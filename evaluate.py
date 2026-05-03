"""
evaluate.py
-----------
Standalone evaluation script — runs on a saved checkpoint without retraining.
Supports normal evaluation and Test-Time Augmentation (TTA).

Usage:
    python evaluate.py --checkpoint "./checkpoints_v10/best_model.pt"
    python evaluate.py --checkpoint "./checkpoints_v10/best_model.pt" --tta 10
    python evaluate.py --checkpoint "./checkpoints_v10/best_model.pt" --tta 20
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import load_data, compute_sbp_tail_multipliers
from siamese_cnn import build_model
from metrics import print_metrics
from train import run_multi_anchor_inference


# ──────────────────────────────────────────────
# TTA augmentation (GPU-side, fast)
# ──────────────────────────────────────────────

def tta_augment_batch(tensor, noise_std=0.03, jitter_range=0.03):
    """Light augmentation: Gaussian noise + per-channel amplitude jitter."""
    noise = torch.randn_like(tensor) * noise_std
    if tensor.ndim == 3:      # (B, C, W) waveform
        scale_shape = (tensor.shape[0], tensor.shape[1], 1)
    elif tensor.ndim == 4:    # (B, C, F, T) spectrogram
        scale_shape = (tensor.shape[0], tensor.shape[1], 1, 1)
    else:
        return tensor + noise
    scale = 1.0 + (torch.rand(*scale_shape, device=tensor.device) * 2 - 1) * jitter_range
    return tensor * scale + noise


# ──────────────────────────────────────────────
# Evaluation loops
# ──────────────────────────────────────────────

def unpack_batch(batch, device, fusion=False):
    """Move a dataset batch to device and normalize batch structure across modes."""
    if fusion:
        anchor, current, d_sbp, d_dbp, anc_sbp_norm, anc_dbp_norm, anc_spec, cur_spec, anc_hf, cur_hf = batch
        anc_spec = anc_spec.to(device)
        cur_spec = cur_spec.to(device)
    else:
        anchor, current, d_sbp, d_dbp, anc_sbp_norm, anc_dbp_norm, anc_hf, cur_hf = batch
        anc_spec = cur_spec = None

    anchor = anchor.to(device)
    current = current.to(device)
    anc_sbp_norm = anc_sbp_norm.to(device)
    anc_dbp_norm = anc_dbp_norm.to(device)
    anc_hf = anc_hf.to(device)
    cur_hf = cur_hf.to(device)
    target = torch.stack([d_sbp, d_dbp], dim=1).to(device)
    return anchor, current, target, anc_sbp_norm, anc_dbp_norm, anc_spec, cur_spec, anc_hf, cur_hf


def run_eval(model, loader, device, fusion=False):
    """Standard (no TTA) evaluation."""
    model.eval()
    all_pred, all_tgt = [], []
    with torch.no_grad():
        for batch in loader:
            anchor, current, target, anc_sbp_norm, anc_dbp_norm, anc_spec, cur_spec, anc_hf, cur_hf = unpack_batch(
                batch, device, fusion=fusion
            )

            pred = model(anchor, current, anc_sbp_norm, anc_dbp_norm,
                         anchor_spec=anc_spec, current_spec=cur_spec,
                         anchor_hf=anc_hf, current_hf=cur_hf)
            all_pred.append(pred.cpu().numpy())
            all_tgt.append(target.cpu().numpy())

    return np.concatenate(all_pred), np.concatenate(all_tgt)


def run_eval_tta(model, loader, device, n_aug=10, fusion=False):
    """TTA evaluation: average 1 clean + n_aug augmented predictions."""
    model.eval()
    all_pred, all_tgt = [], []

    with torch.no_grad():
        for batch in loader:
            anchor, current, target, anc_sbp_norm, anc_dbp_norm, anc_spec, cur_spec, anc_hf, cur_hf = unpack_batch(
                batch, device, fusion=fusion
            )

            # 1) Clean prediction
            preds = [model(anchor, current, anc_sbp_norm, anc_dbp_norm,
                           anchor_spec=anc_spec, current_spec=cur_spec,
                           anchor_hf=anc_hf, current_hf=cur_hf)]

            # 2) Augmented predictions
            for _ in range(n_aug):
                aug_anc = tta_augment_batch(anchor)
                aug_cur = tta_augment_batch(current)
                aug_anc_spec = tta_augment_batch(anc_spec) if anc_spec is not None else None
                aug_cur_spec = tta_augment_batch(cur_spec) if cur_spec is not None else None

                p = model(aug_anc, aug_cur, anc_sbp_norm, anc_dbp_norm,
                          anchor_spec=aug_anc_spec, current_spec=aug_cur_spec,
                          anchor_hf=anc_hf, current_hf=cur_hf)
                preds.append(p)

            avg_pred = torch.stack(preds, dim=0).mean(dim=0)
            all_pred.append(avg_pred.cpu().numpy())
            all_tgt.append(target.cpu().numpy())

    return np.concatenate(all_pred), np.concatenate(all_tgt)


def fit_linear_calibration(pred_mmhg, tgt_mmhg, targets="sbp"):
    """Fit y_true ~= scale * y_pred + bias on validation predictions."""
    scale = np.ones(2, dtype=np.float32)
    bias = np.zeros(2, dtype=np.float32)

    target_map = {
        "sbp": [0],
        "dbp": [1],
        "both": [0, 1],
    }
    for idx in target_map[targets]:
        design = np.column_stack([pred_mmhg[:, idx], np.ones(len(pred_mmhg), dtype=np.float32)])
        coef, _, _, _ = np.linalg.lstsq(design, tgt_mmhg[:, idx], rcond=None)
        scale[idx] = float(coef[0])
        bias[idx] = float(coef[1])
    return scale, bias


def apply_linear_calibration(pred_mmhg, scale, bias):
    calibrated = pred_mmhg.copy()
    calibrated = calibrated * scale.reshape(1, -1) + bias.reshape(1, -1)
    return calibrated


def fit_subject_specific_calibration(pred_mmhg, tgt_mmhg, patient_ids,
                                     targets="sbp", mode="bias",
                                     min_samples=8, shrinkage=12.0):
    """Fit per-patient calibration from validation predictions to targets."""
    pred = np.asarray(pred_mmhg, dtype=np.float32)
    tgt = np.asarray(tgt_mmhg, dtype=np.float32)
    patient_ids = np.asarray(patient_ids, dtype=np.int64)

    target_mask = np.array([
        targets in ("sbp", "both"),
        targets in ("dbp", "both"),
    ], dtype=bool)

    scale_by_pid = {}
    bias_by_pid = {}
    counts_by_pid = {}

    for pid in np.unique(patient_ids):
        mask = patient_ids == int(pid)
        count = int(mask.sum())
        if count < int(min_samples):
            continue

        alpha = count / (count + float(shrinkage))
        scale = np.ones(2, dtype=np.float32)
        bias = np.zeros(2, dtype=np.float32)

        for t in range(2):
            if not target_mask[t]:
                continue
            x = pred[mask, t].astype(np.float64)
            y = tgt[mask, t].astype(np.float64)

            if mode == "affine" and count >= max(int(min_samples), 3):
                x_mean = float(x.mean())
                y_mean = float(y.mean())
                denom = float(np.sum((x - x_mean) ** 2))
                if denom > 1e-6:
                    slope_hat = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
                else:
                    slope_hat = 1.0
                intercept_hat = y_mean - slope_hat * x_mean
                scale[t] = np.float32(1.0 + alpha * (slope_hat - 1.0))
                bias[t] = np.float32(alpha * intercept_hat)
            else:
                mean_err = float((y - x).mean())
                bias[t] = np.float32(alpha * mean_err)

        pid = int(pid)
        scale_by_pid[pid] = scale
        bias_by_pid[pid] = bias
        counts_by_pid[pid] = count

    return {
        "targets": targets,
        "mode": mode,
        "min_samples": int(min_samples),
        "shrinkage": float(shrinkage),
        "scale_by_pid": scale_by_pid,
        "bias_by_pid": bias_by_pid,
        "counts_by_pid": counts_by_pid,
    }


def apply_subject_specific_calibration(pred_mmhg, patient_ids, calib_state):
    """Apply per-patient calibration to predictions."""
    corrected = np.asarray(pred_mmhg, dtype=np.float32).copy()
    patient_ids = np.asarray(patient_ids, dtype=np.int64)

    scale_by_pid = calib_state.get("scale_by_pid", {})
    bias_by_pid = calib_state.get("bias_by_pid", {})
    for pid in np.unique(patient_ids):
        pid = int(pid)
        if pid not in scale_by_pid:
            continue
        mask = patient_ids == pid
        corrected[mask] = (
            corrected[mask] * scale_by_pid[pid].reshape(1, -1)
            + bias_by_pid[pid].reshape(1, -1)
        )
    return corrected


def build_sbp_residual_feature_matrix(dataset, pred_mmhg):
    """Build inference-time features for a second-stage SBP residual model."""
    indices = dataset.indices
    n = len(indices)
    pred_sbp = pred_mmhg[:, 0].astype(np.float32)
    pred_dbp = pred_mmhg[:, 1].astype(np.float32)
    pred_pp = pred_sbp - pred_dbp

    anchor_stats = np.zeros((n, 8), dtype=np.float32)
    if dataset.hand_features is not None:
        current_hf = dataset.hand_features[indices].astype(np.float32)
        hf_diff = np.zeros_like(current_hf)
    else:
        current_hf = np.zeros((n, 0), dtype=np.float32)
        hf_diff = np.zeros((n, 0), dtype=np.float32)

    for row_i, idx in enumerate(indices):
        pid = int(dataset.patient_ids[idx])
        anc_idx = np.asarray(dataset.anchor_lists[pid], dtype=np.int64)
        anc_sbp = dataset.y_sbp[anc_idx].astype(np.float32)
        anc_dbp = dataset.y_dbp[anc_idx].astype(np.float32)

        anc_sbp_mean = float(anc_sbp.mean())
        anc_dbp_mean = float(anc_dbp.mean())
        pred_gap = np.abs(pred_sbp[row_i] - anc_sbp)

        anchor_stats[row_i, 0] = anc_sbp_mean
        anchor_stats[row_i, 1] = float(anc_sbp.std())
        anchor_stats[row_i, 2] = anc_dbp_mean
        anchor_stats[row_i, 3] = float(anc_dbp.std())
        anchor_stats[row_i, 4] = anc_sbp_mean - anc_dbp_mean
        anchor_stats[row_i, 5] = float(pred_gap.mean())
        anchor_stats[row_i, 6] = float(pred_gap.min())
        anchor_stats[row_i, 7] = float(pred_gap.max())

        if current_hf.shape[1] > 0:
            anc_hf_mean = dataset.hand_features[anc_idx].astype(np.float32).mean(axis=0)
            hf_diff[row_i] = np.abs(current_hf[row_i] - anc_hf_mean)

    base = np.column_stack([
        pred_sbp,
        pred_dbp,
        pred_pp,
        anchor_stats[:, 0],
        anchor_stats[:, 1],
        anchor_stats[:, 2],
        anchor_stats[:, 3],
        anchor_stats[:, 4],
        pred_sbp - anchor_stats[:, 0],
        np.abs(pred_sbp - anchor_stats[:, 0]),
        anchor_stats[:, 5],
        anchor_stats[:, 6],
        anchor_stats[:, 7],
        np.maximum(0.0, 90.0 - pred_sbp),
        np.maximum(0.0, pred_sbp - 150.0),
        np.maximum(0.0, 110.0 - pred_sbp),
        np.maximum(0.0, pred_sbp - 130.0),
    ]).astype(np.float32)

    poly = np.column_stack([
        pred_sbp ** 2,
        pred_dbp ** 2,
        pred_pp ** 2,
        pred_sbp * anchor_stats[:, 0],
        pred_sbp * pred_dbp,
    ]).astype(np.float32)

    parts = [base, poly]
    if current_hf.shape[1] > 0:
        parts.extend([current_hf, hf_diff])
    return np.concatenate(parts, axis=1).astype(np.float32)


def _normalise_piecewise_edges(piecewise_edges):
    if piecewise_edges is None:
        return np.zeros(0, dtype=np.float32)
    if isinstance(piecewise_edges, str):
        items = [s.strip() for s in piecewise_edges.split(",") if s.strip()]
        if not items:
            return np.zeros(0, dtype=np.float32)
        edges = np.asarray([float(s) for s in items], dtype=np.float32)
    else:
        edges = np.asarray(list(piecewise_edges), dtype=np.float32)
        if edges.size == 0:
            return np.zeros(0, dtype=np.float32)
    return np.sort(np.unique(edges.astype(np.float32)))


def _fit_weighted_ridge_state(features, residual_target, alpha=64.0, sample_weight=None):
    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(residual_target, dtype=np.float64)

    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    x_std[x_std < 1e-8] = 1.0
    Xs = (X - x_mean) / x_std

    if sample_weight is None:
        sample_weight = np.ones(len(Xs), dtype=np.float64)
    else:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)

    y_mean = float(np.average(y, weights=sample_weight))
    yc = y - y_mean
    sqrt_w = np.sqrt(sample_weight)
    Xw = Xs * sqrt_w[:, None]
    yw = yc * sqrt_w
    reg = np.eye(Xw.shape[1], dtype=np.float64) * float(alpha)
    coef = np.linalg.solve(Xw.T @ Xw + reg, Xw.T @ yw)
    return {
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "coef": coef.astype(np.float32),
        "y_mean": np.float32(y_mean),
    }


def _predict_weighted_ridge_state(features, ridge_state):
    X = np.asarray(features, dtype=np.float32)
    Xs = (X - ridge_state["x_mean"]) / ridge_state["x_std"]
    return (Xs @ ridge_state["coef"] + ridge_state["y_mean"]).astype(np.float32)


def fit_sbp_residual_corrector(features, pred_sbp, true_sbp, alpha=64.0,
                               sample_weight=None, clip_mmHg=12.0,
                               model_type="ridge", piecewise_edges="110,130",
                               piecewise_min_samples=1024,
                               piecewise_blend_width=6.0):
    """Fit a weighted residual model to predict SBP error."""
    X = np.asarray(features, dtype=np.float32)
    pred_sbp = np.asarray(pred_sbp, dtype=np.float32)
    residual_target = np.asarray(true_sbp - pred_sbp, dtype=np.float32)

    if sample_weight is None:
        sample_weight = np.ones(len(X), dtype=np.float32)
    else:
        sample_weight = np.asarray(sample_weight, dtype=np.float32)

    global_state = _fit_weighted_ridge_state(
        X, residual_target, alpha=alpha, sample_weight=sample_weight
    )
    if model_type == "ridge":
        return {
            "model_type": "ridge",
            "global_state": global_state,
            "clip_mmHg": float(clip_mmHg),
        }

    if model_type != "piecewise_ridge":
        raise ValueError(f"Unsupported residual model type: {model_type}")

    edges = _normalise_piecewise_edges(piecewise_edges)
    if edges.size == 0:
        return {
            "model_type": "ridge",
            "global_state": global_state,
            "clip_mmHg": float(clip_mmHg),
        }

    region_ids = np.digitize(pred_sbp, edges, right=False)
    piece_states = []
    piece_counts = []
    piece_is_local = []
    for region_i in range(len(edges) + 1):
        mask = region_ids == region_i
        count = int(mask.sum())
        piece_counts.append(count)
        if count >= int(piecewise_min_samples):
            piece_states.append(
                _fit_weighted_ridge_state(
                    X[mask], residual_target[mask], alpha=alpha, sample_weight=sample_weight[mask]
                )
            )
            piece_is_local.append(True)
        else:
            piece_states.append(global_state)
            piece_is_local.append(False)

    return {
        "model_type": "piecewise_ridge",
        "global_state": global_state,
        "piecewise_edges": edges.astype(np.float32),
        "piece_states": piece_states,
        "piece_counts": np.asarray(piece_counts, dtype=np.int32),
        "piece_is_local": np.asarray(piece_is_local, dtype=bool),
        "piecewise_blend_width": float(piecewise_blend_width),
        "clip_mmHg": float(clip_mmHg),
    }


def apply_sbp_residual_corrector(pred_mmhg, features, model_state):
    corrected = pred_mmhg.copy()
    X = np.asarray(features, dtype=np.float32)

    if "model_type" not in model_state:
        residual = _predict_weighted_ridge_state(X, model_state)
    elif model_state["model_type"] == "ridge":
        residual = _predict_weighted_ridge_state(X, model_state["global_state"])
    elif model_state["model_type"] == "piecewise_ridge":
        pred_sbp = corrected[:, 0].astype(np.float32)
        edges = np.asarray(model_state.get("piecewise_edges", []), dtype=np.float32)
        piece_states = model_state["piece_states"]
        stacked_pred = np.column_stack([
            _predict_weighted_ridge_state(X, piece_state) for piece_state in piece_states
        ]).astype(np.float32)
        region_ids = np.digitize(pred_sbp, edges, right=False)
        residual = stacked_pred[np.arange(len(pred_sbp)), region_ids]

        blend_width = float(model_state.get("piecewise_blend_width", 0.0))
        if blend_width > 0.0 and edges.size > 0:
            for edge_i, edge in enumerate(edges):
                lo = edge - blend_width
                hi = edge + blend_width
                mask = (pred_sbp >= lo) & (pred_sbp < hi)
                if not np.any(mask):
                    continue
                t = ((pred_sbp[mask] - lo) / max(hi - lo, 1e-6)).astype(np.float32)
                left_pred = stacked_pred[mask, edge_i]
                right_pred = stacked_pred[mask, edge_i + 1]
                residual[mask] = (1.0 - t) * left_pred + t * right_pred
    else:
        raise ValueError(f"Unsupported residual model type: {model_state['model_type']}")

    clip_mmHg = float(model_state.get("clip_mmHg", 12.0))
    if clip_mmHg > 0:
        residual = np.clip(residual, -clip_mmHg, clip_mmHg)
    corrected[:, 0] = corrected[:, 0] + residual
    return corrected


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*56}")
    print(f"  Evaluation Script")
    print(f"  Device     : {device}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  TTA        : {'OFF' if args.tta == 0 else f'{args.tta} augmentations'}")
    if args.fit_output_calibration:
        print(f"  Calibration: val-fitted linear correction ({args.calibration_targets})")
    if args.fit_subject_calibration:
        print(
            "  SubjectCal : "
            f"val-fitted per-patient {args.subject_calibration_mode} correction "
            f"({args.subject_calibration_targets}, min={args.subject_calibration_min_samples}, "
            f"shrink={args.subject_calibration_shrinkage})"
        )
    if args.fit_sbp_residual_corrector:
        residual_desc = "ridge"
        if args.residual_model_type == "piecewise_ridge":
            residual_desc = (
                "piecewise ridge "
                f"(edges={args.residual_piecewise_edges}, min={args.residual_piecewise_min_samples}, "
                f"blend={args.residual_piecewise_blend_width})"
            )
        print(
            "  Residual   : "
            f"val-fitted SBP {residual_desc} corrector (alpha={args.residual_corrector_alpha}, "
            f"clip={args.residual_corrector_clip})"
        )
    if args.use_beat_features:
        print("  Beat aware : ON (beat-summary feature stream)")
    if args.use_ecg:
        if args.ecg_input_mode == "both":
            ecg_desc = "ECG raw channel + timing features"
        elif args.ecg_input_mode == "features_only":
            ecg_desc = "ECG timing features only"
        else:
            ecg_desc = "ECG raw channel only"
        print(f"  Hybrid input: {ecg_desc}")
    if args.use_absolute_refinement:
        print("  Abs refine : ON (aux absolute-BP refinement path)")
    if args.multi_anchor_eval:
        if args.anchor_similarity_mode != "none":
            topk_txt = f", top-k={args.anchor_top_k}" if args.anchor_top_k > 0 else ""
            eval_mode = (
                f"similarity-guided multi-anchor ({args.anchor_similarity_mode}{topk_txt}, "
                f"temp={args.anchor_similarity_temp})"
            )
        else:
            eval_mode = "confidence-weighted multi-anchor" if args.use_pair_uncertainty else "multi-anchor average"
    else:
        eval_mode = "single-anchor"
    print(f"  Eval mode  : {eval_mode}")
    print(f"  Pair block : {'gated interaction' if args.use_gated_pair_interaction else 'diff + hadamard'}")
    print(f"{'='*56}\n")

    # Load data (only need test set)
    train_ds, val_ds, test_ds, anchors, y_sbp, y_dbp, _ = load_data(
        args.data_dir, random_anchor=False, min_sbp_std=args.min_sbp_std,
        num_anchors=args.num_anchors,
        use_spectrograms=args.use_spectrograms,
        use_fusion=args.use_fusion,
        target_scale=args.target_scale,
        quality_threshold=args.quality_threshold,
        shuffle_eval=args.shuffle_eval,
        preserve_ppg_amplitude=args.preserve_ppg_amplitude,
        use_beat_features=args.use_beat_features,
        use_ecg=args.use_ecg,
        ecg_input_mode=args.ecg_input_mode,
    )

    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device == "cuda"),
    )

    # Load model
    model = build_model(embed_dim=256, dropout=0.3, device=device,
                        use_spectrograms=args.use_spectrograms,
                        use_fusion=args.use_fusion,
                        use_pair_uncertainty=args.use_pair_uncertainty,
                        use_gated_pair_interaction=args.use_gated_pair_interaction,
                        use_absolute_refinement=args.use_absolute_refinement,
                        hand_feature_dim=test_ds.hand_feature_dim,
                        input_channels=test_ds.num_channels,
                        target_scale=args.target_scale)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)

    cal_scale = None
    cal_bias = None
    subject_cal_state = None
    val_pred_mmhg = None
    val_tgt_mmhg = None
    if args.fit_output_calibration:
        print("\n" + "=" * 56)
        print("  Validation Calibration Fit")
        print("=" * 56)
        if args.multi_anchor_eval:
            val_pred_mmhg, val_tgt_mmhg = run_multi_anchor_inference(
                model, val_ds, device,
                batch_size=args.eval_batch_size,
                fusion=args.use_fusion,
                n_aug=0,
                anchor_similarity_mode=args.anchor_similarity_mode,
                anchor_top_k=args.anchor_top_k,
                anchor_similarity_temp=args.anchor_similarity_temp,
            )
        else:
            val_pred, val_tgt = run_eval(model, val_loader, device, fusion=args.use_fusion)
            val_pred_mmhg = val_pred * args.target_scale
            val_tgt_mmhg = val_tgt * args.target_scale
        cal_scale, cal_bias = fit_linear_calibration(
            val_pred_mmhg, val_tgt_mmhg, targets=args.calibration_targets
        )
        print(f"  Scale      : SBP={cal_scale[0]:.4f}  DBP={cal_scale[1]:.4f}")
        print(f"  Bias       : SBP={cal_bias[0]:.4f}  DBP={cal_bias[1]:.4f}")
    elif args.fit_subject_calibration or args.fit_sbp_residual_corrector:
        if args.multi_anchor_eval:
            val_pred_mmhg, val_tgt_mmhg = run_multi_anchor_inference(
                model, val_ds, device,
                batch_size=args.eval_batch_size,
                fusion=args.use_fusion,
                n_aug=0,
                anchor_similarity_mode=args.anchor_similarity_mode,
                anchor_top_k=args.anchor_top_k,
                anchor_similarity_temp=args.anchor_similarity_temp,
            )
        else:
            val_pred, val_tgt = run_eval(model, val_loader, device, fusion=args.use_fusion)
            val_pred_mmhg = val_pred * args.target_scale
            val_tgt_mmhg = val_tgt * args.target_scale

    val_patient_ids = val_ds.patient_ids[val_ds.indices].astype(np.int64)
    test_patient_ids = test_ds.patient_ids[test_ds.indices].astype(np.int64)
    if args.fit_subject_calibration:
        print("\n" + "=" * 56)
        print("  Validation Subject Calibration Fit")
        print("=" * 56)
        val_cal_base = val_pred_mmhg
        if cal_scale is not None and cal_bias is not None:
            val_cal_base = apply_linear_calibration(val_pred_mmhg, cal_scale, cal_bias)
        subject_cal_state = fit_subject_specific_calibration(
            val_cal_base,
            val_tgt_mmhg,
            val_patient_ids,
            targets=args.subject_calibration_targets,
            mode=args.subject_calibration_mode,
            min_samples=args.subject_calibration_min_samples,
            shrinkage=args.subject_calibration_shrinkage,
        )
        print(f"  Patients    : {len(subject_cal_state['scale_by_pid'])}")
        print(f"  Mode        : {subject_cal_state['mode']}  |  Targets: {subject_cal_state['targets']}")
        val_pred_sub = apply_subject_specific_calibration(val_cal_base, val_patient_ids, subject_cal_state)
        sub_label = "Val+SubCal" if cal_scale is None else "Val+Cal+SubCal"
        print_metrics(val_pred_sub, val_tgt_mmhg, label=sub_label)

    # ── Standard eval ─────────────────────────
    print("\n" + "=" * 56)
    print("  Standard Test Evaluation")
    print("=" * 56)
    t0 = time.time()
    if args.multi_anchor_eval:
        pred_mmhg, tgt_mmhg = run_multi_anchor_inference(
            model, test_ds, device,
            batch_size=args.eval_batch_size,
            fusion=args.use_fusion,
            n_aug=0,
            anchor_similarity_mode=args.anchor_similarity_mode,
            anchor_top_k=args.anchor_top_k,
            anchor_similarity_temp=args.anchor_similarity_temp,
        )
    else:
        pred, tgt = run_eval(model, test_loader, device, fusion=args.use_fusion)
        pred_mmhg = pred * args.target_scale
        tgt_mmhg = tgt * args.target_scale
    elapsed = time.time() - t0
    m_std = print_metrics(pred_mmhg, tgt_mmhg, label="Test")
    print(f"  Time: {elapsed:.1f}s")
    pred_base = pred_mmhg
    if cal_scale is not None and cal_bias is not None:
        pred_mmhg_cal = apply_linear_calibration(pred_mmhg, cal_scale, cal_bias)
        print_metrics(pred_mmhg_cal, tgt_mmhg, label="Test+Cal")
        pred_base = pred_mmhg_cal
    pred_mmhg_sub = None
    if subject_cal_state is not None:
        pred_mmhg_sub = apply_subject_specific_calibration(pred_base, test_patient_ids, subject_cal_state)
        sub_label = "Test+SubCal" if cal_scale is None else "Test+Cal+SubCal"
        print_metrics(pred_mmhg_sub, tgt_mmhg, label=sub_label)
        pred_base = pred_mmhg_sub

    pred_mmhg_res = None
    if args.fit_sbp_residual_corrector:
        print("\n" + "=" * 56)
        print("  Validation SBP Residual Fit")
        print("=" * 56)
        val_base = val_pred_mmhg
        if cal_scale is not None and cal_bias is not None:
            val_base = apply_linear_calibration(val_pred_mmhg, cal_scale, cal_bias)
        if subject_cal_state is not None:
            val_base = apply_subject_specific_calibration(val_base, val_patient_ids, subject_cal_state)
        val_feat = build_sbp_residual_feature_matrix(val_ds, val_base)
        val_weights = None
        if args.residual_tail_weighting:
            val_weights = compute_sbp_tail_multipliers(
                val_tgt_mmhg[:, 0],
                shoulder_low=args.residual_shoulder_low,
                tail_low=args.residual_tail_low,
                shoulder_high=args.residual_shoulder_high,
                tail_high=args.residual_tail_high,
                shoulder_boost=args.residual_shoulder_boost,
                tail_boost=args.residual_tail_boost,
            ).astype(np.float32)
        residual_model = fit_sbp_residual_corrector(
            val_feat,
            val_base[:, 0],
            val_tgt_mmhg[:, 0],
            alpha=args.residual_corrector_alpha,
            sample_weight=val_weights,
            clip_mmHg=args.residual_corrector_clip,
            model_type=args.residual_model_type,
            piecewise_edges=args.residual_piecewise_edges,
            piecewise_min_samples=args.residual_piecewise_min_samples,
            piecewise_blend_width=args.residual_piecewise_blend_width,
        )
        val_pred_res = apply_sbp_residual_corrector(val_base, val_feat, residual_model)
        print_metrics(val_pred_res, val_tgt_mmhg, label="Val+SBPRes")

        test_feat = build_sbp_residual_feature_matrix(test_ds, pred_base)
        pred_mmhg_res = apply_sbp_residual_corrector(pred_base, test_feat, residual_model)
        if subject_cal_state is not None:
            res_label = "Test+SubCal+SBPRes" if cal_scale is None else "Test+Cal+SubCal+SBPRes"
        else:
            res_label = "Test+SBPRes" if cal_scale is None else "Test+Cal+SBPRes"
        print_metrics(pred_mmhg_res, tgt_mmhg, label=res_label)

    # ── TTA eval ──────────────────────────────
    if args.tta > 0:
        print(f"\n{'='*56}")
        print(f"  Test + TTA (n_aug={args.tta})")
        print(f"{'='*56}")
        t0 = time.time()
        if args.multi_anchor_eval:
            tta_pred_mmhg, tta_tgt_mmhg = run_multi_anchor_inference(
                model, test_ds, device,
                batch_size=args.eval_batch_size,
                fusion=args.use_fusion,
                n_aug=args.tta,
                anchor_similarity_mode=args.anchor_similarity_mode,
                anchor_top_k=args.anchor_top_k,
                anchor_similarity_temp=args.anchor_similarity_temp,
            )
        else:
            tta_pred, tta_tgt = run_eval_tta(
                model, test_loader, device,
                n_aug=args.tta, fusion=args.use_fusion,
            )
            tta_pred_mmhg = tta_pred * args.target_scale
            tta_tgt_mmhg = tta_tgt * args.target_scale
        elapsed = time.time() - t0
        m_tta = print_metrics(tta_pred_mmhg, tta_tgt_mmhg, label="Test+TTA")
        print(f"  Time: {elapsed:.1f}s")
        tta_base = tta_pred_mmhg
        if args.fit_output_calibration:
            if args.multi_anchor_eval:
                val_pred_tta_mmhg, val_tgt_tta_mmhg = run_multi_anchor_inference(
                    model, val_ds, device,
                    batch_size=args.eval_batch_size,
                    fusion=args.use_fusion,
                    n_aug=args.tta,
                    anchor_similarity_mode=args.anchor_similarity_mode,
                    anchor_top_k=args.anchor_top_k,
                    anchor_similarity_temp=args.anchor_similarity_temp,
                )
            else:
                val_pred_tta, val_tgt_tta = run_eval_tta(
                    model, val_loader, device,
                    n_aug=args.tta, fusion=args.use_fusion,
                )
                val_pred_tta_mmhg = val_pred_tta * args.target_scale
                val_tgt_tta_mmhg = val_tgt_tta * args.target_scale
            cal_scale_tta, cal_bias_tta = fit_linear_calibration(
                val_pred_tta_mmhg, val_tgt_tta_mmhg, targets=args.calibration_targets
            )
            tta_pred_mmhg_cal = apply_linear_calibration(tta_pred_mmhg, cal_scale_tta, cal_bias_tta)
            m_tta_cal = print_metrics(tta_pred_mmhg_cal, tta_tgt_mmhg, label="Test+TTA+Cal")
            tta_base = tta_pred_mmhg_cal
        elif args.fit_subject_calibration or args.fit_sbp_residual_corrector:
            if args.multi_anchor_eval:
                val_pred_tta_mmhg, val_tgt_tta_mmhg = run_multi_anchor_inference(
                    model, val_ds, device,
                    batch_size=args.eval_batch_size,
                    fusion=args.use_fusion,
                    n_aug=args.tta,
                    anchor_similarity_mode=args.anchor_similarity_mode,
                    anchor_top_k=args.anchor_top_k,
                    anchor_similarity_temp=args.anchor_similarity_temp,
                )
            else:
                val_pred_tta, val_tgt_tta = run_eval_tta(
                    model, val_loader, device,
                    n_aug=args.tta, fusion=args.use_fusion,
                )
                val_pred_tta_mmhg = val_pred_tta * args.target_scale
                val_tgt_tta_mmhg = val_tgt_tta * args.target_scale

        tta_pred_mmhg_sub = None
        subject_cal_state_tta = None
        if args.fit_subject_calibration:
            val_tta_cal_base = val_pred_tta_mmhg
            if args.fit_output_calibration:
                val_tta_cal_base = apply_linear_calibration(val_pred_tta_mmhg, cal_scale_tta, cal_bias_tta)
            subject_cal_state_tta = fit_subject_specific_calibration(
                val_tta_cal_base,
                val_tgt_tta_mmhg,
                val_patient_ids,
                targets=args.subject_calibration_targets,
                mode=args.subject_calibration_mode,
                min_samples=args.subject_calibration_min_samples,
                shrinkage=args.subject_calibration_shrinkage,
            )
            tta_pred_mmhg_sub = apply_subject_specific_calibration(tta_base, test_patient_ids, subject_cal_state_tta)
            tta_sub_label = "Test+TTA+SubCal" if not args.fit_output_calibration else "Test+TTA+Cal+SubCal"
            print_metrics(tta_pred_mmhg_sub, tta_tgt_mmhg, label=tta_sub_label)
            tta_base = tta_pred_mmhg_sub

        tta_pred_mmhg_res = None
        if args.fit_sbp_residual_corrector:
            val_tta_base = val_pred_tta_mmhg
            if args.fit_output_calibration:
                val_tta_base = apply_linear_calibration(val_pred_tta_mmhg, cal_scale_tta, cal_bias_tta)
            if subject_cal_state_tta is not None:
                val_tta_base = apply_subject_specific_calibration(val_tta_base, val_patient_ids, subject_cal_state_tta)
            val_tta_feat = build_sbp_residual_feature_matrix(val_ds, val_tta_base)
            val_tta_weights = None
            if args.residual_tail_weighting:
                val_tta_weights = compute_sbp_tail_multipliers(
                    val_tgt_tta_mmhg[:, 0],
                    shoulder_low=args.residual_shoulder_low,
                    tail_low=args.residual_tail_low,
                    shoulder_high=args.residual_shoulder_high,
                    tail_high=args.residual_tail_high,
                    shoulder_boost=args.residual_shoulder_boost,
                    tail_boost=args.residual_tail_boost,
                ).astype(np.float32)
            residual_model_tta = fit_sbp_residual_corrector(
                val_tta_feat,
                val_tta_base[:, 0],
                val_tgt_tta_mmhg[:, 0],
                alpha=args.residual_corrector_alpha,
                sample_weight=val_tta_weights,
                clip_mmHg=args.residual_corrector_clip,
                model_type=args.residual_model_type,
                piecewise_edges=args.residual_piecewise_edges,
                piecewise_min_samples=args.residual_piecewise_min_samples,
                piecewise_blend_width=args.residual_piecewise_blend_width,
            )
            tta_feat = build_sbp_residual_feature_matrix(test_ds, tta_base)
            tta_pred_mmhg_res = apply_sbp_residual_corrector(tta_base, tta_feat, residual_model_tta)
            if subject_cal_state_tta is not None:
                tta_res_label = "Test+TTA+SubCal+SBPRes" if not args.fit_output_calibration else "Test+TTA+Cal+SubCal+SBPRes"
            else:
                tta_res_label = "Test+TTA+SBPRes" if not args.fit_output_calibration else "Test+TTA+Cal+SBPRes"
            print_metrics(tta_pred_mmhg_res, tta_tgt_mmhg, label=tta_res_label)

        # ── Comparison ────────────────────────
        print(f"\n{'='*56}")
        print(f"  TTA Improvement")
        print(f"{'='*56}")
        print(f"  {'Metric':<20} {'Standard':>10} {'TTA':>10} {'Delta':>10}")
        print(f"  {'-'*52}")
        for key, label in [("mae_sbp", "SBP MAE"), ("mae_dbp", "DBP MAE"),
                           ("combined_mae", "Combined MAE"),
                           ("std_sbp", "SBP Std Err"), ("std_dbp", "DBP Std Err")]:
            d = m_tta[key] - m_std[key]
            print(f"  {label:<20} {m_std[key]:>10.2f} {m_tta[key]:>10.2f} {d:>+10.2f}")
        print(f"  {'SBP BHS':<20} {m_std['bhs_sbp']:>10} {m_tta['bhs_sbp']:>10}")
        print(f"  {'DBP BHS':<20} {m_std['bhs_dbp']:>10} {m_tta['bhs_dbp']:>10}")

    # Save
    save_dir = os.path.dirname(args.checkpoint)
    if save_dir:
        np.save(os.path.join(save_dir, "eval_pred.npy"), pred_mmhg)
        if args.fit_output_calibration and cal_scale is not None:
            np.save(os.path.join(save_dir, "eval_pred_cal.npy"), pred_mmhg_cal)
        if args.fit_subject_calibration and pred_mmhg_sub is not None:
            np.save(os.path.join(save_dir, "eval_pred_subcal.npy"), pred_mmhg_sub)
        if args.fit_sbp_residual_corrector and pred_mmhg_res is not None:
            np.save(os.path.join(save_dir, "eval_pred_sbpres.npy"), pred_mmhg_res)
        if args.tta > 0:
            np.save(os.path.join(save_dir, "eval_pred_tta.npy"), tta_pred_mmhg)
            if args.fit_output_calibration:
                np.save(os.path.join(save_dir, "eval_pred_tta_cal.npy"), tta_pred_mmhg_cal)
            if args.fit_subject_calibration and tta_pred_mmhg_sub is not None:
                np.save(os.path.join(save_dir, "eval_pred_tta_subcal.npy"), tta_pred_mmhg_sub)
            if args.fit_sbp_residual_corrector and tta_pred_mmhg_res is not None:
                np.save(os.path.join(save_dir, "eval_pred_tta_sbpres.npy"), tta_pred_mmhg_res)
        print(f"\n  Predictions saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=r"C:\MIMIC2_out_v8")
    parser.add_argument("--checkpoint", default=r".\checkpoints_v10\best_model.pt")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--target_scale", type=float, default=10.0)
    parser.add_argument("--num_anchors", type=int, default=5)
    parser.add_argument("--tta", type=int, default=10, help="Number of TTA augmentations (0=off)")
    parser.add_argument("--min_sbp_std", type=float, default=5.0)
    parser.add_argument("--quality_threshold", type=float, default=0.4)
    parser.add_argument("--use_spectrograms", action="store_true", default=False)
    parser.add_argument("--use_fusion", action="store_true", default=True)
    parser.add_argument("--no_fusion", dest="use_fusion", action="store_false")
    parser.add_argument("--use_beat_features", action="store_true", default=False)
    parser.add_argument("--no_beat_features", dest="use_beat_features", action="store_false")
    parser.add_argument("--use_ecg", action="store_true", default=False)
    parser.add_argument("--no_ecg", dest="use_ecg", action="store_false")
    parser.add_argument("--ecg_input_mode", choices=["both", "features_only", "raw_only"],
                        default="both",
                        help="How to use ECG when --use_ecg is enabled")
    parser.add_argument("--use_gated_pair_interaction", action="store_true", default=False)
    parser.add_argument("--no_gated_pair_interaction", dest="use_gated_pair_interaction", action="store_false")
    parser.add_argument("--use_absolute_refinement", action="store_true", default=False)
    parser.add_argument("--no_absolute_refinement", dest="use_absolute_refinement", action="store_false")
    parser.add_argument("--use_pair_uncertainty", action="store_true", default=False)
    parser.add_argument("--no_pair_uncertainty", dest="use_pair_uncertainty", action="store_false")
    parser.add_argument("--shuffle_eval", action="store_true", default=True)
    parser.add_argument("--no_shuffle_eval", dest="shuffle_eval", action="store_false")
    parser.add_argument("--preserve_ppg_amplitude", action="store_true", default=True)
    parser.add_argument("--no_preserve_ppg_amplitude", dest="preserve_ppg_amplitude", action="store_false")
    parser.add_argument("--multi_anchor_eval", action="store_true", default=True)
    parser.add_argument("--no_multi_anchor_eval", dest="multi_anchor_eval", action="store_false")
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--anchor_similarity_mode", choices=["none", "embedding", "handcrafted"], default="none")
    parser.add_argument("--anchor_top_k", type=int, default=0)
    parser.add_argument("--anchor_similarity_temp", type=float, default=0.35)
    parser.add_argument("--fit_output_calibration", action="store_true", default=False)
    parser.add_argument("--calibration_targets", choices=["sbp", "dbp", "both"], default="sbp")
    parser.add_argument("--fit_subject_calibration", action="store_true", default=False)
    parser.add_argument("--subject_calibration_targets", choices=["sbp", "dbp", "both"], default="sbp")
    parser.add_argument("--subject_calibration_mode", choices=["bias", "affine"], default="bias")
    parser.add_argument("--subject_calibration_min_samples", type=int, default=8)
    parser.add_argument("--subject_calibration_shrinkage", type=float, default=12.0)
    parser.add_argument("--fit_sbp_residual_corrector", action="store_true", default=False)
    parser.add_argument("--residual_model_type", choices=["ridge", "piecewise_ridge"], default="ridge")
    parser.add_argument("--residual_corrector_alpha", type=float, default=64.0)
    parser.add_argument("--residual_corrector_clip", type=float, default=12.0)
    parser.add_argument("--residual_piecewise_edges", default="110,130")
    parser.add_argument("--residual_piecewise_min_samples", type=int, default=1024)
    parser.add_argument("--residual_piecewise_blend_width", type=float, default=6.0)
    parser.add_argument("--residual_tail_weighting", action="store_true", default=True)
    parser.add_argument("--no_residual_tail_weighting", dest="residual_tail_weighting", action="store_false")
    parser.add_argument("--residual_tail_low", type=float, default=90.0)
    parser.add_argument("--residual_tail_high", type=float, default=150.0)
    parser.add_argument("--residual_shoulder_low", type=float, default=110.0)
    parser.add_argument("--residual_shoulder_high", type=float, default=130.0)
    parser.add_argument("--residual_shoulder_boost", type=float, default=1.15)
    parser.add_argument("--residual_tail_boost", type=float, default=1.60)
    args = parser.parse_args()
    main(args)
