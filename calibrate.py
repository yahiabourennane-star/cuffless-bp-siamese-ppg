"""
calibrate.py
------------
Per-patient calibration + multi-anchor ensemble for improved BP estimation.

Two techniques that significantly improve SBP accuracy:

1. **Multi-Anchor Ensemble**: Instead of 1 anchor per patient, use K anchors
   and average the absolute BP predictions. Reduces variance on extremes.

2. **Per-Patient Fine-Tuning**: For each patient, take their training windows
   (simulating initial cuff calibration readings) and fine-tune ONLY the
   regression heads. This adapts the model to each patient's BP-PPG relationship.

Usage:
    python calibrate.py --checkpoint ./checkpoints_v13/best_model.pt --save_dir ./checkpoints_v14
    python calibrate.py --checkpoint ./checkpoints_v13/best_model.pt --save_dir ./checkpoints_v14 --n_cal 20 --ft_steps 30
"""

import argparse
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from siamese_cnn import build_model
from dataset import (
    load_data, normalise_windows, compute_spectrogram,
    SiameseBPDataset, N_HAND_FEATURES
)
from metrics import print_metrics, compute_metrics


# ──────────────────────────────────────────────
# Multi-Anchor Ensemble Prediction
# ──────────────────────────────────────────────

def predict_multi_anchor(model, test_ds, anchors, y_sbp, y_dbp, patient_ids,
                         device, target_scale, K=5, fusion=True,
                         hand_features=None, X_spec=None):
    """
    For each test window, predict using K different anchors and average
    the absolute BP predictions.

    This reduces regression-to-mean: if one anchor under-predicts SBP,
    another anchor at a different baseline may compensate.
    """
    model.eval()
    all_pred_sbp = []
    all_pred_dbp = []
    all_true_sbp = []
    all_true_dbp = []

    X = test_ds.X

    with torch.no_grad():
        for i in range(len(test_ds)):
            idx = test_ds.indices[i]
            pid = int(patient_ids[idx])

            # Get this patient's anchors (up to K)
            anc_entry = anchors.get(pid)
            if anc_entry is None:
                continue
            if isinstance(anc_entry, list):
                anchor_indices = anc_entry[:K]
            else:
                anchor_indices = [anc_entry]

            true_sbp = float(y_sbp[idx])
            true_dbp = float(y_dbp[idx])

            # Current window
            cur_sig = torch.tensor(X[idx], dtype=torch.float32).unsqueeze(0).to(device)

            if hand_features is not None:
                cur_hf = torch.tensor(hand_features[idx], dtype=torch.float32).unsqueeze(0).to(device)
            else:
                cur_hf = torch.zeros(1, N_HAND_FEATURES, dtype=torch.float32).to(device)

            if fusion and X_spec is not None:
                cur_spec = torch.tensor(np.array(X_spec[idx]), dtype=torch.float32).unsqueeze(0).to(device)
            else:
                cur_spec = None

            # Predict with each anchor, collect absolute BP
            sbp_preds = []
            dbp_preds = []

            for anc_i in anchor_indices:
                anc_sig = torch.tensor(X[anc_i], dtype=torch.float32).unsqueeze(0).to(device)
                anc_sbp_norm = torch.tensor([float(y_sbp[anc_i]) / 100.0], dtype=torch.float32).to(device)
                anc_dbp_norm = torch.tensor([float(y_dbp[anc_i]) / 100.0], dtype=torch.float32).to(device)

                if hand_features is not None:
                    anc_hf = torch.tensor(hand_features[anc_i], dtype=torch.float32).unsqueeze(0).to(device)
                else:
                    anc_hf = torch.zeros(1, N_HAND_FEATURES, dtype=torch.float32).to(device)

                if fusion and X_spec is not None:
                    anc_sp = torch.tensor(np.array(X_spec[anc_i]), dtype=torch.float32).unsqueeze(0).to(device)
                else:
                    anc_sp = None

                pred = model(anc_sig, cur_sig, anc_sbp_norm, anc_dbp_norm,
                             anchor_spec=anc_sp, current_spec=cur_spec,
                             anchor_hf=anc_hf, current_hf=cur_hf)
                delta = pred.cpu().numpy()[0] * target_scale

                sbp_preds.append(float(y_sbp[anc_i]) + delta[0])
                dbp_preds.append(float(y_dbp[anc_i]) + delta[1])

            # Average across anchors
            all_pred_sbp.append(np.mean(sbp_preds))
            all_pred_dbp.append(np.mean(dbp_preds))
            all_true_sbp.append(true_sbp)
            all_true_dbp.append(true_dbp)

    pred = np.column_stack([
        np.array(all_pred_sbp) - np.array(all_true_sbp),  # error as "delta"
        np.array(all_pred_dbp) - np.array(all_true_dbp),
    ])
    # For metrics: pred = errors, target = zeros (so MAE = |pred|)
    # Actually, let's return as (N,2) absolute predictions and targets
    pred_abs = np.column_stack([all_pred_sbp, all_pred_dbp])
    true_abs = np.column_stack([all_true_sbp, all_true_dbp])
    return pred_abs, true_abs


# ──────────────────────────────────────────────
# Per-Patient Fine-Tuning
# ──────────────────────────────────────────────

def finetune_patient(model, patient_train_data, device, lr=1e-4,
                     n_steps=30, target_scale=10.0):
    """
    Fine-tune regression heads on a single patient's calibration windows.

    Only tunes: shared_trunk + sbp_head + dbp_head (freezes encoders).
    This adapts the BP mapping to the patient's specific PPG-BP relationship.

    Returns the fine-tuned model (a deepcopy — original is untouched).
    """
    ft_model = deepcopy(model)
    ft_model.train()

    # Freeze everything except regression heads
    for name, param in ft_model.named_parameters():
        if any(k in name for k in ['shared_trunk', 'sbp_head', 'dbp_head']):
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Count trainable params
    n_trainable = sum(p.numel() for p in ft_model.parameters() if p.requires_grad)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, ft_model.parameters()),
        lr=lr,
    )
    criterion = nn.HuberLoss(delta=1.0)

    # Unpack calibration data
    anchors_sig = patient_train_data["anchors_sig"].to(device)      # (N_cal, 3, W)
    currents_sig = patient_train_data["currents_sig"].to(device)    # (N_cal, 3, W)
    anc_sbp_norm = patient_train_data["anc_sbp_norm"].to(device)    # (N_cal,)
    anc_dbp_norm = patient_train_data["anc_dbp_norm"].to(device)    # (N_cal,)
    targets = patient_train_data["targets"].to(device)              # (N_cal, 2)
    anc_hf = patient_train_data["anc_hf"].to(device)                # (N_cal, 8)
    cur_hf = patient_train_data["cur_hf"].to(device)                # (N_cal, 8)
    anc_specs = patient_train_data.get("anc_specs")
    cur_specs = patient_train_data.get("cur_specs")
    if anc_specs is not None:
        anc_specs = anc_specs.to(device)
        cur_specs = cur_specs.to(device)

    n_cal = len(anchors_sig)

    for step in range(n_steps):
        # Mini-batch: use all calibration windows (usually small, 10-30)
        pred = ft_model(anchors_sig, currents_sig, anc_sbp_norm, anc_dbp_norm,
                        anchor_spec=anc_specs, current_spec=cur_specs,
                        anchor_hf=anc_hf, current_hf=cur_hf)
        loss = criterion(pred, targets)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(ft_model.parameters(), max_norm=2.0)
        optimizer.step()

    ft_model.eval()
    return ft_model


def build_patient_calibration_data(pid, train_indices, anchor_indices,
                                   X, y_sbp, y_dbp, patient_ids,
                                   target_scale, hand_features=None,
                                   X_spec=None, n_cal=20):
    """
    Build calibration tensors for a patient's training windows.

    Uses up to n_cal training windows paired with the patient's anchors.
    """
    # Limit calibration windows
    if len(train_indices) > n_cal:
        rng = np.random.default_rng(42)
        cal_indices = rng.choice(train_indices, n_cal, replace=False)
    else:
        cal_indices = train_indices

    # Pick anchor (first anchor)
    anc_i = anchor_indices[0] if isinstance(anchor_indices, list) else anchor_indices

    anchors_sig = []
    currents_sig = []
    targets = []
    anc_sbp_norms = []
    anc_dbp_norms = []
    anc_hfs = []
    cur_hfs = []
    anc_specs_list = []
    cur_specs_list = []

    for idx in cal_indices:
        anchors_sig.append(X[anc_i])
        currents_sig.append(X[idx])

        delta_sbp = (float(y_sbp[idx]) - float(y_sbp[anc_i])) / target_scale
        delta_dbp = (float(y_dbp[idx]) - float(y_dbp[anc_i])) / target_scale
        targets.append([delta_sbp, delta_dbp])

        anc_sbp_norms.append(float(y_sbp[anc_i]) / 100.0)
        anc_dbp_norms.append(float(y_dbp[anc_i]) / 100.0)

        if hand_features is not None:
            anc_hfs.append(hand_features[anc_i])
            cur_hfs.append(hand_features[idx])
        else:
            anc_hfs.append(np.zeros(N_HAND_FEATURES, dtype=np.float32))
            cur_hfs.append(np.zeros(N_HAND_FEATURES, dtype=np.float32))

        if X_spec is not None:
            anc_specs_list.append(np.array(X_spec[anc_i]))
            cur_specs_list.append(np.array(X_spec[idx]))

    data = {
        "anchors_sig": torch.tensor(np.array(anchors_sig), dtype=torch.float32),
        "currents_sig": torch.tensor(np.array(currents_sig), dtype=torch.float32),
        "targets": torch.tensor(np.array(targets), dtype=torch.float32),
        "anc_sbp_norm": torch.tensor(np.array(anc_sbp_norms), dtype=torch.float32),
        "anc_dbp_norm": torch.tensor(np.array(anc_dbp_norms), dtype=torch.float32),
        "anc_hf": torch.tensor(np.array(anc_hfs), dtype=torch.float32),
        "cur_hf": torch.tensor(np.array(cur_hfs), dtype=torch.float32),
    }

    if X_spec is not None:
        data["anc_specs"] = torch.tensor(np.array(anc_specs_list), dtype=torch.float32)
        data["cur_specs"] = torch.tensor(np.array(cur_specs_list), dtype=torch.float32)

    return data


def evaluate_with_calibration(base_model, test_ds, anchors, y_sbp, y_dbp,
                              patient_ids, device, target_scale,
                              train_mask, hand_features=None, X_spec=None,
                              n_cal=20, ft_steps=30, ft_lr=1e-4,
                              K_anchors=5, fusion=True):
    """
    Full calibrated evaluation pipeline:

    For each test patient:
    1. Build calibration data from their training windows
    2. Fine-tune regression heads on calibration data
    3. Predict test windows using fine-tuned model + multi-anchor ensemble

    Returns (pred_abs, true_abs) as (N_test, 2) arrays.
    """
    X = test_ds.X
    all_pred_sbp = []
    all_pred_dbp = []
    all_true_sbp = []
    all_true_dbp = []

    # Group test indices by patient
    pid_to_test_indices = {}
    for i in range(len(test_ds)):
        idx = test_ds.indices[i]
        pid = int(patient_ids[idx])
        pid_to_test_indices.setdefault(pid, []).append(idx)

    # Group train indices by patient
    pid_to_train_indices = {}
    train_indices_all = np.where(train_mask)[0]
    for idx in train_indices_all:
        pid = int(patient_ids[idx])
        pid_to_train_indices.setdefault(pid, []).append(idx)

    n_patients = len(pid_to_test_indices)
    print(f"\nCalibrating {n_patients} patients (n_cal={n_cal}, ft_steps={ft_steps}, "
          f"K={K_anchors}, lr={ft_lr})...")

    t0 = time.time()
    for pi, (pid, test_indices) in enumerate(pid_to_test_indices.items()):
        # Get patient's anchors
        anc_entry = anchors.get(pid)
        if anc_entry is None:
            continue

        anchor_list = anc_entry if isinstance(anc_entry, list) else [anc_entry]

        # Get training windows for calibration
        train_indices = pid_to_train_indices.get(pid, [])
        if len(train_indices) < 3:
            # Not enough calibration data — use base model
            ft_model = base_model
        else:
            # Build calibration data and fine-tune
            cal_data = build_patient_calibration_data(
                pid, train_indices, anchor_list,
                X, y_sbp, y_dbp, patient_ids,
                target_scale, hand_features, X_spec, n_cal=n_cal,
            )
            ft_model = finetune_patient(
                base_model, cal_data, device,
                lr=ft_lr, n_steps=ft_steps, target_scale=target_scale,
            )

        # Predict test windows with fine-tuned model + multi-anchor
        ft_model.eval()
        with torch.no_grad():
            for idx in test_indices:
                cur_sig = torch.tensor(X[idx], dtype=torch.float32).unsqueeze(0).to(device)

                if hand_features is not None:
                    cur_hf = torch.tensor(hand_features[idx], dtype=torch.float32).unsqueeze(0).to(device)
                else:
                    cur_hf = torch.zeros(1, N_HAND_FEATURES, dtype=torch.float32).to(device)

                if fusion and X_spec is not None:
                    cur_spec = torch.tensor(np.array(X_spec[idx]), dtype=torch.float32).unsqueeze(0).to(device)
                else:
                    cur_spec = None

                sbp_preds = []
                dbp_preds = []

                for anc_i in anchor_list[:K_anchors]:
                    anc_sig = torch.tensor(X[anc_i], dtype=torch.float32).unsqueeze(0).to(device)
                    anc_sbp_norm = torch.tensor([float(y_sbp[anc_i]) / 100.0], dtype=torch.float32).to(device)
                    anc_dbp_norm = torch.tensor([float(y_dbp[anc_i]) / 100.0], dtype=torch.float32).to(device)

                    if hand_features is not None:
                        anc_hf = torch.tensor(hand_features[anc_i], dtype=torch.float32).unsqueeze(0).to(device)
                    else:
                        anc_hf = torch.zeros(1, N_HAND_FEATURES, dtype=torch.float32).to(device)

                    if fusion and X_spec is not None:
                        anc_sp = torch.tensor(np.array(X_spec[anc_i]), dtype=torch.float32).unsqueeze(0).to(device)
                    else:
                        anc_sp = None

                    pred = ft_model(anc_sig, cur_sig, anc_sbp_norm, anc_dbp_norm,
                                    anchor_spec=anc_sp, current_spec=cur_spec,
                                    anchor_hf=anc_hf, current_hf=cur_hf)
                    delta = pred.cpu().numpy()[0] * target_scale

                    sbp_preds.append(float(y_sbp[anc_i]) + delta[0])
                    dbp_preds.append(float(y_dbp[anc_i]) + delta[1])

                all_pred_sbp.append(np.mean(sbp_preds))
                all_pred_dbp.append(np.mean(dbp_preds))
                all_true_sbp.append(float(y_sbp[idx]))
                all_true_dbp.append(float(y_dbp[idx]))

        # Clean up fine-tuned model to free memory
        if ft_model is not base_model:
            del ft_model
            if device == "cuda":
                torch.cuda.empty_cache()

        if (pi + 1) % 50 == 0 or pi == n_patients - 1:
            elapsed = time.time() - t0
            # Running metrics
            if len(all_pred_sbp) > 10:
                running_sbp = np.mean(np.abs(np.array(all_pred_sbp) - np.array(all_true_sbp)))
                running_dbp = np.mean(np.abs(np.array(all_pred_dbp) - np.array(all_true_dbp)))
                print(f"  Patient {pi+1}/{n_patients}  |  "
                      f"Running MAE: SBP={running_sbp:.2f}  DBP={running_dbp:.2f}  |  "
                      f"{elapsed:.0f}s")

    # Convert to delta format for metrics (pred - true as "errors" in delta space)
    pred_abs = np.column_stack([all_pred_sbp, all_pred_dbp])
    true_abs = np.column_stack([all_true_sbp, all_true_dbp])

    # Convert to delta format: pred_delta = pred_abs - true_abs + true_delta
    # Actually, for metrics we want pred and target in same format
    # Use: pred_col = [pred_sbp, pred_dbp], target_col = [true_sbp, true_dbp]
    # Then error = pred - target, which is what compute_metrics expects
    return pred_abs, true_abs


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  Per-Patient Calibration + Multi-Anchor Ensemble")
    print(f"  Device      : {device}")
    print(f"  Checkpoint  : {args.checkpoint}")
    print(f"  Cal windows : {args.n_cal}")
    print(f"  FT steps    : {args.ft_steps}")
    print(f"  FT lr       : {args.ft_lr}")
    print(f"  K anchors   : {args.k_anchors}")
    print(f"{'='*60}\n")

    # Load data
    train_ds, val_ds, test_ds, anchors, y_sbp, y_dbp, _ = load_data(
        args.data_dir,
        random_anchor=False,
        min_sbp_std=args.min_sbp_std,
        num_anchors=args.k_anchors,
        use_spectrograms=False,
        use_fusion=args.use_fusion,
        target_scale=args.target_scale,
        quality_threshold=args.quality_threshold,
        shuffle_eval=True,
    )

    patient_ids = np.load(os.path.join(args.data_dir, "patient_ids.npy"))

    # Reconstruct train_mask from train_ds indices
    train_mask = np.zeros(len(y_sbp), dtype=bool)
    train_mask[train_ds.indices] = True

    # Load hand features and spectrograms
    hf_path = os.path.join(args.data_dir, "X_hand_features.npy")
    hand_features = np.load(hf_path) if os.path.exists(hf_path) else None

    X_spec = None
    if args.use_fusion:
        spec_path = os.path.join(args.data_dir, "X_spectrograms.npy")
        if os.path.exists(spec_path):
            X_spec = np.load(spec_path, mmap_mode="r")

    # Load model
    model = build_model(embed_dim=256, dropout=0.3, device=device,
                        use_fusion=args.use_fusion)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # ── Step 1: Baseline (no calibration) ─────
    print("\n" + "=" * 56)
    print("  Step 1: Baseline — Single Anchor, No Fine-Tuning")
    print("=" * 56)

    # Use single anchor (first in list)
    single_anchors = {}
    for pid, anc in anchors.items():
        single_anchors[pid] = anc[0] if isinstance(anc, list) else anc

    baseline_pred, baseline_true = predict_multi_anchor(
        model, test_ds, single_anchors, y_sbp, y_dbp, patient_ids,
        device, args.target_scale, K=1, fusion=args.use_fusion,
        hand_features=hand_features, X_spec=X_spec,
    )
    # Convert absolute to delta for metrics
    baseline_delta_pred = baseline_pred - baseline_true
    zeros = np.zeros_like(baseline_delta_pred)
    m_base = print_metrics(
        np.column_stack([baseline_delta_pred[:, 0], baseline_delta_pred[:, 1]]),
        zeros,
        label="Baseline (1 anchor, no FT)"
    )

    # ── Step 2: Multi-Anchor Ensemble ─────────
    print("\n" + "=" * 56)
    print(f"  Step 2: Multi-Anchor Ensemble (K={args.k_anchors})")
    print("=" * 56)

    ma_pred, ma_true = predict_multi_anchor(
        model, test_ds, anchors, y_sbp, y_dbp, patient_ids,
        device, args.target_scale, K=args.k_anchors, fusion=args.use_fusion,
        hand_features=hand_features, X_spec=X_spec,
    )
    ma_delta_pred = ma_pred - ma_true
    m_ma = print_metrics(
        np.column_stack([ma_delta_pred[:, 0], ma_delta_pred[:, 1]]),
        zeros[:len(ma_delta_pred)],
        label=f"Multi-Anchor (K={args.k_anchors})"
    )

    # ── Step 3: Per-Patient Fine-Tuning + Multi-Anchor ──
    print("\n" + "=" * 56)
    print(f"  Step 3: Calibrated — Fine-Tune + Multi-Anchor")
    print("=" * 56)

    cal_pred, cal_true = evaluate_with_calibration(
        model, test_ds, anchors, y_sbp, y_dbp, patient_ids,
        device, args.target_scale,
        train_mask=train_mask,
        hand_features=hand_features,
        X_spec=X_spec,
        n_cal=args.n_cal,
        ft_steps=args.ft_steps,
        ft_lr=args.ft_lr,
        K_anchors=args.k_anchors,
        fusion=args.use_fusion,
    )
    cal_delta_pred = cal_pred - cal_true
    m_cal = print_metrics(
        np.column_stack([cal_delta_pred[:, 0], cal_delta_pred[:, 1]]),
        np.zeros_like(cal_delta_pred),
        label="Calibrated (FT + Multi-Anchor)"
    )

    # ── Summary Comparison ────────────────────
    print("\n" + "=" * 60)
    print("  COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\n  {'Method':<35} {'SBP MAE':>10} {'DBP MAE':>10} {'SBP BHS':>10} {'DBP BHS':>10}")
    print(f"  {'-'*75}")
    print(f"  {'Baseline (1 anchor)':<35} {m_base['mae_sbp']:>10.2f} {m_base['mae_dbp']:>10.2f} {m_base['bhs_sbp']:>10} {m_base['bhs_dbp']:>10}")
    print(f"  {'Multi-Anchor (K=' + str(args.k_anchors) + ')':<35} {m_ma['mae_sbp']:>10.2f} {m_ma['mae_dbp']:>10.2f} {m_ma['bhs_sbp']:>10} {m_ma['bhs_dbp']:>10}")
    print(f"  {'Calibrated (FT + MA)':<35} {m_cal['mae_sbp']:>10.2f} {m_cal['mae_dbp']:>10.2f} {m_cal['bhs_sbp']:>10} {m_cal['bhs_dbp']:>10}")
    print(f"\n  Schlesinger et al. (reference):")
    print(f"  {'  Siamese':<35} {'5.95':>10} {'3.41':>10}")
    print(f"  {'  Calibration-free':<35} {'7.34':>10} {'3.91':>10}")

    # ── Save results ──────────────────────────
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    np.save(save_path / "test_pred_baseline.npy", baseline_pred)
    np.save(save_path / "test_true_baseline.npy", baseline_true)
    np.save(save_path / "test_pred_multianchor.npy", ma_pred)
    np.save(save_path / "test_pred_calibrated.npy", cal_pred)
    np.save(save_path / "test_true_calibrated.npy", cal_true)

    # Save for plot_results.py compatibility
    np.save(save_path / "test_pred.npy", cal_delta_pred)
    np.save(save_path / "test_tgt.npy", np.zeros_like(cal_delta_pred))

    print(f"\n  Results saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-patient calibration evaluation")
    parser.add_argument("--data_dir", default=r"C:\MIMIC2_out_v8")
    parser.add_argument("--checkpoint", default=r".\checkpoints_v13\best_model.pt")
    parser.add_argument("--save_dir", default=r".\checkpoints_v14")
    parser.add_argument("--target_scale", type=float, default=10.0)
    parser.add_argument("--min_sbp_std", type=float, default=5.0)
    parser.add_argument("--quality_threshold", type=float, default=0.4)
    parser.add_argument("--use_fusion", action="store_true", default=True)
    parser.add_argument("--no_fusion", dest="use_fusion", action="store_false")

    # Calibration params
    parser.add_argument("--n_cal", type=int, default=20,
                        help="Number of calibration windows per patient")
    parser.add_argument("--ft_steps", type=int, default=30,
                        help="Fine-tuning gradient steps per patient")
    parser.add_argument("--ft_lr", type=float, default=1e-4,
                        help="Fine-tuning learning rate")
    parser.add_argument("--k_anchors", type=int, default=5,
                        help="Number of anchors for ensemble")

    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    main(args)
