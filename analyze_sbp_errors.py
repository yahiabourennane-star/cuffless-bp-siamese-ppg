"""
analyze_sbp_errors.py
---------------------
Focused SBP error analysis for a saved checkpoint.

Reports where systolic error spread comes from:
  - true SBP range bins
  - mean anchor-gap bins
  - waveform quality bands
  - worst patients

Optionally fits a simple linear calibration on the validation set first.
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import load_data, make_derived_cache_path, compute_sbp_tail_multipliers
from evaluate import (
    run_eval,
    fit_linear_calibration,
    apply_linear_calibration,
    fit_subject_specific_calibration,
    apply_subject_specific_calibration,
    build_sbp_residual_feature_matrix,
    fit_sbp_residual_corrector,
    apply_sbp_residual_corrector,
)
from metrics import compute_metrics
from siamese_cnn import build_model
from train import run_multi_anchor_inference


def detect_fs_and_window(data_dir: str):
    ppg_path = os.path.join(data_dir, "X_ppg_windows.npy")
    ppg = np.load(ppg_path, mmap_mode="r", allow_pickle=False)
    window_len = int(ppg.shape[1])
    if window_len == 800:
        fs = 200
    elif window_len == 1000:
        fs = 100
    else:
        fs = 125
    return fs, window_len


def load_quality_scores(data_dir: str):
    fs, window_len = detect_fs_and_window(data_dir)
    qc_cache = make_derived_cache_path(data_dir, "X_quality_scores", fs, window_len)
    legacy_qc_cache = os.path.join(data_dir, "X_quality_scores.npy")
    if os.path.exists(qc_cache):
        return np.load(qc_cache, allow_pickle=False)
    if os.path.exists(legacy_qc_cache):
        return np.load(legacy_qc_cache, allow_pickle=False)
    return None


def parse_edges(edge_text: str):
    edges = [float(x.strip()) for x in edge_text.split(",") if x.strip()]
    if len(edges) < 2:
        raise ValueError("Need at least two bin edges.")
    return edges


def summarise_errors(errors: np.ndarray):
    abs_err = np.abs(errors)
    return {
        "count": int(len(errors)),
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mean_err": float(errors.mean()),
        "std_err": float(errors.std()),
        "p95_abs": float(np.percentile(abs_err, 95)),
    }


def table_by_edges(values: np.ndarray, errors: np.ndarray, edges, label: str):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == edges[-1]:
            mask = (values >= lo) & (values <= hi)
        else:
            mask = (values >= lo) & (values < hi)
        if not np.any(mask):
            continue
        row = summarise_errors(errors[mask])
        row[label] = f"[{lo:.1f}, {hi:.1f}{']' if hi == edges[-1] else ')'}"
        rows.append(row)
    return rows


def table_by_quantiles(values: np.ndarray, errors: np.ndarray, n_bins: int, label: str):
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values, qs)
    rows = []
    for i in range(len(edges) - 1):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if hi <= lo:
            continue
        if i == len(edges) - 2:
            mask = (values >= lo) & (values <= hi)
        else:
            mask = (values >= lo) & (values < hi)
        if not np.any(mask):
            continue
        row = summarise_errors(errors[mask])
        row[label] = f"Q{i+1} [{lo:.3f}, {hi:.3f}{']' if i == len(edges) - 2 else ')'}"
        rows.append(row)
    return rows


def patient_table(patient_ids: np.ndarray, errors: np.ndarray, min_count: int = 20, top_k: int = 10):
    rows = []
    for pid in np.unique(patient_ids):
        mask = patient_ids == pid
        if int(mask.sum()) < min_count:
            continue
        row = summarise_errors(errors[mask])
        row["patient_id"] = int(pid)
        rows.append(row)
    rows.sort(key=lambda r: (-r["mae"], -r["p95_abs"], -r["count"]))
    return rows[:top_k]


def print_table(title: str, rows, first_col: str):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)
    if not rows:
        print("  No rows to report.")
        return
    print(
        f"  {first_col:<20} {'N':>7} {'MAE':>8} {'RMSE':>8} "
        f"{'Mean':>8} {'Std':>8} {'P95|e|':>9}"
    )
    print(f"  {'-' * 70}")
    for row in rows:
        print(
            f"  {str(row[first_col]):<20} {row['count']:>7d} {row['mae']:>8.2f} "
            f"{row['rmse']:>8.2f} {row['mean_err']:>8.2f} {row['std_err']:>8.2f} {row['p95_abs']:>9.2f}"
        )


def save_csv(path: Path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_anchor_gap_stats(dataset, y_sbp):
    gaps = np.zeros(len(dataset.indices), dtype=np.float32)
    for row_i, idx in enumerate(dataset.indices):
        pid = int(dataset.patient_ids[idx])
        anc_list = dataset.anchor_lists[pid]
        anc_sbp = y_sbp[np.asarray(anc_list, dtype=np.int64)].astype(np.float32)
        gaps[row_i] = float(np.mean(np.abs(float(y_sbp[idx]) - anc_sbp)))
    return gaps


def predict_absolute_bp(model, dataset, loader, device, args):
    if args.multi_anchor_eval:
        pred_mmhg, tgt_mmhg = run_multi_anchor_inference(
            model, dataset, device,
            batch_size=args.eval_batch_size,
            fusion=args.use_fusion,
            n_aug=0,
            anchor_similarity_mode=args.anchor_similarity_mode,
            anchor_top_k=args.anchor_top_k,
            anchor_similarity_temp=args.anchor_similarity_temp,
        )
    else:
        pred, tgt = run_eval(model, loader, device, fusion=args.use_fusion)
        pred_mmhg = pred * args.target_scale
        tgt_mmhg = tgt * args.target_scale
    return pred_mmhg, tgt_mmhg


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n" + "=" * 72)
    print("  SBP Error Analysis")
    print("=" * 72)
    print(f"  Device      : {device}")
    print(f"  Checkpoint  : {args.checkpoint}")
    print(f"  Calibration : {'ON' if args.fit_output_calibration else 'OFF'}")
    if args.use_beat_features:
        print("  Beat aware  : ON (beat-summary feature stream)")
    if args.use_ecg:
        if args.ecg_input_mode == "both":
            ecg_desc = "ECG raw channel + timing features"
        elif args.ecg_input_mode == "features_only":
            ecg_desc = "ECG timing features only"
        else:
            ecg_desc = "ECG raw channel only"
        print(f"  Hybrid input: {ecg_desc}")
    if args.use_absolute_refinement:
        print("  Abs refine  : ON (aux absolute-BP refinement path)")
    if args.fit_subject_calibration:
        print(
            "  SubjectCal  : "
            f"ON ({args.subject_calibration_mode}, {args.subject_calibration_targets}, "
            f"min={args.subject_calibration_min_samples}, shrink={args.subject_calibration_shrinkage})"
        )
    if args.fit_sbp_residual_corrector:
        residual_desc = args.residual_model_type
        if args.residual_model_type == "piecewise_ridge":
            residual_desc = (
                f"piecewise_ridge(edges={args.residual_piecewise_edges}, "
                f"min={args.residual_piecewise_min_samples}, "
                f"blend={args.residual_piecewise_blend_width})"
            )
        print(f"  Residual    : ON ({residual_desc})")
    else:
        print("  Residual    : OFF")
    print(f"  Pair block  : {'gated interaction' if args.use_gated_pair_interaction else 'diff + hadamard'}")
    if args.multi_anchor_eval and args.anchor_similarity_mode != "none":
        topk_txt = f", top-k={args.anchor_top_k}" if args.anchor_top_k > 0 else ""
        eval_mode = f"multi-anchor ({args.anchor_similarity_mode}{topk_txt}, temp={args.anchor_similarity_temp})"
    else:
        eval_mode = "multi-anchor" if args.multi_anchor_eval else "single-anchor"
    print(f"  Eval mode   : {eval_mode}")

    train_ds, val_ds, test_ds, anchors, y_sbp, y_dbp, _ = load_data(
        args.data_dir,
        random_anchor=False,
        min_sbp_std=args.min_sbp_std,
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

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device == "cuda"),
    )

    model = build_model(
        embed_dim=256,
        dropout=0.3,
        device=device,
        use_spectrograms=args.use_spectrograms,
        use_fusion=args.use_fusion,
        use_pair_uncertainty=args.use_pair_uncertainty,
        use_gated_pair_interaction=args.use_gated_pair_interaction,
        use_absolute_refinement=args.use_absolute_refinement,
        hand_feature_dim=test_ds.hand_feature_dim,
        input_channels=test_ds.num_channels,
        target_scale=args.target_scale,
    )
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)

    cal_scale = np.ones(2, dtype=np.float32)
    cal_bias = np.zeros(2, dtype=np.float32)
    subject_cal_state = None
    val_pred_mmhg = None
    val_tgt_mmhg = None
    if args.fit_output_calibration:
        val_pred_mmhg, val_tgt_mmhg = predict_absolute_bp(model, val_ds, val_loader, device, args)
        cal_scale, cal_bias = fit_linear_calibration(
            val_pred_mmhg, val_tgt_mmhg, targets=args.calibration_targets
        )
        print(f"\n  Val calibration scale: SBP={cal_scale[0]:.4f} DBP={cal_scale[1]:.4f}")
        print(f"  Val calibration bias : SBP={cal_bias[0]:.4f} DBP={cal_bias[1]:.4f}")
    elif args.fit_subject_calibration or args.fit_sbp_residual_corrector:
        val_pred_mmhg, val_tgt_mmhg = predict_absolute_bp(model, val_ds, val_loader, device, args)

    val_patient_ids = val_ds.patient_ids[val_ds.indices].astype(np.int64)
    test_patient_ids = test_ds.patient_ids[test_ds.indices].astype(np.int64)
    if args.fit_subject_calibration:
        val_sub_base = val_pred_mmhg
        if args.fit_output_calibration:
            val_sub_base = apply_linear_calibration(val_pred_mmhg, cal_scale, cal_bias)
        subject_cal_state = fit_subject_specific_calibration(
            val_sub_base,
            val_tgt_mmhg,
            val_patient_ids,
            targets=args.subject_calibration_targets,
            mode=args.subject_calibration_mode,
            min_samples=args.subject_calibration_min_samples,
            shrinkage=args.subject_calibration_shrinkage,
        )

    test_pred_mmhg, test_tgt_mmhg = predict_absolute_bp(model, test_ds, test_loader, device, args)
    raw_metrics = compute_metrics(test_pred_mmhg, test_tgt_mmhg)
    print(
        f"\n  Raw test SBP: MAE={raw_metrics['mae_sbp']:.2f}  RMSE={raw_metrics['rmse_sbp']:.2f}  "
        f"ME={raw_metrics['me_sbp']:.2f}  STD={raw_metrics['std_sbp']:.2f}"
    )

    pred_for_analysis = test_pred_mmhg
    analysis_tag = "raw"
    if args.fit_output_calibration:
        pred_for_analysis = apply_linear_calibration(test_pred_mmhg, cal_scale, cal_bias)
        cal_metrics = compute_metrics(pred_for_analysis, test_tgt_mmhg)
        analysis_tag = f"cal_{args.calibration_targets}"
        print(
            f"  Cal test SBP: MAE={cal_metrics['mae_sbp']:.2f}  RMSE={cal_metrics['rmse_sbp']:.2f}  "
            f"ME={cal_metrics['me_sbp']:.2f}  STD={cal_metrics['std_sbp']:.2f}"
        )
    if subject_cal_state is not None:
        pred_for_analysis = apply_subject_specific_calibration(pred_for_analysis, test_patient_ids, subject_cal_state)
        sub_metrics = compute_metrics(pred_for_analysis, test_tgt_mmhg)
        analysis_tag = f"{analysis_tag}_subcal" if analysis_tag != "raw" else "subcal"
        print(
            f"  SubCal test SBP: MAE={sub_metrics['mae_sbp']:.2f}  RMSE={sub_metrics['rmse_sbp']:.2f}  "
            f"ME={sub_metrics['me_sbp']:.2f}  STD={sub_metrics['std_sbp']:.2f}"
        )

    if args.fit_sbp_residual_corrector:
        val_base = val_pred_mmhg
        if args.fit_output_calibration:
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
        test_feat = build_sbp_residual_feature_matrix(test_ds, pred_for_analysis)
        pred_for_analysis = apply_sbp_residual_corrector(pred_for_analysis, test_feat, residual_model)
        res_metrics = compute_metrics(pred_for_analysis, test_tgt_mmhg)
        analysis_tag = f"{analysis_tag}_sbpres" if analysis_tag != "raw" else "sbpres"
        print(
            f"  Res test SBP: MAE={res_metrics['mae_sbp']:.2f}  RMSE={res_metrics['rmse_sbp']:.2f}  "
            f"ME={res_metrics['me_sbp']:.2f}  STD={res_metrics['std_sbp']:.2f}"
        )

    indices = test_ds.indices
    true_sbp = test_tgt_mmhg[:, 0]
    pred_sbp = pred_for_analysis[:, 0]
    sbp_errors = pred_sbp - true_sbp
    patient_ids = test_ds.patient_ids[indices].astype(np.int64)

    quality_scores = load_quality_scores(args.data_dir)
    test_quality = quality_scores[indices] if quality_scores is not None else None
    anchor_gap = compute_anchor_gap_stats(test_ds, y_sbp)

    overall = summarise_errors(sbp_errors)
    print("\n" + "=" * 72)
    print("  Overall SBP Error")
    print("=" * 72)
    print(
        f"  Analyzed prediction set : {analysis_tag}\n"
        f"  Samples                 : {overall['count']}\n"
        f"  SBP MAE                 : {overall['mae']:.2f}\n"
        f"  SBP RMSE                : {overall['rmse']:.2f}\n"
        f"  SBP Mean Error          : {overall['mean_err']:.2f}\n"
        f"  SBP Std Error           : {overall['std_err']:.2f}\n"
        f"  SBP 95th |error|        : {overall['p95_abs']:.2f}"
    )

    sbp_rows = table_by_edges(true_sbp, sbp_errors, parse_edges(args.sbp_bins), "sbp_range")
    print_table("SBP Error by True SBP Range", sbp_rows, "sbp_range")

    gap_rows = table_by_edges(anchor_gap, sbp_errors, parse_edges(args.anchor_gap_bins), "anchor_gap")
    print_table("SBP Error by Mean Anchor Gap", gap_rows, "anchor_gap")

    if test_quality is not None:
        quality_rows = table_by_quantiles(test_quality, sbp_errors, n_bins=4, label="quality_band")
        print_table("SBP Error by Quality Quartile", quality_rows, "quality_band")
    else:
        quality_rows = []

    patient_rows = patient_table(patient_ids, sbp_errors, min_count=args.min_patient_windows, top_k=args.top_k_patients)
    print_table("Worst Patients by SBP MAE", patient_rows, "patient_id")

    save_dir = Path(os.path.dirname(args.checkpoint) or ".")
    stem = f"sbp_error_analysis_{analysis_tag}"
    save_csv(save_dir / f"{stem}_by_sbp_range.csv", sbp_rows,
             ["sbp_range", "count", "mae", "rmse", "mean_err", "std_err", "p95_abs"])
    save_csv(save_dir / f"{stem}_by_anchor_gap.csv", gap_rows,
             ["anchor_gap", "count", "mae", "rmse", "mean_err", "std_err", "p95_abs"])
    if quality_rows:
        save_csv(save_dir / f"{stem}_by_quality.csv", quality_rows,
                 ["quality_band", "count", "mae", "rmse", "mean_err", "std_err", "p95_abs"])
    save_csv(save_dir / f"{stem}_worst_patients.csv", patient_rows,
             ["patient_id", "count", "mae", "rmse", "mean_err", "std_err", "p95_abs"])
    np.save(save_dir / f"{stem}_sbp_errors.npy", sbp_errors.astype(np.float32))
    print(f"\n  Analysis files saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=r"C:\MIMIC2_out_v9")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--target_scale", type=float, default=10.0)
    parser.add_argument("--num_anchors", type=int, default=5)
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
    parser.add_argument("--sbp_bins", default="75,90,110,130,150,170")
    parser.add_argument("--anchor_gap_bins", default="0,5,10,15,20,30,60")
    parser.add_argument("--top_k_patients", type=int, default=10)
    parser.add_argument("--min_patient_windows", type=int, default=20)
    main(parser.parse_args())
