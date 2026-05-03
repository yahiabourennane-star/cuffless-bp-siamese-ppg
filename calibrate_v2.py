"""
calibrate_v2.py
---------------
Per-patient calibration using simple statistical corrections.

Instead of fine-tuning neural network weights (which didn't help),
this uses the patient's calibration windows to fit a linear correction:

    corrected_bp = a * predicted_bp + b

This fixes two problems:
1. Systematic bias (patient-specific offset)
2. Regression-to-mean (predictions too conservative → scale them up)

This is exactly what clinical BP devices do — calibrate with a cuff reading.

Usage:
    python calibrate_v2.py --checkpoint ./checkpoints_v13/best_model.pt --save_dir ./checkpoints_v14
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LinearRegression

from siamese_cnn import build_model
from dataset import load_data, N_HAND_FEATURES
from metrics import print_metrics


# ──────────────────────────────────────────────
# Predict all windows for a set of indices
# ──────────────────────────────────────────────

def predict_indices(model, indices, anchor_indices, X, y_sbp, y_dbp,
                    patient_ids, device, target_scale, K_anchors=5,
                    hand_features=None, X_spec=None, fusion=True):
    """
    Predict absolute SBP/DBP for a list of global indices.
    Uses multi-anchor ensemble (average over K anchors).
    Returns (pred_sbp, pred_dbp, true_sbp, true_dbp) as arrays.
    """
    model.eval()
    pred_sbp_list = []
    pred_dbp_list = []
    true_sbp_list = []
    true_dbp_list = []

    with torch.no_grad():
        for idx in indices:
            pid = int(patient_ids[idx])
            anc_list = anchor_indices.get(pid, [])
            if not isinstance(anc_list, list):
                anc_list = [anc_list]
            anc_list = anc_list[:K_anchors]

            if len(anc_list) == 0:
                continue

            cur_sig = torch.tensor(X[idx], dtype=torch.float32).unsqueeze(0).to(device)
            cur_hf = torch.tensor(hand_features[idx], dtype=torch.float32).unsqueeze(0).to(device) if hand_features is not None else torch.zeros(1, N_HAND_FEATURES, dtype=torch.float32).to(device)
            cur_spec = torch.tensor(np.array(X_spec[idx]), dtype=torch.float32).unsqueeze(0).to(device) if (fusion and X_spec is not None) else None

            sbp_preds = []
            dbp_preds = []

            for anc_i in anc_list:
                anc_sig = torch.tensor(X[anc_i], dtype=torch.float32).unsqueeze(0).to(device)
                anc_sbp_norm = torch.tensor([float(y_sbp[anc_i]) / 100.0], dtype=torch.float32).to(device)
                anc_dbp_norm = torch.tensor([float(y_dbp[anc_i]) / 100.0], dtype=torch.float32).to(device)
                anc_hf = torch.tensor(hand_features[anc_i], dtype=torch.float32).unsqueeze(0).to(device) if hand_features is not None else torch.zeros(1, N_HAND_FEATURES, dtype=torch.float32).to(device)
                anc_sp = torch.tensor(np.array(X_spec[anc_i]), dtype=torch.float32).unsqueeze(0).to(device) if (fusion and X_spec is not None) else None

                pred = model(anc_sig, cur_sig, anc_sbp_norm, anc_dbp_norm,
                             anchor_spec=anc_sp, current_spec=cur_spec,
                             anchor_hf=anc_hf, current_hf=cur_hf)
                delta = pred.cpu().numpy()[0] * target_scale

                sbp_preds.append(float(y_sbp[anc_i]) + delta[0])
                dbp_preds.append(float(y_dbp[anc_i]) + delta[1])

            pred_sbp_list.append(np.mean(sbp_preds))
            pred_dbp_list.append(np.mean(dbp_preds))
            true_sbp_list.append(float(y_sbp[idx]))
            true_dbp_list.append(float(y_dbp[idx]))

    return (np.array(pred_sbp_list), np.array(pred_dbp_list),
            np.array(true_sbp_list), np.array(true_dbp_list))


# ──────────────────────────────────────────────
# Calibration methods
# ──────────────────────────────────────────────

def bias_correction(cal_pred, cal_true, test_pred):
    """Simple offset: subtract mean error from predictions."""
    bias = np.mean(cal_pred - cal_true)
    return test_pred - bias


def linear_correction(cal_pred, cal_true, test_pred):
    """Linear regression: true = a*pred + b, then apply to test."""
    if len(cal_pred) < 3:
        return bias_correction(cal_pred, cal_true, test_pred)

    reg = LinearRegression()
    reg.fit(cal_pred.reshape(-1, 1), cal_true)
    corrected = reg.predict(test_pred.reshape(-1, 1))
    return corrected


def robust_linear_correction(cal_pred, cal_true, test_pred, clip_scale=3.0):
    """Linear regression with outlier clipping for robustness."""
    if len(cal_pred) < 5:
        return linear_correction(cal_pred, cal_true, test_pred)

    # Remove outliers from calibration data
    errors = cal_pred - cal_true
    mean_err = np.mean(errors)
    std_err = np.std(errors) + 1e-8
    keep = np.abs(errors - mean_err) < clip_scale * std_err

    if keep.sum() < 3:
        return linear_correction(cal_pred, cal_true, test_pred)

    reg = LinearRegression()
    reg.fit(cal_pred[keep].reshape(-1, 1), cal_true[keep])
    corrected = reg.predict(test_pred.reshape(-1, 1))
    return corrected


# ──────────────────────────────────────────────
# Main evaluation
# ──────────────────────────────────────────────

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  Per-Patient Calibration v2 (Statistical Correction)")
    print(f"  Device       : {device}")
    print(f"  Checkpoint   : {args.checkpoint}")
    print(f"  Cal windows  : {args.n_cal}")
    print(f"  K anchors    : {args.k_anchors}")
    print(f"  Method       : bias / linear / robust-linear")
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
    X = test_ds.X  # shared reference

    # Reconstruct train_mask
    train_mask = np.zeros(len(y_sbp), dtype=bool)
    train_mask[train_ds.indices] = True

    # Load features
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

    # ── Group indices by patient ──────────────
    pid_to_train = {}
    for idx in train_ds.indices:
        pid = int(patient_ids[idx])
        pid_to_train.setdefault(pid, []).append(idx)

    pid_to_test = {}
    for idx in test_ds.indices:
        pid = int(patient_ids[idx])
        pid_to_test.setdefault(pid, []).append(idx)

    # ── Step 1: Get all raw predictions (no correction) ──
    print("Step 1: Computing raw predictions for ALL windows...")
    t0 = time.time()

    # Predict all test windows
    all_test_pred_sbp, all_test_pred_dbp, all_test_true_sbp, all_test_true_dbp = predict_indices(
        model, test_ds.indices, anchors, X, y_sbp, y_dbp, patient_ids,
        device, args.target_scale, K_anchors=args.k_anchors,
        hand_features=hand_features, X_spec=X_spec, fusion=args.use_fusion,
    )
    print(f"  Test predictions done ({time.time()-t0:.0f}s)")

    # Baseline metrics
    print("\n" + "=" * 56)
    print("  Baseline — Raw Predictions (no calibration)")
    print("=" * 56)
    baseline_errors = np.column_stack([
        all_test_pred_sbp - all_test_true_sbp,
        all_test_pred_dbp - all_test_true_dbp,
    ])
    # For proper Pearson r, pass absolute pred and true
    baseline_pred_abs = np.column_stack([all_test_pred_sbp, all_test_pred_dbp])
    baseline_true_abs = np.column_stack([all_test_true_sbp, all_test_true_dbp])
    m_base = print_metrics_absolute(baseline_pred_abs, baseline_true_abs, "Baseline")

    # ── Step 2: Per-patient calibration predictions ──
    print(f"\nStep 2: Computing calibration predictions ({args.n_cal} windows/patient)...")
    t0 = time.time()

    # For each patient, predict their calibration (training) windows
    patient_cal_data = {}
    n_patients = len(pid_to_test)
    for pi, (pid, test_indices) in enumerate(pid_to_test.items()):
        train_indices = pid_to_train.get(pid, [])
        if len(train_indices) == 0:
            continue

        # Sample calibration windows
        rng = np.random.default_rng(42)
        n_cal = min(args.n_cal, len(train_indices))
        if len(train_indices) > n_cal:
            cal_indices = rng.choice(train_indices, n_cal, replace=False)
        else:
            cal_indices = np.array(train_indices)

        # Predict calibration windows
        cal_pred_sbp, cal_pred_dbp, cal_true_sbp, cal_true_dbp = predict_indices(
            model, cal_indices, anchors, X, y_sbp, y_dbp, patient_ids,
            device, args.target_scale, K_anchors=args.k_anchors,
            hand_features=hand_features, X_spec=X_spec, fusion=args.use_fusion,
        )

        patient_cal_data[pid] = {
            "cal_pred_sbp": cal_pred_sbp,
            "cal_pred_dbp": cal_pred_dbp,
            "cal_true_sbp": cal_true_sbp,
            "cal_true_dbp": cal_true_dbp,
        }

        if (pi + 1) % 100 == 0:
            print(f"  Patient {pi+1}/{n_patients} calibrated ({time.time()-t0:.0f}s)")

    print(f"  All calibration done ({time.time()-t0:.0f}s)")

    # ── Step 3: Apply corrections per patient ──
    methods = {
        "Bias Correction": bias_correction,
        "Linear Correction": linear_correction,
        "Robust Linear": robust_linear_correction,
    }

    # Track which test predictions belong to which patient
    # We need to map test_ds.indices to our prediction arrays
    idx_to_pos = {int(idx): pos for pos, idx in enumerate(test_ds.indices)}

    for method_name, correction_fn in methods.items():
        print(f"\n{'='*56}")
        print(f"  {method_name}")
        print(f"{'='*56}")

        corrected_sbp = all_test_pred_sbp.copy()
        corrected_dbp = all_test_pred_dbp.copy()

        n_corrected = 0
        for pid, test_indices in pid_to_test.items():
            cal = patient_cal_data.get(pid)
            if cal is None or len(cal["cal_pred_sbp"]) < 2:
                continue

            # Get positions of this patient's test windows in prediction arrays
            positions = [idx_to_pos[idx] for idx in test_indices if idx in idx_to_pos]
            if len(positions) == 0:
                continue

            positions = np.array(positions)

            # Apply correction
            corrected_sbp[positions] = correction_fn(
                cal["cal_pred_sbp"], cal["cal_true_sbp"],
                all_test_pred_sbp[positions]
            )
            corrected_dbp[positions] = correction_fn(
                cal["cal_pred_dbp"], cal["cal_true_dbp"],
                all_test_pred_dbp[positions]
            )
            n_corrected += 1

        print(f"  Corrected {n_corrected} patients")
        pred_abs = np.column_stack([corrected_sbp, corrected_dbp])
        true_abs = np.column_stack([all_test_true_sbp, all_test_true_dbp])
        m = print_metrics_absolute(pred_abs, true_abs, method_name)

    # ── Summary ───────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Schlesinger et al. (reference):")
    print(f"    Siamese:          SBP 5.95  DBP 3.41")
    print(f"    Calibration-free: SBP 7.34  DBP 3.91")
    print(f"{'='*60}")

    # Save best results
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    np.save(save_path / "test_pred_raw_sbp.npy", all_test_pred_sbp)
    np.save(save_path / "test_pred_raw_dbp.npy", all_test_pred_dbp)
    np.save(save_path / "test_true_sbp.npy", all_test_true_sbp)
    np.save(save_path / "test_true_dbp.npy", all_test_true_dbp)
    print(f"\n  Results saved to {save_path}")


def print_metrics_absolute(pred_abs, true_abs, label=""):
    """Compute and print metrics from absolute BP predictions."""
    err_sbp = pred_abs[:, 0] - true_abs[:, 0]
    err_dbp = pred_abs[:, 1] - true_abs[:, 1]

    mae_sbp = np.abs(err_sbp).mean()
    mae_dbp = np.abs(err_dbp).mean()
    combined = (mae_sbp + mae_dbp) / 2

    rmse_sbp = np.sqrt((err_sbp**2).mean())
    rmse_dbp = np.sqrt((err_dbp**2).mean())

    me_sbp = err_sbp.mean()
    me_dbp = err_dbp.mean()
    std_sbp = err_sbp.std()
    std_dbp = err_dbp.std()

    # Pearson r on absolute values (not errors)
    r_sbp = np.corrcoef(pred_abs[:, 0], true_abs[:, 0])[0, 1]
    r_dbp = np.corrcoef(pred_abs[:, 1], true_abs[:, 1])[0, 1]

    # BHS
    from metrics import bhs_grade
    bhs_sbp = bhs_grade(err_sbp)
    bhs_dbp = bhs_grade(err_dbp)

    # BHS detail
    abs_sbp = np.abs(err_sbp)
    abs_dbp = np.abs(err_dbp)
    p5_sbp  = (abs_sbp <= 5).mean() * 100
    p10_sbp = (abs_sbp <= 10).mean() * 100
    p15_sbp = (abs_sbp <= 15).mean() * 100
    p5_dbp  = (abs_dbp <= 5).mean() * 100
    p10_dbp = (abs_dbp <= 10).mean() * 100
    p15_dbp = (abs_dbp <= 15).mean() * 100

    tag = f"[{label}] " if label else ""
    print(f"\n{tag}Results ({len(pred_abs):,} windows)")
    print(f"  {'Metric':<20} {'SBP':>10} {'DBP':>10}")
    print(f"  {'-'*42}")
    print(f"  {'MAE (mmHg)':<20} {mae_sbp:>10.2f} {mae_dbp:>10.2f}")
    print(f"  {'Combined MAE':<20} {combined:>10.2f}")
    print(f"  {'RMSE (mmHg)':<20} {rmse_sbp:>10.2f} {rmse_dbp:>10.2f}")
    print(f"  {'Mean Error':<20} {me_sbp:>10.2f} {me_dbp:>10.2f}")
    print(f"  {'Std Error':<20} {std_sbp:>10.2f} {std_dbp:>10.2f}")
    print(f"  {'Pearson r':<20} {r_sbp:>10.3f} {r_dbp:>10.3f}")
    print(f"  {'BHS Grade':<20} {bhs_sbp:>10} {bhs_dbp:>10}")
    print(f"  {'≤5 mmHg %':<20} {p5_sbp:>9.1f}% {p5_dbp:>9.1f}%")
    print(f"  {'≤10 mmHg %':<20} {p10_sbp:>9.1f}% {p10_dbp:>9.1f}%")
    print(f"  {'≤15 mmHg %':<20} {p15_sbp:>9.1f}% {p15_dbp:>9.1f}%")

    return {
        "mae_sbp": mae_sbp, "mae_dbp": mae_dbp,
        "bhs_sbp": bhs_sbp, "bhs_dbp": bhs_dbp,
        "r_sbp": r_sbp, "r_dbp": r_dbp,
        "p5_sbp": p5_sbp, "p5_dbp": p5_dbp,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=r"C:\MIMIC2_out_v8")
    parser.add_argument("--checkpoint", default=r".\checkpoints_v13\best_model.pt")
    parser.add_argument("--save_dir", default=r".\checkpoints_v14")
    parser.add_argument("--target_scale", type=float, default=10.0)
    parser.add_argument("--min_sbp_std", type=float, default=5.0)
    parser.add_argument("--quality_threshold", type=float, default=0.4)
    parser.add_argument("--use_fusion", action="store_true", default=True)
    parser.add_argument("--no_fusion", dest="use_fusion", action="store_false")
    parser.add_argument("--n_cal", type=int, default=30,
                        help="Calibration windows per patient")
    parser.add_argument("--k_anchors", type=int, default=5,
                        help="Anchors for ensemble")
    args = parser.parse_args()
    main(args)
