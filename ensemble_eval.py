"""
ensemble_eval.py
----------------
Evaluate an ensemble of Siamese CNN models trained with different seeds.

Averaging predictions from multiple models reduces random error:
  - Each model makes different errors on the same window
  - Averaging cancels out uncorrelated noise
  - Expected improvement: ~15-25% MAE reduction with 3 models

Usage:
    python ensemble_eval.py
    python ensemble_eval.py --checkpoints ./checkpoints_ens_s42/best_model.pt ./checkpoints_ens_s123/best_model.pt ./checkpoints_ens_s7/best_model.pt
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from siamese_cnn import build_model
from dataset import load_data, N_HAND_FEATURES
from metrics import bhs_grade


# ──────────────────────────────────────────────
# Predict with a single model
# ──────────────────────────────────────────────

def predict_all(model, test_ds, anchors, y_sbp, y_dbp, patient_ids,
                device, target_scale, K=5, hand_features=None,
                X_spec=None, fusion=True):
    """Predict absolute SBP/DBP for all test windows using multi-anchor."""
    model.eval()
    X = test_ds.X
    pred_sbp = []
    pred_dbp = []
    true_sbp = []
    true_dbp = []

    with torch.no_grad():
        for i in range(len(test_ds)):
            idx = test_ds.indices[i]
            pid = int(patient_ids[idx])

            anc_entry = anchors.get(pid)
            if anc_entry is None:
                continue
            anc_list = anc_entry if isinstance(anc_entry, list) else [anc_entry]
            anc_list = anc_list[:K]

            cur_sig = torch.tensor(X[idx], dtype=torch.float32).unsqueeze(0).to(device)
            cur_hf = (torch.tensor(hand_features[idx], dtype=torch.float32).unsqueeze(0).to(device)
                      if hand_features is not None
                      else torch.zeros(1, N_HAND_FEATURES, dtype=torch.float32).to(device))
            cur_spec = (torch.tensor(np.array(X_spec[idx]), dtype=torch.float32).unsqueeze(0).to(device)
                        if (fusion and X_spec is not None) else None)

            s_preds, d_preds = [], []
            for anc_i in anc_list:
                anc_sig = torch.tensor(X[anc_i], dtype=torch.float32).unsqueeze(0).to(device)
                anc_sbp_n = torch.tensor([float(y_sbp[anc_i]) / 100.0], dtype=torch.float32).to(device)
                anc_dbp_n = torch.tensor([float(y_dbp[anc_i]) / 100.0], dtype=torch.float32).to(device)
                anc_hf = (torch.tensor(hand_features[anc_i], dtype=torch.float32).unsqueeze(0).to(device)
                          if hand_features is not None
                          else torch.zeros(1, N_HAND_FEATURES, dtype=torch.float32).to(device))
                anc_sp = (torch.tensor(np.array(X_spec[anc_i]), dtype=torch.float32).unsqueeze(0).to(device)
                          if (fusion and X_spec is not None) else None)

                pred = model(anc_sig, cur_sig, anc_sbp_n, anc_dbp_n,
                             anchor_spec=anc_sp, current_spec=cur_spec,
                             anchor_hf=anc_hf, current_hf=cur_hf)
                delta = pred.cpu().numpy()[0] * target_scale
                s_preds.append(float(y_sbp[anc_i]) + delta[0])
                d_preds.append(float(y_dbp[anc_i]) + delta[1])

            pred_sbp.append(np.mean(s_preds))
            pred_dbp.append(np.mean(d_preds))
            true_sbp.append(float(y_sbp[idx]))
            true_dbp.append(float(y_dbp[idx]))

            if (i + 1) % 10000 == 0:
                running_mae_s = np.mean(np.abs(np.array(pred_sbp) - np.array(true_sbp)))
                running_mae_d = np.mean(np.abs(np.array(pred_dbp) - np.array(true_dbp)))
                print(f"    {i+1}/{len(test_ds)}  SBP={running_mae_s:.2f}  DBP={running_mae_d:.2f}")

    return np.array(pred_sbp), np.array(pred_dbp), np.array(true_sbp), np.array(true_dbp)


def print_results(pred_sbp, pred_dbp, true_sbp, true_dbp, label=""):
    """Print comprehensive metrics."""
    err_sbp = pred_sbp - true_sbp
    err_dbp = pred_dbp - true_dbp

    mae_sbp = np.abs(err_sbp).mean()
    mae_dbp = np.abs(err_dbp).mean()
    combined = (mae_sbp + mae_dbp) / 2
    rmse_sbp = np.sqrt((err_sbp**2).mean())
    rmse_dbp = np.sqrt((err_dbp**2).mean())
    me_sbp = err_sbp.mean()
    me_dbp = err_dbp.mean()
    std_sbp = err_sbp.std()
    std_dbp = err_dbp.std()
    r_sbp = np.corrcoef(pred_sbp, true_sbp)[0, 1]
    r_dbp = np.corrcoef(pred_dbp, true_dbp)[0, 1]
    g_sbp = bhs_grade(err_sbp)
    g_dbp = bhs_grade(err_dbp)

    abs_sbp = np.abs(err_sbp)
    abs_dbp = np.abs(err_dbp)
    p5s  = (abs_sbp <= 5).mean() * 100
    p10s = (abs_sbp <= 10).mean() * 100
    p15s = (abs_sbp <= 15).mean() * 100
    p5d  = (abs_dbp <= 5).mean() * 100
    p10d = (abs_dbp <= 10).mean() * 100
    p15d = (abs_dbp <= 15).mean() * 100

    # AAMI pass/fail
    aami_sbp = "PASS" if abs(me_sbp) <= 5 and std_sbp <= 8 else "FAIL"
    aami_dbp = "PASS" if abs(me_dbp) <= 5 and std_dbp <= 8 else "FAIL"

    tag = f"[{label}] " if label else ""
    print(f"\n{tag}Results ({len(pred_sbp):,} windows)")
    print(f"  {'Metric':<20} {'SBP':>10} {'DBP':>10}")
    print(f"  {'-'*42}")
    print(f"  {'MAE (mmHg)':<20} {mae_sbp:>10.2f} {mae_dbp:>10.2f}")
    print(f"  {'Combined MAE':<20} {combined:>10.2f}")
    print(f"  {'RMSE (mmHg)':<20} {rmse_sbp:>10.2f} {rmse_dbp:>10.2f}")
    print(f"  {'Mean Error':<20} {me_sbp:>10.2f} {me_dbp:>10.2f}")
    print(f"  {'Std Error':<20} {std_sbp:>10.2f} {std_dbp:>10.2f}")
    print(f"  {'Pearson r':<20} {r_sbp:>10.3f} {r_dbp:>10.3f}")
    print(f"  {'BHS Grade':<20} {g_sbp:>10} {g_dbp:>10}")
    print(f"  {'AAMI':<20} {aami_sbp:>10} {aami_dbp:>10}")
    print(f"  {'≤5 mmHg %':<20} {p5s:>9.1f}% {p5d:>9.1f}%")
    print(f"  {'≤10 mmHg %':<20} {p10s:>9.1f}% {p10d:>9.1f}%")
    print(f"  {'≤15 mmHg %':<20} {p15s:>9.1f}% {p15d:>9.1f}%")

    return {
        "mae_sbp": mae_sbp, "mae_dbp": mae_dbp, "combined": combined,
        "bhs_sbp": g_sbp, "bhs_dbp": g_dbp,
        "r_sbp": r_sbp, "r_dbp": r_dbp,
        "p5_sbp": p5s, "p5_dbp": p5d,
        "aami_sbp": aami_sbp, "aami_dbp": aami_dbp,
    }


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoints = args.checkpoints
    n_models = len(checkpoints)

    print(f"\n{'='*60}")
    print(f"  Ensemble Evaluation — {n_models} models")
    print(f"  Device: {device}")
    for i, cp in enumerate(checkpoints):
        print(f"  Model {i+1}: {cp}")
    print(f"{'='*60}\n")

    # Load data (same split for all models — deterministic with shuffle_eval + seed=42)
    train_ds, val_ds, test_ds, anchors, y_sbp, y_dbp, _ = load_data(
        args.data_dir,
        random_anchor=False,
        min_sbp_std=5.0,
        num_anchors=5,
        use_spectrograms=False,
        use_fusion=True,
        target_scale=args.target_scale,
        quality_threshold=0.4,
        shuffle_eval=True,
    )

    patient_ids = np.load(os.path.join(args.data_dir, "patient_ids.npy"))

    hf_path = os.path.join(args.data_dir, "X_hand_features.npy")
    hand_features = np.load(hf_path) if os.path.exists(hf_path) else None

    X_spec = None
    spec_path = os.path.join(args.data_dir, "X_spectrograms.npy")
    if os.path.exists(spec_path):
        X_spec = np.load(spec_path, mmap_mode="r")

    # ── Predict with each model ───────────────
    all_sbp_preds = []
    all_dbp_preds = []
    true_sbp = None
    true_dbp = None

    for i, cp in enumerate(checkpoints):
        print(f"\n{'='*56}")
        print(f"  Model {i+1}/{n_models}: {cp}")
        print(f"{'='*56}")

        model = build_model(embed_dim=256, dropout=0.3, device=device, use_fusion=True)
        state = torch.load(cp, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        t0 = time.time()
        ps, pd, ts, td = predict_all(
            model, test_ds, anchors, y_sbp, y_dbp, patient_ids,
            device, args.target_scale, K=5,
            hand_features=hand_features, X_spec=X_spec, fusion=True,
        )
        elapsed = time.time() - t0
        print(f"  Predictions done ({elapsed:.0f}s)")

        m = print_results(ps, pd, ts, td, f"Model {i+1} (seed)")

        all_sbp_preds.append(ps)
        all_dbp_preds.append(pd)
        if true_sbp is None:
            true_sbp = ts
            true_dbp = td

        # Free GPU memory
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # ── Ensemble averaging ────────────────────
    print(f"\n{'='*60}")
    print(f"  ENSEMBLE — Average of {n_models} models")
    print(f"{'='*60}")

    ens_sbp = np.mean(all_sbp_preds, axis=0)
    ens_dbp = np.mean(all_dbp_preds, axis=0)

    m_ens = print_results(ens_sbp, ens_dbp, true_sbp, true_dbp, f"Ensemble ({n_models} models)")

    # ── Comparison ────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"\n  {'Method':<30} {'SBP MAE':>8} {'DBP MAE':>8} {'SBP BHS':>8} {'DBP BHS':>8}")
    print(f"  {'-'*65}")
    for i in range(n_models):
        ms = np.abs(all_sbp_preds[i] - true_sbp).mean()
        md = np.abs(all_dbp_preds[i] - true_dbp).mean()
        gs = bhs_grade(all_sbp_preds[i] - true_sbp)
        gd = bhs_grade(all_dbp_preds[i] - true_dbp)
        print(f"  {'Model ' + str(i+1):<30} {ms:>8.2f} {md:>8.2f} {gs:>8} {gd:>8}")
    print(f"  {'ENSEMBLE':<30} {m_ens['mae_sbp']:>8.2f} {m_ens['mae_dbp']:>8.2f} {m_ens['bhs_sbp']:>8} {m_ens['bhs_dbp']:>8}")
    print(f"\n  Schlesinger et al.:")
    print(f"  {'  Siamese':<30} {'5.95':>8} {'3.41':>8}")
    print(f"  {'  Calibration-free':<30} {'7.34':>8} {'3.91':>8}")

    # ── Save ──────────────────────────────────
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    np.save(save_path / "ensemble_pred_sbp.npy", ens_sbp)
    np.save(save_path / "ensemble_pred_dbp.npy", ens_dbp)
    np.save(save_path / "ensemble_true_sbp.npy", true_sbp)
    np.save(save_path / "ensemble_true_dbp.npy", true_dbp)

    # Save in delta format for plot_results.py
    test_pred = np.column_stack([ens_sbp - true_sbp, ens_dbp - true_dbp])
    test_tgt = np.zeros_like(test_pred)
    np.save(save_path / "test_pred.npy", test_pred)
    np.save(save_path / "test_tgt.npy", test_tgt)

    print(f"\n  Results saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=r"C:\MIMIC2_out_v8")
    parser.add_argument("--checkpoints", nargs="+", default=[
        r".\checkpoints_ens_s42\best_model.pt",
        r".\checkpoints_ens_s123\best_model.pt",
        r".\checkpoints_ens_s7\best_model.pt",
    ])
    parser.add_argument("--save_dir", default=r".\checkpoints_ensemble")
    parser.add_argument("--target_scale", type=float, default=10.0)
    args = parser.parse_args()
    main(args)
