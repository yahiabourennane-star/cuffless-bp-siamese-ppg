"""
evaluate_v2.py
==============

Test-set evaluation script for the Siamese BP estimation pipeline.

Notes:
- Anchor indices are recomputed from the evaluation split mask only.
- Combined MAE uses 0.6*SBP + 0.4*DBP to match the training metric.
- Metrics are reported in mmHg after denormalising model outputs.
- The script also prints BHS grades and saves diagnostic plots.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from siamese_dataset_v2 import SiameseMIMICDataset, SampleMeta
from siamese_model_v2 import SiameseNetwork


def collate_with_meta(batch):
    x_pairs, y_pairs, metas = zip(*batch)
    x_curr   = torch.stack([xp[0] for xp in x_pairs], dim=0)
    x_anchor = torch.stack([xp[1] for xp in x_pairs], dim=0)
    y_sbp    = torch.stack([yp[0] for yp in y_pairs], dim=0)
    y_dbp    = torch.stack([yp[1] for yp in y_pairs], dim=0)
    return (x_curr, x_anchor), (y_sbp, y_dbp), list(metas)


def cap_windows_per_patient(mask, patient_ids, max_windows, seed=42):
    rng = np.random.default_rng(seed)
    new_mask = np.zeros_like(mask, dtype=bool)
    for pid in np.unique(patient_ids[mask]):
        indices = np.where((patient_ids == pid) & mask)[0]
        if len(indices) > max_windows:
            indices = rng.choice(indices, max_windows, replace=False)
        new_mask[indices] = True
    return new_mask


def compute_anchor_indices(patient_ids, y_sbp_raw, split_mask):
    """
    FIX: Anchor computed from the evaluation split mask only.
    Test patients have no windows in train+val, so they must use their own
    windows to select a valid anchor. Anchor = window closest to median SBP.
    """
    pids = np.asarray(patient_ids, dtype=np.int64)
    n_patients = int(pids.max()) + 1
    anchor_k = np.zeros(n_patients, dtype=np.int64)
    for pid in range(n_patients):
        patient_global = np.where((pids == pid) & split_mask)[0]
        if len(patient_global) == 0:
            anchor_k[pid] = 0
            continue
        sbp_vals   = y_sbp_raw[patient_global].astype(np.float32)
        median_sbp = float(np.median(sbp_vals))
        anchor_k[pid] = int(np.argmin(np.abs(sbp_vals - median_sbp)))
    return anchor_k


def compute_metrics(pred, true):
    err  = pred - true
    return {
        "mae":  float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "me":   float(np.mean(err)),
        "std":  float(np.std(err)),
    }


def bhs_percentages(pred, true):
    err = np.abs(pred - true)
    return {
        "pct_le_5":  float(np.mean(err <= 5.0)  * 100),
        "pct_le_10": float(np.mean(err <= 10.0) * 100),
        "pct_le_15": float(np.mean(err <= 15.0) * 100),
    }


def bhs_grade(pct):
    if pct["pct_le_5"] >= 60 and pct["pct_le_10"] >= 85 and pct["pct_le_15"] >= 95: return "A"
    if pct["pct_le_5"] >= 50 and pct["pct_le_10"] >= 75 and pct["pct_le_15"] >= 90: return "B"
    if pct["pct_le_5"] >= 40 and pct["pct_le_10"] >= 65 and pct["pct_le_15"] >= 85: return "C"
    return "D"


def per_patient_summary(pred, true, pids):
    out, maes = {}, []
    for pid in np.unique(pids):
        idx = np.where(pids == pid)[0]
        mae = float(np.mean(np.abs(pred[idx] - true[idx])))
        out[int(pid)] = {"mae": mae, "n": int(len(idx))}
        maes.append(mae)
    maes = np.array(maes)
    return {"mean_patient_mae": float(maes.mean()), "std_patient_mae": float(maes.std()),
            "n_patients": int(len(maes)), "per_patient": out}


def bland_altman_plot(pred, true, title, save_path):
    mean, diff = (pred + true) / 2, pred - true
    md, sd = float(np.mean(diff)), float(np.std(diff))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(mean, diff, alpha=0.3, s=2)
    ax.axhline(md,              color="red",  linestyle="--", label=f"Bias: {md:.2f} mmHg")
    ax.axhline(md + 1.96 * sd, color="blue", linestyle="--", label=f"+1.96 SD: {md+1.96*sd:.2f}")
    ax.axhline(md - 1.96 * sd, color="blue", linestyle="--", label=f"-1.96 SD: {md-1.96*sd:.2f}")
    ax.axhline(0, color="black", linestyle="-", alpha=0.2)
    ax.set_xlabel("Mean of True & Predicted (mmHg)"); ax.set_ylabel("Predicted − True (mmHg)")
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches="tight"); plt.close()


def scatter_plot(pred, true, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(true, pred, alpha=0.3, s=2)
    lo, hi = float(min(true.min(), pred.min())), float(max(true.max(), pred.max()))
    ax.plot([lo, hi], [lo, hi], "r--", label="Identity")
    corr = float(np.corrcoef(true, pred)[0, 1]) if len(true) > 1 else float("nan")
    ax.set_xlabel("True (mmHg)"); ax.set_ylabel("Predicted (mmHg)")
    ax.set_title(f"{title}\nr = {corr:.3f}"); ax.legend(); ax.grid(True, alpha=0.3); ax.set_aspect("equal")
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches="tight"); plt.close()


def error_histogram(pred, true, title, save_path):
    errors = pred - true
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(errors, bins=60, edgecolor="black", alpha=0.7)
    ax.axvline(0,                      color="red",   linestyle="--", label="Zero error")
    ax.axvline(float(np.mean(errors)), color="green", linestyle="--", label=f"Mean: {np.mean(errors):.2f} mmHg")
    ax.set_xlabel("Error: Predicted − True (mmHg)"); ax.set_ylabel("Frequency")
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches="tight"); plt.close()


@torch.no_grad()
def run_inference(model, loader, device, amp, sbp_mean, sbp_std, dbp_mean, dbp_std):
    model.eval()
    use_amp = bool(amp and device.type == "cuda")
    _sbp_std  = torch.tensor(sbp_std,  device=device)
    _dbp_std  = torch.tensor(dbp_std,  device=device)
    dataset = loader.dataset
    abs_pred_sbp, abs_pred_dbp, abs_true_sbp, abs_true_dbp, pids_all = [], [], [], [], []

    for (x_curr, x_anchor), (y_sbp_norm, y_dbp_norm), meta in loader:
        x_curr   = x_curr.to(device,   non_blocking=True)
        x_anchor = x_anchor.to(device, non_blocking=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                sbp_delta_norm, dbp_delta_norm = model(x_curr, x_anchor)
        else:
            sbp_delta_norm, dbp_delta_norm = model(x_curr, x_anchor)

        sbp_delta_mm = (sbp_delta_norm * _sbp_std).detach().float().cpu().numpy().reshape(-1)
        dbp_delta_mm = (dbp_delta_norm * _dbp_std).detach().float().cpu().numpy().reshape(-1)

        anchor_indices  = np.array([m.anchor_idx for m in meta], dtype=np.int64)
        anchor_sbp_norm = dataset.sbp[anchor_indices].astype(np.float64)
        anchor_dbp_norm = dataset.dbp[anchor_indices].astype(np.float64)
        anchor_sbp_mm   = anchor_sbp_norm * sbp_std + sbp_mean
        anchor_dbp_mm   = anchor_dbp_norm * dbp_std + dbp_mean

        pred_sbp_mm = anchor_sbp_mm + sbp_delta_mm
        pred_dbp_mm = anchor_dbp_mm + dbp_delta_mm
        true_sbp_mm = anchor_sbp_mm + y_sbp_norm.numpy().reshape(-1) * sbp_std
        true_dbp_mm = anchor_dbp_mm + y_dbp_norm.numpy().reshape(-1) * dbp_std

        abs_pred_sbp.append(pred_sbp_mm); abs_pred_dbp.append(pred_dbp_mm)
        abs_true_sbp.append(true_sbp_mm); abs_true_dbp.append(true_dbp_mm)
        pids_all.extend([m.pid for m in meta])

    return (np.concatenate(abs_pred_sbp).astype(np.float64),
            np.concatenate(abs_pred_dbp).astype(np.float64),
            np.concatenate(abs_true_sbp).astype(np.float64),
            np.concatenate(abs_true_dbp).astype(np.float64),
            np.array(pids_all, dtype=np.int64))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--split",      type=str, default="test", choices=["test", "val"])
    p.add_argument("--model_base", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers",type=int, default=0)
    p.add_argument("--max_windows_per_patient", type=int, default=500)
    p.add_argument("--amp", action="store_true")
    args = p.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Checkpoint] Loading {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    norm_stats = ckpt["norm_stats"]
    sbp_mean, sbp_std = float(norm_stats["sbp_mean"]), float(norm_stats["sbp_std"])
    dbp_mean, dbp_std = float(norm_stats["dbp_mean"]), float(norm_stats["dbp_std"])
    ckpt_epoch = ckpt.get("epoch", "?")
    cfg        = ckpt.get("config", {})
    model_base = cfg.get("model_base", args.model_base)
    print(f"[Checkpoint] Epoch={ckpt_epoch} | model_base={model_base}")
    print(f"[Norm] SBP mean={sbp_mean:.1f}, std={sbp_std:.1f}")
    print(f"[Norm] DBP mean={dbp_mean:.1f}, std={dbp_std:.1f}")

    model = SiameseNetwork(dropout=0.0, base=model_base).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    print(f"[Model] Loaded. Params: {sum(p.numel() for p in model.parameters()):,}")

    dd = Path(args.data_dir)
    ppg_path   = dd / "X_ppg_windows.npy"
    multi_path = dd / "X_multi_channel.npy"
    if ppg_path.exists():
        X_ppg = np.load(ppg_path,          mmap_mode="r")
        X_vpg = np.load(dd / "X_vpg.npy",  mmap_mode="r")
        X_apg = np.load(dd / "X_apg.npy",  mmap_mode="r")
    elif multi_path.exists():
        X_multi = np.load(multi_path, mmap_mode="r")
        X_ppg, X_vpg, X_apg = X_multi[..., 0], X_multi[..., 1], X_multi[..., 2]
    else:
        raise FileNotFoundError(f"No signal arrays found in {dd}")

    y_sbp_raw   = np.load(dd / "y_sbp.npy",      mmap_mode="r")
    y_dbp_raw   = np.load(dd / "y_dbp.npy",      mmap_mode="r")
    patient_ids = np.load(dd / "patient_ids.npy", mmap_mode="r")
    split_mask  = np.load(dd / f"{args.split}_mask.npy")

    seed = cfg.get("seed", 42)
    cap  = args.max_windows_per_patient
    split_mask_capped = cap_windows_per_patient(split_mask, patient_ids, cap, seed) if cap > 0 else split_mask

    # FIX: use split_mask (not train+val) for anchor computation
    anchor_k = compute_anchor_indices(patient_ids, y_sbp_raw, split_mask_capped)
    print(f"[Anchor] Recomputed from {args.split} split mask (cap={cap}).")

    y_sbp_norm = ((y_sbp_raw - sbp_mean) / sbp_std).astype(np.float32)
    y_dbp_norm = ((y_dbp_raw - dbp_mean) / dbp_std).astype(np.float32)

    unique_pids = np.unique(patient_ids[split_mask])
    print(f"[Split] {args.split}: {int(split_mask.sum()):,} windows | {len(unique_pids)} patients")

    ds = SiameseMIMICDataset(
    X_ppg, X_vpg, X_apg, y_sbp_norm, y_dbp_norm,
    patient_ids, anchor_k, split_mask_capped,
        use_delta=True, random_anchor=False, augment=False, return_meta=True, verbose=True,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                        collate_fn=collate_with_meta)

    print(f"\n[Inference] Running on {args.split} set...")
    sbp_pred, dbp_pred, sbp_true, dbp_true, pids = run_inference(
        model, loader, device, use_amp, sbp_mean, sbp_std, dbp_mean, dbp_std)

    sbp_m  = compute_metrics(sbp_pred, sbp_true)
    dbp_m  = compute_metrics(dbp_pred, dbp_true)
    sbp_b  = bhs_percentages(sbp_pred, sbp_true)
    dbp_b  = bhs_percentages(dbp_pred, dbp_true)
    sbp_pp = per_patient_summary(sbp_pred, sbp_true, pids)
    dbp_pp = per_patient_summary(dbp_pred, dbp_true, pids)
    combined_mae = 0.6 * sbp_m["mae"] + 0.4 * dbp_m["mae"]

    print("\n" + "=" * 60)
    print(f"RESULTS ON {args.split.upper()} SET  (epoch {ckpt_epoch})")
    print("=" * 60)
    print(f"  SBP MAE:      {sbp_m['mae']:.2f} mmHg")
    print(f"  SBP RMSE:     {sbp_m['rmse']:.2f} mmHg")
    print(f"  SBP Bias:     {sbp_m['me']:.2f} ± {sbp_m['std']:.2f} mmHg")
    print(f"  DBP MAE:      {dbp_m['mae']:.2f} mmHg")
    print(f"  DBP RMSE:     {dbp_m['rmse']:.2f} mmHg")
    print(f"  DBP Bias:     {dbp_m['me']:.2f} ± {dbp_m['std']:.2f} mmHg")
    print(f"  Combined MAE: {combined_mae:.3f} mmHg  (0.6*SBP + 0.4*DBP)")
    print(f"\n  BHS Grading:")
    print(f"  SBP: ≤5mmHg={sbp_b['pct_le_5']:.1f}%  ≤10mmHg={sbp_b['pct_le_10']:.1f}%  ≤15mmHg={sbp_b['pct_le_15']:.1f}%  -> Grade {bhs_grade(sbp_b)}")
    print(f"  DBP: ≤5mmHg={dbp_b['pct_le_5']:.1f}%  ≤10mmHg={dbp_b['pct_le_10']:.1f}%  ≤15mmHg={dbp_b['pct_le_15']:.1f}%  -> Grade {bhs_grade(dbp_b)}")
    print(f"\n  Per-patient SBP MAE: {sbp_pp['mean_patient_mae']:.2f} ± {sbp_pp['std_patient_mae']:.2f} mmHg")
    print(f"  Per-patient DBP MAE: {dbp_pp['mean_patient_mae']:.2f} ± {dbp_pp['std_patient_mae']:.2f} mmHg")

    all_pp = {}
    for pid_k in set(list(sbp_pp["per_patient"]) + list(dbp_pp["per_patient"])):
        s = sbp_pp["per_patient"].get(pid_k, {}).get("mae", float("nan"))
        d = dbp_pp["per_patient"].get(pid_k, {}).get("mae", float("nan"))
        n = sbp_pp["per_patient"].get(pid_k, {}).get("n", 0)
        all_pp[pid_k] = {"sbp_mae": s, "dbp_mae": d, "combined": 0.6*s + 0.4*d, "n": n}
    worst = sorted(all_pp.items(), key=lambda x: x[1]["combined"], reverse=True)[:5]
    print(f"\n  Worst patients:")
    for pid_w, s in worst:
        print(f"    pid={pid_w} | n={s['n']} | SBP={s['sbp_mae']:.2f} DBP={s['dbp_mae']:.2f} Combined={s['combined']:.2f}")

    results = {
        "split": args.split, "checkpoint_epoch": ckpt_epoch,
        "n_samples": int(len(sbp_pred)), "n_patients": int(len(unique_pids)),
        "sbp": {**sbp_m, "bhs": sbp_b, "bhs_grade": bhs_grade(sbp_b), "per_patient": sbp_pp},
        "dbp": {**dbp_m, "bhs": dbp_b, "bhs_grade": bhs_grade(dbp_b), "per_patient": dbp_pp},
        "combined_mae_06_04": float(combined_mae),
        "combined_mae_05_05": float((sbp_m["mae"] + dbp_m["mae"]) / 2),
        "norm_stats": norm_stats,
    }
    results_path = out_dir / f"{args.split}_metrics.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\n[Saved] Metrics -> {results_path}")

    bland_altman_plot(sbp_pred, sbp_true, "SBP Bland-Altman",  out_dir / "bland_altman_sbp.png")
    bland_altman_plot(dbp_pred, dbp_true, "DBP Bland-Altman",  out_dir / "bland_altman_dbp.png")
    scatter_plot(sbp_pred,      sbp_true, "SBP: True vs Predicted", out_dir / "scatter_sbp.png")
    scatter_plot(dbp_pred,      dbp_true, "DBP: True vs Predicted", out_dir / "scatter_dbp.png")
    error_histogram(sbp_pred,   sbp_true, "SBP Error Distribution", out_dir / "error_hist_sbp.png")
    error_histogram(dbp_pred,   dbp_true, "DBP Error Distribution", out_dir / "error_hist_dbp.png")
    print(f"[Saved] Plots -> {out_dir}")

    print("\nEvaluation complete")
    print(f"   SBP MAE: {sbp_m['mae']:.2f} mmHg | DBP MAE: {dbp_m['mae']:.2f} mmHg | Combined: {combined_mae:.3f} mmHg")


if __name__ == "__main__":
    main()
