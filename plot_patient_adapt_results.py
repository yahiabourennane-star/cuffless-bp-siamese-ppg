"""
Create plots from a saved prediction/target bundle.

Default input is the best personalized run:
    patient_adapt_full_heads_noaug_tta
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from metrics import compute_metrics


plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
})


def _load_manifest(results_dir: Path):
    manifest_path = results_dir / "patient_adaptation_manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return None


def _bhs_percentages(abs_err: np.ndarray):
    return {
        5: float((abs_err <= 5.0).mean() * 100.0),
        10: float((abs_err <= 10.0).mean() * 100.0),
        15: float((abs_err <= 15.0).mean() * 100.0),
    }


def _style_axis(ax):
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _density_hexbin(ax, x: np.ndarray, y: np.ndarray, cmap: str, gridsize: int = 75):
    return ax.hexbin(
        x,
        y,
        gridsize=gridsize,
        cmap=cmap,
        mincnt=1,
        bins="log",
        linewidths=0.0,
    )


def plot_pred_vs_true(pred: np.ndarray, tgt: np.ndarray, out_path: Path):
    metrics = compute_metrics(pred, tgt)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    specs = [
        (0, "SBP", "Blues", metrics["mae_sbp"], metrics["r_sbp"]),
        (1, "DBP", "Reds", metrics["mae_dbp"], metrics["r_dbp"]),
    ]
    for idx, title, cmap, mae, corr in specs:
        ax = axes[idx]
        x = tgt[:, idx]
        y = pred[:, idx]
        lo = float(min(x.min(), y.min()) - 2.0)
        hi = float(max(x.max(), y.max()) + 2.0)
        _density_hexbin(ax, x, y, cmap=cmap, gridsize=85)
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0, alpha=0.8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"True {title} (mmHg)")
        ax.set_ylabel(f"Predicted {title} (mmHg)")
        ax.set_title(f"{title}: Predicted vs True")
        ax.text(
            0.04, 0.96,
            f"MAE = {mae:.2f} mmHg\nr = {corr:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", alpha=0.92),
        )
        _style_axis(ax)
    fig.suptitle("Predicted vs True Blood Pressure on the Personalized Test Set", fontsize=13)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_bhs_curves(pred: np.ndarray, tgt: np.ndarray, out_path: Path):
    err = np.abs(pred - tgt)
    thresholds = np.linspace(0.0, 20.0, 401)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    specs = [
        (0, "SBP", "#1f77b4"),
        (1, "DBP", "#d62728"),
    ]
    for idx, title, color in specs:
        ax = axes[idx]
        cumulative = np.array([(err[:, idx] <= t).mean() * 100.0 for t in thresholds], dtype=np.float32)
        ax.plot(thresholds, cumulative, color=color, linewidth=2.2)
        for t in (5, 10, 15):
            ax.axvline(t, color="#888888", linestyle="--", linewidth=0.9)
        for y_thr in (50, 75, 90):
            ax.axhline(y_thr, color="#9a9a9a", linestyle=":", linewidth=0.9, alpha=0.9)
        pct = _bhs_percentages(err[:, idx])
        ax.scatter([5, 10, 15], [pct[5], pct[10], pct[15]], color=color, s=34, zorder=3)
        ax.text(
            0.04, 0.24,
            f"<=5 mmHg: {pct[5]:.1f}%\n<=10 mmHg: {pct[10]:.1f}%\n<=15 mmHg: {pct[15]:.1f}%",
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", alpha=0.92),
        )
        ax.set_xlabel("Absolute Error Threshold (mmHg)")
        ax.set_ylabel("Predictions Within Threshold (%)")
        ax.set_title(f"{title}: BHS Cumulative Error Curve")
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 100)
        _style_axis(ax)
    fig.suptitle("BHS-Style Cumulative Error Curves", fontsize=13)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_bland_altman(pred: np.ndarray, tgt: np.ndarray, out_path: Path):
    diff = pred - tgt
    avg = (pred + tgt) / 2.0
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    specs = [(0, "SBP", "Blues"), (1, "DBP", "Reds")]
    for idx, title, cmap in specs:
        ax = axes[idx]
        bias = float(diff[:, idx].mean())
        sd = float(diff[:, idx].std())
        loa = 1.96 * sd
        _density_hexbin(ax, avg[:, idx], diff[:, idx], cmap=cmap, gridsize=80)
        ax.axhline(bias, color="black", linewidth=1.2)
        ax.axhline(bias + loa, color="#666666", linestyle="--", linewidth=1.0)
        ax.axhline(bias - loa, color="#666666", linestyle="--", linewidth=1.0)
        ax.set_xlabel(f"Mean of True and Predicted {title} (mmHg)")
        ax.set_ylabel(f"Prediction Error ({title}) (mmHg)")
        ax.set_title(f"{title}: Bland-Altman Plot")
        ax.text(
            0.97, 0.97,
            f"Bias = {bias:.2f}\n+1.96 SD = {bias + loa:.2f}\n-1.96 SD = {bias - loa:.2f}",
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", alpha=0.92),
        )
        _style_axis(ax)
    fig.suptitle("Bland-Altman Analysis", fontsize=13)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_sbp_residuals(pred: np.ndarray, tgt: np.ndarray, out_path: Path):
    true_sbp = tgt[:, 0]
    resid = pred[:, 0] - tgt[:, 0]
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    _density_hexbin(ax, true_sbp, resid, cmap="Blues", gridsize=95)
    ax.axhline(0.0, color="black", linewidth=1.0)

    bins = np.linspace(true_sbp.min(), true_sbp.max(), 13)
    centers = []
    medians = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (true_sbp >= lo) & (true_sbp < hi)
        if mask.sum() < 10:
            continue
        centers.append((lo + hi) / 2.0)
        medians.append(float(np.median(resid[mask])))
    if centers:
        ax.plot(centers, medians, color="#d62728", linewidth=2.2, marker="o", markersize=4, label="Bin median residual")
        ax.legend(loc="upper left", frameon=True)

    ax.set_xlabel("True SBP (mmHg)")
    ax.set_ylabel("SBP Residual (Predicted - True, mmHg)")
    ax.set_title("SBP Residuals vs True SBP")
    _style_axis(ax)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_range_mae(pred: np.ndarray, tgt: np.ndarray, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # SBP bins
    sbp_bins = np.array([75, 90, 110, 130, 150, 166], dtype=np.float32)
    sbp_labels = ["75-90", "90-110", "110-130", "130-150", "150-165"]
    sbp_mae = []
    for lo, hi in zip(sbp_bins[:-1], sbp_bins[1:]):
        mask = (tgt[:, 0] >= lo) & (tgt[:, 0] < hi)
        sbp_mae.append(float(np.abs(pred[mask, 0] - tgt[mask, 0]).mean()) if mask.any() else np.nan)
    axes[0].bar(sbp_labels, sbp_mae, color="#1f77b4", alpha=0.85)
    axes[0].set_title("SBP MAE by True SBP Range")
    axes[0].set_xlabel("True SBP Range (mmHg)")
    axes[0].set_ylabel("MAE (mmHg)")
    axes[0].tick_params(axis="x", rotation=20)
    _style_axis(axes[0])

    # DBP bins
    dbp_bins = np.array([40, 50, 60, 70, 86], dtype=np.float32)
    dbp_labels = ["40-50", "50-60", "60-70", "70-85"]
    dbp_mae = []
    for lo, hi in zip(dbp_bins[:-1], dbp_bins[1:]):
        mask = (tgt[:, 1] >= lo) & (tgt[:, 1] < hi)
        dbp_mae.append(float(np.abs(pred[mask, 1] - tgt[mask, 1]).mean()) if mask.any() else np.nan)
    axes[1].bar(dbp_labels, dbp_mae, color="#d62728", alpha=0.85)
    axes[1].set_title("DBP MAE by True DBP Range")
    axes[1].set_xlabel("True DBP Range (mmHg)")
    axes[1].set_ylabel("MAE (mmHg)")
    axes[1].tick_params(axis="x", rotation=20)
    _style_axis(axes[1])

    fig.suptitle("Range-Stratified Mean Absolute Error", fontsize=13)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_summary(pred: np.ndarray, tgt: np.ndarray, manifest: dict | None, out_path: Path):
    personalized_metrics = compute_metrics(pred, tgt)
    subject_calibrated = bool(manifest and manifest.get("fit_subject_calibration", False))
    personalized_label = "Personalized\nensemble"
    if subject_calibrated:
        personalized_label = "Personalized +\nsubject affine"

    labels = [
        "Single model\nPPG-only",
        "3-model ensemble",
        personalized_label,
        "ECG+PPG hybrid",
    ]
    combined = np.array([
        5.72,
        5.65,
        personalized_metrics["combined_mae"],
        7.33,
    ], dtype=np.float32)
    sbp = np.array([
        7.50,
        7.39,
        personalized_metrics["mae_sbp"],
        9.42,
    ], dtype=np.float32)
    dbp = np.array([
        3.93,
        3.91,
        personalized_metrics["mae_dbp"],
        5.24,
    ], dtype=np.float32)

    x = np.arange(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    ax.bar(x - width, sbp, width, label="SBP MAE", color="#1f77b4")
    ax.bar(x, dbp, width, label="DBP MAE", color="#d62728")
    ax.bar(x + width, combined, width, label="Combined MAE", color="#2ca02c")
    for xpos, val in zip(x + width, combined):
        ax.text(xpos, val + 0.06, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MAE (mmHg)")
    ax.set_title("Comparison of Final System Variants")
    ax.legend(loc="upper right", frameon=True)
    _style_axis(ax)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_summary(pred: np.ndarray, tgt: np.ndarray, manifest: dict | None, out_path: Path):
    metrics = compute_metrics(pred, tgt)
    err = np.abs(pred - tgt)
    bhs_sbp = _bhs_percentages(err[:, 0])
    bhs_dbp = _bhs_percentages(err[:, 1])
    lines = []
    if manifest is not None:
        lines.append(f"Source directory: {manifest.get('data_dir', 'unknown')}")
        lines.append(f"TTA passes: {manifest.get('tta', 'unknown')}")
    lines.extend([
        "",
        "Final personalized test metrics",
        f"SBP MAE: {metrics['mae_sbp']:.3f} mmHg",
        f"DBP MAE: {metrics['mae_dbp']:.3f} mmHg",
        f"Combined MAE: {metrics['combined_mae']:.3f} mmHg",
        f"SBP RMSE: {metrics['rmse_sbp']:.3f} mmHg",
        f"DBP RMSE: {metrics['rmse_dbp']:.3f} mmHg",
        f"SBP mean error: {metrics['me_sbp']:.3f} mmHg",
        f"DBP mean error: {metrics['me_dbp']:.3f} mmHg",
        f"SBP std error: {metrics['std_sbp']:.3f} mmHg",
        f"DBP std error: {metrics['std_dbp']:.3f} mmHg",
        f"SBP Pearson r: {metrics['r_sbp']:.3f}",
        f"DBP Pearson r: {metrics['r_dbp']:.3f}",
        f"SBP BHS grade: {metrics['bhs_sbp']}",
        f"DBP BHS grade: {metrics['bhs_dbp']}",
        "",
        "BHS percentages",
        f"SBP <=5 / <=10 / <=15 mmHg: {bhs_sbp[5]:.2f}% / {bhs_sbp[10]:.2f}% / {bhs_sbp[15]:.2f}%",
        f"DBP <=5 / <=10 / <=15 mmHg: {bhs_dbp[5]:.2f}% / {bhs_dbp[10]:.2f}% / {bhs_dbp[15]:.2f}%",
    ])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate final result plots from saved prediction arrays.")
    parser.add_argument("--results_dir", default=r".\patient_adapt_full_heads_noaug_tta")
    parser.add_argument("--pred_file", default="adapted_test_pred.npy")
    parser.add_argument("--tgt_file", default="adapted_test_tgt.npy")
    parser.add_argument("--out_dir", default="")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = np.load(results_dir / args.pred_file)
    tgt = np.load(results_dir / args.tgt_file)
    manifest = _load_manifest(results_dir)

    plot_pred_vs_true(pred, tgt, out_dir / "figure_pred_vs_true.png")
    plot_bhs_curves(pred, tgt, out_dir / "figure_bhs_curves.png")
    plot_bland_altman(pred, tgt, out_dir / "figure_bland_altman.png")
    plot_sbp_residuals(pred, tgt, out_dir / "figure_sbp_residuals.png")
    plot_range_mae(pred, tgt, out_dir / "figure_range_stratified_mae.png")
    plot_comparison_summary(pred, tgt, manifest, out_dir / "figure_performance_summary.png")
    write_summary(pred, tgt, manifest, out_dir / "figure_metrics_summary.txt")

    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()
