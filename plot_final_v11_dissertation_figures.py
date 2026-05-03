from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from metrics import compute_metrics


plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
})


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "dissertation_figures_final_v11"


def bhs_percentages(err_abs: np.ndarray) -> dict[int, float]:
    return {thr: float((err_abs <= thr).mean() * 100.0) for thr in (5, 10, 15)}


def style_axis(ax):
    ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def density_hexbin(ax, x, y, cmap, gridsize=80):
    return ax.hexbin(x, y, gridsize=gridsize, cmap=cmap, bins="log", mincnt=1, linewidths=0.0)


def metrics_with_bhs(pred: np.ndarray, tgt: np.ndarray) -> dict:
    m = compute_metrics(pred, tgt)
    err = np.abs(pred - tgt)
    m["sbp_bhs_pct"] = bhs_percentages(err[:, 0])
    m["dbp_bhs_pct"] = bhs_percentages(err[:, 1])
    return m


def plot_performance_summary(out_path: Path, clean_metrics: dict, personalized_metrics: dict):
    labels = [
        "Single model\nPPG-only",
        "Clean ensemble\nbeat-level labels",
        "Personalized\nsubject-calibrated",
        "ECG+PPG\nhybrid",
    ]
    sbp = np.array([7.50, clean_metrics["mae_sbp"], personalized_metrics["mae_sbp"], 9.42])
    dbp = np.array([3.93, clean_metrics["mae_dbp"], personalized_metrics["mae_dbp"], 5.24])
    combined = np.array([5.72, clean_metrics["combined_mae"], personalized_metrics["combined_mae"], 7.33])

    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.4, 5.6), constrained_layout=True)
    ax.bar(x - width, sbp, width, label="SBP MAE", color="#2d6f9f")
    ax.bar(x, dbp, width, label="DBP MAE", color="#c84630")
    ax.bar(x + width, combined, width, label="Combined MAE", color="#3b8f4f")
    for xpos, val in zip(x + width, combined):
        ax.text(xpos, val + 0.08, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MAE (mmHg)")
    ax.set_title("Comparison of Final System Variants")
    ax.legend(loc="upper left", frameon=True)
    ax.set_ylim(0, max(sbp.max(), combined.max()) + 1.2)
    style_axis(ax)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_pred_vs_true(pred: np.ndarray, tgt: np.ndarray, out_path: Path, title_suffix: str):
    m = compute_metrics(pred, tgt)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    for idx, label, cmap, mae_key, r_key in [
        (0, "SBP", "Blues", "mae_sbp", "r_sbp"),
        (1, "DBP", "Reds", "mae_dbp", "r_dbp"),
    ]:
        ax = axes[idx]
        true = tgt[:, idx]
        pred_i = pred[:, idx]
        lo = float(min(true.min(), pred_i.min()) - 2.0)
        hi = float(max(true.max(), pred_i.max()) + 2.0)
        density_hexbin(ax, true, pred_i, cmap=cmap, gridsize=85)
        ax.plot([lo, hi], [lo, hi], color="#111111", linestyle="--", linewidth=1.0)
        ax.plot([lo, hi], [lo + 5, hi + 5], color="#777777", linestyle=":", linewidth=0.8)
        ax.plot([lo, hi], [lo - 5, hi - 5], color="#777777", linestyle=":", linewidth=0.8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"True {label} (mmHg)")
        ax.set_ylabel(f"Predicted {label} (mmHg)")
        ax.set_title(f"{label}: Predicted vs True")
        ax.text(
            0.04,
            0.96,
            f"MAE = {m[mae_key]:.2f} mmHg\nr = {m[r_key]:.3f}",
            transform=ax.transAxes,
            va="top",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", alpha=0.94),
        )
        style_axis(ax)
    fig.suptitle(f"Predicted vs True Blood Pressure ({title_suffix})", fontsize=13)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_bhs_curves(pred: np.ndarray, tgt: np.ndarray, out_path: Path, title_suffix: str):
    err = np.abs(pred - tgt)
    thresholds = np.linspace(0.0, 20.0, 401)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    for idx, label, color in [(0, "SBP", "#2d6f9f"), (1, "DBP", "#c84630")]:
        ax = axes[idx]
        cumulative = np.array([(err[:, idx] <= t).mean() * 100.0 for t in thresholds])
        pct = bhs_percentages(err[:, idx])
        ax.plot(thresholds, cumulative, color=color, linewidth=2.2)
        for x_thr in (5, 10, 15):
            ax.axvline(x_thr, color="#777777", linestyle="--", linewidth=0.85)
        for y_thr in (50, 75, 90):
            ax.axhline(y_thr, color="#9a9a9a", linestyle=":", linewidth=0.85)
        ax.scatter([5, 10, 15], [pct[5], pct[10], pct[15]], color=color, s=35, zorder=3)
        ax.text(
            0.04,
            0.25,
            f"<=5 mmHg: {pct[5]:.1f}%\n<=10 mmHg: {pct[10]:.1f}%\n<=15 mmHg: {pct[15]:.1f}%",
            transform=ax.transAxes,
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", alpha=0.94),
        )
        ax.set_xlabel("Absolute Error Threshold (mmHg)")
        ax.set_ylabel("Predictions Within Threshold (%)")
        ax.set_title(f"{label}: Cumulative Error Curve")
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 100)
        style_axis(ax)
    fig.suptitle(f"BHS-Style Cumulative Error Curves ({title_suffix})", fontsize=13)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_bland_altman(pred: np.ndarray, tgt: np.ndarray, out_path: Path, title_suffix: str):
    diff = pred - tgt
    mean = (pred + tgt) / 2.0
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    for idx, label, cmap in [(0, "SBP", "Blues"), (1, "DBP", "Reds")]:
        ax = axes[idx]
        bias = float(diff[:, idx].mean())
        sd = float(diff[:, idx].std())
        upper = bias + 1.96 * sd
        lower = bias - 1.96 * sd
        density_hexbin(ax, mean[:, idx], diff[:, idx], cmap=cmap, gridsize=80)
        ax.axhline(bias, color="#111111", linewidth=1.1)
        ax.axhline(upper, color="#666666", linestyle="--", linewidth=0.95)
        ax.axhline(lower, color="#666666", linestyle="--", linewidth=0.95)
        ax.set_xlabel(f"Mean of True and Predicted {label} (mmHg)")
        ax.set_ylabel(f"Prediction Error ({label}) (mmHg)")
        ax.set_title(f"{label}: Bland-Altman")
        ax.text(
            0.97,
            0.97,
            f"Bias = {bias:.2f}\n+1.96 SD = {upper:.2f}\n-1.96 SD = {lower:.2f}",
            transform=ax.transAxes,
            va="top",
            ha="right",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", alpha=0.94),
        )
        style_axis(ax)
    fig.suptitle(f"Bland-Altman Agreement ({title_suffix})", fontsize=13)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_sbp_residuals(pred: np.ndarray, tgt: np.ndarray, out_path: Path, title_suffix: str):
    true_sbp = tgt[:, 0]
    resid = pred[:, 0] - tgt[:, 0]
    fig, ax = plt.subplots(figsize=(9.2, 5.5), constrained_layout=True)
    density_hexbin(ax, true_sbp, resid, cmap="Blues", gridsize=95)
    ax.axhline(0.0, color="#111111", linewidth=1.0)
    bins = np.linspace(true_sbp.min(), true_sbp.max(), 13)
    centers = []
    medians = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (true_sbp >= lo) & (true_sbp < hi)
        if mask.sum() >= 10:
            centers.append((lo + hi) / 2.0)
            medians.append(float(np.median(resid[mask])))
    if centers:
        ax.plot(centers, medians, color="#c84630", linewidth=2.2, marker="o", markersize=4, label="Bin median residual")
        ax.legend(loc="upper left", frameon=True)
    ax.set_xlabel("True SBP (mmHg)")
    ax.set_ylabel("SBP Residual (Predicted - True, mmHg)")
    ax.set_title(f"SBP Residuals vs True SBP ({title_suffix})")
    style_axis(ax)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_range_mae(pred: np.ndarray, tgt: np.ndarray, out_path: Path, title_suffix: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    configs = [
        (0, np.array([75, 90, 110, 130, 150, 166], dtype=np.float32), ["75-90", "90-110", "110-130", "130-150", "150-165"], "#2d6f9f", "SBP"),
        (1, np.array([40, 50, 60, 70, 86], dtype=np.float32), ["40-50", "50-60", "60-70", "70-85"], "#c84630", "DBP"),
    ]
    for idx, bins, labels, color, name in configs:
        vals = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (tgt[:, idx] >= lo) & (tgt[:, idx] < hi)
            vals.append(float(np.abs(pred[mask, idx] - tgt[mask, idx]).mean()) if mask.any() else np.nan)
        axes[idx].bar(labels, vals, color=color, alpha=0.88)
        axes[idx].set_title(f"{name} MAE by True {name} Range")
        axes[idx].set_xlabel(f"True {name} Range (mmHg)")
        axes[idx].set_ylabel("MAE (mmHg)")
        axes[idx].tick_params(axis="x", rotation=18)
        for xpos, val in enumerate(vals):
            if np.isfinite(val):
                axes[idx].text(xpos, val + 0.08, f"{val:.1f}", ha="center", va="bottom", fontsize=8.5)
        style_axis(axes[idx])
    fig.suptitle(f"Range-Stratified Mean Absolute Error ({title_suffix})", fontsize=13)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_readme(clean_m: dict, personalized_m: dict, out_path: Path):
    clean_sbp = clean_m["sbp_bhs_pct"]
    clean_dbp = clean_m["dbp_bhs_pct"]
    pers_sbp = personalized_m["sbp_bhs_pct"]
    pers_dbp = personalized_m["dbp_bhs_pct"]
    lines = [
        "Figure bundle generated from saved prediction arrays.",
        "",
        "Clean core v11 ensemble:",
        f"- SBP MAE = {clean_m['mae_sbp']:.2f} mmHg",
        f"- DBP MAE = {clean_m['mae_dbp']:.2f} mmHg",
        f"- Combined MAE = {clean_m['combined_mae']:.2f} mmHg",
        f"- RMSE = {clean_m['rmse_sbp']:.2f} / {clean_m['rmse_dbp']:.2f} mmHg",
        f"- Mean error = {clean_m['me_sbp']:.2f} / {clean_m['me_dbp']:.2f} mmHg",
        f"- Pearson r = {clean_m['r_sbp']:.3f} / {clean_m['r_dbp']:.3f}",
        f"- BHS = {clean_m['bhs_sbp']} / {clean_m['bhs_dbp']}",
        f"- SBP <=5/<=10/<=15 = {clean_sbp[5]:.2f}% / {clean_sbp[10]:.2f}% / {clean_sbp[15]:.2f}%",
        f"- DBP <=5/<=10/<=15 = {clean_dbp[5]:.2f}% / {clean_dbp[10]:.2f}% / {clean_dbp[15]:.2f}%",
        "",
        "Personalized subject-calibrated extension:",
        f"- SBP MAE = {personalized_m['mae_sbp']:.2f} mmHg",
        f"- DBP MAE = {personalized_m['mae_dbp']:.2f} mmHg",
        f"- Combined MAE = {personalized_m['combined_mae']:.2f} mmHg",
        f"- BHS = {personalized_m['bhs_sbp']} / {personalized_m['bhs_dbp']}",
        f"- SBP <=5/<=10/<=15 = {pers_sbp[5]:.2f}% / {pers_sbp[10]:.2f}% / {pers_sbp[15]:.2f}%",
        f"- DBP <=5/<=10/<=15 = {pers_dbp[5]:.2f}% / {pers_dbp[10]:.2f}% / {pers_dbp[15]:.2f}%",
        "",
        "Suggested use:",
        "- Use clean_v11 figures for the clean ensemble/core result section.",
        "- Use personalized figures only for the personalized subject-calibrated extension section or appendix.",
        "- Use Figure_6_performance_summary_v11.png for the overall comparison chart.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    clean_pred = np.load(ROOT / "ensemble_v11_abpbeat" / "ensemble_pred.npy").astype(np.float32)
    clean_tgt = np.load(ROOT / "ensemble_v11_abpbeat" / "ensemble_tgt.npy").astype(np.float32)
    personalized_pred = np.load(ROOT / "patient_adapt_full_heads_noaug_tta_subaffine" / "adapted_test_pred.npy").astype(np.float32)
    personalized_tgt = np.load(ROOT / "patient_adapt_full_heads_noaug_tta_subaffine" / "adapted_test_tgt.npy").astype(np.float32)

    clean_m = metrics_with_bhs(clean_pred, clean_tgt)
    personalized_m = metrics_with_bhs(personalized_pred, personalized_tgt)

    plot_performance_summary(OUT / "Figure_6_performance_summary_v11.png", clean_m, personalized_m)
    plot_pred_vs_true(clean_pred, clean_tgt, OUT / "Figure_7_clean_v11_pred_vs_true.png", "clean v11 ensemble")
    plot_bhs_curves(clean_pred, clean_tgt, OUT / "Figure_8_clean_v11_bhs_curves.png", "clean v11 ensemble")
    plot_sbp_residuals(clean_pred, clean_tgt, OUT / "Figure_9_clean_v11_sbp_residuals.png", "clean v11 ensemble")
    plot_range_mae(clean_pred, clean_tgt, OUT / "Figure_10_clean_v11_range_stratified_mae.png", "clean v11 ensemble")
    plot_bland_altman(clean_pred, clean_tgt, OUT / "Figure_A2_clean_v11_bland_altman.png", "clean v11 ensemble")

    # Keep a clearly separated personalized set so figures are not mixed into the clean ensemble section.
    plot_pred_vs_true(personalized_pred, personalized_tgt, OUT / "Personalized_pred_vs_true.png", "personalized subject-calibrated extension")
    plot_bhs_curves(personalized_pred, personalized_tgt, OUT / "Personalized_bhs_curves.png", "personalized subject-calibrated extension")
    plot_sbp_residuals(personalized_pred, personalized_tgt, OUT / "Personalized_sbp_residuals.png", "personalized subject-calibrated extension")
    plot_range_mae(personalized_pred, personalized_tgt, OUT / "Personalized_range_stratified_mae.png", "personalized subject-calibrated extension")
    plot_bland_altman(personalized_pred, personalized_tgt, OUT / "Personalized_bland_altman.png", "personalized subject-calibrated extension")

    write_readme(clean_m, personalized_m, OUT / "README.txt")
    (OUT / "metrics_summary.json").write_text(
        json.dumps({"clean_v11": clean_m, "personalized_extension": personalized_m}, indent=2),
        encoding="utf-8",
    )

    shutil.make_archive(str(OUT), "zip", OUT)
    print(f"Saved final v11 figures to {OUT}")
    print(f"Saved zip to {OUT}.zip")


if __name__ == "__main__":
    main()
