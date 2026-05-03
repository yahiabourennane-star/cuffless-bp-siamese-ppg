"""
plot_results.py
---------------
Generate publication-ready plots for supervisor presentation.

Produces:
  1. Training curves (loss + MAE over epochs)
  2. Bland-Altman plots (SBP and DBP)
  3. Scatter plots (predicted vs true)
  4. Error distribution histograms with BHS thresholds
  5. Summary table

Usage:
    python plot_results.py --save_dir "./checkpoints_v11"
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from metrics import compute_metrics, bhs_grade


def load_results(save_dir):
    history = np.load(os.path.join(save_dir, "history.npy"), allow_pickle=True).item()
    test_pred = np.load(os.path.join(save_dir, "test_pred.npy"))
    test_tgt = np.load(os.path.join(save_dir, "test_tgt.npy"))

    # Check for TTA results
    tta_path = os.path.join(save_dir, "test_pred_tta.npy")
    test_pred_tta = np.load(tta_path) if os.path.exists(tta_path) else None

    return history, test_pred, test_tgt, test_pred_tta


# ──────────────────────────────────────────────
# Plot 1: Training curves
# ──────────────────────────────────────────────

def plot_training_curves(history, save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Progress", fontsize=16, fontweight="bold")
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], label="Train (augmented)", color="#2196F3", linewidth=1.5)
    ax.plot(epochs, history["train_eval_loss"], label="Train (clean)", color="#4CAF50", linewidth=1.5)
    ax.plot(epochs, history["val_loss"], label="Validation", color="#FF5722", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Gap
    ax = axes[0, 1]
    gap = [v - t for v, t in zip(history["val_loss"], history["train_eval_loss"])]
    ax.plot(epochs, gap, color="#9C27B0", linewidth=1.5)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val - Train Loss")
    ax.set_title("Generalisation Gap")
    ax.grid(True, alpha=0.3)

    # SBP MAE
    ax = axes[1, 0]
    ax.plot(epochs, history["val_mae_sbp"], color="#F44336", linewidth=1.5, label="SBP MAE")
    ax.axhline(y=5.0, color="green", linestyle="--", alpha=0.5, label="BHS-A target (~5)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (mmHg)")
    ax.set_title("Systolic BP - Validation MAE")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # DBP MAE
    ax = axes[1, 1]
    ax.plot(epochs, history["val_mae_dbp"], color="#2196F3", linewidth=1.5, label="DBP MAE")
    ax.axhline(y=5.0, color="green", linestyle="--", alpha=0.5, label="BHS-A target (~5)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (mmHg)")
    ax.set_title("Diastolic BP - Validation MAE")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "fig1_training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────
# Plot 2: Bland-Altman plots
# ──────────────────────────────────────────────

def plot_bland_altman(test_pred, test_tgt, save_dir, label=""):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    tag = f" ({label})" if label else ""
    fig.suptitle(f"Bland-Altman Analysis{tag}", fontsize=16, fontweight="bold")

    for idx, (ax, name) in enumerate(zip(axes, ["SBP", "DBP"])):
        pred = test_pred[:, idx]
        true = test_tgt[:, idx]
        mean_vals = (pred + true) / 2
        diff = pred - true

        mean_diff = diff.mean()
        std_diff = diff.std()

        ax.scatter(mean_vals, diff, alpha=0.05, s=3, color="#2196F3", rasterized=True)
        ax.axhline(mean_diff, color="red", linewidth=1.5, label=f"Mean: {mean_diff:.2f}")
        ax.axhline(mean_diff + 1.96 * std_diff, color="orange", linestyle="--", linewidth=1,
                    label=f"+1.96 SD: {mean_diff + 1.96 * std_diff:.1f}")
        ax.axhline(mean_diff - 1.96 * std_diff, color="orange", linestyle="--", linewidth=1,
                    label=f"-1.96 SD: {mean_diff - 1.96 * std_diff:.1f}")
        ax.set_xlabel(f"Mean of Predicted and True {name} (mmHg)")
        ax.set_ylabel(f"Predicted - True {name} (mmHg)")
        ax.set_title(f"{name}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    suffix = f"_{label}" if label else ""
    path = os.path.join(save_dir, f"fig2_bland_altman{suffix}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────
# Plot 3: Scatter plots (pred vs true delta)
# ──────────────────────────────────────────────

def plot_scatter(test_pred, test_tgt, save_dir, label=""):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    tag = f" ({label})" if label else ""
    fig.suptitle(f"Predicted vs True BP Delta{tag}", fontsize=16, fontweight="bold")

    for idx, (ax, name) in enumerate(zip(axes, ["SBP", "DBP"])):
        pred = test_pred[:, idx]
        true = test_tgt[:, idx]

        r = np.corrcoef(pred, true)[0, 1]
        mae = np.abs(pred - true).mean()

        ax.scatter(true, pred, alpha=0.05, s=3, color="#4CAF50", rasterized=True)

        # Identity line
        lim_min = min(true.min(), pred.min()) - 5
        lim_max = max(true.max(), pred.max()) + 5
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", linewidth=1, alpha=0.7)

        ax.set_xlabel(f"True Delta {name} (mmHg)")
        ax.set_ylabel(f"Predicted Delta {name} (mmHg)")
        ax.set_title(f"{name}  |  r={r:.3f}  MAE={mae:.2f}")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    suffix = f"_{label}" if label else ""
    path = os.path.join(save_dir, f"fig3_scatter{suffix}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────
# Plot 4: Error distribution with BHS thresholds
# ──────────────────────────────────────────────

def plot_error_distribution(test_pred, test_tgt, save_dir, label=""):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    tag = f" ({label})" if label else ""
    fig.suptitle(f"Error Distribution with BHS Thresholds{tag}", fontsize=16, fontweight="bold")

    for idx, (ax, name) in enumerate(zip(axes, ["SBP", "DBP"])):
        errors = test_pred[:, idx] - test_tgt[:, idx]
        abs_err = np.abs(errors)

        ax.hist(errors, bins=100, density=True, alpha=0.7, color="#2196F3", edgecolor="none")

        # BHS thresholds
        p5 = (abs_err <= 5).mean() * 100
        p10 = (abs_err <= 10).mean() * 100
        p15 = (abs_err <= 15).mean() * 100
        grade = bhs_grade(errors)

        ax.axvline(-5, color="green", linestyle="--", alpha=0.7, linewidth=1)
        ax.axvline(5, color="green", linestyle="--", alpha=0.7, linewidth=1, label=f"<=5: {p5:.1f}% (need 60%)")
        ax.axvline(-10, color="orange", linestyle="--", alpha=0.7, linewidth=1)
        ax.axvline(10, color="orange", linestyle="--", alpha=0.7, linewidth=1, label=f"<=10: {p10:.1f}% (need 85%)")
        ax.axvline(-15, color="red", linestyle="--", alpha=0.7, linewidth=1)
        ax.axvline(15, color="red", linestyle="--", alpha=0.7, linewidth=1, label=f"<=15: {p15:.1f}% (need 95%)")

        mae = abs_err.mean()
        me = errors.mean()
        std = errors.std()

        ax.set_xlabel(f"{name} Error (mmHg)")
        ax.set_ylabel("Density")
        ax.set_title(f"{name}  |  BHS Grade: {grade}  |  MAE={mae:.2f}  ME={me:.2f}  SD={std:.2f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    suffix = f"_{label}" if label else ""
    path = os.path.join(save_dir, f"fig4_error_dist{suffix}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────
# Plot 5: Summary comparison table (as figure)
# ──────────────────────────────────────────────

def plot_summary_table(test_pred, test_tgt, save_dir, test_pred_tta=None):
    m = compute_metrics(test_pred, test_tgt)

    rows = [
        ["MAE (mmHg)", f"{m['mae_sbp']:.2f}", f"{m['mae_dbp']:.2f}"],
        ["Combined MAE", f"{m['combined_mae']:.2f}", ""],
        ["RMSE (mmHg)", f"{m['rmse_sbp']:.2f}", f"{m['rmse_dbp']:.2f}"],
        ["Mean Error", f"{m['me_sbp']:.2f}", f"{m['me_dbp']:.2f}"],
        ["Std Error", f"{m['std_sbp']:.2f}", f"{m['std_dbp']:.2f}"],
        ["Pearson r", f"{m['r_sbp']:.3f}", f"{m['r_dbp']:.3f}"],
        ["BHS Grade", m["bhs_sbp"], m["bhs_dbp"]],
    ]

    if test_pred_tta is not None:
        mt = compute_metrics(test_pred_tta, test_tgt)
        for i, key in enumerate(["mae_sbp", "mae_dbp", "combined_mae", "rmse_sbp",
                                   "rmse_dbp", "me_sbp", "me_dbp", "std_sbp", "std_dbp",
                                   "r_sbp", "r_dbp", "bhs_sbp", "bhs_dbp"]):
            pass  # TTA column could be added as extension

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Metric", "SBP", "DBP"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)

    # Style header
    for j in range(3):
        table[0, j].set_facecolor("#2196F3")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight BHS grades
    for i, row in enumerate(rows):
        for j in range(1, 3):
            if row[j] == "A":
                table[i+1, j].set_facecolor("#C8E6C9")
            elif row[j] == "D":
                table[i+1, j].set_facecolor("#FFCDD2")

    ax.set_title("Test Set Results Summary", fontsize=14, fontweight="bold", pad=20)
    path = os.path.join(save_dir, "fig5_summary_table.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main(args):
    print(f"\nGenerating plots from {args.save_dir}...\n")

    history, test_pred, test_tgt, test_pred_tta = load_results(args.save_dir)

    # All plots
    plot_training_curves(history, args.save_dir)
    plot_bland_altman(test_pred, test_tgt, args.save_dir)
    plot_scatter(test_pred, test_tgt, args.save_dir)
    plot_error_distribution(test_pred, test_tgt, args.save_dir)
    plot_summary_table(test_pred, test_tgt, args.save_dir, test_pred_tta)

    if test_pred_tta is not None:
        plot_bland_altman(test_pred_tta, test_tgt, args.save_dir, label="TTA")
        plot_scatter(test_pred_tta, test_tgt, args.save_dir, label="TTA")
        plot_error_distribution(test_pred_tta, test_tgt, args.save_dir, label="TTA")

    # Print metrics to console too
    m = compute_metrics(test_pred, test_tgt)
    print(f"\n{'='*50}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*50}")
    print(f"  SBP MAE: {m['mae_sbp']:.2f} mmHg  |  BHS: {m['bhs_sbp']}")
    print(f"  DBP MAE: {m['mae_dbp']:.2f} mmHg  |  BHS: {m['bhs_dbp']}")
    print(f"  Combined MAE: {m['combined_mae']:.2f} mmHg")
    print(f"  Pearson r: SBP={m['r_sbp']:.3f}  DBP={m['r_dbp']:.3f}")
    print(f"{'='*50}")

    if test_pred_tta is not None:
        mt = compute_metrics(test_pred_tta, test_tgt)
        print(f"\n  With TTA:")
        print(f"  SBP MAE: {mt['mae_sbp']:.2f} mmHg  |  BHS: {mt['bhs_sbp']}")
        print(f"  DBP MAE: {mt['mae_dbp']:.2f} mmHg  |  BHS: {mt['bhs_dbp']}")
        print(f"  Combined MAE: {mt['combined_mae']:.2f} mmHg")

    print(f"\nAll figures saved to {args.save_dir}")
    print("Open with:  start <filename>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default=r".\checkpoints_v11")
    args = parser.parse_args()
    main(args)
