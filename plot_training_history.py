import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_history(path: Path):
    obj = np.load(path, allow_pickle=True).item()
    required = [
        "train_loss",
        "train_eval_loss",
        "val_loss",
        "gap",
        "selection_score",
        "val_mae_sbp",
        "val_mae_dbp",
    ]
    missing = [k for k in required if k not in obj]
    if missing:
        raise ValueError(f"Missing history keys: {missing}")
    return obj


def write_summary(history: dict, out_path: Path):
    ss = np.asarray(history["selection_score"], dtype=np.float32)
    tr = np.asarray(history["train_loss"], dtype=np.float32)
    tre = np.asarray(history["train_eval_loss"], dtype=np.float32)
    vl = np.asarray(history["val_loss"], dtype=np.float32)
    gap = np.asarray(history["gap"], dtype=np.float32)
    sbp = np.asarray(history["val_mae_sbp"], dtype=np.float32)
    dbp = np.asarray(history["val_mae_dbp"], dtype=np.float32)

    best = int(ss.argmin())
    lines = [
        f"epochs={len(ss)}",
        f"best_epoch={best + 1}",
        f"best_selection_score={ss[best]:.6f}",
        f"last_selection_score={ss[-1]:.6f}",
        f"best_train_loss={tr[best]:.6f}",
        f"last_train_loss={tr[-1]:.6f}",
        f"best_train_eval_loss={tre[best]:.6f}",
        f"last_train_eval_loss={tre[-1]:.6f}",
        f"best_val_loss={vl[best]:.6f}",
        f"last_val_loss={vl[-1]:.6f}",
        f"best_gap={gap[best]:.6f}",
        f"last_gap={gap[-1]:.6f}",
        f"best_val_mae_sbp={sbp[best]:.6f}",
        f"last_val_mae_sbp={sbp[-1]:.6f}",
        f"best_val_mae_dbp={dbp[best]:.6f}",
        f"last_val_mae_dbp={dbp[-1]:.6f}",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def plot_history(history: dict, title: str, out_path: Path):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    best_epoch = int(np.argmin(history["selection_score"])) + 1

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle(title, fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], label="train loss", linewidth=2)
    ax.plot(epochs, history["train_eval_loss"], label="train-eval loss", linewidth=2)
    ax.plot(epochs, history["val_loss"], label="val loss", linewidth=2)
    ax.axvline(best_epoch, color="black", linestyle="--", linewidth=1.5, label=f"best epoch ({best_epoch})")
    ax.set_title("Loss Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(frameon=True)

    ax = axes[0, 1]
    ax.plot(epochs, history["val_mae_sbp"], label="val SBP MAE", linewidth=2, color="#d62728")
    ax.plot(epochs, history["val_mae_dbp"], label="val DBP MAE", linewidth=2, color="#1f77b4")
    ax.axvline(best_epoch, color="black", linestyle="--", linewidth=1.5)
    ax.set_title("Validation MAE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (mmHg)")
    ax.legend(frameon=True)

    ax = axes[1, 0]
    ax.plot(epochs, history["selection_score"], label="selection score", linewidth=2, color="#2ca02c")
    ax.axvline(best_epoch, color="black", linestyle="--", linewidth=1.5)
    ax.set_title("Selection Score")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.legend(frameon=True)

    ax = axes[1, 1]
    ax.plot(epochs, history["gap"], label="gap", linewidth=2, color="#9467bd")
    ax.axvline(best_epoch, color="black", linestyle="--", linewidth=1.5)
    ax.set_title("Generalization Gap")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gap")
    ax.legend(frameon=True)

    best_text = (
        f"Best epoch: {best_epoch}\n"
        f"SBP MAE: {history['val_mae_sbp'][best_epoch - 1]:.3f}\n"
        f"DBP MAE: {history['val_mae_dbp'][best_epoch - 1]:.3f}\n"
        f"Score: {history['selection_score'][best_epoch - 1]:.3f}"
    )
    fig.text(
        0.84,
        0.03,
        best_text,
        ha="left",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#666666"),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", required=True, help="Path to history.npy")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--summary", default=None, help="Optional text summary path")
    parser.add_argument("--title", default="Training History")
    args = parser.parse_args()

    history = load_history(Path(args.history))
    plot_history(history, args.title, Path(args.output))
    if args.summary:
        write_summary(history, Path(args.summary))


if __name__ == "__main__":
    main()
