"""
run_ensemble.py
---------------
Train and/or evaluate an ensemble of PPG-only Siamese BP models, then fit
calibration and SBP residual correction on the ensemble validation predictions.

The script writes a manifest with the exact arguments used for each run. This
matters because older checkpoints did not always store the full command line.

Typical usage:

    python run_ensemble.py --n_seeds 3

    python run_ensemble.py --n_seeds 2 --start_seed 43 ^
        --existing_checkpoint ".\\checkpoints_v26_beataware\\best_model.pt"

    python run_ensemble.py --skip_training ^
        --existing_checkpoint ".\\checkpoints_v26_beataware\\best_model.pt" ^
        --existing_checkpoint ".\\ensemble_ppg_clean\\seed_43\\best_model.pt" ^
        --existing_checkpoint ".\\ensemble_ppg_clean\\seed_44\\best_model.pt"
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import load_data, compute_sbp_tail_multipliers
from evaluate import (
    apply_linear_calibration,
    apply_sbp_residual_corrector,
    build_sbp_residual_feature_matrix,
    fit_linear_calibration,
    fit_sbp_residual_corrector,
)
from metrics import print_metrics
from siamese_cnn import build_model
from train import run_multi_anchor_inference


DEFAULT_RECIPE = {
    "name": "clean_ppg_huber_v1",
    "description": (
        "PPG-only Siamese recipe used for the clean v11 ensemble: "
        "fusion + beat-aware features + gated pair interaction, train-only "
        "normalisation stats, Huber SBP/DBP losses, uniform BP sampling."
    ),
    "train_args": [
        "--use_fusion",
        "--use_beat_features",
        "--use_gated_pair_interaction",
        "--no_mixup",
        "--no_balanced_sampling",
        "--sbp_loss_type", "huber",
        "--ccc_weight", "0.06",
        "--dbp_ccc_boost", "0.0",
        "--sbp_scale_weight", "0.03",
        "--dbp_scale_weight", "0.0",
        "--dbp_corr_boost", "1.0",
        "--epochs", "150",
        "--patience", "30",
        "--batch_size", "256",
        "--eval_batch_size", "512",
        "--lr", "3e-4",
        "--dropout", "0.30",
        "--embed_dim", "256",
        "--weight_decay", "1e-3",
        "--num_anchors", "5",
        "--target_scale", "10.0",
        "--quality_threshold", "0.4",
        "--preserve_ppg_amplitude",
        "--multi_anchor_eval",
        "--use_ema",
        "--ema_decay", "0.999",
        "--gap_penalty", "0.15",
        "--gap_target", "1.0",
        "--corr_weight", "0.05",
        "--sbp_weight", "1.5",
        "--tta", "0",
        "--workers", "0",
    ],
}


def load_recipe(recipe_json: str | None) -> dict:
    if recipe_json is None:
        return {
            "name": DEFAULT_RECIPE["name"],
            "description": DEFAULT_RECIPE["description"],
            "train_args": list(DEFAULT_RECIPE["train_args"]),
        }

    recipe_path = Path(recipe_json)
    with recipe_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return {
            "name": recipe_path.stem,
            "description": "Loaded from recipe JSON list.",
            "train_args": [str(x) for x in payload],
        }

    if not isinstance(payload, dict) or "train_args" not in payload:
        raise ValueError("recipe_json must contain either a JSON list or an object with 'train_args'.")

    return {
        "name": str(payload.get("name", recipe_path.stem)),
        "description": str(payload.get("description", "Loaded from recipe JSON object.")),
        "train_args": [str(x) for x in payload["train_args"]],
    }


def seed_range(start_seed: int, n_seeds: int) -> list[int]:
    return list(range(start_seed, start_seed + n_seeds))


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def build_train_command(args, recipe: dict, seed: int, save_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "train.py",
        "--data_dir", args.data_dir,
        "--save_dir", str(save_dir),
        "--seed", str(seed),
    ]
    cmd.extend(recipe["train_args"])
    return cmd


def train_seed(args, recipe: dict, seed: int, save_dir: Path) -> bool:
    save_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_train_command(args, recipe, seed, save_dir)
    (save_dir / "train_command.txt").write_text(" ".join(cmd), encoding="utf-8")

    print(f"\n{'=' * 64}")
    print(f"  ENSEMBLE: training seed {seed} -> {save_dir}")
    print(f"{'=' * 64}")
    print(f"  Recipe : {recipe['name']}")
    print(f"  Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f"  WARNING: seed {seed} exited with code {result.returncode}")
    return result.returncode == 0


def member_label_from_checkpoint(checkpoint_path: Path) -> str:
    parent = checkpoint_path.parent.name
    if parent:
        return parent
    return checkpoint_path.stem


def collect_members(args) -> list[dict]:
    members = []
    seen = set()

    for seed in seed_range(args.start_seed, args.n_seeds):
        ckpt = Path(args.ensemble_dir) / f"seed_{seed}" / "best_model.pt"
        if ckpt.exists():
            resolved = str(ckpt.resolve())
            if resolved not in seen:
                seen.add(resolved)
                members.append({
                    "label": f"seed_{seed}",
                    "checkpoint": resolved,
                    "source": "trained_seed",
                })

    for raw_path in args.existing_checkpoint:
        ckpt = Path(raw_path)
        if not ckpt.exists():
            print(f"  WARNING: existing checkpoint not found, skipping: {ckpt}")
            continue
        resolved = str(ckpt.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        members.append({
            "label": member_label_from_checkpoint(ckpt),
            "checkpoint": resolved,
            "source": "existing_checkpoint",
        })

    return members


def build_member_model(device: str, hand_feature_dim: int, input_channels: int):
    model = build_model(
        embed_dim=256,
        dropout=0.30,
        device=device,
        use_spectrograms=False,
        use_fusion=True,
        use_pair_uncertainty=False,
        use_gated_pair_interaction=True,
        use_absolute_refinement=False,
        hand_feature_dim=hand_feature_dim,
        input_channels=input_channels,
        target_scale=10.0,
    )
    return model


def run_multi_anchor_for_model(
    checkpoint_path: str,
    dataset,
    device: str,
    batch_size: int,
    n_aug: int,
    hand_feature_dim: int,
    input_channels: int,
):
    model = build_member_model(device, hand_feature_dim, input_channels)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    pred, tgt = run_multi_anchor_inference(
        model,
        dataset,
        device,
        batch_size=batch_size,
        fusion=True,
        n_aug=n_aug,
        anchor_similarity_mode="none",
        anchor_top_k=0,
        anchor_similarity_temp=0.35,
    )
    return pred, tgt


def load_ppg_eval_data(data_dir: str):
    print("\nLoading cleaned PPG-only dataset...")
    train_ds, val_ds, test_ds, anchors, y_sbp, y_dbp, _ = load_data(
        data_dir,
        random_anchor=False,
        min_sbp_std=5.0,
        num_anchors=5,
        use_spectrograms=False,
        use_fusion=True,
        target_scale=10.0,
        quality_threshold=0.4,
        shuffle_eval=True,
        preserve_ppg_amplitude=True,
        use_beat_features=True,
        use_ecg=False,
    )
    return train_ds, val_ds, test_ds, anchors, y_sbp, y_dbp


def report_metrics(label: str, pred: np.ndarray, tgt: np.ndarray, results_lines: list[str]) -> dict:
    metrics = print_metrics(pred, tgt, label=label)
    sbp_abs_err = np.abs(pred[:, 0] - tgt[:, 0])
    within5 = float((sbp_abs_err <= 5.0).mean() * 100.0)
    within10 = float((sbp_abs_err <= 10.0).mean() * 100.0)
    within15 = float((sbp_abs_err <= 15.0).mean() * 100.0)
    results_lines.append(
        f"{label}: SBP MAE={metrics['mae_sbp']:.2f}  DBP MAE={metrics['mae_dbp']:.2f}  "
        f"Comb={metrics['combined_mae']:.2f}  BHS SBP={metrics['bhs_sbp']}  BHS DBP={metrics['bhs_dbp']}  "
        f"SBP<=5={within5:.2f}%  <=10={within10:.2f}%  <=15={within15:.2f}%"
    )
    return metrics


def ensemble_and_calibrate(args, recipe: dict) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensemble_dir = Path(args.ensemble_dir)
    members = collect_members(args)

    if len(members) < 2:
        raise RuntimeError("Need at least two checkpoints to form an ensemble.")

    print(f"\n{'=' * 64}")
    print(f"  ENSEMBLE: evaluating {len(members)} members on {device}")
    print(f"{'=' * 64}")
    for member in members:
        print(f"  - {member['label']}: {member['checkpoint']}")

    _, val_ds, test_ds, _, _, _ = load_ppg_eval_data(args.data_dir)
    hf_dim = test_ds.hand_feature_dim
    n_ch = test_ds.num_channels

    val_preds = []
    val_tta_preds = []
    test_preds = []
    test_tta_preds = []
    val_tgt = None
    test_tgt = None
    results_lines: list[str] = []

    print(f"\n{'=' * 64}")
    print("  PER-MEMBER INFERENCE")
    print(f"{'=' * 64}")
    for member in members:
        ckpt = member["checkpoint"]
        label = member["label"]
        print(f"\n  {label}: running multi-anchor inference...")

        vp, vt = run_multi_anchor_for_model(
            ckpt, val_ds, device, args.eval_batch_size, 0, hf_dim, n_ch
        )
        tp, tt = run_multi_anchor_for_model(
            ckpt, test_ds, device, args.eval_batch_size, 0, hf_dim, n_ch
        )

        val_preds.append(vp)
        test_preds.append(tp)
        if val_tgt is None:
            val_tgt = vt
        if test_tgt is None:
            test_tgt = tt

        report_metrics(f"{label} (no TTA)", tp, test_tgt, results_lines)

        if args.tta > 0:
            vp_tta, _ = run_multi_anchor_for_model(
                ckpt, val_ds, device, args.eval_batch_size, args.tta, hf_dim, n_ch
            )
            tp_tta, _ = run_multi_anchor_for_model(
                ckpt, test_ds, device, args.eval_batch_size, args.tta, hf_dim, n_ch
            )
            val_tta_preds.append(vp_tta)
            test_tta_preds.append(tp_tta)
            report_metrics(f"{label} (TTA)", tp_tta, test_tgt, results_lines)

    val_ensemble = np.mean(val_preds, axis=0)
    test_ensemble = np.mean(test_preds, axis=0)

    print(f"\n{'=' * 64}")
    print("  ENSEMBLE RESULTS")
    print(f"{'=' * 64}")
    ensemble_metrics = {}
    ensemble_metrics["ensemble_no_tta"] = report_metrics(
        "Ensemble (no TTA)", test_ensemble, test_tgt, results_lines
    )

    cal_scale, cal_bias = fit_linear_calibration(val_ensemble, val_tgt, targets="sbp")
    test_cal = apply_linear_calibration(test_ensemble, cal_scale, cal_bias)
    ensemble_metrics["ensemble_cal"] = report_metrics(
        "Ensemble + Cal", test_cal, test_tgt, results_lines
    )

    val_cal = apply_linear_calibration(val_ensemble, cal_scale, cal_bias)
    val_feat = build_sbp_residual_feature_matrix(val_ds, val_cal)
    test_feat = build_sbp_residual_feature_matrix(test_ds, test_cal)
    val_weights = compute_sbp_tail_multipliers(val_tgt[:, 0]).astype(np.float32)
    residual_model = fit_sbp_residual_corrector(
        val_feat,
        val_cal[:, 0],
        val_tgt[:, 0],
        alpha=64.0,
        sample_weight=val_weights,
        clip_mmHg=12.0,
        model_type="piecewise_ridge",
        piecewise_edges="110,130",
    )
    test_cal_res = apply_sbp_residual_corrector(test_cal, test_feat, residual_model)
    ensemble_metrics["ensemble_cal_sbpres"] = report_metrics(
        "Ensemble + Cal + SBPRes", test_cal_res, test_tgt, results_lines
    )

    final_pred = test_cal_res
    final_label = "Ensemble + Cal + SBPRes"

    if args.tta > 0 and len(test_tta_preds) == len(members):
        val_tta_ensemble = np.mean(val_tta_preds, axis=0)
        test_tta_ensemble = np.mean(test_tta_preds, axis=0)

        ensemble_metrics["ensemble_tta"] = report_metrics(
            "Ensemble + TTA", test_tta_ensemble, test_tgt, results_lines
        )

        cal_scale_tta, cal_bias_tta = fit_linear_calibration(
            val_tta_ensemble, val_tgt, targets="sbp"
        )
        test_tta_cal = apply_linear_calibration(test_tta_ensemble, cal_scale_tta, cal_bias_tta)
        ensemble_metrics["ensemble_tta_cal"] = report_metrics(
            "Ensemble + TTA + Cal", test_tta_cal, test_tgt, results_lines
        )

        val_tta_cal = apply_linear_calibration(val_tta_ensemble, cal_scale_tta, cal_bias_tta)
        val_tta_feat = build_sbp_residual_feature_matrix(val_ds, val_tta_cal)
        test_tta_feat = build_sbp_residual_feature_matrix(test_ds, test_tta_cal)
        residual_model_tta = fit_sbp_residual_corrector(
            val_tta_feat,
            val_tta_cal[:, 0],
            val_tgt[:, 0],
            alpha=64.0,
            sample_weight=val_weights,
            clip_mmHg=12.0,
            model_type="piecewise_ridge",
            piecewise_edges="110,130",
        )
        test_tta_cal_res = apply_sbp_residual_corrector(
            test_tta_cal, test_tta_feat, residual_model_tta
        )
        ensemble_metrics["ensemble_tta_cal_sbpres"] = report_metrics(
            "Ensemble + TTA + Cal + SBPRes", test_tta_cal_res, test_tgt, results_lines
        )
        final_pred = test_tta_cal_res
        final_label = "Ensemble + TTA + Cal + SBPRes"

    np.save(ensemble_dir / "ensemble_pred.npy", final_pred)
    np.save(ensemble_dir / "ensemble_tgt.npy", test_tgt)
    np.save(ensemble_dir / "ensemble_pred_raw.npy", test_ensemble)

    results_path = ensemble_dir / "ensemble_results.txt"
    results_path.write_text(
        "\n".join([
            f"Recipe: {recipe['name']}",
            recipe.get("description", ""),
            f"Members: {[m['label'] for m in members]}",
            f"Final label: {final_label}",
            "=" * 64,
            *results_lines,
        ]),
        encoding="utf-8",
    )

    manifest = {
        "recipe": recipe,
        "data_dir": args.data_dir,
        "eval_batch_size": args.eval_batch_size,
        "tta": args.tta,
        "members": members,
        "final_label": final_label,
        "results_path": str(results_path.resolve()),
        "pred_path": str((ensemble_dir / "ensemble_pred.npy").resolve()),
    }
    write_manifest(ensemble_dir / "ensemble_manifest.json", manifest)

    print(f"\n  Results saved to {results_path}")
    print(f"  Manifest saved to {ensemble_dir / 'ensemble_manifest.json'}")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate an ensemble of cleaned PPG-only Siamese BP models."
    )
    parser.add_argument("--n_seeds", type=int, default=3, help="Number of contiguous seeds to train/evaluate.")
    parser.add_argument("--start_seed", type=int, default=42, help="First seed in the contiguous seed range.")
    parser.add_argument("--ensemble_dir", default=r".\ensemble_ppg_clean", help="Output directory.")
    parser.add_argument("--data_dir", default=r"C:\MIMIC2_out_v9", help="PPG-only dataset directory.")
    parser.add_argument("--eval_batch_size", type=int, default=512, help="Batch size for ensemble inference.")
    parser.add_argument("--tta", type=int, default=10, help="Number of test-time augmentations for ensemble inference.")
    parser.add_argument("--skip_training", action="store_true", help="Skip training and only ensemble available checkpoints.")
    parser.add_argument(
        "--existing_checkpoint",
        action="append",
        default=[],
        help="Additional checkpoint(s) to include in the ensemble without retraining.",
    )
    parser.add_argument(
        "--recipe_json",
        default=None,
        help="Optional JSON file containing {'name','description','train_args'} or just a JSON list of train args.",
    )
    args = parser.parse_args()

    ensemble_dir = Path(args.ensemble_dir)
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    recipe = load_recipe(args.recipe_json)
    write_manifest(
        ensemble_dir / "planned_manifest.json",
        {
            "recipe": recipe,
            "data_dir": args.data_dir,
            "n_seeds": args.n_seeds,
            "start_seed": args.start_seed,
            "skip_training": args.skip_training,
            "existing_checkpoint": [str(Path(p).resolve()) for p in args.existing_checkpoint],
            "eval_batch_size": args.eval_batch_size,
            "tta": args.tta,
        },
    )

    if not args.skip_training:
        for seed in seed_range(args.start_seed, args.n_seeds):
            save_dir = ensemble_dir / f"seed_{seed}"
            if (save_dir / "best_model.pt").exists():
                print(f"\n  Seed {seed} already exists at {save_dir}; skipping training.")
                continue
            train_seed(args, recipe, seed, save_dir)

    ensemble_and_calibrate(args, recipe)


if __name__ == "__main__":
    main()
