"""
patient_specific_adapt.py
-------------------------
Patient-specific adaptation on top of one or more cleaned PPG-only Siamese
checkpoints.

Workflow:
  1. Load the cleaned v9 PPG-only datasets (train/val/test).
  2. For each patient, fine-tune a small part of the model on that patient's
     training windows only.
  3. Early-stop on that patient's validation windows.
  4. Evaluate on that patient's held-out test windows.
  5. Optionally average multiple adapted members, then fit the usual global
     linear calibration and SBP residual correction on the aggregated
     validation predictions before scoring test performance.

This stays inside the MIMIC-II setting while making the system more explicitly
personalized, which is the most realistic remaining route to improving BHS
threshold performance without changing the dataset family.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import SiameseBPDataset, compute_sbp_tail_multipliers, load_data
from evaluate import (
    apply_linear_calibration,
    apply_subject_specific_calibration,
    apply_sbp_residual_corrector,
    build_sbp_residual_feature_matrix,
    fit_linear_calibration,
    fit_subject_specific_calibration,
    fit_sbp_residual_corrector,
)
from metrics import compute_metrics, print_metrics
from siamese_cnn import build_model
from train import AdvancedBPLoss, run_epoch, run_multi_anchor_inference


def build_member_model(device: str, hand_feature_dim: int, input_channels: int):
    return build_model(
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


def load_ppg_datasets(data_dir: str):
    print("\nLoading cleaned PPG-only datasets...")
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


def collect_members(args) -> list[dict]:
    members = []
    seen = set()

    if args.ensemble_dir:
        ensemble_dir = Path(args.ensemble_dir)
        for ckpt in sorted(ensemble_dir.glob("seed_*\\best_model.pt")):
            resolved = str(ckpt.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            members.append({
                "label": ckpt.parent.name,
                "checkpoint": resolved,
                "source": "ensemble_dir",
            })

    for raw_path in args.checkpoint:
        ckpt = Path(raw_path)
        if not ckpt.exists():
            print(f"  WARNING: checkpoint not found, skipping: {ckpt}")
            continue
        resolved = str(ckpt.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        members.append({
            "label": ckpt.parent.name or ckpt.stem,
            "checkpoint": resolved,
            "source": "explicit_checkpoint",
        })

    if not members:
        raise RuntimeError("No checkpoints found. Pass --ensemble_dir or one/more --checkpoint entries.")
    return members


def make_patient_group_view(base_ds: SiameseBPDataset, pid_list, augment: bool | None = None):
    pid_array = np.asarray([int(pid) for pid in pid_list], dtype=np.int64)
    mask = np.zeros(len(base_ds.patient_ids), dtype=bool)
    keep = np.isin(base_ds.patient_ids[base_ds.indices], pid_array)
    pid_indices = base_ds.indices[keep]
    mask[pid_indices] = True
    if augment is None:
        augment = bool(base_ds.augment)
    return SiameseBPDataset(
        base_ds.X,
        base_ds.y_sbp,
        base_ds.y_dbp,
        base_ds.patient_ids,
        mask,
        base_ds.anchors,
        pid_array.tolist(),
        randomize_anchor=False,
        augment=augment,
        X_spec=base_ds.X_spec,
        target_scale=base_ds.target_scale,
        hand_features=base_ds.hand_features,
        anchor_selection=base_ds.anchor_selection,
    )


def make_patient_view(base_ds: SiameseBPDataset, pid: int, augment: bool | None = None):
    return make_patient_group_view(base_ds, [int(pid)], augment=augment)


def set_adaptation_mode(model, mode: str) -> list[str]:
    for param in model.parameters():
        param.requires_grad = False

    train_modules = []
    if mode == "full":
        for param in model.parameters():
            param.requires_grad = True
    elif mode == "heads":
        train_modules.extend([model.shared_trunk, model.sbp_head, model.dbp_head])
        for name in ("hand_feature_proj", "pair_interaction", "fusion_proj"):
            module = getattr(model, name, None)
            if module is not None:
                train_modules.append(module)
        for module in train_modules:
            for param in module.parameters():
                param.requires_grad = True
    else:
        raise ValueError(f"Unsupported adaptation mode: {mode}")

    return [name for name, p in model.named_parameters() if p.requires_grad]


def make_optimizer(model, lr: float, weight_decay: float):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters selected for adaptation.")
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def make_criterion():
    return AdvancedBPLoss(
        sbp_weight=1.5,
        dbp_weight=1.0,
        huber_delta_sbp=0.1,
        huber_delta_dbp=0.2,
        label_noise_std=0.03,
        corr_weight=0.05,
        dbp_corr_boost=1.0,
        ccc_weight=0.06,
        dbp_ccc_boost=1.0,
        direction_weight=0.05,
        sbp_scale_weight=0.03,
        dbp_scale_weight=0.0,
        sbp_loss_type="huber",
    )


def choose_batch_size(dataset_len: int, requested: int) -> int:
    if requested <= 0:
        return max(1, dataset_len)
    return max(1, min(dataset_len, requested))


def adapt_one_patient(
    base_state: dict,
    pid: int,
    train_pid_ds: SiameseBPDataset,
    val_pid_ds: SiameseBPDataset,
    test_pid_ds: SiameseBPDataset,
    args,
    device: str,
    hand_feature_dim: int,
    input_channels: int,
):
    model = build_member_model(device, hand_feature_dim, input_channels)
    model.load_state_dict(base_state)
    model.to(device)

    adapted = False
    epochs_run = 0
    best_val_score = None
    best_state = deepcopy(model.state_dict())

    # Baseline per-patient validation score before any tuning.
    val_pred_0, val_tgt_0 = run_multi_anchor_inference(
        model, val_pid_ds, device,
        batch_size=args.eval_batch_size,
        fusion=True,
        n_aug=0,
    )
    baseline_val_score = compute_metrics(val_pred_0, val_tgt_0)["combined_mae"]
    best_val_score = baseline_val_score

    enough_support = (
        len(train_pid_ds) >= args.min_train_windows and
        len(val_pid_ds) >= args.min_val_windows and
        len(test_pid_ds) >= args.min_test_windows
    )

    if enough_support and args.adapt_epochs > 0:
        adapted = True
        trainable_names = set_adaptation_mode(model, args.adapt_mode)
        trainable_param_count = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        optimizer = make_optimizer(model, args.adapt_lr, args.adapt_weight_decay)
        criterion = make_criterion()

        support_loader = DataLoader(
            train_pid_ds,
            batch_size=choose_batch_size(len(train_pid_ds), args.adapt_batch_size),
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

        patience_ctr = 0
        for epoch in range(1, args.adapt_epochs + 1):
            epochs_run = epoch
            run_epoch(
                model,
                support_loader,
                criterion,
                optimizer,
                device,
                train=True,
                fusion=True,
                use_mixup=False,
                target_scale=10.0,
            )

            val_pred, val_tgt = run_multi_anchor_inference(
                model, val_pid_ds, device,
                batch_size=args.eval_batch_size,
                fusion=True,
                n_aug=0,
            )
            val_score = compute_metrics(val_pred, val_tgt)["combined_mae"]

            if val_score < (best_val_score - args.adapt_min_delta):
                best_val_score = val_score
                best_state = deepcopy(model.state_dict())
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= args.adapt_patience:
                    break

        model.load_state_dict(best_state)
    else:
        trainable_names = []
        trainable_param_count = 0

    # Final predictions used for aggregate scoring/calibration.
    val_pred, val_tgt = run_multi_anchor_inference(
        model, val_pid_ds, device,
        batch_size=args.eval_batch_size,
        fusion=True,
        n_aug=args.tta,
    )
    test_pred, test_tgt = run_multi_anchor_inference(
        model, test_pid_ds, device,
        batch_size=args.eval_batch_size,
        fusion=True,
        n_aug=args.tta,
    )
    final_val_metrics = compute_metrics(val_pred, val_tgt)
    final_test_metrics = compute_metrics(test_pred, test_tgt)

    summary = {
        "pid": int(pid),
        "n_train": int(len(train_pid_ds)),
        "n_val": int(len(val_pid_ds)),
        "n_test": int(len(test_pid_ds)),
        "adapted": bool(adapted),
        "epochs_run": int(epochs_run),
        "baseline_val_combined_mae": float(baseline_val_score),
        "best_val_combined_mae": float(best_val_score),
        "final_val_combined_mae": float(final_val_metrics["combined_mae"]),
        "final_test_combined_mae": float(final_test_metrics["combined_mae"]),
        "final_test_sbp_mae": float(final_test_metrics["mae_sbp"]),
        "final_test_dbp_mae": float(final_test_metrics["mae_dbp"]),
        "support_augment": bool(args.support_augment),
        "trainable_param_count": trainable_param_count,
        "trainable_param_groups": ",".join(trainable_names[:8]) if trainable_names else "",
    }

    return val_pred, val_tgt, test_pred, test_tgt, summary


def assign_subset_predictions(storage: np.ndarray, subset_ds: SiameseBPDataset,
                              pred: np.ndarray, pos_by_idx: dict[int, int]):
    for row_i, global_idx in enumerate(subset_ds.indices):
        storage[pos_by_idx[int(global_idx)]] = pred[row_i]


def save_patient_summaries(path: Path, rows: list[dict]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Patient-specific adaptation on top of one or more PPG-only Siamese checkpoints."
    )
    parser.add_argument("--data_dir", default=r"C:\MIMIC2_out_v9")
    parser.add_argument("--ensemble_dir", default=r".\ensemble_ppg_clean")
    parser.add_argument("--checkpoint", action="append", default=[],
                        help="Optional extra checkpoint(s) to adapt.")
    parser.add_argument("--save_dir", default=r".\patient_adapt_results")
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--adapt_batch_size", type=int, default=0,
                        help="0 means full-patient batch for adaptation.")
    parser.add_argument("--adapt_epochs", type=int, default=8)
    parser.add_argument("--adapt_patience", type=int, default=2)
    parser.add_argument("--adapt_min_delta", type=float, default=0.01)
    parser.add_argument("--adapt_lr", type=float, default=5e-5)
    parser.add_argument("--adapt_weight_decay", type=float, default=1e-4)
    parser.add_argument("--adapt_mode", choices=["heads", "full"], default="heads")
    parser.add_argument("--support_augment", dest="support_augment", action="store_true",
                        help="Use augmentation on each patient's support/train windows during adaptation.")
    parser.add_argument("--no_support_augment", dest="support_augment", action="store_false",
                        help="Disable augmentation on the support/train windows during adaptation.")
    parser.add_argument("--tta", type=int, default=0,
                        help="TTA passes for final per-patient val/test predictions.")
    parser.add_argument("--fit_subject_calibration", action="store_true", default=False)
    parser.add_argument("--subject_calibration_targets", choices=["sbp", "dbp", "both"], default="sbp")
    parser.add_argument("--subject_calibration_mode", choices=["bias", "affine"], default="affine")
    parser.add_argument("--subject_calibration_min_samples", type=int, default=8)
    parser.add_argument("--subject_calibration_shrinkage", type=float, default=12.0)
    parser.add_argument("--min_train_windows", type=int, default=16)
    parser.add_argument("--min_val_windows", type=int, default=4)
    parser.add_argument("--min_test_windows", type=int, default=4)
    parser.add_argument("--patient_limit", type=int, default=0,
                        help="0 means all valid patients; otherwise adapt only the first N.")
    parser.add_argument("--patient_offset", type=int, default=0,
                        help="Skip this many patients before applying patient_limit.")
    parser.set_defaults(support_augment=True)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'=' * 64}")
    print(f"  PATIENT-SPECIFIC ADAPTATION")
    print(f"  Device      : {device}")
    print(f"  Data        : {args.data_dir}")
    print(f"  Ensemble dir: {args.ensemble_dir}")
    print(f"  Adapt mode  : {args.adapt_mode}")
    print(f"  Adapt epochs: {args.adapt_epochs}  |  Patience: {args.adapt_patience}")
    print(f"  Adapt lr    : {args.adapt_lr:.2e}  |  TTA: {args.tta}")
    print(f"  Support aug : {'ON' if args.support_augment else 'OFF'}")
    print(f"{'=' * 64}")

    members = collect_members(args)
    print("\nMembers:")
    for member in members:
        print(f"  - {member['label']}: {member['checkpoint']}")

    train_ds, val_ds, test_ds, _, _, _ = load_ppg_datasets(args.data_dir)
    hand_feature_dim = test_ds.hand_feature_dim
    input_channels = test_ds.num_channels

    patient_ids = sorted({int(test_ds.patient_ids[idx]) for idx in test_ds.indices})
    if args.patient_offset > 0:
        patient_ids = patient_ids[args.patient_offset:]
    if args.patient_limit > 0:
        patient_ids = patient_ids[:args.patient_limit]
    print(f"\nAdapting {len(patient_ids)} patients")

    val_eval_ds = make_patient_group_view(val_ds, patient_ids, augment=False)
    test_eval_ds = make_patient_group_view(test_ds, patient_ids, augment=False)
    val_tgt = np.column_stack([
        val_eval_ds.y_sbp[val_eval_ds.indices],
        val_eval_ds.y_dbp[val_eval_ds.indices],
    ]).astype(np.float32)
    val_patient_ids = val_eval_ds.patient_ids[val_eval_ds.indices].astype(np.int64)
    test_tgt = np.column_stack([
        test_eval_ds.y_sbp[test_eval_ds.indices],
        test_eval_ds.y_dbp[test_eval_ds.indices],
    ]).astype(np.float32)
    test_patient_ids = test_eval_ds.patient_ids[test_eval_ds.indices].astype(np.int64)
    val_pos_by_idx = {int(idx): pos for pos, idx in enumerate(val_eval_ds.indices)}
    test_pos_by_idx = {int(idx): pos for pos, idx in enumerate(test_eval_ds.indices)}

    member_val_preds = []
    member_test_preds = []
    patient_summary_rows = []

    t0 = time.time()
    for member in members:
        print(f"\n{'=' * 64}")
        print(f"  Adapting member: {member['label']}")
        print(f"{'=' * 64}")

        base_state = torch.load(member["checkpoint"], map_location="cpu", weights_only=True)
        val_pred_full = np.zeros_like(val_tgt, dtype=np.float32)
        test_pred_full = np.zeros_like(test_tgt, dtype=np.float32)

        for p_i, pid in enumerate(patient_ids, start=1):
            train_pid_ds = make_patient_view(train_ds, pid, augment=args.support_augment)
            val_pid_ds = make_patient_view(val_ds, pid, augment=False)
            test_pid_ds = make_patient_view(test_ds, pid, augment=False)

            val_pred, _, test_pred, _, summary = adapt_one_patient(
                base_state,
                pid,
                train_pid_ds,
                val_pid_ds,
                test_pid_ds,
                args,
                device,
                hand_feature_dim,
                input_channels,
            )
            summary["member"] = member["label"]
            patient_summary_rows.append(summary)

            assign_subset_predictions(val_pred_full, val_pid_ds, val_pred, val_pos_by_idx)
            assign_subset_predictions(test_pred_full, test_pid_ds, test_pred, test_pos_by_idx)

            if p_i % 25 == 0 or p_i == len(patient_ids):
                print(
                    f"  {member['label']}: patient {p_i:>3}/{len(patient_ids)}  "
                    f"(pid={pid}, adapted={summary['adapted']}, best val={summary['best_val_combined_mae']:.2f})"
                )

            del train_pid_ds, val_pid_ds, test_pid_ds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        member_val_preds.append(val_pred_full)
        member_test_preds.append(test_pred_full)
        print_metrics(test_pred_full, test_tgt, label=f"{member['label']} adapted")

    val_ensemble = np.mean(member_val_preds, axis=0)
    test_ensemble = np.mean(member_test_preds, axis=0)

    print(f"\n{'=' * 64}")
    print("  AGGREGATE RESULTS")
    print(f"{'=' * 64}")
    metrics_summary = {}
    label_base = "Adapted Ensemble"
    if args.tta > 0:
        label_base += f" + TTA{args.tta}"
    metrics_summary["raw"] = print_metrics(test_ensemble, test_tgt, label=label_base)

    cal_scale, cal_bias = fit_linear_calibration(val_ensemble, val_tgt, targets="sbp")
    test_cal = apply_linear_calibration(test_ensemble, cal_scale, cal_bias)
    val_cal = apply_linear_calibration(val_ensemble, cal_scale, cal_bias)
    metrics_summary["cal"] = print_metrics(test_cal, test_tgt, label=label_base + " + Cal")

    residual_val_base = val_cal
    residual_test_base = test_cal
    subject_cal_state = None
    if args.fit_subject_calibration:
        subject_cal_state = fit_subject_specific_calibration(
            val_cal,
            val_tgt,
            val_patient_ids,
            targets=args.subject_calibration_targets,
            mode=args.subject_calibration_mode,
            min_samples=args.subject_calibration_min_samples,
            shrinkage=args.subject_calibration_shrinkage,
        )
        residual_val_base = apply_subject_specific_calibration(val_cal, val_patient_ids, subject_cal_state)
        residual_test_base = apply_subject_specific_calibration(test_cal, test_patient_ids, subject_cal_state)
        metrics_summary["cal_sub"] = print_metrics(
            residual_test_base,
            test_tgt,
            label=label_base + " + Cal + SubCal",
        )

    val_feat = build_sbp_residual_feature_matrix(val_eval_ds, residual_val_base)
    test_feat = build_sbp_residual_feature_matrix(test_eval_ds, residual_test_base)
    val_weights = compute_sbp_tail_multipliers(val_tgt[:, 0]).astype(np.float32)
    residual_model = fit_sbp_residual_corrector(
        val_feat,
        residual_val_base[:, 0],
        val_tgt[:, 0],
        alpha=64.0,
        sample_weight=val_weights,
        clip_mmHg=12.0,
        model_type="piecewise_ridge",
        piecewise_edges="110,130",
    )
    test_final = apply_sbp_residual_corrector(residual_test_base, test_feat, residual_model)
    final_label = label_base + " + Cal + SBPRes"
    if args.fit_subject_calibration:
        final_label = label_base + " + Cal + SubCal + SBPRes"
    metrics_summary["final"] = print_metrics(test_final, test_tgt, label=final_label)

    np.save(save_dir / "adapted_val_pred_raw.npy", val_ensemble)
    np.save(save_dir / "adapted_test_pred_raw.npy", test_ensemble)
    np.save(save_dir / "adapted_val_pred_cal.npy", val_cal)
    np.save(save_dir / "adapted_test_pred_cal.npy", test_cal)
    if args.fit_subject_calibration:
        np.save(save_dir / "adapted_val_pred_subcal.npy", residual_val_base)
        np.save(save_dir / "adapted_test_pred_subcal.npy", residual_test_base)
    np.save(save_dir / "adapted_val_pred.npy", residual_val_base)
    np.save(save_dir / "adapted_test_pred.npy", test_final)
    np.save(save_dir / "adapted_test_tgt.npy", test_tgt)
    save_patient_summaries(save_dir / "patient_adaptation_summary.csv", patient_summary_rows)

    manifest = {
        "data_dir": args.data_dir,
        "device": device,
        "tta": args.tta,
        "members": members,
        "patient_count": len(patient_ids),
        "adapt_mode": args.adapt_mode,
        "adapt_epochs": args.adapt_epochs,
        "adapt_patience": args.adapt_patience,
        "adapt_lr": args.adapt_lr,
        "adapt_weight_decay": args.adapt_weight_decay,
        "fit_subject_calibration": bool(args.fit_subject_calibration),
        "subject_calibration_targets": args.subject_calibration_targets,
        "subject_calibration_mode": args.subject_calibration_mode,
        "subject_calibration_min_samples": args.subject_calibration_min_samples,
        "subject_calibration_shrinkage": args.subject_calibration_shrinkage,
        "min_train_windows": args.min_train_windows,
        "min_val_windows": args.min_val_windows,
        "min_test_windows": args.min_test_windows,
        "elapsed_sec": time.time() - t0,
        "final_metrics": metrics_summary["final"],
    }
    if subject_cal_state is not None:
        manifest["subject_calibrated_patients"] = int(len(subject_cal_state["scale_by_pid"]))
    with (save_dir / "patient_adaptation_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved results to {save_dir}")


if __name__ == "__main__":
    main()
