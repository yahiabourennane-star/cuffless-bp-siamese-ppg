"""
train.py
--------
Main training and evaluation loop for SiameseBPNet.
Most experiment settings are passed in through command-line flags.

Usage:
    python train.py --data_dir "C:\MIMIC2_out_v8" --save_dir ".\checkpoints" --workers 0
"""

import argparse
from copy import copy
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

# flat imports — all files in same folder
from dataset import load_data
from siamese_cnn import build_model
from metrics import print_metrics


# ──────────────────────────────────────────────
# Loss - Huber is less sensitive to occasional large label errors than MAE
# ──────────────────────────────────────────────

class BPLoss(nn.Module):
    """
    Loss used for delta SBP/DBP prediction.

    The point-error term handles the main regression target. The correlation
    term was added because early runs tended to predict changes too close to
    zero. The direction term gives a small extra penalty when the sign of a
    larger BP change is wrong.
    """

    def __init__(self, sbp_weight=1.5, dbp_weight=1.0,
                 huber_delta_sbp=0.1, huber_delta_dbp=0.2,
                 label_noise_std=0.0,
                 corr_weight=0.05, dbp_corr_boost=1.25,
                 ccc_weight=0.12, dbp_ccc_boost=1.15,
                 direction_weight=0.05,
                 sbp_scale_weight=0.08, dbp_scale_weight=0.02,
                 sbp_loss_type="mse"):
        super().__init__()
        self.sbp_w = sbp_weight
        self.dbp_w = dbp_weight
        self.sbp_loss_type = sbp_loss_type
        if sbp_loss_type == "mse":
            self.loss_sbp = nn.MSELoss(reduction="mean")
        else:
            self.loss_sbp = nn.HuberLoss(delta=huber_delta_sbp, reduction="mean")
        self.huber_dbp = nn.HuberLoss(delta=huber_delta_dbp, reduction="mean")
        self.label_noise_std = label_noise_std
        self.corr_w = corr_weight
        self.corr_w_sbp = corr_weight
        self.corr_w_dbp = corr_weight * dbp_corr_boost
        self.ccc_w_sbp = ccc_weight
        self.ccc_w_dbp = ccc_weight * dbp_ccc_boost
        self.dir_w = direction_weight
        self.scale_w_sbp = sbp_scale_weight
        self.scale_w_dbp = dbp_scale_weight

    def _correlation_loss(self, pred, target):
        """1 - Pearson r, so predictions follow BP ups and downs."""
        # pred, target: (B,)
        p = pred - pred.mean()
        t = target - target.mean()
        num = (p * t).sum()
        den = (p.norm() * t.norm()).clamp(min=1e-8)
        r = num / den
        return 1.0 - r  # 0 when perfectly correlated

    def _ccc_loss(self, pred, target):
        """1 - CCC, covering correlation, bias, and scale mismatch."""
        pred_mean = pred.mean()
        target_mean = target.mean()
        pred_var = pred.var(unbiased=False)
        target_var = target.var(unbiased=False)
        cov = ((pred - pred_mean) * (target - target_mean)).mean()
        ccc = (2.0 * cov) / (
            pred_var + target_var + (pred_mean - target_mean).pow(2) + 1e-8
        )
        return 1.0 - ccc

    def _scale_bias_loss(self, pred, target):
        """Extra penalty for slope and offset mismatch."""
        target_centered = target - target.mean()
        pred_centered = pred - pred.mean()
        target_var = target_centered.pow(2).mean().clamp(min=1e-8)
        slope = (pred_centered * target_centered).mean() / target_var
        pred_std = pred.std(unbiased=False)
        target_std = target.std(unbiased=False).clamp(min=1e-8)
        std_ratio = pred_std / target_std
        intercept = pred.mean() - slope * target.mean()
        return (
            (slope - 1.0).abs()
            + 0.5 * (std_ratio - 1.0).abs()
            + 0.1 * intercept.abs()
        )

    def _direction_loss(self, pred, target):
        """Penalty for predicting the wrong direction of BP change."""
        # Sign agreement: +1 if same sign, -1 if different.
        sign_match = torch.sign(pred) * torch.sign(target)  # +1 or -1
        # Wrong signs matter more on larger BP changes.
        magnitude = target.abs()
        # Penalty is zero when the sign is correct.
        penalty = torch.relu(-sign_match) * magnitude  # 0 if correct sign
        return penalty.mean()

    def forward(self, pred, target):
        if isinstance(pred, tuple):
            pred = pred[0]
        # Small target noise during training only.
        if self.training and self.label_noise_std > 0:
            noise = torch.randn_like(target) * self.label_noise_std
            target = target + noise

        # 1. Primary loss (MSE for SBP if configured, Huber for DBP)
        loss_sbp = self.loss_sbp(pred[:, 0], target[:, 0])
        loss_dbp = self.huber_dbp(pred[:, 1], target[:, 1])
        huber = self.sbp_w * loss_sbp + self.dbp_w * loss_dbp

        # 2. Correlation term: discourages flat delta predictions.
        corr_sbp = self._correlation_loss(pred[:, 0], target[:, 0])
        corr_dbp = self._correlation_loss(pred[:, 1], target[:, 1])
        corr = self.sbp_w * corr_sbp + self.dbp_w * corr_dbp

        # 3. Direction term: penalise wrong-sign predictions.
        dir_sbp = self._direction_loss(pred[:, 0], target[:, 0])
        dir_dbp = self._direction_loss(pred[:, 1], target[:, 1])
        direction = self.sbp_w * dir_sbp + self.dbp_w * dir_dbp

        return huber + self.corr_w * corr + self.dir_w * direction


class AdvancedBPLoss(nn.Module):
    """
    Loss variant used for most later experiments.

    It combines point error with correlation, CCC, direction, and scale terms.
    This helped reduce the tendency to predict small changes for every window.
    """

    def __init__(self, sbp_weight=1.5, dbp_weight=1.0,
                 huber_delta_sbp=0.1, huber_delta_dbp=0.2,
                 label_noise_std=0.0,
                 corr_weight=0.05, dbp_corr_boost=1.25,
                 ccc_weight=0.12, dbp_ccc_boost=1.15,
                 direction_weight=0.05,
                 sbp_scale_weight=0.08, dbp_scale_weight=0.02,
                 sbp_loss_type="mse"):
        super().__init__()
        self.sbp_w = sbp_weight
        self.dbp_w = dbp_weight
        self.label_noise_std = label_noise_std
        self.corr_w_sbp = corr_weight
        self.corr_w_dbp = corr_weight * dbp_corr_boost
        self.ccc_w_sbp = ccc_weight
        self.ccc_w_dbp = ccc_weight * dbp_ccc_boost
        self.dir_w = direction_weight
        self.scale_w_sbp = sbp_scale_weight
        self.scale_w_dbp = dbp_scale_weight
        self.sbp_loss_type = sbp_loss_type
        self.huber_delta_sbp = huber_delta_sbp
        self.huber_delta_dbp = huber_delta_dbp
        if sbp_loss_type == "mse":
            self.loss_sbp = nn.MSELoss(reduction="mean")
        else:
            self.loss_sbp = nn.HuberLoss(delta=huber_delta_sbp, reduction="mean")
        self.loss_dbp = nn.HuberLoss(delta=huber_delta_dbp, reduction="mean")

    @staticmethod
    def _correlation_loss(pred, target):
        p = pred - pred.mean()
        t = target - target.mean()
        num = (p * t).sum()
        den = (p.norm() * t.norm()).clamp(min=1e-8)
        return 1.0 - (num / den)

    @staticmethod
    def _ccc_loss(pred, target):
        pred_mean = pred.mean()
        target_mean = target.mean()
        pred_var = pred.var(unbiased=False)
        target_var = target.var(unbiased=False)
        cov = ((pred - pred_mean) * (target - target_mean)).mean()
        ccc = (2.0 * cov) / (
            pred_var + target_var + (pred_mean - target_mean).pow(2) + 1e-8
        )
        return 1.0 - ccc

    @staticmethod
    def _scale_bias_loss(pred, target):
        target_centered = target - target.mean()
        pred_centered = pred - pred.mean()
        target_var = target_centered.pow(2).mean().clamp(min=1e-8)
        slope = (pred_centered * target_centered).mean() / target_var
        pred_std = pred.std(unbiased=False)
        target_std = target.std(unbiased=False).clamp(min=1e-8)
        std_ratio = pred_std / target_std
        intercept = pred.mean() - slope * target.mean()
        return (
            (slope - 1.0).abs()
            + 0.5 * (std_ratio - 1.0).abs()
            + 0.1 * intercept.abs()
        )

    @staticmethod
    def _direction_loss(pred, target):
        sign_match = torch.sign(pred) * torch.sign(target)
        magnitude = target.abs()
        penalty = torch.relu(-sign_match) * magnitude
        return penalty.mean()

    def forward(self, pred, target):
        log_var = None
        if isinstance(pred, tuple):
            pred, log_var = pred
        if self.training and self.label_noise_std > 0:
            target = target + torch.randn_like(target) * self.label_noise_std

        pred_sbp, pred_dbp = pred[:, 0], pred[:, 1]
        tgt_sbp, tgt_dbp = target[:, 0], target[:, 1]

        if self.sbp_loss_type == "mse":
            reg_sbp_elem = F.mse_loss(pred_sbp, tgt_sbp, reduction="none")
        else:
            reg_sbp_elem = F.huber_loss(
                pred_sbp, tgt_sbp, delta=self.huber_delta_sbp, reduction="none"
            )
        reg_dbp_elem = F.huber_loss(
            pred_dbp, tgt_dbp, delta=self.huber_delta_dbp, reduction="none"
        )

        if log_var is not None:
            log_var = torch.clamp(log_var, min=-4.0, max=4.0)
            reg_sbp = (torch.exp(-log_var[:, 0]) * reg_sbp_elem + log_var[:, 0]).mean()
            reg_dbp = (torch.exp(-log_var[:, 1]) * reg_dbp_elem + log_var[:, 1]).mean()
        else:
            reg_sbp = reg_sbp_elem.mean()
            reg_dbp = reg_dbp_elem.mean()

        reg = self.sbp_w * reg_sbp
        reg = reg + self.dbp_w * reg_dbp

        pearson = self.sbp_w * self.corr_w_sbp * self._correlation_loss(pred_sbp, tgt_sbp)
        pearson = pearson + self.dbp_w * self.corr_w_dbp * self._correlation_loss(pred_dbp, tgt_dbp)

        concordance = self.sbp_w * self.ccc_w_sbp * self._ccc_loss(pred_sbp, tgt_sbp)
        concordance = concordance + self.dbp_w * self.ccc_w_dbp * self._ccc_loss(pred_dbp, tgt_dbp)

        scale = self.sbp_w * self.scale_w_sbp * self._scale_bias_loss(pred_sbp, tgt_sbp)
        scale = scale + self.dbp_w * self.scale_w_dbp * self._scale_bias_loss(pred_dbp, tgt_dbp)

        direction = self.sbp_w * self.dir_w * self._direction_loss(pred_sbp, tgt_sbp)
        direction = direction + self.dbp_w * self.dir_w * self._direction_loss(pred_dbp, tgt_dbp)

        return reg + pearson + concordance + scale + direction


class ModelEMA:
    """Exponential moving average of trainable parameters for evaluation."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.backup = {}

    def _ensure_shadow_entry(self, name: str, param: torch.Tensor):
        if name not in self.shadow:
            self.shadow[name] = param.detach().clone()

    def update(self, model: nn.Module):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                self._ensure_shadow_entry(name, param)
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def store(self, model: nn.Module):
        self.backup = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def copy_to(self, model: nn.Module):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self._ensure_shadow_entry(name, param)
                    param.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.backup:
                    param.copy_(self.backup[name])
        self.backup = {}


def set_feature_extractors_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze or unfreeze shared Siamese feature extractors for fine-tuning."""
    module_names = ("encoder", "encoder_wave", "encoder_spec", "fusion_proj")
    for name in module_names:
        module = getattr(model, name, None)
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad = trainable


# ──────────────────────────────────────────────
# Window cap per patient (balance dataset)
# ──────────────────────────────────────────────

def cap_dataset(dataset, patient_ids_full, max_per_patient, seed=42):
    """
    Cap the number of windows per patient to max_per_patient.
    Prevents high-window patients from dominating training.
    """
    if max_per_patient <= 0:
        return dataset

    rng = np.random.default_rng(seed)
    indices = dataset.indices  # global indices into X

    # Group by patient
    pid_to_local = {}
    for local_i, global_i in enumerate(indices):
        pid = int(patient_ids_full[global_i])
        pid_to_local.setdefault(pid, []).append(local_i)

    kept = []
    for pid, local_idxs in pid_to_local.items():
        if len(local_idxs) > max_per_patient:
            chosen = rng.choice(local_idxs, max_per_patient, replace=False)
        else:
            chosen = local_idxs
        kept.extend(chosen)

    kept = np.array(sorted(kept))
    dataset.indices = indices[kept]
    if hasattr(dataset, "refresh_index_cache"):
        dataset.refresh_index_cache()
    return dataset


def forward_pair(model, anchor, current, anc_sbp_norm, anc_dbp_norm,
                 anchor_spec=None, current_spec=None,
                 anchor_hf=None, current_hf=None,
                 return_extras=False):
    """Forward one Siamese pair with optional uncertainty / auxiliary outputs."""
    needs_extras = bool(return_extras)
    if getattr(model, "use_pair_uncertainty", False) or getattr(model, "use_absolute_refinement", False):
        return model(
            anchor, current, anc_sbp_norm, anc_dbp_norm,
            anchor_spec=anchor_spec, current_spec=current_spec,
            anchor_hf=anchor_hf, current_hf=current_hf,
            return_uncertainty=needs_extras,
            return_auxiliary=needs_extras,
        )

    pred = model(
        anchor, current, anc_sbp_norm, anc_dbp_norm,
        anchor_spec=anchor_spec, current_spec=current_spec,
        anchor_hf=anchor_hf, current_hf=current_hf,
    )
    if return_extras:
        return pred, {}
    return pred


def compute_absolute_refinement_losses(extras, target, anc_sbp_norm, anc_dbp_norm,
                                       target_scale,
                                       tail_low=90.0, tail_high=150.0,
                                       shoulder_low=110.0, shoulder_high=130.0,
                                       shoulder_boost=1.15, tail_boost=1.60):
    """Auxiliary losses for the absolute-refinement branch."""
    absolute_norm = extras.get("absolute_norm")
    delta_raw = extras.get("delta_raw")
    delta_from_abs = extras.get("delta_from_abs")
    if absolute_norm is None or delta_raw is None or delta_from_abs is None:
        device = target.device
        zero = torch.zeros((), device=device, dtype=target.dtype)
        return zero, zero

    anchor_norm = torch.stack([anc_sbp_norm, anc_dbp_norm], dim=1)
    current_norm = anchor_norm + target * (float(target_scale) / 100.0)

    abs_pred_mmhg = absolute_norm * 100.0
    abs_tgt_mmhg = current_norm * 100.0

    sbp_elem = F.huber_loss(
        abs_pred_mmhg[:, 0], abs_tgt_mmhg[:, 0],
        delta=4.0, reduction="none"
    )
    dbp_elem = F.huber_loss(
        abs_pred_mmhg[:, 1], abs_tgt_mmhg[:, 1],
        delta=3.0, reduction="none"
    )

    sbp_tgt = abs_tgt_mmhg[:, 0]
    sbp_weights = torch.ones_like(sbp_tgt)
    shoulder_mask = ((sbp_tgt < shoulder_low) | (sbp_tgt > shoulder_high)).to(sbp_tgt.dtype)
    tail_mask = ((sbp_tgt < tail_low) | (sbp_tgt > tail_high)).to(sbp_tgt.dtype)
    sbp_weights = sbp_weights + (float(shoulder_boost) - 1.0) * shoulder_mask
    sbp_weights = sbp_weights + (float(tail_boost) - 1.0) * tail_mask

    abs_loss = (sbp_elem * sbp_weights).mean() + dbp_elem.mean()

    consistency_sbp = F.smooth_l1_loss(
        delta_raw[:, 0], delta_from_abs[:, 0], beta=0.25, reduction="mean"
    )
    consistency_dbp = F.smooth_l1_loss(
        delta_raw[:, 1], delta_from_abs[:, 1], beta=0.20, reduction="mean"
    )
    consistency = consistency_sbp + 0.5 * consistency_dbp
    return abs_loss, consistency


# ──────────────────────────────────────────────
# One epoch
# ──────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train, fusion=False,
              use_mixup=False, mixup_alpha=0.2, ema=None,
              absolute_refine_weight=0.0, absolute_refine_consistency_weight=0.0,
              target_scale=10.0,
              absolute_refine_tail_low=90.0, absolute_refine_tail_high=150.0,
              absolute_refine_shoulder_low=110.0, absolute_refine_shoulder_high=130.0,
              absolute_refine_shoulder_boost=1.15, absolute_refine_tail_boost=1.60):
    if use_mixup:
        raise RuntimeError(
            "Mixup is disabled because branch-only mixing is inconsistent with "
            "anchor-relative delta targets."
        )
    model.train(train)
    criterion.train(train)   # enables label noise only during training
    total_loss = 0.0
    n_seen = 0
    all_pred, all_tgt = [], []

    with torch.set_grad_enabled(train):
        for batch in loader:
            if fusion:
                anchor, current, d_sbp, d_dbp, anc_sbp_norm, anc_dbp_norm, anc_spec, cur_spec, anc_hf, cur_hf = batch
                anc_spec = anc_spec.to(device)
                cur_spec = cur_spec.to(device)
            else:
                anchor, current, d_sbp, d_dbp, anc_sbp_norm, anc_dbp_norm, anc_hf, cur_hf = batch
                anc_spec = cur_spec = None

            anchor = anchor.to(device)
            current = current.to(device)
            anc_sbp_norm = anc_sbp_norm.to(device)
            anc_dbp_norm = anc_dbp_norm.to(device)
            anc_hf = anc_hf.to(device)
            cur_hf = cur_hf.to(device)
            target = torch.stack([d_sbp, d_dbp], dim=1).to(device)

            pred, extras = forward_pair(
                model, anchor, current, anc_sbp_norm, anc_dbp_norm,
                anchor_spec=anc_spec, current_spec=cur_spec,
                anchor_hf=anc_hf, current_hf=cur_hf,
                return_extras=True,
            )
            log_var = extras.get("log_var")
            loss = criterion((pred, log_var), target)
            if getattr(model, "use_absolute_refinement", False):
                abs_loss, consistency_loss = compute_absolute_refinement_losses(
                    extras, target, anc_sbp_norm, anc_dbp_norm,
                    target_scale=target_scale,
                    tail_low=absolute_refine_tail_low,
                    tail_high=absolute_refine_tail_high,
                    shoulder_low=absolute_refine_shoulder_low,
                    shoulder_high=absolute_refine_shoulder_high,
                    shoulder_boost=absolute_refine_shoulder_boost,
                    tail_boost=absolute_refine_tail_boost,
                )
                loss = loss + float(absolute_refine_weight) * abs_loss
                loss = loss + float(absolute_refine_consistency_weight) * consistency_loss

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                if ema is not None:
                    ema.update(model)

            total_loss += loss.item() * len(anchor)
            n_seen += len(anchor)
            all_pred.append(pred.detach().cpu().numpy())
            all_tgt.append(target.detach().cpu().numpy())

    all_pred = np.concatenate(all_pred)
    all_tgt = np.concatenate(all_tgt)
    return total_loss / max(n_seen, 1), all_pred, all_tgt


# ──────────────────────────────────────────────
# Test-Time Augmentation (TTA)
# ──────────────────────────────────────────────

def tta_augment_batch(tensor, noise_std=0.03, jitter_range=0.03):
    """Apply light augmentation to a batch of tensors for TTA.
    Lighter than training augmentation — just noise + amplitude jitter.
    """
    noise = torch.randn_like(tensor) * noise_std
    # Per-channel jitter: scale each channel independently
    if tensor.ndim == 3:  # (B, C, W) waveform
        scale = 1.0 + (torch.rand(tensor.shape[0], tensor.shape[1], 1,
                                   device=tensor.device) * 2 - 1) * jitter_range
    elif tensor.ndim == 4:  # (B, C, F, T) spectrogram
        scale = 1.0 + (torch.rand(tensor.shape[0], tensor.shape[1], 1, 1,
                                   device=tensor.device) * 2 - 1) * jitter_range
    else:
        scale = 1.0
    return tensor * scale + noise


def run_epoch_tta(model, loader, device, n_aug=10, fusion=False):
    """Run inference with TTA: average predictions over n_aug augmented copies.
    The original (clean) prediction is included as one of the copies.
    """
    model.eval()
    all_pred, all_tgt = [], []

    with torch.no_grad():
        for batch in loader:
            if fusion:
                anchor, current, d_sbp, d_dbp, anc_sbp_norm, anc_dbp_norm, anc_spec, cur_spec, anc_hf, cur_hf = batch
                anc_spec = anc_spec.to(device)
                cur_spec = cur_spec.to(device)
            else:
                anchor, current, d_sbp, d_dbp, anc_sbp_norm, anc_dbp_norm, anc_hf, cur_hf = batch
                anc_spec = cur_spec = None

            anchor = anchor.to(device)
            current = current.to(device)
            anc_sbp_norm = anc_sbp_norm.to(device)
            anc_dbp_norm = anc_dbp_norm.to(device)
            anc_hf = anc_hf.to(device)
            cur_hf = cur_hf.to(device)
            target = torch.stack([d_sbp, d_dbp], dim=1).to(device)

            # Clean prediction (1 of n_aug+1 total)
            preds = [forward_pair(
                model, anchor, current, anc_sbp_norm, anc_dbp_norm,
                anchor_spec=anc_spec, current_spec=cur_spec,
                anchor_hf=anc_hf, current_hf=cur_hf,
            )]

            # Augmented predictions
            for _ in range(n_aug):
                aug_anchor = tta_augment_batch(anchor)
                aug_current = tta_augment_batch(current)
                if fusion and anc_spec is not None:
                    aug_anc_spec = tta_augment_batch(anc_spec)
                    aug_cur_spec = tta_augment_batch(cur_spec)
                else:
                    aug_anc_spec = aug_cur_spec = None

                p = forward_pair(
                    model, aug_anchor, aug_current, anc_sbp_norm, anc_dbp_norm,
                    anchor_spec=aug_anc_spec, current_spec=aug_cur_spec,
                    anchor_hf=anc_hf, current_hf=cur_hf,
                )
                preds.append(p)

            # Average all predictions
            avg_pred = torch.stack(preds, dim=0).mean(dim=0)
            all_pred.append(avg_pred.cpu().numpy())
            all_tgt.append(target.cpu().numpy())

    all_pred = np.concatenate(all_pred)
    all_tgt = np.concatenate(all_tgt)
    return all_pred, all_tgt


def predict_delta_with_tta(model, anchor, current, anc_sbp_norm, anc_dbp_norm,
                           anc_hf, cur_hf, anchor_spec=None, current_spec=None, n_aug=0):
    """Average 1 clean + n_aug augmented delta predictions and pair precision."""
    pred0, extras0 = forward_pair(
        model, anchor, current, anc_sbp_norm, anc_dbp_norm,
        anchor_spec=anchor_spec, current_spec=current_spec,
        anchor_hf=anc_hf, current_hf=cur_hf,
        return_extras=True,
    )
    log_var0 = extras0.get("log_var")
    preds = [pred0]
    precisions = []
    if log_var0 is not None:
        precisions.append(torch.exp(-log_var0))

    for _ in range(n_aug):
        aug_anchor = tta_augment_batch(anchor)
        aug_current = tta_augment_batch(current)
        aug_anchor_spec = tta_augment_batch(anchor_spec) if anchor_spec is not None else None
        aug_current_spec = tta_augment_batch(current_spec) if current_spec is not None else None
        pred_aug, extras_aug = forward_pair(
            model, aug_anchor, aug_current, anc_sbp_norm, anc_dbp_norm,
            anchor_spec=aug_anchor_spec, current_spec=aug_current_spec,
            anchor_hf=anc_hf, current_hf=cur_hf,
            return_extras=True,
        )
        log_var_aug = extras_aug.get("log_var")
        preds.append(pred_aug)
        if log_var_aug is not None:
            precisions.append(torch.exp(-log_var_aug))

    avg_pred = torch.stack(preds, dim=0).mean(dim=0)
    avg_precision = None
    if precisions:
        avg_precision = torch.stack(precisions, dim=0).mean(dim=0)
    return avg_pred, avg_precision


def compute_anchor_similarity_scores(model, mode, cur_sub, cur_spec_sub, cur_hf_sub,
                                     anc_wave, anc_spec, anc_hf, temperature):
    """Return per-sample similarity scores between current windows and one anchor."""
    if mode == "embedding":
        cur_feat = model._encode(cur_sub, cur_spec_sub)
        anc_feat = model._encode(anc_wave[:1], anc_spec[:1] if anc_spec is not None else None)
        cur_feat = F.normalize(cur_feat, dim=1)
        anc_feat = F.normalize(anc_feat, dim=1)
        score = torch.matmul(cur_feat, anc_feat.T).squeeze(1)
    elif mode == "handcrafted":
        anc_vec = anc_hf[:1]
        score = -torch.mean(torch.abs(cur_hf_sub - anc_vec), dim=1)
    else:
        score = torch.zeros(cur_sub.shape[0], dtype=torch.float32, device=cur_sub.device)

    temp = max(float(temperature), 1e-4)
    return score / temp


def run_multi_anchor_inference(model, dataset, device, batch_size=256, fusion=False, n_aug=0,
                               anchor_similarity_mode="none", anchor_top_k=0,
                               anchor_similarity_temp=0.35):
    """
    Evaluate a dataset by averaging absolute BP predictions across all anchors
    available for each patient.
    """
    model.eval()
    all_pred_abs, all_tgt_abs = [], []
    indices = dataset.indices

    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            current = torch.tensor(dataset.X[batch_indices], dtype=torch.float32, device=device)
            cur_hf = torch.tensor(dataset.hand_features[batch_indices], dtype=torch.float32, device=device)
            cur_spec = None
            if fusion and dataset.X_spec is not None:
                cur_spec = torch.tensor(dataset.X_spec[batch_indices], dtype=torch.float32, device=device)

            batch_pids = dataset.patient_ids[batch_indices].astype(np.int64)
            batch_pred_sum = torch.zeros((len(batch_indices), 2), dtype=torch.float32, device=device)
            batch_weight_sum = torch.zeros((len(batch_indices), 2), dtype=torch.float32, device=device)

            for pid in np.unique(batch_pids):
                pid = int(pid)
                mask_np = (batch_pids == pid)
                mask = torch.as_tensor(mask_np, dtype=torch.bool, device=device)
                cur_sub = current[mask]
                cur_hf_sub = cur_hf[mask]
                cur_spec_sub = cur_spec[mask] if cur_spec is not None else None
                n_sub = cur_sub.shape[0]
                anc_indices = [int(a) for a in dataset.anchor_lists[pid]]

                similarity_logits = None
                if anchor_similarity_mode != "none" and len(anc_indices) > 1:
                    score_cols = []
                    for anc_i in anc_indices:
                        anc_wave_single = torch.tensor(dataset.X[anc_i], dtype=torch.float32, device=device).unsqueeze(0)
                        anc_hf_single = torch.tensor(dataset.hand_features[anc_i], dtype=torch.float32, device=device).unsqueeze(0)
                        anc_spec_single = None
                        if fusion and dataset.X_spec is not None:
                            anc_spec_single = torch.tensor(dataset.X_spec[anc_i], dtype=torch.float32, device=device).unsqueeze(0)
                        score_cols.append(
                            compute_anchor_similarity_scores(
                                model, anchor_similarity_mode,
                                cur_sub, cur_spec_sub, cur_hf_sub,
                                anc_wave_single, anc_spec_single, anc_hf_single,
                                anchor_similarity_temp,
                            )
                        )
                    similarity_logits = torch.stack(score_cols, dim=1)
                    if anchor_top_k > 0 and anchor_top_k < similarity_logits.shape[1]:
                        top_idx = torch.topk(similarity_logits, k=anchor_top_k, dim=1).indices
                        keep_mask = torch.zeros_like(similarity_logits, dtype=torch.bool)
                        keep_mask.scatter_(1, top_idx, True)
                        similarity_logits = similarity_logits.masked_fill(~keep_mask, -1e9)
                    similarity_weights = torch.softmax(similarity_logits, dim=1)
                else:
                    similarity_weights = None

                for anc_pos, anc_i in enumerate(anc_indices):
                    anc_wave = torch.tensor(dataset.X[anc_i], dtype=torch.float32, device=device).unsqueeze(0).expand(n_sub, -1, -1)
                    anc_hf = torch.tensor(dataset.hand_features[anc_i], dtype=torch.float32, device=device).unsqueeze(0).expand(n_sub, -1)
                    anc_spec = None
                    if fusion and dataset.X_spec is not None:
                        anc_spec = torch.tensor(dataset.X_spec[anc_i], dtype=torch.float32, device=device).unsqueeze(0).expand(n_sub, -1, -1, -1)

                    anc_sbp_norm = torch.full((n_sub,), float(dataset.y_sbp[anc_i]) / 100.0, dtype=torch.float32, device=device)
                    anc_dbp_norm = torch.full((n_sub,), float(dataset.y_dbp[anc_i]) / 100.0, dtype=torch.float32, device=device)

                    delta_pred, pair_precision = predict_delta_with_tta(
                        model, anc_wave, cur_sub, anc_sbp_norm, anc_dbp_norm,
                        anc_hf, cur_hf_sub, anchor_spec=anc_spec,
                        current_spec=cur_spec_sub, n_aug=n_aug,
                    )
                    abs_pred = delta_pred.clone()
                    abs_pred[:, 0] = abs_pred[:, 0] * dataset.target_scale + float(dataset.y_sbp[anc_i])
                    abs_pred[:, 1] = abs_pred[:, 1] * dataset.target_scale + float(dataset.y_dbp[anc_i])

                    if pair_precision is None:
                        pair_weight = torch.ones_like(abs_pred)
                    else:
                        pair_weight = pair_precision.clamp(min=1e-3, max=50.0)
                    if similarity_weights is not None:
                        sim_weight = similarity_weights[:, anc_pos].unsqueeze(1)
                        pair_weight = pair_weight * sim_weight

                    batch_pred_sum[mask] += abs_pred * pair_weight
                    batch_weight_sum[mask] += pair_weight

            batch_pred = batch_pred_sum / batch_weight_sum.clamp(min=1e-6)
            batch_tgt = np.stack([dataset.y_sbp[batch_indices], dataset.y_dbp[batch_indices]], axis=1).astype(np.float32)
            all_pred_abs.append(batch_pred.cpu().numpy())
            all_tgt_abs.append(batch_tgt)

    return np.concatenate(all_pred_abs), np.concatenate(all_tgt_abs)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mixup_requested = bool(args.use_mixup)
    mixup_active = False
    print(f"\n{'='*60}")
    print(f"  Siamese BP Estimation — Training")
    print(f"  Device      : {device}")
    print(f"  Data        : {args.data_dir}")
    print(f"  LR          : {args.lr}  |  Warmup: {args.warmup_epochs} epochs")
    print(f"  SBP loss    : {args.sbp_loss_type.upper()}  |  DBP loss: Huber(delta={args.huber_delta_dbp})")
    print(f"  Huber delta : SBP={args.huber_delta_sbp} DBP={args.huber_delta_dbp}  |  Label noise: {args.label_noise_std}  |  Target scale: {args.target_scale}")
    print(f"  Aux loss    : Pearson={args.corr_weight}  CCC={args.ccc_weight}  |  Scale SBP={args.sbp_scale_weight} DBP={args.dbp_scale_weight}")
    print(f"  SBP weight  : {args.sbp_weight}  |  Epochs: {args.epochs}")
    print(f"  Resume from : {args.resume if args.resume else 'scratch'}")
    print(f"  Dropout     : {args.dropout}  |  Embed dim: {args.embed_dim}")
    print(f"  Weight decay: {args.weight_decay}  |  Max win/pt: {args.max_windows_per_patient}")
    print(f"  Num anchors : {args.num_anchors}  (per patient)")
    if mixup_requested:
        print(f"  Mixup       : requested (alpha={args.mixup_alpha}) but DISABLED for delta supervision")
    else:
        print(f"  Mixup       : OFF")
    if args.use_fusion:
        input_mode = "FUSION (1D + 2D CNN)"
    elif args.use_spectrograms:
        input_mode = "SPECTROGRAMS (2D CNN)"
    else:
        input_mode = "Raw waveforms (1D CNN)"
    print(f"  Input mode  : {input_mode}")
    if args.use_beat_features:
        print(f"  Beat aware  : ON (beat-summary feature stream)")
    if args.use_ecg:
        if args.ecg_input_mode == "both":
            ecg_desc = "ECG raw channel + timing features"
        elif args.ecg_input_mode == "features_only":
            ecg_desc = "ECG timing features only"
        else:
            ecg_desc = "ECG raw channel only"
        print(f"  Hybrid input: {ecg_desc}")
    print(f"  PPG norm    : {'amplitude-preserving' if args.preserve_ppg_amplitude else 'per-window z-score'}")
    print(f"  Quality thr : {args.quality_threshold}")
    print(f"  Eval split  : {'shuffled' if args.shuffle_eval else 'chronological'}")
    if args.multi_anchor_eval:
        if args.anchor_similarity_mode != "none":
            topk_txt = f", top-k={args.anchor_top_k}" if args.anchor_top_k > 0 else ""
            eval_mode = (
                f"similarity-guided multi-anchor ({args.anchor_similarity_mode}{topk_txt}, "
                f"temp={args.anchor_similarity_temp})"
            )
        else:
            eval_mode = "confidence-weighted multi-anchor" if args.use_pair_uncertainty else "multi-anchor average"
    else:
        eval_mode = "single-anchor"
    print(f"  Eval mode   : {eval_mode}")
    if args.multi_anchor_eval and args.multi_anchor_eval_every > 1:
        print(
            f"  Eval cadence: full multi-anchor every {args.multi_anchor_eval_every} epochs "
            f"(first/last always); skipped epochs use direct-val proxy"
        )
    if args.train_eval_every > 1:
        print(
            f"  Train eval  : full train-set eval every {args.train_eval_every} epochs "
            f"(first/last + scored epochs always)"
        )
    print(f"  Pair block  : {'gated interaction' if args.use_gated_pair_interaction else 'diff + hadamard'}")
    if args.use_absolute_refinement:
        print(
            "  Abs refine  : ON "
            f"(w={args.absolute_refine_weight}, cons={args.absolute_refine_consistency_weight}, "
            f"tails <{args.absolute_refine_tail_low:.0f}/>{args.absolute_refine_tail_high:.0f} "
            f"x{args.absolute_refine_tail_boost:.2f})"
        )
    sampling_modes = []
    if args.balanced_sampling:
        sampling_modes.append("delta-tail")
    if args.sbp_tail_sampling:
        sampling_modes.append("SBP-tail")
    print(f"  BP sampling : {' + '.join(sampling_modes) if sampling_modes else 'uniform'}")
    if args.sbp_tail_sampling:
        print(
            "  SBP tails   : "
            f"<{args.sbp_tail_low:.0f} or >{args.sbp_tail_high:.0f} x{args.sbp_tail_boost:.2f}  |  "
            f"shoulders {args.sbp_tail_low:.0f}-{args.sbp_shoulder_low:.0f}/"
            f"{args.sbp_shoulder_high:.0f}-{args.sbp_tail_high:.0f} x{args.sbp_shoulder_boost:.2f}"
        )
    print(f"  Pair uncert : {'ON' if args.use_pair_uncertainty else 'OFF'}")
    if args.freeze_feature_epochs > 0:
        print(f"  Fine-tune    : freeze feature extractors for first {args.freeze_feature_epochs} epochs")
    print(
        f"  EMA eval    : {'ON' if args.use_ema else 'OFF'}"
        + (f" (decay={args.ema_decay}, start={args.ema_start_epoch})" if args.use_ema else "")
    )
    if args.gap_penalty > 0:
        print(f"  Gap score   : penalty={args.gap_penalty} after target gap {args.gap_target}")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────
    train_ds, val_ds, test_ds, anchors, y_sbp_all, y_dbp_all, sample_weights = load_data(
        args.data_dir, random_anchor=False, min_sbp_std=args.min_sbp_std,
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

    patient_ids = np.load(os.path.join(args.data_dir, "patient_ids.npy"))

    # Cap windows per patient on train set only
    if args.max_windows_per_patient > 0:
        print(f"\nCapping train to {args.max_windows_per_patient} windows/patient...")
        before = len(train_ds)
        train_ds = cap_dataset(train_ds, patient_ids, args.max_windows_per_patient, seed=args.seed)
        print(f"  Train windows: {before:,} -> {len(train_ds):,}")

    # Clean, non-augmented copy for fair train-vs-val evaluation
    train_eval_ds = copy(train_ds)
    if hasattr(train_eval_ds, "augment"):
        train_eval_ds.augment = False
    if hasattr(train_eval_ds, "anchor_selection"):
        train_eval_ds.anchor_selection = "first"
    if hasattr(train_eval_ds, "fixed_pairs"):
        train_eval_ds.fixed_pairs = None

    # Pair-aware sampling: optionally focus on delta tails, absolute SBP tails,
    # or both, after train-set capping.
    if args.balanced_sampling or args.sbp_tail_sampling:
        from dataset import build_pair_pool
        train_pairs, sample_weights = build_pair_pool(
            train_ds.indices, patient_ids, anchors, y_sbp_all, y_dbp_all,
            use_delta_weights=args.balanced_sampling,
            use_sbp_tail_weights=args.sbp_tail_sampling,
            sbp_tail_low=args.sbp_tail_low,
            sbp_tail_high=args.sbp_tail_high,
            sbp_shoulder_low=args.sbp_shoulder_low,
            sbp_shoulder_high=args.sbp_shoulder_high,
            sbp_shoulder_boost=args.sbp_shoulder_boost,
            sbp_tail_boost=args.sbp_tail_boost,
        )
        train_ds.fixed_pairs = train_pairs
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_ds.indices),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.workers,
            pin_memory=(device == "cuda"),
        )
        sampler_desc = []
        if args.balanced_sampling:
            sampler_desc.append("delta-tail")
        if args.sbp_tail_sampling:
            sampler_desc.append("SBP-tail")
        print(
            f"  Using pair-aware WeightedRandomSampler ({len(sample_weights):,} pairs; "
            f"{' + '.join(sampler_desc)})"
        )
        print(f"  Pair weight range: {sample_weights.min():.2f} - {sample_weights.max():.2f}")
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=(device == "cuda"),
        )
    train_eval_loader = DataLoader(
        train_eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device == "cuda"),
    )

    # ── Model ─────────────────────────────────
    model = build_model(embed_dim=args.embed_dim, dropout=args.dropout, device=device,
                        use_spectrograms=args.use_spectrograms,
                        use_fusion=args.use_fusion,
                        use_pair_uncertainty=args.use_pair_uncertainty,
                        use_gated_pair_interaction=args.use_gated_pair_interaction,
                        use_absolute_refinement=args.use_absolute_refinement,
                        hand_feature_dim=train_ds.hand_feature_dim,
                        input_channels=train_ds.num_channels,
                        target_scale=args.target_scale)

    # Resume from checkpoint if specified
    if args.resume:
        print(f"  Resuming from: {args.resume}")
        state = torch.load(args.resume, map_location=device, weights_only=True)
        if args.resume_partial:
            model_state = model.state_dict()
            compatible_state = {}
            skipped = []
            for key, value in state.items():
                if key in model_state and model_state[key].shape == value.shape:
                    compatible_state[key] = value
                else:
                    skipped.append(key)
            model_state.update(compatible_state)
            model.load_state_dict(model_state)
            print(
                f"  Loaded {len(compatible_state)}/{len(model_state)} tensors from checkpoint "
                f"(partial resume; skipped {len(skipped)} mismatched tensors)"
            )
            if skipped:
                preview = ", ".join(skipped[:8])
                suffix = " ..." if len(skipped) > 8 else ""
                print(f"  Skipped keys: {preview}{suffix}")
        else:
            model.load_state_dict(state)
            print(f"  Loaded checkpoint weights successfully")
    features_frozen = False
    if args.freeze_feature_epochs > 0:
        set_feature_extractors_trainable(model, trainable=False)
        features_frozen = True
        print("  Feature extractors frozen for fine-tuning warm start")

    criterion = AdvancedBPLoss(
        sbp_weight=args.sbp_weight,
        dbp_weight=1.0,
        huber_delta_sbp=args.huber_delta_sbp,
        huber_delta_dbp=args.huber_delta_dbp,
        label_noise_std=args.label_noise_std,
        corr_weight=args.corr_weight,
        dbp_corr_boost=args.dbp_corr_boost,
        ccc_weight=args.ccc_weight,
        dbp_ccc_boost=args.dbp_ccc_boost,
        direction_weight=args.direction_weight,
        sbp_scale_weight=args.sbp_scale_weight,
        dbp_scale_weight=args.dbp_scale_weight,
        sbp_loss_type=args.sbp_loss_type,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    ema = ModelEMA(model, decay=args.ema_decay) if args.use_ema else None

    # Start gently, then switch to cosine decay.
    # This made early training less jumpy in the longer runs.
    warmup_epochs = args.warmup_epochs
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - warmup_epochs,
        eta_min=args.lr * 0.01,
    )
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,     # start at lr * 0.1
        end_factor=1.0,       # ramp to full lr
        total_iters=warmup_epochs,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    # ── Training ──────────────────────────────
    best_val_score = float("inf")
    patience_ctr = 0
    history = {
        "train_loss": [],
        "train_eval_loss": [],
        "val_loss": [],
        "gap": [],
        "selection_score": [],
        "val_mae_sbp": [],
        "val_mae_dbp": [],
    }

    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    last_train_eval_loss = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        if features_frozen and epoch > args.freeze_feature_epochs:
            set_feature_extractors_trainable(model, trainable=True)
            features_frozen = False
            print(f"  Unfroze feature extractors at epoch {epoch}")

        run_full_multi_anchor_eval = (
            args.multi_anchor_eval and (
                args.multi_anchor_eval_every <= 1
                or epoch == 1
                or epoch == args.epochs
                or (epoch % args.multi_anchor_eval_every == 0)
            )
        )

        run_train_eval = (
            (not args.multi_anchor_eval)
            or run_full_multi_anchor_eval
            or args.train_eval_every <= 1
            or epoch == 1
            or epoch == args.epochs
            or (epoch % args.train_eval_every == 0)
        )

        use_ema_epoch = ema is not None and epoch >= args.ema_start_epoch
        train_loss, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True,
            fusion=args.use_fusion, use_mixup=mixup_active,
            mixup_alpha=args.mixup_alpha,
            ema=ema,
            absolute_refine_weight=args.absolute_refine_weight,
            absolute_refine_consistency_weight=args.absolute_refine_consistency_weight,
            target_scale=args.target_scale,
            absolute_refine_tail_low=args.absolute_refine_tail_low,
            absolute_refine_tail_high=args.absolute_refine_tail_high,
            absolute_refine_shoulder_low=args.absolute_refine_shoulder_low,
            absolute_refine_shoulder_high=args.absolute_refine_shoulder_high,
            absolute_refine_shoulder_boost=args.absolute_refine_shoulder_boost,
            absolute_refine_tail_boost=args.absolute_refine_tail_boost,
        )
        if use_ema_epoch:
            ema.store(model)
            ema.copy_to(model)

        if run_train_eval:
            train_eval_loss, _, _ = run_epoch(
                model, train_eval_loader, criterion, None, device, train=False,
                fusion=args.use_fusion,
                absolute_refine_weight=args.absolute_refine_weight,
                absolute_refine_consistency_weight=args.absolute_refine_consistency_weight,
                target_scale=args.target_scale,
                absolute_refine_tail_low=args.absolute_refine_tail_low,
                absolute_refine_tail_high=args.absolute_refine_tail_high,
                absolute_refine_shoulder_low=args.absolute_refine_shoulder_low,
                absolute_refine_shoulder_high=args.absolute_refine_shoulder_high,
                absolute_refine_shoulder_boost=args.absolute_refine_shoulder_boost,
                absolute_refine_tail_boost=args.absolute_refine_tail_boost,
            )
            last_train_eval_loss = train_eval_loss
            train_eval_mode_tag = "full"
        else:
            train_eval_loss = train_loss if last_train_eval_loss is None else last_train_eval_loss
            train_eval_mode_tag = "proxy"

        val_loss, val_pred, val_tgt = run_epoch(
            model, val_loader, criterion, None, device, train=False,
            fusion=args.use_fusion,
            absolute_refine_weight=args.absolute_refine_weight,
            absolute_refine_consistency_weight=args.absolute_refine_consistency_weight,
            target_scale=args.target_scale,
            absolute_refine_tail_low=args.absolute_refine_tail_low,
            absolute_refine_tail_high=args.absolute_refine_tail_high,
            absolute_refine_shoulder_low=args.absolute_refine_shoulder_low,
            absolute_refine_shoulder_high=args.absolute_refine_shoulder_high,
            absolute_refine_shoulder_boost=args.absolute_refine_shoulder_boost,
            absolute_refine_tail_boost=args.absolute_refine_tail_boost,
        )

        proxy_mae_sbp = float(np.abs(val_pred[:, 0] - val_tgt[:, 0]).mean()) * args.target_scale
        proxy_mae_dbp = float(np.abs(val_pred[:, 1] - val_tgt[:, 1]).mean()) * args.target_scale

        if run_full_multi_anchor_eval:
            val_pred_mmhg, val_tgt_mmhg = run_multi_anchor_inference(
                model, val_ds, device,
                batch_size=args.eval_batch_size,
                fusion=args.use_fusion,
                n_aug=0,
                anchor_similarity_mode=args.anchor_similarity_mode,
                anchor_top_k=args.anchor_top_k,
                anchor_similarity_temp=args.anchor_similarity_temp,
            )
            mae_sbp = float(np.abs(val_pred_mmhg[:, 0] - val_tgt_mmhg[:, 0]).mean())
            mae_dbp = float(np.abs(val_pred_mmhg[:, 1] - val_tgt_mmhg[:, 1]).mean())
            eval_mode_tag = "full"
        else:
            mae_sbp = proxy_mae_sbp
            mae_dbp = proxy_mae_dbp
            eval_mode_tag = "proxy" if args.multi_anchor_eval else "direct"
        combined_mae = (mae_sbp + mae_dbp) / 2.0
        gap_value = val_loss - train_eval_loss

        history["train_loss"].append(train_loss)
        history["train_eval_loss"].append(train_eval_loss)
        history["val_loss"].append(val_loss)
        history["gap"].append(gap_value)
        history["val_mae_sbp"].append(mae_sbp)
        history["val_mae_dbp"].append(mae_dbp)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Give SBP more weight when choosing checkpoints. The optional gap
        # term avoids picking a model with a much worse val/train gap.
        if args.multi_anchor_eval and not run_full_multi_anchor_eval:
            score = float("nan")
            history["selection_score"].append(score)
            tag = " (proxy val)"
        else:
            sbp_weighted_mae = 0.7 * mae_sbp + 0.3 * mae_dbp
            score = sbp_weighted_mae + args.gap_penalty * max(0.0, gap_value - args.gap_target)
            history["selection_score"].append(score)
            improved = score < (best_val_score - args.min_delta)
            if improved:
                best_val_score = score
                patience_ctr = 0
                torch.save(model.state_dict(), save_path / "best_model.pt")
                tag = " * saved"
            else:
                patience_ctr += 1
                tag = f" (patience {patience_ctr}/{args.patience})"

        if use_ema_epoch:
            ema.restore(model)

        elapsed = time.time() - t0
        score_txt = ""
        if args.gap_penalty > 0:
            score_txt = f"  score={score:.2f}" if np.isfinite(score) else "  score=proxy"
        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"tr={train_loss:.2f}  tr_eval={train_eval_loss:.2f}({train_eval_mode_tag})  val={val_loss:.2f}  "
            f"gap={gap_value:.2f}  "
            f"SBP={mae_sbp:.2f}  DBP={mae_dbp:.2f}  eval={eval_mode_tag}  "
            f"comb={combined_mae:.2f}{score_txt}  lr={current_lr:.2e}  [{elapsed:.0f}s]{tag}"
        )

        if patience_ctr >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    # ── Test ──────────────────────────────────
    print("\n" + "=" * 56)
    print("  Test Set Evaluation (best model)")
    print("=" * 56)

    model.load_state_dict(torch.load(save_path / "best_model.pt", map_location=device, weights_only=True))
    if args.multi_anchor_eval:
        test_pred_mmhg, test_tgt_mmhg = run_multi_anchor_inference(
            model, test_ds, device,
            batch_size=args.eval_batch_size,
            fusion=args.use_fusion,
            n_aug=0,
            anchor_similarity_mode=args.anchor_similarity_mode,
            anchor_top_k=args.anchor_top_k,
            anchor_similarity_temp=args.anchor_similarity_temp,
        )
    else:
        _, test_pred, test_tgt = run_epoch(model, test_loader, criterion, None, device, train=False,
                                           fusion=args.use_fusion,
                                           absolute_refine_weight=args.absolute_refine_weight,
                                           absolute_refine_consistency_weight=args.absolute_refine_consistency_weight,
                                           target_scale=args.target_scale,
                                           absolute_refine_tail_low=args.absolute_refine_tail_low,
                                           absolute_refine_tail_high=args.absolute_refine_tail_high,
                                           absolute_refine_shoulder_low=args.absolute_refine_shoulder_low,
                                           absolute_refine_shoulder_high=args.absolute_refine_shoulder_high,
                                           absolute_refine_shoulder_boost=args.absolute_refine_shoulder_boost,
                                           absolute_refine_tail_boost=args.absolute_refine_tail_boost)
        test_pred_mmhg = test_pred * args.target_scale
        test_tgt_mmhg  = test_tgt  * args.target_scale
    print_metrics(test_pred_mmhg, test_tgt_mmhg, label="Test")

    # ── Test-Time Augmentation ────────────────
    if args.tta > 0:
        print(f"\n{'='*56}")
        print(f"  Test Set + TTA (n_aug={args.tta})")
        print(f"{'='*56}")
        if args.multi_anchor_eval:
            tta_pred_mmhg, tta_tgt_mmhg = run_multi_anchor_inference(
                model, test_ds, device,
                batch_size=args.eval_batch_size,
                fusion=args.use_fusion,
                n_aug=args.tta,
                anchor_similarity_mode=args.anchor_similarity_mode,
                anchor_top_k=args.anchor_top_k,
                anchor_similarity_temp=args.anchor_similarity_temp,
            )
        else:
            tta_pred, tta_tgt = run_epoch_tta(
                model, test_loader, device,
                n_aug=args.tta, fusion=args.use_fusion,
            )
            tta_pred_mmhg = tta_pred * args.target_scale
            tta_tgt_mmhg  = tta_tgt  * args.target_scale
        print_metrics(tta_pred_mmhg, tta_tgt_mmhg, label="Test+TTA")
        np.save(save_path / "test_pred_tta.npy", tta_pred_mmhg)

    np.save(save_path / "history.npy", history)
    np.save(save_path / "test_pred.npy", test_pred_mmhg)
    np.save(save_path / "test_tgt.npy", test_tgt_mmhg)
    print(f"\nOutputs saved to {save_path}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=r"C:\MIMIC2_out_v8")
    parser.add_argument("--save_dir", default=r".\checkpoints")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=256)       # larger batch = more stable gradients
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.30)       # let model learn — multi-anchor handles regularization
    parser.add_argument("--embed_dim", type=int, default=256)        # full capacity
    parser.add_argument("--weight_decay", type=float, default=1e-3)  # lighter — was over-regularized
    parser.add_argument("--huber_delta_sbp", type=float, default=0.1)  # tighter for SBP — penalise large errors harder
    parser.add_argument("--huber_delta_dbp", type=float, default=0.2)  # keep original for DBP (already grade A)
    parser.add_argument("--sbp_weight", type=float, default=1.5)     # upweight SBP — separate heads prevent DBP starvation
    parser.add_argument("--target_scale", type=float, default=10.0)  # divide delta targets by this → loss in 0.1-0.3 range
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min_delta", type=float, default=0.02)
    parser.add_argument("--max_windows_per_patient", type=int, default=300)  # was 400 → less per-patient memorization
    parser.add_argument("--min_sbp_std", type=float, default=5.0)    # was 8.0 → keep more patients with v8
    parser.add_argument("--warmup_epochs", type=int, default=5)      # NEW: linear warmup
    parser.add_argument("--label_noise_std", type=float, default=0.03)# scaled: was 0.3 mmHg, now /target_scale
    parser.add_argument("--corr_weight", type=float, default=0.05)    # Pearson auxiliary weight
    parser.add_argument("--dbp_corr_boost", type=float, default=1.25) # slightly stronger trend tracking for DBP
    parser.add_argument("--ccc_weight", type=float, default=0.12)     # CCC penalty (correlation + bias + scale)
    parser.add_argument("--dbp_ccc_boost", type=float, default=1.15)  # slightly stronger DBP concordance term
    parser.add_argument("--direction_weight", type=float, default=0.05) # direction loss weight (penalise wrong sign)
    parser.add_argument("--sbp_scale_weight", type=float, default=0.08) # slope/range penalty for SBP
    parser.add_argument("--dbp_scale_weight", type=float, default=0.02) # lighter slope/range penalty for DBP
    parser.add_argument("--num_anchors", type=int, default=5)       # K=5 anchors — more SBP delta diversity per patient
    parser.add_argument("--use_spectrograms", action="store_true") # use STFT spectrograms instead of raw waveforms
    parser.add_argument("--use_fusion", action="store_true")      # fuse waveform + spectrogram encoders
    parser.add_argument("--use_beat_features", action="store_true", default=False,
                        help="Append beat-summary PPG features to the Siamese feature stream")
    parser.add_argument("--no_beat_features", dest="use_beat_features", action="store_false")
    parser.add_argument("--use_ecg", action="store_true", default=False,
                        help="Load hybrid ECG+PPG windows when X_ecg.npy is present")
    parser.add_argument("--no_ecg", dest="use_ecg", action="store_false")
    parser.add_argument("--ecg_input_mode", choices=["both", "features_only", "raw_only"],
                        default="both",
                        help="How to use ECG when --use_ecg is enabled")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (e.g. ./checkpoints_v13/best_model.pt)")
    parser.add_argument("--resume_partial", action="store_true", default=False,
                        help="Load only checkpoint tensors whose names and shapes match the current model")
    parser.add_argument("--sbp_loss_type", type=str, default="mse",
                        choices=["huber", "mse"],
                        help="Loss type for SBP: 'mse' (default, stronger on large misses) or 'huber'")
    parser.add_argument("--use_mixup", action="store_true", default=False,
                        help="Deprecated; disabled because branch-only mixup is inconsistent with anchor-relative deltas.")
    parser.add_argument("--no_mixup", dest="use_mixup", action="store_false")
    parser.add_argument("--mixup_alpha", type=float, default=0.2) # Beta distribution parameter
    parser.add_argument("--tta", type=int, default=10)            # TTA augmentations (0=off, 10=default)
    parser.add_argument("--quality_threshold", type=float, default=0.4) # min signal quality (0=off)
    parser.add_argument("--shuffle_eval", action="store_true", default=True) # shuffle val/test split
    parser.add_argument("--no_shuffle_eval", dest="shuffle_eval", action="store_false")
    parser.add_argument("--preserve_ppg_amplitude", action="store_true", default=True)
    parser.add_argument("--no_preserve_ppg_amplitude", dest="preserve_ppg_amplitude", action="store_false")
    parser.add_argument("--multi_anchor_eval", action="store_true", default=True)
    parser.add_argument("--no_multi_anchor_eval", dest="multi_anchor_eval", action="store_false")
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--multi_anchor_eval_every", type=int, default=1,
                        help="Run expensive full multi-anchor validation every N epochs; "
                             "skipped epochs use direct-val proxy metrics for logging only")
    parser.add_argument("--train_eval_every", type=int, default=1,
                        help="Run full train-set evaluation every N epochs; "
                             "skipped epochs reuse the last scored value or the current train loss")
    parser.add_argument("--anchor_similarity_mode", choices=["none", "embedding", "handcrafted"], default="none")
    parser.add_argument("--anchor_top_k", type=int, default=0,
                        help="Use only the top-k most similar anchors during multi-anchor aggregation (0 = all anchors)")
    parser.add_argument("--anchor_similarity_temp", type=float, default=0.35,
                        help="Softmax temperature for similarity-guided anchor weighting")
    parser.add_argument("--balanced_sampling", action="store_true", default=True) # oversample extreme SBP
    parser.add_argument("--no_balanced_sampling", dest="balanced_sampling", action="store_false")
    parser.add_argument("--sbp_tail_sampling", action="store_true", default=False,
                        help="Oversample absolute SBP tails/shoulders without re-enabling full delta-tail sampling")
    parser.add_argument("--no_sbp_tail_sampling", dest="sbp_tail_sampling", action="store_false")
    parser.add_argument("--sbp_tail_low", type=float, default=90.0,
                        help="Lower SBP tail threshold for sampling emphasis")
    parser.add_argument("--sbp_tail_high", type=float, default=150.0,
                        help="Upper SBP tail threshold for sampling emphasis")
    parser.add_argument("--sbp_shoulder_low", type=float, default=110.0,
                        help="Upper edge of the low-SBP shoulder region")
    parser.add_argument("--sbp_shoulder_high", type=float, default=130.0,
                        help="Lower edge of the high-SBP shoulder region")
    parser.add_argument("--sbp_shoulder_boost", type=float, default=1.15,
                        help="Sampling multiplier for SBP shoulder ranges")
    parser.add_argument("--sbp_tail_boost", type=float, default=1.60,
                        help="Sampling multiplier for SBP tail ranges")
    parser.add_argument("--use_gated_pair_interaction", action="store_true", default=False)
    parser.add_argument("--no_gated_pair_interaction", dest="use_gated_pair_interaction", action="store_false")
    parser.add_argument("--use_absolute_refinement", action="store_true", default=False,
                        help="Add an auxiliary absolute-BP head that refines delta predictions")
    parser.add_argument("--no_absolute_refinement", dest="use_absolute_refinement", action="store_false")
    parser.add_argument("--absolute_refine_weight", type=float, default=0.30,
                        help="Weight for auxiliary absolute-BP supervision")
    parser.add_argument("--absolute_refine_consistency_weight", type=float, default=0.08,
                        help="Weight for consistency between raw-delta and absolute-guided delta paths")
    parser.add_argument("--absolute_refine_tail_low", type=float, default=90.0)
    parser.add_argument("--absolute_refine_tail_high", type=float, default=150.0)
    parser.add_argument("--absolute_refine_shoulder_low", type=float, default=110.0)
    parser.add_argument("--absolute_refine_shoulder_high", type=float, default=130.0)
    parser.add_argument("--absolute_refine_shoulder_boost", type=float, default=1.20)
    parser.add_argument("--absolute_refine_tail_boost", type=float, default=1.75)
    parser.add_argument("--use_pair_uncertainty", action="store_true", default=False)
    parser.add_argument("--no_pair_uncertainty", dest="use_pair_uncertainty", action="store_false")
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--no_ema", dest="use_ema", action="store_false")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--ema_start_epoch", type=int, default=5)
    parser.add_argument("--gap_penalty", type=float, default=0.15)
    parser.add_argument("--gap_target", type=float, default=1.0)
    parser.add_argument("--freeze_feature_epochs", type=int, default=0,
                        help="Freeze shared feature extractors for the first N epochs (useful for checkpoint fine-tuning)")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    train(args)
