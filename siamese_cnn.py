"""
siamese_cnn.py
--------------
Siamese CNN for cuff-less blood pressure estimation from PPG signals.

Architecture follows the calibration-style setup from the project brief:

    ┌─────────────────┐      ┌─────────────────┐
    │  Anchor window  │      │  Current window │
    │   (3 × W)       │      │    (3 × W)      │
    └────────┬────────┘      └────────┬────────┘
             │                        │
             └──────── CNN ───────────┘   (shared weights)
                          │
                    concatenate
                          │
                    ReLU → FC → ReLU → FC
                          │
                  [ΔSBP,  ΔDBP]  (mmHg difference from anchor)
"""

import torch
import torch.nn as nn


# ──────────────────────────────────────────────
# Shared feature extractor
# ──────────────────────────────────────────────

class SEBlock1d(nn.Module):
    """Channel weighting block for 1D signals.

    This lets the CNN give more weight to useful channels and less weight to
    noisy ones without adding many parameters.
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, T)
        w = x.mean(dim=-1)         # squeeze: (B, C)
        w = self.fc(w).unsqueeze(-1)  # excite: (B, C, 1)
        return x * w


class SEBlock2d(nn.Module):
    """Squeeze-and-Excitation block for 2D feature maps."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, H, W)
        w = x.mean(dim=(-2, -1))   # squeeze: (B, C)
        w = self.fc(w).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return x * w


class TemporalAttention(nn.Module):
    """Multi-head self-attention over temporal positions.

    After the CNN extracts local features at each time step, this module lets
    the model compare positions across the full window. That is useful for
    pulse timing and shape features such as the systolic peak and dicrotic
    notch.
    """
    def __init__(self, channels, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # x: (B, C, T) -> (B, T, C) for attention
        x_t = x.permute(0, 2, 1)
        attn_out, _ = self.attn(x_t, x_t, x_t)
        x_t = self.norm(x_t + attn_out)  # residual connection
        return x_t.permute(0, 2, 1)      # back to (B, C, T)


class ConvBlock(nn.Module):
    """Conv1d -> GroupNorm -> ReLU -> SE -> optional MaxPool -> optional Dropout1d

    Includes Squeeze-and-Excitation (SE) channel attention to learn which
    feature channels are most important for BP estimation.
    """

    def __init__(self, in_ch, out_ch, kernel=7, stride=1, pool=True,
                 spatial_dropout=0.0, use_se=True):
        super().__init__()
        num_groups = min(8, out_ch)
        self.conv_gn_relu = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel,
                      stride=stride, padding=kernel // 2, bias=False),
            nn.GroupNorm(num_groups, out_ch),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock1d(out_ch) if use_se else nn.Identity()
        post = []
        if pool:
            post.append(nn.MaxPool1d(kernel_size=2, stride=2))
        if spatial_dropout > 0:
            post.append(nn.Dropout1d(spatial_dropout))
        self.post = nn.Sequential(*post) if post else nn.Identity()

    def forward(self, x):
        x = self.conv_gn_relu(x)
        x = self.se(x)
        return self.post(x)


class PPGEncoder(nn.Module):
    """
    CNN → Dual-pool encoder shared between Siamese branches.

    Input:  (B, C, W)   — waveform channels (PPG / VPG / APG [+ ECG])
    Output: (B, embed_dim)

    Architecture:

    Stage 1 — CNN: extracts local pulse morphology features (systolic peak
        amplitude/shape, dicrotic notch, diastolic decay) at multiple scales.
        Four ConvBlocks with progressive pooling to build hierarchical features.

    Stage 2 — Dual pooling: average pool captures mean morphology across the
        window; max pool captures peak activations (sharp features like the
        systolic peak and dicrotic notch). Concatenating both gives a richer
        fixed-size summary than either alone, without any learnable temporal
        weights that could overfit to patient-specific waveform sequences.

    Stage 3 — Normalise + project to embed_dim.

    Note on the removed BiLSTM+Attention version:
        The recurrent version fitted the training patients too closely in the
        early experiments. Dual pooling keeps the waveform summary simpler while
        still retaining the main morphology cues used for BP estimation.
    """

    def __init__(self, window_len: int = 625, embed_dim: int = 256,
                 dropout: float = 0.1, in_channels: int = 3):
        super().__init__()

        # Spatial dropout rate for CNN blocks — scales with regressor dropout
        # but lighter (×0.5). Drops entire channels to prevent filter co-adaptation.
        sdrop = dropout * 0.5

        # ── Stage 1: Local feature extraction ──────────────────────────
        # Channel sizes used in the final runs. Smaller versions underfit SBP.
        self.cnn = nn.Sequential(
            ConvBlock(in_channels, 32, kernel=7, pool=True,  spatial_dropout=0.0),     # (B, 32,  W/2) — no drop on first block
            ConvBlock(32,  64, kernel=5, pool=True,  spatial_dropout=sdrop),   # (B, 64,  W/4)
            ConvBlock(64, 128, kernel=3, pool=True,  spatial_dropout=sdrop),   # (B, 128, W/8)
            ConvBlock(128,128, kernel=3, pool=False, spatial_dropout=sdrop),   # (B, 128, W/8) — refine
        )

        # ── Stage 2: Temporal self-attention ──────────────────────────
        # Attend to pulse positions such as the systolic peak and dicrotic notch.
        self.temporal_attn = TemporalAttention(128, num_heads=4, dropout=dropout)

        # ── Stage 3: Dual pooling ───────────────────────────────────────
        self.pool_drop = nn.Dropout(dropout)

        # ── Stage 4: Normalise + project ───────────────────────────────
        # concat_dim = 128 * 2 = 256; project to embed_dim
        self.fc = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1: CNN — local morphology features
        x = self.cnn(x)                        # (B, 128, T)  T = W/8 ~ 100

        # Stage 2: Temporal attention — focus on BP-relevant positions
        x = self.temporal_attn(x)              # (B, 128, T)

        # Stage 3: Dual pooling
        x = self.pool_drop(x)
        avg = x.mean(dim=-1)                   # (B, 128) — mean morphology
        mx  = x.amax(dim=-1)                   # (B, 128) — peak activations
        x   = torch.cat([avg, mx], dim=1)      # (B, 256)

        # Stage 4: Normalise + project
        return self.fc(x)                       # (B, embed_dim)


# ──────────────────────────────────────────────
# 2D Spectrogram encoder (Schlesinger-style)
# ──────────────────────────────────────────────

class ConvBlock2d(nn.Module):
    """Conv2d -> GroupNorm -> ReLU -> SE -> MaxPool2d -> optional Dropout2d"""

    def __init__(self, in_ch, out_ch, kernel=3, pool=True, spatial_dropout=0.0,
                 use_se=True):
        super().__init__()
        num_groups = min(8, out_ch)
        self.conv_gn_relu = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel,
                      padding=kernel // 2, bias=False),
            nn.GroupNorm(num_groups, out_ch),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock2d(out_ch) if use_se else nn.Identity()
        post = []
        if pool:
            post.append(nn.MaxPool2d(kernel_size=2, stride=2))
        if spatial_dropout > 0:
            post.append(nn.Dropout2d(spatial_dropout))
        self.post = nn.Sequential(*post) if post else nn.Identity()

    def forward(self, x):
        x = self.conv_gn_relu(x)
        x = self.se(x)
        return self.post(x)


class SpectrogramEncoder(nn.Module):
    """
    2D CNN encoder for spectrogram input (Schlesinger et al. style).

    Input:  (B, C, F, T)  — multi-channel spectrogram (PPG/VPG/APG [+ ECG])
    Output: (B, embed_dim)

    Uses 2D convolutions to process the time-frequency representation,
    capturing both spectral and temporal patterns that are harder for
    1D CNNs to extract from raw waveforms.
    """

    def __init__(self, embed_dim: int = 256, dropout: float = 0.1,
                 in_channels: int = 3):
        super().__init__()
        sdrop = dropout * 0.5

        # 5-block CNN inspired by AlexNet (like Schlesinger)
        self.cnn = nn.Sequential(
            ConvBlock2d(in_channels, 32, kernel=5, pool=True,  spatial_dropout=0.0),     # → (32, F/2, T/2)
            ConvBlock2d(32,  64, kernel=3, pool=True,  spatial_dropout=sdrop),   # → (64, F/4, T/4)
            ConvBlock2d(64, 128, kernel=3, pool=True,  spatial_dropout=sdrop),   # → (128, F/8, T/8)
            ConvBlock2d(128,128, kernel=3, pool=False, spatial_dropout=sdrop),   # → (128, F/8, T/8) refine
            ConvBlock2d(128,128, kernel=3, pool=True,  spatial_dropout=sdrop),   # → (128, F/16, T/16)
        )

        # Adaptive pooling → fixed size regardless of spectrogram dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))  # → (128, 2, 2) = 512
        self.pool_drop = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)                                # (B, 128, F', T')
        x = self.adaptive_pool(x)                      # (B, 128, 2, 2)
        x = self.pool_drop(x)
        x = x.flatten(1)                               # (B, 512)
        return self.fc(x)                               # (B, embed_dim)


class GatedPairInteraction(nn.Module):
    """
    Learn a richer anchor-current comparison than plain diff/hadamard alone.

    The shared Siamese encoders stay unchanged; this block only improves how
    the two branch embeddings are fused before regression.
    """

    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        hidden = max(embed_dim // 2, 64)
        self.gate = nn.Sequential(
            nn.LayerNorm(embed_dim * 4),
            nn.Linear(embed_dim * 4, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Sigmoid(),
        )
        self.diff_gate = nn.Sequential(
            nn.LayerNorm(embed_dim * 4),
            nn.Linear(embed_dim * 4, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Tanh(),
        )
        self.mix_proj = nn.Sequential(
            nn.LayerNorm(embed_dim * 3),
            nn.Linear(embed_dim * 3, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, feat_anc: torch.Tensor, feat_cur: torch.Tensor) -> torch.Tensor:
        diff = feat_cur - feat_anc
        hadamard = feat_cur * feat_anc
        pair = torch.cat([feat_cur, feat_anc, diff, hadamard], dim=1)

        gate = self.gate(pair)
        diff_gate = self.diff_gate(pair)

        blended = gate * feat_cur + (1.0 - gate) * feat_anc
        emphasized_diff = diff_gate * diff
        return self.mix_proj(torch.cat([blended, emphasized_diff, hadamard], dim=1))


# ──────────────────────────────────────────────
# Siamese model
# ──────────────────────────────────────────────

class SiameseBPNet(nn.Module):
    """
    Full Siamese network.

    forward(anchor, current) → (pred_delta_sbp, pred_delta_dbp)

    Modes:
        - waveform:    1D CNN on raw PPG/VPG/APG signals
        - spectrogram: 2D CNN on STFT spectrograms
        - fusion:      BOTH encoders, concatenated embeddings

    To recover absolute BP:
        sbp_pred = anchor_sbp + pred_delta_sbp
        dbp_pred = anchor_dbp + pred_delta_dbp
    """

    def __init__(self, window_len: int = 625, embed_dim: int = 256, dropout: float = 0.3,
                 use_spectrograms: bool = False, use_fusion: bool = False,
                 use_pair_uncertainty: bool = False,
                 use_gated_pair_interaction: bool = False,
                 use_absolute_refinement: bool = False,
                 hand_feature_dim: int = 8,
                 input_channels: int = 3,
                 target_scale: float = 10.0):
        super().__init__()
        self.use_fusion = use_fusion
        self.use_pair_uncertainty = use_pair_uncertainty
        self.use_gated_pair_interaction = use_gated_pair_interaction
        self.use_absolute_refinement = use_absolute_refinement
        self.hand_feature_dim = int(hand_feature_dim)
        self.hand_feature_out_dim = self.hand_feature_dim
        self.input_channels = int(input_channels)
        self.target_scale = float(target_scale)

        enc_drop = dropout * 0.5

        if use_fusion:
            # Both encoders — each produces embed_dim features
            self.encoder_wave = PPGEncoder(window_len=window_len, embed_dim=embed_dim,
                                           dropout=enc_drop,
                                           in_channels=self.input_channels)
            self.encoder_spec = SpectrogramEncoder(embed_dim=embed_dim,
                                                    dropout=enc_drop,
                                                    in_channels=self.input_channels)
            # Fusion projection: 2*embed_dim → embed_dim (compress before Siamese diff)
            self.fusion_proj = nn.Sequential(
                nn.LayerNorm(embed_dim * 2),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(enc_drop),
            )
            fused_dim = embed_dim  # after projection
        elif use_spectrograms:
            self.encoder = SpectrogramEncoder(embed_dim=embed_dim, dropout=enc_drop,
                                              in_channels=self.input_channels)
            fused_dim = embed_dim
        else:
            self.encoder = PPGEncoder(window_len=window_len, embed_dim=embed_dim,
                                      dropout=enc_drop,
                                      in_channels=self.input_channels)
            fused_dim = embed_dim

        # Separate regression heads for SBP and DBP
        # Each head gets the full feature vector but learns independently,
        # so SBP gradients can't starve DBP learning (and vice versa).
        hand_dim = self.hand_feature_dim
        if hand_dim > 8:
            beat_embed_dim = min(16, hand_dim)
            self.hand_feature_proj = nn.Sequential(
                nn.LayerNorm(hand_dim),
                nn.Linear(hand_dim, beat_embed_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout * 0.25),
            )
            hand_dim = beat_embed_dim
            self.hand_feature_out_dim = beat_embed_dim
        else:
            self.hand_feature_proj = None
        if self.use_gated_pair_interaction:
            self.pair_interaction = GatedPairInteraction(fused_dim, dropout=dropout * 0.5)
        input_dim = fused_dim * (3 if self.use_gated_pair_interaction else 2) + 2 + hand_dim * 2
        # [diff, hadamard, optional gated pair ctx, bp_ctx, beat-aware hf_diff, hf_hadamard]
        hidden = min(192, fused_dim)

        self.shared_trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.sbp_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 1),
        )
        self.dbp_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        if self.use_absolute_refinement:
            self.absolute_head = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden // 2, 2),
            )
            self.absolute_gate = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden // 2, 2),
                nn.Sigmoid(),
            )
        if self.use_pair_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden // 2, 2),
            )

    @staticmethod
    def _decode_absolute_norm(raw_abs: torch.Tensor) -> torch.Tensor:
        """
        Map unconstrained head outputs to plausible BP ranges (normalised by /100).

        A bounded absolute branch is more stable than unconstrained regression
        when it is only used as a refinement signal.
        """
        sig = torch.sigmoid(raw_abs)
        sbp = 0.55 + 1.25 * sig[:, :1]  # 55–180 mmHg
        dbp = 0.30 + 0.80 * sig[:, 1:]  # 30–110 mmHg
        return torch.cat([sbp, dbp], dim=1)

    def _encode(self, wave, spec):
        """Encode one branch (anchor or current)."""
        if self.use_fusion:
            f_wave = self.encoder_wave(wave)    # (B, embed_dim)
            f_spec = self.encoder_spec(spec)    # (B, embed_dim)
            fused  = torch.cat([f_wave, f_spec], dim=1)  # (B, 2*embed_dim)
            return self.fusion_proj(fused)      # (B, embed_dim)
        else:
            return self.encoder(wave)           # wave is either waveform or spectrogram

    def forward(self, anchor, current,
                anchor_sbp: torch.Tensor, anchor_dbp: torch.Tensor,
                anchor_spec=None, current_spec=None,
                anchor_hf=None, current_hf=None,
                return_uncertainty: bool = False,
                return_auxiliary: bool = False):
        """
        anchor/current:      (B, 3, W) waveforms or (B, 3, F, T) spectrograms
        anchor/current_spec: (B, 3, F, T) spectrograms (only used in fusion mode)
        anchor/current_hf:   (B, F_hand) handcrafted + optional beat-aware PPG features
        anchor_sbp/dbp:      (B,) normalised by /100
        returns:             (B, 2) — [delta_sbp, delta_dbp]
        """
        feat_anc = self._encode(anchor, anchor_spec)
        feat_cur = self._encode(current, current_spec)

        diff     = feat_cur - feat_anc
        hadamard = feat_cur * feat_anc

        bp_ctx   = torch.stack([anchor_sbp, anchor_dbp], dim=1)

        parts = [diff, hadamard, bp_ctx]
        if self.use_gated_pair_interaction:
            parts.insert(2, self.pair_interaction(feat_anc, feat_cur))

        # Handcrafted features: diff and element-wise product (like CNN features)
        if anchor_hf is not None and current_hf is not None:
            if self.hand_feature_proj is not None:
                anchor_hf = self.hand_feature_proj(anchor_hf)
                current_hf = self.hand_feature_proj(current_hf)
            hf_diff = current_hf - anchor_hf
            hf_had  = current_hf * anchor_hf
            parts.extend([hf_diff, hf_had])
        else:
            # Pad with zeros if not provided (backward compat)
            B = anchor.shape[0]
            zeros = torch.zeros(B, self.hand_feature_out_dim, device=anchor.device)
            parts.extend([zeros, zeros])

        combined = torch.cat(parts, dim=1)

        trunk = self.shared_trunk(combined)          # (B, hidden)
        d_sbp = self.sbp_head(trunk)                 # (B, 1)
        d_dbp = self.dbp_head(trunk)                 # (B, 1)
        delta_raw = torch.cat([d_sbp, d_dbp], dim=1)    # (B, 2)

        absolute_norm = None
        delta_from_abs = None
        refine_gate = None
        deltas = delta_raw
        if self.use_absolute_refinement:
            absolute_norm = self._decode_absolute_norm(self.absolute_head(trunk))
            delta_from_abs = (absolute_norm - bp_ctx) * (100.0 / self.target_scale)
            refine_gate = self.absolute_gate(trunk)
            deltas = delta_raw + refine_gate * (delta_from_abs - delta_raw)

        log_var = None
        if self.use_pair_uncertainty:
            log_var = torch.clamp(self.uncertainty_head(trunk), min=-4.0, max=4.0)

        if return_uncertainty or return_auxiliary:
            extras = {
                "log_var": log_var,
                "delta_raw": delta_raw,
                "absolute_norm": absolute_norm,
                "delta_from_abs": delta_from_abs,
                "refine_gate": refine_gate,
            }
            return deltas, extras

        return deltas


# ──────────────────────────────────────────────
# Model factory
# ──────────────────────────────────────────────

def build_model(window_len: int = 625, embed_dim: int = 256,
                dropout: float = 0.3, device: str = "cpu",
                use_spectrograms: bool = False,
                use_fusion: bool = False,
                use_pair_uncertainty: bool = False,
                use_gated_pair_interaction: bool = False,
                use_absolute_refinement: bool = False,
                hand_feature_dim: int = 8,
                input_channels: int = 3,
                target_scale: float = 10.0) -> SiameseBPNet:
    model = SiameseBPNet(window_len=window_len, embed_dim=embed_dim, dropout=dropout,
                         use_spectrograms=use_spectrograms, use_fusion=use_fusion,
                         use_pair_uncertainty=use_pair_uncertainty,
                         use_gated_pair_interaction=use_gated_pair_interaction,
                         use_absolute_refinement=use_absolute_refinement,
                         hand_feature_dim=hand_feature_dim,
                         input_channels=input_channels,
                         target_scale=target_scale)
    model = model.to(device)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SiameseBPNet  |  params: {total:,}  |  device: {device}")
    return model


if __name__ == "__main__":
    # Quick sanity check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = build_model(window_len=625, device=device)

    B, C, W = 8, 3, 625
    anchor      = torch.randn(B, C, W).to(device)
    current     = torch.randn(B, C, W).to(device)
    anc_sbp     = torch.randn(B).to(device)
    anc_dbp     = torch.randn(B).to(device)

    out = model(anchor, current, anc_sbp, anc_dbp)
    print(f"Output shape: {out.shape}")   # expect (8, 2)
    print("Sanity check passed")
