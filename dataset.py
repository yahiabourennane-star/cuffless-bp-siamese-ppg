"""
dataset.py
----------
Siamese dataset for cuff-less BP estimation from PPG signals.

Expected files in data_dir:
    X_ppg_windows.npy  – (N, W)  float32   raw PPG windows
    X_vpg.npy          – (N, W)  float32   velocity PPG (1st derivative)
    X_apg.npy          – (N, W)  float32   acceleration PPG (2nd derivative)
    y_sbp.npy          – (N,)    float32   SBP labels [mmHg]
    y_dbp.npy          – (N,)    float32   DBP labels [mmHg]
    patient_ids.npy    – (N,)    int64     patient index per window
    train_mask.npy     – (N,)    bool
    val_mask.npy       – (N,)    bool
    test_mask.npy      – (N,)    bool

Dataset note:
    Patients with very low BP variability (SBP std < min_sbp_std) are excluded.
    Otherwise, predicting almost no BP change gives a misleadingly strong
    baseline and the Siamese task becomes less useful.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import stft as scipy_stft, find_peaks, peak_prominences, peak_widths


# ──────────────────────────────────────────────
# Signal normalisation
# ──────────────────────────────────────────────

def normalise_windows(x: np.ndarray) -> np.ndarray:
    """
    Per-window z-score normalisation.
    Input: (N, W) → output: (N, W) float32.
    Note: signals are already pre-normalised in preprocessing; this keeps the
    channel scales consistent inside the model.
    """
    mu  = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1,  keepdims=True) + 1e-8
    return ((x - mu) / std).astype(np.float32)


def normalise_windows_global_scale(x: np.ndarray, global_std: float | None = None) -> np.ndarray:
    """
    Preserve inter-window amplitude variation while keeping values numerically
    stable for the CNN.

    Each window is mean-centered, but all windows in the channel share one
    global scale factor instead of using their own per-window std.
    """
    mu = x.mean(axis=-1, keepdims=True)
    if global_std is None:
        global_std = float(np.std(x)) + 1e-8
    return ((x - mu) / global_std).astype(np.float32)


def compute_spectrogram(signal_1d: np.ndarray, fs: int = 125,
                        nperseg: int = 64, noverlap: int = 56) -> np.ndarray:
    """
    Compute log-power spectrogram for a single 1D signal.

    Input:  (W,) float32  — one channel of one window
    Output: (F, T) float32 — log-magnitude spectrogram

    Parameters match Schlesinger et al.'s approach adapted for short windows:
        nperseg=64 (0.51s) with 87.5% overlap → captures ~1 heartbeat per frame
        Gives ~33 freq bins × ~93 time frames for W=800
    """
    _, _, Zxx = scipy_stft(signal_1d, fs=fs, nperseg=nperseg,
                           noverlap=noverlap, window='hamming')
    # Log-magnitude (add small epsilon to avoid log(0))
    spec = np.log1p(np.abs(Zxx)).astype(np.float32)
    return spec


def signals_to_spectrograms(X: np.ndarray, fs: int = 125,
                             nperseg: int = 64, noverlap: int = 56,
                             cache_path: str = None,
                             batch_size: int = 2048) -> np.ndarray:
    """
    Convert (N, 3, W) waveforms → (N, 3, F, T) spectrograms.
    Each channel is independently converted to a log-power spectrogram.
    """
    N, C, W = X.shape
    # Compute one to get output shape
    sample = compute_spectrogram(X[0, 0], fs, nperseg, noverlap)
    F, T = sample.shape
    print(f"  Spectrogram shape per channel: ({F}, {T})")

    cache_dtype = np.float16 if cache_path is not None else np.float32
    total_gb = (N * C * F * T * np.dtype(cache_dtype).itemsize) / 1e9

    if cache_path is None:
        specs = np.zeros((N, C, F, T), dtype=np.float32)
        for i in range(N):
            for c in range(C):
                specs[i, c] = compute_spectrogram(X[i, c], fs, nperseg, noverlap)
            if (i + 1) % 50000 == 0:
                print(f"  Computed {i+1:,}/{N:,} spectrograms...")
        return specs

    if os.path.exists(cache_path):
        os.remove(cache_path)

    print(
        f"  Streaming spectrogram cache to disk (~{total_gb:.1f} GB, {np.dtype(cache_dtype).name}) "
        f"in batches of {batch_size:,} windows..."
    )
    specs = np.lib.format.open_memmap(
        cache_path,
        mode="w+",
        dtype=cache_dtype,
        shape=(N, C, F, T),
    )

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        for i in range(start, end):
            for c in range(C):
                specs[i, c] = compute_spectrogram(X[i, c], fs, nperseg, noverlap)
        specs.flush()
        if end % 50000 == 0 or end == N:
            print(f"  Computed {end:,}/{N:,} spectrograms...")

    specs.flush()
    mmap_obj = getattr(specs, "_mmap", None)
    if mmap_obj is not None:
        mmap_obj.close()
    del specs
    return np.load(cache_path, mmap_mode="r", allow_pickle=False)


def stack_signals(ppg, vpg, apg, preserve_ppg_amplitude: bool = True,
                  ppg_global_std: float | None = None) -> np.ndarray:
    """Stack three (N, W) arrays → (N, 3, W) float32, each normalised."""
    ppg_norm = (
        normalise_windows_global_scale(ppg, global_std=ppg_global_std)
        if preserve_ppg_amplitude
        else normalise_windows(ppg)
    )
    return np.stack([
        ppg_norm,
        normalise_windows(vpg),
        normalise_windows(apg),
    ], axis=1)


# ──────────────────────────────────────────────
# Handcrafted PPG features (time-invariant)
# ──────────────────────────────────────────────

N_HAND_FEATURES = 8   # number of base handcrafted features per window
N_BEAT_FEATURES = 10  # number of beat-aware summary features per window
N_HYBRID_BEAT_FEATURES = 6  # number of ECG+PPG hybrid beat features per window


def make_derived_cache_path(data_dir, stem, fs, window_len, extra=None):
    """Version derived caches by fs/window length so stale caches are not reused."""
    parts = [stem, f"fs{fs}", f"w{window_len}"]
    if extra:
        parts.append(extra)
    return os.path.join(data_dir, "_".join(parts) + ".npy")


def compute_global_std_from_indices(x: np.ndarray, indices, batch_size: int = 8192) -> float:
    """Compute a global std using only the selected window indices."""
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        raise ValueError("Need at least one training index to compute global std.")

    total_count = 0
    total_sum = 0.0
    total_sumsq = 0.0
    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        chunk = np.asarray(x[indices[start:end]], dtype=np.float64)
        total_count += chunk.size
        total_sum += float(chunk.sum())
        total_sumsq += float(np.square(chunk).sum())

    mean = total_sum / max(total_count, 1)
    var = max(total_sumsq / max(total_count, 1) - mean * mean, 0.0)
    return float(np.sqrt(var)) + 1e-8


def standardize_features_from_train_stats(features: np.ndarray, train_mask: np.ndarray,
                                          eps: float = 1e-8) -> np.ndarray:
    """Z-score features using training windows only, then apply to all windows."""
    train_mask = np.asarray(train_mask, dtype=bool)
    if len(features) != len(train_mask):
        raise ValueError(
            f"Feature matrix rows {len(features)} do not match train_mask length {len(train_mask)}"
        )
    if not np.any(train_mask):
        raise ValueError("Training mask is empty; cannot fit feature standardization stats.")

    feats = np.asarray(features, dtype=np.float32)
    train_feats = feats[train_mask]
    mu = train_feats.mean(axis=0, keepdims=True)
    std = train_feats.std(axis=0, keepdims=True) + float(eps)
    return ((feats - mu) / std).astype(np.float32)


# ──────────────────────────────────────────────
# Signal quality assessment
# ──────────────────────────────────────────────

def compute_signal_quality(ppg_window: np.ndarray, fs: int = 125) -> float:
    """Score a PPG window's quality from 0 (garbage) to 1 (clean).

    Checks peak detection, clipping, flatness, HR range, interval regularity.
    Returns float in [0, 1].
    """
    x = ppg_window.astype(np.float32)
    score = 1.0

    # Flat signal
    std = x.std()
    if std < 1e-6:
        return 0.0

    # Clipping: >5% of samples at extremes
    x_min, x_max = x.min(), x.max()
    span = x_max - x_min
    if span < 1e-6:
        return 0.0
    near_min = (np.abs(x - x_min) < span * 0.01).mean()
    near_max = (np.abs(x - x_max) < span * 0.01).mean()
    if near_min > 0.05 or near_max > 0.05:
        score -= 0.4

    # Peak detection
    min_dist = int(0.3 * fs)
    peaks, _ = find_peaks(x, distance=min_dist, height=np.percentile(x, 40))
    if len(peaks) < 2:
        return max(0.0, score - 0.6)

    # HR range (30-200 bpm)
    intervals = np.diff(peaks) / fs
    hr = 60.0 / (np.mean(intervals) + 1e-8)
    if hr < 30 or hr > 200:
        score -= 0.3

    # Interval regularity
    if len(intervals) > 2:
        cv = np.std(intervals) / (np.mean(intervals) + 1e-8)
        if cv > 0.5:
            score -= 0.3

    return max(0.0, min(1.0, score))


def compute_quality_scores(X_ppg: np.ndarray, fs: int = 125) -> np.ndarray:
    """Compute quality scores for all windows. Returns (N,) float32."""
    N = len(X_ppg)
    scores = np.zeros(N, dtype=np.float32)
    for i in range(N):
        scores[i] = compute_signal_quality(X_ppg[i], fs)
        if (i + 1) % 50000 == 0:
            print(f"  Quality scored {i+1:,}/{N:,}...")
    return scores


def extract_ppg_features(ppg_window: np.ndarray, fs: int = 125) -> np.ndarray:
    """Extract physiological features from a single PPG window (1D, length W).

    These features capture SBP-correlated morphology that is more stable
    across time than raw waveform shape, helping reduce temporal drift.

    Returns (N_HAND_FEATURES,) float32 array.
    """
    x = ppg_window.astype(np.float32)
    W = len(x)
    feats = np.zeros(N_HAND_FEATURES, dtype=np.float32)

    # Find systolic peaks
    min_dist = int(0.4 * fs)  # min 0.4s between peaks (~150 bpm max)
    peaks, props = find_peaks(x, distance=min_dist, height=np.percentile(x, 50))

    if len(peaks) < 2:
        return feats  # return zeros if can't detect peaks

    # 1. Mean pulse rate (beats per minute)
    intervals = np.diff(peaks) / fs  # seconds between peaks
    mean_hr = 60.0 / (np.mean(intervals) + 1e-8)
    feats[0] = mean_hr / 100.0  # normalise to ~0.5-2.0 range

    # 2. Heart rate variability (std of intervals)
    feats[1] = np.std(intervals) * 10.0  # scale up for visibility

    # 3. Mean systolic peak amplitude
    peak_heights = x[peaks]
    feats[2] = np.mean(peak_heights)

    # 4. Systolic peak amplitude variability
    feats[3] = np.std(peak_heights)

    # 5. Mean pulse width at half maximum
    half_heights = []
    for pk in peaks:
        threshold = x[pk] * 0.5
        # Search left for half-height crossing
        left = pk
        while left > 0 and x[left] > threshold:
            left -= 1
        # Search right
        right = pk
        while right < W - 1 and x[right] > threshold:
            right += 1
        half_heights.append((right - left) / fs)  # width in seconds
    feats[4] = np.mean(half_heights) * 5.0  # scale

    # 6. Augmentation index (dicrotic notch detection)
    # Ratio of second peak to first peak amplitude in each beat
    aug_indices = []
    for i in range(len(peaks) - 1):
        segment = x[peaks[i]:peaks[i+1]]
        if len(segment) > 10:
            # Find dicrotic notch (minimum after systolic peak)
            notch_region = segment[len(segment)//4:]  # skip initial descent
            if len(notch_region) > 5:
                local_peaks, _ = find_peaks(notch_region, distance=5)
                if len(local_peaks) > 0:
                    diastolic_peak = notch_region[local_peaks[0]]
                    aug_idx = diastolic_peak / (segment[0] + 1e-8)
                    aug_indices.append(aug_idx)
    feats[5] = np.mean(aug_indices) if aug_indices else 0.5

    # 6. Diastolic decay slope (mean downslope after systolic peak)
    slopes = []
    for pk in peaks:
        end = min(pk + int(0.2 * fs), W)  # 200ms after peak
        if end > pk + 2:
            slope = (x[end-1] - x[pk]) / ((end - pk) / fs + 1e-8)
            slopes.append(slope)
    feats[6] = np.mean(slopes) if slopes else 0.0

    # 7. Signal energy (RMS)
    feats[7] = np.sqrt(np.mean(x**2))

    return feats


def extract_all_features(X_ppg: np.ndarray, fs: int = 125) -> np.ndarray:
    """Extract handcrafted features for all windows.

    X_ppg: (N, W) raw PPG windows (NOT normalised — use raw PPG).
    Returns: (N, N_HAND_FEATURES) float32.
    """
    N = len(X_ppg)
    feats = np.zeros((N, N_HAND_FEATURES), dtype=np.float32)
    for i in range(N):
        feats[i] = extract_ppg_features(X_ppg[i], fs)
        if (i + 1) % 50000 == 0:
            print(f"  Extracted features {i+1:,}/{N:,}...")

    return feats


def extract_beat_features(ppg_window: np.ndarray, fs: int = 125) -> np.ndarray:
    """Extract beat-aware summary features from one raw PPG window."""
    x = ppg_window.astype(np.float32)
    feats = np.zeros(N_BEAT_FEATURES, dtype=np.float32)

    min_dist = int(0.4 * fs)
    min_height = float(np.percentile(x, 50))
    prominence_floor = max(float(np.std(x)) * 0.08, 1e-4)
    peaks, _ = find_peaks(x, distance=min_dist, height=min_height, prominence=prominence_floor)
    if len(peaks) < 3:
        return feats

    duration_sec = max(len(x) / float(fs), 1e-6)
    intervals = np.diff(peaks).astype(np.float32) / float(fs)
    prominences = peak_prominences(x, peaks)[0].astype(np.float32)
    widths = peak_widths(x, peaks, rel_height=0.5)[0].astype(np.float32) / float(fs)

    rise_times = []
    decay_times = []
    amplitudes = []
    resampled_beats = []
    interp_t = np.linspace(0.0, 1.0, 32, dtype=np.float32)

    for i in range(1, len(peaks) - 1):
        pk = int(peaks[i])
        left = int(peaks[i - 1])
        right = int(peaks[i + 1])
        if right <= left + 4:
            continue

        pre_seg = x[left:pk + 1]
        post_seg = x[pk:right + 1]
        if len(pre_seg) < 2 or len(post_seg) < 2:
            continue

        left_min = left + int(np.argmin(pre_seg))
        right_min = pk + int(np.argmin(post_seg))
        if right_min <= left_min + 4:
            continue

        rise_times.append((pk - left_min) / float(fs))
        decay_times.append((right_min - pk) / float(fs))
        amplitudes.append(float(x[pk] - x[left_min]))

        beat = x[left_min:right_min + 1]
        src_t = np.linspace(0.0, 1.0, len(beat), dtype=np.float32)
        resampled_beats.append(np.interp(interp_t, src_t, beat).astype(np.float32))

    if len(rise_times) == 0:
        return feats

    feats[0] = len(peaks) / duration_sec / 2.0
    feats[1] = float(np.mean(intervals))
    feats[2] = float(np.std(intervals))
    feats[3] = float(np.mean(prominences))
    feats[4] = float(np.std(prominences))
    feats[5] = float(np.mean(widths))
    feats[6] = float(np.std(widths))

    rise_times = np.asarray(rise_times, dtype=np.float32)
    decay_times = np.asarray(decay_times, dtype=np.float32)
    total_times = rise_times + decay_times + 1e-6
    amplitudes = np.asarray(amplitudes, dtype=np.float32)
    feats[7] = float(np.mean(rise_times / total_times))
    feats[8] = float(np.mean(amplitudes))

    if len(resampled_beats) >= 2:
        beats = np.stack(resampled_beats, axis=0)
        template = beats.mean(axis=0)
        template_centered = template - template.mean()
        template_norm = np.linalg.norm(template_centered) + 1e-6
        corr_scores = []
        for beat in beats:
            beat_centered = beat - beat.mean()
            denom = (np.linalg.norm(beat_centered) * template_norm) + 1e-6
            corr_scores.append(float(np.dot(beat_centered, template_centered) / denom))
        feats[9] = float(np.mean(corr_scores))
    else:
        feats[9] = 0.0

    return feats


def extract_all_beat_features(X_ppg: np.ndarray, fs: int = 125) -> np.ndarray:
    """Extract beat-aware summary features for all raw PPG windows."""
    N = len(X_ppg)
    feats = np.zeros((N, N_BEAT_FEATURES), dtype=np.float32)
    for i in range(N):
        feats[i] = extract_beat_features(X_ppg[i], fs)
        if (i + 1) % 50000 == 0:
            print(f"  Extracted beat features {i+1:,}/{N:,}...")

    return feats


# ──────────────────────────────────────────────
# Patient filtering
# ──────────────────────────────────────────────

def get_valid_patients(y_sbp, patient_ids, min_sbp_std=5.0):
    """
    Return set of patient IDs whose SBP has sufficient variability.

    Patients with flat BP (std < min_sbp_std) are excluded because:
    - Predicting delta=0 already works well for these patients
    - They add very little signal for the Siamese delta task
    - Schlesinger et al. used a similar quality filtering step

    Default threshold: 5 mmHg (removes ~25/162 patients with flat BP)
    """
    valid = set()
    for pid in np.unique(patient_ids):
        idx = np.where(patient_ids == pid)[0]
        if y_sbp[idx].std() >= min_sbp_std:
            valid.add(int(pid))
    return valid


# ──────────────────────────────────────────────
# Anchor selection
# ──────────────────────────────────────────────

def make_within_patient_masks(patient_ids, valid_patients,
                              train_frac=0.70, val_frac=0.15,
                              shuffle_eval=True, seed=42):
    """
    Within-patient temporal split (70 / 15 / 15 by default).

    Train: first train_frac of each patient's windows (chronological).
    Val/Test: the remaining windows are SHUFFLED and split 50/50.

    shuffle_eval=True (default, NEW in v13):
        The last 30% of windows are randomly split into val/test.
        This removes the systematic temporal bias where test windows
        are always later than val windows, closing the val-to-test gap.

    shuffle_eval=False (legacy):
        Val = next val_frac, Test = remainder (strictly chronological).
    """
    N          = len(patient_ids)
    train_mask = np.zeros(N, dtype=bool)
    val_mask   = np.zeros(N, dtype=bool)
    test_mask  = np.zeros(N, dtype=bool)
    rng        = np.random.default_rng(seed)

    for pid in np.unique(patient_ids):
        if int(pid) not in valid_patients:
            continue
        idx     = np.where(patient_ids == pid)[0]   # temporal order assumed
        n       = len(idx)
        n_train = max(1, int(n * train_frac))

        train_mask[idx[:n_train]] = True

        eval_idx = idx[n_train:]  # remaining windows
        if len(eval_idx) == 0:
            continue

        if shuffle_eval:
            # Shuffle the eval windows, then split 50/50
            shuffled = eval_idx.copy()
            rng.shuffle(shuffled)
            n_val = max(1, len(shuffled) // 2)
            val_mask[shuffled[:n_val]]  = True
            test_mask[shuffled[n_val:]] = True
        else:
            n_val = max(1, int(n * val_frac))
            val_mask[eval_idx[:n_val]]  = True
            test_mask[eval_idx[n_val:]] = True

    return train_mask, val_mask, test_mask


def select_anchors(y_sbp, y_dbp, patient_ids, valid_patients,
                   random_anchor=False, seed=42, train_mask=None,
                   num_anchors=1):
    """
    Pick calibration (anchor) windows per patient.

    num_anchors=1 (default): returns {pid: int} — single best anchor
    num_anchors=K (K>1): returns {pid: list[int]} — top-K near-median anchors

    Multi-anchor mode picks the K windows closest to the patient's joint
    SBP+DBP median. During training, each window randomly pairs with one
    of these K anchors — providing target diversity to prevent memorisation
    while keeping targets stable (all anchors are near-median, so deltas
    stay similar across the K choices).

    Returns: {patient_id: int_or_list}
    """
    rng = np.random.default_rng(seed)
    anchors = {}

    for pid in np.unique(patient_ids):
        if int(pid) not in valid_patients:
            continue
        idx = np.where(patient_ids == pid)[0]
        # Restrict anchor candidates to training windows only.
        if train_mask is not None:
            train_idx = idx[train_mask[idx]]
            if len(train_idx) > 0:
                idx = train_idx
        if random_anchor:
            anchors[int(pid)] = int(rng.choice(idx))
        else:
            sbp_vals   = y_sbp[idx]
            dbp_vals   = y_dbp[idx]
            median_sbp = np.median(sbp_vals)
            median_dbp = np.median(dbp_vals)
            sbp_std    = sbp_vals.std() + 1e-8
            dbp_std    = dbp_vals.std() + 1e-8
            joint_dist = (np.abs(sbp_vals - median_sbp) / sbp_std +
                          np.abs(dbp_vals - median_dbp) / dbp_std)

            if num_anchors <= 1:
                best_local = int(np.argmin(joint_dist))
                anchors[int(pid)] = int(idx[best_local])
            else:
                # Top-K closest to median
                K = min(num_anchors, len(idx))
                top_k_local = np.argsort(joint_dist)[:K]
                anchors[int(pid)] = [int(idx[j]) for j in top_k_local]

    return anchors


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class SiameseBPDataset(Dataset):
    """
    Returns (anchor, current, delta_sbp, delta_dbp, anc_sbp_norm, anc_dbp_norm).

    anchor / current  : (3, W) float32 tensors
    delta_sbp / dbp   : scalar float32 [mmHg] — current minus anchor
    anc_sbp/dbp_norm  : anchor BP / 100  (unit-scale input to regressor)

    randomize_anchor=False (training + val + test)
        → uses the fixed median-SBP anchor from `anchors` dict.
          Stable delta targets allow the model to converge on SBP features.

    augment=True (training only)
        → adds independent Gaussian noise to anchor and current signals
          each call. Prevents exact waveform memorisation while keeping
          BP-predictive morphology features intact.

    Absolute BP at inference:
        sbp_pred = known_anchor_sbp + delta_sbp_pred
        dbp_pred = known_anchor_dbp + delta_dbp_pred
    """

    def __init__(self, X, y_sbp, y_dbp, patient_ids, mask,
                 anchors, valid_patients, randomize_anchor=False,
                 augment=False, X_spec=None, target_scale=1.0,
                 hand_features=None, anchor_selection="random_list",
                 fixed_pairs=None):
        self.X                = X
        self.num_channels     = int(X.shape[1])
        self.X_spec           = X_spec    # (N, 3, F, T) or None
        self.hand_features    = hand_features  # (N, F_hand) or None
        self.hand_feature_dim = int(hand_features.shape[1]) if hand_features is not None else N_HAND_FEATURES
        self.target_scale     = target_scale   # divide deltas by this
        self.y_sbp            = y_sbp
        self.y_dbp            = y_dbp
        self.patient_ids      = patient_ids
        self.anchors          = anchors
        self.randomize_anchor = randomize_anchor
        self.augment          = augment
        self.anchor_selection = anchor_selection
        self.fixed_pairs      = None if fixed_pairs is None else np.asarray(fixed_pairs, dtype=np.int64)
        self.anchor_lists     = {
            int(pid): ([int(a) for a in anc] if isinstance(anc, list) else [int(anc)])
            for pid, anc in anchors.items()
        }

        # Only keep windows from valid (high-variability) patients
        base_idx = np.where(mask)[0]
        self.indices = np.array([
            i for i in base_idx
            if int(patient_ids[i]) in valid_patients
        ])

        self.refresh_index_cache()

    def refresh_index_cache(self):
        """Rebuild per-patient lookup tables after dataset index changes."""
        # Per-patient index lookup for random anchor selection
        self.pid_to_indices = {}
        for global_i in self.indices:
            pid = int(self.patient_ids[global_i])
            self.pid_to_indices.setdefault(pid, []).append(global_i)

    def __len__(self):
        if self.fixed_pairs is not None:
            return len(self.fixed_pairs)
        return len(self.indices)

    @staticmethod
    def _augment_signal(x: np.ndarray, noise_std: float = 0.05,
                        apply_masking: bool = True) -> np.ndarray:
        """
        Augment signal with Gaussian noise + amplitude jitter + optional time masking.

        - Gaussian noise (std=0.05): prevents exact waveform memorisation.
        - Amplitude jitter (±5%): tolerates inter-window amplitude drift.
        - Time masking (current only, 20% chance, ≤W//32 samples):
            Sub-beat masking (~0.2s max at 125 Hz) — preserves systolic peak,
            dicrotic notch, and VPG/APG derivative structure. NOT applied to
            the anchor branch since it is the stable calibration reference.
        """
        # Gaussian noise
        x = x + np.random.normal(0.0, noise_std, x.shape).astype(np.float32)
        # Per-channel amplitude jitter ±5%
        # Shape-agnostic: works for (C, W) waveforms AND (C, F, T) spectrograms
        scale_shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (C,1) or (C,1,1)
        scale = np.random.uniform(0.95, 1.05, size=scale_shape).astype(np.float32)
        x = x * scale
        # Time masking: current branch only, short segment, low probability
        # For 1D: masks along last axis (time). For 2D spec: masks time columns.
        if apply_masking and np.random.rand() < 0.20:
            T = x.shape[-1]  # last axis is always time
            mask_len = np.random.randint(1, max(2, T // 32))
            mask_start = np.random.randint(0, T - mask_len)
            x = x.copy()
            x[..., mask_start:mask_start + mask_len] = 0.0
        return x

    def __getitem__(self, i):
        if self.fixed_pairs is not None:
            idx, anc_i = [int(v) for v in self.fixed_pairs[i]]
            pid = int(self.patient_ids[idx])
        else:
            idx = self.indices[i]
            pid = int(self.patient_ids[idx])

            if self.randomize_anchor:
                candidates = self.pid_to_indices[pid]
                if len(candidates) > 1:
                    candidates = [j for j in candidates if j != idx]
                anc_i = int(np.random.choice(candidates))
            else:
                anc_entry = self.anchor_lists[pid]
                if len(anc_entry) > 1 and self.anchor_selection == "random_list":
                    anc_i = int(np.random.choice(anc_entry))
                else:
                    anc_i = int(anc_entry[0])

        anc_sig = self.X[anc_i]   # (3, W) already z-scored
        cur_sig = self.X[idx]

        if self.augment:
            anc_sig = self._augment_signal(anc_sig, apply_masking=False)
            cur_sig = self._augment_signal(cur_sig, apply_masking=True)

        anchor  = torch.tensor(anc_sig, dtype=torch.float32)
        current = torch.tensor(cur_sig, dtype=torch.float32)

        delta_sbp = torch.tensor(
            (float(self.y_sbp[idx]) - float(self.y_sbp[anc_i])) / self.target_scale,
            dtype=torch.float32
        )
        delta_dbp = torch.tensor(
            (float(self.y_dbp[idx]) - float(self.y_dbp[anc_i])) / self.target_scale,
            dtype=torch.float32
        )

        anc_sbp_norm = torch.tensor(float(self.y_sbp[anc_i]) / 100.0, dtype=torch.float32)
        anc_dbp_norm = torch.tensor(float(self.y_dbp[anc_i]) / 100.0, dtype=torch.float32)

        # Handcrafted features (always included if available)
        if self.hand_features is not None:
            anc_hf = torch.tensor(self.hand_features[anc_i], dtype=torch.float32)
            cur_hf = torch.tensor(self.hand_features[idx],   dtype=torch.float32)
        else:
            anc_hf = torch.zeros(self.hand_feature_dim, dtype=torch.float32)
            cur_hf = torch.zeros(self.hand_feature_dim, dtype=torch.float32)

        # Fusion mode: return spectrograms alongside waveforms
        if self.X_spec is not None:
            anc_spec = torch.tensor(self.X_spec[anc_i], dtype=torch.float32)
            cur_spec = torch.tensor(self.X_spec[idx],   dtype=torch.float32)
            if self.augment:
                anc_spec_np = self._augment_signal(self.X_spec[anc_i], apply_masking=False)
                cur_spec_np = self._augment_signal(self.X_spec[idx],   apply_masking=False)
                anc_spec = torch.tensor(anc_spec_np, dtype=torch.float32)
                cur_spec = torch.tensor(cur_spec_np, dtype=torch.float32)
            return anchor, current, delta_sbp, delta_dbp, anc_sbp_norm, anc_dbp_norm, anc_spec, cur_spec, anc_hf, cur_hf

        return anchor, current, delta_sbp, delta_dbp, anc_sbp_norm, anc_dbp_norm, anc_hf, cur_hf


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────

def extract_ecg_ppg_hybrid_features(ecg_window: np.ndarray,
                                    ppg_window: np.ndarray,
                                    fs: int = 125) -> np.ndarray:
    """Extract compact ECG+PPG beat timing features from one window."""
    ecg = ecg_window.astype(np.float32)
    ppg = ppg_window.astype(np.float32)
    feats = np.zeros(N_HYBRID_BEAT_FEATURES, dtype=np.float32)

    if len(ecg) != len(ppg):
        return feats

    ecg_centered = ecg - np.median(ecg)
    ecg_energy = np.abs(ecg_centered)
    ecg_prom = max(float(np.std(ecg_energy)) * 0.25, 1e-4)
    ecg_height = float(np.percentile(ecg_energy, 75))
    r_peaks, _ = find_peaks(
        ecg_energy,
        distance=max(int(0.25 * fs), 1),
        height=ecg_height,
        prominence=ecg_prom,
    )
    if len(r_peaks) < 3:
        return feats

    ppg_prom = max(float(np.std(ppg)) * 0.08, 1e-4)
    ppg_height = float(np.percentile(ppg, 50))
    ppg_peaks, _ = find_peaks(
        ppg,
        distance=max(int(0.4 * fs), 1),
        height=ppg_height,
        prominence=ppg_prom,
    )
    if len(ppg_peaks) < 3:
        return feats

    duration_sec = max(len(ecg) / float(fs), 1e-6)
    rr = np.diff(r_peaks).astype(np.float32) / float(fs)
    feats[0] = len(r_peaks) / duration_sec / 2.0
    feats[1] = float(np.mean(rr))
    feats[2] = float(np.std(rr))

    pat_delays = []
    max_delay = int(0.45 * fs)
    min_delay = int(0.08 * fs)
    for r_peak in r_peaks[:-1]:
        after = ppg_peaks[(ppg_peaks > r_peak + min_delay) & (ppg_peaks < r_peak + max_delay)]
        if len(after) == 0:
            continue
        pat_delays.append(float(after[0] - r_peak) / float(fs))

    if pat_delays:
        pat_delays = np.asarray(pat_delays, dtype=np.float32)
        feats[3] = float(np.mean(pat_delays))
        feats[4] = float(np.std(pat_delays))
        feats[5] = float(len(pat_delays) / max(len(r_peaks) - 1, 1))
    return feats


def extract_all_ecg_ppg_hybrid_features(X_ecg: np.ndarray,
                                        X_ppg: np.ndarray,
                                        fs: int = 125) -> np.ndarray:
    """Extract ECG+PPG hybrid beat features for all windows."""
    N = len(X_ppg)
    feats = np.zeros((N, N_HYBRID_BEAT_FEATURES), dtype=np.float32)
    for i in range(N):
        feats[i] = extract_ecg_ppg_hybrid_features(X_ecg[i], X_ppg[i], fs=fs)
        if (i + 1) % 50000 == 0:
            print(f"  Extracted ECG+PPG beat features {i+1:,}/{N:,}...")

    return feats


def compute_sample_weights(y_sbp, y_dbp, indices, patient_ids, anchors):
    """Compute pair-aware weights to oversample anchor-relative BP tails.

    Each weight is based on the current window's delta relative to that
    patient's anchor set, rather than only its absolute BP value.
    """
    sbp_vals = []
    dbp_vals = []
    for idx in indices:
        pid = int(patient_ids[idx])
        anc_entry = anchors.get(pid)
        if anc_entry is None:
            sbp_vals.append(0.0)
            dbp_vals.append(0.0)
            continue
        anc_candidates = anc_entry if isinstance(anc_entry, list) else [anc_entry]
        sbp_deltas = np.abs(float(y_sbp[idx]) - y_sbp[anc_candidates].astype(np.float32))
        dbp_deltas = np.abs(float(y_dbp[idx]) - y_dbp[anc_candidates].astype(np.float32))
        sbp_vals.append(0.7 * float(sbp_deltas.mean()) + 0.3 * float(sbp_deltas.max()))
        dbp_vals.append(0.7 * float(dbp_deltas.mean()) + 0.3 * float(dbp_deltas.max()))
    sbp_vals = np.asarray(sbp_vals, dtype=np.float32)
    dbp_vals = np.asarray(dbp_vals, dtype=np.float32)

    sbp_norm = sbp_vals / (sbp_vals.std() + 1e-8)
    dbp_norm = dbp_vals / (dbp_vals.std() + 1e-8)
    sbp_q85, sbp_q95 = np.percentile(sbp_vals, [85, 95])
    dbp_q85, dbp_q95 = np.percentile(dbp_vals, [85, 95])

    weights = 1.0 + 0.75 * sbp_norm + 0.35 * dbp_norm
    weights += 1.00 * (sbp_vals >= sbp_q85)
    weights += 0.75 * (sbp_vals >= sbp_q95)
    weights += 0.40 * (dbp_vals >= dbp_q85)
    weights += 0.25 * (dbp_vals >= dbp_q95)
    return weights.astype(np.float64)


def compute_sbp_tail_multipliers(
    sbp_values,
    shoulder_low=110.0,
    tail_low=90.0,
    shoulder_high=130.0,
    tail_high=150.0,
    shoulder_boost=1.15,
    tail_boost=1.60,
):
    """Return mild multiplicative boosts for SBP shoulder/tail ranges."""
    sbp_values = np.asarray(sbp_values, dtype=np.float32)
    multipliers = np.ones_like(sbp_values, dtype=np.float32)

    low_tail = sbp_values < tail_low
    low_shoulder = (sbp_values >= tail_low) & (sbp_values < shoulder_low)
    high_shoulder = (sbp_values > shoulder_high) & (sbp_values <= tail_high)
    high_tail = sbp_values > tail_high

    multipliers[low_shoulder | high_shoulder] *= float(shoulder_boost)
    multipliers[low_tail | high_tail] *= float(tail_boost)
    return multipliers


def build_pair_pool(indices, patient_ids, anchors, y_sbp, y_dbp,
                    use_delta_weights=True,
                    use_sbp_tail_weights=False,
                    sbp_tail_low=90.0,
                    sbp_tail_high=150.0,
                    sbp_shoulder_low=110.0,
                    sbp_shoulder_high=130.0,
                    sbp_shoulder_boost=1.15,
                    sbp_tail_boost=1.60):
    """Build explicit (current, anchor) training pairs with configurable weights.

    Parameters let us keep the old delta-tail sampler, add absolute-SBP tail
    emphasis, or combine both.
    """
    pair_rows = []
    sbp_vals = []
    dbp_vals = []
    cur_sbp_vals = []

    for idx in indices:
        idx = int(idx)
        pid = int(patient_ids[idx])
        anc_entry = anchors.get(pid)
        if anc_entry is None:
            continue

        anc_candidates = anc_entry if isinstance(anc_entry, list) else [anc_entry]
        anc_candidates = [int(a) for a in anc_candidates]
        if len(anc_candidates) > 1:
            anc_candidates = [a for a in anc_candidates if a != idx] or anc_candidates

        for anc_i in anc_candidates:
            pair_rows.append((idx, anc_i))
            sbp_vals.append(abs(float(y_sbp[idx]) - float(y_sbp[anc_i])))
            dbp_vals.append(abs(float(y_dbp[idx]) - float(y_dbp[anc_i])))
            cur_sbp_vals.append(float(y_sbp[idx]))

    if not pair_rows:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.float64)

    sbp_vals = np.asarray(sbp_vals, dtype=np.float32)
    dbp_vals = np.asarray(dbp_vals, dtype=np.float32)
    cur_sbp_vals = np.asarray(cur_sbp_vals, dtype=np.float32)
    weights = np.ones(len(pair_rows), dtype=np.float32)

    if use_delta_weights:
        sbp_norm = sbp_vals / (sbp_vals.std() + 1e-8)
        dbp_norm = dbp_vals / (dbp_vals.std() + 1e-8)
        sbp_q85, sbp_q95 = np.percentile(sbp_vals, [85, 95])
        dbp_q85, dbp_q95 = np.percentile(dbp_vals, [85, 95])

        weights += 0.90 * sbp_norm + 0.45 * dbp_norm
        weights += 1.25 * (sbp_vals >= sbp_q85)
        weights += 1.00 * (sbp_vals >= sbp_q95)
        weights += 0.45 * (dbp_vals >= dbp_q85)
        weights += 0.30 * (dbp_vals >= dbp_q95)

    if use_sbp_tail_weights:
        weights *= compute_sbp_tail_multipliers(
            cur_sbp_vals,
            shoulder_low=sbp_shoulder_low,
            tail_low=sbp_tail_low,
            shoulder_high=sbp_shoulder_high,
            tail_high=sbp_tail_high,
            shoulder_boost=sbp_shoulder_boost,
            tail_boost=sbp_tail_boost,
        )

    return np.asarray(pair_rows, dtype=np.int64), weights.astype(np.float64)


def stack_signals(ppg, vpg, apg, preserve_ppg_amplitude: bool = True,
                  ecg: np.ndarray | None = None,
                  ppg_global_std: float | None = None) -> np.ndarray:
    """Stack waveform arrays into (N, C, W) float32, each normalised."""
    ppg_norm = (
        normalise_windows_global_scale(ppg, global_std=ppg_global_std)
        if preserve_ppg_amplitude
        else normalise_windows(ppg)
    )
    channels = [
        ppg_norm,
        normalise_windows(vpg),
        normalise_windows(apg),
    ]
    if ecg is not None:
        channels.append(normalise_windows(ecg))
    return np.stack(channels, axis=1)


def stack_signals_streamed(ppg, vpg, apg, preserve_ppg_amplitude: bool = True,
                           ecg: np.ndarray | None = None,
                           ppg_global_std: float | None = None,
                           cache_path: str | None = None,
                           batch_size: int = 2048) -> np.ndarray:
    """
    Stream waveform stacking + normalisation to disk to avoid large temporary
    allocations on big datasets, especially on Windows where memory pressure
    during evaluation can be high.

    Returns a memory-mapped array loaded from cache_path when provided.
    """
    if cache_path is None:
        return stack_signals(
            ppg, vpg, apg,
            preserve_ppg_amplitude=preserve_ppg_amplitude,
            ecg=ecg,
            ppg_global_std=ppg_global_std,
        )

    N, W = ppg.shape
    n_channels = 4 if ecg is not None else 3
    total_gb = (N * n_channels * W * np.dtype(np.float32).itemsize) / 1e9

    if os.path.exists(cache_path):
        os.remove(cache_path)

    print(
        f"  Streaming stacked waveform cache to disk (~{total_gb:.1f} GB) "
        f"in batches of {batch_size:,} windows..."
    )
    X = np.lib.format.open_memmap(
        cache_path,
        mode="w+",
        dtype=np.float32,
        shape=(N, n_channels, W),
    )

    if preserve_ppg_amplitude and ppg_global_std is None:
        ppg_global_std = float(np.std(ppg)) + 1e-8

    def _write_zscore_channel(src, channel_idx, use_global_scale=False, global_std=None):
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            chunk = np.asarray(src[start:end], dtype=np.float32)
            mu = chunk.mean(axis=-1, keepdims=True)
            if use_global_scale:
                X[start:end, channel_idx] = (chunk - mu) / global_std
            else:
                std = chunk.std(axis=-1, keepdims=True) + 1e-8
                X[start:end, channel_idx] = (chunk - mu) / std
            if channel_idx == 0:
                X.flush()
                if end % 50000 == 0 or end == N:
                    print(f"  Stacked {end:,}/{N:,} windows...")

    _write_zscore_channel(
        ppg, 0,
        use_global_scale=preserve_ppg_amplitude,
        global_std=ppg_global_std,
    )
    _write_zscore_channel(vpg, 1)
    _write_zscore_channel(apg, 2)
    if ecg is not None:
        _write_zscore_channel(ecg, 3)

    X.flush()
    mmap_obj = getattr(X, "_mmap", None)
    if mmap_obj is not None:
        mmap_obj.close()
    del X
    return np.load(cache_path, mmap_mode="r", allow_pickle=False)


def load_data(data_dir, random_anchor=False, min_sbp_std=5.0,
              num_anchors=1, use_spectrograms=False, use_fusion=False,
              target_scale=1.0, quality_threshold=0.4, shuffle_eval=True,
              preserve_ppg_amplitude=True, use_beat_features=False,
              use_ecg=False, ecg_input_mode="both"):
    """
    Load all arrays -> (train_ds, val_ds, test_ds, anchors, y_sbp, y_dbp, sample_weights).

    quality_threshold : float
        Minimum signal quality score (0-1) to include a window.
        Set to 0.0 to disable quality filtering.
    shuffle_eval : bool
        If True, randomly split the eval windows 50/50 into val/test
        instead of chronological ordering. Removes temporal bias.
    """
    p = lambda f: os.path.join(data_dir, f)

    valid_ecg_modes = {"both", "features_only", "raw_only"}
    if ecg_input_mode not in valid_ecg_modes:
        raise ValueError(
            f"ecg_input_mode must be one of {sorted(valid_ecg_modes)}, got {ecg_input_mode!r}"
        )

    use_ecg_raw_channel = bool(use_ecg and ecg_input_mode in {"both", "raw_only"})
    use_ecg_features = bool(use_ecg and ecg_input_mode in {"both", "features_only"})
    need_ecg = use_ecg_raw_channel or use_ecg_features

    print("Loading signals (memory-mapped)...")
    ppg = np.load(p("X_ppg_windows.npy"), mmap_mode="r", allow_pickle=False)
    vpg = np.load(p("X_vpg.npy"),         mmap_mode="r", allow_pickle=False)
    apg = np.load(p("X_apg.npy"),         mmap_mode="r", allow_pickle=False)
    ecg = None
    if need_ecg:
        ecg_path = p("X_ecg.npy")
        if not os.path.exists(ecg_path):
            raise FileNotFoundError(
                f"ECG mode requires {ecg_path}, but it was not found. "
                "Run the ECG+PPG preprocessing pipeline first."
            )
        ecg = np.load(ecg_path, mmap_mode="r", allow_pickle=False)
        if ecg.shape != ppg.shape:
            raise ValueError(
                f"X_ecg.npy shape {ecg.shape} does not match X_ppg_windows.npy {ppg.shape}"
            )
        print(f"  PPG {ppg.shape}  VPG {vpg.shape}  APG {apg.shape}  ECG {ecg.shape}")
    else:
        print(f"  PPG {ppg.shape}  VPG {vpg.shape}  APG {apg.shape}")

    # Auto-detect sampling rate from window length
    # v8: 800 samples = 4s @ 200Hz;  v9: 1000 samples = 10s @ 100Hz
    W = ppg.shape[1]
    if W == 800:
        detected_fs = 200
    elif W == 1000:
        detected_fs = 100
    else:
        detected_fs = 125  # fallback
    print(f"  Detected fs={detected_fs} Hz (W={W})")

    y_sbp       = np.load(p("y_sbp.npy"),      allow_pickle=False)
    y_dbp       = np.load(p("y_dbp.npy"),      allow_pickle=False)
    patient_ids = np.load(p("patient_ids.npy"), allow_pickle=False)

    # ── Signal quality filtering ─────────────────
    quality_scores = None
    if quality_threshold > 0:
        qc_cache = make_derived_cache_path(data_dir, "X_quality_scores", detected_fs, W)
        legacy_qc_cache = p("X_quality_scores.npy")
        if os.path.exists(legacy_qc_cache) and not os.path.exists(qc_cache):
            print(f"Found legacy quality-score cache at {legacy_qc_cache}; regenerating fs-aware cache.")
        if os.path.exists(qc_cache):
            print(f"Loading cached quality scores from {qc_cache}...")
            quality_scores = np.load(qc_cache, allow_pickle=False)
        else:
            print("Computing signal quality scores (cached for next run)...")
            quality_scores = compute_quality_scores(ppg, fs=detected_fs)
            np.save(qc_cache, quality_scores)
            print(f"  Saved cache to {qc_cache}")

        n_total_windows = len(quality_scores)
        n_good = (quality_scores >= quality_threshold).sum()
        n_bad = n_total_windows - n_good
        print(f"  Quality filter: {n_good:,} good, {n_bad:,} removed (threshold={quality_threshold})")

    print(f"\nFiltering patients (SBP std >= {min_sbp_std} mmHg)...")
    valid_patients = get_valid_patients(y_sbp, patient_ids, min_sbp_std)
    n_total = len(np.unique(patient_ids))
    print(f"  Kept {len(valid_patients)}/{n_total} patients")

    split_mode = "shuffled" if shuffle_eval else "chronological"
    print(f"Creating within-patient temporal splits (70 / 15 / 15, eval={split_mode})...")
    train_mask, val_mask, test_mask = make_within_patient_masks(
        patient_ids, valid_patients, train_frac=0.70, val_frac=0.15,
        shuffle_eval=shuffle_eval)

    # Apply quality filter to masks
    if quality_scores is not None and quality_threshold > 0:
        bad_mask = quality_scores < quality_threshold
        n_train_before = train_mask.sum()
        n_val_before = val_mask.sum()
        n_test_before = test_mask.sum()
        train_mask = train_mask & ~bad_mask
        val_mask = val_mask & ~bad_mask
        test_mask = test_mask & ~bad_mask
        print(f"  After quality filter: train {n_train_before}->{train_mask.sum()}, "
              f"val {n_val_before}->{val_mask.sum()}, test {n_test_before}->{test_mask.sum()}")

    train_stat_idx = np.where(train_mask)[0]
    if train_stat_idx.size == 0:
        raise ValueError("Training split is empty after filtering; cannot fit train-only statistics.")

    norm_tag = "ppg_amp_trainv2" if preserve_ppg_amplitude else "ppg_zscore"
    if use_ecg_raw_channel:
        norm_tag += "_ecg"
    norm_desc = "global-scale centered PPG (train-only scale)" if preserve_ppg_amplitude else "per-window z-scored PPG"
    channel_desc = "(N, 4, W)" if use_ecg_raw_channel else "(N, 3, W)"
    extra_desc = " + z-scored ECG" if use_ecg_raw_channel else ""
    print(f"Stacking + normalising -> {channel_desc}...  [{norm_desc}{extra_desc}]")
    wave_cache = make_derived_cache_path(data_dir, "X_stacked", detected_fs, W, extra=norm_tag)
    if os.path.exists(wave_cache):
        print(f"Loading cached stacked waveforms from {wave_cache}...")
        X = np.load(wave_cache, mmap_mode="r", allow_pickle=False)
    else:
        ppg_train_std = None
        if preserve_ppg_amplitude:
            print("  Fitting train-only global PPG scale...")
            ppg_train_std = compute_global_std_from_indices(ppg, train_stat_idx)
            print(f"  Train-only PPG global std: {ppg_train_std:.6f}")
        X = stack_signals_streamed(
            ppg, vpg, apg,
            preserve_ppg_amplitude=preserve_ppg_amplitude,
            ecg=ecg if use_ecg_raw_channel else None,
            ppg_global_std=ppg_train_std,
            cache_path=wave_cache,
        )
        print(f"  Saved cache to {wave_cache} ({os.path.getsize(wave_cache) / 1e9:.1f} GB)")
    print(f"  X {X.shape}  {X.dtype}")

    X_spec = None  # spectrograms array (only loaded for spectrogram/fusion modes)

    if use_spectrograms or use_fusion:
        spec_cache = make_derived_cache_path(
            data_dir, "X_spectrograms", detected_fs, W, extra=f"nper64_ov56_{norm_tag}"
        )
        legacy_spec_cache = p("X_spectrograms.npy")
        if os.path.exists(legacy_spec_cache) and not os.path.exists(spec_cache):
            print(f"Found legacy spectrogram cache at {legacy_spec_cache}; regenerating fs-aware cache.")
        if os.path.exists(spec_cache):
            print(f"Loading cached spectrograms from {spec_cache}...")
            try:
                X_spec_data = np.load(spec_cache, mmap_mode="r", allow_pickle=False)
                print(f"  X_spec {X_spec_data.shape}  {X_spec_data.dtype}")
            except Exception as exc:
                print(f"  Spectrogram cache load failed ({exc}); rebuilding cache...")
                try:
                    os.remove(spec_cache)
                except OSError:
                    pass
                X_spec_data = None
        else:
            X_spec_data = None

        if X_spec_data is None:
            print("Computing spectrograms (streamed to disk, cached for next run)...")
            X_spec_data = signals_to_spectrograms(
                X,
                fs=detected_fs,
                nperseg=64,
                noverlap=56,
                cache_path=spec_cache,
            )
            print(f"  X_spec {X_spec_data.shape}  {X_spec_data.dtype}")
            print(f"  Saved cache to {spec_cache} ({os.path.getsize(spec_cache) / 1e9:.1f} GB)")

        if use_fusion:
            X_spec = X_spec_data
        else:
            X = X_spec_data

    # ── Handcrafted PPG features ─────────────────
    hf_cache = make_derived_cache_path(data_dir, "X_hand_features", detected_fs, W, extra="raw_v2")
    if os.path.exists(hf_cache):
        print(f"Loading cached handcrafted features from {hf_cache}...")
        hand_features = np.load(hf_cache, allow_pickle=False)
        print(f"  hand_features(raw) {hand_features.shape}  {hand_features.dtype}")
    else:
        print("Extracting handcrafted PPG features (cached for next run)...")
        hand_features = extract_all_features(ppg, fs=detected_fs)
        print(f"  hand_features(raw) {hand_features.shape}")
        np.save(hf_cache, hand_features)
        print(f"  Saved cache to {hf_cache}")
    hand_features = standardize_features_from_train_stats(hand_features, train_mask)
    print(f"  hand_features(train-z) {hand_features.shape}  {hand_features.dtype}")

    if use_beat_features:
        beat_cache = make_derived_cache_path(data_dir, "X_beat_features", detected_fs, W, extra="raw_v2")
        if os.path.exists(beat_cache):
            print(f"Loading cached beat-aware features from {beat_cache}...")
            beat_features = np.load(beat_cache, allow_pickle=False)
            print(f"  beat_features(raw) {beat_features.shape}  {beat_features.dtype}")
        else:
            print("Extracting beat-aware PPG features (cached for next run)...")
            beat_features = extract_all_beat_features(ppg, fs=detected_fs)
            print(f"  beat_features(raw) {beat_features.shape}")
            np.save(beat_cache, beat_features)
            print(f"  Saved cache to {beat_cache}")
        beat_features = standardize_features_from_train_stats(beat_features, train_mask)
        hand_features = np.concatenate([hand_features, beat_features], axis=1).astype(np.float32)
        dim_note = f"base {N_HAND_FEATURES} + beat {N_BEAT_FEATURES}"

        if use_ecg_features:
            hybrid_cache = make_derived_cache_path(
                data_dir, "X_ecg_ppg_beat_features", detected_fs, W, extra="raw_v2"
            )
            if os.path.exists(hybrid_cache):
                print(f"Loading cached ECG+PPG beat features from {hybrid_cache}...")
                hybrid_features = np.load(hybrid_cache, allow_pickle=False)
                print(f"  ecg_ppg_beat_features(raw) {hybrid_features.shape}  {hybrid_features.dtype}")
            else:
                print("Extracting ECG+PPG hybrid beat features (cached for next run)...")
                hybrid_features = extract_all_ecg_ppg_hybrid_features(ecg, ppg, fs=detected_fs)
                print(f"  ecg_ppg_beat_features(raw) {hybrid_features.shape}")
                np.save(hybrid_cache, hybrid_features)
                print(f"  Saved cache to {hybrid_cache}")
            hybrid_features = standardize_features_from_train_stats(hybrid_features, train_mask)
            hand_features = np.concatenate([hand_features, hybrid_features], axis=1).astype(np.float32)
            dim_note += f" + hybrid {N_HYBRID_BEAT_FEATURES}"

        print(f"  Combined feature dim: {hand_features.shape[1]} ({dim_note}; train-only z-score)")

    # ── Anchor selection ───────────────────────
    print(f"Selecting anchors (joint SBP+DBP median, K={num_anchors}, training windows only)...")
    anchors = select_anchors(y_sbp, y_dbp, patient_ids, valid_patients,
                             random_anchor=random_anchor,
                             train_mask=train_mask,
                             num_anchors=num_anchors)
    print(f"  {len(anchors)} patients  (K={num_anchors}, random={random_anchor})")

    # Keep full anchor lists for multi-anchor evaluation, while the default
    # batch path stays deterministic by using the first anchor.
    if num_anchors > 1:
        val_anchors = {pid: ancs[0] for pid, ancs in anchors.items()}
        eval_anchor_lists = {pid: list(ancs) for pid, ancs in anchors.items()}
    else:
        val_anchors = anchors
        eval_anchor_lists = {pid: [anc] for pid, anc in anchors.items()}

    # ── Build datasets ─────────────────────────
    train_ds = SiameseBPDataset(X, y_sbp, y_dbp, patient_ids,
                                train_mask, anchors, valid_patients,
                                randomize_anchor=False,
                                augment=True, X_spec=X_spec,
                                target_scale=target_scale,
                                hand_features=hand_features,
                                anchor_selection="random_list")
    val_ds   = SiameseBPDataset(X, y_sbp, y_dbp, patient_ids,
                                val_mask,   eval_anchor_lists, valid_patients,
                                randomize_anchor=False, augment=False, X_spec=X_spec,
                                target_scale=target_scale,
                                hand_features=hand_features,
                                anchor_selection="first")
    test_ds  = SiameseBPDataset(X, y_sbp, y_dbp, patient_ids,
                                test_mask,  eval_anchor_lists, valid_patients,
                                randomize_anchor=False, augment=False, X_spec=X_spec,
                                target_scale=target_scale,
                                hand_features=hand_features,
                                anchor_selection="first")

    print(f"\n  Train : {len(train_ds):,} windows")
    print(f"  Val   : {len(val_ds):,} windows")
    print(f"  Test  : {len(test_ds):,} windows")
    print(f"  SBP   : {y_sbp.min():.0f}-{y_sbp.max():.0f} mmHg")
    print(f"  DBP   : {y_dbp.min():.0f}-{y_dbp.max():.0f} mmHg")

    # ── Balanced sampling weights ─────────────
    print("Computing balanced sampling weights...")
    sample_weights = compute_sample_weights(y_sbp, y_dbp, train_ds.indices, patient_ids, anchors)
    print(f"  Weight range: {sample_weights.min():.2f} - {sample_weights.max():.2f}")

    # ── New baseline after filtering ───────────
    val_idx   = np.array([i for i in np.where(val_mask)[0]
                          if int(patient_ids[i]) in valid_patients])
    anc_sbp   = np.array([y_sbp[val_anchors[int(patient_ids[i])]] for i in val_idx])
    delta_sbp = np.abs(y_sbp[val_idx] - anc_sbp)
    delta_dbp_arr = np.abs(y_dbp[val_idx] -
                    np.array([y_dbp[val_anchors[int(patient_ids[i])]] for i in val_idx]))
    print(f"\n  New baseline (predict-zero) SBP MAE: {delta_sbp.mean():.2f} mmHg")
    print(f"  New baseline (predict-zero) DBP MAE: {delta_dbp_arr.mean():.2f} mmHg")

    return train_ds, val_ds, test_ds, anchors, y_sbp, y_dbp, sample_weights
