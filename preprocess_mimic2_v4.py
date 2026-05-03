"""
MIMIC-II PPG Preprocessing - v4 (beat-level ABP labels)
===================================================

This keeps the earlier PPG windowing setup, but changes how the ABP labels are
picked:

1. Beat-level ABP labeling
   - detect systolic peaks and preceding diastolic troughs per beat
   - reject implausible or unstable beats
   - aggregate window SBP/DBP with the median, not raw-window extrema means

2. Window periodicity quality gates on BOTH PPG and ABP
   - lightweight autocorrelation score to reject noisy windows

3. CLI-configurable window length
   - default remains 10 s because that is what the main pipeline uses, but
     this script can also generate 20 s / 30 s
     variants for future experiments without touching v9.

Output format matches the existing training pipeline:
    X_ppg_windows.npy, X_vpg.npy, X_apg.npy,
    y_sbp.npy, y_dbp.npy, patient_ids.npy,
    train_mask.npy, val_mask.npy, test_mask.npy
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import butter, filtfilt, find_peaks, resample

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="MIMIC-II PPG preprocessing with beat-level ABP labels.")
    parser.add_argument("--input_dir", type=str, default=r"C:\MIMIC 2")
    parser.add_argument("--output_dir", type=str, default=r"C:\MIMIC2_out_v11")
    parser.add_argument("--window_sec", type=float, default=10.0)
    parser.add_argument("--target_fs", type=int, default=100)
    parser.add_argument("--sbp_min", type=float, default=75.0)
    parser.add_argument("--sbp_max", type=float, default=165.0)
    parser.add_argument("--dbp_min", type=float, default=40.0)
    parser.add_argument("--dbp_max", type=float, default=85.0)
    parser.add_argument("--min_pp", type=float, default=15.0)
    parser.add_argument("--hr_min", type=float, default=50.0)
    parser.add_argument("--hr_max", type=float, default=140.0)
    parser.add_argument("--min_windows", type=int, default=100)
    parser.add_argument("--max_windows_per_patient", type=int, default=500)
    parser.add_argument("--per_patient_bp_cap", type=float, default=40.0)
    parser.add_argument("--train_frac", type=float, default=0.60)
    parser.add_argument("--val_frac", type=float, default=0.20)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--ppg_periodicity_min", type=float, default=0.55)
    parser.add_argument("--abp_periodicity_min", type=float, default=0.65)
    parser.add_argument("--min_valid_beats", type=int, default=3)
    return parser.parse_args()


def bandpass(signal, fs, low=0.5, high=8.0, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def compute_vpg_apg(ppg):
    taper = np.hanning(len(ppg))
    ppg_tapered = ppg * taper
    vpg = np.gradient(ppg_tapered)
    apg = np.gradient(vpg)
    return vpg.astype(np.float32), apg.astype(np.float32)


def is_flat(sig, threshold=1e-4):
    return np.std(sig) < threshold


def smooth_signal(x: np.ndarray, fs: float, width_sec: float = 0.08) -> np.ndarray:
    k = max(3, int(round(width_sec * fs)))
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(x.astype(np.float32), kernel, mode="same")


def periodicity_score(sig: np.ndarray, fs: float, hr_min: float, hr_max: float) -> float:
    x = sig.astype(np.float32) - float(np.mean(sig))
    denom = float(np.std(x))
    if denom < 1e-6:
        return 0.0
    x = x / denom
    ac = np.correlate(x, x, mode="full")
    ac = ac[len(x) - 1 :]
    if ac[0] <= 0:
        return 0.0
    ac = ac / ac[0]
    lag_lo = max(1, int(fs * 60.0 / hr_max))
    lag_hi = min(len(ac) - 1, int(fs * 60.0 / hr_min))
    if lag_hi <= lag_lo:
        return 0.0
    return float(np.clip(np.max(ac[lag_lo : lag_hi + 1]), 0.0, 1.0))


def estimate_hr(ppg, fs, hr_max):
    min_dist = max(1, int(fs * 60 / hr_max))
    peaks, _ = find_peaks(ppg, distance=min_dist, prominence=0.01)
    if len(peaks) < 2:
        return None
    intervals = np.diff(peaks) / fs
    mean_rr = float(np.mean(intervals))
    if mean_rr <= 0:
        return None
    return 60.0 / mean_rr


def robust_extract_bp(
    abp_win: np.ndarray,
    fs: float,
    sbp_min: float,
    sbp_max: float,
    dbp_min: float,
    dbp_max: float,
    min_pp: float,
    hr_max: float,
    min_valid_beats: int,
):
    x = smooth_signal(abp_win.astype(np.float32), fs)
    min_dist = max(1, int(fs * 60.0 / hr_max))

    peaks, _ = find_peaks(x, distance=min_dist, prominence=5.0)
    troughs, _ = find_peaks(-x, distance=min_dist, prominence=3.0)

    if len(peaks) == 0 or len(troughs) < 2:
        return None

    sbp_vals = []
    dbp_vals = []

    for p in peaks:
        left_pos = np.searchsorted(troughs, p, side="left") - 1
        right_pos = left_pos + 1
        if left_pos < 0 or right_pos >= len(troughs):
            continue

        left_t = int(troughs[left_pos])
        right_t = int(troughs[right_pos])
        if not (left_t < p < right_t):
            continue

        sbp = float(x[p])
        dbp = float(x[left_t])
        pp = sbp - dbp
        beat_len_sec = (right_t - left_t) / float(fs)

        if beat_len_sec <= 0:
            continue
        if not (sbp_min <= sbp <= sbp_max):
            continue
        if not (dbp_min <= dbp <= dbp_max):
            continue
        if pp < min_pp or pp > 120.0:
            continue
        if beat_len_sec < 60.0 / 180.0 or beat_len_sec > 60.0 / 35.0:
            continue

        sbp_vals.append(sbp)
        dbp_vals.append(dbp)

    if len(sbp_vals) < min_valid_beats:
        return None

    sbp_vals = np.asarray(sbp_vals, dtype=np.float32)
    dbp_vals = np.asarray(dbp_vals, dtype=np.float32)

    # Reject windows whose beatwise labels are too unstable.
    sbp_iqr = float(np.percentile(sbp_vals, 75) - np.percentile(sbp_vals, 25))
    dbp_iqr = float(np.percentile(dbp_vals, 75) - np.percentile(dbp_vals, 25))
    if sbp_iqr > 18.0 or dbp_iqr > 12.0:
        return None

    sbp = float(np.median(sbp_vals))
    dbp = float(np.median(dbp_vals))
    return sbp, dbp


def process_patient(patient_dir: Path, cfg: argparse.Namespace):
    hea_files = list(patient_dir.glob("*.hea"))
    if not hea_files:
        return None

    waveform = [f for f in hea_files if not f.stem.endswith("n")]
    record_path = None

    for hf in waveform:
        candidate = str(hf.with_suffix(""))
        try:
            header = wfdb.rdheader(candidate)
            sig_names = [s.upper() for s in header.sig_name]
            has_ppg = any(s in sig_names for s in ["PLETH", "PPG"])
            has_abp = any(s in sig_names for s in ["ABP", "ART", "BP"])
            if has_ppg and has_abp:
                record_path = candidate
                break
        except Exception:
            continue

    if record_path is None:
        return None

    try:
        record = wfdb.rdrecord(record_path)
    except Exception:
        return None

    sig_names = [s.upper() for s in record.sig_name]
    fs = float(record.fs)

    ppg_idx = next((i for i, s in enumerate(sig_names) if s in ["PLETH", "PPG"]), None)
    abp_idx = next((i for i, s in enumerate(sig_names) if s in ["ABP", "ART", "BP"]), None)
    if ppg_idx is None or abp_idx is None:
        return None

    ppg_raw = record.p_signal[:, ppg_idx].astype(np.float32)
    abp_raw = record.p_signal[:, abp_idx].astype(np.float32)
    valid = np.isfinite(ppg_raw) & np.isfinite(abp_raw)
    ppg_raw = ppg_raw[valid]
    abp_raw = abp_raw[valid]

    if len(ppg_raw) < int(fs * cfg.window_sec * 2):
        return None

    try:
        ppg_filt = bandpass(ppg_raw, fs)
    except Exception:
        return None

    win_samples = int(fs * cfg.window_sec)
    n_windows = len(ppg_filt) // win_samples

    ppg_list, vpg_list, apg_list = [], [], []
    sbp_list, dbp_list = [], []
    abp_q_list, ppg_q_list = [], []

    for w in range(n_windows):
        start = w * win_samples
        end = start + win_samples

        ppg_win = ppg_filt[start:end]
        abp_win = abp_raw[start:end]

        if not np.all(np.isfinite(ppg_win)) or not np.all(np.isfinite(abp_win)):
            continue
        if is_flat(ppg_win) or is_flat(abp_win):
            continue

        ppg_q = periodicity_score(ppg_win, fs, cfg.hr_min, cfg.hr_max)
        abp_q = periodicity_score(abp_win, fs, cfg.hr_min, cfg.hr_max)
        if ppg_q < cfg.ppg_periodicity_min or abp_q < cfg.abp_periodicity_min:
            continue

        bp = robust_extract_bp(
            abp_win,
            fs=fs,
            sbp_min=cfg.sbp_min,
            sbp_max=cfg.sbp_max,
            dbp_min=cfg.dbp_min,
            dbp_max=cfg.dbp_max,
            min_pp=cfg.min_pp,
            hr_max=cfg.hr_max,
            min_valid_beats=cfg.min_valid_beats,
        )
        if bp is None:
            continue
        sbp, dbp = bp

        hr = estimate_hr(ppg_win, fs, cfg.hr_max)
        if hr is None or not (cfg.hr_min <= hr <= cfg.hr_max):
            continue

        ppg_res = resample(ppg_win, int(cfg.window_sec * cfg.target_fs)).astype(np.float32)
        vpg, apg = compute_vpg_apg(ppg_res)

        ppg_list.append(ppg_res)
        vpg_list.append(vpg)
        apg_list.append(apg)
        sbp_list.append(sbp)
        dbp_list.append(dbp)
        ppg_q_list.append(ppg_q)
        abp_q_list.append(abp_q)

    if len(ppg_list) < cfg.min_windows:
        return None

    sbp_arr = np.array(sbp_list, dtype=np.float32)
    dbp_arr = np.array(dbp_list, dtype=np.float32)
    ref_sbp = float(sbp_arr[0])
    ref_dbp = float(dbp_arr[0])
    keep_mask = (
        (np.abs(sbp_arr - ref_sbp) <= cfg.per_patient_bp_cap)
        & (np.abs(dbp_arr - ref_dbp) <= cfg.per_patient_bp_cap)
    )

    n_before = len(ppg_list)
    ppg_list = [ppg_list[i] for i in range(n_before) if keep_mask[i]]
    vpg_list = [vpg_list[i] for i in range(n_before) if keep_mask[i]]
    apg_list = [apg_list[i] for i in range(n_before) if keep_mask[i]]
    sbp_list = [sbp_list[i] for i in range(n_before) if keep_mask[i]]
    dbp_list = [dbp_list[i] for i in range(n_before) if keep_mask[i]]
    ppg_q_list = [ppg_q_list[i] for i in range(n_before) if keep_mask[i]]
    abp_q_list = [abp_q_list[i] for i in range(n_before) if keep_mask[i]]

    if len(ppg_list) < cfg.min_windows:
        return None

    if len(ppg_list) > cfg.max_windows_per_patient:
        rng = np.random.default_rng()
        idx = np.sort(rng.choice(len(ppg_list), cfg.max_windows_per_patient, replace=False))
        ppg_list = [ppg_list[i] for i in idx]
        vpg_list = [vpg_list[i] for i in idx]
        apg_list = [apg_list[i] for i in idx]
        sbp_list = [sbp_list[i] for i in idx]
        dbp_list = [dbp_list[i] for i in idx]
        ppg_q_list = [ppg_q_list[i] for i in idx]
        abp_q_list = [abp_q_list[i] for i in idx]

    return {
        "ppg": np.array(ppg_list, dtype=np.float32),
        "vpg": np.array(vpg_list, dtype=np.float32),
        "apg": np.array(apg_list, dtype=np.float32),
        "sbp": np.array(sbp_list, dtype=np.float32),
        "dbp": np.array(dbp_list, dtype=np.float32),
        "ppg_q": np.array(ppg_q_list, dtype=np.float32),
        "abp_q": np.array(abp_q_list, dtype=np.float32),
        "n_before_cap": n_before,
        "n_after_cap": len(ppg_list),
    }


def main():
    cfg = parse_args()
    input_path = Path(cfg.input_dir)
    output_path = Path(cfg.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("MIMIC-II PPG PREPROCESSING - v4 (robust ABP labels)")
    print(f"Input:   {cfg.input_dir}")
    print(f"Output:  {cfg.output_dir}")
    print(f"Window:  {cfg.window_sec:.1f}s @ {cfg.target_fs}Hz = {int(cfg.window_sec * cfg.target_fs)} samples")
    print(f"BP gate: SBP [{cfg.sbp_min:.0f},{cfg.sbp_max:.0f}]  DBP [{cfg.dbp_min:.0f},{cfg.dbp_max:.0f}]")
    print(f"PPG/ABP periodicity min: {cfg.ppg_periodicity_min:.2f} / {cfg.abp_periodicity_min:.2f}")
    print(f"Per-patient BP cap: +/-{cfg.per_patient_bp_cap:.0f} mmHg from first kept window")
    print("=" * 68)

    patient_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()])
    print(f"Found {len(patient_dirs)} patient directories\n")

    all_patients = []
    total_before_cap = 0
    total_dropped_by_cap = 0

    for p_dir in patient_dirs:
        result = process_patient(p_dir, cfg)
        if result is None:
            print(f"  [---] {p_dir.name:20s}  skipped")
            continue

        result["name"] = p_dir.name
        all_patients.append(result)
        n = len(result["sbp"])
        dropped = result["n_before_cap"] - result["n_after_cap"]
        total_before_cap += result["n_before_cap"]
        total_dropped_by_cap += dropped

        sbp_med = float(np.median(result["sbp"]))
        dbp_med = float(np.median(result["dbp"]))
        ppg_q_med = float(np.median(result["ppg_q"]))
        abp_q_med = float(np.median(result["abp_q"]))
        print(
            f"  [{len(all_patients):3d}] {p_dir.name:20s}  {n:5d} windows  "
            f"SBP={sbp_med:.0f}  DBP={dbp_med:.0f}  Qppg={ppg_q_med:.2f}  Qabp={abp_q_med:.2f}  "
            f"(cap dropped {dropped})"
        )

    n_patients = len(all_patients)
    print(f"\n{'=' * 68}")
    print(f"Patients accepted: {n_patients} / {len(patient_dirs)}")
    if total_before_cap > 0:
        print(
            f"Windows removed by +/-{cfg.per_patient_bp_cap:.0f} mmHg cap: "
            f"{total_dropped_by_cap:,} / {total_before_cap:,} "
            f"({100.0 * total_dropped_by_cap / total_before_cap:.1f}%)"
        )
    if n_patients < 10:
        print("ERROR: Too few patients after preprocessing.")
        return

    rng = np.random.default_rng(cfg.split_seed)
    idx = rng.permutation(n_patients)
    n_train = int(cfg.train_frac * n_patients)
    n_val = int(cfg.val_frac * n_patients)
    train_pids = set(idx[:n_train].tolist())
    val_pids = set(idx[n_train : n_train + n_val].tolist())
    test_pids = set(idx[n_train + n_val :].tolist())
    print(f"\nSplit: train={len(train_pids)}  val={len(val_pids)}  test={len(test_pids)} patients")

    all_ppg, all_vpg, all_apg = [], [], []
    all_sbp, all_dbp = [], []
    all_pids = []
    train_mask, val_mask, test_mask = [], [], []
    patient_names = []

    for pid, patient in enumerate(all_patients):
        n = len(patient["sbp"])
        all_ppg.append(patient["ppg"])
        all_vpg.append(patient["vpg"])
        all_apg.append(patient["apg"])
        all_sbp.append(patient["sbp"])
        all_dbp.append(patient["dbp"])
        all_pids.extend([pid] * n)
        patient_names.append(patient["name"])

        train_mask.extend([pid in train_pids] * n)
        val_mask.extend([pid in val_pids] * n)
        test_mask.extend([pid in test_pids] * n)

    X_ppg = np.concatenate(all_ppg, axis=0)
    X_vpg = np.concatenate(all_vpg, axis=0)
    X_apg = np.concatenate(all_apg, axis=0)
    y_sbp = np.concatenate(all_sbp, axis=0)
    y_dbp = np.concatenate(all_dbp, axis=0)
    pids = np.array(all_pids, dtype=np.int64)
    train_mask = np.array(train_mask, dtype=bool)
    val_mask = np.array(val_mask, dtype=bool)
    test_mask = np.array(test_mask, dtype=bool)

    print(f"\nTotal windows: {len(y_sbp):,}")
    print(f"  Train: {train_mask.sum():,}")
    print(f"  Val:   {val_mask.sum():,}")
    print(f"  Test:  {test_mask.sum():,}")
    print(
        f"\nSBP: mean={y_sbp.mean():.1f}  std={y_sbp.std():.1f}  min={y_sbp.min():.1f}  max={y_sbp.max():.1f}"
    )
    print(
        f"DBP: mean={y_dbp.mean():.1f}  std={y_dbp.std():.1f}  min={y_dbp.min():.1f}  max={y_dbp.max():.1f}"
    )

    np.save(output_path / "X_ppg_windows.npy", X_ppg)
    np.save(output_path / "X_vpg.npy", X_vpg)
    np.save(output_path / "X_apg.npy", X_apg)
    np.save(output_path / "y_sbp.npy", y_sbp)
    np.save(output_path / "y_dbp.npy", y_dbp)
    np.save(output_path / "patient_ids.npy", pids)
    np.save(output_path / "train_mask.npy", train_mask)
    np.save(output_path / "val_mask.npy", val_mask)
    np.save(output_path / "test_mask.npy", test_mask)

    with open(output_path / "patient_names.txt", "w", encoding="utf-8") as f:
        for name in patient_names:
            f.write(name + "\n")

    metadata = {
        "input_dir": cfg.input_dir,
        "output_dir": cfg.output_dir,
        "window_sec": cfg.window_sec,
        "target_fs": cfg.target_fs,
        "ppg_periodicity_min": cfg.ppg_periodicity_min,
        "abp_periodicity_min": cfg.abp_periodicity_min,
        "min_valid_beats": cfg.min_valid_beats,
        "per_patient_bp_cap": cfg.per_patient_bp_cap,
        "accepted_patients": int(n_patients),
        "total_windows": int(len(y_sbp)),
        "sbp_mean": float(y_sbp.mean()),
        "dbp_mean": float(y_dbp.mean()),
    }
    (output_path / "preprocess_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nOK Saved to {cfg.output_dir}")
    print("\nSuggested next step:")
    print(
        "  python train.py --data_dir \"{0}\" --save_dir .\\checkpoints_v4_abpbeat "
        "--use_fusion --use_beat_features --use_gated_pair_interaction "
        "--quality_threshold 0.4 --shuffle_eval --workers 0".format(cfg.output_dir)
    )


if __name__ == "__main__":
    main()
