"""
preprocess_mimic2_ecgppg.py
---------------------------
Build a hybrid ECG+PPG MIMIC-II dataset directly from the raw WFDB folders.

Outputs:
    X_ppg_windows.npy
    X_vpg.npy
    X_apg.npy
    X_ecg.npy
    y_sbp.npy
    y_dbp.npy
    patient_ids.npy
    train_mask.npy
    val_mask.npy
    test_mask.npy
    patient_names.txt

This mirrors the current PPG-only preprocessing, but keeps an aligned ECG
window for each accepted PPG/ABP window so the downstream Siamese model can
compare ECG+PPG pairs with the same anchor-relative BP formulation.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import butter, filtfilt, find_peaks, resample

warnings.filterwarnings("ignore")

PPG_NAMES = ("PLETH", "PPG")
ABP_NAMES = ("ABP", "ART", "BP")
ECG_NAMES = (
    "II", "ECG", "ECG1", "MLII", "MCL1", "I", "III",
    "V", "V1", "V2", "V3", "V4", "V5", "AVL", "AVR", "AVF",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess raw MIMIC-II into ECG+PPG windows")
    parser.add_argument("--input_dir", default=r"C:\MIMIC 2")
    parser.add_argument("--output_dir", default=r"C:\MIMIC2_out_v10_ecgppg")
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_patients", type=int, default=0,
                        help="Optional limit for quick sanity runs (0 = all patients)")
    return parser.parse_args()


def bandpass(signal, fs, low, high, order=4):
    nyq = fs / 2.0
    low = max(float(low), 1e-4)
    high = min(float(high), nyq - 1e-3)
    if high <= low:
        high = min(nyq - 1e-3, low * 2.0)
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


def extract_bp(abp_win, fs, hr_max):
    min_dist = int(fs * 60 / hr_max)
    peaks, _ = find_peaks(abp_win, distance=min_dist, prominence=0.5)
    troughs, _ = find_peaks(-abp_win, distance=min_dist, prominence=0.5)
    if len(peaks) >= 2 and len(troughs) >= 2:
        sbp = float(np.mean(abp_win[peaks]))
        dbp = float(np.mean(abp_win[troughs]))
    else:
        sbp = float(np.percentile(abp_win, 98))
        dbp = float(np.percentile(abp_win, 2))
    return sbp, dbp


def estimate_hr(ppg, fs, hr_max):
    min_dist = int(fs * 60 / hr_max)
    peaks, _ = find_peaks(ppg, distance=min_dist, prominence=0.01)
    if len(peaks) < 2:
        return None
    intervals = np.diff(peaks) / fs
    mean_rr = np.mean(intervals)
    if mean_rr <= 0:
        return None
    return 60.0 / mean_rr


def pick_signal_index(sig_names, candidates):
    sig_names = [name.upper() for name in sig_names]
    for candidate in candidates:
        if candidate in sig_names:
            return sig_names.index(candidate)
    return None


def process_patient(patient_dir: Path, args):
    hea_files = list(patient_dir.glob("*.hea"))
    if not hea_files:
        return None

    waveform_headers = [f for f in hea_files if not f.stem.endswith("n")]
    record_name = None
    for header_path in waveform_headers:
        candidate = str(header_path.with_suffix(""))
        try:
            header = wfdb.rdheader(candidate)
        except Exception:
            continue
        sig_names = header.sig_name
        if sig_names is None:
            layout_path = Path(candidate + "_layout.hea")
            if layout_path.exists():
                try:
                    sig_names = wfdb.rdheader(str(layout_path.with_suffix(""))).sig_name
                except Exception:
                    sig_names = None
        if sig_names is None:
            continue

        ppg_idx = pick_signal_index(sig_names, PPG_NAMES)
        abp_idx = pick_signal_index(sig_names, ABP_NAMES)
        ecg_idx = pick_signal_index(sig_names, ECG_NAMES)
        if ppg_idx is not None and abp_idx is not None and ecg_idx is not None:
            record_name = candidate
            break

    if record_name is None:
        return None

    try:
        record = wfdb.rdrecord(record_name)
    except Exception:
        return None

    sig_names = [name.upper() for name in record.sig_name]
    ppg_idx = pick_signal_index(sig_names, PPG_NAMES)
    abp_idx = pick_signal_index(sig_names, ABP_NAMES)
    ecg_idx = pick_signal_index(sig_names, ECG_NAMES)
    if ppg_idx is None or abp_idx is None or ecg_idx is None:
        return None

    fs = float(record.fs)
    ppg_raw = record.p_signal[:, ppg_idx].astype(np.float32)
    abp_raw = record.p_signal[:, abp_idx].astype(np.float32)
    ecg_raw = record.p_signal[:, ecg_idx].astype(np.float32)

    valid = np.isfinite(ppg_raw) & np.isfinite(abp_raw) & np.isfinite(ecg_raw)
    ppg_raw = ppg_raw[valid]
    abp_raw = abp_raw[valid]
    ecg_raw = ecg_raw[valid]

    if len(ppg_raw) < int(fs * args.window_sec * 2):
        return None

    try:
        ppg_filt = bandpass(ppg_raw, fs, low=0.5, high=8.0)
        ecg_filt = bandpass(ecg_raw, fs, low=0.5, high=min(30.0, fs / 2.0 - 1.0))
    except Exception:
        return None

    win_samples = int(fs * args.window_sec)
    target_samples = int(args.window_sec * args.target_fs)
    n_windows = len(ppg_filt) // win_samples

    ppg_list, vpg_list, apg_list, ecg_list = [], [], [], []
    sbp_list, dbp_list = [], []

    for w in range(n_windows):
        start = w * win_samples
        end = start + win_samples

        ppg_win = ppg_filt[start:end]
        abp_win = abp_raw[start:end]
        ecg_win = ecg_filt[start:end]

        if not np.all(np.isfinite(ppg_win)) or not np.all(np.isfinite(abp_win)) or not np.all(np.isfinite(ecg_win)):
            continue
        if is_flat(ppg_win) or is_flat(abp_win) or is_flat(ecg_win):
            continue

        sbp, dbp = extract_bp(abp_win, fs, args.hr_max)
        if not (args.sbp_min <= sbp <= args.sbp_max):
            continue
        if not (args.dbp_min <= dbp <= args.dbp_max):
            continue
        if (sbp - dbp) < args.min_pp:
            continue

        hr = estimate_hr(ppg_win, fs, args.hr_max)
        if hr is None or not (args.hr_min <= hr <= args.hr_max):
            continue

        ppg_res = resample(ppg_win, target_samples).astype(np.float32)
        ecg_res = resample(ecg_win, target_samples).astype(np.float32)
        vpg_res, apg_res = compute_vpg_apg(ppg_res)

        ppg_list.append(ppg_res)
        vpg_list.append(vpg_res)
        apg_list.append(apg_res)
        ecg_list.append(ecg_res)
        sbp_list.append(sbp)
        dbp_list.append(dbp)

    if len(ppg_list) < args.min_windows:
        return None

    sbp_arr = np.asarray(sbp_list, dtype=np.float32)
    dbp_arr = np.asarray(dbp_list, dtype=np.float32)
    ref_sbp = float(sbp_arr[0])
    ref_dbp = float(dbp_arr[0])

    keep_mask = (
        (np.abs(sbp_arr - ref_sbp) <= args.per_patient_bp_cap) &
        (np.abs(dbp_arr - ref_dbp) <= args.per_patient_bp_cap)
    )

    n_before_cap = len(ppg_list)
    ppg_list = [ppg_list[i] for i in range(n_before_cap) if keep_mask[i]]
    vpg_list = [vpg_list[i] for i in range(n_before_cap) if keep_mask[i]]
    apg_list = [apg_list[i] for i in range(n_before_cap) if keep_mask[i]]
    ecg_list = [ecg_list[i] for i in range(n_before_cap) if keep_mask[i]]
    sbp_list = [sbp_list[i] for i in range(n_before_cap) if keep_mask[i]]
    dbp_list = [dbp_list[i] for i in range(n_before_cap) if keep_mask[i]]

    if len(ppg_list) < args.min_windows:
        return None

    if len(ppg_list) > args.max_windows_per_patient:
        patient_seed = args.seed + int(patient_dir.name)
        rng = np.random.default_rng(patient_seed)
        keep_idx = np.sort(rng.choice(len(ppg_list), args.max_windows_per_patient, replace=False))
        ppg_list = [ppg_list[i] for i in keep_idx]
        vpg_list = [vpg_list[i] for i in keep_idx]
        apg_list = [apg_list[i] for i in keep_idx]
        ecg_list = [ecg_list[i] for i in keep_idx]
        sbp_list = [sbp_list[i] for i in keep_idx]
        dbp_list = [dbp_list[i] for i in keep_idx]

    return {
        "ppg": np.asarray(ppg_list, dtype=np.float32),
        "vpg": np.asarray(vpg_list, dtype=np.float32),
        "apg": np.asarray(apg_list, dtype=np.float32),
        "ecg": np.asarray(ecg_list, dtype=np.float32),
        "sbp": np.asarray(sbp_list, dtype=np.float32),
        "dbp": np.asarray(dbp_list, dtype=np.float32),
        "n_before_cap": n_before_cap,
        "n_after_cap": len(ppg_list),
    }


def main():
    args = parse_args()
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    target_samples = int(args.window_sec * args.target_fs)
    print("=" * 72)
    print("MIMIC-II ECG+PPG PREPROCESSING")
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Window: {args.window_sec}s @ {args.target_fs}Hz = {target_samples} samples")
    print(f"BP range: SBP [{args.sbp_min},{args.sbp_max}]  DBP [{args.dbp_min},{args.dbp_max}]")
    print(f"Per-patient BP cap: +/-{args.per_patient_bp_cap} mmHg from first valid window")
    print(f"Min windows/patient: {args.min_windows}")
    print("=" * 72)

    patient_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()])
    if args.max_patients > 0:
        patient_dirs = patient_dirs[:args.max_patients]
    print(f"Found {len(patient_dirs)} patient directories\n")

    all_patients = []
    total_before_cap = 0
    total_dropped_by_cap = 0

    for patient_dir in patient_dirs:
        result = process_patient(patient_dir, args)
        if result is None:
            print(f"  [---] {patient_dir.name:20s}  skipped")
            continue

        result["name"] = patient_dir.name
        all_patients.append(result)
        n_windows = len(result["sbp"])
        sbp_med = float(np.median(result["sbp"]))
        dbp_med = float(np.median(result["dbp"]))
        dropped = result["n_before_cap"] - result["n_after_cap"]
        total_before_cap += result["n_before_cap"]
        total_dropped_by_cap += dropped
        print(
            f"  [{len(all_patients):3d}] {patient_dir.name:20s}  {n_windows:5d} windows  "
            f"SBP={sbp_med:.0f}  DBP={dbp_med:.0f}  (cap dropped {dropped})"
        )

    n_patients = len(all_patients)
    print(f"\n{'=' * 72}")
    print(f"Patients accepted: {n_patients} / {len(patient_dirs)}")
    if total_before_cap > 0:
        pct = 100.0 * total_dropped_by_cap / total_before_cap
        print(
            f"Windows removed by +/-{args.per_patient_bp_cap}mmHg cap: "
            f"{total_dropped_by_cap:,} / {total_before_cap:,} ({pct:.1f}%)"
        )

    if n_patients < 10:
        print("ERROR: Too few accepted patients.")
        return

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_patients)
    n_train = int(args.train_frac * n_patients)
    n_val = int(args.val_frac * n_patients)
    train_pids = set(perm[:n_train].tolist())
    val_pids = set(perm[n_train:n_train + n_val].tolist())
    test_pids = set(perm[n_train + n_val:].tolist())

    print(f"\nSplit: train={len(train_pids)}  val={len(val_pids)}  test={len(test_pids)} patients")

    all_ppg, all_vpg, all_apg, all_ecg = [], [], [], []
    all_sbp, all_dbp = [], []
    all_pids = []
    train_mask, val_mask, test_mask = [], [], []
    patient_names = []

    for pid, patient in enumerate(all_patients):
        n = len(patient["sbp"])
        all_ppg.append(patient["ppg"])
        all_vpg.append(patient["vpg"])
        all_apg.append(patient["apg"])
        all_ecg.append(patient["ecg"])
        all_sbp.append(patient["sbp"])
        all_dbp.append(patient["dbp"])
        all_pids.extend([pid] * n)
        patient_names.append(patient["name"])

        in_train = pid in train_pids
        in_val = pid in val_pids
        in_test = pid in test_pids
        train_mask.extend([in_train] * n)
        val_mask.extend([in_val] * n)
        test_mask.extend([in_test] * n)

    X_ppg = np.concatenate(all_ppg, axis=0)
    X_vpg = np.concatenate(all_vpg, axis=0)
    X_apg = np.concatenate(all_apg, axis=0)
    X_ecg = np.concatenate(all_ecg, axis=0)
    y_sbp = np.concatenate(all_sbp, axis=0)
    y_dbp = np.concatenate(all_dbp, axis=0)
    patient_ids = np.asarray(all_pids, dtype=np.int64)
    train_mask = np.asarray(train_mask, dtype=bool)
    val_mask = np.asarray(val_mask, dtype=bool)
    test_mask = np.asarray(test_mask, dtype=bool)

    print(f"\nTotal windows: {len(y_sbp):,}")
    print(f"  Train: {train_mask.sum():,}")
    print(f"  Val:   {val_mask.sum():,}")
    print(f"  Test:  {test_mask.sum():,}")
    print(
        f"\nSBP: mean={y_sbp.mean():.1f}  std={y_sbp.std():.1f}  "
        f"min={y_sbp.min():.1f}  max={y_sbp.max():.1f}"
    )
    print(
        f"DBP: mean={y_dbp.mean():.1f}  std={y_dbp.std():.1f}  "
        f"min={y_dbp.min():.1f}  max={y_dbp.max():.1f}"
    )

    np.save(output_path / "X_ppg_windows.npy", X_ppg)
    np.save(output_path / "X_vpg.npy", X_vpg)
    np.save(output_path / "X_apg.npy", X_apg)
    np.save(output_path / "X_ecg.npy", X_ecg)
    np.save(output_path / "y_sbp.npy", y_sbp)
    np.save(output_path / "y_dbp.npy", y_dbp)
    np.save(output_path / "patient_ids.npy", patient_ids)
    np.save(output_path / "train_mask.npy", train_mask)
    np.save(output_path / "val_mask.npy", val_mask)
    np.save(output_path / "test_mask.npy", test_mask)

    with open(output_path / "patient_names.txt", "w", encoding="utf-8") as handle:
        for name in patient_names:
            handle.write(name + "\n")

    print(f"\nOK Saved ECG+PPG dataset to {args.output_dir}")
    print("Next step:")
    print(
        f"  python train.py --data_dir \"{args.output_dir}\" --save_dir .\\checkpoints_v27_ecgppg "
        "--use_fusion --use_beat_features --use_ecg --use_gated_pair_interaction "
        "--no_mixup --sbp_loss_type huber --no_balanced_sampling"
    )


if __name__ == "__main__":
    main()
