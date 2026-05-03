"""
MIMIC-II PPG Preprocessing - v3 (Schlesinger-inspired)
======================================================
Based on preprocess_mimic2_v2.py but with Schlesinger et al. (2020) ideas:

  1. LONGER WINDOWS (10s instead of 4s) — more beats per window, more
     stable BP reference, richer spectrogram content.
  2. ±40 mmHg PER-PATIENT OUTLIER CAP — remove windows whose SBP/DBP
     differs by more than 40 mmHg from the patient's first valid window.
     This is the single biggest change in Schlesinger's preprocessing:
     it prunes rare extreme values that are hard to predict and inflate
     SBP MAE.
  3. STRICTER PATIENT FILTER (min 100 windows, not 50).

Everything else preserved: SBP/DBP range, HR filter, bandpass, VPG/APG
derivatives, patient-level split, ±40 windows/patient cap.

Output: C:\\MIMIC2_out_v9 (new dir, won't overwrite v8)
New shape: X_ppg_windows (N, 1000) float32   [was (N, 800)]
"""

import warnings
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import butter, filtfilt, resample, find_peaks

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

INPUT_DIR  = r"C:\MIMIC 2"
OUTPUT_DIR = r"C:\MIMIC2_out_v9"

# Window settings — SCHLESINGER-INSPIRED
WINDOW_SEC    = 10.0      # was 4.0 — longer windows, more beats, more stable BP
TARGET_FS     = 100       # was 200 — 100Hz matches MIMIC-II 125Hz source better
TARGET_SAMPLES = int(WINDOW_SEC * TARGET_FS)  # 1000 samples

# BP range (Schlesinger et al. 2020)
SBP_MIN = 75
SBP_MAX = 165
DBP_MIN = 40
DBP_MAX = 85
MIN_PP  = 15   # minimum pulse pressure

# HR range
HR_MIN = 50
HR_MAX = 140

# Minimum windows to keep a patient (Schlesinger used 100)
MIN_WINDOWS = 100

# Maximum windows to keep per patient
MAX_WINDOWS_PER_PATIENT = 500

# ★ NEW: Per-patient BP outlier cap (Schlesinger's key idea)
# Remove windows whose BP differs by more than this from patient's first window.
# Schlesinger used ±40 mmHg.
PER_PATIENT_BP_CAP = 40.0

# Dataset split (by patient)
TRAIN_FRAC = 0.60
VAL_FRAC   = 0.20
SPLIT_SEED = 42

# ============================================================
# SIGNAL PROCESSING HELPERS (unchanged from v2)
# ============================================================

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


def extract_bp(abp_win, fs):
    min_dist = int(fs * 60 / HR_MAX)
    peaks, _  = find_peaks( abp_win, distance=min_dist, prominence=0.5)
    troughs, _ = find_peaks(-abp_win, distance=min_dist, prominence=0.5)

    if len(peaks) >= 2 and len(troughs) >= 2:
        sbp = float(np.mean(abp_win[peaks]))
        dbp = float(np.mean(abp_win[troughs]))
    else:
        sbp = float(np.percentile(abp_win, 98))
        dbp = float(np.percentile(abp_win,  2))

    return sbp, dbp


def estimate_hr(ppg, fs):
    min_dist = int(fs * 60 / HR_MAX)
    peaks, _ = find_peaks(ppg, distance=min_dist, prominence=0.01)
    if len(peaks) < 2:
        return None
    intervals = np.diff(peaks) / fs
    mean_rr = np.mean(intervals)
    if mean_rr <= 0:
        return None
    hr = 60.0 / mean_rr
    return hr


# ============================================================
# PROCESS ONE PATIENT
# ============================================================

def process_patient(patient_dir):
    patient_dir = Path(patient_dir)

    hea_files = list(patient_dir.glob("*.hea"))
    if not hea_files:
        return None

    waveform = [f for f in hea_files if not f.stem.endswith("n")]

    ppg_record_path = None

    for hf in waveform:
        record_name = str(hf.with_suffix(""))
        try:
            header = wfdb.rdheader(record_name)
            sig_names = [s.upper() for s in header.sig_name]
            has_ppg = any(s in sig_names for s in ["PLETH", "PPG"])
            has_abp = any(s in sig_names for s in ["ABP", "ART", "BP"])
            if has_ppg and has_abp:
                ppg_record_path = record_name
                break
        except Exception:
            continue

    if ppg_record_path is None:
        return None

    try:
        record = wfdb.rdrecord(ppg_record_path)
    except Exception:
        return None

    sig_names = [s.upper() for s in record.sig_name]
    fs = record.fs

    ppg_idx = next((i for i, s in enumerate(sig_names) if s in ["PLETH", "PPG"]), None)
    abp_idx = next((i for i, s in enumerate(sig_names) if s in ["ABP", "ART", "BP"]), None)

    if ppg_idx is None or abp_idx is None:
        return None

    ppg_raw = record.p_signal[:, ppg_idx].astype(np.float32)
    abp_raw = record.p_signal[:, abp_idx].astype(np.float32)

    valid = np.isfinite(ppg_raw) & np.isfinite(abp_raw)
    ppg_raw = ppg_raw[valid]
    abp_raw = abp_raw[valid]

    if len(ppg_raw) < int(fs * WINDOW_SEC * 2):
        return None

    try:
        ppg_filt = bandpass(ppg_raw, fs)
    except Exception:
        return None

    win_samples = int(fs * WINDOW_SEC)
    n_windows = len(ppg_filt) // win_samples

    ppg_list, vpg_list, apg_list = [], [], []
    sbp_list, dbp_list = [], []

    for w in range(n_windows):
        start = w * win_samples
        end   = start + win_samples

        ppg_win = ppg_filt[start:end]
        abp_win = abp_raw[start:end]

        if not np.all(np.isfinite(ppg_win)) or not np.all(np.isfinite(abp_win)):
            continue

        if is_flat(ppg_win) or is_flat(abp_win):
            continue

        sbp, dbp = extract_bp(abp_win, fs)

        if not (SBP_MIN <= sbp <= SBP_MAX):
            continue
        if not (DBP_MIN <= dbp <= DBP_MAX):
            continue
        if (sbp - dbp) < MIN_PP:
            continue

        hr = estimate_hr(ppg_win, fs)
        if hr is None or not (HR_MIN <= hr <= HR_MAX):
            continue

        ppg_res = resample(ppg_win, TARGET_SAMPLES).astype(np.float32)
        vpg, apg = compute_vpg_apg(ppg_res)

        ppg_list.append(ppg_res)
        vpg_list.append(vpg)
        apg_list.append(apg)
        sbp_list.append(sbp)
        dbp_list.append(dbp)

    if len(ppg_list) < MIN_WINDOWS:
        return None

    # ★ SCHLESINGER PER-PATIENT ±40 mmHg OUTLIER CAP ★
    # Reference = patient's first valid window BP
    sbp_arr = np.array(sbp_list)
    dbp_arr = np.array(dbp_list)
    ref_sbp = sbp_arr[0]
    ref_dbp = dbp_arr[0]

    keep_mask = (
        (np.abs(sbp_arr - ref_sbp) <= PER_PATIENT_BP_CAP) &
        (np.abs(dbp_arr - ref_dbp) <= PER_PATIENT_BP_CAP)
    )

    n_before = len(ppg_list)
    ppg_list = [ppg_list[i] for i in range(n_before) if keep_mask[i]]
    vpg_list = [vpg_list[i] for i in range(n_before) if keep_mask[i]]
    apg_list = [apg_list[i] for i in range(n_before) if keep_mask[i]]
    sbp_list = [sbp_list[i] for i in range(n_before) if keep_mask[i]]
    dbp_list = [dbp_list[i] for i in range(n_before) if keep_mask[i]]

    if len(ppg_list) < MIN_WINDOWS:
        return None

    # Cap windows per patient
    if len(ppg_list) > MAX_WINDOWS_PER_PATIENT:
        rng = np.random.default_rng()
        idx = rng.choice(len(ppg_list), MAX_WINDOWS_PER_PATIENT, replace=False)
        idx = np.sort(idx)
        ppg_list = [ppg_list[i] for i in idx]
        vpg_list = [vpg_list[i] for i in idx]
        apg_list = [apg_list[i] for i in idx]
        sbp_list = [sbp_list[i] for i in idx]
        dbp_list = [dbp_list[i] for i in idx]

    return {
        "ppg": np.array(ppg_list, dtype=np.float32),
        "vpg": np.array(vpg_list, dtype=np.float32),
        "apg": np.array(apg_list, dtype=np.float32),
        "sbp": np.array(sbp_list, dtype=np.float32),
        "dbp": np.array(dbp_list, dtype=np.float32),
        "n_before_cap": n_before,
        "n_after_cap": len(ppg_list) if len(ppg_list) <= MAX_WINDOWS_PER_PATIENT else MAX_WINDOWS_PER_PATIENT,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    input_path  = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MIMIC-II PPG PREPROCESSING - v3 (Schlesinger-inspired)")
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Window: {WINDOW_SEC}s @ {TARGET_FS}Hz = {TARGET_SAMPLES} samples")
    print(f"BP range: SBP [{SBP_MIN},{SBP_MAX}]  DBP [{DBP_MIN},{DBP_MAX}]")
    print(f"Per-patient BP cap: ±{PER_PATIENT_BP_CAP} mmHg from first window")
    print(f"Min windows/patient: {MIN_WINDOWS}")
    print("=" * 60)

    patient_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()])
    print(f"Found {len(patient_dirs)} patient directories\n")

    all_patients = []
    total_dropped_by_cap = 0
    total_before_cap = 0

    for i, p_dir in enumerate(patient_dirs):
        result = process_patient(p_dir)
        if result is not None:
            result["name"] = p_dir.name
            all_patients.append(result)
            n = len(result["sbp"])
            sbp_med = float(np.median(result["sbp"]))
            dbp_med = float(np.median(result["dbp"]))
            dropped = result["n_before_cap"] - result["n_after_cap"]
            total_dropped_by_cap += dropped
            total_before_cap += result["n_before_cap"]
            print(f"  [{len(all_patients):3d}] {p_dir.name:20s}  {n:5d} windows  "
                  f"SBP={sbp_med:.0f}  DBP={dbp_med:.0f}  (cap dropped {dropped})")
        else:
            print(f"  [---] {p_dir.name:20s}  skipped")

    n_patients = len(all_patients)
    print(f"\n{'='*60}")
    print(f"Patients accepted: {n_patients} / {len(patient_dirs)}")
    if total_before_cap > 0:
        print(f"Windows removed by ±{PER_PATIENT_BP_CAP}mmHg cap: "
              f"{total_dropped_by_cap:,} / {total_before_cap:,} "
              f"({100*total_dropped_by_cap/total_before_cap:.1f}%)")

    if n_patients < 10:
        print("ERROR: Too few patients.")
        return

    # Split by patient
    rng = np.random.default_rng(SPLIT_SEED)
    idx = rng.permutation(n_patients)

    n_train = int(TRAIN_FRAC * n_patients)
    n_val   = int(VAL_FRAC   * n_patients)

    train_pids = set(idx[:n_train].tolist())
    val_pids   = set(idx[n_train:n_train + n_val].tolist())
    test_pids  = set(idx[n_train + n_val:].tolist())

    print(f"\nSplit: train={len(train_pids)}  val={len(val_pids)}  test={len(test_pids)} patients")

    # Assemble flat arrays
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

        in_train = pid in train_pids
        in_val   = pid in val_pids
        in_test  = pid in test_pids

        train_mask.extend([in_train] * n)
        val_mask.extend([in_val]   * n)
        test_mask.extend([in_test]  * n)

    X_ppg = np.concatenate(all_ppg, axis=0)
    X_vpg = np.concatenate(all_vpg, axis=0)
    X_apg = np.concatenate(all_apg, axis=0)
    y_sbp = np.concatenate(all_sbp, axis=0)
    y_dbp = np.concatenate(all_dbp, axis=0)
    pids  = np.array(all_pids, dtype=np.int64)
    train_mask = np.array(train_mask, dtype=bool)
    val_mask   = np.array(val_mask,   dtype=bool)
    test_mask  = np.array(test_mask,  dtype=bool)

    N = len(y_sbp)
    print(f"\nTotal windows: {N:,}")
    print(f"  Train: {train_mask.sum():,}")
    print(f"  Val:   {val_mask.sum():,}")
    print(f"  Test:  {test_mask.sum():,}")
    print(f"\nSBP: mean={y_sbp.mean():.1f}  std={y_sbp.std():.1f}  "
          f"min={y_sbp.min():.1f}  max={y_sbp.max():.1f}")
    print(f"DBP: mean={y_dbp.mean():.1f}  std={y_dbp.std():.1f}  "
          f"min={y_dbp.min():.1f}  max={y_dbp.max():.1f}")

    # Save
    np.save(output_path / "X_ppg_windows.npy", X_ppg)
    np.save(output_path / "X_vpg.npy",         X_vpg)
    np.save(output_path / "X_apg.npy",         X_apg)
    np.save(output_path / "y_sbp.npy",         y_sbp)
    np.save(output_path / "y_dbp.npy",         y_dbp)
    np.save(output_path / "patient_ids.npy",   pids)
    np.save(output_path / "train_mask.npy",    train_mask)
    np.save(output_path / "val_mask.npy",      val_mask)
    np.save(output_path / "test_mask.npy",     test_mask)

    with open(output_path / "patient_names.txt", "w") as f:
        for name in patient_names:
            f.write(name + "\n")

    print(f"\nOK Saved to {OUTPUT_DIR}")
    print(f"\nNext steps:")
    print(f"  1. Spectrograms and handcrafted features will auto-regenerate on first train")
    print(f"  2. Train with your v13 recipe but pointing at new data:")
    print(f"     python train.py --data_dir \"{OUTPUT_DIR}\" --save_dir .\\checkpoints_v17 \\")
    print(f"       --use_fusion --epochs 100 --patience 25 --lr 3e-4 --num_anchors 5 \\")
    print(f"       --quality_threshold 0.4 --shuffle_eval --balanced_sampling --workers 0")


if __name__ == "__main__":
    main()
