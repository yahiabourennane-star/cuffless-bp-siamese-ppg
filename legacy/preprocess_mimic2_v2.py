"""
MIMIC-II PPG Preprocessing - v7 (clean / simple)
==================================================
Reads WFDB .dat/.hea files and produces a Siamese-ready dataset.

Quality checks (kept simple):
  - HR in [50, 140] bpm
  - SBP in [75, 165] mmHg  (Schlesinger 2020 range)
  - DBP in [40,  85] mmHg
  - Pulse pressure >= 10 mmHg
  - Signal not flat, no NaN/Inf
  - Min 50 windows per patient

Output files (all saved to OUTPUT_DIR):
  X_ppg_windows.npy  [N, 800]  float32
  X_vpg.npy          [N, 800]  float32
  X_apg.npy          [N, 800]  float32
  y_sbp.npy          [N]       float32
  y_dbp.npy          [N]       float32
  patient_ids.npy    [N]       int64   (0 .. P-1)
  train_mask.npy     [N]       bool
  val_mask.npy       [N]       bool
  test_mask.npy      [N]       bool
  patient_names.txt            one name per line

Requirements:
  pip install wfdb scipy numpy
"""

import warnings
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import butter, filtfilt, resample, find_peaks

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG — edit these
# ============================================================

INPUT_DIR  = r"C:\MIMIC 2"
OUTPUT_DIR = r"C:\MIMIC2_out_v8"

# Window settings
WINDOW_SEC    = 4.0       # seconds per window
TARGET_FS     = 200       # resample to this
TARGET_SAMPLES = int(WINDOW_SEC * TARGET_FS)  # 800 samples

# BP range (Schlesinger et al. 2020)
SBP_MIN = 75
SBP_MAX = 165
DBP_MIN = 40
DBP_MAX = 85
MIN_PP  = 15   # minimum pulse pressure (SBP - DBP); below 15 suggests damped arterial line

# HR range
HR_MIN = 50
HR_MAX = 140

# Minimum windows to keep a patient
MIN_WINDOWS = 50

# Maximum windows to keep per patient (keeps dataset balanced + memory safe)
MAX_WINDOWS_PER_PATIENT = 500

# Dataset split (by patient)
TRAIN_FRAC = 0.60
VAL_FRAC   = 0.20
# TEST_FRAC  = remaining (~0.20)
SPLIT_SEED = 42

# ============================================================
# SIGNAL PROCESSING HELPERS
# ============================================================

def bandpass(signal, fs, low=0.5, high=8.0, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def compute_vpg_apg(ppg):
    """First and second derivatives (VPG, APG).
    A Hann window tapers the edges before differentiation to prevent
    boundary spikes caused by signal discontinuities at window cuts.
    """
    taper = np.hanning(len(ppg))
    ppg_tapered = ppg * taper
    vpg = np.gradient(ppg_tapered)
    apg = np.gradient(vpg)
    return vpg.astype(np.float32), apg.astype(np.float32)


def is_flat(sig, threshold=1e-4):
    """True if signal has near-zero variance (flat/artifact)."""
    return np.std(sig) < threshold


def extract_bp(abp_win, fs):
    """Extract SBP and DBP from ABP window using peak/trough detection.
    Falls back to percentiles if peak detection finds too few beats.
    """
    min_dist = int(fs * 60 / HR_MAX)
    peaks, _  = find_peaks( abp_win, distance=min_dist, prominence=0.5)
    troughs, _ = find_peaks(-abp_win, distance=min_dist, prominence=0.5)

    if len(peaks) >= 2 and len(troughs) >= 2:
        sbp = float(np.mean(abp_win[peaks]))
        dbp = float(np.mean(abp_win[troughs]))
    else:
        # Fallback to robust percentiles
        sbp = float(np.percentile(abp_win, 98))
        dbp = float(np.percentile(abp_win,  2))

    return sbp, dbp


def estimate_hr(ppg, fs):
    """Estimate heart rate from PPG peaks. Returns HR in bpm or None."""
    min_dist = int(fs * 60 / HR_MAX)
    peaks, _ = find_peaks(ppg, distance=min_dist, prominence=0.01)
    if len(peaks) < 2:
        return None
    intervals = np.diff(peaks) / fs  # seconds between peaks
    mean_rr = np.mean(intervals)
    if mean_rr <= 0:
        return None
    hr = 60.0 / mean_rr
    return hr


# ============================================================
# PROCESS ONE PATIENT
# ============================================================

def process_patient(patient_dir):
    """
    Returns a dict with keys: ppg, vpg, apg, sbp, dbp
    Each is a list of 1-D arrays (one per accepted window).
    Returns None if the patient has no usable data.
    """
    patient_dir = Path(patient_dir)
    
    # Find the WFDB record — look for a .hea file
    hea_files = list(patient_dir.glob("*.hea"))
    if not hea_files:
        return None

    # We want the numerics record (contains ABP) — usually ends in 'n'
    # Try numerics first, then fall back to any record
    numerics = [f for f in hea_files if f.stem.endswith("n")]
    waveform = [f for f in hea_files if not f.stem.endswith("n")]

    ppg_record_path = None
    abp_record_path = None

    # Waveform record has PPG + ABP
    for hf in waveform:
        record_name = str(hf.with_suffix(""))
        try:
            header = wfdb.rdheader(record_name)
            sig_names = [s.upper() for s in header.sig_name]
            has_ppg = any(s in sig_names for s in ["PLETH", "PPG"])
            has_abp = any(s in sig_names for s in ["ABP", "ART", "BP"])
            if has_ppg and has_abp:
                ppg_record_path = record_name
                abp_record_path = record_name
                break
        except Exception:
            continue

    if ppg_record_path is None:
        return None

    # Load record
    try:
        record = wfdb.rdrecord(ppg_record_path)
    except Exception:
        return None

    sig_names = [s.upper() for s in record.sig_name]
    fs = record.fs

    # Find PPG and ABP channel indices
    ppg_idx = next((i for i, s in enumerate(sig_names) if s in ["PLETH", "PPG"]), None)
    abp_idx = next((i for i, s in enumerate(sig_names) if s in ["ABP", "ART", "BP"]), None)

    if ppg_idx is None or abp_idx is None:
        return None

    ppg_raw = record.p_signal[:, ppg_idx].astype(np.float32)
    abp_raw = record.p_signal[:, abp_idx].astype(np.float32)

    # Remove NaN
    valid = np.isfinite(ppg_raw) & np.isfinite(abp_raw)
    ppg_raw = ppg_raw[valid]
    abp_raw = abp_raw[valid]

    if len(ppg_raw) < int(fs * WINDOW_SEC * 2):
        return None  # need at least 2 windows worth of data

    # Filter PPG
    try:
        ppg_filt = bandpass(ppg_raw, fs)
    except Exception:
        return None

    # Segment into non-overlapping windows
    win_samples = int(fs * WINDOW_SEC)
    n_windows = len(ppg_filt) // win_samples

    ppg_list, vpg_list, apg_list = [], [], []
    sbp_list, dbp_list = [], []

    for w in range(n_windows):
        start = w * win_samples
        end   = start + win_samples

        ppg_win = ppg_filt[start:end]
        abp_win = abp_raw[start:end]

        # Skip if any NaN/Inf
        if not np.all(np.isfinite(ppg_win)) or not np.all(np.isfinite(abp_win)):
            continue

        # Skip flat signals
        if is_flat(ppg_win) or is_flat(abp_win):
            continue

        # Extract BP from ABP using peak/trough detection
        sbp, dbp = extract_bp(abp_win, fs)

        # BP range check
        if not (SBP_MIN <= sbp <= SBP_MAX):
            continue
        if not (DBP_MIN <= dbp <= DBP_MAX):
            continue
        if (sbp - dbp) < MIN_PP:
            continue

        # HR check
        hr = estimate_hr(ppg_win, fs)
        if hr is None or not (HR_MIN <= hr <= HR_MAX):
            continue

        # Resample to TARGET_FS
        ppg_res = resample(ppg_win, TARGET_SAMPLES).astype(np.float32)

        # Compute VPG and APG
        vpg, apg = compute_vpg_apg(ppg_res)

        ppg_list.append(ppg_res)
        vpg_list.append(vpg)
        apg_list.append(apg)
        sbp_list.append(sbp)
        dbp_list.append(dbp)

    if len(ppg_list) < MIN_WINDOWS:
        return None

    # Cap windows per patient — keeps dataset balanced and memory safe
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
    }


# ============================================================
# MAIN
# ============================================================

def main():
    input_path  = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MIMIC-II PPG PREPROCESSING - v7")
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"BP range: SBP [{SBP_MIN},{SBP_MAX}]  DBP [{DBP_MIN},{DBP_MAX}]")
    print(f"Min windows per patient: {MIN_WINDOWS}")
    print("=" * 60)

    patient_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()])
    print(f"Found {len(patient_dirs)} patient directories\n")

    all_patients = []

    for i, p_dir in enumerate(patient_dirs):
        result = process_patient(p_dir)
        if result is not None:
            result["name"] = p_dir.name
            all_patients.append(result)
            n = len(result["sbp"])
            sbp_med = float(np.median(result["sbp"]))
            dbp_med = float(np.median(result["dbp"]))
            print(f"  [{len(all_patients):3d}] {p_dir.name:20s}  {n:5d} windows  "
                  f"SBP={sbp_med:.0f}  DBP={dbp_med:.0f}")
        else:
            print(f"  [---] {p_dir.name:20s}  skipped")

    n_patients = len(all_patients)
    print(f"\n{'='*60}")
    print(f"Patients accepted: {n_patients} / {len(patient_dirs)}")

    if n_patients < 10:
        print("ERROR: Too few patients. Check INPUT_DIR or lower MIN_WINDOWS.")
        return

    # ── Split by patient ──────────────────────────────────────
    rng = np.random.default_rng(SPLIT_SEED)
    idx = rng.permutation(n_patients)

    n_train = int(TRAIN_FRAC * n_patients)
    n_val   = int(VAL_FRAC   * n_patients)

    train_pids = set(idx[:n_train].tolist())
    val_pids   = set(idx[n_train:n_train + n_val].tolist())
    test_pids  = set(idx[n_train + n_val:].tolist())

    print(f"\nSplit: train={len(train_pids)}  val={len(val_pids)}  test={len(test_pids)} patients")

    # ── Assemble flat arrays ───────────────────────────────────
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

    # ── Save ──────────────────────────────────────────────────
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

    print(f"\nSaved to {OUTPUT_DIR}")
    print(f"\nTrain with:")
    print(f'  python train_v2.py --data_dir "{OUTPUT_DIR}" '
          f'--output_dir "C:\\training_outputs\\run_v7_seed42" '
          f'--model_base 32 --epochs 30 --lr 3e-4 --dropout 0.20 '
          f'--weight_decay 2e-4 --grad_clip 5.0 --patience 10 '
          f'--min_delta 0.02 --huber_delta 2.0 --sbp_weight 1.2 '
          f'--cosine_tmax 20 --max_windows_per_patient 500 '
          f'--max_val_windows_per_patient 500 --no_augment '
          f'--train_fixed_anchor --max_delta_sbp 0 --seed 42 --num_workers 0')


if __name__ == "__main__":
    main()
