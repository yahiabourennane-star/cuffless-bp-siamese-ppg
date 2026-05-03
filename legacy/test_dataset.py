"""
Quick test of SiameseMIMICDataset
"""
import numpy as np
from siamese_dataset_v2 import SiameseMIMICDataset

print("=" * 70)
print("TESTING SIAMESE DATASET")
print("=" * 70)

# Load data
DATA_DIR = r"C:\MIMIC2_out_improved"

print("\nLoading data files...")
X_ppg = np.load(f"{DATA_DIR}/X_ppg_windows.npy")
X_vpg = np.load(f"{DATA_DIR}/X_vpg.npy")
X_apg = np.load(f"{DATA_DIR}/X_apg.npy")
y_sbp = np.load(f"{DATA_DIR}/y_sbp.npy")
y_dbp = np.load(f"{DATA_DIR}/y_dbp.npy")
patient_ids = np.load(f"{DATA_DIR}/patient_ids.npy")
anchor_indices = np.load(f"{DATA_DIR}/anchor_indices.npy")
train_mask = np.load(f"{DATA_DIR}/train_mask.npy")

print("Loaded data:")
print(f"  Total windows: {len(X_ppg):,}")
print(f"  Train windows: {train_mask.sum():,}")
print(f"  Unique patients: {len(anchor_indices)}")

# ==============================================================
# Test 1: Fixed anchor (RECOMMENDED)
# ==============================================================

print("\n" + "=" * 70)
print("TEST 1: Fixed Anchor Dataset (RECOMMENDED)")
print("=" * 70)

dataset_fixed = SiameseMIMICDataset(
    X_ppg, X_vpg, X_apg, y_sbp, y_dbp,
    patient_ids, anchor_indices, train_mask,
    use_delta=True,
    random_anchor=False,   # Fixed anchor
    normalize_signals=True,
    return_meta=True       # Explicit for clarity
)

print(f"\nDataset created: {len(dataset_fixed):,} windows")

# Get a sample (FIXED UNPACKING)
(x_curr, x_anchor), (target_sbp, target_dbp), meta = dataset_fixed[0]

print(f"\nSample batch:")
print(f"  Current shape:  {x_curr.shape}")
print(f"  Anchor shape:   {x_anchor.shape}")
print(f"  SBP delta:      {target_sbp.item():.1f} mmHg")
print(f"  DBP delta:      {target_dbp.item():.1f} mmHg")
print(f"  Meta:           {meta}")

# ==============================================================
# Test 2: Random anchor (NOT recommended)
# ==============================================================

print("\n" + "=" * 70)
print("TEST 2: Random Anchor Dataset (NOT RECOMMENDED)")
print("=" * 70)

dataset_random = SiameseMIMICDataset(
    X_ppg, X_vpg, X_apg, y_sbp, y_dbp,
    patient_ids, anchor_indices, train_mask,
    use_delta=True,
    random_anchor=True,   # Random anchor
    seed=42,
    normalize_signals=True,
    return_meta=True
)

print(f"\nDataset created: {len(dataset_random):,} windows")

print("\n" + "=" * 70)
print("Both dataset versions loaded.")
print("=" * 70)
print("\nFor the main runs, use:")
print("  random_anchor=False  - fixed best-quality anchor")
print("\nOnly use for comparison/debugging:")
print("  random_anchor=True   - random anchor selection")
