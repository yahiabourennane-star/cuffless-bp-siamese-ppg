"""
check_signals.py
----------------
Quickly plots a few PPG/VPG/APG windows to verify signals look correct.
Run from C:\siamese network

Usage: python check_signals.py --data_dir "C:\MIMIC2_out_v7"
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default=r"C:\MIMIC2_out_v7")
args = parser.parse_args()

import os
p = lambda f: os.path.join(args.data_dir, f)

ppg = np.load(p("X_ppg_windows.npy"), mmap_mode="r")
vpg = np.load(p("X_vpg.npy"),         mmap_mode="r")
apg = np.load(p("X_apg.npy"),         mmap_mode="r")
y_sbp = np.load(p("y_sbp.npy"))
y_dbp = np.load(p("y_dbp.npy"))

# Pick 4 windows with varied SBP
indices = [0, len(ppg)//4, len(ppg)//2, 3*len(ppg)//4]

fig, axes = plt.subplots(4, 3, figsize=(14, 10))
fig.suptitle("PPG Signal Quality Check", fontsize=13, fontweight="bold")

for row, idx in enumerate(indices):
    sbp = y_sbp[idx]
    dbp = y_dbp[idx]
    t   = np.arange(ppg.shape[1]) / 125.0  # assume 125 Hz

    axes[row, 0].plot(t, ppg[idx], color="#c17a5b", linewidth=0.8)
    axes[row, 0].set_title(f"PPG  (SBP={sbp:.0f}, DBP={dbp:.0f} mmHg)", fontsize=9)
    axes[row, 0].set_ylabel("Amplitude")

    axes[row, 1].plot(t, vpg[idx], color="#4a7c59", linewidth=0.8)
    axes[row, 1].set_title("VPG (1st derivative)", fontsize=9)

    axes[row, 2].plot(t, apg[idx], color="#5b7ac1", linewidth=0.8)
    axes[row, 2].set_title("APG (2nd derivative)", fontsize=9)

    for ax in axes[row]:
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("signal_check.png", dpi=120, bbox_inches="tight")
print("Saved: signal_check.png  — open this file to check your signals look like real PPG")

# Also print basic stats
print(f"\nPPG value range: {ppg.min():.3f} to {ppg.max():.3f}")
print(f"VPG value range: {vpg.min():.3f} to {vpg.max():.3f}")
print(f"APG value range: {apg.min():.3f} to {apg.max():.3f}")
print(f"\nSample PPG window[0] stats: mean={ppg[0].mean():.3f}  std={ppg[0].std():.3f}  min={ppg[0].min():.3f}  max={ppg[0].max():.3f}")
