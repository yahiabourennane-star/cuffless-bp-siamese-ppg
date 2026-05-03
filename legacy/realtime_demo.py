"""
realtime_demo.py
----------------
Real-time PPG signal visualisation with live BP estimation.

Simulates a continuous PPG stream from test-set patients and displays:
  - Scrolling PPG / VPG / APG waveforms
  - Live SBP / DBP predictions updating per window
  - Ground-truth BP for comparison
  - Prediction error bars

Satisfies Stage 4 requirement:
  "A real-time visualisation of the signal together with the analysis outcome"

Usage:
    python realtime_demo.py --checkpoint "./checkpoints_v10/best_model.pt"
    python realtime_demo.py --checkpoint "./checkpoints_v10/best_model.pt" --patient_idx 5
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive — saves to file
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from scipy.signal import stft as scipy_stft

# Project imports
from siamese_cnn import build_model
from dataset import normalise_windows, compute_spectrogram


# ──────────────────────────────────────────────
# Data loading (lightweight — test patient only)
# ──────────────────────────────────────────────

def load_patient_data(data_dir, patient_idx=0, min_sbp_std=5.0):
    """Load data for a single patient for the demo."""
    ppg = np.load(os.path.join(data_dir, "X_ppg_windows.npy"), mmap_mode="r")
    vpg = np.load(os.path.join(data_dir, "X_vpg.npy"), mmap_mode="r")
    apg = np.load(os.path.join(data_dir, "X_apg.npy"), mmap_mode="r")
    y_sbp = np.load(os.path.join(data_dir, "y_sbp.npy"))
    y_dbp = np.load(os.path.join(data_dir, "y_dbp.npy"))
    patient_ids = np.load(os.path.join(data_dir, "patient_ids.npy"))

    # Get unique patients with enough variability
    unique_pids = np.unique(patient_ids)
    valid_pids = []
    for pid in unique_pids:
        mask = patient_ids == pid
        if y_sbp[mask].std() >= min_sbp_std:
            valid_pids.append(pid)

    if patient_idx >= len(valid_pids):
        print(f"Patient index {patient_idx} out of range (0-{len(valid_pids)-1}). Using 0.")
        patient_idx = 0

    pid = valid_pids[patient_idx]
    mask = patient_ids == pid
    indices = np.where(mask)[0]

    print(f"Patient {pid} (index {patient_idx}/{len(valid_pids)-1})")
    print(f"  Windows: {len(indices)}")
    print(f"  SBP range: {y_sbp[indices].min():.0f}-{y_sbp[indices].max():.0f} mmHg")
    print(f"  DBP range: {y_dbp[indices].min():.0f}-{y_dbp[indices].max():.0f} mmHg")

    # Stack and normalise signals for this patient
    ppg_p = normalise_windows(np.array(ppg[indices]))
    vpg_p = normalise_windows(np.array(vpg[indices]))
    apg_p = normalise_windows(np.array(apg[indices]))
    X_patient = np.stack([ppg_p, vpg_p, apg_p], axis=1)  # (N_p, 3, W)

    # Use first window as anchor (closest to median SBP+DBP)
    sbp_vals = y_sbp[indices]
    dbp_vals = y_dbp[indices]
    median_sbp = np.median(sbp_vals)
    median_dbp = np.median(dbp_vals)
    sbp_std = sbp_vals.std() + 1e-8
    dbp_std = dbp_vals.std() + 1e-8
    joint_dist = (np.abs(sbp_vals - median_sbp) / sbp_std +
                  np.abs(dbp_vals - median_dbp) / dbp_std)
    anchor_local = int(np.argmin(joint_dist))

    anchor_sig = X_patient[anchor_local]  # (3, W)
    anchor_sbp = float(sbp_vals[anchor_local])
    anchor_dbp = float(dbp_vals[anchor_local])

    print(f"  Anchor window: {anchor_local} (SBP={anchor_sbp:.0f}, DBP={anchor_dbp:.0f})")

    # Raw PPG for display (unnormalised, just the PPG channel)
    ppg_raw = np.array(ppg[indices]).astype(np.float32)

    # Handcrafted features (load cached)
    hf_path = os.path.join(data_dir, "X_hand_features.npy")
    if os.path.exists(hf_path):
        all_hf = np.load(hf_path, allow_pickle=False)
        hand_features = all_hf[indices]  # (N_p, 8)
        anchor_hf = hand_features[anchor_local]  # (8,)
    else:
        hand_features = None
        anchor_hf = None

    return {
        "X": X_patient,          # (N_p, 3, W) normalised
        "ppg_raw": ppg_raw,      # (N_p, W) raw PPG for display
        "y_sbp": sbp_vals,       # (N_p,)
        "y_dbp": dbp_vals,       # (N_p,)
        "anchor_sig": anchor_sig,
        "anchor_sbp": anchor_sbp,
        "anchor_dbp": anchor_dbp,
        "anchor_hf": anchor_hf,
        "hand_features": hand_features,
        "pid": pid,
    }


# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────

def predict_bp(model, anchor_sig, current_sig, anchor_sbp, anchor_dbp,
               device, target_scale, use_fusion=False,
               anchor_spec=None, current_spec=None,
               anchor_hf=None, current_hf=None):
    """Run single-window inference and return predicted absolute SBP, DBP."""
    model.eval()
    with torch.no_grad():
        anc = torch.tensor(anchor_sig, dtype=torch.float32).unsqueeze(0).to(device)
        cur = torch.tensor(current_sig, dtype=torch.float32).unsqueeze(0).to(device)
        anc_sbp_norm = torch.tensor([anchor_sbp / 100.0], dtype=torch.float32).to(device)
        anc_dbp_norm = torch.tensor([anchor_dbp / 100.0], dtype=torch.float32).to(device)

        if use_fusion and anchor_spec is not None:
            anc_sp = torch.tensor(anchor_spec, dtype=torch.float32).unsqueeze(0).to(device)
            cur_sp = torch.tensor(current_spec, dtype=torch.float32).unsqueeze(0).to(device)
        else:
            anc_sp = cur_sp = None

        # Handcrafted features
        if anchor_hf is not None and current_hf is not None:
            anc_hf = torch.tensor(anchor_hf, dtype=torch.float32).unsqueeze(0).to(device)
            cur_hf = torch.tensor(current_hf, dtype=torch.float32).unsqueeze(0).to(device)
        else:
            anc_hf = torch.zeros(1, 8, dtype=torch.float32).to(device)
            cur_hf = torch.zeros(1, 8, dtype=torch.float32).to(device)

        pred = model(anc, cur, anc_sbp_norm, anc_dbp_norm,
                     anchor_spec=anc_sp, current_spec=cur_sp,
                     anchor_hf=anc_hf, current_hf=cur_hf)
        delta = pred.cpu().numpy()[0] * target_scale  # unscale

    pred_sbp = anchor_sbp + delta[0]
    pred_dbp = anchor_dbp + delta[1]
    return float(pred_sbp), float(pred_dbp)


# ──────────────────────────────────────────────
# Real-time visualisation
# ──────────────────────────────────────────────

def run_demo(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load patient data
    data = load_patient_data(args.data_dir, args.patient_idx)

    # Load model
    use_fusion = args.use_fusion
    model = build_model(embed_dim=256, dropout=0.3, device=device,
                        use_fusion=use_fusion)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print("Model loaded.\n")

    # Precompute spectrograms for anchor if fusion
    anchor_spec = None
    all_specs = None
    if use_fusion:
        print("Computing spectrograms for demo windows...")
        anchor_spec = np.stack([
            compute_spectrogram(data["anchor_sig"][c]) for c in range(3)
        ])  # (3, F, T)
        # Precompute all for speed
        N = len(data["X"])
        sample_spec = compute_spectrogram(data["X"][0, 0])
        F, T = sample_spec.shape
        all_specs = np.zeros((N, 3, F, T), dtype=np.float32)
        for i in range(N):
            for c in range(3):
                all_specs[i, c] = compute_spectrogram(data["X"][i, c])
        print(f"  Done ({N} windows)")

    # ── Setup figure ──────────────────────────
    N_windows = len(data["X"])
    W = data["X"].shape[2]
    fs = 125  # Hz
    window_duration = W / fs  # seconds per window

    # How many windows to show in the scrolling view
    display_windows = min(8, N_windows)
    display_samples = display_windows * W

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("#1a1a2e")
    gs = GridSpec(4, 2, figure=fig, width_ratios=[3, 1],
                  hspace=0.35, wspace=0.3)

    # PPG waveform (top, spanning most of width)
    ax_ppg = fig.add_subplot(gs[0, 0])
    ax_ppg.set_facecolor("#16213e")
    ax_ppg.set_title("PPG Signal (Real-Time)", color="white", fontsize=12, fontweight="bold")
    ax_ppg.set_ylabel("Amplitude", color="white")
    ax_ppg.tick_params(colors="white")
    for spine in ax_ppg.spines.values():
        spine.set_color("#333")

    # VPG waveform
    ax_vpg = fig.add_subplot(gs[1, 0])
    ax_vpg.set_facecolor("#16213e")
    ax_vpg.set_title("VPG (1st Derivative)", color="white", fontsize=12, fontweight="bold")
    ax_vpg.set_ylabel("Amplitude", color="white")
    ax_vpg.tick_params(colors="white")
    for spine in ax_vpg.spines.values():
        spine.set_color("#333")

    # APG waveform
    ax_apg = fig.add_subplot(gs[2, 0])
    ax_apg.set_facecolor("#16213e")
    ax_apg.set_title("APG (2nd Derivative)", color="white", fontsize=12, fontweight="bold")
    ax_apg.set_ylabel("Amplitude", color="white")
    ax_apg.set_xlabel("Time (s)", color="white")
    ax_apg.tick_params(colors="white")
    for spine in ax_apg.spines.values():
        spine.set_color("#333")

    # BP display panel (right column, top)
    ax_bp = fig.add_subplot(gs[0:2, 1])
    ax_bp.set_facecolor("#16213e")
    ax_bp.set_title("Blood Pressure", color="white", fontsize=12, fontweight="bold")
    ax_bp.axis("off")

    # BP history plot (right column, bottom half)
    ax_hist = fig.add_subplot(gs[2:4, 1])
    ax_hist.set_facecolor("#16213e")
    ax_hist.set_title("BP History", color="white", fontsize=10, fontweight="bold")
    ax_hist.set_ylabel("mmHg", color="white")
    ax_hist.set_xlabel("Window", color="white")
    ax_hist.tick_params(colors="white")
    for spine in ax_hist.spines.values():
        spine.set_color("#333")

    # Info panel (bottom left)
    ax_info = fig.add_subplot(gs[3, 0])
    ax_info.set_facecolor("#16213e")
    ax_info.axis("off")

    fig.suptitle(
        f"Real-Time BP Estimation  |  Patient {data['pid']}  |  Siamese CNN + Fusion",
        color="#00d4ff", fontsize=14, fontweight="bold"
    )

    # ── Initialise plot elements ──────────────
    t_axis = np.arange(display_samples) / fs

    line_ppg, = ax_ppg.plot([], [], color="#00ff88", linewidth=0.8)
    line_vpg, = ax_vpg.plot([], [], color="#ff6b6b", linewidth=0.8)
    line_apg, = ax_apg.plot([], [], color="#4ecdc4", linewidth=0.8)

    # BP text elements
    bp_texts = {}

    # History arrays
    hist_sbp_pred, hist_dbp_pred = [], []
    hist_sbp_true, hist_dbp_true = [], []

    line_sbp_pred, = ax_hist.plot([], [], "o-", color="#ff4444", markersize=3, linewidth=1.2, label="SBP pred")
    line_sbp_true, = ax_hist.plot([], [], "s--", color="#ff9999", markersize=2, linewidth=0.8, label="SBP true")
    line_dbp_pred, = ax_hist.plot([], [], "o-", color="#4444ff", markersize=3, linewidth=1.2, label="DBP pred")
    line_dbp_true, = ax_hist.plot([], [], "s--", color="#9999ff", markersize=2, linewidth=0.8, label="DBP true")
    ax_hist.legend(loc="upper left", fontsize=7, facecolor="#16213e",
                   edgecolor="#333", labelcolor="white")

    # ── Animation state ───────────────────────
    state = {"window_idx": 0, "start_time": time.time()}

    def update(frame):
        wi = state["window_idx"]
        if wi >= N_windows:
            wi = 0  # loop
            hist_sbp_pred.clear(); hist_dbp_pred.clear()
            hist_sbp_true.clear(); hist_dbp_true.clear()
            state["window_idx"] = 0

        # Current window signals (normalised for display)
        current_sig = data["X"][wi]  # (3, W)

        # Build scrolling buffer (last display_windows windows)
        start_w = max(0, wi - display_windows + 1)
        end_w = wi + 1
        ppg_buf = data["X"][start_w:end_w, 0, :].flatten()
        vpg_buf = data["X"][start_w:end_w, 1, :].flatten()
        apg_buf = data["X"][start_w:end_w, 2, :].flatten()

        n_samples = len(ppg_buf)
        t = np.arange(n_samples) / fs

        # Update waveforms
        line_ppg.set_data(t, ppg_buf)
        ax_ppg.set_xlim(t[0], t[-1])
        ax_ppg.set_ylim(ppg_buf.min() - 0.3, ppg_buf.max() + 0.3)

        line_vpg.set_data(t, vpg_buf)
        ax_vpg.set_xlim(t[0], t[-1])
        ax_vpg.set_ylim(vpg_buf.min() - 0.3, vpg_buf.max() + 0.3)

        line_apg.set_data(t, apg_buf)
        ax_apg.set_xlim(t[0], t[-1])
        ax_apg.set_ylim(apg_buf.min() - 0.3, apg_buf.max() + 0.3)

        # ── Inference ─────────────────────────
        cur_spec = all_specs[wi] if all_specs is not None else None
        cur_hf = data["hand_features"][wi] if data["hand_features"] is not None else None
        pred_sbp, pred_dbp = predict_bp(
            model, data["anchor_sig"], current_sig,
            data["anchor_sbp"], data["anchor_dbp"],
            device, args.target_scale, use_fusion=use_fusion,
            anchor_spec=anchor_spec, current_spec=cur_spec,
            anchor_hf=data["anchor_hf"], current_hf=cur_hf,
        )

        true_sbp = float(data["y_sbp"][wi])
        true_dbp = float(data["y_dbp"][wi])

        # ── BP display ────────────────────────
        ax_bp.clear()
        ax_bp.set_facecolor("#16213e")
        ax_bp.axis("off")

        # Large BP readout
        sbp_color = "#ff4444" if abs(pred_sbp - true_sbp) > 10 else "#00ff88"
        dbp_color = "#4444ff" if abs(pred_dbp - true_dbp) > 10 else "#00ff88"

        ax_bp.text(0.5, 0.85, "PREDICTED", ha="center", va="center",
                   color="#aaa", fontsize=10, transform=ax_bp.transAxes)
        ax_bp.text(0.5, 0.68, f"{pred_sbp:.0f}/{pred_dbp:.0f}",
                   ha="center", va="center", color="white",
                   fontsize=32, fontweight="bold", transform=ax_bp.transAxes)
        ax_bp.text(0.5, 0.55, "mmHg", ha="center", va="center",
                   color="#666", fontsize=10, transform=ax_bp.transAxes)

        ax_bp.text(0.5, 0.38, "GROUND TRUTH", ha="center", va="center",
                   color="#aaa", fontsize=10, transform=ax_bp.transAxes)
        ax_bp.text(0.5, 0.22, f"{true_sbp:.0f}/{true_dbp:.0f}",
                   ha="center", va="center", color="#00d4ff",
                   fontsize=26, fontweight="bold", transform=ax_bp.transAxes)

        err_sbp = pred_sbp - true_sbp
        err_dbp = pred_dbp - true_dbp
        ax_bp.text(0.5, 0.06,
                   f"Error: SBP {err_sbp:+.1f}  DBP {err_dbp:+.1f}",
                   ha="center", va="center", color="#ffaa00",
                   fontsize=9, transform=ax_bp.transAxes)

        # ── History ───────────────────────────
        hist_sbp_pred.append(pred_sbp)
        hist_dbp_pred.append(pred_dbp)
        hist_sbp_true.append(true_sbp)
        hist_dbp_true.append(true_dbp)

        x_hist = list(range(len(hist_sbp_pred)))
        line_sbp_pred.set_data(x_hist, hist_sbp_pred)
        line_sbp_true.set_data(x_hist, hist_sbp_true)
        line_dbp_pred.set_data(x_hist, hist_dbp_pred)
        line_dbp_true.set_data(x_hist, hist_dbp_true)

        if len(x_hist) > 1:
            ax_hist.set_xlim(0, len(x_hist))
            all_bp = hist_sbp_pred + hist_sbp_true + hist_dbp_pred + hist_dbp_true
            ax_hist.set_ylim(min(all_bp) - 5, max(all_bp) + 5)

        # ── Info bar ──────────────────────────
        ax_info.clear()
        ax_info.set_facecolor("#16213e")
        ax_info.axis("off")
        elapsed = time.time() - state["start_time"]

        # Running MAE
        if len(hist_sbp_pred) > 0:
            mae_sbp = np.mean(np.abs(np.array(hist_sbp_pred) - np.array(hist_sbp_true)))
            mae_dbp = np.mean(np.abs(np.array(hist_dbp_pred) - np.array(hist_dbp_true)))
        else:
            mae_sbp = mae_dbp = 0.0

        info_text = (
            f"Window {wi+1}/{N_windows}  |  "
            f"Anchor SBP/DBP: {data['anchor_sbp']:.0f}/{data['anchor_dbp']:.0f}  |  "
            f"Running MAE: SBP={mae_sbp:.1f}  DBP={mae_dbp:.1f}  |  "
            f"Time: {elapsed:.1f}s"
        )
        ax_info.text(0.5, 0.5, info_text, ha="center", va="center",
                     color="#00d4ff", fontsize=9, transform=ax_info.transAxes,
                     family="monospace")

        state["window_idx"] = wi + 1

        return [line_ppg, line_vpg, line_apg,
                line_sbp_pred, line_sbp_true, line_dbp_pred, line_dbp_true]

    # ── Run animation ─────────────────────────
    interval_ms = args.interval  # ms between frames

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save N snapshot frames as images + create animated GIF
    n_frames = min(args.n_frames, N_windows)
    print(f"Generating {n_frames} demo frames...")

    frames_dir = os.path.join(os.path.dirname(args.checkpoint), "demo_frames")
    os.makedirs(frames_dir, exist_ok=True)

    saved_paths = []
    for f in range(n_frames):
        update(f)
        path = os.path.join(frames_dir, f"frame_{f:03d}.png")
        fig.savefig(path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
        saved_paths.append(path)
        if (f + 1) % 10 == 0:
            print(f"  Frame {f+1}/{n_frames}")

    # Save key snapshot (last frame)
    snapshot_path = os.path.join(os.path.dirname(args.checkpoint), "demo_snapshot.png")
    fig.savefig(snapshot_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nDemo snapshot saved: {snapshot_path}")
    print(f"All {n_frames} frames saved to: {frames_dir}")

    # Try to create animated GIF
    try:
        from PIL import Image
        images = [Image.open(p) for p in saved_paths]
        gif_path = os.path.join(os.path.dirname(args.checkpoint), "demo_animation.gif")
        images[0].save(gif_path, save_all=True, append_images=images[1:],
                       duration=interval_ms, loop=0)
        print(f"Animated GIF saved: {gif_path}")
    except ImportError:
        print("(Install Pillow for animated GIF: pip install Pillow)")

    plt.close()


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time BP estimation demo")
    parser.add_argument("--data_dir", default=r"C:\MIMIC2_out_v8")
    parser.add_argument("--checkpoint", default=r".\checkpoints_v10\best_model.pt")
    parser.add_argument("--patient_idx", type=int, default=0,
                        help="Which patient to demo (index into valid patients)")
    parser.add_argument("--target_scale", type=float, default=10.0)
    parser.add_argument("--use_fusion", action="store_true", default=True)
    parser.add_argument("--no_fusion", dest="use_fusion", action="store_false")
    parser.add_argument("--interval", type=int, default=500,
                        help="Milliseconds between window updates (lower=faster)")
    parser.add_argument("--n_frames", type=int, default=30,
                        help="Number of frames to generate for the demo")
    args = parser.parse_args()
    run_demo(args)
