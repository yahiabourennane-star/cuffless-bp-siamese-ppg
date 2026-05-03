"""
app.py
------
Streamlit real-time BP estimation demo.

Satisfies Stage 4 requirement:
  "A real-time visualisation of the signal together with the analysis outcome"

Usage:
    streamlit run app.py
"""

import os
import time
import numpy as np
import torch
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Project imports
from siamese_cnn import build_model
from dataset import (
    compute_global_std_from_indices,
    compute_spectrogram,
    make_derived_cache_path,
    make_within_patient_masks,
    stack_signals,
    standardize_features_from_train_stats,
)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
DATA_DIR = r"C:\MIMIC2_out_v9"
CHECKPOINT = r".\checkpoints_v26_beataware\best_model.pt"
TARGET_SCALE = 10.0
FS = 100  # Hz (v9 data is 100 Hz)
HAND_FEATURE_DIM = 18
MODEL_USES_FUSION = True

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Real-Time BP Estimation",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS for dark medical-monitor look
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .bp-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        border: 1px solid #333;
        margin: 5px;
    }
    .bp-value {
        font-size: 52px;
        font-weight: bold;
        color: #ffffff;
        line-height: 1.1;
    }
    .bp-label {
        font-size: 14px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .bp-unit {
        font-size: 14px;
        color: #555;
    }
    .error-text {
        font-size: 16px;
        font-weight: bold;
        padding: 5px;
    }
    .metric-box {
        background: #16213e;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border: 1px solid #333;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #00d4ff;
    }
    .metric-label {
        font-size: 12px;
        color: #888;
        text-transform: uppercase;
    }
    .good { color: #00ff88; }
    .warn { color: #ffaa00; }
    .bad  { color: #ff4444; }
    .stApp header { background-color: transparent; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Cached data loading
# ──────────────────────────────────────────────
@st.cache_resource
def load_model(checkpoint_path, device):
    """Load the trained Siamese CNN model."""
    model = build_model(
        embed_dim=256,
        dropout=0.3,
        device=device,
        use_fusion=MODEL_USES_FUSION,
        use_gated_pair_interaction=True,
        hand_feature_dim=HAND_FEATURE_DIM,
        input_channels=3,
    )
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@st.cache_data
def load_all_patient_data(data_dir, min_sbp_std=5.0):
    """Load data and return valid patient list + all arrays."""
    ppg = np.load(os.path.join(data_dir, "X_ppg_windows.npy"), mmap_mode="r")
    vpg = np.load(os.path.join(data_dir, "X_vpg.npy"), mmap_mode="r")
    apg = np.load(os.path.join(data_dir, "X_apg.npy"), mmap_mode="r")
    y_sbp = np.load(os.path.join(data_dir, "y_sbp.npy"))
    y_dbp = np.load(os.path.join(data_dir, "y_dbp.npy"))
    patient_ids = np.load(os.path.join(data_dir, "patient_ids.npy"))

    # Valid patients (enough BP variability)
    unique_pids = np.unique(patient_ids)
    valid_pids = []
    for pid in unique_pids:
        mask = patient_ids == pid
        if y_sbp[mask].std() >= min_sbp_std:
            valid_pids.append(int(pid))

    train_mask, _, _ = make_within_patient_masks(
        patient_ids, valid_pids, train_frac=0.70, val_frac=0.15,
        shuffle_eval=True, seed=42
    )
    train_stat_idx = np.where(train_mask)[0]
    ppg_global_std = compute_global_std_from_indices(ppg, train_stat_idx)

    window_len = int(ppg.shape[1])
    hand_cache = make_derived_cache_path(data_dir, "X_hand_features", FS, window_len, extra="raw_v2")
    beat_cache = make_derived_cache_path(data_dir, "X_beat_features", FS, window_len, extra="raw_v2")

    hand_base = np.load(hand_cache)
    beat_features = np.load(beat_cache)
    hand_features = np.concatenate([hand_base, beat_features], axis=1).astype(np.float32)
    hand_features = standardize_features_from_train_stats(hand_features, train_mask)

    return {
        "ppg": ppg, "vpg": vpg, "apg": apg,
        "y_sbp": y_sbp, "y_dbp": y_dbp,
        "patient_ids": patient_ids,
        "hand_features": hand_features,
        "valid_pids": valid_pids,
        "train_mask": train_mask,
        "ppg_global_std": ppg_global_std,
    }


def get_patient_data(all_data, patient_idx):
    """Extract and prepare data for one patient."""
    pid = all_data["valid_pids"][patient_idx]
    mask = all_data["patient_ids"] == pid
    indices = np.where(mask)[0]

    X = stack_signals(
        np.array(all_data["ppg"][indices]),
        np.array(all_data["vpg"][indices]),
        np.array(all_data["apg"][indices]),
        preserve_ppg_amplitude=True,
        ppg_global_std=all_data["ppg_global_std"],
    )

    sbp_vals = all_data["y_sbp"][indices]
    dbp_vals = all_data["y_dbp"][indices]

    # Match the training/eval pipeline: anchor chosen from this patient's training windows only.
    train_local_idx = np.where(all_data["train_mask"][indices])[0]
    if len(train_local_idx) == 0:
        train_local_idx = np.arange(len(indices))

    cand_sbp = sbp_vals[train_local_idx]
    cand_dbp = dbp_vals[train_local_idx]
    median_sbp, median_dbp = np.median(cand_sbp), np.median(cand_dbp)
    sbp_std = cand_sbp.std() + 1e-8
    dbp_std = cand_dbp.std() + 1e-8
    joint_dist = (
        np.abs(cand_sbp - median_sbp) / sbp_std +
        np.abs(cand_dbp - median_dbp) / dbp_std
    )
    anchor_idx = int(train_local_idx[np.argmin(joint_dist)])

    hf = all_data["hand_features"][indices] if all_data["hand_features"] is not None else None

    return {
        "X": X,
        "y_sbp": sbp_vals,
        "y_dbp": dbp_vals,
        "anchor_sig": X[anchor_idx],
        "anchor_sbp": float(sbp_vals[anchor_idx]),
        "anchor_dbp": float(dbp_vals[anchor_idx]),
        "anchor_hf": hf[anchor_idx] if hf is not None else None,
        "hand_features": hf,
        "anchor_local_idx": anchor_idx,
        "pid": pid,
        "n_windows": len(indices),
    }


# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────
def predict_bp(model, anchor_sig, current_sig, anchor_sbp, anchor_dbp,
               device, anchor_hf=None, current_hf=None,
               anchor_spec=None, current_spec=None):
    """Single-window inference → absolute SBP, DBP."""
    with torch.no_grad():
        anc = torch.tensor(anchor_sig, dtype=torch.float32).unsqueeze(0).to(device)
        cur = torch.tensor(current_sig, dtype=torch.float32).unsqueeze(0).to(device)
        anc_sbp_t = torch.tensor([anchor_sbp / 100.0], dtype=torch.float32).to(device)
        anc_dbp_t = torch.tensor([anchor_dbp / 100.0], dtype=torch.float32).to(device)

        # Spectrograms for fusion
        if anchor_spec is not None and current_spec is not None:
            anc_sp = torch.tensor(anchor_spec, dtype=torch.float32).unsqueeze(0).to(device)
            cur_sp = torch.tensor(current_spec, dtype=torch.float32).unsqueeze(0).to(device)
        else:
            anc_sp = cur_sp = None

        # Handcrafted features
        if anchor_hf is not None and current_hf is not None:
            anc_hf = torch.tensor(anchor_hf, dtype=torch.float32).unsqueeze(0).to(device)
            cur_hf = torch.tensor(current_hf, dtype=torch.float32).unsqueeze(0).to(device)
        else:
            hand_dim = int(getattr(model, "hand_feature_dim", HAND_FEATURE_DIM))
            anc_hf = torch.zeros(1, hand_dim, dtype=torch.float32).to(device)
            cur_hf = torch.zeros(1, hand_dim, dtype=torch.float32).to(device)

        pred = model(anc, cur, anc_sbp_t, anc_dbp_t,
                     anchor_spec=anc_sp, current_spec=cur_sp,
                     anchor_hf=anc_hf, current_hf=cur_hf)
        delta = pred.cpu().numpy()[0] * TARGET_SCALE

    return float(anchor_sbp + delta[0]), float(anchor_dbp + delta[1])


# ──────────────────────────────────────────────
# Precompute spectrograms for a patient
# ──────────────────────────────────────────────
@st.cache_data
def precompute_specs(X_patient, anchor_idx):
    """Compute spectrograms for all windows of a patient."""
    N = len(X_patient)
    sample = compute_spectrogram(X_patient[0, 0], fs=FS)
    F, T = sample.shape
    all_specs = np.zeros((N, 3, F, T), dtype=np.float32)
    for i in range(N):
        for c in range(3):
            all_specs[i, c] = compute_spectrogram(X_patient[i, c], fs=FS)
    anchor_spec = all_specs[anchor_idx]
    return all_specs, anchor_spec


# ──────────────────────────────────────────────
# Helper: colour based on error
# ──────────────────────────────────────────────
def error_color(err):
    abs_err = abs(err)
    if abs_err <= 5:
        return "#00ff88"  # green
    elif abs_err <= 10:
        return "#ffaa00"  # amber
    else:
        return "#ff4444"  # red


def error_class(err):
    abs_err = abs(err)
    if abs_err <= 5:
        return "good"
    elif abs_err <= 10:
        return "warn"
    else:
        return "bad"


# ──────────────────────────────────────────────
# Main app
# ──────────────────────────────────────────────
def main():
    # Title
    st.markdown("""
    <h1 style='text-align: center; color: #00d4ff; margin-bottom: 0;'>
        🫀 Real-Time Blood Pressure Estimation
    </h1>
    <p style='text-align: center; color: #666; margin-top: 5px;'>
        Siamese CNN + Temporal Attention  |  PPG → BP  |  MIMIC-II
    </p>
    """, unsafe_allow_html=True)

    # Load model and data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(CHECKPOINT, device)
    all_data = load_all_patient_data(DATA_DIR)

    n_patients = len(all_data["valid_pids"])

    # ── Sidebar ────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Controls")

        patient_idx = st.selectbox(
            "Patient",
            range(n_patients),
            format_func=lambda i: f"Patient {i} (ID: {all_data['valid_pids'][i]})",
            index=5,
        )

        speed = st.select_slider(
            "Speed",
            options=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
            value=1.0,
            format_func=lambda x: f"{x}x",
        )

        display_windows = st.slider("Waveform display (windows)", 4, 16, 8)

        use_fusion = MODEL_USES_FUSION

        st.markdown("---")
        st.markdown("### 📊 Model Info")
        st.markdown(f"""
        - **Architecture:** Siamese CNN
        - **Encoder:** 1D + 2D Fusion
        - **Attention:** Temporal (4-head)
        - **Features:** PPG + VPG + APG + beat-aware handcrafted stream
        - **Pair block:** Gated pair interaction
        - **Device:** `{device}`
        - **Checkpoint:** v26_beataware
        """)

        st.markdown("---")
        st.markdown("### 📋 BHS Grading")
        st.markdown("""
        | Grade | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg |
        |-------|---------|----------|----------|
        | **A** | ≥60%    | ≥85%     | ≥95%     |
        | **B** | ≥50%    | ≥75%     | ≥90%     |
        | **C** | ≥40%    | ≥65%     | ≥85%     |
        """)

    # ── Load patient data ─────────────────────
    pdata = get_patient_data(all_data, patient_idx)

    # Show patient info
    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    with col_info1:
        st.metric("Patient ID", pdata["pid"])
    with col_info2:
        st.metric("Windows", pdata["n_windows"])
    with col_info3:
        st.metric("SBP Range", f"{pdata['y_sbp'].min():.0f}–{pdata['y_sbp'].max():.0f}")
    with col_info4:
        st.metric("DBP Range", f"{pdata['y_dbp'].min():.0f}–{pdata['y_dbp'].max():.0f}")

    # Precompute spectrograms if fusion
    if use_fusion:
        with st.spinner("Computing spectrograms..."):
            all_specs, anchor_spec = precompute_specs(pdata["X"], pdata["anchor_local_idx"])
    else:
        all_specs = anchor_spec = None

    st.markdown("---")

    # ── Session state for animation ───────────
    if "running" not in st.session_state:
        st.session_state.running = False
    if "window_idx" not in st.session_state:
        st.session_state.window_idx = 0
    if "history" not in st.session_state:
        st.session_state.history = {
            "window": [],
            "sbp_pred": [], "dbp_pred": [],
            "sbp_true": [], "dbp_true": [],
        }
    # Reset history if patient changed
    if "current_patient" not in st.session_state or st.session_state.current_patient != patient_idx:
        st.session_state.current_patient = patient_idx
        st.session_state.window_idx = 0
        st.session_state.window_slider = 0
        st.session_state.history = {
            "window": [],
            "sbp_pred": [], "dbp_pred": [],
            "sbp_true": [], "dbp_true": [],
        }

    if "window_slider" not in st.session_state:
        st.session_state.window_slider = int(st.session_state.window_idx)
    max_window_idx = pdata["n_windows"] - 1
    st.session_state.window_idx = int(max(0, min(max_window_idx, st.session_state.window_idx)))
    st.session_state.window_slider = int(max(0, min(max_window_idx, st.session_state.window_slider)))

    def reset_demo_history():
        st.session_state.history = {
            "window": [],
            "sbp_pred": [], "dbp_pred": [],
            "sbp_true": [], "dbp_true": [],
        }

    def jump_to_window(target):
        target = int(max(0, min(pdata["n_windows"] - 1, target)))
        st.session_state.window_idx = target
        st.session_state.running = False
        reset_demo_history()

    def jump_from_slider():
        jump_to_window(st.session_state.window_slider)

    # ── Control buttons ───────────────────────
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([1, 1, 1, 3])
    with btn_col1:
        if st.button("▶️ Play", use_container_width=True):
            st.session_state.running = True
    with btn_col2:
        if st.button("⏸️ Pause", use_container_width=True):
            st.session_state.running = False
    with btn_col3:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.window_idx = 0
            st.session_state.window_slider = 0
            st.session_state.running = False
            reset_demo_history()
    with btn_col4:
        st.slider(
            "Window", 0, pdata["n_windows"] - 1,
            key="window_slider",
            on_change=jump_from_slider,
        )

    # ── Current window index ──────────────────
    wi = st.session_state.window_idx
    if wi >= pdata["n_windows"]:
        wi = 0
        st.session_state.window_idx = 0
        st.session_state.window_slider = 0
        reset_demo_history()

    # ── Run inference ─────────────────────────
    current_sig = pdata["X"][wi]
    cur_spec = all_specs[wi] if all_specs is not None else None
    cur_hf = pdata["hand_features"][wi] if pdata["hand_features"] is not None else None

    pred_sbp, pred_dbp = predict_bp(
        model, pdata["anchor_sig"], current_sig,
        pdata["anchor_sbp"], pdata["anchor_dbp"],
        device,
        anchor_hf=pdata["anchor_hf"], current_hf=cur_hf,
        anchor_spec=anchor_spec, current_spec=cur_spec,
    )

    true_sbp = float(pdata["y_sbp"][wi])
    true_dbp = float(pdata["y_dbp"][wi])
    err_sbp = pred_sbp - true_sbp
    err_dbp = pred_dbp - true_dbp

    # Update history
    hist = st.session_state.history
    if "window" not in hist:
        hist["window"] = []
    if not hist["window"] or hist["window"][-1] != wi:
        hist["window"].append(wi)
        hist["sbp_pred"].append(pred_sbp)
        hist["dbp_pred"].append(pred_dbp)
        hist["sbp_true"].append(true_sbp)
        hist["dbp_true"].append(true_dbp)

    # ── BP Display Cards ──────────────────────
    col_pred, col_true, col_err = st.columns([2, 2, 2])

    with col_pred:
        st.markdown(f"""
        <div class="bp-card">
            <div class="bp-label">Predicted</div>
            <div class="bp-value">{pred_sbp:.0f}/{pred_dbp:.0f}</div>
            <div class="bp-unit">mmHg</div>
        </div>
        """, unsafe_allow_html=True)

    with col_true:
        st.markdown(f"""
        <div class="bp-card">
            <div class="bp-label">Ground Truth</div>
            <div class="bp-value" style="color: #00d4ff;">{true_sbp:.0f}/{true_dbp:.0f}</div>
            <div class="bp-unit">mmHg</div>
        </div>
        """, unsafe_allow_html=True)

    with col_err:
        sbp_col = error_color(err_sbp)
        dbp_col = error_color(err_dbp)
        # Running MAE
        if hist["sbp_pred"]:
            r_mae_sbp = np.mean(np.abs(np.array(hist["sbp_pred"]) - np.array(hist["sbp_true"])))
            r_mae_dbp = np.mean(np.abs(np.array(hist["dbp_pred"]) - np.array(hist["dbp_true"])))
        else:
            r_mae_sbp = r_mae_dbp = 0.0

        st.markdown(f"""
        <div class="bp-card">
            <div class="bp-label">Error</div>
            <div style="font-size: 28px; font-weight: bold;">
                <span style="color: {sbp_col};">SBP {err_sbp:+.1f}</span> /
                <span style="color: {dbp_col};">DBP {err_dbp:+.1f}</span>
            </div>
            <div style="color: #888; font-size: 13px; margin-top: 8px;">
                Running MAE — SBP: {r_mae_sbp:.1f} | DBP: {r_mae_dbp:.1f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Waveform plots ────────────────────────
    start_w = max(0, wi - display_windows + 1)
    end_w = wi + 1

    ppg_buf = pdata["X"][start_w:end_w, 0, :].flatten()
    vpg_buf = pdata["X"][start_w:end_w, 1, :].flatten()
    apg_buf = pdata["X"][start_w:end_w, 2, :].flatten()
    t = np.arange(len(ppg_buf)) / FS

    fig_waves = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("PPG Signal", "VPG (1st Derivative)", "APG (2nd Derivative)"),
    )

    fig_waves.add_trace(
        go.Scatter(x=t, y=ppg_buf, mode="lines",
                   line=dict(color="#00ff88", width=1.2), name="PPG",
                   showlegend=False),
        row=1, col=1,
    )
    fig_waves.add_trace(
        go.Scatter(x=t, y=vpg_buf, mode="lines",
                   line=dict(color="#ff6b6b", width=1.2), name="VPG",
                   showlegend=False),
        row=2, col=1,
    )
    fig_waves.add_trace(
        go.Scatter(x=t, y=apg_buf, mode="lines",
                   line=dict(color="#4ecdc4", width=1.2), name="APG",
                   showlegend=False),
        row=3, col=1,
    )

    fig_waves.update_layout(
        height=450,
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#16213e",
        margin=dict(l=50, r=20, t=30, b=40),
        font=dict(size=11),
    )
    fig_waves.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig_waves.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig_waves.update_yaxes(title_text="Amplitude", row=2, col=1)
    fig_waves.update_yaxes(title_text="Amplitude", row=3, col=1)

    st.plotly_chart(fig_waves, use_container_width=True)

    # ── BP History chart ──────────────────────
    if len(hist["sbp_pred"]) > 1:
        x_hist = hist.get("window", list(range(len(hist["sbp_pred"]))))

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=x_hist, y=hist["sbp_pred"],
            mode="lines+markers", name="SBP Predicted",
            line=dict(color="#ff4444", width=2),
            marker=dict(size=4),
        ))
        fig_hist.add_trace(go.Scatter(
            x=x_hist, y=hist["sbp_true"],
            mode="lines", name="SBP Ground Truth",
            line=dict(color="#ff9999", width=1, dash="dash"),
        ))
        fig_hist.add_trace(go.Scatter(
            x=x_hist, y=hist["dbp_pred"],
            mode="lines+markers", name="DBP Predicted",
            line=dict(color="#4444ff", width=2),
            marker=dict(size=4),
        ))
        fig_hist.add_trace(go.Scatter(
            x=x_hist, y=hist["dbp_true"],
            mode="lines", name="DBP Ground Truth",
            line=dict(color="#9999ff", width=1, dash="dash"),
        ))

        fig_hist.update_layout(
            title="BP Prediction History",
            height=300,
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#16213e",
            margin=dict(l=50, r=20, t=40, b=40),
            xaxis_title="Window",
            yaxis_title="mmHg",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Progress bar ──────────────────────────
    progress = (wi + 1) / pdata["n_windows"]
    st.progress(progress, text=f"Window {wi + 1} / {pdata['n_windows']}  |  "
                               f"Anchor: {pdata['anchor_sbp']:.0f}/{pdata['anchor_dbp']:.0f} mmHg")

    # ── Auto-advance if running ───────────────
    if st.session_state.running:
        delay = 0.5 / speed  # base 500ms per frame
        time.sleep(delay)
        st.session_state.window_idx = wi + 1
        st.rerun()


if __name__ == "__main__":
    main()
