#app.py
import os
import numpy as np
from PIL import Image
import streamlit as st

st.set_page_config(
    page_title="PerfusionAI Clinical Decision Support System",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE = "outputs"

PATHS = {
    "predictions": os.path.join(BASE, "predictions"),
    "overlays": os.path.join(BASE, "overlays"),
    "errors": os.path.join(BASE, "errors"),
    "confidence": os.path.join(BASE, "confidence"),
    "uncertainty": os.path.join(BASE, "uncertainty"),
    "polar": os.path.join(BASE, "polar"),
    "gradcam": os.path.join(BASE, "gradcam"),
    "defect_map": os.path.join(BASE, "defect_map"),
    "region_dice": os.path.join(BASE, "region_dice.npy"),
    "selected_meta": os.path.join(BASE, "selected_meta.npy"),
    "dice_curve": os.path.join(BASE, "dice_curve.png"),
    "loss_curve": os.path.join(BASE, "loss_curve.png"),
    "best_sample": os.path.join(BASE, "best_sample.png"),
    "worst_sample": os.path.join(BASE, "worst_sample.png"),
}

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.0rem;
        font-weight: 700;
        margin-bottom: 0.15rem;
    }
    .subtle {
        color: #666;
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1rem 1rem 0.6rem 1rem;
        border: 1px solid rgba(128,128,128,0.18);
        border-radius: 14px;
        background-color: rgba(250,250,250,0.03);
        margin-bottom: 1rem;
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Load helpers
# -----------------------------
def safe_exists(path: str) -> bool:
    return os.path.exists(path)

def load_image(path: str):
    if safe_exists(path):
        return Image.open(path)
    return None

def load_npy(path: str):
    if safe_exists(path):
        return np.load(path, allow_pickle=True)
    return None

def load_dict_npy(path: str):
    arr = load_npy(path)
    if arr is None:
        return {}
    return arr.item()

def render_img(path: str, caption: str | None = None):
    img = load_image(path)
    if img is None:
        st.warning(f"Missing: {os.path.basename(path)}")
    else:
        st.image(img, caption=caption, use_container_width=True)

def available_ui_indices(max_n: int = 10):
    idxs = []
    for i in range(max_n):
        pred_path = os.path.join(PATHS["predictions"], f"pred_{i}.png")
        if safe_exists(pred_path):
            idxs.append(i)
    return idxs

# -----------------------------
# Reliability score
# -----------------------------
def compute_reliability(slice_idx: int):
    """
    Reliability score uses only saved quantitative artifacts:
    - global Dice from selected_meta
    - region Dice from region_dice.npy
    - confidence map (inside ROI only)
    - uncertainty map (inside ROI only)

    Score range: 0 to 1
    """
    selected_meta = load_dict_npy(PATHS["selected_meta"])
    region_dice_dict = load_dict_npy(PATHS["region_dice"])

    meta = selected_meta.get(slice_idx, {})
    dice_global = float(meta.get("dice", 0.0))

    region_scores = region_dice_dict.get(slice_idx, {"top": 0.0, "middle": 0.0, "bottom": 0.0})
    region_vals = np.array(list(region_scores.values()), dtype=np.float32)
    region_mean = float(region_vals.mean()) if len(region_vals) > 0 else 0.0
    region_std = float(region_vals.std()) if len(region_vals) > 0 else 0.0

    conf_path = os.path.join(PATHS["confidence"], f"conf_{slice_idx}.npy")
    unc_path = os.path.join(PATHS["uncertainty"], f"unc_{slice_idx}.npy")

    conf = load_npy(conf_path)
    unc = load_npy(unc_path)

    mean_conf = 0.0
    mean_unc = 1.0
    if conf is not None:
        conf = np.asarray(conf).astype(np.float32)
        roi = conf > 0.05
        if roi.sum() > 0:
            mean_conf = float(conf[roi].mean())
        else:
            mean_conf = float(conf.mean())

    if unc is not None:
        unc = np.asarray(unc).astype(np.float32)
        roi = (conf > 0.05) if conf is not None else np.ones_like(unc, dtype=bool)
        if roi.sum() > 0:
            unc_roi = unc[roi]
        else:
            unc_roi = unc.flatten()

        unc_max = float(np.max(unc_roi)) if unc_roi.size > 0 else 0.0
        if unc_max > 0:
            unc_norm = unc_roi / (unc_max + 1e-6)
            mean_unc = float(unc_norm.mean())
        else:
            mean_unc = 0.0

    # Region consistency rewards balanced performance across regions
    if region_mean > 1e-6:
        region_consistency = max(0.0, 1.0 - (region_std / (region_mean + 1e-6)))
    else:
        region_consistency = 0.0

    # Weighted score
    reliability = (
        0.40 * dice_global +
        0.25 * mean_conf +
        0.20 * (1.0 - mean_unc) +
        0.15 * region_consistency
    )
    reliability = float(np.clip(reliability, 0.0, 1.0))

    # Hard safety rules
    hard_fail = (
        dice_global < 0.60 or
        mean_conf < 0.45 or
        mean_unc > 0.55
    )

    if hard_fail:
        status = "Unreliable"
    elif reliability >= 0.72:
        status = "Reliable"
    elif reliability >= 0.55:
        status = "Review Required"
    else:
        status = "Unreliable"

    return {
        "reliability": reliability,
        "status": status,
        "dice_global": dice_global,
        "region_mean": region_mean,
        "region_std": region_std,
        "mean_conf": mean_conf,
        "mean_unc": mean_unc,
        "region_scores": region_scores,
    }

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="main-title">🫀 PerfusionAI Clinical Decision Support System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtle">Interpretable myocardial perfusion SPECT analysis with segmentation, confidence, Grad-CAM, polar mapping, and reliability scoring.</div>',
    unsafe_allow_html=True
)

ui_indices = available_ui_indices(10)
if not ui_indices:
    st.error("No UI outputs found in outputs/predictions. Run inference first.")
    st.stop()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Slice Viewer"], index=1)

st.sidebar.markdown("---")
st.sidebar.markdown("**Available Slices**")
st.sidebar.caption("Selected clinically meaningful slices only.")
st.sidebar.write(f"0 to {len(ui_indices)-1}")

# -----------------------------
# Dashboard
# -----------------------------
if page == "Dashboard":
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="card"><div class="section-title">Best Representative Case</div>', unsafe_allow_html=True)
        render_img(PATHS["best_sample"])
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card"><div class="section-title">Worst Representative Case</div>', unsafe_allow_html=True)
        render_img(PATHS["worst_sample"])
        st.markdown("</div>", unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="card"><div class="section-title">Training Dice Curve</div>', unsafe_allow_html=True)
        render_img(PATHS["dice_curve"])
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="card"><div class="section-title">Training Loss Curve</div>', unsafe_allow_html=True)
        render_img(PATHS["loss_curve"])
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Slice Viewer
# -----------------------------
else:
    slice_idx = st.sidebar.slider("Selected Slice Index", min_value=0, max_value=min(9, len(ui_indices)-1), value=0, step=1)

    metrics = compute_reliability(slice_idx)
    reliability = metrics["reliability"]
    status = metrics["status"]

    if status == "Reliable":
        status_color = "green"
        status_icon = "✅"
    elif status == "Review Required":
        status_color = "orange"
        status_icon = "⚠️"
    else:
        status_color = "red"
        status_icon = "❌"

    # Top summary
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Slice", f"{slice_idx}")
    m2.metric("Global Dice", f"{metrics['dice_global']:.3f}")
    m3.metric("Reliability Score", f"{reliability:.3f}")
    m4.markdown(
        f"""
        <div class="card" style="text-align:center; padding:0.8rem;">
            <div style="font-size:0.9rem; color:#666;">Decision Status</div>
            <div style="font-size:1.2rem; font-weight:700; color:{status_color};">{status_icon} {status}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    left, right = st.columns([3, 1])

    with left:
        tabs = st.tabs([
            "Prediction",
            "Overlay",
            "Error Map",
            "Confidence Map",
            "Grad-CAM",
            "Polar Map",
            "Defect Map",
        ])

        with tabs[0]:
            render_img(os.path.join(PATHS["predictions"], f"pred_{slice_idx}.png"))

        with tabs[1]:
            render_img(os.path.join(PATHS["overlays"], f"overlay_{slice_idx}.png"))

        with tabs[2]:
            render_img(os.path.join(PATHS["errors"], f"pred_{slice_idx}.png"))

        with tabs[3]:
            render_img(os.path.join(PATHS["confidence"], f"pred_{slice_idx}.png"))

        with tabs[4]:
            render_img(os.path.join(PATHS["gradcam"], f"gradcam_{slice_idx}.png"))

        with tabs[5]:
            render_img(os.path.join(PATHS["polar"], f"pred_{slice_idx}.png"))

        with tabs[6]:
            render_img(os.path.join(PATHS["defect_map"], f"defect_{slice_idx}.png"))

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Reliability Breakdown")
        st.write(f"**Mean Confidence:** {metrics['mean_conf']:.3f}")
        st.write(f"**Mean Uncertainty:** {metrics['mean_unc']:.3f}")
        st.write(f"**Mean Region Dice:** {metrics['region_mean']:.3f}")
        st.write(f"**Region Dice Std:** {metrics['region_std']:.3f}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Region-wise Dice")
        region_scores = metrics["region_scores"]
        st.write(f"**Anterior / Top:** {region_scores.get('top', 0.0):.3f}")
        st.write(f"**Mid:** {region_scores.get('middle', 0.0):.3f}")
        st.write(f"**Inferior / Bottom:** {region_scores.get('bottom', 0.0):.3f}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Clinical Interpretation")
        if status == "Reliable":
            st.success(
                "Prediction is quantitatively consistent. Confidence is adequate and uncertainty is controlled. Suitable for supportive clinical review."
            )
        elif status == "Review Required":
            st.warning(
                "Prediction is partially reliable, but caution is needed. Check overlay, error map, and Grad-CAM before clinical interpretation."
            )
        else:
            st.error(
                "Prediction is not sufficiently reliable. High uncertainty, weak confidence, or low Dice suggests this slice should not be trusted without manual review."
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Scoring Logic")
        st.caption(
            "Reliability Score = 0.40 × Global Dice + 0.25 × Mean Confidence + "
            "0.20 × (1 − Mean Uncertainty) + 0.15 × Region Consistency"
        )
        st.caption(
            "Hard fail rules mark a slice as Unreliable if Dice < 0.60, "
            "Mean Confidence < 0.45, or Mean Uncertainty > 0.55."
        )
        st.markdown("</div>", unsafe_allow_html=True)