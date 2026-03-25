import streamlit as st
import numpy as np
import os
from PIL import Image
import plotly.graph_objects as go

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(layout="wide", page_title="PerfusionAI Clinical System")
st.title("🫀 PerfusionAI Clinical Decision Support System")

# -----------------------------
# PATHS
# -----------------------------
base = "outputs"
paths = {
    "prediction": f"{base}/predictions",
    "overlay": f"{base}/overlays",
    "polar": f"{base}/polar",
    "error": f"{base}/errors",
    "confidence": f"{base}/confidence",
    "gradcam": f"{base}/gradcam",
    "region_dice": f"{base}/region_dice.npy",
    "dice_curve": f"{base}/dice_curve.png",
    "loss_curve": f"{base}/loss_curve.png",
    "best_sample": f"{base}/best_sample.png",
    "worst_sample": f"{base}/worst_sample.png"
}

# -----------------------------
# UTILS
# -----------------------------
def load_img(path):
    try:
        if os.path.exists(path):
            return Image.open(path)
    except:
        pass
    return None

def safe_image(img):
    if img is None:
        st.warning("Image not available")
    else:
        st.image(img, use_container_width=True)

def load_uncertainty(idx):
    path = os.path.join(paths["confidence"], f"conf_{idx}.npy")
    return np.load(path) if os.path.exists(path) else None

def load_region_dice():
    return np.load(paths["region_dice"], allow_pickle=True).item() if os.path.exists(paths["region_dice"]) else {}

def compute_error_ratio(error_img):
    if error_img is None:
        return 1.0
    arr = np.array(error_img)
    error_ratio = np.mean(arr / 255.0)
    return error_ratio  # % of error pixels

def compute_gradcam_focus(gradcam_img):
    if gradcam_img is None:
        return 0.0
    arr = np.array(gradcam_img).astype(float)
    return np.std(arr)  # higher std → more focused

# -----------------------------
# LOAD DATA
# -----------------------------
files = os.listdir(paths["prediction"])
all_indices = sorted([int(f.split("_")[1].split(".")[0]) for f in files])
indices = all_indices[:10]

region_dice_dict = load_region_dice()

# -----------------------------
# RELIABILITY MODEL (CORE)
# -----------------------------
def compute_reliability(uncertainty_map, error_img, dice_regions, gradcam_img):

    if uncertainty_map is None:
        return "⚠️ Unknown", {}, 0

    mean_conf = float(np.mean(uncertainty_map))
    std_conf = float(np.std(uncertainty_map))
    
    dice_mean = float(np.mean(list(dice_regions.values())))

    error_ratio = compute_error_ratio(error_img)
    gradcam_focus = compute_gradcam_focus(gradcam_img)

    # NORMALIZE
    error_score = 1 - error_ratio
    variance_penalty = std_conf

    # WEIGHTED SCORE
    score = (
        0.5 * mean_conf +
        0.2 * error_score +
        0.2 * dice_mean -
        0.1 * variance_penalty
    )

    # DECISION
    if score > 0.65:
        status = "🟢 Reliable"
    elif score > 0.45:
        status = "⚠️ Caution"
    else:
        status = "🔴 Unreliable"

    reasons = {
        "Mean Confidence": round(mean_conf, 3),
        "Confidence Std": round(std_conf, 3),
        "Dice Score": round(dice_mean, 3),
        "Error Ratio": round(error_ratio, 3),
        "GradCAM Focus": round(gradcam_focus, 3),
        "Final Score": round(score, 3)
    }

    return status, reasons, score

# -----------------------------
# DEFECT FROM POLAR MAP
# -----------------------------
def analyze_polar_map(polar_img):

    if polar_img is None:
        return "Unknown", "Unknown", {}

    arr = np.array(polar_img).astype(float)

    if len(arr.shape) == 3:
        arr = np.mean(arr, axis=2)

    h = arr.shape[0]

    top = np.mean(arr[:h//3])
    mid = np.mean(arr[h//3:2*h//3])
    bottom = np.mean(arr[2*h//3:])

    region_values = {
        "Anterior": top,
        "Mid": mid,
        "Inferior": bottom
    }

    worst_region = min(region_values, key=region_values.get)

    ratio = region_values[worst_region] / (max(region_values.values()) + 1e-8)

    if ratio < 0.5:
        defect = "Severe Defect"
    elif ratio < 0.75:
        defect = "Mild Defect"
    else:
        defect = "Normal"

    return defect, worst_region, region_values

# -----------------------------
# PLOT
# -----------------------------
def plot_overlay(img_path, prob=None):
    img = load_img(img_path)
    if img is None:
        st.warning("Overlay missing")
        return

    arr = np.array(img)
    fig = go.Figure(go.Image(z=arr))

    if prob is not None:
        prob = prob / (np.max(prob) + 1e-8)
        fig.add_trace(go.Heatmap(z=prob, opacity=0.4, showscale=False))

    fig.update_layout(margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# NAVIGATION
# -----------------------------
page = st.sidebar.radio("Navigation", ["Dashboard", "Slice Viewer"])

# =========================================================
# DASHBOARD
# =========================================================
if page == "Dashboard":

    st.subheader("System Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Best Case")
        safe_image(load_img(paths["best_sample"]))

    with col2:
        st.markdown("### Worst Case")
        safe_image(load_img(paths["worst_sample"]))

# =========================================================
# SLICE VIEWER
# =========================================================
else:

    selected_idx = st.sidebar.slider("Slice", 0, len(indices)-1, 0)
    slice_idx = indices[selected_idx]

    uncertainty_map = load_uncertainty(slice_idx)
    error_img = load_img(f"{paths['error']}/pred_{slice_idx}.png")
    gradcam_img = load_img(f"{paths['gradcam']}/gradcam_{slice_idx}.png")

    dice_regions = region_dice_dict.get(slice_idx, {"top":0,"middle":0,"bottom":0})

    reliability, reasons, score = compute_reliability(
        uncertainty_map, error_img, dice_regions, gradcam_img
    )

    polar_img = load_img(f"{paths['polar']}/pred_{slice_idx}.png")
    defect, region, region_values = analyze_polar_map(polar_img)

    # ---------------- TOP PANEL ----------------
    col1, col2, col3 = st.columns(3)

    col1.metric("Slice", selected_idx)
    col2.metric("Reliability Score", f"{score:.3f}")
    col3.metric("Decision", reliability)

    st.markdown("---")

    # ---------------- MAIN ----------------
    col_main, col_right = st.columns([3,1])

    with col_main:

        tabs = st.tabs(["Overlay", "Prediction", "Error", "Confidence", "Grad-CAM"])

        with tabs[0]:
            plot_overlay(f"{paths['overlay']}/overlay_{slice_idx}.png", uncertainty_map)

        with tabs[1]:
            safe_image(load_img(f"{paths['prediction']}/pred_{slice_idx}.png"))

        with tabs[2]:
            safe_image(error_img)

        with tabs[3]:
            safe_image(load_img(f"{paths['confidence']}/pred_{slice_idx}.png"))

        with tabs[4]:
            safe_image(gradcam_img)

    # ---------------- RIGHT PANEL ----------------
    with col_right:

        st.markdown("### Polar Map")
        safe_image(polar_img)

        st.markdown("### Region Dice")
        st.table(dice_regions)

        # ✅ YOUR REQUIRED SECTION (ADDED BEFORE DECISION)
        st.markdown("### Clinical Evidence")

        st.write("**Reliability Factors:**")
        st.json(reasons)

        st.write("**Perfusion by Region:**")
        st.json({k: round(v,2) for k,v in region_values.items()})

        st.markdown("---")

        st.markdown("### Clinical Decision")

        if reliability == "🔴 Unreliable":
            st.error("Model output NOT reliable → Do not trust")
        elif reliability == "⚠️ Caution":
            st.warning("Review required with clinician")
        else:
            st.success("Model output supported by quantitative evidence")

        st.markdown(f"""
        **Detected Defect:** {defect}  
        **Most Affected Region:** {region}
        """)

    # ---------------- BOTTOM ----------------
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        safe_image(load_img(paths["dice_curve"]))

    with col2:
        safe_image(load_img(paths["loss_curve"]))