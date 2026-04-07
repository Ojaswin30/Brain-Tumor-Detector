import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
import time

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Detection · NeuroVista",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
IMG_SIZE = 128
PORTFOLIO_URL = "https://ojaswin30.github.io/"

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
    /* Main page background */
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(255, 182, 193, 0.35), transparent 28%),
            radial-gradient(circle at top right, rgba(173, 216, 230, 0.35), transparent 26%),
            radial-gradient(circle at bottom left, rgba(255, 230, 180, 0.35), transparent 24%),
            linear-gradient(180deg, #fcfdff 0%, #f7faff 45%, #ffffff 100%);
        color: #16324f;
    }

    /* Hide default Streamlit chrome bits */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Top-left home button */
    .home-btn {
        position: fixed;
        top: 16px;
        left: 16px;
        z-index: 9999;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 10px 16px;
        border-radius: 999px;
        text-decoration: none;
        font-weight: 700;
        font-size: 0.92rem;
        color: #144d8a !important;
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid rgba(102, 153, 255, 0.18);
        box-shadow: 0 12px 32px rgba(77, 116, 177, 0.15);
        backdrop-filter: blur(12px);
        transition: transform 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
    }
    .home-btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 16px 40px rgba(77, 116, 177, 0.22);
        background: rgba(255, 255, 255, 0.92);
    }

    /* Wrapper spacing */
    .page-wrap {
        padding-top: 3.2rem;
        padding-bottom: 2rem;
    }

    /* Hero section */
    .hero {
        background: linear-gradient(135deg, rgba(255,255,255,0.82), rgba(245,250,255,0.92));
        border: 1px solid rgba(120, 154, 210, 0.18);
        box-shadow: 0 18px 50px rgba(70, 99, 145, 0.10);
        border-radius: 28px;
        padding: 2.2rem 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: "";
        position: absolute;
        inset: -60px auto auto -60px;
        width: 180px;
        height: 180px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(255, 160, 122, 0.35), rgba(255, 160, 122, 0));
        filter: blur(6px);
    }
    .hero::after {
        content: "";
        position: absolute;
        inset: auto -80px -80px auto;
        width: 240px;
        height: 240px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(116, 185, 255, 0.28), rgba(116, 185, 255, 0));
        filter: blur(8px);
    }

    .eyebrow {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background: linear-gradient(90deg, rgba(255, 159, 67, 0.15), rgba(48, 207, 208, 0.16));
        color: #0c5a73;
        font-weight: 700;
        font-size: 0.8rem;
        letter-spacing: 0.2px;
        margin-bottom: 14px;
        position: relative;
        z-index: 1;
    }

    .hero h1 {
        margin: 0;
        font-size: clamp(2rem, 3.8vw, 3.7rem);
        line-height: 1.05;
        color: #153a64;
        position: relative;
        z-index: 1;
    }

    .hero h1 .accent {
        background: linear-gradient(90deg, #ff7a59, #ffb347, #29b6f6, #6c63ff);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }

    .hero p {
        margin-top: 14px;
        max-width: 760px;
        color: #4e6782;
        font-size: 1.03rem;
        line-height: 1.7;
        position: relative;
        z-index: 1;
    }

    /* Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid rgba(120, 154, 210, 0.16);
        border-radius: 24px;
        box-shadow: 0 14px 34px rgba(70, 99, 145, 0.08);
        padding: 1.2rem;
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #173e6d;
        margin-bottom: 0.75rem;
    }

    .muted {
        color: #5f748d;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, #ffffff, #f4f8ff) !important;
        border: 2px dashed #7aa7ff !important;
        border-radius: 20px !important;
        padding: 1rem !important;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    }

    /* Inner drop zone */
    [data-testid="stFileUploader"] div {
        background: transparent !important;
        color: #1b3c5a !important;
    }

    /* Upload button */
    [data-testid="stFileUploader"] button {
        background: linear-gradient(90deg, #4facfe, #6c63ff) !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
    }

    /* Drag text */
    [data-testid="stFileUploader"] span {
        color: #355c7d !important;
    }

    /* Buttons */
    .stButton > button {
        border: none;
        border-radius: 16px;
        padding: 0.78rem 1.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #2d9cdb, #6c63ff);
        color: white;
        box-shadow: 0 12px 28px rgba(80, 109, 207, 0.22);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 16px 34px rgba(80, 109, 207, 0.28);
    }

    /* Result cards */
    .result-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 14px;
        margin-top: 10px;
    }

    .result-card {
        border-radius: 22px;
        padding: 1rem 1.1rem;
        border: 1px solid rgba(120, 154, 210, 0.16);
        box-shadow: 0 12px 28px rgba(70, 99, 145, 0.08);
        background: rgba(255, 255, 255, 0.85);
    }

    .result-title {
        font-weight: 800;
        margin-bottom: 0.35rem;
        font-size: 1rem;
    }

    .result-value {
        font-size: 1.7rem;
        font-weight: 900;
        letter-spacing: -0.03em;
    }

    .safe {
        background: linear-gradient(135deg, rgba(230, 255, 243, 0.95), rgba(240, 255, 250, 0.9));
        border-color: rgba(68, 184, 112, 0.18);
    }
    .safe .result-title { color: #16835c; }
    .safe .result-value { color: #0b8f63; }

    .danger {
        background: linear-gradient(135deg, rgba(255, 239, 239, 0.96), rgba(255, 248, 244, 0.92));
        border-color: rgba(240, 97, 97, 0.16);
    }
    .danger .result-title { color: #b93232; }
    .danger .result-value { color: #db3d3d; }

    .score-box {
        background: linear-gradient(135deg, rgba(245, 249, 255, 0.95), rgba(255, 255, 255, 0.92));
        border: 1px solid rgba(100, 133, 192, 0.15);
        border-radius: 20px;
        padding: 0.9rem 1rem;
        box-shadow: 0 10px 22px rgba(70, 99, 145, 0.07);
    }

    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 1rem;
    }

    .pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 8px 12px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 700;
        color: #33526f;
        background: rgba(239, 245, 255, 0.92);
        border: 1px solid rgba(120, 154, 210, 0.14);
    }

    .footer-note {
        margin-top: 1rem;
        font-size: 0.82rem;
        color: #7588a0;
        text-align: center;
    }

    /* Small screens */
    @media (max-width: 780px) {
        .hero { padding: 1.5rem 1.2rem; }
        .result-grid { grid-template-columns: 1fr; }
        .home-btn { top: 12px; left: 12px; }
    }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# HOME BUTTON
# ─────────────────────────────────────────────
st.markdown(
    f"""
<a class="home-btn" href="{PORTFOLIO_URL}" target="_blank" rel="noopener noreferrer">
    ⟵ Home
</a>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("brain_tumor_model.h5")

model = load_model()

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown(
    """
<div class="page-wrap">
    <div class="hero">
        <div class="eyebrow">AI-Assisted MRI Review</div>
        <h1>Brain Tumor <span class="accent">Detection</span></h1>
        <p>
            Upload an MRI scan and get a clean, visually guided prediction interface.
            This page uses a light theme with layered color accents, soft shadows, and a
            more polished presentation for a better user experience.
        </p>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")

st.write("")

# ─────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────
left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown(
        """
<div class="glass-card">
    <div class="section-title">Upload MRI Scan</div>
    <div class="muted">
        Choose a brain MRI image in JPG, JPEG, or PNG format. The preview and analysis
        are shown in a clean side-by-side layout.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.write("")

    uploaded_file = st.file_uploader(
        "Drop your MRI image here",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True, caption="Uploaded MRI preview")
    else:
        st.markdown(
            """
<div class="glass-card">
    <div class="muted" style="text-align:center; padding: 1.4rem 0;">
        Your preview will appear here after you upload an MRI image.
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

with right:
    st.markdown(
        """
<div class="glass-card">
    <div class="section-title">Analysis Panel</div>
    <div class="muted">
        Run the model after uploading an image to see the prediction result and confidence.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.write("")

    run_analysis = st.button("Run Analysis")

    if uploaded_file and run_analysis:
        with st.spinner("Analyzing MRI scan..."):
            img = np.array(image)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            time.sleep(0.35)
            pred = float(model.predict(img, verbose=0)[0][0])

        tumor = pred > 0.5
        confidence = pred if tumor else 1 - pred
        confidence_pct = round(confidence * 100, 2)
        raw_pct = round(pred * 100, 2)

        if tumor:
            result_html = f"""
<div class="result-card danger">
    <div class="result-title">⚠ Tumor Detected</div>
    <div class="result-value">{confidence_pct:.2f}%</div>
    <div class="muted">The model predicts a tumor with higher confidence.</div>
</div>
"""
        else:
            result_html = f"""
<div class="result-card safe">
    <div class="result-title">✓ No Tumor Found</div>
    <div class="result-value">{confidence_pct:.2f}%</div>
    <div class="muted">The model predicts a non-tumor case with higher confidence.</div>
</div>
"""

        st.markdown(result_html, unsafe_allow_html=True)

        st.write("")

        st.markdown(
            f"""
<div class="score-box">
    <div style="font-weight:800; color:#24496f; margin-bottom: 6px;">Model Score</div>
    <div class="muted">
        Raw prediction output: <strong>{raw_pct:.2f}%</strong><br>
        Threshold used: <strong>50%</strong>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.write("")
        st.markdown(
            """
<div class="glass-card">
    <div class="section-title">Reading Guide</div>
    <div class="muted">
        A higher score indicates stronger model confidence. This interface is designed for
        demonstration and educational use, not clinical diagnosis.
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

    elif run_analysis and not uploaded_file:
        st.warning("Please upload an MRI image first.")

# ─────────────────────────────────────────────
# EXTRA VISUAL SECTION
# ─────────────────────────────────────────────
st.write("")


# ─────────────────────────────────────────────
# DISCLAIMER
# ─────────────────────────────────────────────
st.write("")
st.markdown(
    """
<p class="footer-note">
    Disclaimer: This tool is for educational purposes only and does not replace professional medical diagnosis.
</p>
""",
    unsafe_allow_html=True,
)