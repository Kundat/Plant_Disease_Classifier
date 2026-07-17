"""Run with: streamlit run app/streamlit_app.py"""

import sys
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.predict import predict_image
from src import config

st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="centered",
)

# --- Custom CSS for extra polish ---
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #4E6B4E;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    .prediction-card {
        background-color: #F1F8F2;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 0.75rem;
        border-left: 5px solid #2E7D32;
    }
    .prediction-label {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1B2E1F;
    }
    .prediction-pct {
        font-size: 1rem;
        color: #2E7D32;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("🌿 About this project")
    st.write(
        "An image classifier that detects plant diseases from leaf photos, "
        "built with PyTorch using transfer learning on ResNet18."
    )
    st.markdown("---")
    st.write("**Classes detected:** 15")
    st.write("**Model:** ResNet18 (transfer learning)")
    st.markdown("---")
    st.caption("Built as a portfolio project.")

# --- Main content ---
st.markdown('<p class="main-title">🌿 Plant Disease Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a leaf photo and get an instant diagnosis</p>', unsafe_allow_html=True)

if not config.MODEL_PATH.exists():
    st.error(f"No trained model at `{config.MODEL_PATH}`. Run `python -m src.train` first.")
    st.stop()

uploaded_file = st.file_uploader(
    "Drag and drop a leaf image, or click to browse",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])

    image = Image.open(uploaded_file).convert("RGB")
    with col1:
        st.image(image, caption="Uploaded image", use_column_width=True)

    temp_path = Path(tempfile.gettempdir()) / "uploaded_image.jpg"
    image.save(temp_path)

    with st.spinner("Analyzing leaf..."):
        predictions = predict_image(str(temp_path))

    with col2:
        st.markdown("#### Predictions")
        for label, prob in predictions:
            display_label = label.replace("_", " ").replace("___", " — ")
            st.markdown(
                f"""
                <div class="prediction-card">
                    <span class="prediction-label">{display_label}</span><br>
                    <span class="prediction-pct">{prob * 100:.1f}%</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(prob)
else:
    st.info("👆 Upload an image above to get started.")