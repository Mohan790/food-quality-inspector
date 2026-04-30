"""
app.py — Streamlit Demo for Food Quality Inspector

Run:
    streamlit run app.py
"""

import os
import sys
import torch
import numpy as np
import streamlit as st
from PIL import Image
import plotly.graph_objects as go

sys.path.append(os.path.dirname(__file__))
from models.model import build_model
from data.dataset import get_transforms
from utils.gradcam import GradCAM, overlay_heatmap


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FreshCheck AI",
    page_icon="🍎",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .fresh-card {
        background: linear-gradient(135deg, #d1fae5, #a7f3d0);
        border-radius: 20px;
        padding: 28px;
        text-align: center;
        border: 2px solid #10b981;
    }
    .rotten-card {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        border-radius: 20px;
        padding: 28px;
        text-align: center;
        border: 2px solid #ef4444;
    }
    .info-card {
        background: white;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        margin-bottom: 16px;
    }
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        font-size: 13px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ── Food safety tips per prediction ───────────────────────────────────────────
FOOD_TIPS = {
    "Fresh": {
        "emoji": "✅",
        "color": "#10b981",
        "message": "This food appears fresh and safe to consume!",
        "tips": [
            "Store properly to maintain freshness",
            "Consume within recommended timeframe",
            "Keep refrigerated if required",
        ]
    },
    "Rotten": {
        "emoji": "❌",
        "color": "#ef4444",
        "message": "This food appears spoiled. Do NOT consume!",
        "tips": [
            "Dispose of this food immediately",
            "Clean the storage area to prevent contamination",
            "Check nearby foods for spoilage",
        ]
    }
}


# ── Load model (cached so it only loads once) ──────────────────────────────────
@st.cache_resource
def load_model(checkpoint_path: str = "./best_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint  = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint.get("class_names", ["Fresh", "Rotten"])
    model       = build_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, class_names, device


# ── Predict ────────────────────────────────────────────────────────────────────
def predict(model, image: Image.Image, class_names: list, device):
    transform = get_transforms("val")
    tensor    = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    results = sorted(
        zip(class_names, probs.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    return results, tensor


# ── Main UI ────────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div style='text-align:center; padding: 1.5rem 0 1rem;'>
        <h1 style='font-size:3rem; margin-bottom:0;'>🍎 FreshCheck AI</h1>
        <p style='color:#6b7280; font-size:1.1rem; margin-top:4px;'>
            Deep Learning Food Quality Inspector · MobileNetV2 + Grad-CAM
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    """, unsafe_allow_html=True)

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        model_path   = st.text_input("Model path", value="./best_model.pth")
        show_gradcam = st.toggle("Show Grad-CAM heatmap", value=True)
        alpha        = st.slider("Heatmap opacity", 0.2, 0.8, 0.5) if show_gradcam else 0.5

        st.markdown("---")
        st.markdown("**Model:** MobileNetV2 (fine-tuned)")
        st.markdown("**Task:** Fresh vs Rotten classification")
        st.markdown("**Input:** 224×224 RGB image")

        st.markdown("---")
        st.markdown("### 🥦 Supported Foods")
        foods = ["🍎 Apples", "🍌 Bananas", "🥭 Mangoes",
                 "🍊 Oranges", "🥔 Potatoes", "🍅 Tomatoes"]
        for f in foods:
            st.markdown(f"- {f}")

    # ── File uploader ──────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "📸 Upload a food image (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded is None:
        # Show example cards when no image uploaded
        st.markdown("---")
        st.subheader("How it works")
        c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class='info-card' style='text-align:center; background:white;'>
        <h2>📸</h2>
        <h4 style='color:#111827;'>Upload</h4>
        <p style='color:#4b5563;font-size:13px'>
        Take a photo of any fruit or vegetable
        </p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='info-card' style='text-align:center; background:white;'>
        <h2>🧠</h2>
        <h4 style='color:#111827;'>Analyze</h4>
        <p style='color:#4b5563;font-size:13px'>
        AI inspects color, texture and surface patterns
        </p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class='info-card' style='text-align:center; background:white;'>
        <h2>✅</h2>
        <h4 style='color:#111827;'>Result</h4>
        <p style='color:#4b5563;font-size:13px'>
        Get instant Fresh or Rotten verdict + heatmap
        </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Check model exists ─────────────────────────────────────────────────────
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at `{model_path}`. Train first with: `python train.py`")
        return

    # ── Load & predict ─────────────────────────────────────────────────────────
    image = Image.open(uploaded).convert("RGB")

    with st.spinner("🔍 Analyzing food quality..."):
        model, class_names, device = load_model(model_path)
        results, tensor = predict(model, image, class_names, device)

    top_class, top_conf = results[0]
    info = FOOD_TIPS[top_class]

    # ── Layout ─────────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([1, 1, 1.2])

    with col1:
        st.subheader("📷 Input Image")
        st.image(image, use_container_width=True)

    with col2:
        if show_gradcam:
            st.subheader("🔥 Grad-CAM Heatmap")
            with st.spinner("Computing heatmap..."):
                gradcam     = GradCAM(model)
                transform   = get_transforms("val")
                tensor_grad = transform(image).unsqueeze(0).to(device)
                tensor_grad.requires_grad_(True)
                heatmap, _  = gradcam.generate(tensor_grad)
                overlay     = overlay_heatmap(image, heatmap, alpha=alpha)
            st.image(overlay, use_container_width=True)
            st.caption("🔴 Red = model focused here | 🔵 Blue = ignored")

    with col3:
        st.subheader("📊 Result")

        # Result card
        card_class = "fresh-card" if top_class == "Fresh" else "rotten-card"
        st.markdown(f"""
        <div class='{card_class}'>
            <h1 style='margin:0; font-size:3rem;'>{info["emoji"]}</h1>
            <h2 style='margin:8px 0; color:{info["color"]};'>{top_class}</h2>
            <h3 style='margin:0;'>{top_conf*100:.1f}% confidence</h3>
            <p style='margin-top:12px; color:#374151;'>{info["message"]}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Confidence bar chart
        fig = go.Figure(go.Bar(
            x=[r[1] * 100 for r in results],
            y=[r[0] for r in results],
            orientation="h",
            marker_color=["#10b981" if r[0] == "Fresh" else "#ef4444" for r in results],
            text=[f"{r[1]*100:.1f}%" for r in results],
            textposition="outside",
        ))
        fig.update_layout(
            margin=dict(l=0, r=50, t=10, b=10),
            xaxis=dict(range=[0, 115], visible=False),
            height=120,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Safety tips
        st.markdown("### 💡 Food Safety Tips")
        for tip in info["tips"]:
            st.markdown(f"- {tip}")


if __name__ == "__main__":
    main()