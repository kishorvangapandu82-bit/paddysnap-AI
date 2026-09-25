"""
============================================================
PADDY GUARD AI -- CLEAN & HIGH-CONTRAST STREAMLIT DASHBOARD
============================================================
100% Native Streamlit components for crystal-clear readability
in both Light and Dark themes.
============================================================
"""

import sys
import json
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_image
from src.recommendations import DISEASE_KNOWLEDGE_BASE


# --------------------------------------------------------------
# 1. PAGE SETUP
# --------------------------------------------------------------
st.set_page_config(
    page_title="PaddySnap AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------------------
# 2. SIDEBAR
# --------------------------------------------------------------
with st.sidebar:
    st.title("🌾 PaddySnap AI")
    st.caption("Paddy Disease Diagnosis & Fertilizer System")
    st.divider()

    page = st.radio(
        "Navigation Menu:",
        [
            "🔍 Leaf Disease Diagnosis",
            "📊 Model Comparison Leaderboard",
            "📈 Training Curves & Confusion Matrices",
            "📖 Disease Encyclopedia",
            "🧪 Test Predictions (Raw vs Diagnosed)"
        ]
    )

    st.divider()
    gpu_active = torch.cuda.is_available()
    st.write(f"**Hardware:** {'🟢 NVIDIA RTX 5050 GPU' if gpu_active else '🟡 CPU'}")
    st.write("**Top Model:** EfficientNetV2-S (97.38% Test Acc)")
    st.write("**Test Set:** 1,562 Images")


# ==============================================================
# SECTION 1: LEAF DISEASE DIAGNOSIS
# ==============================================================
if page == "🔍 Leaf Disease Diagnosis":
    st.title("🔍 Paddy Leaf Disease Diagnosis & Advisory")
    st.write("Upload a paddy leaf photo to get instant AI classification, confidence scores, and agronomy advice.")
    st.divider()

    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.subheader("1. Input Image")
        source = st.radio("Choose Input:", ["Upload an Image", "Choose from Dataset Samples"], horizontal=True)

        target_img_path = None
        temp_file = Path("dataset/temp_input.jpg")

        if source == "Upload an Image":
            uploaded_file = st.file_uploader("Upload Leaf Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                img = Image.open(uploaded_file).convert("RGB")
                img.save(temp_file)
                target_img_path = str(temp_file)
                st.image(img, caption="Uploaded Image", use_container_width=True)
        else:
            sample_options = {
                "Rice Blast (Sample)": "dataset/train_images/blast/100004.jpg",
                "Bacterial Leaf Blight (Sample)": "dataset/train_images/bacterial_leaf_blight/100330.jpg",
                "Brown Spot (Sample)": "dataset/train_images/brown_spot/100078.jpg",
                "Dead Heart (Sample)": "dataset/train_images/dead_heart/100046.jpg",
                "Rice Hispa (Sample)": "dataset/train_images/hispa/100057.jpg",
                "Healthy Leaf (Normal)": "dataset/train_images/normal/100002.jpg"
            }
            chosen_sample = st.selectbox("Pick a Sample:", list(sample_options.keys()))
            sample_path = Path(sample_options[chosen_sample])
            if sample_path.exists():
                target_img_path = str(sample_path)
                st.image(Image.open(sample_path), caption=chosen_sample, use_container_width=True)

        st.subheader("2. Select Model")
        model_options = {
            "EfficientNetV2-S (Rank #1 — 97.38% Test Acc)": "efficientnet_v2_s",
            "ConvNeXt-Tiny (Rank #2 — 97.06% Test Acc)": "convnext_tiny",
            "ResNet-50 (Rank #3 — 96.99% Test Acc)": "resnet50",
            "DenseNet-121 (Rank #4 — 96.93% Test Acc)": "densenet121",
            "Custom PaddySnapNet (From Scratch — 93.09% Test Acc)": "custom_cnn"
        }
        chosen_model_label = st.selectbox("Architecture:", list(model_options.keys()))
        chosen_model_code = model_options[chosen_model_label]

        predict_btn = st.button("🚀 Run Diagnosis", type="primary", use_container_width=True)

    with col2:
        st.subheader("3. AI Diagnosis & Agronomy Report")

        if predict_btn and target_img_path:
            with st.spinner("Analyzing image..."):
                try:
                    res = predict_image(target_img_path, model_name=chosen_model_code)
                    predicted_class = res["predicted_class"]
                    confidence = res["confidence"]
                    rec = res["recommendation"]

                    # Top Metrics
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric("Predicted Disease", rec["common_name"])
                    with m2:
                        st.metric("AI Confidence", f"{confidence:.2f}%")

                    st.write(f"**Causal Pathogen:** *{rec['causal_organism']}*")
                    st.write(f"**Severity Level:** `{rec['severity']}`")

                    # Probabilities Chart
                    st.subheader("Probability Distribution Across 10 Classes")
                    probs_data = pd.DataFrame([
                        {"Class": k.replace("_", " ").title(), "Probability (%)": v}
                        for k, v in res["all_probabilities"].items()
                    ]).sort_values(by="Probability (%)", ascending=True)

                    fig = px.bar(
                        probs_data,
                        x="Probability (%)",
                        y="Class",
                        orientation="h",
                        text="Probability (%)",
                        color="Probability (%)",
                        color_continuous_scale="Blues"
                    )
                    fig.update_layout(
                        height=280,
                        margin=dict(l=10, r=20, t=10, b=10),
                        xaxis=dict(range=[0, 105]),
                        yaxis=dict(title=None),
                        coloraxis_showscale=False
                    )
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)

                    # Actionable Advisory
                    st.subheader("Actionable Agronomy Recommendations")
                    t1, t2, t3 = st.tabs(["🛡️ Treatment & Chemical Control", "🌾 Fertilizer & N-P-K Guidance", "🔍 Symptoms & Causes"])

                    with t1:
                        st.write("#### Organic & Cultural Management:")
                        for o in rec["organic_cultural_management"]:
                            st.write(f"- {o}")
                        st.write("#### Chemical Control:")
                        for c in rec["chemical_control"]:
                            st.write(f"- {c}")

                    with t2:
                        fg = rec["fertilizer_nutrient_guidance"]
                        st.info(f"**⚡ Nitrogen (N) Guidance:**\n\n{fg['nitrogen_action']}")
                        st.success(f"**🛡️ Potassium (K) Guidance:**\n\n{fg['potassium_action']}")
                        st.warning(f"**🧪 Phosphorus & Micronutrients:**\n\n{fg['phosphorus_action']} | {fg['micronutrients']}")

                    with t3:
                        st.write("#### Key Symptoms:")
                        for s in rec["symptoms"]:
                            st.write(f"- {s}")
                        st.write("#### Favorable Conditions:")
                        for cause in rec["causes"]:
                            st.write(f"- {cause}")

                except Exception as ex:
                    st.error(f"Error: {ex}")
        else:
            st.info("👈 Upload an image or select a sample from the left panel and click **Run Diagnosis**.")


# ==============================================================
# SECTION 2: MODEL COMPARISON LEADERBOARD
# ==============================================================
elif page == "📊 Model Comparison Leaderboard":
    st.title("📊 Multi-Model Performance Leaderboard")
    st.write("Formal evaluation results across all 5 CNN architectures on the **1,562 test images**.")
    st.divider()

    csv_path = Path("results/metrics/model_comparison.csv")
    if csv_path.exists():
        df_comp = pd.read_csv(csv_path)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

    st.divider()
    c1, c2 = st.columns([1.2, 1])
    with c1:
        bar_png = Path("results/graphs/model_comparison_bar_chart.png")
        if bar_png.exists():
            st.image(str(bar_png), caption="Model Performance Benchmark (Accuracy, Precision, Recall, F1-Score)", use_container_width=True)
    with c2:
        table_png = Path("results/graphs/model_comparison_table.png")
        if table_png.exists():
            st.image(str(table_png), caption="Summary Performance Table", use_container_width=True)

    st.subheader("Key Scientific Findings:")
    st.write("1. **Champion Model:** **EfficientNetV2-S** achieved the highest score with **97.38% Test Accuracy** and **97.35% Macro F1**.")
    st.write("2. **Efficiency Leader:** **DenseNet-121** achieved **96.93% Accuracy** with only **6.9641M parameters** (70% smaller than ResNet-50).")
    st.write("3. **Custom CNN (From Scratch):** Our custom **PaddySnapNet** reached **93.09% Test Accuracy** without any pretrained ImageNet weights.")
    st.write("4. **Transfer Learning Value:** Pretrained representations gave a **+4.29% accuracy boost** compared to training from scratch.")


# ==============================================================
# SECTION 3: TRAINING CURVES & CONFUSION MATRICES
# ==============================================================
elif page == "📈 Training Curves & Confusion Matrices":
    st.title("📈 Neural Training Curves & Confusion Matrices")
    st.write("Inspect loss/accuracy training curves and test set confusion matrix heatmaps for each model.")
    st.divider()

    model_sel = st.selectbox(
        "Choose Architecture:",
        [
            ("efficientnet_v2_s", "EfficientNetV2-S (Rank #1 — 97.38%)"),
            ("convnext_tiny", "ConvNeXt-Tiny (Rank #2 — 97.06%)"),
            ("resnet50", "ResNet-50 (Rank #3 — 96.99%)"),
            ("densenet121", "DenseNet-121 (Rank #4 — 96.93%)"),
            ("custom_cnn", "Custom PaddySnapNet (Rank #5 — 93.09%)")
        ],
        format_func=lambda x: x[1]
    )[0]

    c_left, c_right = st.columns(2, gap="large")

    with c_left:
        st.subheader(f"Loss & Accuracy Curves ({model_sel.upper()})")
        acc_p = Path(f"results/graphs/{model_sel}_accuracy.png")
        loss_p = Path(f"results/graphs/{model_sel}_loss.png")
        if acc_p.exists():
            st.image(str(acc_p), use_container_width=True)
        if loss_p.exists():
            st.image(str(loss_p), use_container_width=True)

    with c_right:
        st.subheader(f"Test Confusion Matrix ({model_sel.upper()})")
        cm_p = Path(f"results/confusion_matrices/{model_sel}_confusion_matrix.png")
        if cm_p.exists():
            st.image(str(cm_p), caption=f"Confusion Matrix ({model_sel})", use_container_width=True)

    # Per-epoch data table
    hist_file = Path(f"results/training_history/{model_sel}_history.csv")
    if hist_file.exists():
        st.subheader(f"Per-Epoch Training Log ({model_sel})")
        st.dataframe(pd.read_csv(hist_file), use_container_width=True)


# ==============================================================
# SECTION 4: DISEASE ENCYCLOPEDIA
# ==============================================================
elif page == "📖 Disease Encyclopedia":
    st.title("📖 Paddy Disease & Fertilizer Encyclopedia")
    st.write("Detailed pathology profiles, symptoms, and treatment guidelines for all 10 paddy classes.")
    st.divider()

    dis_name = st.selectbox(
        "Choose a Paddy Disease:",
        [k.replace("_", " ").title() for k in DISEASE_KNOWLEDGE_BASE.keys()]
    )
    dis_k = dis_name.lower().replace(" ", "_")
    d_info = DISEASE_KNOWLEDGE_BASE[dis_k]

    st.header(d_info["common_name"])
    st.write(f"**Pathogen:** *{d_info['causal_organism']}* | **Severity:** `{d_info['severity']}`")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("🔍 Visual Symptoms")
        for s in d_info["symptoms"]:
            st.write(f"- {s}")
        st.subheader("🌧️ Favorable Conditions")
        for c in d_info["causes"]:
            st.write(f"- {c}")

    with c2:
        st.subheader("🛡️ Organic & Cultural Management")
        for o in d_info["organic_cultural_management"]:
            st.write(f"- {o}")
        st.subheader("💊 Chemical Treatments")
        for ch in d_info["chemical_control"]:
            st.write(f"- {ch}")

    st.subheader("🌾 Fertilizer & N-P-K Guidance")
    fg = d_info["fertilizer_nutrient_guidance"]
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        st.info(f"**⚡ Nitrogen (N):**\n\n{fg['nitrogen_action']}")
    with fc2:
        st.success(f"**🛡️ Potassium (K):**\n\n{fg['potassium_action']}")
    with fc3:
        st.warning(f"**🧪 Phosphorus & Micronutrients:**\n\n{fg['phosphorus_action']}\n\n{fg['micronutrients']}")


# ==============================================================
# SECTION 5: TEST PREDICTIONS (RAW VS DIAGNOSED)
# ==============================================================
if page == "🧪 Test Predictions (Raw vs Diagnosed)":
    st.title("🧪 Batch Test Predictions — Raw Images vs AI Diagnoses")
    st.write(
        "Evaluate unlabelled field test images from `dataset/test_images/` "
        "before and after clinical diagnosis by **EfficientNetV2-S** (97.38% Test Accuracy)."
    )

    pred_dir = PROJECT_ROOT / "results" / "test_predictions"
    raw_grid_path = pred_dir / "raw_20_images_grid.png"
    pred_grid_path = pred_dir / "all_20_predictions_grid.png"
    master_grid_path = pred_dir / "all_20_raw_vs_predicted_grid.png"
    csv_path = PROJECT_ROOT / "results" / "metrics" / "test_predictions.csv"

    # KPI row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Evaluated Images", "20 Samples")
    kpi2.metric("Mean AI Confidence", "84.34%")
    kpi3.metric("Normal / Healthy", "5 / 20 (25%)")
    kpi4.metric("Pathological / Pest", "15 / 20 (75%)")

    st.divider()

    view_mode = st.radio(
        "Select Inspection View:",
        [
            "🌾 EfficientNetV2-S Multi-Model Grid (1 PNG)",
            "🖼️ All 20 in One Master PNG (Raw vs Diagnosed)",
            "⚡ ONNX FP32 vs INT8 Quantization Comparison (20 Images)",
            "⚖️ Side-by-Side 4x5 Collages (Raw vs Diagnosed)",
            "🌿 Raw Images Only (Without Predictions)",
            "🏷️ Diagnosed Images Only (With Overlays)",
            "🔍 Single-Sample Comparative Inspector"
        ],
        horizontal=True
    )

    eff_grid_path = pred_dir / "efficientnet_v2_s_20_predictions_comparison.png"

    if view_mode == "🌾 EfficientNetV2-S Multi-Model Grid (1 PNG)":
        st.subheader("🌾 EfficientNetV2-S Multi-Model Comparison Grid (1 PNG)")
        st.caption("Complete 20-leaf diagnostic breakdown comparing PyTorch Baseline vs ONNX FP32 vs ONNX INT8 with leaf photos and confidence meters.")
        if eff_grid_path.exists():
            st.image(str(eff_grid_path), caption="EfficientNetV2-S: Multi-Model Parity Verification (2400x2150)", use_container_width=True)
        else:
            st.warning("EfficientNet comparison grid not found.")

    elif view_mode == "🖼️ All 20 in One Master PNG (Raw vs Diagnosed)":
        st.subheader("🖼️ Master 20-Sample Diagnostic Grid (All in One PNG)")
        st.caption("Side-by-side unlabelled field images vs AI diagnostic overlays for all 20 test leaves.")
        if master_grid_path.exists():
            st.image(str(master_grid_path), caption="PaddySnap AI: 20-Sample Master Comparison Grid (3690x2130)", use_container_width=True)
        else:
            st.warning("Master comparison grid not found.")

    elif view_mode == "⚡ ONNX FP32 vs INT8 Quantization Comparison (20 Images)":
        st.subheader("⚡ ONNX FP32 vs INT8 Dynamic Quantization Benchmark (20 Test Images)")
        st.caption("Side-by-side diagnostic consistency and latency comparison on held-out field test leaves.")

        onnx_csv = PROJECT_ROOT / "results" / "metrics" / "onnx_fp32_vs_int8_comparison.csv"
        if onnx_csv.exists():
            df_onnx = pd.read_csv(onnx_csv)
            o1, o2, o3, o4 = st.columns(4)
            o1.metric("Class Agreement", "100.0%", "20 / 20 Identical")
            o2.metric("Size Reduction", "73.9%", "76.87 MB → 20.07 MB")
            o3.metric("Mean Conf. Drift", "1.44%", "Quantization Noise")
            o4.metric("Engine Tested", "EfficientNetV2-S", "Edge CPU")

            st.dataframe(df_onnx, use_container_width=True, hide_index=True)

            # Visual Infographic
            infographic_path = PROJECT_ROOT / "results" / "graphs" / "onnx_fp32_vs_int8_visual_comparison.png"
            if infographic_path.exists():
                st.divider()
                st.subheader("🖼️ Visual Parity & Quantization Benchmark Infographic")
                st.image(str(infographic_path), caption="PaddySnap AI: ONNX FP32 vs INT8 20-Sample Visual Comparison", use_container_width=True)
        else:
            st.warning("ONNX comparison CSV not found.")

    elif view_mode == "⚖️ Side-by-Side 4x5 Collages (Raw vs Diagnosed)":
        c_left, c_right = st.columns(2, gap="medium")
        with c_left:
            st.subheader("🌿 Clean Field Images (Without Predictions)")
            if raw_grid_path.exists():
                st.image(str(raw_grid_path), caption="20 Raw Unlabelled Images", use_container_width=True)
            else:
                st.warning("Raw collage grid not found.")
        with c_right:
            st.subheader("🏷️ AI Diagnosed Predictions (With Banners)")
            if pred_grid_path.exists():
                st.image(str(pred_grid_path), caption="20 AI Classified & Annotated Images", use_container_width=True)
            else:
                st.warning("Predicted collage grid not found.")

    elif view_mode == "🌿 Raw Images Only (Without Predictions)":
        st.subheader("🌿 Clean Field Images (Without Predictions / Ground Truth)")
        if raw_grid_path.exists():
            st.image(str(raw_grid_path), caption="4x5 Collage: 20 Raw Test Images", use_container_width=True)

    elif view_mode == "🏷️ Diagnosed Images Only (With Overlays)":
        st.subheader("🏷️ AI Diagnosed Predictions (With Overlays & Confidence)")
        if pred_grid_path.exists():
            st.image(str(pred_grid_path), caption="4x5 Collage: 20 AI Classified Images", use_container_width=True)

    elif view_mode == "🔍 Single-Sample Comparative Inspector":
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            sample_idx = st.selectbox(
                "Choose a sample to inspect:",
                range(len(df)),
                format_func=lambda i: f"Sample #{df.iloc[i]['ID']}: {df.iloc[i]['Image Filename']} — {df.iloc[i]['Predicted Class']} ({df.iloc[i]['Confidence (%)']}%)"
            )
            row = df.iloc[sample_idx]
            raw_img_file = pred_dir / str(row["Raw Image (Without Predictions)"])
            pred_img_file = pred_dir / str(row["Annotated Image (With Predictions)"])

            col1, col2, col3 = st.columns([1.2, 1.2, 1.6], gap="medium")
            with col1:
                st.markdown("**🌿 Raw Field Image (Without Predictions)**")
                if raw_img_file.exists():
                    st.image(str(raw_img_file), caption=str(row["Raw Image (Without Predictions)"]), use_container_width=True)
            with col2:
                st.markdown("**🏷️ Diagnosed Image (With AI Prediction)**")
                if pred_img_file.exists():
                    st.image(str(pred_img_file), caption=str(row["Annotated Image (With Predictions)"]), use_container_width=True)
            with col3:
                st.markdown("### 📋 Diagnostic Assessment")
                st.write(f"**Image Filename:** `{row['Image Filename']}`")
                st.write(f"**Predicted Pathology:** `{row['Predicted Class']}`")
                st.write(f"**Pathogen / Vector:** {row['Pathogen / Vector Type']}")
                st.write(f"**Confidence:** **{row['Confidence (%)']}%**")
                st.write(f"**Severity Status:** `{row['Priority Status']}`")
                
                # Knowledge base advisory lookup
                dis_key = str(row["Predicted Class"]).lower().replace(" ", "_")
                if dis_key in DISEASE_KNOWLEDGE_BASE:
                    kb = DISEASE_KNOWLEDGE_BASE[dis_key]
                    st.info(f"**Advisory:** {kb['organic_cultural_management'][0]}")

    st.divider()
    st.subheader("📋 20-Sample Batch Predictions Log")
    if csv_path.exists():
        df_log = pd.read_csv(csv_path)
        st.dataframe(df_log, use_container_width=True, hide_index=True)

