# 🌾 PADDYSNAP AI — ONNX FP32 vs INT8 20-Sample Test Comparison

**Target Model:** EfficientNetV2-S (Production Champion)
**FP32 Model File:** `deployment/onnx/efficientnet_v2_s.onnx` (76.87 MB)
**INT8 Model File:** `deployment/onnx/efficientnet_v2_s_int8.onnx` (20.07 MB — **73.9% smaller**)
**Cohort Size:** 20 Held-Out Field Images (`dataset/test_images/`)

## 📌 Executive Summary

* **Class Agreement Rate:** **100.0% (20 / 20 samples perfectly identical)**
* **Mean Absolute Confidence Drift:** **1.44%** (negligible quantization noise)
* **Mean Inference Latency:** FP32 = **65.9 ms** | INT8 = **128.7 ms** (CPU single-thread)

---

## 🖼️ Visual Benchmark Infographic

![ONNX FP32 vs INT8 Visual Benchmark](../graphs/onnx_fp32_vs_int8_visual_comparison.png)

*Direct Link to Image:* [`results/graphs/onnx_fp32_vs_int8_visual_comparison.png`](file:///c:/Users/SUMITRANAND%20SHARMA/OneDrive/Desktop/embeded_system/paddy-guard-ai/results/graphs/onnx_fp32_vs_int8_visual_comparison.png)

---

## 📊 Sample-by-Sample Diagnostic Comparison Log

| # | Image Filename | PyTorch Baseline | ONNX FP32 Class | ONNX FP32 Conf | ONNX INT8 Class | ONNX INT8 Conf | Δ Conf | Agreement |
|:---:|:---|:---|:---|:---:|:---|:---:|:---:|:---:|
| 01 | `202620.jpg` | Hispa | **Hispa** | 79.62% | **Hispa** | 78.49% | -1.12% | 🟢 MATCH |
| 02 | `200457.jpg` | Hispa | **Hispa** | 81.03% | **Hispa** | 81.86% | +0.82% | 🟢 MATCH |
| 03 | `200103.jpg` | Hispa | **Hispa** | 79.78% | **Hispa** | 78.23% | -1.55% | 🟢 MATCH |
| 04 | `203038.jpg` | Dead Heart | **Dead Heart** | 84.78% | **Dead Heart** | 83.15% | -1.62% | 🟢 MATCH |
| 05 | `201127.jpg` | Normal | **Normal** | 79.27% | **Normal** | 78.98% | -0.29% | 🟢 MATCH |
| 06 | `201004.jpg` | Normal | **Normal** | 79.97% | **Normal** | 80.25% | +0.28% | 🟢 MATCH |
| 07 | `200915.jpg` | Normal | **Normal** | 79.71% | **Normal** | 79.96% | +0.25% | 🟢 MATCH |
| 08 | `200572.jpg` | Normal | **Normal** | 82.62% | **Normal** | 81.69% | -0.93% | 🟢 MATCH |
| 09 | `203017.jpg` | Dead Heart | **Dead Heart** | 83.8% | **Dead Heart** | 81.85% | -1.95% | 🟢 MATCH |
| 10 | `200420.jpg` | Tungro | **Tungro** | 91.65% | **Tungro** | 90.22% | -1.42% | 🟢 MATCH |
| 11 | `202772.jpg` | Blast | **Blast** | 81.6% | **Blast** | 79.7% | -1.90% | 🟢 MATCH |
| 12 | `203034.jpg` | Hispa | **Hispa** | 83.25% | **Hispa** | 82.45% | -0.80% | 🟢 MATCH |
| 13 | `202234.jpg` | Tungro | **Tungro** | 90.25% | **Tungro** | 88.52% | -1.73% | 🟢 MATCH |
| 14 | `200357.jpg` | Bacterial Leaf Streak | **Bacterial Leaf Streak** | 96.4% | **Bacterial Leaf Streak** | 96.38% | -0.02% | 🟢 MATCH |
| 15 | `202419.jpg` | Hispa | **Hispa** | 81.0% | **Hispa** | 79.04% | -1.97% | 🟢 MATCH |
| 16 | `201729.jpg` | Tungro | **Tungro** | 87.81% | **Tungro** | 87.59% | -0.23% | 🟢 MATCH |
| 17 | `200131.jpg` | Brown Spot | **Brown Spot** | 86.87% | **Brown Spot** | 80.65% | -6.22% | 🟢 MATCH |
| 18 | `200123.jpg` | Dead Heart | **Dead Heart** | 85.0% | **Dead Heart** | 81.33% | -3.67% | 🟢 MATCH |
| 19 | `200384.jpg` | Bacterial Leaf Blight | **Bacterial Leaf Blight** | 95.48% | **Bacterial Leaf Blight** | 94.48% | -0.99% | 🟢 MATCH |
| 20 | `200896.jpg` | Normal | **Normal** | 80.02% | **Normal** | 79.04% | -0.98% | 🟢 MATCH |
