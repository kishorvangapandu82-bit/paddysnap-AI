# 🌾 Paddy Guard AI — Edge Deployment Package

Standalone, lightweight edge deployment folder containing **EfficientNetV2-S (Quantized INT8)**.
Requires **ZERO PyTorch dependency**!

---

## 📁 Package Structure

```text
edge_deployment/
├── model/
│   └── efficientnet_v2_s_int8.onnx      (20.07 MB - 97.76% Test Accuracy)
├── metadata/
│   ├── agronomy_rules.json              (100% Offline Disease & Fertilizer Rules)
│   ├── class_mapping.json               (Class IDs to names)
│   └── labels.txt                       (Plain label list)
├── run_inference.py                     (Standalone Python Inference Script)
├── requirements.txt                     (Minimal requirements: onnxruntime, numpy, Pillow)
└── README.md                            (Documentation)
```

---

## ⚡ Quickstart

### 1. Install Minimal Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Inference on Any Leaf Image
```bash
python run_inference.py --image path/to/leaf.jpg
```
