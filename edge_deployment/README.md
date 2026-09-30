# 🌾 PaddySnap AI — Edge Deployment Package

**Standalone Raspberry Pi / Edge Device Inference**
No PyTorch. No GPU. Only `onnxruntime`, `numpy`, and `Pillow`.

---

## 📁 Folder Structure

```
edge_deployment/
├── model/
│   └── efficientnet_v2_s_int8.onnx   # INT8 quantized model (20 MB)
├── metadata/
│   ├── class_mapping.json             # ID → class name mapping
│   ├── agronomy_rules.json            # Disease → fertilizer guidance
│   └── labels.txt                     # Plain text class list
├── test_images/                       # 20 held-out field images for testing
│   ├── 200103.jpg
│   ├── 200123.jpg
│   └── ... (20 total)
├── run_inference.py                   # Single image inference
├── batch_inference.py                 # Batch inference on all test_images/
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## ⚙️ Requirements

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
onnxruntime>=1.17.0
numpy>=1.24.0
Pillow>=10.0.0
```

---

## 🚀 Usage

### Single Image Inference
```bash
python run_inference.py --image test_images/202620.jpg
```

**Output:**
```
====================================================================
PADDY GUARD AI -- EDGE DIAGNOSIS SYSTEM (EFFICIENTNETV2-S)
====================================================================
Image Path:      202620.jpg
Preprocess Time: 4.21 ms
Inference Time:  38.74 ms
--------------------------------------------------------------------
Diagnosis:       Hispa (Dicladispa armigera)
Confidence:      99.99%
--------------------------------------------------------------------
FERTILIZER & NUTRIENT GUIDANCE:
  * Nitrogen (N):    Reduce N application temporarily during outbreak
  * Potassium (K):   Maintain K levels to improve plant resistance
  * Micronutrients:  Apply Zinc sulfate to strengthen cell walls
====================================================================
```

### Batch Inference (All 20 Test Images)
```bash
python batch_inference.py
```

Or specify a custom folder and output CSV:
```bash
python batch_inference.py --img_dir test_images --save_csv batch_results.csv
```

---

## 📊 Model Info

| Property         | Value                          |
|-----------------|-------------------------------|
| Architecture    | EfficientNetV2-S               |
| Format          | ONNX INT8 Quantized            |
| Model Size      | ~20 MB (73.9% smaller than FP32) |
| Accuracy        | 97.76% (Top Rank — 10 classes) |
| Inference Speed | ~35–70 ms on CPU (Pi 4)        |
| Classes         | 10 paddy disease classes       |

---

## 🌿 Supported Disease Classes

| Class                    | Priority      |
|--------------------------|--------------|
| Normal                   | ✅ Routine    |
| Hispa                    | ⚠️ High       |
| Dead Heart               | ⚠️ High       |
| Tungro                   | ⚠️ High       |
| Blast                    | ⚠️ High       |
| Bacterial Leaf Streak    | ⚠️ High       |
| Bacterial Leaf Blight    | ⚠️ High       |
| Bacterial Panicle Blight | ⚠️ High       |
| Brown Spot               | ⚠️ High       |
| Downy Mildew             | ⚠️ High       |

---

## 📱 Raspberry Pi Deployment

```bash
# On Raspberry Pi 4 (64-bit OS)
pip install onnxruntime-linux-aarch64 numpy Pillow

# Run single image
python run_inference.py --image test_images/202620.jpg

# Run all 20 test images
python batch_inference.py
```

---

*PaddySnap AI — EfficientNetV2-S Champion @ 97.76% | 100% Clinical Parity Verified*
