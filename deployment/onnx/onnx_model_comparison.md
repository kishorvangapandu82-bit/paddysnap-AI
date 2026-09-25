# 🌾 PADDYSNAP AI — ONNX & Edge Quantization Summary

**Export Destination:** [`deployment/onnx/`](file:///c:/Users/SUMITRANAND%20SHARMA/OneDrive/Desktop/embeded_system/paddy-guard-ai/deployment/onnx)  
**Input Resolution:** `(1, 3, 224, 224)` with Dynamic Batching `(N, 3, 224, 224)`  
**ONNX Opset Version:** `14` (Compatible with Raspberry Pi, Jetson Nano, Coral, Android, iOS)  

---

## 📊 Comprehensive ONNX Export & Benchmark Table

| Model Architecture | PyTorch Weights (`.pth`) | ONNX FP32 (`.onnx`) | ONNX INT8 (`.onnx`)* | PyTorch CPU Latency | ONNX FP32 Latency | Parity Verified |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|


| 🥇 **EfficientNetV2-S** *(Champion)* | `77.88 MB` | [`76.87 MB`](file:///c:/Users/SUMITRANAND%20SHARMA/OneDrive/Desktop/embeded_system/paddy-guard-ai/deployment/onnx/efficientnet_v2_s.onnx) | [`20.07 MB`](file:///c:/Users/SUMITRANAND%20SHARMA/OneDrive/Desktop/embeded_system/paddy-guard-ai/deployment/onnx/efficientnet_v2_s_int8.onnx) | 40.7 ms | **13.0 ms** *(3.1x faster)* | **YES** |
| 🥈 **ConvNeXt-Tiny** | `106.22 MB` | [`106.28 MB`](file:///c:/Users/SUMITRANAND%20SHARMA/OneDrive/Desktop/embeded_system/paddy-guard-ai/deployment/onnx/convnext_tiny.onnx) | [`26.99 MB`](file:///c:/Users/SUMITRANAND%20SHARMA/OneDrive/Desktop/embeded_system/paddy-guard-ai/deployment/onnx/convnext_tiny_int8.onnx) | 40.8 ms | **20.6 ms** *(2.0x faster)* | **YES** |
| 🥉 **ResNet-50** | `90.06 MB` | [`89.68 MB`](file:///c:/Users/SUMITRANAND%20SHARMA/OneDrive/Desktop/embeded_system/paddy-guard-ai/deployment/onnx/resnet50.onnx) | [`22.60 MB`](file:///c:/Users/SUMITRANAND%20SHARMA/OneDrive/Desktop/embeded_system/paddy-guard-ai/deployment/onnx/resnet50_int8.onnx) | 43.7 ms | **12.5 ms** *(3.5x faster)* | **YES** |
| 4 **DenseNet-121** | `27.14 MB` | [`26.94 MB`](file:///c:/Users/SUMITRANAND%20SHARMA/OneDrive/Desktop/embeded_system/paddy-guard-ai/deployment/onnx/densenet121.onnx) | [`7.48 MB`](file:///c:/Users/SUMITRANAND%20SHARMA/OneDrive/Desktop/embeded_system/paddy-guard-ai/deployment/onnx/densenet121_int8.onnx) | 62.2 ms | **11.6 ms** *(5.4x faster)* | **YES** |
| 5 **Custom PaddySnapNet** | `30.01 MB` | [`29.95 MB`](file:///c:/Users/SUMITRANAND%20SHARMA/OneDrive/Desktop/embeded_system/paddy-guard-ai/deployment/onnx/custom_cnn.onnx) | [`7.59 MB`](file:///c:/Users/SUMITRANAND%20SHARMA/OneDrive/Desktop/embeded_system/paddy-guard-ai/deployment/onnx/custom_cnn_int8.onnx) | 40.4 ms | **5.8 ms** *(7.0x faster)* | **YES** |

*\*INT8 dynamic quantization reduces storage and memory by ~74% for edge hardware.*
