# 🌾 PADDYSNAP AI — ONNX & Edge Quantization Summary

| Model Architecture | PyTorch Weights (.pth) | ONNX FP32 (.onnx) | ONNX INT8 (.onnx) | PyTorch Latency | ONNX FP32 Latency | ONNX INT8 Latency | Parity Verified |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **efficientnet_v2_s** | 77.88 MB | 76.87 MB | 20.07 MB | 54.0 ms | 14.5 ms | 65.5 ms | YES |
| **densenet121** | 27.14 MB | 26.94 MB | 7.48 MB | 53.2 ms | 12.9 ms | 41.1 ms | YES |
| **custom_cnn** | 30.01 MB | 29.95 MB | 7.59 MB | 43.0 ms | 5.9 ms | 12.5 ms | YES |
| **resnet50** | 90.06 MB | 89.68 MB | 22.60 MB | 45.2 ms | 12.6 ms | 26.0 ms | YES |
| **convnext_tiny** | 106.22 MB | 106.28 MB | 26.99 MB | 36.3 ms | 27.3 ms | 80.4 ms | YES |
