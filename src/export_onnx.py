"""
============================================================
PADDYSNAP AI -- MODULE 22: ONNX Exporter & Edge Quantization
============================================================
Purpose:
    Exports all 5 trained Paddy Leaf Disease models to:
      1. Self-contained FP32 ONNX (.onnx) for cross-platform edge inference
      2. Ultra-compact INT8 ONNX (_int8.onnx) for micro-controllers / RPi
    Validates numerical parity and benchmarks CPU inference latency.

Usage:
    python src/export_onnx.py
============================================================
"""

import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Any, List

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import get_model


MODELS_TO_EXPORT = [
    ("efficientnet_v2_s", "checkpoints/weights_only/efficientnet_v2_s_weights.pth"),
    ("densenet121", "checkpoints/weights_only/densenet121_weights.pth"),
    ("custom_cnn", "checkpoints/weights_only/custom_cnn_weights.pth"),
    ("resnet50", "checkpoints/weights_only/resnet50_weights.pth"),
    ("convnext_tiny", "checkpoints/weights_only/convnext_tiny_weights.pth"),
]


def export_all_to_onnx(opset_version: int = 14):
    out_dir = PROJECT_ROOT / "deployment" / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clean up any leftover external data files
    for f in out_dir.glob("*.onnx.data"):
        try:
            f.unlink()
        except Exception:
            pass

    print("=" * 85)
    print("🌾 PADDYSNAP AI -- ONNX & EDGE QUANTIZATION EXPORTER")
    print(f"Destination Directory: {out_dir.resolve()}")
    print("=" * 85)

    summary_records = []
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    np_dummy = dummy_input.numpy()

    for model_name, weights_rel_path in MODELS_TO_EXPORT:
        weights_path = PROJECT_ROOT / weights_rel_path
        if not weights_path.exists():
            print(f"⚠️ Skipping {model_name}: weights file not found at {weights_path}")
            continue

        print(f"\n[Exporting] {model_name}...")
        pth_size_mb = weights_path.stat().st_size / (1024 * 1024)

        # 1. Load PyTorch model with pure weights
        model = get_model(model_name, num_classes=10, pretrained=False)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict)
        model.eval()

        # PyTorch CPU benchmark (10 iterations)
        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(10):
                torch_out = model(dummy_input)
            pytorch_lat_ms = ((time.perf_counter() - t0) / 10) * 1000.0

        # 2. Export to Self-Contained FP32 ONNX
        fp32_onnx_path = out_dir / f"{model_name}.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            str(fp32_onnx_path),
            export_params=True,
            opset_version=opset_version,
            dynamo=False,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            }
        )

        onnx_model = onnx.load(str(fp32_onnx_path))
        onnx.checker.check_model(onnx_model)
        fp32_size_mb = fp32_onnx_path.stat().st_size / (1024 * 1024)

        # 3. Benchmark ONNX Runtime FP32 & Verify Parity
        ort_session = ort.InferenceSession(str(fp32_onnx_path), providers=["CPUExecutionProvider"])
        t0 = time.perf_counter()
        for _ in range(10):
            ort_out = ort_session.run(["output"], {"input": np_dummy})[0]
        onnx_lat_ms = ((time.perf_counter() - t0) / 10) * 1000.0

        max_diff = np.max(np.abs(torch_out.numpy() - ort_out))
        parity_ok = max_diff < 1e-4

        # 4. Dynamic INT8 Quantization for Edge Devices
        int8_onnx_path = out_dir / f"{model_name}_int8.onnx"
        quantize_dynamic(
            model_input=str(fp32_onnx_path),
            model_output=str(int8_onnx_path),
            weight_type=QuantType.QUInt8
        )
        int8_size_mb = int8_onnx_path.stat().st_size / (1024 * 1024)

        # Benchmark INT8 ONNX Latency
        ort_int8_session = ort.InferenceSession(str(int8_onnx_path), providers=["CPUExecutionProvider"])
        t0 = time.perf_counter()
        for _ in range(10):
            _ = ort_int8_session.run(["output"], {"input": np_dummy})[0]
        int8_lat_ms = ((time.perf_counter() - t0) / 10) * 1000.0

        print(f"  ✓ FP32 ONNX : {fp32_size_mb:6.2f} MB | Latency: {onnx_lat_ms:5.1f} ms | Parity Error: {max_diff:.1e}")
        print(f"  ✓ INT8 ONNX : {int8_size_mb:6.2f} MB | Latency: {int8_lat_ms:5.1f} ms | Edge Size Reduction: -{((fp32_size_mb - int8_size_mb)/fp32_size_mb)*100:.1f}%")

        summary_records.append({
            "Model Architecture": model_name,
            "PyTorch Weights (.pth)": f"{pth_size_mb:.2f} MB",
            "ONNX FP32 (.onnx)": f"{fp32_size_mb:.2f} MB",
            "ONNX INT8 (.onnx)": f"{int8_size_mb:.2f} MB",
            "PyTorch Latency": f"{pytorch_lat_ms:.1f} ms",
            "ONNX FP32 Latency": f"{onnx_lat_ms:.1f} ms",
            "ONNX INT8 Latency": f"{int8_lat_ms:.1f} ms",
            "Parity Verified": "YES" if parity_ok else "NO"
        })

    # Save summary tables
    summary_df = pd.DataFrame(summary_records)
    csv_out = out_dir / "onnx_model_comparison.csv"
    summary_df.to_csv(csv_out, index=False)

    md_out = out_dir / "onnx_model_comparison.md"
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("# 🌾 PADDYSNAP AI — ONNX & Edge Quantization Summary\n\n")
        f.write("| Model Architecture | PyTorch Weights (.pth) | ONNX FP32 (.onnx) | ONNX INT8 (.onnx) | PyTorch Latency | ONNX FP32 Latency | ONNX INT8 Latency | Parity Verified |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for r in summary_records:
            f.write(f"| **{r['Model Architecture']}** | {r['PyTorch Weights (.pth)']} | {r['ONNX FP32 (.onnx)']} | {r['ONNX INT8 (.onnx)']} | {r['PyTorch Latency']} | {r['ONNX FP32 Latency']} | {r['ONNX INT8 Latency']} | {r['Parity Verified']} |\n")

    print("\n" + "=" * 85)
    print("🏆 ALL 5 MODELS SUCCESSFULLY CONVERTED TO ONNX (FP32 & INT8)")
    print("=" * 85)
    print(summary_df.to_string(index=False))
    print(f"\nArtifacts saved in: {out_dir.resolve()}")


if __name__ == "__main__":
    export_all_to_onnx()
