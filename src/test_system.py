"""
============================================================
PADDYSNAP AI -- MODULE 23: End-to-End System Validation
============================================================
Purpose:
    Automated test suite that verifies:
      1. Dataset integrity & split CSV consistency
      2. All 5 model architecture builders and parameter counts
      3. All 5 saved model checkpoints (.pth) loading & forward pass
      4. Prediction engine & recommendation engine accuracy
      5. Generated artifacts (metrics JSONs, history CSVs, confusion matrices, graphs)

Usage:
    python src/test_system.py
============================================================
"""

import sys
import json
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pandas as pd
from PIL import Image

from src.utils import get_device, set_seed
from src.models import get_model, count_parameters, SUPPORTED_MODELS
from src.predict import predict_image
from src.recommendations import DISEASE_KNOWLEDGE_BASE, get_recommendation


def test_suite():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    set_seed(42)
    device = get_device()

    print("=" * 70)
    print("  PADDYSNAP AI -- SYSTEM VALIDATION TEST SUITE")
    print("=" * 70)

    passed = 0
    total_tests = 0

    # ----------------------------------------------------------
    # TEST 1: Dataset & Split CSV Integrity
    # ----------------------------------------------------------
    total_tests += 1
    print("\n[TEST 1] Verifying Dataset Splits & Class Mapping...")
    try:
        train_df = pd.read_csv("dataset/train_split.csv")
        val_df   = pd.read_csv("dataset/val_split.csv")
        test_df  = pd.read_csv("dataset/test_split.csv")
        with open("dataset/class_mapping.json") as f:
            mapping = json.load(f)

        assert len(train_df) == 7284, f"Train split count mismatch: {len(train_df)}"
        assert len(val_df)   == 1561, f"Val split count mismatch: {len(val_df)}"
        assert len(test_df)  == 1562, f"Test split count mismatch: {len(test_df)}"
        assert len(mapping["class_to_id"]) == 10, "Class count mismatch"
        print(f"  --> PASSED: 10,407 total images split correctly (7,284 Train, 1,561 Val, 1,562 Test)")
        passed += 1
    except Exception as e:
        print(f"  --> FAILED: {e}")

    # ----------------------------------------------------------
    # TEST 2: Model Architecture Builders & Checkpoints
    # ----------------------------------------------------------
    total_tests += 1
    print("\n[TEST 2] Verifying Checkpoints & GPU Forward Passes...")
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    models_ok = True
    for model_name in SUPPORTED_MODELS.keys():
        ckpt_path = Path(f"checkpoints/{model_name}_best.pth")
        if not ckpt_path.exists():
            print(f"  --> FAILED: Missing checkpoint {ckpt_path}")
            models_ok = False
            continue

        try:
            ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
            model = get_model(model_name, num_classes=10, pretrained=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model = model.to(device)
            model.eval()

            with torch.no_grad():
                out = model(dummy_input)
            assert out.shape == (1, 10), f"Output shape mismatch: {out.shape}"
            print(f"    OK: {model_name:<20} | Weights Loaded | GPU Output Shape: {tuple(out.shape)}")
        except Exception as e:
            print(f"    ERROR: {model_name} failed: {e}")
            models_ok = False

    if models_ok:
        print("  --> PASSED: All 5 model checkpoints verified on GPU.")
        passed += 1

    # ----------------------------------------------------------
    # TEST 3: Prediction Engine & Agronomy Advisory
    # ----------------------------------------------------------
    total_tests += 1
    print("\n[TEST 3] Verifying Single-Image Prediction & Agronomy Engine...")
    sample_img = "dataset/train_images/blast/100004.jpg"
    try:
        res = predict_image(sample_img, model_name="efficientnet_v2_s", device=device)
        assert res["predicted_class"] == "blast", f"Expected blast, got {res['predicted_class']}"
        assert res["confidence"] > 50.0, f"Confidence too low: {res['confidence']}"
        assert "recommendation" in res, "Missing recommendation object"
        assert res["recommendation"]["common_name"] != "", "Empty common name"
        print(f"  --> PASSED: Correctly diagnosed Blast with {res['confidence']:.2f}% confidence & agronomy guidance.")
        passed += 1
    except Exception as e:
        print(f"  --> FAILED: {e}")

    # ----------------------------------------------------------
    # TEST 4: Agronomy Knowledge Base Coverage
    # ----------------------------------------------------------
    total_tests += 1
    print("\n[TEST 4] Verifying Agronomy Knowledge Base (All 10 Classes)...")
    try:
        with open("dataset/class_mapping.json") as f:
            classes = list(json.load(f)["class_to_id"].keys())

        for cls in classes:
            rec = get_recommendation(cls)
            assert "symptoms" in rec and len(rec["symptoms"]) > 0, f"Missing symptoms for {cls}"
            assert "chemical_control" in rec, f"Missing chemical control for {cls}"
            assert "fertilizer_nutrient_guidance" in rec, f"Missing fertilizer guidance for {cls}"
        print(f"  --> PASSED: Full clinical protocols and fertilizer rules verified for all 10 classes.")
        passed += 1
    except Exception as e:
        print(f"  --> FAILED: {e}")

    # ----------------------------------------------------------
    # TEST 5: Results & Visual Artifacts Verification
    # ----------------------------------------------------------
    total_tests += 1
    print("\n[TEST 5] Verifying Generated Artifacts & Comparison Tables...")
    try:
        assert Path("results/metrics/model_comparison.csv").exists(), "Missing model_comparison.csv"
        assert Path("results/metrics/model_comparison.md").exists(), "Missing model_comparison.md"
        assert Path("results/graphs/model_comparison_table.png").exists(), "Missing model_comparison_table.png"
        assert Path("results/graphs/model_comparison_bar_chart.png").exists(), "Missing model_comparison_bar_chart.png"
        assert len(list(Path("results/confusion_matrices").glob("*.png"))) == 5, "Missing confusion matrices"
        assert len(list(Path("results/metrics").glob("*_metrics.json"))) == 5, "Missing JSON metrics"
        assert len(list(Path("results/training_history").glob("*.csv"))) == 5, "Missing training history CSVs"
        print(f"  --> PASSED: All 5 CSVs, 5 JSONs, 5 Confusion Matrices, and comparison charts present.")
        passed += 1
    except Exception as e:
        print(f"  --> FAILED: {e}")

    # ----------------------------------------------------------
    # TEST 6: Pure Weights Only Checkpoints (Optimizer Removed)
    # ----------------------------------------------------------
    total_tests += 1
    print("\n[TEST 6] Verifying Pure Weight Files (checkpoints/weights_only/)...")
    weights_ok = True
    for model_name in SUPPORTED_MODELS.keys():
        weights_file = Path(f"checkpoints/weights_only/{model_name}_weights.pth")
        if not weights_file.exists():
            print(f"  --> FAILED: Missing weights file {weights_file}")
            weights_ok = False
            continue
        try:
            state_dict = torch.load(str(weights_file), map_location="cpu", weights_only=True)
            assert isinstance(state_dict, dict) and len(state_dict) > 0
            file_mb = weights_file.stat().st_size / (1024 * 1024)
            print(f"    OK: {model_name:<20} | Weights Loaded | Size: {file_mb:.2f} MB")
        except Exception as e:
            print(f"    ERROR loading {weights_file}: {e}")
            weights_ok = False

    if weights_ok:
        print("  --> PASSED: All 5 pure weights checkpoints verified.")
        passed += 1

    # ----------------------------------------------------------
    # TEST 7: ONNX Models (FP32 & INT8 Quantized) Runtime Inference
    # ----------------------------------------------------------
    total_tests += 1
    print("\n[TEST 7] Verifying ONNX Models (FP32 & INT8 Quantized) via ONNX Runtime...")
    try:
        import onnxruntime as ort
        import numpy as np
        onnx_ok = True
        dummy_onnx_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

        for model_name in SUPPORTED_MODELS.keys():
            fp32_path = Path(f"deployment/onnx/{model_name}.onnx")
            int8_path = Path(f"deployment/onnx/{model_name}_int8.onnx")

            if not fp32_path.exists() or not int8_path.exists():
                print(f"  --> FAILED: Missing ONNX files for {model_name}")
                onnx_ok = False
                continue

            session_fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
            out_fp32 = session_fp32.run(None, {"input": dummy_onnx_input})[0]
            assert out_fp32.shape == (1, 10), f"FP32 shape mismatch: {out_fp32.shape}"

            session_int8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
            out_int8 = session_int8.run(None, {"input": dummy_onnx_input})[0]
            assert out_int8.shape == (1, 10), f"INT8 shape mismatch: {out_int8.shape}"

            fp32_mb = fp32_path.stat().st_size / (1024 * 1024)
            int8_mb = int8_path.stat().st_size / (1024 * 1024)
            print(f"    OK: {model_name:<20} | FP32: {fp32_mb:6.2f} MB | INT8: {int8_mb:5.2f} MB | Both Run Cleanly")

        if onnx_ok:
            print("  --> PASSED: All 10 ONNX models verified with working inference on ONNX Runtime.")
            passed += 1
    except Exception as e:
        print(f"  --> FAILED ONNX tests: {e}")

    # ----------------------------------------------------------
    # TEST 8: 20-Sample Diagnostic Grids & Comparison Artifacts
    # ----------------------------------------------------------
    total_tests += 1
    print("\n[TEST 8] Verifying Multi-Model Visual Artifacts & 20-Sample Test Grids...")
    try:
        assert Path("deployment/onnx/onnx_model_comparison.csv").exists(), "Missing onnx_model_comparison.csv"
        assert Path("results/metrics/onnx_fp32_vs_int8_comparison.csv").exists(), "Missing onnx_fp32_vs_int8_comparison.csv"
        assert Path("results/metrics/test_predictions.csv").exists(), "Missing test_predictions.csv"
        assert Path("results/metrics/test_results.html").exists(), "Missing test_results.html"
        assert Path("results/graphs/onnx_fp32_vs_int8_visual_comparison.png").exists(), "Missing ONNX infographic"
        assert Path("results/test_predictions/efficientnet_v2_s_20_predictions_comparison.png").exists(), "Missing effnet grid"
        assert Path("results/test_predictions/all_20_raw_vs_predicted_grid.png").exists(), "Missing master grid"
        assert len(list(Path("results/test_predictions").glob("raw_*.jpg"))) == 20, "Missing raw test photos"
        assert len(list(Path("results/test_predictions").glob("pred_*.jpg"))) == 20, "Missing annotated test photos"
        print("  --> PASSED: All multi-model comparison CSVs, HTML viewers, and PNG visual grids verified.")
        passed += 1
    except Exception as e:
        print(f"  --> FAILED: {e}")

    # ----------------------------------------------------------
    # FINAL SUMMARY
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"  SYSTEM VALIDATION SUMMARY: {passed} / {total_tests} TESTS PASSED ({passed/total_tests*100:.1f}%)")
    print("=" * 70)

    if passed == total_tests:
        print("  🎉 ALL SYSTEM MODULES ARE FULLY OPERATIONAL AND VERIFIED!")
    else:
        print("  ⚠️ Some tests failed. Please review errors above.")


if __name__ == "__main__":
    test_suite()
