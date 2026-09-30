"""
============================================================
PADDY GUARD AI -- STANDALONE EDGE INFERENCE (EFFICIENTNETV2-S)
============================================================
Requires ONLY: onnxruntime, numpy, Pillow.
Zero PyTorch dependency!
============================================================
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import onnxruntime as ort

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "model" / "efficientnet_v2_s_int8.onnx"
LABELS_PATH = SCRIPT_DIR / "metadata" / "class_mapping.json"
AGRONOMY_PATH = SCRIPT_DIR / "metadata" / "agronomy_rules.json"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(image_path: str, image_size: int = 224) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0)


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def predict(image_path: str):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 2

    session = ort.InferenceSession(str(MODEL_PATH), opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    t0 = time.perf_counter()
    tensor = preprocess_image(image_path)
    t1 = time.perf_counter()
    outputs = session.run(None, {input_name: tensor})[0]
    t2 = time.perf_counter()

    probs = softmax(outputs[0])
    pred_id = int(np.argmax(probs))
    conf = float(probs[pred_id]) * 100.0

    with open(LABELS_PATH, "r") as f:
        id_to_class = json.load(f)["id_to_class"]
    with open(AGRONOMY_PATH, "r") as f:
        agronomy = json.load(f)

    pred_key = id_to_class[str(pred_id)]
    rec = agronomy.get(pred_key, agronomy.get("normal", {}))

    common_name = rec.get("common_name", pred_key)
    fg = rec.get("fertilizer_nutrient_guidance", {})

    print(f"\n" + "="*68)
    print(f"PADDY GUARD AI -- EDGE DIAGNOSIS SYSTEM (EFFICIENTNETV2-S)")
    print(f"="*68)
    print(f"Image Path:      {Path(image_path).name}")
    print(f"Preprocess Time: {(t1 - t0)*1000:.2f} ms")
    print(f"Inference Time:  {(t2 - t1)*1000:.2f} ms")
    print(f"-"*68)
    print(f"Diagnosis:       {common_name}")
    print(f"Confidence:      {conf:.2f}%")
    print(f"-"*68)
    print(f"FERTILIZER & NUTRIENT GUIDANCE:")
    print(f"  * Nitrogen (N):    {fg.get('nitrogen_action', 'N/A')}")
    print(f"  * Potassium (K):   {fg.get('potassium_action', 'N/A')}")
    print(f"  * Micronutrients:  {fg.get('micronutrients', 'N/A')}")
    print(f"="*68 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone EfficientNetV2 Edge Predictor")
    parser.add_argument("--image", type=str, required=True, help="Path to input leaf image")
    args = parser.parse_args()
    predict(args.image)
