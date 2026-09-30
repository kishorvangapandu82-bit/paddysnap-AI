"""
============================================================
PADDY GUARD AI -- BATCH EDGE INFERENCE (20 TEST IMAGES)
============================================================
Runs the INT8 ONNX model on all images in test_images/
and prints a full summary table + saves results to CSV.
No PyTorch required â€” only onnxruntime, numpy, Pillow.

Usage:
    python batch_inference.py
    python batch_inference.py --img_dir test_images --save_csv results.csv
============================================================
"""

import os, json, time, argparse, csv
from pathlib import Path
import numpy as np
from PIL import Image
import onnxruntime as ort

SCRIPT_DIR    = Path(__file__).resolve().parent
MODEL_PATH    = SCRIPT_DIR / "model"    / "efficientnet_v2_s_int8.onnx"
LABELS_PATH   = SCRIPT_DIR / "metadata" / "class_mapping.json"
AGRONOMY_PATH = SCRIPT_DIR / "metadata" / "agronomy_rules.json"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# â”€â”€ Load model once â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.intra_op_num_threads = 2
session    = ort.InferenceSession(str(MODEL_PATH), opts,
                                  providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
print(f"  Model loaded: {MODEL_PATH.name}")

with open(LABELS_PATH)   as f: id_to_class = json.load(f)["id_to_class"]
with open(AGRONOMY_PATH) as f: agronomy    = json.load(f)


def preprocess(path):
    img = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = (np.array(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return np.expand_dims(np.transpose(arr, (2, 0, 1)), 0)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def run_batch(img_dir: Path, save_csv: Path | None = None):
    images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not images:
        print(f"  No images found in {img_dir}")
        return

    print(f"\n  Running inference on {len(images)} images...\n")
    header = f"  {'#':<4} {'Image':<18} {'Diagnosis':<28} {'Conf':>7}  {'Latency':>9}"
    print(header)
    print("  " + "-" * 72)

    records  = []
    latencies = []

    for i, img_path in enumerate(images, 1):
        tensor = preprocess(img_path)
        t0     = time.perf_counter()
        out    = session.run(None, {input_name: tensor})[0]
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat_ms)

        probs   = softmax(out[0])
        pred_id = int(np.argmax(probs))
        conf    = float(probs[pred_id]) * 100.0
        key     = id_to_class[str(pred_id)]
        rec     = agronomy.get(key, {})
        label   = rec.get("common_name", key.replace("_", " ").title())

        print(f"  {i:<4} {img_path.name:<18} {label:<28} {conf:>6.2f}%  {lat_ms:>7.1f} ms")
        records.append({
            "id":        i,
            "filename":  img_path.name,
            "diagnosis": label,
            "conf_pct":  round(conf, 2),
            "latency_ms": round(lat_ms, 1),
        })

    # â”€â”€ Summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n  " + "-" * 72)
    print(f"  Total images  : {len(records)}")
    print(f"  Mean latency  : {np.mean(latencies):.1f} ms  |  "
          f"Min: {np.min(latencies):.1f} ms  |  Max: {np.max(latencies):.1f} ms")
    print(f"  Mean confidence: {np.mean([r['conf_pct'] for r in records]):.2f}%")

    # â”€â”€ Save CSV â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if save_csv:
        with open(save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        print(f"\n  Results saved â†’ {save_csv}")

    print("  DONE.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Edge Inference â€” PaddySnap AI")
    parser.add_argument("--img_dir",  default="test_images",  help="Folder of images to process")
    parser.add_argument("--save_csv", default="batch_results.csv", help="Output CSV filename")
    args = parser.parse_args()

    img_dir  = SCRIPT_DIR / args.img_dir
    save_csv = SCRIPT_DIR / args.save_csv
    run_batch(img_dir, save_csv)
