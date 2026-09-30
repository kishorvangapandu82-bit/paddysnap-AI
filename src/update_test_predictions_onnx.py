"""
Run real ONNX FP32 + INT8 inference on the 20 test images,
update test_predictions.csv with accurate confidence scores,
then regenerate the comparison PNG.
"""
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Paths ────────────────────────────────────────────────
ref_csv      = PROJECT_ROOT / "results" / "metrics" / "test_predictions.csv"
fp32_model   = PROJECT_ROOT / "deployment" / "onnx" / "efficientnet_v2_s.onnx"
int8_model   = PROJECT_ROOT / "deployment" / "onnx" / "efficientnet_v2_s_int8.onnx"
class_map    = PROJECT_ROOT / "dataset" / "class_mapping.json"
test_img_dir = PROJECT_ROOT / "dataset" / "test_images"

# ── Load references ──────────────────────────────────────
ref_df = pd.read_csv(ref_csv)
with open(class_map) as f:
    id_to_class = json.load(f)["id_to_class"]

# ── ImageNet normalisation ───────────────────────────────
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(img_path):
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    arr = (np.array(img, dtype=np.float32) / 255.0 - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0)

def softmax_conf(logits, idx):
    e = np.exp(logits - np.max(logits))
    return float(e[0, idx] / np.sum(e)) * 100.0

# ── Load ONNX sessions ───────────────────────────────────
print("  Loading ONNX FP32 model...")
sess_fp32 = ort.InferenceSession(str(fp32_model), providers=["CPUExecutionProvider"])
print("  Loading ONNX INT8 model...")
sess_int8 = ort.InferenceSession(str(int8_model), providers=["CPUExecutionProvider"])
print()

# ── Run inference ────────────────────────────────────────
print(f"  {'#':<4} {'Image':<18} {'PyTorch':>8} {'FP32':>8} {'INT8':>8} {'Delta':>7}  Match")
print("  " + "-" * 65)

fp32_confs, int8_confs, deltas = [], [], []

for idx, row in ref_df.iterrows():
    img_path = test_img_dir / row["Image Filename"]
    tensor   = preprocess(img_path)

    # FP32
    out_fp32   = sess_fp32.run(["output"], {"input": tensor})[0]
    pid_fp32   = int(np.argmax(out_fp32))
    conf_fp32  = softmax_conf(out_fp32, pid_fp32)

    # INT8
    out_int8   = sess_int8.run(["output"], {"input": tensor})[0]
    pid_int8   = int(np.argmax(out_int8))
    conf_int8  = softmax_conf(out_int8, pid_int8)

    delta = conf_int8 - conf_fp32
    match = "OK" if pid_fp32 == pid_int8 else "MISMATCH"

    fp32_confs.append(round(conf_fp32, 2))
    int8_confs.append(round(conf_int8, 2))
    deltas.append(round(delta, 2))

    pt_c = float(row["PyTorch Conf (%)"])
    print(f"  {int(row['ID']):<4} {row['Image Filename']:<18} {pt_c:>7.2f}% {conf_fp32:>7.2f}% {conf_int8:>7.2f}% {delta:>+6.2f}%  {match}")

# ── Patch test_predictions.csv ───────────────────────────
ref_df["ONNX FP32 Conf (%)"]         = fp32_confs
ref_df["ONNX INT8 Conf (%)"]         = int8_confs
ref_df["Delta Conf (INT8-FP32) (%)"] = deltas
ref_df.to_csv(ref_csv, index=False)
print(f"\n  CSV updated: {ref_csv}")

# ── Regenerate comparison PNG ────────────────────────────
print("  Regenerating comparison PNG...")
import importlib.util
spec = importlib.util.spec_from_file_location(
    "gen_png", PROJECT_ROOT / "src" / "generate_efficientnet_comparison_png.py")
mod  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print("  PNG regenerated.")
