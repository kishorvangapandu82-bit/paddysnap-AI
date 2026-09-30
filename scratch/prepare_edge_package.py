"""
============================================================
PADDY GUARD AI -- Edge Package Builder
============================================================
Assembles all metadata, models, and standalone scripts
into deployment/edge_package/ for Raspberry Pi, Jetson, Mobile, & C++.
"""

import json
import shutil
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recommendations import DISEASE_KNOWLEDGE_BASE

EDGE_DIR = PROJECT_ROOT / "deployment" / "edge_package"
MODELS_DIR = EDGE_DIR / "models"
META_DIR = EDGE_DIR / "metadata"

EDGE_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

# 1. Export Agronomy Rules to standalone JSON
agronomy_json_path = META_DIR / "agronomy_rules.json"
with open(agronomy_json_path, "w", encoding="utf-8") as f:
    json.dump(DISEASE_KNOWLEDGE_BASE, f, indent=2)
print(f" Saved: {agronomy_json_path}")

# 2. Copy class_mapping.json
class_map_src = PROJECT_ROOT / "dataset" / "class_mapping.json"
class_map_dst = META_DIR / "class_mapping.json"
shutil.copy(class_map_src, class_map_dst)
print(f" Copied: {class_map_dst}")

# 3. Create labels.txt (plain 1 line per label for Android/Mobile/C++)
with open(class_map_src, "r") as f:
    cmap = json.load(f)["id_to_class"]

labels_txt_path = META_DIR / "labels.txt"
with open(labels_txt_path, "w", encoding="utf-8") as f:
    for i in range(len(cmap)):
        f.write(f"{cmap[str(i)]}\n")
print(f" Saved: {labels_txt_path}")

# 4. Copy INT8 & FP32 ONNX models for edge deployment
onnx_src_dir = PROJECT_ROOT / "deployment" / "onnx"
models_to_copy = [
    "efficientnet_v2_s_int8.onnx",
    "efficientnet_v2_s.onnx",
    "densenet121_int8.onnx",
    "custom_cnn_int8.onnx"
]

for m in models_to_copy:
    src_file = onnx_src_dir / m
    dst_file = MODELS_DIR / m
    if src_file.exists():
        shutil.copy(src_file, dst_file)
        size_mb = dst_file.stat().st_size / (1024 * 1024)
        print(f" Copied Model: {m:<30} ({size_mb:.2f} MB)")
    else:
        print(f"⚠️ Warning: Model {m} not found at {src_file}")

print("\n Edge deployment package base prepared successfully!")
