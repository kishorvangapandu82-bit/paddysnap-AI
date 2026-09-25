"""
============================================================
PADDY GUARD AI -- MODULE 3: Dataset Inspector
============================================================
Purpose:
    Inspect the extracted paddy dataset.
    Determine class names, image counts, image dimensions,
    class imbalance, and generate a class distribution chart.

Usage:
    python src/inspect_dataset.py

Output:
    - Printed dataset report in terminal
    - results/graphs/class_distribution.png
============================================================
"""

import os
import sys
from pathlib import Path
from collections import Counter
import random

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from PIL import Image


# --------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------

TRAIN_DIR = Path("dataset/train_images")
OUTPUT_DIR = Path("results/graphs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DIMENSION_SAMPLE_SIZE = 100


# --------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------

def get_class_counts(train_dir):
    counts = {}
    for class_dir in sorted(train_dir.iterdir()):
        if class_dir.is_dir():
            images = (list(class_dir.glob("*.jpg")) +
                      list(class_dir.glob("*.jpeg")) +
                      list(class_dir.glob("*.png")))
            counts[class_dir.name] = len(images)
    return counts


def sample_image_dimensions(train_dir, sample_size=100):
    all_images = []
    for class_dir in train_dir.iterdir():
        if class_dir.is_dir():
            imgs = (list(class_dir.glob("*.jpg")) +
                    list(class_dir.glob("*.jpeg")) +
                    list(class_dir.glob("*.png")))
            all_images.extend(imgs)

    random.seed(42)
    sample = random.sample(all_images, min(sample_size, len(all_images)))

    dims = []
    for img_path in sample:
        try:
            with Image.open(img_path) as img:
                dims.append(img.size)
        except Exception:
            pass
    return dims


def check_corrupted(train_dir):
    corrupted = []
    for class_dir in train_dir.iterdir():
        if class_dir.is_dir():
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    try:
                        with Image.open(img_path) as img:
                            img.verify()
                    except Exception:
                        corrupted.append(str(img_path))
    return corrupted


def plot_class_distribution(counts, save_path):
    classes = list(counts.keys())
    values  = list(counts.values())
    colors  = plt.cm.viridis(np.linspace(0.2, 0.85, len(classes)))

    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.bar(classes, values, color=colors, edgecolor="white", linewidth=0.7)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 15,
            str(val),
            ha="center", va="bottom",
            fontsize=10, fontweight="bold"
        )

    ax.set_title("Paddy Disease Dataset -- Class Distribution",
                 fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Disease Class", fontsize=12)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=35, ha="right", fontsize=10)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_facecolor("#f9f9f9")
    fig.patch.set_facecolor("#ffffff")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Chart saved -> {save_path}")


# --------------------------------------------------------------
# MAIN INSPECTION REPORT
# --------------------------------------------------------------

def main():
    print("=" * 60)
    print("  PADDY GUARD AI -- DATASET INSPECTION REPORT")
    print("=" * 60)

    if not TRAIN_DIR.exists():
        print(f"\n[ERROR] Could not find: {TRAIN_DIR}")
        print("  -> Please extract the ZIP first (see command below).")
        sys.exit(1)

    print(f"\n  Dataset path : {TRAIN_DIR.resolve()}")

    # 1. Class counts
    counts = get_class_counts(TRAIN_DIR)

    print("\n  CLASS DISTRIBUTION")
    print(f"  {'Class':<30} {'Count':>7}  {'% of Total':>12}")
    print(f"  {'-'*30} {'-'*7}  {'-'*12}")
    total = sum(counts.values())
    for cls, cnt in counts.items():
        pct = (cnt / total) * 100
        print(f"  {cls:<30} {cnt:>7}  {pct:>11.1f}%")

    print(f"\n  Total Classes      : {len(counts)}")
    print(f"  Total Train Images : {total}")

    # 2. Imbalance
    max_cls = max(counts, key=counts.get)
    min_cls = min(counts, key=counts.get)
    ratio   = counts[max_cls] / counts[min_cls]
    print(f"\n  IMBALANCE ANALYSIS")
    print(f"  Largest  class : {max_cls} ({counts[max_cls]})")
    print(f"  Smallest class : {min_cls} ({counts[min_cls]})")
    print(f"  Imbalance ratio: {ratio:.2f}x")
    if ratio > 3:
        print("  WARNING: Significant imbalance. Use class weights in training.")
    elif ratio > 1.5:
        print("  NOTE: Mild imbalance. Monitor per-class F1.")
    else:
        print("  OK: Dataset is relatively balanced.")

    # 3. Image dimensions
    print(f"\n  IMAGE DIMENSIONS (sample of {DIMENSION_SAMPLE_SIZE})")
    dims = sample_image_dimensions(TRAIN_DIR, DIMENSION_SAMPLE_SIZE)
    if dims:
        widths  = [d[0] for d in dims]
        heights = [d[1] for d in dims]
        print(f"  Sampled         : {len(dims)} images")
        print(f"  Width  min/max/avg : {min(widths)} / {max(widths)} / {sum(widths)//len(widths)}")
        print(f"  Height min/max/avg : {min(heights)} / {max(heights)} / {sum(heights)//len(heights)}")
        unique_sizes = Counter(dims)
        print(f"  Unique sizes    : {len(unique_sizes)}")
        if len(unique_sizes) > 1:
            print("  WARNING: Mixed sizes -> resizing required (224x224).")

    # 4. Corrupted check
    print(f"\n  CORRUPTED IMAGE CHECK")
    print("  Scanning all images (this may take a moment)...")
    corrupted = check_corrupted(TRAIN_DIR)
    if corrupted:
        print(f"  WARNING: {len(corrupted)} corrupted image(s):")
        for c in corrupted:
            print(f"    {c}")
    else:
        print("  OK: No corrupted images found.")

    # 5. Formats
    formats = Counter()
    for class_dir in TRAIN_DIR.iterdir():
        if class_dir.is_dir():
            for f in class_dir.iterdir():
                if f.is_file():
                    formats[f.suffix.lower()] += 1
    print(f"\n  IMAGE FORMATS")
    for fmt, cnt in formats.most_common():
        print(f"  {fmt}: {cnt}")

    # 6. Test set
    test_dir = Path("dataset/test_images")
    if test_dir.exists():
        test_imgs = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.jpeg"))
        print(f"\n  UNLABELED TEST SET")
        print(f"  Images : {len(test_imgs)}")
        print("  NOTE: These have NO labels and CANNOT be used for metrics.")

    # 7. train.csv
    train_csv = Path("dataset/train.csv")
    if train_csv.exists():
        import csv
        with open(train_csv, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows   = list(reader)
        print(f"\n  TRAIN CSV")
        print(f"  Columns : {header}")
        print(f"  Rows    : {len(rows)}")
        print("  Sample rows:")
        for row in rows[:3]:
            print(f"    {row}")

    # 8. Chart
    print(f"\n  Generating class distribution chart...")
    plot_class_distribution(counts, OUTPUT_DIR / "class_distribution.png")

    print("\n" + "=" * 60)
    print("  INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
