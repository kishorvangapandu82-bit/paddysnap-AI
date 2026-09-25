"""
============================================================
PADDY GUARD AI -- MODULE 5: Stratified Dataset Splitter
============================================================
Purpose:
    Performs a reproducible, stratified 70% Train / 15% Val / 15% Test split
    on the 10,407 labeled paddy leaf images.

Outputs:
    - dataset/train_split.csv
    - dataset/val_split.csv
    - dataset/test_split.csv
    - dataset/class_mapping.json

Usage:
    python src/split_dataset.py
============================================================
"""

import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------
DATASET_DIR = Path("dataset/train_images")
OUTPUT_DIR  = Path("dataset")
SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15


def scan_dataset(dataset_dir: Path) -> pd.DataFrame:
    """Scans dataset directory and builds a DataFrame of image paths and labels."""
    records = []
    class_names = sorted([d.name for d in dataset_dir.iterdir() if d.is_dir()])
    
    for class_id, class_name in enumerate(class_names):
        class_dir = dataset_dir / class_name
        for img_path in class_dir.glob("*.jpg"):
            records.append({
                "image_path": str(img_path.as_posix()),
                "filename": img_path.name,
                "label_name": class_name,
                "label_id": class_id
            })
            
    df = pd.DataFrame(records)
    return df, class_names


def create_splits(df: pd.DataFrame, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Performs two-step stratified splitting:
      Step 1: 70% Train, 30% Temp
      Step 2: 15% Val, 15% Test (50% of Temp each)
    """
    # Step 1: Separate Train (70%) and Temp (30%)
    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        stratify=df["label_id"],
        random_state=seed,
        shuffle=True
    )
    
    # Step 2: Split Temp into Validation (15%) and Test (15%)
    val_proportion_of_temp = val_ratio / (val_ratio + test_ratio)  # 0.15 / 0.30 = 0.50
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_proportion_of_temp),
        stratify=temp_df["label_id"],
        random_state=seed,
        shuffle=True
    )
    
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def main():
    print("=" * 65)
    print("  PADDY GUARD AI -- DATASET SPLIT GENERATOR")
    print("=" * 65)
    
    if not DATASET_DIR.exists():
        print(f"[ERROR] Directory not found: {DATASET_DIR}")
        return

    # 1. Scan images
    df, class_names = scan_dataset(DATASET_DIR)
    print(f"\n  Total images found : {len(df)}")
    print(f"  Total classes      : {len(class_names)}")
    
    # 2. Save class mapping dictionary
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    id_to_class = {idx: name for idx, name in enumerate(class_names)}
    mapping_data = {
        "class_to_id": class_to_id,
        "id_to_class": id_to_class,
        "num_classes": len(class_names)
    }
    
    mapping_path = OUTPUT_DIR / "class_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping_data, f, indent=2)
    print(f"  Class mapping saved -> {mapping_path}")

    # 3. Create Stratified Splits
    train_df, val_df, test_df = create_splits(
        df,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SEED
    )

    # 4. Save CSV splits
    train_csv = OUTPUT_DIR / "train_split.csv"
    val_csv   = OUTPUT_DIR / "val_split.csv"
    test_csv  = OUTPUT_DIR / "test_split.csv"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"\n  Split CSVs Generated (Random Seed = {SEED}):")
    print(f"    - Training Split   (70%) : {len(train_df):>5} images -> {train_csv}")
    print(f"    - Validation Split (15%) : {len(val_df):>5} images -> {val_csv}")
    print(f"    - Test Split       (15%) : {len(test_df):>5} images -> {test_csv}")
    print(f"    - Total                  : {len(train_df) + len(val_df) + len(test_df):>5} images")

    # 5. Verification Table (Per-Class Stratification Confirmation)
    print("\n" + "=" * 65)
    print("  STRATIFICATION VERIFICATION TABLE")
    print("=" * 65)
    print(f"  {'Class Name':<28} {'Total':>6}  {'Train':>6}  {'Val':>5}  {'Test':>5}  {'Train %':>7}")
    print(f"  {'-'*28} {'-'*6}  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*7}")
    
    for cls in class_names:
        tot = (df["label_name"] == cls).sum()
        trn = (train_df["label_name"] == cls).sum()
        val = (val_df["label_name"] == cls).sum()
        tst = (test_df["label_name"] == cls).sum()
        pct = (trn / tot) * 100
        print(f"  {cls:<28} {tot:>6}  {trn:>6}  {val:>5}  {tst:>5}  {pct:>6.1f}%")

    print("\n" + "=" * 65)
    print("  STRATIFIED SPLIT COMPLETED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    main()
