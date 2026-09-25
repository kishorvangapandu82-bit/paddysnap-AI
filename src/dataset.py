"""
============================================================
PADDY GUARD AI -- MODULE 5: PyTorch Dataset & DataLoaders
============================================================
Purpose:
    Defines the PyTorch Dataset class, DataLoader factory,
    and class weighting utility for class imbalance mitigation.

Usage:
    python src/dataset.py
============================================================
"""

import sys
import json
from pathlib import Path
from typing import Tuple, Optional, Dict

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from src.preprocessing import get_train_transforms, get_val_transforms, get_test_transforms


class PaddyDataset(Dataset):
    """
    PyTorch Dataset for Paddy Leaf Images.
    Loads image paths and numerical class labels from a split CSV file.
    """
    def __init__(self, csv_file: str, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        
        # Verify required columns exist
        required_cols = {"image_path", "label_id"}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(f"CSV {csv_file} must contain columns: {required_cols}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        label = int(row["label_id"])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image: {img_path}. Error: {e}")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def compute_class_weights(csv_path: str = "dataset/train_split.csv", num_classes: int = 10) -> torch.Tensor:
    """
    Calculates balanced class weights using inverse class frequency:
        weight[c] = total_samples / (num_classes * count[c])
    Helps loss function treat minority classes fairly during gradient updates.
    """
    df = pd.read_csv(csv_path)
    counts = df["label_id"].value_counts().sort_index()
    total_samples = len(df)
    
    weights = np.zeros(num_classes, dtype=np.float32)
    for cls_id in range(num_classes):
        cnt = counts.get(cls_id, 1)
        weights[cls_id] = total_samples / (num_classes * cnt)
        
    return torch.tensor(weights, dtype=torch.float32)


def get_dataloaders(
    data_dir: str = "dataset",
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 2,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Creates and returns train, validation, and test PyTorch DataLoaders.
    
    Returns:
        (train_loader, val_loader, test_loader, metadata_dict)
    """
    data_path = Path(data_dir)
    train_csv = data_path / "train_split.csv"
    val_csv   = data_path / "val_split.csv"
    test_csv  = data_path / "test_split.csv"
    map_json  = data_path / "class_mapping.json"

    for path in [train_csv, val_csv, test_csv, map_json]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}. Run src/split_dataset.py first.")

    with open(map_json, "r") as f:
        mapping = json.load(f)

    # 1. Instantiate Datasets with appropriate transforms
    train_dataset = PaddyDataset(str(train_csv), transform=get_train_transforms(image_size))
    val_dataset   = PaddyDataset(str(val_csv),   transform=get_val_transforms(image_size))
    test_dataset  = PaddyDataset(str(test_csv),  transform=get_test_transforms(image_size))

    # 2. Build DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # 3. Compute Class Weights for Loss Function
    class_weights = compute_class_weights(str(train_csv), num_classes=mapping["num_classes"])

    metadata = {
        "num_classes": mapping["num_classes"],
        "class_to_id": mapping["class_to_id"],
        "id_to_class": mapping["id_to_class"],
        "class_weights": class_weights,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset)
    }

    return train_loader, val_loader, test_loader, metadata


# --------------------------------------------------------------
# Self-Test Verification
# --------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 65)
    print("  PADDY GUARD AI -- DATALOADER VERIFICATION TEST")
    print("=" * 65)

    train_loader, val_loader, test_loader, meta = get_dataloaders(
        data_dir="dataset",
        batch_size=32,
        image_size=224,
        num_workers=0  # Use 0 workers for quick single-process unit test on Windows
    )

    print(f"\n  Classes ({meta['num_classes']}):")
    for id_str, name in meta["id_to_class"].items():
        weight = meta["class_weights"][int(id_str)].item()
        print(f"    Class {id_str}: {name:<28} (Loss Weight = {weight:.3f})")

    print(f"\n  Dataset Counts:")
    print(f"    Train Samples : {meta['train_samples']} ({len(train_loader)} batches of 32)")
    print(f"    Val Samples   : {meta['val_samples']} ({len(val_loader)} batches of 32)")
    print(f"    Test Samples  : {meta['test_samples']} ({len(test_loader)} batches of 32)")

    # Test pulling 1 batch from train_loader
    images, labels = next(iter(train_loader))
    print(f"\n  [Sample Batch Verification]")
    print(f"    Batch Images Tensor Shape : {tuple(images.shape)}")
    print(f"    Batch Labels Tensor Shape : {tuple(labels.shape)}")
    print(f"    Labels in Batch Sample    : {labels[:8].tolist()}")

    print("\n" + "=" * 65)
    print("  DATALOADER PIPELINE VERIFIED SUCCESSFULLY!")
    print("=" * 65)
