"""
============================================================
PADDY GUARD AI -- Generate Kaggle Submission CSV
============================================================
Purpose:
    Runs batch inference on all 3,469 unlabeled test images
    using the Champion Model (EfficientNetV2-S) on GPU and
    outputs a fully completed submission.csv.

Usage:
    python src/generate_submission.py
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
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from tqdm import tqdm

from src.utils import get_device, set_seed
from src.models import get_model
from src.preprocessing import get_val_transforms


class UnlabeledTestDataset(Dataset):
    """Dataset for Kaggle unlabeled test images."""
    def __init__(self, image_ids, images_dir, transform=None):
        self.image_ids = list(image_ids)
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = self.image_ids[idx]
        img_path = self.images_dir / img_name
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_name


def generate_submission(model_name: str = "efficientnet_v2_s", batch_size: int = 64):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    set_seed(42)
    device = get_device()

    print("=" * 70)
    print("  PADDY GUARD AI -- BATCH INFERENCE FOR KAGGLE TEST SET")
    print("=" * 70)

    # 1. Load Sample Submission
    sample_sub_path = Path("dataset/sample_submission.csv")
    if not sample_sub_path.exists():
        raise FileNotFoundError("Missing dataset/sample_submission.csv")
    sub_df = pd.read_csv(sample_sub_path)
    print(f"  Target Image Count : {len(sub_df)} unlabeled test images")

    # 2. Load Class Mapping
    with open("dataset/class_mapping.json") as f:
        mapping = json.load(f)
    id_to_class = mapping["id_to_class"]
    num_classes = len(id_to_class)

    # 3. Load Model Checkpoint
    ckpt_path = Path(f"checkpoints/{model_name}_best.pth")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    print(f"  Loading Champion Model : [{model_name.upper()}] from {ckpt_path}")
    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = get_model(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # 4. DataLoader
    test_dataset = UnlabeledTestDataset(
        image_ids=sub_df["image_id"],
        images_dir="dataset/test_images",
        transform=get_val_transforms(image_size=224)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # 5. Batch Inference
    print("  Running GPU Batch Inference...")
    predictions = []
    image_names = []

    with torch.no_grad():
        for images, names in tqdm(test_loader, desc="  Inference Progress"):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            for p, name in zip(preds, names):
                predictions.append(id_to_class[str(p)])
                image_names.append(name)

    # 6. Save Submissions
    out_df = pd.DataFrame({"image_id": image_names, "label": predictions})

    # Save to root and results
    out_path_1 = Path("submission.csv")
    out_path_2 = Path("results/submission.csv")
    out_df.to_csv(out_path_1, index=False)
    out_df.to_csv(out_path_2, index=False)

    print("\n" + "=" * 70)
    print(f"  PREDICTIONS SAVED SUCCESSFULLY!")
    print(f"  --> {out_path_1} ({len(out_df)} rows)")
    print(f"  --> {out_path_2} ({len(out_df)} rows)")
    print("=" * 70)

    # 7. Distribution Breakdown
    print("\n  Predicted Disease Distribution Across 3,469 Test Images:")
    print("  " + "-" * 50)
    counts = out_df["label"].value_counts()
    for cls, count in counts.items():
        pct = count / len(out_df) * 100
        print(f"    {cls:<28} : {count:>5} images ({pct:>5.1f}%)")
    print("  " + "-" * 50 + "\n")


if __name__ == "__main__":
    generate_submission()
