"""
============================================================
PADDY GUARD AI -- MODULE 4: Data Preprocessing & Augmentation
============================================================
Purpose:
    Defines standard torchvision transform pipelines for:
      - Training (Resizing + Mild Augmentation + ImageNet Normalization)
      - Validation / Testing (Resizing + ImageNet Normalization)
      - Un-normalization (for display in dashboard / plotting)

Usage:
    python src/preprocessing.py
============================================================
"""

from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms

# --------------------------------------------------------------
# ImageNet Standard Normalization Constants
# --------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Returns torchvision transforms for the TRAINING dataset.
    Includes gentle spatial and color augmentations suitable for plant leaves.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Returns torchvision transforms for the VALIDATION dataset.
    Only deterministic resizing and normalization (NO random augmentations).
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_test_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Returns torchvision transforms for the TEST dataset.
    Identical to validation transforms for fair and reproducible evaluation.
    """
    return get_val_transforms(image_size=image_size)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reverses ImageNet normalization on a tensor (C, H, W) or (B, C, H, W)
    for rendering images back in standard RGB [0, 1] range.
    """
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(-1, 1, 1)
    std  = torch.tensor(IMAGENET_STD, device=tensor.device).view(-1, 1, 1)
    
    if tensor.ndim == 4:
        mean = mean.unsqueeze(0)
        std  = std.unsqueeze(0)
        
    return torch.clamp((tensor * std) + mean, 0.0, 1.0)


# --------------------------------------------------------------
# Self-Test Verification
# --------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  PADDY GUARD AI -- PREPROCESSING MODULE TEST")
    print("=" * 60)

    # Pick an actual image from the dataset for testing
    sample_dir = Path("dataset/train_images/normal")
    if sample_dir.exists():
        sample_images = list(sample_dir.glob("*.jpg"))
        if sample_images:
            sample_path = sample_images[0]
            print(f"\n  Testing with actual image: {sample_path.name}")
            raw_img = Image.open(sample_path).convert("RGB")
            print(f"  Raw image size (W x H)   : {raw_img.size}")

            # 1. Test Train Transform
            train_tf = get_train_transforms(image_size=224)
            train_tensor = train_tf(raw_img)
            print(f"\n  [Train Transform]")
            print(f"    Output Tensor Shape   : {tuple(train_tensor.shape)}")
            print(f"    Data Type             : {train_tensor.dtype}")
            print(f"    Min Value             : {train_tensor.min().item():.3f}")
            print(f"    Max Value             : {train_tensor.max().item():.3f}")

            # 2. Test Val Transform
            val_tf = get_val_transforms(image_size=224)
            val_tensor = val_tf(raw_img)
            print(f"\n  [Val/Test Transform]")
            print(f"    Output Tensor Shape   : {tuple(val_tensor.shape)}")
            print(f"    Min Value             : {val_tensor.min().item():.3f}")
            print(f"    Max Value             : {val_tensor.max().item():.3f}")

            # 3. Test Denormalize
            unnorm = denormalize(val_tensor)
            print(f"\n  [Denormalization Test]")
            print(f"    Un-normalized Shape   : {tuple(unnorm.shape)}")
            print(f"    Un-normalized Min     : {unnorm.min().item():.3f} (Expected >= 0.0)")
            print(f"    Un-normalized Max     : {unnorm.max().item():.3f} (Expected <= 1.0)")

            print("\n" + "=" * 60)
            print("  PREPROCESSING PIPELINE VERIFIED SUCCESSFULLY!")
            print("=" * 60)
        else:
            print("  [WARN] No sample images found in dataset/train_images/normal")
    else:
        print(f"  [ERROR] Directory not found: {sample_dir}")
