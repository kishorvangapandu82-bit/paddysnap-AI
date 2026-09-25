"""
============================================================
PADDY GUARD AI -- MODULE 6: Universal Training Pipeline
============================================================
Purpose:
    Trains any of the 5 supported CNN architectures with:
      - Class-weighted Loss (mitigates class imbalance)
      - AdamW Optimizer + Cosine Annealing LR Schedule
      - Mixed Precision (AMP) for maximum RTX 5050 GPU speed
      - Per-Epoch CSV Logging to results/training_history/
      - Best Model Checkpointing to checkpoints/
      - Early Stopping to prevent overfitting

Usage:
    python src/train.py --model resnet50 --epochs 20 --batch_size 32
============================================================
"""

import sys
import os
import time
import argparse
import csv
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.utils import set_seed, get_device, AverageMeter, calculate_accuracy, format_time, save_checkpoint
from src.dataset import get_dataloaders
from src.models import get_model, count_parameters, SUPPORTED_MODELS


def parse_args():
    parser = argparse.ArgumentParser(description="PADDY GUARD AI -- Model Training Engine")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(SUPPORTED_MODELS.keys()),
        help="Architecture to train (resnet50, densenet121, efficientnet_v2_s, convnext_tiny, custom_cnn)"
    )
    parser.add_argument("--epochs", type=int, default=20, help="Total training epochs (default: 20)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate (default: 1e-4)")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for AdamW (default: 1e-2)")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience (default: 7)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader subprocess workers (default: 2)")
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    loss_meter = AverageMeter("Train Loss")
    acc_meter  = AverageMeter("Train Acc")

    pbar = tqdm(loader, desc="  Training  ", leave=False, dynamic_ncols=True)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Automatic Mixed Precision for fast RTX GPU training
        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        acc = calculate_accuracy(outputs, labels)
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc, images.size(0))

        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}", "acc": f"{acc_meter.avg:.2f}%"})

    return loss_meter.avg, acc_meter.avg


def validate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter("Val Loss")
    acc_meter  = AverageMeter("Val Acc")

    pbar = tqdm(loader, desc="  Validating", leave=False, dynamic_ncols=True)
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            acc = calculate_accuracy(outputs, labels)
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(acc, images.size(0))

            pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}", "acc": f"{acc_meter.avg:.2f}%"})

    return loss_meter.avg, acc_meter.avg


def main():
    # Ensure Windows terminal encoding safety
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    args = parse_args()

    # 1. Setup paths & seed
    set_seed(args.seed)
    device = get_device()

    history_dir = Path("results/training_history")
    checkpoints_dir = Path("checkpoints")
    history_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    csv_path = history_dir / f"{args.model}_history.csv"
    checkpoint_path = checkpoints_dir / f"{args.model}_best.pth"

    print("\n" + "=" * 70)
    print(f"  PADDY GUARD AI -- TRAINING: [{args.model.upper()}]")
    print("=" * 70)
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch Size   : {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Weight Decay : {args.weight_decay}")
    print(f"  Patience     : {args.patience}")
    print(f"  Seed         : {args.seed}")
    print(f"  History CSV  : {csv_path}")
    print(f"  Checkpoint   : {checkpoint_path}")

    # 2. DataLoaders
    print("\n  Loading Dataset Splits...")
    train_loader, val_loader, test_loader, meta = get_dataloaders(
        data_dir="dataset",
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"  Train: {meta['train_samples']} | Val: {meta['val_samples']} | Test: {meta['test_samples']}")

    # 3. Model
    print(f"\n  Building Model Architecture [{args.model}]...")
    is_pretrained = (args.model != "custom_cnn")
    model = get_model(args.model, num_classes=meta["num_classes"], pretrained=is_pretrained)
    model = model.to(device)

    total_p, train_p = count_parameters(model)
    print(f"  Total Parameters     : {total_p:,}")
    print(f"  Trainable Parameters : {train_p:,}")
    print(f"  Pretrained Weights   : {'ImageNet-1K' if is_pretrained else 'None (From Scratch)'}")

    # 4. Loss, Optimizer, Scheduler, AMP Scaler
    class_weights = meta["class_weights"].to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu")

    # 5. Initialize History CSV
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy", "learning_rate", "epoch_time_sec"])

    # 6. Training Loop
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    start_total_time = time.time()

    print("\n" + "-" * 70)
    print(f" {'Epoch':<7} | {'Train Loss':<10} | {'Train Acc':<10} | {'Val Loss':<10} | {'Val Acc':<10} | {'LR':<10} | {'Time':<8}")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc     = validate(model, val_loader, criterion, device)

        scheduler.step()
        epoch_time = time.time() - epoch_start

        # Append to CSV immediately
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{train_acc:.2f}", f"{val_acc:.2f}", f"{current_lr:.6f}", f"{epoch_time:.1f}"])

        # Display Epoch Status
        is_best_epoch = val_acc > best_val_acc
        mark = " [BEST]" if is_best_epoch else ""
        print(f" {epoch:>3}/{args.epochs:<3} | {train_loss:>10.4f} | {train_acc:>9.2f}% | {val_loss:>10.4f} | {val_acc:>9.2f}% | {current_lr:>10.6f} | {epoch_time:>6.1f}s{mark}")

        # Checkpoint Saving
        if is_best_epoch:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            save_checkpoint({
                "epoch": epoch,
                "model_name": args.model,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
                "val_loss": val_loss,
                "classes": meta["id_to_class"]
            }, str(checkpoint_path))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  [Early Stopping] No validation improvement for {args.patience} epochs. Stopping early.")
                break

    total_training_time = time.time() - start_total_time

    # 7. Final Training Summary
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE -- SUMMARY REPORT")
    print("=" * 70)
    print(f"  Model Trained           : {args.model}")
    print(f"  Best Epoch              : {best_epoch} / {epoch}")
    print(f"  Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"  Best Validation Loss    : {best_val_loss:.4f}")
    print(f"  Total Training Duration : {format_time(total_training_time)}")
    print(f"  Saved Best Checkpoint   : {checkpoint_path}")
    print(f"  Saved Training History  : {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
