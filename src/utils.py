"""
============================================================
PADDY GUARD AI -- Helper Utilities
============================================================
Purpose:
    Provides seed fixing, device detection (GPU/CPU), metric
    meters, checkpoint management, and logging helpers.
============================================================
"""

import os
import random
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across Python, NumPy, and PyTorch for 100% deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """Returns CUDA GPU device if available (RTX 5050), else falls back to CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        device_name = torch.cuda.get_device_name(0)
        print(f"  [Device] Using GPU: {device_name}")
    else:
        device = torch.device("cpu")
        print("  [Device] GPU not available. Using CPU.")
    return device


class AverageMeter:
    """Computes and stores the average and current value of a metric."""
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0.0


def calculate_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Computes classification accuracy percentage (0.0 to 100.0)."""
    with torch.no_grad():
        preds = torch.argmax(outputs, dim=1)
        correct = (preds == targets).sum().item()
        total = targets.size(0)
        return (correct / total) * 100.0 if total > 0 else 0.0


def format_time(seconds: float) -> str:
    """Formats elapsed seconds into MM:SS or HH:MM:SS string."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def save_checkpoint(state: Dict[str, Any], filepath: str) -> None:
    """Saves model weights and training metadata dictionary to disk."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))
