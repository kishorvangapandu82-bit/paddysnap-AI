"""
============================================================
PADDY GUARD AI -- MODULE 15: Formal Test Set Evaluation
============================================================
Purpose:
    Evaluates all 5 trained CNN models on the test set
    (1,562 images never seen during training or validation).

Outputs per model:
    - Test Accuracy, Macro F1, Weighted F1
    - Per-class Precision, Recall, F1-Score
    - Confusion Matrix (PNG heatmap)
    - Full metrics saved to results/metrics/{model}_metrics.json

Usage:
    python src/evaluate.py --model all
    python src/evaluate.py --model resnet50
============================================================
"""

import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)

from src.utils import get_device, set_seed
from src.models import get_model, SUPPORTED_MODELS
from src.dataset import get_dataloaders


METRICS_DIR = Path("results/metrics")
GRAPHS_DIR  = Path("results/graphs")
METRICS_DIR.mkdir(parents=True, exist_ok=True)
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(model_name: str, device: torch.device, test_loader, id_to_class: dict):
    """
    Loads a saved checkpoint and evaluates the model on the test set.
    Returns all_preds, all_labels, class_names.
    """
    checkpoint_path = Path(f"checkpoints/{model_name}_best.pth")
    if not checkpoint_path.exists():
        print(f"  [SKIP] Checkpoint not found: {checkpoint_path}")
        return None, None, None

    print(f"\n  Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    
    is_pretrained = (model_name != "custom_cnn")
    model = get_model(model_name, num_classes=len(id_to_class), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"  Evaluating [{model_name}]", leave=False):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def generate_confusion_matrix(preds, labels, class_names, model_name):
    """Generates and saves a high-quality confusion matrix heatmap."""
    cm = confusion_matrix(labels, preds)
    
    # Normalize (row-wise: recall per class)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(12, 10), dpi=130)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="lightgray",
        ax=ax,
        cbar_kws={"label": "Recall per Class (%)"}
    )

    ax.set_title(f"Confusion Matrix -- {model_name.upper()}", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=10)
    ax.set_ylabel("True Label", fontsize=11, labelpad=10)
    ax.tick_params(axis="x", rotation=40, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    plt.tight_layout()

    output_path = Path("results/confusion_matrices") / f"{model_name}_confusion_matrix.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), bbox_inches="tight")
    plt.close()
    print(f"  Confusion Matrix saved -> {output_path}")


def run_evaluation(model_name: str, device, test_loader, meta):
    """Full evaluation pipeline for a single model."""
    id_to_class = meta["id_to_class"]
    class_names = [id_to_class[str(i)] for i in range(meta["num_classes"])]

    preds, labels = evaluate_model(model_name, device, test_loader, id_to_class)
    if preds is None:
        return None

    # 1. Core Metrics
    test_accuracy   = accuracy_score(labels, preds) * 100
    macro_f1        = f1_score(labels, preds, average="macro") * 100
    weighted_f1     = f1_score(labels, preds, average="weighted") * 100

    # 2. Per-Class Report
    report = classification_report(
        labels, preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    # 3. Display Results
    print(f"\n  ======================================================")
    print(f"  MODEL: {model_name.upper()}")
    print(f"  ======================================================")
    print(f"  Test Accuracy      : {test_accuracy:.2f}%")
    print(f"  Macro F1-Score     : {macro_f1:.2f}%")
    print(f"  Weighted F1-Score  : {weighted_f1:.2f}%")
    print(f"\n  {'Class':<28} {'Precision':>10} {'Recall':>8} {'F1-Score':>10} {'Support':>9}")
    print(f"  {'-'*28} {'-'*10} {'-'*8} {'-'*10} {'-'*9}")
    for cls in class_names:
        r = report[cls]
        print(f"  {cls:<28} {r['precision']*100:>9.2f}% {r['recall']*100:>7.2f}% {r['f1-score']*100:>9.2f}% {int(r['support']):>9}")
    print(f"  {'-'*28} {'-'*10} {'-'*8} {'-'*10} {'-'*9}")
    print(f"  {'macro avg':<28} {report['macro avg']['precision']*100:>9.2f}% {report['macro avg']['recall']*100:>7.2f}% {macro_f1:>9.2f}%")
    print(f"  {'weighted avg':<28} {report['weighted avg']['precision']*100:>9.2f}% {report['weighted avg']['recall']*100:>7.2f}% {weighted_f1:>9.2f}%")

    # 4. Save Confusion Matrix
    generate_confusion_matrix(preds, labels, class_names, model_name)

    # 5. Save JSON metrics
    metrics_data = {
        "model_name": model_name,
        "test_accuracy": round(test_accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": {
            cls: {
                "precision": round(report[cls]["precision"] * 100, 4),
                "recall": round(report[cls]["recall"] * 100, 4),
                "f1_score": round(report[cls]["f1-score"] * 100, 4),
                "support": int(report[cls]["support"])
            }
            for cls in class_names
        }
    }
    json_path = METRICS_DIR / f"{model_name}_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"  Metrics JSON saved  -> {json_path}")

    return metrics_data


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="PADDY GUARD AI -- Test Set Evaluation")
    parser.add_argument("--model", type=str, default="all",
                        help="Model name or 'all' (default: all)")
    args = parser.parse_args()

    set_seed(42)
    device = get_device()

    print("\n" + "=" * 70)
    print("  PADDY GUARD AI -- MODULE 15: FORMAL TEST SET EVALUATION")
    print("=" * 70)

    _, _, test_loader, meta = get_dataloaders(
        data_dir="dataset",
        batch_size=32,
        num_workers=0,
        pin_memory=False
    )
    print(f"  Test Set: {meta['test_samples']} images | {meta['num_classes']} classes")

    # Models to evaluate
    if args.model.lower() == "all":
        models_to_eval = list(SUPPORTED_MODELS.keys())
    else:
        models_to_eval = [args.model.lower().replace("-", "_")]

    all_results = {}
    for model_name in models_to_eval:
        result = run_evaluation(model_name, device, test_loader, meta)
        if result:
            all_results[model_name] = result

    # Final Comparison Table
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("  FINAL MODEL COMPARISON -- TEST SET RESULTS")
        print("=" * 70)
        print(f"  {'Model':<25} {'Test Accuracy':>14} {'Macro F1':>10} {'Weighted F1':>13}")
        print(f"  {'-'*25} {'-'*14} {'-'*10} {'-'*13}")
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]["test_accuracy"], reverse=True)
        for model_name, metrics in sorted_results:
            print(f"  {model_name:<25} {metrics['test_accuracy']:>13.2f}% {metrics['macro_f1']:>9.2f}% {metrics['weighted_f1']:>12.2f}%")
        print("=" * 70)

    print("\n  MODULE 15 EVALUATION COMPLETE!")


if __name__ == "__main__":
    main()
