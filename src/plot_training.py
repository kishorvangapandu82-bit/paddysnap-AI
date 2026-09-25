"""
============================================================
PADDY GUARD AI -- Training Curve Plotter
============================================================
Purpose:
    Reads actual training history CSVs from results/training_history/
    and generates high-resolution Loss and Accuracy curve plots.

Usage:
    python src/plot_training.py --model resnet50
    python src/plot_training.py --model all
============================================================
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


HISTORY_DIR = Path("results/training_history")
GRAPHS_DIR  = Path("results/graphs")
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)


def plot_single_model(model_name: str):
    """Generates Loss and Accuracy plots for a single model."""
    csv_file = HISTORY_DIR / f"{model_name}_history.csv"
    if not csv_file.exists():
        print(f"  [ERROR] History file not found: {csv_file}")
        print("  -> Train the model first using: python src/train.py --model " + model_name)
        return False

    df = pd.read_csv(csv_file)
    epochs = df["epoch"].values
    train_loss = df["train_loss"].values
    val_loss   = df["val_loss"].values
    train_acc  = df["train_accuracy"].values
    val_acc    = df["val_accuracy"].values

    # 1. Loss Curve Plot
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    ax.plot(epochs, train_loss, label="Training Loss", color="#1f77b4", linewidth=2.2, marker="o", markersize=4)
    ax.plot(epochs, val_loss,   label="Validation Loss", color="#d62728", linewidth=2.2, marker="s", markersize=4, linestyle="--")
    
    # Mark Best Val Loss
    best_loss_idx = val_loss.argmin()
    ax.scatter(epochs[best_loss_idx], val_loss[best_loss_idx], color="#d62728", s=100, zorder=5, edgecolors="black")
    ax.annotate(f"Best: {val_loss[best_loss_idx]:.4f} (Ep {epochs[best_loss_idx]})",
                (epochs[best_loss_idx], val_loss[best_loss_idx]),
                textcoords="offset points", xytext=(0, 12), ha="center",
                fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="#ffebee", ec="#d62728", lw=1))

    ax.set_title(f"Training & Validation Loss -- [{model_name.upper()}]", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("CrossEntropy Loss", fontsize=11)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="lightgray", fontsize=10)
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("#ffffff")
    plt.tight_layout()

    loss_png = GRAPHS_DIR / f"{model_name}_loss.png"
    plt.savefig(loss_png, bbox_inches="tight")
    plt.close()
    print(f"  Saved Loss Curve     -> {loss_png}")

    # 2. Accuracy Curve Plot
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    ax.plot(epochs, train_acc, label="Training Accuracy", color="#2ca02c", linewidth=2.2, marker="o", markersize=4)
    ax.plot(epochs, val_acc,   label="Validation Accuracy", color="#ff7f0e", linewidth=2.2, marker="s", markersize=4, linestyle="--")

    # Mark Best Val Accuracy
    best_acc_idx = val_acc.argmax()
    ax.scatter(epochs[best_acc_idx], val_acc[best_acc_idx], color="#ff7f0e", s=100, zorder=5, edgecolors="black")
    ax.annotate(f"Best: {val_acc[best_acc_idx]:.2f}% (Ep {epochs[best_acc_idx]})",
                (epochs[best_acc_idx], val_acc[best_acc_idx]),
                textcoords="offset points", xytext=(0, -18), ha="center",
                fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="#fff3e0", ec="#ff7f0e", lw=1))

    ax.set_title(f"Training & Validation Accuracy -- [{model_name.upper()}]", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="lightgray", fontsize=10)
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("#ffffff")
    plt.tight_layout()

    acc_png = GRAPHS_DIR / f"{model_name}_accuracy.png"
    plt.savefig(acc_png, bbox_inches="tight")
    plt.close()
    print(f"  Saved Accuracy Curve -> {acc_png}")
    return True


def main():
    parser = argparse.ArgumentParser(description="PADDY GUARD AI -- Plot Training Curves")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g. resnet50) or 'all'")
    args = parser.parse_args()

    print("=" * 65)
    print("  PADDY GUARD AI -- GENERATING TRAINING PLOTS")
    print("=" * 65)

    if args.model.lower() == "all":
        csv_files = list(HISTORY_DIR.glob("*_history.csv"))
        if not csv_files:
            print("  [WARN] No training history CSV files found in results/training_history/")
            return
        for f in csv_files:
            model_name = f.stem.replace("_history", "")
            print(f"\n  Plotting: {model_name}")
            plot_single_model(model_name)
    else:
        plot_single_model(args.model.lower())

    print("\n" + "=" * 65)
    print("  PLOTTING COMPLETED!")
    print("=" * 65)


if __name__ == "__main__":
    main()
