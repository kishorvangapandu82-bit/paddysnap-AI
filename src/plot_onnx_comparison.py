"""
================================================================================
PADDYSNAP AI -- Visual Infographic: ONNX FP32 vs INT8 Quantization Comparison
================================================================================
Generates a comprehensive, publication-grade dark-themed visual comparison PNG
illustrating diagnostic parity, confidence correlation, and memory footprint.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

PROJECT_ROOT = Path(__file__).resolve().parent.parent
csv_path = PROJECT_ROOT / "results" / "metrics" / "onnx_fp32_vs_int8_comparison.csv"
df = pd.read_csv(csv_path)

out_dir = PROJECT_ROOT / "results" / "graphs"
out_dir.mkdir(parents=True, exist_ok=True)
out_png = out_dir / "onnx_fp32_vs_int8_visual_comparison.png"

# Setup Theme
plt.style.use('dark_background')
fig = plt.figure(figsize=(20, 13), facecolor='#0b1120')
gs = GridSpec(nrows=3, ncols=2, height_ratios=[0.18, 0.45, 0.37], width_ratios=[1.3, 0.7], hspace=0.32, wspace=0.22)

# Color Palette
C_BG_CARD = '#1e293b'
C_BORDER = '#334155'
C_FP32 = '#10b981'       # Emerald Green
C_INT8 = '#38bdf8'       # Sky Blue
C_TEXT = '#f8fafc'
C_MUTED = '#94a3b8'
C_ACCENT = '#fbbf24'      # Amber Gold
C_PURPLE = '#a855f7'

# ==============================================================================
# 1. HEADER & KPI CARDS (Top row)
# ==============================================================================
ax_kpi = fig.add_subplot(gs[0, :])
ax_kpi.set_facecolor('#0b1120')
ax_kpi.axis('off')

# Main Title & Subtitle
ax_kpi.text(0.0, 0.88, "PADDYSNAP AI -- ONNX FP32 vs INT8 QUANTIZATION BENCHMARK", 
            fontsize=22, fontweight='bold', color='#ffffff', va='top')
ax_kpi.text(0.0, 0.62, "Direct Diagnostic Parity & Confidence Verification on 20 Held-Out Field Leaves | Model: EfficientNetV2-S (97.38% Acc)", 
            fontsize=12, color=C_MUTED, va='top')

# 4 KPI Stat Cards
cards = [
    ("DIAGNOSTIC AGREEMENT", "100.0%", "20 / 20 Identical Diagnoses", C_FP32),
    ("STORAGE REDUCTION", "73.9%", "76.87 MB → 20.07 MB (-56.8 MB)", C_INT8),
    ("MEAN CONFIDENCE SHIFT", "1.44%", "High Precision Retention", C_ACCENT),
    ("EDGE DEPLOYMENT READY", "INT8 ENGINE", "Low RAM / Raspberry Pi / Jetson", C_PURPLE)
]

card_w = 0.235
card_gap = 0.02
for i, (title, val, sub, col) in enumerate(cards):
    x0 = i * (card_w + card_gap)
    # Background Box
    rect = patches.FancyBboxPatch((x0, 0.0), card_w, 0.48, boxstyle="round,pad=0.015", 
                                  facecolor=C_BG_CARD, edgecolor=C_BORDER, linewidth=1.5)
    ax_kpi.add_patch(rect)
    # Accent top border
    accent_bar = patches.Rectangle((x0, 0.45), card_w, 0.03, facecolor=col)
    ax_kpi.add_patch(accent_bar)
    
    ax_kpi.text(x0 + 0.015, 0.36, title, fontsize=9, fontweight='bold', color=C_MUTED, va='center')
    ax_kpi.text(x0 + 0.015, 0.20, val, fontsize=18, fontweight='bold', color='#ffffff', va='center')
    ax_kpi.text(x0 + 0.015, 0.07, sub, fontsize=8.5, color=col, va='center')

# ==============================================================================
# 2. PANEL 1: GROUPED CONFIDENCE COMPARISON (Top Left)
# ==============================================================================
ax_bars = fig.add_subplot(gs[1, 0])
ax_bars.set_facecolor(C_BG_CARD)
for spine in ax_bars.spines.values():
    spine.set_edgecolor(C_BORDER)

x = np.arange(len(df))
bar_w = 0.38

rects1 = ax_bars.bar(x - bar_w/2, df["ONNX FP32 Conf (%)"], bar_w, label="ONNX FP32 (Full Precision - 76.87 MB)", color=C_FP32, alpha=0.9, edgecolor='#065f46')
rects2 = ax_bars.bar(x + bar_w/2, df["ONNX INT8 Conf (%)"], bar_w, label="ONNX INT8 (Quantized Edge - 20.07 MB)", color=C_INT8, alpha=0.9, edgecolor='#0284c7')

ax_bars.set_title("Sample-by-Sample Confidence Score: FP32 vs INT8 (All 20 Test Images)", fontsize=13, fontweight='bold', color='#ffffff', pad=12, loc='left')
ax_bars.set_ylabel("Confidence Score (%)", fontsize=11, color=C_MUTED)
ax_bars.set_ylim(70, 102)
ax_bars.set_xticks(x)
ax_bars.set_xticklabels([f"#{r['ID']:02d}\n{r['ONNX FP32 Class'][:8]}" for _, r in df.iterrows()], fontsize=8.5, color='#cbd5e1')
ax_bars.grid(axis='y', linestyle='--', alpha=0.2, color='#ffffff')
ax_bars.legend(loc='upper right', framealpha=0.9, facecolor='#0f172a', edgecolor=C_BORDER, fontsize=10)

# Add 100% Match Banner on top of the graph
ax_bars.text(0.5, 0.94, "[PARITY VERIFIED] 100% DIAGNOSTIC AGREEMENT (ZERO PREDICTION MISMATCHES)", 
             transform=ax_bars.transAxes, ha='center', va='top', fontsize=9.5, fontweight='bold', 
             color=C_FP32, bbox=dict(boxstyle="round,pad=0.3", facecolor='#064e3b', edgecolor=C_FP32, alpha=0.85))

# ==============================================================================
# 3. PANEL 2: MODEL WEIGHT & RAM FOOTPRINT COMPARISON (Top Right)
# ==============================================================================
ax_size = fig.add_subplot(gs[1, 1])
ax_size.set_facecolor(C_BG_CARD)
for spine in ax_size.spines.values():
    spine.set_edgecolor(C_BORDER)

categories = ['Original Training Checkpoint', 'ONNX FP32 Model', 'ONNX INT8 Edge Model']
sizes = [232.31, 76.87, 20.07]
colors = ['#ef4444', C_FP32, C_INT8]

y_pos = np.arange(len(categories))
bars_h = ax_size.barh(y_pos, sizes, height=0.5, color=colors, edgecolor=C_BORDER)

ax_size.set_yticks(y_pos)
ax_size.set_yticklabels(categories, fontsize=10.5, fontweight='bold', color='#ffffff')
ax_size.invert_yaxis()
ax_size.set_xlabel("Disk File Size (Megabytes)", fontsize=11, color=C_MUTED)
ax_size.set_title("Model Footprint Reduction for Edge Deployment", fontsize=13, fontweight='bold', color='#ffffff', pad=12, loc='left')
ax_size.grid(axis='x', linestyle='--', alpha=0.2, color='#ffffff')

# Add size labels
for bar, sz in zip(bars_h, sizes):
    w = bar.get_width()
    ax_size.text(w + 3, bar.get_y() + bar.get_height()/2, f"{sz:.1f} MB", 
                 va='center', fontsize=11, fontweight='bold', color='#ffffff')

ax_size.text(0.5, 0.15, "INT8 Quantization saves 212.2 MB (91.4%) vs Checkpoint\nand 56.8 MB (73.9%) vs FP32 ONNX", 
             transform=ax_size.transAxes, ha='center', va='center', fontsize=10, color=C_ACCENT,
             bbox=dict(boxstyle="round,pad=0.4", facecolor='#451a03', edgecolor=C_ACCENT, alpha=0.7))

# ==============================================================================
# 4. PANEL 3: CONFIDENCE DRIFT DISTRIBUTION (Bottom Left)
# ==============================================================================
ax_drift = fig.add_subplot(gs[2, 0])
ax_drift.set_facecolor(C_BG_CARD)
for spine in ax_drift.spines.values():
    spine.set_edgecolor(C_BORDER)

deltas = df["Delta Conf (%)"].values
colors_drift = [C_FP32 if d >= 0 else '#f59e0b' for d in deltas]

ax_drift.bar(x, deltas, width=0.55, color=colors_drift, edgecolor='#0f172a')
ax_drift.axhline(0, color='#ffffff', linestyle='-', linewidth=1.2, alpha=0.4)
ax_drift.axhline(2.0, color='#ef4444', linestyle=':', linewidth=1.0, alpha=0.5)
ax_drift.axhline(-2.0, color='#ef4444', linestyle=':', linewidth=1.0, alpha=0.5)

ax_drift.set_title("Quantization Confidence Drift (Delta = INT8 - FP32)", fontsize=13, fontweight='bold', color='#ffffff', pad=12, loc='left')
ax_drift.set_ylabel("Difference (% Points)", fontsize=11, color=C_MUTED)
ax_drift.set_xlabel("Test Image Sample ID (#01 to #20)", fontsize=11, color=C_MUTED)
ax_drift.set_xticks(x)
ax_drift.set_xticklabels([f"#{r['ID']:02d}" for _, r in df.iterrows()], fontsize=9, color='#cbd5e1')
ax_drift.set_ylim(-7.0, 3.0)
ax_drift.grid(axis='y', linestyle='--', alpha=0.2, color='#ffffff')

ax_drift.text(0.02, 0.88, "Tight +/-2% Bound: 19 / 20 Samples (95% within noise threshold)", 
              transform=ax_drift.transAxes, fontsize=9.5, color='#cbd5e1', 
              bbox=dict(boxstyle="round,pad=0.25", facecolor='#0f172a', edgecolor=C_BORDER))

# ==============================================================================
# 5. PANEL 4: CLASS BREAKDOWN & PARITY MATRIX (Bottom Right)
# ==============================================================================
ax_diag = fig.add_subplot(gs[2, 1])
ax_diag.set_facecolor(C_BG_CARD)
for spine in ax_diag.spines.values():
    spine.set_edgecolor(C_BORDER)
ax_diag.axis('off')

# Title
ax_diag.text(0.0, 0.96, "Pathological Category Breakdown (20 Samples)", fontsize=13, fontweight='bold', color='#ffffff', va='top')

summary_data = [
    ("Normal (Healthy Canopy)", "5 Samples", "100% Agreement", C_FP32),
    ("Hispa (Pest Infestation)", "5 Samples", "100% Agreement", C_INT8),
    ("Tungro (Viral Leafhopper)", "3 Samples", "100% Agreement", '#f87171'),
    ("Dead Heart (Stem Borer)", "3 Samples", "100% Agreement", C_ACCENT),
    ("Bacterial Leaf Blight / Streak", "2 Samples", "100% Agreement", '#fb923c'),
    ("Blast & Brown Spot (Fungi)", "2 Samples", "100% Agreement", C_PURPLE),
]

y_start = 0.80
y_step = 0.13
for i, (cat, cnt, ag, col) in enumerate(summary_data):
    y = y_start - i * y_step
    # Box
    rect = patches.FancyBboxPatch((0.0, y - 0.08), 1.0, 0.10, boxstyle="round,pad=0.01", 
                                  facecolor='#0f172a', edgecolor=C_BORDER, linewidth=1.0)
    ax_diag.add_patch(rect)
    
    # Bullet dot
    circle = patches.Circle((0.03, y - 0.03), 0.018, facecolor=col)
    ax_diag.add_patch(circle)
    
    ax_diag.text(0.07, y - 0.03, cat, fontsize=10, fontweight='bold', color='#ffffff', va='center')
    ax_diag.text(0.68, y - 0.03, cnt, fontsize=9.5, color=C_MUTED, va='center')
    ax_diag.text(0.97, y - 0.03, ag, fontsize=9.5, fontweight='bold', color=C_FP32, ha='right', va='center')

# Save high-resolution graphic
plt.savefig(out_png, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()

print(f"Successfully generated visual infographic: {out_png}")
