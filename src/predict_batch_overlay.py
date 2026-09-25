"""
============================================================
PADDY GUARD AI: 20-Sample Test Set Batch Inference & Visual Overlay
============================================================
Samples 20 random images from dataset/test_images, runs inference
using the champion EfficientNetV2-S model, overlays the predicted
disease label and confidence score directly onto each image,
and saves the annotated images plus a 4x5 summary collage grid.
============================================================
"""

import sys
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

from src.models import get_model
from src.utils import get_device


def run_batch_overlay(num_samples: int = 20, seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    device = get_device()

    test_dir = Path("dataset/test_images")
    if not test_dir.exists():
        test_dir = Path("test_images")

    out_dir = Path("results/test_predictions")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load class mapping
    mapping_path = Path("dataset/class_mapping.json")
    with open(mapping_path, "r") as f:
        mapping = json.load(f)
    id_to_class = {int(k): v for k, v in mapping["id_to_class"].items()}

    # 2. Load Champion Model (EfficientNetV2-S)
    ckpt_path = Path("checkpoints/efficientnet_v2_s_best.pth")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    print(f"\n[1/3] Loading Champion Architecture: EfficientNetV2-S (97.38% Accuracy)")
    model = get_model("efficientnet_v2_s", num_classes=len(id_to_class), pretrained=False)
    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()

    # 3. Preprocessing transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. Sample images
    all_images = sorted(list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpeg")))
    print(f"[2/3] Found {len(all_images)} test images in '{test_dir}'")
    sampled = random.sample(all_images, min(num_samples, len(all_images)))

    # Try loading a true-type font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 22)
        font_sm = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
        font_sm = font

    print("\n" + "=" * 80)
    print(f"{'#':<3} | {'Image Filename':<16} | {'Predicted Class':<26} | {'Confidence':<12} | {'Output Path'}")
    print("=" * 80)

    annotated_images = []
    raw_images_list = []

    # 5. Inference & Visual Overlay
    for idx, img_path in enumerate(sampled, 1):
        raw_img = Image.open(img_path).convert("RGB")
        tensor = transform(raw_img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)[0]
            conf, pred_id = torch.max(probs, dim=0)

        pred_class = id_to_class[pred_id.item()]
        conf_pct = conf.item() * 100.0
        display_name = pred_class.replace("_", " ").title()

        # Terminal Log
        out_filename = f"pred_{img_path.name}"
        save_path = out_dir / out_filename
        print(f"{idx:02d}  | {img_path.name:<16} | {display_name:<26} | {conf_pct:>6.2f}%     | {save_path.name}")

        # Draw Overlay directly on Image
        annotated = raw_img.copy()
        draw = ImageDraw.Draw(annotated)
        w, h = annotated.size

        # Color coding: Green if Normal (Healthy), Red/Amber if Disease
        if pred_class == "normal":
            badge_bg = (16, 128, 64)      # Forest Green
            badge_txt = "#FFFFFF"
        elif conf_pct >= 90.0:
            badge_bg = (180, 20, 20)      # Crimson Red
            badge_txt = "#FFFFFF"
        else:
            badge_bg = (200, 100, 10)     # Amber Orange
            badge_txt = "#FFFFFF"

        # Banner at top
        banner_h = max(58, int(h * 0.12))
        draw.rectangle([(0, 0), (w, banner_h)], fill=(15, 23, 42, 230)) # Dark slate header
        draw.rectangle([(0, banner_h - 4), (w, banner_h)], fill=badge_bg) # Accent bar

        title_text = f"PADDYSNAP AI: {display_name}"
        conf_text = f"Confidence: {conf_pct:.2f}%"

        draw.text((12, 8), title_text, fill=(255, 255, 255), font=font)
        draw.text((12, 34), conf_text, fill=(254, 240, 138), font=font_sm) # Pale gold

        # Save raw clean image (without predictions)
        raw_save_path = out_dir / f"raw_{img_path.name}"
        raw_img.save(raw_save_path, quality=95)

        annotated.save(save_path, quality=95)
        annotated_images.append((annotated, display_name, conf_pct))
        raw_images_list.append(raw_img)

    print("=" * 80)
    print(f"\n[3/3] Successfully generated {len(annotated_images)} prediction images + clean raw images in '{out_dir}'")

    # 6. Generate Summary Collage Grids (Both WITH and WITHOUT predictions)
    cols = 5
    rows = (len(annotated_images) + cols - 1) // cols
    cell_w, cell_h = 360, 480

    # Grid A: Annotated with Predictions
    grid_img = Image.new("RGB", (cols * cell_w, rows * cell_h), color=(15, 23, 42))
    for i, (img, name, conf) in enumerate(annotated_images):
        r = i // cols
        c = i % cols
        thumb = img.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
        grid_img.paste(thumb, (c * cell_w, r * cell_h))

    grid_path = out_dir / f"all_{len(annotated_images)}_predictions_grid.png"
    grid_img.save(grid_path, quality=95)
    print(f"Annotated Collage Grid (WITH predictions) saved -> {grid_path}")

    # Grid B: Clean Images (WITHOUT predictions)
    grid_raw = Image.new("RGB", (cols * cell_w, rows * cell_h), color=(15, 23, 42))
    for i, img in enumerate(raw_images_list):
        r = i // cols
        c = i % cols
        thumb = img.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
        grid_raw.paste(thumb, (c * cell_w, r * cell_h))

    grid_raw_path = out_dir / f"raw_{len(raw_images_list)}_images_grid.png"
    grid_raw.save(grid_raw_path, quality=95)
    print(f"Raw Collage Grid (WITHOUT predictions) saved    -> {grid_raw_path}")

    # Grid C: Master Combined One PNG (Raw vs Predicted Side-by-Side)
    header_h = 140
    footer_h = 70
    padding = 30
    gap = 30
    gw, gh = grid_raw.size
    canvas_w = padding * 2 + gw * 2 + gap
    canvas_h = header_h + gh + footer_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(15, 23, 42))
    draw_c = ImageDraw.Draw(canvas)

    draw_c.rectangle([(0, 0), (canvas_w, header_h)], fill=(10, 16, 30))
    draw_c.text((padding, 25), f"PADDYSNAP AI — {len(annotated_images)}-SAMPLE BATCH TEST EVALUATION (ALL-IN-ONE COMPARISON)", fill=(255, 255, 255), font=font)
    draw_c.text((padding, 85), "Inference Model: EfficientNetV2-S (97.38% Test Acc) | Held-Out Field Test Leaves", fill=(148, 163, 184), font=font_sm)

    col1_x = padding
    col2_x = padding + gw + gap
    grid_y = header_h + 45

    draw_c.text((col1_x, header_h + 10), "🌿 RAW FIELD IMAGES (WITHOUT PREDICTIONS / GROUND TRUTH)", fill=(52, 211, 153), font=font)
    draw_c.text((col2_x, header_h + 10), "🏷️ AI DIAGNOSTIC OVERLAYS (WITH PREDICTIONS & CONFIDENCE)", fill=(96, 165, 250), font=font)

    canvas.paste(grid_raw, (col1_x, grid_y))
    canvas.paste(grid_img, (col2_x, grid_y))

    draw_c.rectangle([(col1_x - 2, grid_y - 2), (col1_x + gw + 1, grid_y + gh + 1)], outline=(51, 65, 85), width=2)
    draw_c.rectangle([(col2_x - 2, grid_y - 2), (col2_x + gw + 1, grid_y + gh + 1)], outline=(51, 65, 85), width=2)

    foot_y = canvas_h - footer_h + 20
    draw_c.rectangle([(0, canvas_h - footer_h), (canvas_w, canvas_h)], fill=(10, 16, 30))
    draw_c.text((padding, foot_y), "🌾 PaddySnap AI Diagnostics Engine • Comprehensive Held-Out Field Batch", fill=(148, 163, 184), font=font_sm)

    master_path = out_dir / f"all_{len(annotated_images)}_raw_vs_predicted_grid.png"
    canvas.save(master_path, quality=95)
    print(f"Master One PNG Comparison (Raw vs Predicted) saved -> {master_path}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PaddySnap AI -- Batch Test Predictions & Overlay")
    parser.add_argument("--num", type=int, default=20, help="Number of test images to evaluate (e.g. 50, 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for selection (default: 42)")
    args = parser.parse_args()

    run_batch_overlay(num_samples=args.num, seed=args.seed)
