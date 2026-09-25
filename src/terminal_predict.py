"""
============================================================
PADDY GUARD AI: Terminal Image Viewer & Diagnosis Engine
============================================================
Displays the actual leaf image directly inside the terminal
using 24-bit ANSI TrueColor blocks, with predicted disease
class, confidence score, and top-3 breakdown displayed
both on the image itself and in the terminal.
============================================================
"""

import sys
import json
import random
import argparse
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


def render_image_in_terminal(image: Image.Image, cols: int = 56, rows: int = 40) -> str:
    """
    Renders a PIL image directly in the terminal using 24-bit ANSI TrueColor
    and Unicode half-block characters (▀). Each character displays two vertical pixels.
    """
    thumb = image.resize((cols, rows))
    rgb = thumb.convert("RGB")
    pixels = rgb.load()
    
    lines = []
    for y in range(0, rows, 2):
        row_str = []
        for x in range(cols):
            r1, g1, b1 = pixels[x, y]
            if y + 1 < rows:
                r2, g2, b2 = pixels[x, y + 1]
            else:
                r2, g2, b2 = (0, 0, 0)
            # \033[38;2;R;G;Bm = foreground color (top pixel)
            # \033[48;2;R;G;Bm = background color (bottom pixel)
            row_str.append(f"\033[38;2;{r1};{g1};{b1}m\033[48;2;{r2};{g2};{b2}m\u2580\033[0m")
        lines.append("".join(row_str))
    return "\n".join(lines)


def draw_label_on_image(image: Image.Image, label: str, confidence: float) -> Image.Image:
    """Draws a clean, high-contrast banner directly on the image with class and confidence."""
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    w, h = annotated.size
    
    try:
        font_title = ImageFont.truetype("arial.ttf", max(18, int(h * 0.045)))
        font_sub = ImageFont.truetype("arial.ttf", max(14, int(h * 0.035)))
    except Exception:
        font_title = ImageFont.load_default()
        font_sub = font_title

    banner_h = max(55, int(h * 0.12))
    # Dark header background
    draw.rectangle([(0, 0), (w, banner_h)], fill=(15, 23, 42))
    
    # Accent strip: Green if healthy normal, Red/Amber for diseases
    is_normal = (label.lower() == "normal")
    accent_color = (16, 185, 129) if is_normal else (239, 68, 68)
    draw.rectangle([(0, banner_h - 4), (w, banner_h)], fill=accent_color)
    
    # Header texts
    draw.text((12, 6), f"CLASS: {label.upper()}", fill=(255, 255, 255), font=font_title)
    draw.text((12, int(banner_h * 0.52)), f"CONFIDENCE: {confidence:.2f}%", fill=(254, 240, 138), font=font_sub)
    
    return annotated


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Paddy Guard AI -- Terminal Visual Diagnosis")
    parser.add_argument("--num", type=int, default=20, help="Number of random test images (default: 20)")
    parser.add_argument("--cols", type=int, default=58, help="Terminal display width in columns (default: 58)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for selection (default: 42)")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()

    # Paths
    test_dir = Path("dataset/test_images")
    if not test_dir.exists():
        test_dir = Path("test_images")
    out_dir = Path("results/test_predictions")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Class Mapping
    with open("dataset/class_mapping.json", "r") as f:
        mapping = json.load(f)
    id_to_class = {int(k): v for k, v in mapping["id_to_class"].items()}

    # Model (EfficientNetV2-S Champion)
    model = get_model("efficientnet_v2_s", num_classes=len(id_to_class), pretrained=False)
    checkpoint = torch.load("checkpoints/efficientnet_v2_s_best.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    all_images = sorted(list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpeg")))
    sampled = random.sample(all_images, min(args.num, len(all_images)))

    print("\n" + "=" * 70)
    print("  🌾 PADDYSNAP AI — TERMINAL VISUAL DIAGNOSIS")
    print(f"  Champion Model: EfficientNetV2-S (97.38% Test Accuracy)")
    print(f"  Evaluating {len(sampled)} Random Images from: {test_dir}")
    print("=" * 70)

    for idx, img_path in enumerate(sampled, 1):
        raw_img = Image.open(img_path).convert("RGB")
        tensor = transform(raw_img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)[0]
            topk_probs, topk_indices = torch.topk(probs, k=3)

        pred_id = topk_indices[0].item()
        pred_class = id_to_class[pred_id]
        conf_pct = topk_probs[0].item() * 100.0
        display_name = pred_class.replace("_", " ").title()

        # 1. Overlay banner on image
        annotated_img = draw_label_on_image(raw_img, display_name, conf_pct)
        save_path = out_dir / f"pred_{img_path.name}"
        annotated_img.save(save_path, quality=95)

        # 2. Render image directly in the terminal
        # Aspect ratio: 480x640 -> cols=58, rows=54 (27 terminal lines)
        term_img_str = render_image_in_terminal(annotated_img, cols=args.cols, rows=52)

        # 3. Print image & diagnostic output in terminal
        print(f"\n[{idx:02d} / {len(sampled):02d}] 📸 IMAGE: {img_path.name}")
        print(term_img_str)
        print("-" * args.cols)
        print(f"  🎯 PREDICTED CLASS : \033[1;33m{display_name}\033[0m")
        print(f"  📊 CONFIDENCE      : \033[1;32m{conf_pct:.2f}%\033[0m")
        print("  🏆 TOP-3 PROBABILITIES:")
        for rank, (k_prob, k_idx) in enumerate(zip(topk_probs, topk_indices), 1):
            k_name = id_to_class[k_idx.item()].replace("_", " ").title()
            bar = "█" * int(k_prob.item() * 20)
            print(f"     {rank}. {k_name:<24} : {k_prob.item()*100:>5.2f}%  {bar}")
        print("-" * args.cols)

    print("\n" + "=" * 70)
    print(f"  ✅ Completed {len(sampled)} visual terminal diagnoses!")
    print(f"  📁 High-res annotated images saved to: {out_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
