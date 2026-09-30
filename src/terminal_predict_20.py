"""
============================================================
PADDY GUARD AI -- Terminal Predictor for 20 Test Images
============================================================
Purpose:
    Runs inference on 20 random test images from dataset/test_images/
    and displays predicted disease, confidence score, and
    fertilizer/nutrient recommendations.

Usage:
    venv\\Scripts\\python.exe src/terminal_predict_20.py
    venv\\Scripts\\python.exe src/terminal_predict_20.py --model convnext_tiny
============================================================
"""

import sys
import json
import random
import argparse
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from PIL import Image

from src.utils import set_seed, get_device
from src.models import get_model
from src.preprocessing import get_val_transforms
from src.recommendations import get_recommendation


def load_class_mapping():
    with open("dataset/class_mapping.json", "r") as f:
        data = json.load(f)
    return data["class_to_id"], data["id_to_class"]


def render_ascii_thumbnail(img: Image.Image, width: int = 44, height: int = 14) -> str:
    """Renders a low-res ASCII visual thumbnail of the leaf image for the terminal."""
    resized = img.resize((width, height)).convert("L")
    ascii_chars = " .:-=+*#%@"
    pixels = resized.getdata()
    lines = []
    for y in range(height):
        row = ""
        for x in range(width):
            val = pixels[y * width + x]
            char_idx = int((val / 255.0) * (len(ascii_chars) - 1))
            row += ascii_chars[char_idx]
        lines.append("| " + row + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run 20 test image inference with ASCII leaf preview and fertilizer recommendations")
    parser.add_argument("--model", type=str, default="efficientnet_v2_s", help="Model name")
    parser.add_argument("--num_images", type=int, default=20, help="Number of random test images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no_ascii", action="store_true", help="Disable ASCII image preview")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    print(f"\n" + "="*70)
    print(f"PADDY GUARD AI -- BATCH TEST PREDICTION & TERMINAL LEAF PREVIEW")
    print(f"="*70)
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Number of test images: {args.num_images}\n")

    # Load class mapping
    class_to_id, id_to_class = load_class_mapping()
    num_classes = len(id_to_class)

    # Load model weights
    clean_name = args.model.lower().replace("-", "_")
    weights_path = Path(f"checkpoints/weights_only/{clean_name}_weights.pth")
    checkpoint_path = Path(f"checkpoints/{clean_name}_best.pth")

    if weights_path.exists():
        target_path = weights_path
    elif checkpoint_path.exists():
        target_path = checkpoint_path
    else:
        print(f"Error: Weights for {args.model} not found.")
        sys.exit(1)

    model = get_model(args.model, num_classes=num_classes, pretrained=False).to(device)
    state_dict = torch.load(target_path, map_location=device)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    transforms = get_val_transforms(image_size=224)

    # Pick random test images
    test_dir = Path("dataset/test_images")
    image_paths = sorted(list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png")))

    if not image_paths:
        print("No images found in dataset/test_images/")
        sys.exit(1)

    random.seed(args.seed)
    selected_paths = random.sample(image_paths, min(args.num_images, len(image_paths)))

    print(f"Loaded {len(selected_paths)} test images. Starting inference...\n")

    for idx, img_path in enumerate(selected_paths, start=1):
        raw_image = Image.open(img_path).convert("RGB")
        tensor = transforms(raw_image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probs = F.softmax(outputs, dim=1)[0]
            pred_id = torch.argmax(probs).item()
            conf = probs[pred_id].item() * 100.0

        pred_class = id_to_class[str(pred_id)]
        rec = get_recommendation(pred_class)
        common_name = rec.get("common_name", pred_class)
        fg = rec.get("fertilizer_nutrient_guidance", {})

        print(f"+" + "-"*68 + "+")
        print(f"| #{idx:02d} | IMAGE: {img_path.name:<25} CONFIDENCE: [{conf:6.2f}%] |")
        print(f"+" + "-"*68 + "+")
        print(f"| Diagnosis: {common_name:<55} |")
        
        if not args.no_ascii:
            print(f"+" + "-"*68 + "+")
            print(f"| ASCII LEAF PREVIEW (44x14 pixels):                                |")
            ascii_art = render_ascii_thumbnail(raw_image, width=44, height=12)
            for line in ascii_art.splitlines():
                print(f"{line:<69}|")

        print(f"+" + "-"*68 + "+")
        print(f"| FERTILIZER & NUTRIENT ADVISORY:                                    |")
        
        n_act = fg.get("nitrogen_action", "N/A")
        k_act = fg.get("potassium_action", "N/A")
        micro = fg.get("micronutrients", "N/A")

        # Wrap text for clean CLI view
        def print_wrapped(prefix, text):
            words = text.split()
            lines = []
            curr = prefix
            for w in words:
                if len(curr) + len(w) + 1 > 64:
                    lines.append(curr)
                    curr = "|     " + w
                else:
                    curr += (" " if curr != "|     " and curr != prefix else "") + w
            lines.append(curr)
            for l in lines:
                print(f"{l:<69}|")

        print_wrapped("|  * Nitrogen (N): ", n_act)
        print_wrapped("|  * Potassium (K): ", k_act)
        print_wrapped("|  * Micronutrients: ", micro)
        print(f"+" + "-"*68 + "+\n")


if __name__ == "__main__":
    main()
