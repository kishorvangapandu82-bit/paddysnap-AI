"""
============================================================
PADDY GUARD AI -- MODULE 19: Single-Image Disease Prediction
============================================================
Purpose:
    Performs real-time disease diagnosis on any paddy leaf image (.jpg/.png),
    computing class probabilities and retrieving integrated agronomic
    recommendations (Module 20/21).

Usage:
    python src/predict.py --image path/to/leaf.jpg
    python src/predict.py --image path/to/leaf.jpg --model densenet121
============================================================
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
import torch
import torch.nn.functional as F

from src.utils import get_device
from src.models import get_model, SUPPORTED_MODELS
from src.preprocessing import get_val_transforms
from src.recommendations import get_recommendation


CLASS_MAPPING_FILE = Path("dataset/class_mapping.json")


def load_class_mapping() -> Tuple[Dict[str, int], Dict[str, str]]:
    """Loads class-to-id and id-to-class mappings."""
    if not CLASS_MAPPING_FILE.exists():
        raise FileNotFoundError(f"Class mapping file not found: {CLASS_MAPPING_FILE}")
    with open(CLASS_MAPPING_FILE, "r") as f:
        data = json.load(f)
    return data["class_to_id"], data["id_to_class"]


def load_trained_model(model_name: str, device: torch.device, num_classes: int = 10):
    """Loads model architecture and weights from pure weights or best checkpoint."""
    clean_name = model_name.lower().replace("-", "_")
    weights_path = Path(f"checkpoints/weights_only/{clean_name}_weights.pth")
    checkpoint_path = Path(f"checkpoints/{clean_name}_best.pth")

    if weights_path.exists():
        target_path = weights_path
    elif checkpoint_path.exists():
        target_path = checkpoint_path
    else:
        raise FileNotFoundError(f"No weights or checkpoint found for '{model_name}' at {weights_path} or {checkpoint_path}")

    data = torch.load(str(target_path), map_location=device, weights_only=False)
    state_dict = data["model_state_dict"] if isinstance(data, dict) and "model_state_dict" in data else data

    model = get_model(clean_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def predict_image(
    image_path: str,
    model_name: str = "efficientnet_v2_s",
    device: torch.device = None,
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Diagnoses a single paddy leaf image.
    Returns prediction dictionary with top predicted class, probabilities, and recommendations.
    """
    if device is None:
        device = get_device()

    class_to_id, id_to_class = load_class_mapping()
    num_classes = len(id_to_class)

    # 1. Load & Transform Image
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Input image not found: {img_path}")

    raw_image = Image.open(img_path).convert("RGB")
    transform = get_val_transforms(image_size=224)
    tensor = transform(raw_image).unsqueeze(0).to(device)

    # 2. Load Model & Run Inference
    model = load_trained_model(model_name, device, num_classes=num_classes)
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    # 3. Format Probabilities across all 10 classes
    all_probs = {
        id_to_class[str(i)]: float(round(probs[i] * 100, 2))
        for i in range(num_classes)
    }

    # Sort descending
    sorted_probs = sorted(all_probs.items(), key=lambda item: item[1], reverse=True)
    top_class, top_confidence = sorted_probs[0]

    # 4. Fetch Agronomy Guidance
    recommendation = get_recommendation(top_class)

    return {
        "image_path": str(img_path.as_posix()),
        "model_used": model_name,
        "predicted_class": top_class,
        "confidence": top_confidence,
        "top_k_predictions": sorted_probs[:top_k],
        "all_probabilities": all_probs,
        "recommendation": recommendation
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="PADDYSNAP AI -- Leaf Disease Prediction")
    parser.add_argument("--image", type=str, required=True, help="Path to input leaf image (.jpg/.png)")
    parser.add_argument("--model", type=str, default="efficientnet_v2_s",
                        help="Model architecture: resnet50, densenet121, efficientnet_v2_s, convnext_tiny, custom_cnn")
    parser.add_argument("--topk",  type=int, default=3, help="Number of top predictions to display")
    parser.add_argument("--json",  action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    result = predict_image(args.image, model_name=args.model, top_k=args.topk)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    # Formatted Terminal Report
    print("\n" + "=" * 70)
    print("  PADDYSNAP AI -- LEAF DISEASE DIAGNOSIS REPORT")
    print("=" * 70)
    print(f"  Input Image      : {result['image_path']}")
    print(f"  Model Employed   : {result['model_used'].upper()}")
    print(f"  Diagnosis Result : {result['predicted_class'].upper()}")
    print(f"  Confidence Score : {result['confidence']:.2f}%")

    print("\n  Top Predictions:")
    for rank, (cls, conf) in enumerate(result["top_k_predictions"], 1):
        bar = "█" * int(conf / 5)
        print(f"    {rank}. {cls:<26} {conf:>6.2f}%  |{bar}")

    rec = result["recommendation"]
    print("\n" + "-" * 70)
    print(f"  AGRONOMY GUIDANCE: {rec['common_name']}")
    print(f"  Causal Organism  : {rec['causal_organism']}")
    print(f"  Disease Severity : {rec['severity']}")
    print("-" * 70)

    print("\n  Key Symptoms:")
    for s in rec["symptoms"]:
        print(f"    • {s}")

    print("\n  Chemical / Organic Treatment:")
    for c in rec["chemical_control"]:
        print(f"    • {c}")

    fg = rec["fertilizer_nutrient_guidance"]
    print("\n  Fertilizer & Nutrient Adjustments:")
    print(f"    [Nitrogen]   : {fg['nitrogen_action']}")
    print(f"    [Potassium]  : {fg['potassium_action']}")
    print(f"    [Nutrients]  : {fg['micronutrients']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
