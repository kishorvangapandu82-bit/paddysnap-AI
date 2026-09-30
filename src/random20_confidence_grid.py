"""
Picks 20 random images from dataset/test_images/,
runs EfficientNetV2-S inference, and saves a 4x5 grid
with predicted class + confidence score overlaid on each image.
"""
import sys, json, random
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import get_model
from src.preprocessing import get_val_transforms

# ── Config ──────────────────────────────────────────────
SEED         = 42
NUM_IMAGES   = 20
MODEL_NAME   = "efficientnet_v2_s"
TEST_DIR     = PROJECT_ROOT / "dataset" / "test_images"
WEIGHTS_PATH = PROJECT_ROOT / "checkpoints" / "weights_only" / "efficientnet_v2_s_weights.pth"
CLASS_MAP    = PROJECT_ROOT / "dataset" / "class_mapping.json"
OUT_DIR      = PROJECT_ROOT / "results" / "test_predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH     = OUT_DIR / "efficientnet_random20_confidence_grid.png"

# Bright neon colour palette per class
CLASS_COLORS = {
    "bacterial_leaf_blight":   (255,  50,  50),   # vivid red
    "bacterial_leaf_streak":   (255, 165,   0),   # bright orange
    "bacterial_panicle_blight":(230,   0, 255),   # neon magenta
    "blast":                   (255,  20, 147),   # deep pink
    "brown_spot":              (210, 105,  30),   # chocolate orange
    "dead_heart":              (180, 180, 180),   # bright silver
    "downy_mildew":            ( 30, 200, 255),   # electric cyan
    "hispa":                   (255, 230,   0),   # neon yellow
    "normal":                  ( 57, 255, 100),   # neon green
    "tungro":                  (255,  80,   0),   # blazing orange-red
}

# ── Load class mapping ───────────────────────────────────
with open(CLASS_MAP) as f:
    mapping = json.load(f)
id_to_class = mapping["id_to_class"]

# ── Load model ───────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

model = get_model(MODEL_NAME, num_classes=10, pretrained=False)
state_dict = torch.load(str(WEIGHTS_PATH), map_location=device, weights_only=False)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
print(f"  Model loaded: {MODEL_NAME}")

transform = get_val_transforms(image_size=224)

# ── Pick 20 random images ────────────────────────────────
random.seed(SEED)
all_imgs = sorted(list(TEST_DIR.glob("*.jpg")))
chosen   = random.sample(all_imgs, NUM_IMAGES)
print(f"  Picked {NUM_IMAGES} random images from {len(all_imgs)} total")

# ── Run inference on each ────────────────────────────────
THUMB  = 320          # thumbnail size per cell
COLS   = 4
ROWS   = 5
PAD    = 12           # padding between cells
BORDER = 4

cell_w = THUMB + 2 * BORDER
cell_h = THUMB + 2 * BORDER        # no separate header bar — overlay only
grid_w = COLS * cell_w + (COLS + 1) * PAD
grid_h = ROWS * cell_h + (ROWS + 1) * PAD + 60   # +60 for title bar

canvas = Image.new("RGB", (grid_w, grid_h), (18, 18, 28))
draw   = ImageDraw.Draw(canvas)

# Title
try:
    title_font = ImageFont.truetype("arial.ttf", 26)
    label_font = ImageFont.truetype("arial.ttf", 17)
    conf_font  = ImageFont.truetype("arial.ttf", 22)
except:
    title_font = ImageFont.load_default()
    label_font = title_font
    conf_font  = title_font

draw.text((grid_w // 2, 20), "EfficientNetV2-S — 20 Random Test Images",
          fill=(255, 255, 255), font=title_font, anchor="mm")

results = []
print("\n  {'Image':<18} {'Prediction':<28} {'Confidence':>10}")
print("  " + "-" * 58)

for idx, img_path in enumerate(chosen):
    raw  = Image.open(img_path).convert("RGB")
    tens = transform(raw).unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
            logits = model(tens)
            probs  = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_id   = int(np.argmax(probs))
    pred_cls  = id_to_class[str(pred_id)]
    conf      = float(probs[pred_id]) * 100.0
    color     = CLASS_COLORS.get(pred_cls, (200, 200, 200))
    label_str = pred_cls.replace("_", " ").title()

    results.append((img_path.name, label_str, conf))
    print(f"  {img_path.name:<18} {label_str:<28} {conf:>9.2f}%")

    # ── Draw cell ────────────────────────────────────────
    row = idx // COLS
    col = idx  % COLS
    x0  = PAD + col * (cell_w + PAD)
    y0  = 60 + PAD + row * (cell_h + PAD)

    # Coloured border
    draw.rectangle([x0, y0, x0 + cell_w, y0 + cell_h], outline=color, width=BORDER)

    # Thumbnail — paste directly (no header bar)
    thumb = raw.resize((THUMB, THUMB), Image.LANCZOS)
    canvas.paste(thumb, (x0 + BORDER, y0 + BORDER))

    # ── Overlays drawn on top of the thumbnail ───────────
    img_x = x0 + BORDER   # pixel origin of the thumbnail inside canvas
    img_y = y0 + BORDER

    # — TOP strip: "ClassName — XX.X%" (colour-tinted) —
    top_strip_h = 42
    top_label   = f"{label_str}  —  {conf:.1f}%"
    # tint = class colour at 60% opacity blended over near-black base
    tr, tg, tb  = color
    top_bg      = (max(0, tr // 5), max(0, tg // 5), max(0, tb // 5))  # dim tint
    top_strip   = Image.new("RGBA", (THUMB, top_strip_h), (0, 0, 0, 0))
    ts          = ImageDraw.Draw(top_strip)
    ts.rectangle([0, 0, THUMB, top_strip_h], fill=(*top_bg, 210))
    canvas.paste(Image.new("RGB", (THUMB, top_strip_h), top_bg),
                 (img_x, img_y),
                 mask=top_strip.split()[3])
    # bright label text
    draw.text((img_x + THUMB // 2, img_y + top_strip_h // 2),
              top_label, fill=color, font=conf_font, anchor="mm")

    # — BOTTOM strip: filename only (subtle dark) —
    bot_strip_h = 26
    bot_strip_y = img_y + THUMB - bot_strip_h
    bot_strip   = Image.new("RGBA", (THUMB, bot_strip_h), (0, 0, 0, 0))
    bs          = ImageDraw.Draw(bot_strip)
    bs.rectangle([0, 0, THUMB, bot_strip_h], fill=(8, 8, 18, 175))
    canvas.paste(Image.new("RGB", (THUMB, bot_strip_h), (8, 8, 18)),
                 (img_x, bot_strip_y),
                 mask=bot_strip.split()[3])
    draw.text((img_x + THUMB // 2, bot_strip_y + bot_strip_h // 2),
              img_path.name, fill=(200, 200, 200), font=label_font, anchor="mm")

canvas.save(str(OUT_PATH), quality=95)
print(f"\n  Grid saved -> {OUT_PATH}")
print(f"  Mean confidence: {sum(r[2] for r in results)/len(results):.2f}%")
print("  DONE.")
