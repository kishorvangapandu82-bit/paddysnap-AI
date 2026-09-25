"""
PaddySnap AI — Raspberry Pi Edge Inference Script
==================================================
Runs paddy disease detection using ONNX Runtime on Raspberry Pi.
Supports: USB webcam, Pi Camera, or single image file input.

Usage:
    python inference.py --source camera         # live webcam
    python inference.py --source image.jpg      # single image
    python inference.py --source images/        # folder of images
"""

import argparse
import time
import os
import sys
import numpy as np
from PIL import Image
import onnxruntime as ort

# ── Class Labels ────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "bacterial_leaf_blight",
    "bacterial_leaf_streak",
    "bacterial_panicle_blight",
    "blast",
    "brown_spot",
    "dead_heart",
    "downy_mildew",
    "hispa",
    "normal",
    "tungro",
]

# ── Treatment Recommendations ────────────────────────────────────────────────
RECOMMENDATIONS = {
    "bacterial_leaf_blight": "Apply copper-based bactericides. Avoid excess nitrogen fertilizer.",
    "bacterial_leaf_streak": "Remove infected plants. Apply streptomycin sulfate spray.",
    "bacterial_panicle_blight": "Use disease-free seeds. Apply potassium silicate.",
    "blast": "Apply tricyclazole or isoprothiolane fungicide immediately.",
    "brown_spot": "Apply mancozeb or iprodione fungicide. Improve soil nutrition.",
    "dead_heart": "Apply carbofuran granules to soil. Check for stem borer larvae.",
    "downy_mildew": "Apply metalaxyl fungicide. Improve field drainage.",
    "hispa": "Apply chlorpyrifos or quinalphos insecticide. Remove egg masses.",
    "normal": "✅ Plant is healthy! Continue regular care and monitoring.",
    "tungro": "Remove infected plants immediately. Control green leafhopper vectors.",
}

# ── Image Preprocessing ──────────────────────────────────────────────────────
IMG_SIZE = (224, 224)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image: Image.Image) -> np.ndarray:
    """Resize, normalize and convert image to model input tensor."""
    img = image.convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)          # HWC → CHW
    return arr[np.newaxis, ...]            # add batch dim → (1, 3, 224, 224)


# ── Inference ────────────────────────────────────────────────────────────────
def load_model(model_path: str) -> ort.InferenceSession:
    """Load ONNX model with CPU execution provider."""
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        sys.exit(1)
    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )
    print(f"[INFO] Model loaded: {os.path.basename(model_path)}")
    return session


def predict(session: ort.InferenceSession, image: Image.Image) -> tuple:
    """Run inference and return (class_name, confidence, latency_ms)."""
    tensor = preprocess(image)
    input_name = session.get_inputs()[0].name

    t0 = time.perf_counter()
    outputs = session.run(None, {input_name: tensor})
    latency_ms = (time.perf_counter() - t0) * 1000

    logits = outputs[0][0]
    probs  = softmax(logits)
    idx    = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]) * 100, latency_ms


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def print_result(class_name: str, confidence: float, latency_ms: float):
    """Pretty print prediction result."""
    print("\n" + "═" * 50)
    print(f"  🌾 PaddySnap AI — Prediction Result")
    print("═" * 50)
    print(f"  Disease   : {class_name.replace('_', ' ').title()}")
    print(f"  Confidence: {confidence:.1f}%")
    print(f"  Latency   : {latency_ms:.1f} ms")
    print(f"\n  💊 Recommendation:")
    print(f"  {RECOMMENDATIONS.get(class_name, 'Consult an agronomist.')}")
    print("═" * 50 + "\n")


# ── Modes ────────────────────────────────────────────────────────────────────
def run_image(session, path: str):
    """Predict on a single image file."""
    print(f"[INFO] Processing image: {path}")
    img = Image.open(path)
    cls, conf, lat = predict(session, img)
    print_result(cls, conf, lat)


def run_folder(session, folder: str):
    """Predict on all images in a folder."""
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    files = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
    if not files:
        print(f"[WARN] No images found in: {folder}")
        return
    print(f"[INFO] Found {len(files)} images in: {folder}\n")
    for fname in files:
        fpath = os.path.join(folder, fname)
        img = Image.open(fpath)
        cls, conf, lat = predict(session, img)
        print(f"  {fname:<30} → {cls:<28} {conf:5.1f}%  ({lat:.0f}ms)")


def run_camera(session):
    """Live inference from USB webcam or Pi Camera."""
    try:
        import cv2
    except ImportError:
        print("[ERROR] OpenCV not installed. Run: pip install opencv-python-headless")
        sys.exit(1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        sys.exit(1)

    print("[INFO] Camera started. Press 'q' to quit, SPACE to capture & predict.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Show live feed
        cv2.putText(frame, "Press SPACE to predict | Q to quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("PaddySnap AI — Edge Camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            # Convert frame and predict
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            cls, conf, lat = predict(session, img_pil)
            print_result(cls, conf, lat)

            # Overlay result on frame
            label = f"{cls.replace('_',' ').title()} ({conf:.1f}%)"
            cv2.putText(frame, label, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("PaddySnap AI — Edge Camera", frame)
            cv2.waitKey(2000)

    cap.release()
    cv2.destroyAllWindows()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="PaddySnap AI — Raspberry Pi Edge Inference"
    )
    parser.add_argument(
        "--model",
        default="models/efficientnet_v2_s_int8.onnx",
        help="Path to ONNX model file"
    )
    parser.add_argument(
        "--source",
        default="camera",
        help="Input source: 'camera', path to image file, or path to folder"
    )
    args = parser.parse_args()

    print("\n🌾 PaddySnap AI — Edge Deployment")
    print(f"   Model : {args.model}")
    print(f"   Source: {args.source}\n")

    session = load_model(args.model)

    if args.source == "camera":
        run_camera(session)
    elif os.path.isdir(args.source):
        run_folder(session, args.source)
    elif os.path.isfile(args.source):
        run_image(session, args.source)
    else:
        print(f"[ERROR] Invalid source: {args.source}")
        sys.exit(1)


if __name__ == "__main__":
    main()
