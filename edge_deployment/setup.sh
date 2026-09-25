#!/bin/bash
# ============================================================
# PaddySnap AI — Raspberry Pi One-Command Setup Script
# ============================================================
# Run with: bash setup.sh
# Tested on: Raspberry Pi 4 / 5 running Raspberry Pi OS (64-bit)

set -e  # Exit on error

echo ""
echo "🌾 ============================================="
echo "   PaddySnap AI — Raspberry Pi Setup"
echo "   ============================================="
echo ""

# ── Step 1: System Update ────────────────────────────────────
echo "[1/6] Updating system packages..."
sudo apt-get update -qq
sudo apt-get install -y python3-pip python3-venv python3-dev \
    libatlas-base-dev libopenblas-dev libjpeg-dev zlib1g-dev \
    libhdf5-dev git wget curl
echo "      ✅ System packages installed"

# ── Step 2: Create Virtual Environment ──────────────────────
echo ""
echo "[2/6] Creating Python virtual environment..."
python3 -m venv paddysnap_env
source paddysnap_env/bin/activate
pip install --upgrade pip -q
echo "      ✅ Virtual environment created: paddysnap_env/"

# ── Step 3: Install Python Dependencies ─────────────────────
echo ""
echo "[3/6] Installing Python dependencies..."
pip install -r requirements_pi.txt -q
echo "      ✅ Python packages installed"

# ── Step 4: Download ONNX Model ─────────────────────────────
echo ""
echo "[4/6] Setting up model directory..."
mkdir -p models

if [ ! -f "models/efficientnet_v2_s_int8.onnx" ]; then
    echo "      [!] Model not found in models/"
    echo "      Copy your ONNX model from the project's deployment/onnx/ folder:"
    echo ""
    echo "      scp user@your-pc:/path/to/paddy-guard-ai/deployment/onnx/efficientnet_v2_s_int8.onnx models/"
    echo ""
    echo "      Or copy manually from USB drive to: $(pwd)/models/"
else
    echo "      ✅ Model found: models/efficientnet_v2_s_int8.onnx"
fi

# ── Step 5: Test Inference ───────────────────────────────────
echo ""
echo "[5/6] Running system test..."
if [ -f "models/efficientnet_v2_s_int8.onnx" ] && [ -f "test_images/sample.jpg" ]; then
    python3 inference.py --model models/efficientnet_v2_s_int8.onnx \
                         --source test_images/sample.jpg
    echo "      ✅ Inference test passed"
else
    echo "      ⚠️  Skipping test (add a test image to test_images/sample.jpg)"
fi

# ── Step 6: Create Launcher Script ──────────────────────────
echo ""
echo "[6/6] Creating launcher script..."
cat > run.sh << 'EOF'
#!/bin/bash
source paddysnap_env/bin/activate
echo "🌾 Starting PaddySnap AI..."
python3 inference.py --model models/efficientnet_v2_s_int8.onnx --source "$@"
EOF
chmod +x run.sh
echo "      ✅ Launcher created: run.sh"

# ── Done ─────────────────────────────────────────────────────
echo ""
echo "🎉 ============================================="
echo "   Setup Complete!"
echo "   ============================================="
echo ""
echo "   Usage:"
echo "   ./run.sh camera              # Live camera"
echo "   ./run.sh image.jpg           # Single image"
echo "   ./run.sh images/             # Batch folder"
echo ""
echo "   Or manually:"
echo "   source paddysnap_env/bin/activate"
echo "   python3 inference.py --source camera"
echo ""
