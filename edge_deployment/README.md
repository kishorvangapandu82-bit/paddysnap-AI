# 🌾 PaddySnap AI — Raspberry Pi Edge Deployment

Deploy the **PaddySnap AI** paddy disease detector on a Raspberry Pi 4/5 using ONNX Runtime for real-time, offline inference in the field.

---

## 📁 Folder Structure

```
edge_deployment/
├── inference.py          # Main inference script
├── requirements_pi.txt   # Python dependencies for Pi
├── setup.sh              # One-command setup script
├── models/               # Place your .onnx model here
│   └── efficientnet_v2_s_int8.onnx  ← recommended
└── test_images/          # Optional: test images
```

---

## ⚙️ Requirements

| Hardware | Minimum |
|----------|---------|
| Raspberry Pi | 4B (2GB RAM) or Pi 5 |
| Storage | 16GB microSD (Class 10) |
| Camera | USB webcam or Pi Camera Module |
| OS | Raspberry Pi OS 64-bit (Bookworm) |

---

## 🚀 Quick Setup (on Raspberry Pi)

### Step 1 — Clone or copy files to your Pi

```bash
# Option A: Clone from GitHub
git clone https://github.com/kishorvangapandu82-bit/paddysnap-AI.git
cd paddysnap-AI/edge_deployment

# Option B: Copy via SCP from your PC
scp -r edge_deployment/ pi@raspberrypi.local:~/paddysnap/
```

### Step 2 — Copy the ONNX model

```bash
mkdir -p models/
# Copy from project's deployment folder
cp ../deployment/onnx/efficientnet_v2_s_int8.onnx models/
```

### Step 3 — Run the setup script

```bash
bash setup.sh
```

This will:
- ✅ Install system dependencies
- ✅ Create a Python virtual environment
- ✅ Install ONNX Runtime + OpenCV
- ✅ Create a `run.sh` launcher

---

## 🎯 Usage

### Live Camera Mode
```bash
./run.sh camera
# Press SPACE to capture and predict
# Press Q to quit
```

### Single Image
```bash
./run.sh leaf_photo.jpg
```

### Batch Folder
```bash
./run.sh images/
```

### Manual
```bash
source paddysnap_env/bin/activate
python3 inference.py --model models/efficientnet_v2_s_int8.onnx --source camera
```

---

## 📊 Model Performance on Raspberry Pi 4

| Model | Size | Latency (Pi 4) | Accuracy |
|-------|------|----------------|----------|
| EfficientNet-V2-S INT8 | 20 MB | ~350ms | 97.9% |
| DenseNet121 INT8 | 7.5 MB | ~200ms | 94.2% |
| Custom CNN INT8 | 7.6 MB | ~80ms | 88.5% |

> ✅ **Recommended**: `efficientnet_v2_s_int8.onnx` — best accuracy for field use

---

## 🌿 Detected Diseases

| Class | Disease |
|-------|---------|
| `bacterial_leaf_blight` | Bacterial Leaf Blight |
| `bacterial_leaf_streak` | Bacterial Leaf Streak |
| `bacterial_panicle_blight` | Bacterial Panicle Blight |
| `blast` | Rice Blast |
| `brown_spot` | Brown Spot |
| `dead_heart` | Dead Heart |
| `downy_mildew` | Downy Mildew |
| `hispa` | Hispa |
| `normal` | Healthy Plant |
| `tungro` | Tungro Virus |

---

## 🔧 Troubleshooting

**ONNX Runtime install fails:**
```bash
pip install onnxruntime --extra-index-url https://rpi4.piwheels.org/simple
```

**Camera not detected:**
```bash
ls /dev/video*       # Check camera device
vcgencmd get_camera  # Check Pi Camera status
```

**Low memory error:**
```bash
# Add swap space
sudo dphys-swapfile swapoff
sudo sed -i 's/CONF_SWAPSIZE=100/CONF_SWAPSIZE=1024/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

---

## 📜 License
MIT License — PaddySnap AI Project
