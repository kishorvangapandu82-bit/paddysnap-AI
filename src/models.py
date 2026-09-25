"""
============================================================
PADDY GUARD AI -- MODULE 6: Model Architectures
============================================================
Purpose:
    Defines 5 deep learning architectures for Paddy Leaf Disease Classification:
      1. ResNet-50           (Transfer Learning)
      2. DenseNet-121        (Transfer Learning)
      3. EfficientNetV2-S    (Transfer Learning)
      4. ConvNeXt-Tiny       (Transfer Learning)
      5. PaddySnapNet        (Custom CNN with Residual + SE-Attention from Scratch)

Usage:
    python src/models.py
============================================================
"""

from typing import Tuple, Dict
import torch
import torch.nn as nn
import torchvision.models as models


# ==============================================================
# 1. Custom CNN: PaddySnapNet (Modern Residual + SE Attention)
# ==============================================================

class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation Channel Attention Block.
    Dynamically recalibrates channel-wise feature responses to emphasize disease spots.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced_channels = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced_channels, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        scale = self.fc(x).view(b, c, 1, 1)
        return x * scale


class ResidualBlock(nn.Module):
    """
    Residual Convolutional Block with optional Squeeze-and-Excitation.
    x_out = Activation(x + SE(Conv2(Conv1(x))))
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, use_se: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.act1  = nn.SiLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        
        self.se = SqueezeExcitation(out_channels) if use_se else nn.Identity()
        self.act2 = nn.SiLU(inplace=True)
        
        # Shortcut connection for dimension matching
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.act2(out + residual)
        return out


class PaddySnapNet(nn.Module):
    """
    Custom Lightweight CNN with Residual Connections and Squeeze-and-Excitation Attention.
    Designed specifically from scratch for Paddy Leaf Disease Classification.
    Total Parameters: ~7.85 Million (Lightweight, high capacity).
    """
    def __init__(self, num_classes: int = 10, dropout: float = 0.4):
        super().__init__()
        
        # Stem: Initial downsampling and edge feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False), # 224 -> 112
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)                            # 112 -> 56
        )
        
        # Stage 1: 32 -> 64 channels
        self.stage1 = nn.Sequential(
            ResidualBlock(32, 64, stride=1, use_se=True),
            ResidualBlock(64, 64, stride=1, use_se=True)
        )
        
        # Stage 2: 64 -> 128 channels (Downsample 56 -> 28)
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2, use_se=True),
            ResidualBlock(128, 128, stride=1, use_se=True)
        )
        
        # Stage 3: 128 -> 256 channels (Downsample 28 -> 14)
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2, use_se=True),
            ResidualBlock(256, 256, stride=1, use_se=True)
        )
        
        # Stage 4: 256 -> 384 channels (Downsample 14 -> 7)
        self.stage4 = nn.Sequential(
            ResidualBlock(256, 384, stride=2, use_se=True),
            ResidualBlock(384, 384, stride=1, use_se=True)
        )
        
        # Global Pooling and Regularized Classification Head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(384, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


# Alias for backward compatibility
PaddyGuardNet = PaddySnapNet


# ==============================================================
# 2. Transfer Learning Model Builders
# ==============================================================

def build_resnet50(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


def build_densenet121(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.densenet121(weights=weights)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


def build_efficientnet_v2_s(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_v2_s(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


def build_convnext_tiny(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.convnext_tiny(weights=weights)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model


# ==============================================================
# 3. Model Factory & Utilities
# ==============================================================

SUPPORTED_MODELS = {
    "resnet50": build_resnet50,
    "densenet121": build_densenet121,
    "efficientnet_v2_s": build_efficientnet_v2_s,
    "convnext_tiny": build_convnext_tiny,
    "custom_cnn": lambda num_classes, pretrained=False: PaddySnapNet(num_classes=num_classes)
}


def get_model(model_name: str, num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """Factory function to build any of the 5 supported models."""
    name_clean = model_name.lower().replace("-", "_")
    if name_clean not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model: '{model_name}'. Supported options are: {list(SUPPORTED_MODELS.keys())}"
        )
    builder = SUPPORTED_MODELS[name_clean]
    if name_clean == "custom_cnn":
        return builder(num_classes=num_classes, pretrained=False)
    return builder(num_classes=num_classes, pretrained=pretrained)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Returns (total_parameters, trainable_parameters)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# --------------------------------------------------------------
# Self-Test Verification
# --------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("  PADDY GUARD AI -- MODEL ARCHITECTURE TEST")
    print("=" * 70)

    dummy_input = torch.randn(2, 3, 224, 224)

    for name in SUPPORTED_MODELS.keys():
        print(f"\n  Instantiating [{name}]...")
        model = get_model(name, num_classes=10, pretrained=False)
        total_p, train_p = count_parameters(model)
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)

        size_mb = (total_p * 4) / (1024 ** 2)
        print(f"    - Parameters  : {total_p:,} ({train_p:,} trainable)")
        print(f"    - Model Size  : ~{size_mb:.2f} MB")
        print(f"    - Output Shape: {tuple(output.shape)} (Expected: (2, 10))")
        assert output.shape == (2, 10), f"Output shape mismatch: {output.shape}"

    print("\n" + "=" * 70)
    print("  ALL 5 MODELS VERIFIED AND READY FOR TRAINING!")
    print("=" * 70)
