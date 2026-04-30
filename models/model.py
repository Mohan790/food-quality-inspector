"""
model.py — MobileNetV2 Fine-Tuned for Food Quality Classification

Why MobileNetV2?
- Lightweight and fast → trains quickly even on CPU
- Designed for real-world mobile applications → perfect for a food safety app
- Great accuracy with fewer parameters than ResNet or EfficientNet
- Easy to explain every component in the oral defense
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights


class FoodQualityClassifier(nn.Module):
    """
    Fine-tuned MobileNetV2 for binary classification:
        0 = Fresh
        1 = Rotten

    Args:
        num_classes (int): 2 for Fresh/Rotten
        dropout_rate (float): Dropout before classifier head
    """

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.3):
        super(FoodQualityClassifier, self).__init__()

        # ── Load pre-trained backbone ──────────────────────────────────────────
        backbone = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

        # ── Freeze all layers first ────────────────────────────────────────────
        for param in backbone.parameters():
            param.requires_grad = False

        # ── Unfreeze last 3 layers of the feature extractor ───────────────────
        # These layers learn high-level features specific to food quality
        # Earlier layers detect edges/textures → already good from ImageNet
        for param in backbone.features[-3:].parameters():
            param.requires_grad = True

        self.features = backbone.features  # Convolutional backbone

        # ── Custom classification head ─────────────────────────────────────────
        # Replace original 1000-class head with our 2-class head
        # Dropout → Linear(1280→256) → ReLU → Linear(256→2)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(256, num_classes),
        )

        # Global average pooling → converts (B, 1280, 7, 7) to (B, 1280)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x: (B, 3, 224, 224) — batch of normalized RGB images
        returns: (B, 2) — raw logits for [Fresh, Rotten]
        """
        x = self.features(x)       # → (B, 1280, 7, 7)
        x = self.avgpool(x)        # → (B, 1280, 1, 1)
        x = torch.flatten(x, 1)   # → (B, 1280)
        x = self.classifier(x)    # → (B, 2)
        return x


def build_model(num_classes: int = 2) -> FoodQualityClassifier:
    """Factory function — returns model ready for training."""
    return FoodQualityClassifier(num_classes=num_classes)


def count_trainable_params(model: nn.Module) -> int:
    """Returns number of parameters that will be updated during training."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)