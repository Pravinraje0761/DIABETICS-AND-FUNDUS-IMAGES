"""
Model factory for the baseline CNNs (Phase 4).

ImageNet-pretrained backbones from torchvision with the classifier head swapped
for a 5-way DR-grade head. Kept deliberately small - RETFound (Phase 5) is built
from its own repo, not here.

    build_model("resnet50")         ResNet-50  (IMAGENET1K_V2 weights)
    build_model("efficientnet_b3")  EfficientNet-B3 (IMAGENET1K_V1 weights)

`set_dropout_p` bumps the head dropout so the same checkpoint can be reused for
MC-dropout uncertainty in Phase 8.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models

ARCHS = ["resnet50", "efficientnet_b3"]
NUM_CLASSES = 5


def build_model(arch: str, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    arch = arch.lower()
    if arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = models.resnet50(weights=weights)
        in_features = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features, num_classes))
        return m
    if arch == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b3(weights=weights)
        in_features = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.3), nn.Linear(in_features, num_classes)
        )
        return m
    raise ValueError(f"unknown arch {arch!r}; choose from {ARCHS}")


def set_dropout_p(model: nn.Module, p: float) -> None:
    """Set p on every nn.Dropout in the model (used for MC-dropout in Phase 8)."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = p
