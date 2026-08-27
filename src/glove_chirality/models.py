from __future__ import annotations


def build_model(name: str, num_classes: int = 2, pretrained: bool = True):
    """Build an interchangeable PyTorch classifier (imports stay optional)."""
    try:
        from torch import nn
        from torchvision import models
    except ImportError as exc:
        raise RuntimeError("Training requires: pip install -e .[ml]") from exc

    if name == "tiny_cnn":
        return nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2), nn.BatchNorm2d(24), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(96, num_classes),
        )
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if name == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model
    raise ValueError(f"Unknown classifier: {name}")
