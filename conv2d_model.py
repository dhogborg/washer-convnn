"""Shared model definitions and constants for spectrogram classification."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import torch
from torch import nn
from torchvision import transforms

CLASS_NAMES = ["WASHER", "SPIN", "IDLE", "END", "FAN", "DRYER"]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
DEFAULT_IMAGE_SIZE = (171, 286)
CHECKPOINT_EXT = ".pt"


def build_base_transform(img_size: Tuple[int, int] | Sequence[int]) -> transforms.Compose:
    """Return the normalization pipeline shared by training and inference."""
    height, width = int(img_size[0]), int(img_size[1])
    return transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


class ResidualBlock(nn.Module):
    """Basic ResNet-style block with identity skip."""

    def __init__(self, in_channels: int, out_channels: int, stride: int | Tuple[int, int] = 1) -> None:
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.relu = nn.ReLU(inplace=True)
        
        needs_proj = stride != 1 or in_channels != out_channels
        if isinstance(stride, tuple):
            needs_proj = stride != (1, 1) or in_channels != out_channels

        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if needs_proj
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.layer(x)
        
        # Does the in and out dims differ?
        # if so we need to resample before we apply the residual.
        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        return self.relu(out)


class ConvClassifier(nn.Module):
    """Compact ResNet-like CNN for grayscale spectrogram classification."""

    def __init__(self, num_classes: int | None = None) -> None:
        super().__init__()
        out_classes = num_classes or len(CLASS_NAMES)

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(7, 3), padding=(3, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.layer1 = self._make_layer(32, 32, stride=1)
        self.layer2 = self._make_layer(32, 64, stride=(2, 2))
        self.layer3 = self._make_layer(64, 128, stride=(2, 2))
        self.layer4 = self._make_layer(128, 256, stride=(2, 2))

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(256, out_classes),
        )

    def _make_layer(
        self, in_channels: int, out_channels: int,  stride: int | Tuple[int, int]
    ) -> nn.Sequential:
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride=stride),
            ResidualBlock(out_channels, out_channels, stride=1),
            ResidualBlock(out_channels, out_channels, stride=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.classifier(x)


def load_weights(model_path: Path, device: torch.device) -> tuple[ConvClassifier, Sequence[str]]:
    """Create a ConvClassifier and load weights from disk."""
    weights = torch.load(model_path, map_location=device)
    classes = weights.get("classes", CLASS_NAMES)
    model = ConvClassifier(num_classes=len(classes))
    model.load_state_dict(weights["model_state_dict"])
    model.to(device)
    return model, classes
