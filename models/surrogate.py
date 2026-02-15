# ============================================================================
# Latent Stiffness Regressor — Differentiable Physics Surrogate
# ============================================================================
"""
A convolutional neural network that maps VAE latent codes directly
to effective mechanical stiffness, enabling gradient-based inverse
design via Score Distillation Sampling (SDS).

Architecture:
    Input  → (B, 16, 32, 32)  — VAE latent representation
    Self-contained ResNet-18 backbone (16 input channels, no torchvision
    dependency) → adaptive avg pool → (B, 512)
    MLP head → scalar E_eff

Design rationale:
    The surrogate must be *differentiable w.r.t. its input* so that
    ∂L/∂z can be computed and injected into the diffusion sampling loop.
    ResNet-18 is lightweight enough for real-time gradient queries
    yet expressive enough to achieve R² > 0.9 on the latent→stiffness
    mapping.

    The backbone is implemented from scratch (no torchvision dependency)
    to avoid version-coupling issues with the broader PyTorch ecosystem.
"""

from __future__ import annotations

from typing import Type

import torch
import torch.nn as nn


# ── ResNet building blocks ──────────────────────────────────────────────────

class BasicBlock(nn.Module):
    """Standard ResNet basic block (two 3×3 convolutions + skip)."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet18Backbone(nn.Module):
    """Minimal ResNet-18 backbone (no torchvision dependency).

    Layer configuration: [2, 2, 2, 2] BasicBlocks with channels
    [64, 128, 256, 512].
    """

    def __init__(self, input_channels: int = 16) -> None:
        super().__init__()
        self.in_channels = 64

        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Kaiming initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = [BasicBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        return x.view(x.size(0), -1)  # (B, 512)


# ── Regressor ───────────────────────────────────────────────────────────────

class LatentStiffnessRegressor(nn.Module):
    """Physics surrogate predicting stiffness from VAE latent codes.

    Parameters
    ----------
    input_channels : int
        Number of channels in the VAE latent (default: 16).
    output_dim : int
        Dimensionality of the prediction vector (default: 1 for scalar
        stiffness).
    dropout : float
        Dropout probability in the MLP head.
    """

    def __init__(
        self,
        input_channels: int = 16,
        output_dim: int = 1,
        pretrained_backbone: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.backbone = ResNet18Backbone(input_channels=input_channels)

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass: latent → stiffness prediction.

        Parameters
        ----------
        z : torch.Tensor
            Latent code of shape ``(B, 16, 32, 32)``.

        Returns
        -------
        torch.Tensor
            Predicted stiffness of shape ``(B, output_dim)``.
        """
        features = self.backbone(z)  # (B, 512)
        return self.head(features)

    def predict(self, z: torch.Tensor) -> float:
        """Convenience method returning a Python scalar for a single sample.

        Parameters
        ----------
        z : torch.Tensor
            Single latent of shape ``(1, 16, 32, 32)``.

        Returns
        -------
        float
            Predicted stiffness value.
        """
        self.eval()
        with torch.no_grad():
            return self.forward(z).item()
