from __future__ import annotations

import torch
from torch import nn

from shapeembed.models.circular_resnet import CircularResNet18Backbone


class ShapeEmbedEncoder(nn.Module):
    """
    Paper-style ShapeEmbed VAE encoder.

    The same circular ResNet-18-style backbone processes:
        1. original distance matrix D
        2. flipped distance matrix flip(D)

    Their feature vectors are summed, then mapped to mu/logvar.
    """

    def __init__(
        self,
        matrix_size: int,
        latent_dim: int = 128,
        base_channels: int = 64,
    ):
        super().__init__()

        if matrix_size not in [32, 64, 128, 256, 512]:
            raise ValueError(
                f"matrix_size should be a power-of-two >= 32, got {matrix_size}"
            )

        self.matrix_size = matrix_size
        self.latent_dim = latent_dim
        self.base_channels = base_channels

        self.backbone = CircularResNet18Backbone(
            in_channels=1,
            base_channels=base_channels,
        )

        feature_dim = self.backbone.out_channels

        self.fc_mu = nn.Linear(feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(feature_dim, latent_dim)

    def encode_once(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h_original = self.encode_once(x)

        # Reverse both contour axes. This corresponds to mirrored traversal.
        x_flipped = torch.flip(x, dims=(-1, -2))
        h_flipped = self.encode_once(x_flipped)

        h = h_original + h_flipped

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar


class ShapeEmbedDeterministicEncoder(nn.Module):
    """
    Paper-style deterministic encoder for the AE variant.

    Same backbone and same original/flipped feature summation, but outputs z.
    """

    def __init__(
        self,
        matrix_size: int,
        latent_dim: int = 128,
        base_channels: int = 64,
    ):
        super().__init__()

        if matrix_size not in [32, 64, 128, 256, 512]:
            raise ValueError(
                f"matrix_size should be a power-of-two >= 32, got {matrix_size}"
            )

        self.matrix_size = matrix_size
        self.latent_dim = latent_dim
        self.base_channels = base_channels

        self.backbone = CircularResNet18Backbone(
            in_channels=1,
            base_channels=base_channels,
        )

        feature_dim = self.backbone.out_channels
        self.fc_z = nn.Linear(feature_dim, latent_dim)

    def encode_once(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_original = self.encode_once(x)

        x_flipped = torch.flip(x, dims=(-1, -2))
        h_flipped = self.encode_once(x_flipped)

        h = h_original + h_flipped
        z = self.fc_z(h)

        return z
