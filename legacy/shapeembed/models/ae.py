from __future__ import annotations

import torch
from torch import nn

from shapeembed.models.decoder import ShapeEmbedDecoder
from shapeembed.models.encoder import ShapeEmbedDeterministicEncoder


class ShapeEmbedAE(nn.Module):
    """
    Phase 2 standard Autoencoder.

    Difference from VAE:
        no mu/logvar
        no reparameterization
        no KL loss

    Encoder:
        same paper-style circular ResNet-18 feature extractor

    Decoder:
        same mirrored convolutional decoder
    """

    def __init__(
        self,
        matrix_size: int,
        latent_dim: int = 128,
        base_channels: int = 64,
        decoder_channels: int = 512,
    ):
        super().__init__()

        self.matrix_size = matrix_size
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.decoder_channels = decoder_channels

        self.encoder = ShapeEmbedDeterministicEncoder(
            matrix_size=matrix_size,
            latent_dim=latent_dim,
            base_channels=base_channels,
        )

        self.decoder = ShapeEmbedDecoder(
            matrix_size=matrix_size,
            latent_dim=latent_dim,
            base_channels=base_channels,
            decoder_channels=decoder_channels,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)

        return x_hat, z
