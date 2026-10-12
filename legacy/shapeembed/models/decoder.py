from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class ShapeEmbedDecoder(nn.Module):
    """
    Official-style ShapeEmbed decoder.

    The latent vector is mapped to a 512-channel 1x1 seed, then repeatedly
    upsampled with ConvTranspose2d layers until it reaches the matrix size.
    """

    def __init__(
        self,
        matrix_size: int,
        latent_dim: int = 128,
        base_channels: int = 64,
        decoder_channels: int = 512,
    ):
        super().__init__()

        if matrix_size not in [32, 64, 128, 256, 512]:
            raise ValueError(
                f"matrix_size should be a power-of-two >= 32, got {matrix_size}"
            )

        self.matrix_size = matrix_size
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.decoder_channels = decoder_channels

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, self.decoder_channels),
            nn.ReLU(inplace=True),
        )

        n_steps = int(math.ceil(math.log2(matrix_size)))
        layers: list[nn.Module] = []
        in_channels = self.decoder_channels

        for step in range(n_steps, 0, -1):
            if step != 1:
                out_channels = max(1, math.ceil(in_channels / 2))
                layers.extend(
                    [
                        nn.ConvTranspose2d(
                            in_channels,
                            out_channels,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                        ),
                        nn.ReLU(inplace=True),
                    ]
                )
                in_channels = out_channels
            else:
                layers.append(
                    nn.ConvTranspose2d(
                        in_channels,
                        1,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    )
                )

        self.decoder_conv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(z.shape[0], self.decoder_channels, 1, 1)
        x = self.decoder_conv(x)

        if x.shape[-2:] != (self.matrix_size, self.matrix_size):
            x = F.interpolate(
                x,
                size=(self.matrix_size, self.matrix_size),
                mode="nearest",
            )

        return x
