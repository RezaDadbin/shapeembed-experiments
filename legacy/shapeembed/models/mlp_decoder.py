from __future__ import annotations

import torch
from torch import nn


class ShapeEmbedMLPDecoder(nn.Module):
    """
    Phase 2 MLP decoder.

    It replaces the convolutional decoder:
        z -> dense layers -> N x N distance matrix
    """

    def __init__(
        self,
        matrix_size: int,
        latent_dim: int = 128,
        hidden_dim: int = 1024,
        num_layers: int = 3,
    ):
        super().__init__()

        if num_layers < 2:
            raise ValueError("num_layers must be at least 2.")

        self.matrix_size = matrix_size
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        layers = []
        in_dim = latent_dim

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, matrix_size * matrix_size))

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.net(z)
        out = out.view(z.shape[0], 1, self.matrix_size, self.matrix_size)
        return out
