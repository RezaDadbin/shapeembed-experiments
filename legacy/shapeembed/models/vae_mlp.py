from __future__ import annotations

import torch
from torch import nn

from shapeembed.models.encoder import ShapeEmbedEncoder
from shapeembed.models.mlp_decoder import ShapeEmbedMLPDecoder


class ShapeEmbedVAEMLP(nn.Module):
    """
    Phase 2 VAE with MLP decoder.

    Same paper-style VAE encoder as original ShapeEmbed.
    Decoder is replaced by an MLP.
    """

    def __init__(
        self,
        matrix_size: int,
        latent_dim: int = 128,
        base_channels: int = 64,
        mlp_hidden_dim: int = 1024,
        mlp_num_layers: int = 3,
    ):
        super().__init__()

        self.matrix_size = matrix_size
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_num_layers = mlp_num_layers

        self.encoder = ShapeEmbedEncoder(
            matrix_size=matrix_size,
            latent_dim=latent_dim,
            base_channels=base_channels,
        )

        self.decoder = ShapeEmbedMLPDecoder(
            matrix_size=matrix_size,
            latent_dim=latent_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu

        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)

        return x_hat, mu, logvar, z
