from __future__ import annotations

import torch

from shapeembed.models.circular_resnet import MyNet


class ShapeEmbedVAE(MyNet):
    """
    Official ShapeEmbed Phase 1 model.

    This is a thin project-compatible wrapper around the official repo's
    `MyNet` behavior. Its forward return order is:
        preproc_x, recon_x, z, z_mean, z_log_var, og_scale
    """

    def __init__(
        self,
        matrix_size: int,
        latent_dim: int = 128,
        base_channels: int = 64,
        decoder_channels: int = 512,
        circular_padding: bool = True,
        normalize_input: str | None = "fro",
        rescale_input: float | None = None,
        augment_input: bool = False,
    ):
        if base_channels != 64:
            raise ValueError("Official ShapeEmbed Phase 1 uses base_channels=64.")

        super().__init__(
            latent_dim=latent_dim,
            matrix_size=matrix_size,
            padding=circular_padding,
            decoder_channels=decoder_channels,
            normalize_input=normalize_input,
            rescale_input=rescale_input,
            augment_input=augment_input,
        )

        self.base_channels = base_channels
        self.decoder_channels = decoder_channels
        self.circular_padding = circular_padding
        self.normalize_input = normalize_input
        self.rescale_input = rescale_input
        self.augment_input = augment_input

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_preproc_og = self.preproc(x)
        x_preproc_flipped = torch.flip(x_preproc_og, dims=(2, 3))
        x_encoded_og = self.encoder(x_preproc_og)
        x_encoded_flipped = self.encoder(x_preproc_flipped)
        x_encoded = torch.add(x_encoded_og, x_encoded_flipped)
        z_mean = self.z_mean(x_encoded)
        z_log_var = self.z_log_var(x_encoded)
        return z_mean, z_log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
