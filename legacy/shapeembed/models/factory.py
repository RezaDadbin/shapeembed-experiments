from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from shapeembed.models.ae import ShapeEmbedAE
from shapeembed.models.vae import ShapeEmbedVAE
from shapeembed.models.vae_mlp import ShapeEmbedVAEMLP


def get_model_type(config: dict[str, Any]) -> str:
    """
    Default model is original ShapeEmbed VAE.
    """
    return config["model"].get("type", "shapeembed")


def get_run_name(config: dict[str, Any]) -> str:
    dataset_name = config["dataset"]["name"]
    model_type = get_model_type(config)

    return f"{dataset_name}_{model_type}"


def get_checkpoint_path(config: dict[str, Any]) -> Path:
    checkpoint_dir = Path(config["outputs"]["checkpoint_dir"])
    return checkpoint_dir / f"{get_run_name(config)}.pt"


def get_latest_checkpoint_path(config: dict[str, Any]) -> Path:
    checkpoint_dir = Path(config["outputs"]["checkpoint_dir"])
    return checkpoint_dir / f"{get_run_name(config)}_latest.pt"


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    model_cfg = config["model"]
    dataset_cfg = config["dataset"]

    model_type = get_model_type(config)

    matrix_size = int(dataset_cfg["distance_matrix_size"])
    latent_dim = int(model_cfg["latent_dim"])
    base_channels = int(model_cfg["base_channels"])

    if model_type in ["shapeembed", "vae"]:
        return ShapeEmbedVAE(
            matrix_size=matrix_size,
            latent_dim=latent_dim,
            base_channels=base_channels,
            decoder_channels=int(model_cfg.get("decoder_channels", 512)),
            circular_padding=bool(model_cfg.get("circular_padding", True)),
            normalize_input=model_cfg.get("normalize_input", "fro"),
            rescale_input=model_cfg.get("rescale_input", None),
            augment_input=bool(model_cfg.get("augment_input", False)),
        )

    if model_type == "ae":
        return ShapeEmbedAE(
            matrix_size=matrix_size,
            latent_dim=latent_dim,
            base_channels=base_channels,
            decoder_channels=int(model_cfg.get("decoder_channels", 512)),
        )

    if model_type == "vae_mlp":
        return ShapeEmbedVAEMLP(
            matrix_size=matrix_size,
            latent_dim=latent_dim,
            base_channels=base_channels,
            mlp_hidden_dim=int(model_cfg.get("mlp_hidden_dim", 1024)),
            mlp_num_layers=int(model_cfg.get("mlp_num_layers", 3)),
        )

    raise ValueError(f"Unknown model type: {model_type}")


@torch.no_grad()
def extract_latent(
    model: torch.nn.Module,
    config: dict[str, Any],
    D: torch.Tensor,
) -> torch.Tensor:
    """
    Unified latent extraction.

    ShapeEmbed VAE:
        return mu

    VAE-MLP:
        return mu

    AE:
        return z
    """
    model_type = get_model_type(config)

    if model_type in ["shapeembed", "vae"]:
        _, _, _, z_mean, _, _ = model(D)
        return z_mean

    if model_type == "ae":
        return model.encode(D)

    mu, _ = model.encode(D)
    return mu
