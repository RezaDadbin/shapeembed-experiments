from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from shapeembed.datasets import DistanceMatrixDataset
from shapeembed.models.factory import (
    build_model,
    extract_latent,
    get_checkpoint_path,
    get_latest_checkpoint_path,
    get_model_type,
    get_run_name,
)
from shapeembed.utils import ensure_dir, get_device, prepare_distance_batch


def get_selected_checkpoint_path(
    config: dict[str, Any],
    checkpoint: str | None = None,
) -> Path:
    checkpoint = str(
        checkpoint or config.get("evaluation", {}).get("checkpoint", "best")
    ).lower()

    if checkpoint == "best":
        return get_checkpoint_path(config)

    if checkpoint == "latest":
        return get_latest_checkpoint_path(config)

    raise ValueError(f"Unknown checkpoint selection: {checkpoint}")


def load_trained_model(
    config: dict[str, Any],
    device: torch.device,
    checkpoint: str | None = None,
) -> torch.nn.Module:
    checkpoint_path = get_selected_checkpoint_path(config, checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_model(config).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print(f"[encode] loaded checkpoint: {checkpoint_path}")
    return model


@torch.no_grad()
def encode_dataset(
    config: dict[str, Any],
    checkpoint: str | None = None,
) -> None:
    dataset_name = config["dataset"]["name"]
    run_name = get_run_name(config)
    evaluation_cfg = config.get("evaluation", {})

    processed_dir = Path(config["dataset"]["processed_dir"])
    embedding_dir = ensure_dir(config["outputs"]["embedding_dir"])

    device = get_device(config["training"]["device"])
    print(f"[encode] dataset: {dataset_name}")
    print(f"[encode] run name: {run_name}")
    print(f"[encode] device: {device}")

    train_dataset = DistanceMatrixDataset(processed_dir / "train.pt")
    test_dataset = DistanceMatrixDataset(processed_dir / "test.pt")
    dataset = ConcatDataset([train_dataset, test_dataset])

    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"]["num_workers"])

    if device.type == "mps":
        num_workers = 0

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = load_trained_model(config, device, checkpoint=checkpoint)
    model_type = get_model_type(config)
    embedding_source = str(evaluation_cfg.get("embedding_source", "z")).lower()

    if model_type in ["shapeembed", "vae"] and embedding_source not in ["z", "z_mean"]:
        raise ValueError(
            "evaluation.embedding_source must be 'z' or 'z_mean' for Phase 1, "
            f"got {embedding_source!r}"
        )

    if model_type in ["shapeembed", "vae"]:
        print(f"[encode] phase1 embedding_source: {embedding_source}")

    all_z = []
    all_labels = []
    all_norms = []

    for batch in tqdm(loader, desc=f"[encode] {run_name}"):
        D = batch["D"].to(device)
        labels = batch["label"].cpu().numpy()

        if model_type in ["shapeembed", "vae"]:
            _, _, z_sample, z_mean, _, _ = model(D)

            if embedding_source == "z":
                z = z_sample
            elif embedding_source == "z_mean":
                z = z_mean
            else:
                raise ValueError(f"Unknown embedding_source: {embedding_source}")

            norms = batch["norm"].cpu().numpy()
        else:
            D = prepare_distance_batch(D, config)
            z = extract_latent(model, config, D)
            norms = batch["norm"].cpu().numpy()

        all_z.append(z.cpu().numpy())
        all_labels.append(labels)
        all_norms.append(norms)

    embeddings = np.concatenate(all_z, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    norms = np.concatenate(all_norms, axis=0)

    expected_latent_dim = int(config["model"]["latent_dim"])

    if embeddings.ndim != 2 or embeddings.shape[1] != expected_latent_dim:
        raise RuntimeError(
            f"Expected embeddings [num_samples,{expected_latent_dim}], "
            f"got {embeddings.shape}"
        )

    np.save(embedding_dir / f"{run_name}_embeddings.npy", embeddings)
    np.save(embedding_dir / f"{run_name}_labels.npy", labels)
    np.save(embedding_dir / f"{run_name}_norms.npy", norms)

    print(f"[encode] embeddings: {embeddings.shape}")
    print(f"[encode] saved to: {embedding_dir}")
