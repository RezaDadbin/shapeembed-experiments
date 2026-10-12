from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")

    if device_name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")

    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    return torch.device("cpu")


def get_distance_matrix_scale(config: dict[str, Any]) -> float:
    dataset_cfg = config.get("dataset", {})

    if not bool(dataset_cfg.get("scale_distance_matrix_by_size", True)):
        return 1.0

    return float(dataset_cfg["distance_matrix_size"])


def prepare_distance_batch(D: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    if D.ndim != 4:
        raise ValueError(f"Expected distance batch [B,1,N,N], got {tuple(D.shape)}")

    matrix_size = int(config["dataset"]["distance_matrix_size"])

    if D.shape[-2:] != (matrix_size, matrix_size):
        raise ValueError(
            f"Expected distance matrix size {(matrix_size, matrix_size)}, "
            f"got {tuple(D.shape[-2:])}"
        )

    if not bool(config.get("dataset", {}).get("scale_distance_matrix_by_size", True)):
        return D

    norm = torch.linalg.norm(D, ord="fro", dim=(-2, -1), keepdim=True)
    norm = torch.clamp(norm, min=1e-12)

    return (D / norm) * D.new_tensor(float(matrix_size))


def save_json(obj: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r") as f:
        return json.load(f)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_module_parameters(module: torch.nn.Module | None) -> int | None:
    if module is None:
        return None

    return sum(p.numel() for p in module.parameters() if p.requires_grad)
