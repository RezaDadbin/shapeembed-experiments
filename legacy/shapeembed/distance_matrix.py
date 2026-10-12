from __future__ import annotations

import numpy as np


def pairwise_distance_matrix(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected points of shape [N, 2], got {points.shape}")

    diff = points[:, None, :] - points[None, :, :]
    D = np.sqrt(np.sum(diff * diff, axis=-1))

    return D.astype(np.float32)


def normalize_distance_matrix(D: np.ndarray) -> tuple[np.ndarray, float]:
    D = np.asarray(D, dtype=np.float32)

    norm = float(np.linalg.norm(D, ord="fro"))

    if norm <= 1e-12:
        raise ValueError("Distance matrix has zero Frobenius norm.")

    D_norm = D / norm
    return D_norm.astype(np.float32), norm


def compute_normalized_distance_matrix(points: np.ndarray) -> tuple[np.ndarray, float]:
    D = pairwise_distance_matrix(points)
    return normalize_distance_matrix(D)
