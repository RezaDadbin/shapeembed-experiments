from __future__ import annotations

import numpy as np
import torch

from shapeembed.models.factory import extract_latent, get_model_type
from shapeembed.utils import prepare_distance_batch


def clean_distance_matrix(D: np.ndarray) -> np.ndarray:
    """
    Visualization cleanup only; classification uses latent embeddings directly.
    Make decoder output usable as a distance matrix:
    - remove channel/batch dimensions if present
    - clamp negative values
    - symmetrize
    - zero diagonal
    """
    D = np.asarray(D, dtype=np.float64)

    while D.ndim > 2:
        D = D[0]

    D = np.maximum(D, 0.0)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)

    return D


def classical_mds(distance_matrix: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Project cleaned distances into a deterministic low-dimensional point set.

    Double-center squared distances, then retain the largest nonnegative Gram
    eigenvalues. A raw decoder output need not be Euclidean; this projection can
    discard information and does not guarantee a non-self-intersecting contour.
    """
    D = clean_distance_matrix(distance_matrix)
    n = D.shape[0]

    D2 = D ** 2
    J = np.eye(n) - np.full((n, n), 1.0 / n)
    B = -0.5 * J @ D2 @ J
    B = 0.5 * (B + B.T)

    eigvals, eigvecs = np.linalg.eigh(B)          # ascending
    order = np.argsort(eigvals)[::-1][:n_components]
    eigvals = np.clip(eigvals[order], 0.0, None)
    eigvecs = eigvecs[:, order]

    return (eigvecs * np.sqrt(eigvals)).astype(np.float32)


def fourier_smooth_closed(
    points: np.ndarray,
    n_harmonics: int | None,
    n_out: int | None = None,
) -> np.ndarray:
    """
    Smooth (and optionally resample) a closed contour with a truncated complex
    Fourier series (elliptic Fourier descriptors). Keeping only the lowest
    ``n_harmonics`` frequencies removes small-scale jitter; lower -> smoother.
    ``n_harmonics=None`` (or large) disables smoothing.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n_out is None:
        n_out = n
    if n < 3:
        return pts.astype(np.float32)

    z = pts[:, 0] + 1j * pts[:, 1]
    coeffs = np.fft.fft(z) / n
    freqs = np.fft.fftfreq(n, d=1.0 / n)          # integer frequencies

    if n_harmonics is None:
        keep = np.ones(n, dtype=bool)
    else:
        k = max(int(n_harmonics), 1)
        keep = np.abs(freqs) <= k

    t = np.linspace(0.0, 1.0, n_out, endpoint=False)
    basis = np.exp(2j * np.pi * np.outer(t, freqs[keep]))
    zr = basis @ coeffs[keep]
    return np.stack([zr.real, zr.imag], axis=1).astype(np.float32)


def recon_harmonics_for(model_type: str | None, matrix_size: int) -> None:
    """Default figures use an unsmoothed MDS projection for every variant."""
    return None


def distance_matrix_to_points(
    D: np.ndarray,
    random_state: int = 42,
    n_harmonics: int | None = None,
    n_out: int | None = None,
) -> np.ndarray:
    """
    Recover 2-D outline points from a distance matrix via classical MDS, then
    (optionally) Fourier-smooth the resulting closed contour.
    """
    coords = classical_mds(D, n_components=2)
    if n_harmonics is not None or n_out is not None:
        coords = fourier_smooth_closed(coords, n_harmonics=n_harmonics, n_out=n_out)
    return coords


@torch.no_grad()
def reconstruct_batch(model, config: dict, D: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generic reconstruction for both VAE and AE.

    VAE:
        latent = mu (decode the descriptor, not a sampled z)

    AE:
        latent = z
    """
    model.eval()

    if get_model_type(config) in ["shapeembed", "vae"]:
        preproc_x, _, _, z_mean, _, _ = model(D)
        D_hat = model.decode(z_mean)

        if D_hat.shape != preproc_x.shape:
            raise RuntimeError(
                f"Model output shape {tuple(D_hat.shape)} does not match "
                f"preprocessed target shape {tuple(preproc_x.shape)}"
            )

        return D_hat, z_mean

    D = prepare_distance_batch(D, config)
    latent = extract_latent(model, config, D)
    D_hat = model.decode(latent)

    if D_hat.shape != D.shape:
        raise RuntimeError(
            f"Model output shape {tuple(D_hat.shape)} does not match "
            f"input shape {tuple(D.shape)}"
        )

    return D_hat, latent
