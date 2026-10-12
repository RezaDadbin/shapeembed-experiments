# Shared helpers for the Phase 3 (Section 1) proof of concept:
# "SurfaceGraph-ShapeEmbed" -- a permutation-invariant metric-graph extension of
# ShapeEmbed to 3D surfaces.
#
# This module is intentionally self-contained (numpy / scipy / sklearn / torch /
# matplotlib only -- no mesh or graph-NN libraries) so the PoC runs anywhere the
# main repo runs.

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "shapes3d"
OUT_DIR = ROOT / "outputs" / "shapes3d"
CKPT_DIR = OUT_DIR / "checkpoints"
FIG_DIR = OUT_DIR / "figures"
RES_DIR = OUT_DIR / "results"

for _d in (DATA_DIR, CKPT_DIR, FIG_DIR, RES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------

# Number of points sampled per surface (the "nodes" of the metric graph).
N_POINTS = 64
# Class names, in label order.  Four simple convex primitives -- distinct enough to read
# at a glance and to render cleanly as convex-hull surfaces.
CLASSES = ("cube", "sphere", "pyramid", "cylinder")


# ---------------------------------------------------------------------
# Device / reproducibility (mirrors the conventions of the main repo)
# ---------------------------------------------------------------------

def autodetect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


DEVICE = autodetect_device()


def resolve_device(name: str) -> str:
    if name == "auto":
        return autodetect_device()
    if name == "cpu":
        return name
    if name == "cuda" and torch.cuda.is_available():
        return name
    if name == "mps" and torch.backends.mps.is_available():
        return name
    raise ValueError(f"Device {name!r} is unavailable")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    mps = getattr(torch, "mps", None)
    if mps is not None and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------

def farthest_point_sample(points: np.ndarray, n: int, start: int | None = None) -> np.ndarray:
    """Greedy farthest-point sampling. Returns indices into `points` (shape (n,)).

    `start` fixes the first index; pass the point farthest from the centroid for a
    *canonical* (rotation-invariant) ordering, so the resulting distance matrix is a
    deterministic function of the shape -- which lets the decoder learn an ordered
    reconstruction instead of having to average over random orderings."""
    m = points.shape[0]
    idx = np.empty(n, dtype=np.int64)
    idx[0] = int(start) if start is not None else int(np.argmax(np.sum((points - points.mean(0)) ** 2, axis=1)))
    d2 = np.sum((points - points[idx[0]]) ** 2, axis=1)
    for i in range(1, n):
        idx[i] = int(np.argmax(d2))
        d2 = np.minimum(d2, np.sum((points - points[idx[i]]) ** 2, axis=1))
    return idx


def frobenius_normalize(D: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (D / ||D||_F, ||D||_F) -- the paper's scale-invariance trick."""
    nrm = float(np.linalg.norm(D))
    if nrm < 1e-12:
        return D.copy(), nrm
    return D / nrm, nrm


def classical_mds(D: np.ndarray, dim: int = 3) -> np.ndarray:
    """Classical (Torgerson) MDS: distance matrix -> point cloud in R^dim.

    Used to turn a decoded distance matrix back into a 3D shape.
    """
    n = D.shape[0]
    J = np.eye(n) - np.full((n, n), 1.0 / n)
    B = -0.5 * J @ (D ** 2) @ J
    B = 0.5 * (B + B.T)
    w, V = np.linalg.eigh(B)              # ascending
    w = w[::-1][:dim]
    V = V[:, ::-1][:, :dim]
    w = np.clip(w, 0.0, None)
    return V * np.sqrt(w)[None, :]


def convex_hull_faces(P: np.ndarray):
    """Triangular faces of the convex hull of P, as an (F, 3) index array.

    All shape families in this PoC are convex, so the convex hull is the natural
    way to turn a reconstructed point cloud back into a *surface* (the 3D analogue
    of joining 2D contour points into a closed curve).  Returns None if the hull
    is degenerate (e.g. nearly coplanar points early in training)."""
    try:
        from scipy.spatial import ConvexHull
        return ConvexHull(P, qhull_options="QJ").simplices
    except Exception:
        return None


def kabsch(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Rigidly align P onto Q (rotation + translation, shared point ordering).

    Used only for visualisation: the decoder reconstructs a shape only up to a
    rigid motion (it is trained on distances), so we register it to the ground
    truth before plotting."""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1.0, 1.0, d]) @ Vt
    return Pc @ R + Q.mean(0)


# ---------------------------------------------------------------------
# Torch dataset
# ---------------------------------------------------------------------

class Shapes3DTensors:
    """Thin holder for the on-disk arrays + a torch TensorDataset factory."""

    def __init__(self, split: str):
        assert split in ("train", "test")
        z = np.load(DATA_DIR / f"{split}.npz")
        self.D_e = z["D_e"].astype(np.float32)   # (M, N, N) Euclidean, ||.||_F == 1
        self.D_g = z["D_g"].astype(np.float32)   # (M, N, N) geodesic,  ||.||_F == 1
        self.y = z["y"].astype(np.int64)         # (M,)
        self.P = z["P"].astype(np.float32)       # (M, N, 3) FPS points (centred, EDM Frobenius-normalised)
        self.norm_e = z["norm_e"].astype(np.float32)   # (M,) pre-normalisation ||D_e||_F  (size cue)

    def __len__(self):
        return len(self.y)
