# Synthetic 3D shape dataset for the SurfaceGraph-ShapeEmbed proof of concept.
#
# Four simple convex primitives:  cube  |  sphere  |  pyramid  |  cylinder
# (each with randomised aspect ratios and a random pose).  They are deliberately
# distinct so the latent space, the reconstructions and the figures are easy to
# read; being convex, every shape -- ground truth or reconstruction -- renders
# cleanly as the surface of its convex hull.
#
# For each sample we:
#   1. sample a dense, approximately area-uniform point cloud on the surface,
#   2. farthest-point-subsample to N points (uniform spatial coverage),
#   3. build the N x N *Euclidean* distance matrix D^E,
#   4. build an N x N *geodesic* distance matrix D^G by Dijkstra on a kNN graph
#      over the dense cloud (an Isomap-style geodesic estimate -- no mesh needed),
#   5. Frobenius-normalise each matrix (scale invariance, as in the paper),
#   6. also keep the FPS point cloud (centred + scaled so cdist(P) == D^E) so the
#      evaluation can render and compare reconstructions against ground truth.
#
# Output: data/processed/shapes3d/{train,test}.npz with keys D_e, D_g, y, P, norm_e.
#
#   python phase3/generate_dataset.py                 # defaults
#   python phase3/generate_dataset.py --per-class 200 --seed 0

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from common import CLASSES, DATA_DIR, N_POINTS, farthest_point_sample, frobenius_normalize, set_seed

DENSE = 3000   # dense surface samples per shape, before farthest-point subsampling


# ---------------------------------------------------------------------
# Area-uniform surface samplers -- each returns a dense (~DENSE, 3) point cloud
# ---------------------------------------------------------------------

def _alloc(weights, total):
    """Split `total` points across pieces proportionally to `weights` (>= 1 each)."""
    w = np.asarray(weights, float); w = w / w.sum()
    n = np.maximum(1, np.round(w * total).astype(int))
    return n


def _sample_triangle(rng, A, B, C, n):
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    return (1 - s)[:, None] * A + (s * (1 - r2))[:, None] * B + (s * r2)[:, None] * C


def _sample_quad(rng, A, B, C, D, n):
    # split the quad A-B-C-D into two triangles, area-weighted
    a1 = 0.5 * np.linalg.norm(np.cross(B - A, C - A))
    a2 = 0.5 * np.linalg.norm(np.cross(C - A, D - A))
    n1 = int(round(n * a1 / (a1 + a2)))
    return np.vstack([_sample_triangle(rng, A, B, C, n1),
                      _sample_triangle(rng, A, C, D, n - n1)])


def _sample_disk(rng, centre, radius, axis, n):
    # uniform on a disk of given radius, in the plane orthogonal to `axis`
    u = axis / np.linalg.norm(axis)
    e1 = np.cross(u, [1.0, 0.0, 0.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(u, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1); e2 = np.cross(u, e1)
    rr = radius * np.sqrt(rng.random(n)); th = 2 * np.pi * rng.random(n)
    return centre + rr[:, None] * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)


def _jit(base, rng, amp=0.06):
    return np.asarray(base, float) * np.exp(rng.uniform(-amp, amp, size=np.shape(base)))


def sample_cube(rng):
    a, b, c = _jit([1.0, 0.78, 0.62], rng)                # a box: distinct extents -> stable principal axes
    V = np.array([[sx * a, sy * b, sz * c] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    # six faces as quads (vertex order matters only for the triangle split)
    faces = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
    areas = [4 * b * c, 4 * b * c, 4 * a * c, 4 * a * c, 4 * a * b, 4 * a * b]
    nper = _alloc(areas, DENSE)
    return np.vstack([_sample_quad(rng, V[i], V[j], V[k], V[l], int(m))
                      for (i, j, k, l), m in zip(faces, nper)])


def sample_sphere(rng):
    r = float(_jit(1.0, rng))
    x = rng.normal(size=(DENSE, 3))
    return r * x / np.linalg.norm(x, axis=1, keepdims=True)


def sample_pyramid(rng):
    s = float(_jit(0.7, rng))                              # half side of the square base
    h = float(_jit(2.2, rng))                              # apex height (tall -> clear major axis)
    base = np.array([[-s, -s, 0.0], [s, -s, 0.0], [s, s, 0.0], [-s, s, 0.0]])
    apex = np.array([0.0, 0.0, h])
    side_area = 4 * (0.5 * (2 * s) * np.sqrt(h * h + s * s))
    nb, ns = _alloc([4 * s * s, side_area], DENSE)
    pts = [_sample_quad(rng, *base, int(nb))]
    for k in range(4):
        pts.append(_sample_triangle(rng, base[k], base[(k + 1) % 4], apex, int(round(ns / 4))))
    return np.vstack(pts) - np.array([0.0, 0.0, h / 3.0])  # roughly centre it


def sample_cylinder(rng):
    r = float(_jit(0.4, rng))
    h = float(_jit(1.8, rng))                              # slender rod -> clear major axis
    lateral, caps = 2 * np.pi * r * h, 2 * np.pi * r * r
    n_lat, n_cap = _alloc([lateral, caps], DENSE)
    th = 2 * np.pi * rng.random(int(n_lat)); z = rng.uniform(-h / 2, h / 2, int(n_lat))
    lat = np.stack([r * np.cos(th), r * np.sin(th), z], axis=1)
    top = _sample_disk(rng, np.array([0.0, 0.0, h / 2]), r, np.array([0.0, 0.0, 1.0]), int(n_cap) // 2)
    bot = _sample_disk(rng, np.array([0.0, 0.0, -h / 2]), r, np.array([0.0, 0.0, 1.0]), int(n_cap) - int(n_cap) // 2)
    return np.vstack([lat, top, bot])


SAMPLERS = {"cube": sample_cube, "sphere": sample_sphere, "pyramid": sample_pyramid, "cylinder": sample_cylinder}


# ---------------------------------------------------------------------
# Geodesic estimate via Dijkstra on a symmetric kNN graph
# ---------------------------------------------------------------------

def geodesic_matrix(dense: np.ndarray, src_idx: np.ndarray, k: int = 12) -> np.ndarray:
    """Approximate geodesic distances between the `src_idx` points of `dense`
    by Dijkstra on a symmetric kNN graph (Isomap-style estimate; no mesh needed)."""
    m = dense.shape[0]
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, m)).fit(dense)
    w, knn = nbrs.kneighbors(dense)
    rows = np.repeat(np.arange(m), knn.shape[1] - 1)
    cols = knn[:, 1:].ravel()
    G = csr_matrix((w[:, 1:].ravel(), (rows, cols)), shape=(m, m))
    G = G.maximum(G.T)
    dist = dijkstra(G, directed=False, indices=src_idx)
    out = dist[:, src_idx]
    if not np.all(np.isfinite(out)):
        eu = cdist(dense[src_idx], dense[src_idx])
        out = np.where(np.isfinite(out), out, eu)
    return 0.5 * (out + out.T)


# ---------------------------------------------------------------------
# Build one sample
# ---------------------------------------------------------------------

def _random_rotation(rng):
    Q, Rm = np.linalg.qr(rng.normal(size=(3, 3)))
    Q = Q @ np.diag(np.sign(np.diag(Rm)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def _canonical_order(pts: np.ndarray) -> np.ndarray:
    """An ordering of `pts` that is a deterministic function of the *shape* (not the
    pose): orient by principal axes (sign convention from the third moment), then
    sort by azimuth around the major axis, then by height.  Storing points in this
    order makes the distance matrix a consistent target, so the decoder can learn an
    ordered reconstruction with a plain MSE loss."""
    pc = pts - pts.mean(0)
    _, _, Vt = np.linalg.svd(pc, full_matrices=False)
    q = pc @ Vt.T
    for k in range(3):
        if np.sum(q[:, k] ** 3) < 0:
            q[:, k] = -q[:, k]
    return np.lexsort((q[:, 2], np.arctan2(q[:, 1], q[:, 0])))


def make_sample(cls_name: str, rng: np.random.Generator, n: int):
    dense = SAMPLERS[cls_name](rng)
    dense = dense + rng.normal(0.0, 1e-3, size=dense.shape)          # tiny jitter
    dense = (dense - dense.mean(0)) @ _random_rotation(rng).T        # arbitrary pose
    idx = farthest_point_sample(dense, n)
    idx = idx[_canonical_order(dense[idx])]                          # canonical, pose-independent point ordering
    pts = dense[idx]
    D_e, norm_e = frobenius_normalize(cdist(pts, pts))
    D_g, _ = frobenius_normalize(geodesic_matrix(dense, idx))
    P = pts / (norm_e if norm_e > 1e-12 else 1.0)                   # so cdist(P) == D_e
    return D_e.astype(np.float32), D_g.astype(np.float32), P.astype(np.float32), np.float32(norm_e)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-class", type=int, default=200, help="samples generated per shape family")
    ap.add_argument("--n-points", type=int, default=N_POINTS, help="points sampled per surface")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    De_all, Dg_all, P_all, norm_all, y_all = [], [], [], [], []
    t0 = time.time()
    for label, name in enumerate(CLASSES):
        for _ in range(args.per_class):
            De, Dg, P, ne = make_sample(name, rng, args.n_points)
            De_all.append(De); Dg_all.append(Dg); P_all.append(P); norm_all.append(ne); y_all.append(label)
        print(f"  {name:<10s} {args.per_class} samples  ({time.time() - t0:5.1f}s elapsed)")

    De_all = np.stack(De_all); Dg_all = np.stack(Dg_all); P_all = np.stack(P_all)
    norm_all = np.asarray(norm_all, np.float32); y_all = np.asarray(y_all, np.int64)
    idx_tr, idx_te = train_test_split(np.arange(len(y_all)), test_size=args.test_frac,
                                      stratify=y_all, random_state=args.seed)
    for split, idx in (("train", idx_tr), ("test", idx_te)):
        np.savez_compressed(DATA_DIR / f"{split}.npz",
                            D_e=De_all[idx], D_g=Dg_all[idx], y=y_all[idx], P=P_all[idx], norm_e=norm_all[idx])
        print(f"saved {split}: {len(idx)} samples -> {DATA_DIR / f'{split}.npz'}")
    print(f"classes: {CLASSES}  |  N points: {args.n_points}  |  total time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
