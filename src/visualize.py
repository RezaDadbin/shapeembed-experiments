from pathlib import Path
import json
import re
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# =============================================================================
# Global plotting style: simple paper/report style
# =============================================================================

def set_plot_style():
    plt.rcParams.update({
        "figure.dpi": 180,
        "savefig.dpi": 220,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# =============================================================================
# Distance-matrix -> 2D outline reconstruction (point-based, like the old code)
# =============================================================================

def clean_distance_matrix(D: np.ndarray) -> np.ndarray:
    """
    Visualization cleanup only; classification uses latent embeddings directly.

    Make a (possibly raw) decoder output usable as a distance matrix:
        - drop channel/batch dims if present
        - clamp negatives to zero
        - symmetrize
        - zero the diagonal
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

    eigvals, eigvecs = np.linalg.eigh(B)          # ascending order
    order = np.argsort(eigvals)[::-1][:n_components]
    eigvals = np.clip(eigvals[order], 0.0, None)
    eigvecs = eigvecs[:, order]

    coords = eigvecs * np.sqrt(eigvals)
    return coords.astype(np.float32)


def fourier_smooth_closed(
    points: np.ndarray,
    n_harmonics: int | None,
    n_out: int | None = None,
) -> np.ndarray:
    """
    Smooth (and optionally resample) a closed contour with a truncated complex
    Fourier series (elliptic Fourier descriptors).

    Keeping only the lowest ``n_harmonics`` frequencies removes the small-scale
    jitter. This optional helper is not used to smooth the default figures.
    ``n_harmonics=None``
    (or large) disables smoothing.
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
    basis = np.exp(2j * np.pi * np.outer(t, freqs[keep]))     # [n_out, n_keep]
    zr = basis @ coeffs[keep]

    return np.stack([zr.real, zr.imag], axis=1).astype(np.float32)


def _harmonics_for(model_type: str | None, matrix_size: int) -> None:
    """Use the same unsmoothed MDS projection for every model."""
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

    Output:
        points: [n_out (or N), 2] float32
    """
    coords = classical_mds(D, n_components=2)
    if n_harmonics is not None or n_out is not None:
        coords = fourier_smooth_closed(coords, n_harmonics=n_harmonics, n_out=n_out)
    return coords


# Kept for backward compatibility with callers that imported the old name.
def paper_style_mds(distance_matrix: np.ndarray, random_state: int = 42) -> np.ndarray:
    return distance_matrix_to_points(distance_matrix, random_state=random_state)


# =============================================================================
# Display-only alignment helpers (ported from the old visualization)
# =============================================================================

def align_points_for_display(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Align ``source`` points onto ``target`` points for visualization only.

    MDS reconstruction from a distance matrix is only defined up to translation,
    scale, rotation, and reflection. ShapeEmbed is intentionally invariant to
    rotation/reflection/indexing, so a mirrored or rotated MDS plot is not a
    training failure. This helper only makes figures easier to compare; it does
    not touch training, evaluation, embeddings, or any saved outputs.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    if source.shape != target.shape:
        raise ValueError(
            f"source and target must have matching shape [N,2], got "
            f"{source.shape} and {target.shape}"
        )

    if source.ndim != 2 or source.shape[1] != 2:
        raise ValueError(f"Expected source/target shape [N,2], got {source.shape}")

    source_centroid = source.mean(axis=0, keepdims=True)
    target_centroid = target.mean(axis=0, keepdims=True)
    source_centered = source - source_centroid
    target_centered = target - target_centroid

    source_norm = float(np.linalg.norm(source_centered))
    target_norm = float(np.linalg.norm(target_centered))
    eps = 1e-12

    if source_norm <= eps or target_norm <= eps:
        return (source_centered + target_centroid).astype(np.float32)

    target_unit = target_centered / target_norm
    best_aligned_unit = None
    best_error = float("inf")

    for reflect_x in (False, True):
        candidate = source_centered.copy()

        if reflect_x:
            candidate[:, 0] *= -1.0

        candidate_unit = candidate / source_norm
        covariance = candidate_unit.T @ target_unit
        u, _, vt = np.linalg.svd(covariance)
        rotation = u @ vt

        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1.0
            rotation = u @ vt

        aligned_unit = candidate_unit @ rotation
        error = float(np.mean((aligned_unit - target_unit) ** 2))

        if error < best_error:
            best_error = error
            best_aligned_unit = aligned_unit

    if best_aligned_unit is None:
        raise RuntimeError("Point alignment failed to produce a candidate.")

    aligned = best_aligned_unit * target_norm + target_centroid
    return aligned.astype(np.float32)


def align_target_to_prediction(
    target: np.ndarray,
    prediction: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Align a target distance matrix to a prediction for display only.

    Mirrors the invariant reconstruction comparison: every diagonal cyclic shift
    plus its flipped version is compared with L1 distance, and the best target
    view is returned for visualization.
    """
    target_2d = np.asarray(target, dtype=np.float32).squeeze()
    prediction_2d = np.asarray(prediction, dtype=np.float32).squeeze()

    if target_2d.ndim != 2 or prediction_2d.ndim != 2:
        raise ValueError(
            f"Expected 2D distance matrices, got {target_2d.shape} and "
            f"{prediction_2d.shape}"
        )

    if target_2d.shape != prediction_2d.shape:
        raise ValueError(
            f"Target/prediction shape mismatch: {target_2d.shape} vs "
            f"{prediction_2d.shape}"
        )

    matrix_size = target_2d.shape[0]
    best_matrix = target_2d
    best_info: dict[str, Any] = {
        "shift": 0,
        "flipped": False,
        "l1": float(np.mean(np.abs(prediction_2d - target_2d))),
    }

    for shift in range(matrix_size):
        shifted = np.roll(target_2d, shift=(shift, shift), axis=(0, 1))

        for flipped in (False, True):
            candidate = np.flip(shifted, axis=(0, 1)) if flipped else shifted
            l1 = float(np.mean(np.abs(prediction_2d - candidate)))

            if l1 < best_info["l1"]:
                best_matrix = candidate
                best_info = {"shift": shift, "flipped": flipped, "l1": l1}

    return best_matrix.astype(np.float32), best_info


# =============================================================================
# Small helpers
# =============================================================================

def _close_curve(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points)
    if len(points) == 0:
        return points
    return np.concatenate([points, points[:1]], axis=0)


def _plot_outline(
    ax,
    points: np.ndarray,
    title: str = "",
    markers: bool = True,
    markersize: float = 2.0,
):
    """
    Outline plot (closed contour). With ``markers`` the sampled points are drawn
    too (the old "points that lead to a contour" style); without, just the curve.
    """
    pts = _close_curve(points)
    if markers:
        ax.plot(pts[:, 0], pts[:, 1], "-o", markersize=markersize)
    else:
        ax.plot(pts[:, 0], pts[:, 1], "-")
    if title:
        ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")


# Backward-compatible alias (kept so older imports keep working).
def _plot_curve(ax, points: np.ndarray, title: str = "", show_points: bool = True):
    _plot_outline(ax, points, title=title)


# =============================================================================
# Preprocessing visualization
# =============================================================================

def save_preprocessing_figure(
    raw_image: np.ndarray,
    clean_mask: np.ndarray,
    contour: np.ndarray,
    sampled_points: np.ndarray,
    distance_matrix: np.ndarray,
    save_path,
    title_prefix: str = "",
):
    """
    Save a 5-panel preprocessing figure:
        raw image | cleaned mask | contour | sampled contour points | distance matrix
    """
    set_plot_style()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 5, figsize=(14, 3))

    axes[0].imshow(raw_image, cmap="gray")
    axes[0].set_title(f"{title_prefix}Raw")
    axes[0].axis("off")

    axes[1].imshow(clean_mask, cmap="gray")
    axes[1].set_title(f"{title_prefix}Mask")
    axes[1].axis("off")

    _plot_outline(axes[2], contour, title=f"{title_prefix}Contour")
    _plot_outline(
        axes[3],
        sampled_points,
        title=f"{title_prefix}Sampled {len(sampled_points)} points",
    )

    im = axes[4].imshow(distance_matrix, cmap="viridis")
    axes[4].set_title(f"{title_prefix}Distance matrix")
    axes[4].set_xticks([])
    axes[4].set_yticks([])
    fig.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Training curve visualization
# =============================================================================

def save_training_curves(history_json_path, save_path):
    set_plot_style()

    history_json_path = Path(history_json_path)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(history_json_path, "r") as f:
        history = json.load(f)

    epochs = [h["epoch"] for h in history]
    total = [h["total"] for h in history]
    rec = [h["rec"] for h in history]
    kl = [h.get("kl", 0.0) for h in history]
    diag = [h.get("diag", 0.0) for h in history]
    nonneg = [h.get("nonneg", 0.0) for h in history]
    sym = [h.get("sym", 0.0) for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

    axes[0].plot(epochs, total, label="Total")
    axes[0].plot(epochs, rec, label="Reconstruction")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Main training losses")
    axes[0].legend()

    axes[1].plot(epochs, kl, label="KL")
    axes[1].plot(epochs, diag, label="Diagonal")
    axes[1].plot(epochs, nonneg, label="Non-negativity")
    axes[1].plot(epochs, sym, label="Symmetry")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Regularization terms")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Latent space t-SNE visualization
# =============================================================================

def save_tsne_figure(
    latents: np.ndarray,
    labels: np.ndarray,
    save_path,
    title: str = "Latent space t-SNE",
    max_points: int = 3000,
    random_state: int = 42,
):
    """Plot all points in a seeded uniform subsample; never filter by class overlap.

    PCA reduces features to at most 50 dimensions before t-SNE (perplexity 5).
    The sidecar records selected source indices and coordinates for inspection.
    """
    from sklearn.decomposition import PCA
    set_plot_style()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    latents = np.asarray(latents, dtype=np.float64)
    labels = np.asarray(labels)
    if latents.ndim != 2 or len(labels) != len(latents):
        raise ValueError("Expected latents [samples, features] and one label per sample")
    if len(latents) < 3 or max_points < 3:
        raise ValueError("t-SNE needs at least three samples and max_points >= 3")
    rng = np.random.default_rng(random_state)
    indices = np.sort(rng.choice(len(latents), min(max_points, len(latents)), replace=False))
    X, shown_labels = latents[indices], labels[indices]
    if X.shape[1] > 50:
        X = PCA(n_components=min(50, X.shape[1], len(X) - 1),
                random_state=random_state).fit_transform(X)
    emb = TSNE(n_components=2, random_state=random_state, init="pca",
               learning_rate="auto", perplexity=min(5.0, len(X) - 1.0)).fit_transform(X)
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    classes = np.unique(shown_labels)
    cmap = plt.get_cmap("tab10", max(len(classes), 1))
    for i, label in enumerate(classes):
        mask = shown_labels == label
        ax.scatter(emb[mask, 0], emb[mask, 1], s=12, alpha=0.8,
                   label=str(label), color=cmap(i), linewidths=0)
    ax.set(title=title, xlabel="t-SNE 1", ylabel="t-SNE 2", xticks=[], yticks=[])
    ax.legend(title="Class", frameon=False)
    ax.text(0.99, 0.01, f"{len(indices)} of {len(latents)} points; seed {random_state}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    save_path.with_suffix(".json").write_text(json.dumps({
        "sampling": "uniform without replacement; no outlier or class-overlap filtering",
        "seed": random_state, "total_points": len(latents),
        "indices": indices.tolist(), "labels": shown_labels.tolist(),
        "coordinates": emb.tolist(),
    }, indent=2) + "\n")


# =============================================================================
# Reconstruction grid visualization (point-based, ported from the old code)
# =============================================================================

def save_reconstruction_grid(
    input_matrices: np.ndarray,
    recon_matrices: np.ndarray,
    true_contours: np.ndarray,
    save_path,
    model_type: str | None = "original",
):
    """
    Save a report-style (landscape) reconstruction figure: one COLUMN per example,
    four ROWS of panels:

        row 1: original (input) contour — exactly the N connected sample points
               the model was fed (``MATRIX_SIZE`` points: 32 for MNIST, 64 for
               BBBC010), drawn with markers. NOT smoothed — this is the literal
               input, identical in style to the preprocessing figure.
        row 2: target distance matrix, aligned to the prediction for display
        row 3: reconstructed distance matrix
        row 4: unsmoothed classical-MDS projection, aligned by a similarity
               transform (including optional reflection) to the input outline

    Raw decoder values are shown in row 3. The MDS outline is a projection after
    matrix cleanup, not a certificate of valid decoded geometry. ``model_type``
    is retained for caller compatibility; every variant uses the same display.
    """
    set_plot_style()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    k = len(input_matrices)
    panel_w = 2.5 if k <= 7 else 1.9          # keep many-column figures manageable
    fig, axes = plt.subplots(4, k, figsize=(max(panel_w * k, 6.0), 9.6 if k <= 7 else 8.6))

    if k == 1:
        axes = np.array(axes).reshape(4, 1)

    matrix_size = int(np.asarray(input_matrices[0]).shape[-1])
    recon_harmonics = _harmonics_for(model_type, matrix_size)

    row_labels = [
        f"Input contour\n({matrix_size} pts)",
        "Target D\n(aligned)",
        "Reconstructed D",
        "Recon contour\n(MDS)",
    ]

    for col in range(k):
        D_in = np.asarray(input_matrices[col], dtype=np.float64)
        D_out = np.asarray(recon_matrices[col], dtype=np.float64)
        original_points = np.asarray(true_contours[col], dtype=np.float64)

        D_in_aligned, align_info = align_target_to_prediction(D_in, D_out)

        try:
            recon_points = distance_matrix_to_points(
                D_out, n_harmonics=recon_harmonics, n_out=matrix_size
            )
            recon_points = align_points_for_display(recon_points, original_points)
        except Exception as exc:
            print(f"[visualize] MDS/alignment failed at example {col}: {exc}")
            recon_points = np.zeros_like(original_points)

        ax_orig, ax_tgt, ax_rec, ax_recc = axes[:, col]

        _plot_outline(ax_orig, original_points)

        ax_tgt.imshow(D_in_aligned, cmap="viridis")
        align_label = "flip" if align_info["flipped"] else "no flip"
        ax_tgt.set_title(
            f"shift={align_info['shift']}, {align_label}", fontsize=7
        )
        ax_tgt.axis("off")

        ax_rec.imshow(D_out, cmap="viridis")
        ax_rec.axis("off")

        _plot_outline(ax_recc, recon_points)

    # row labels down the left edge (axes have their frame off, so use text)
    for r in range(4):
        axes[r, 0].text(
            -0.12, 0.5, row_labels[r],
            transform=axes[r, 0].transAxes,
            rotation=90, va="center", ha="right", fontsize=8.5,
        )

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
