from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from shapeembed.datasets import DistanceMatrixDataset
from shapeembed.encode import encode_dataset, load_trained_model
from shapeembed.models.factory import extract_latent, get_model_type, get_run_name
from shapeembed.reconstruct import (
    clean_distance_matrix,
    distance_matrix_to_points,
    recon_harmonics_for,
)
from shapeembed.utils import ensure_dir, get_device, prepare_distance_batch


# =============================================================================
# Small helpers
# =============================================================================

def _close_curve(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points)
    if len(points) == 0:
        return points
    return np.concatenate([points, points[:1]], axis=0)


def _plot_outline(ax, points: np.ndarray, title: str = "", markers: bool = True,
                  markersize: float = 2.0) -> None:
    pts = _close_curve(points)
    if markers:
        ax.plot(pts[:, 0], pts[:, 1], "-o", markersize=markersize)
    else:
        ax.plot(pts[:, 0], pts[:, 1], "-")
    if title:
        ax.set_title(title)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")


def _contour_elongation(points: np.ndarray) -> float:
    p = np.asarray(points, dtype=np.float64)
    p = p - p.mean(axis=0, keepdims=True)
    cov = (p.T @ p) / max(len(p), 1)
    w = np.maximum(np.linalg.eigvalsh(cov), 1e-12)
    return float(np.sqrt(w[-1] / w[0]))


def align_points_for_display(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Align source points to target points for visualization only.

    MDS reconstruction from a distance matrix is only defined up to translation,
    scale, rotation, and reflection. ShapeEmbed is intentionally invariant to
    rotation/reflection/indexing, so a mirrored/rotated MDS plot is not a
    training failure. This only makes figures comparable; it does not modify
    training, evaluation, embeddings, or saved outputs.
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
    Align a target distance matrix to a prediction for display only -- mirrors
    the invariant reconstruction comparison: every diagonal cyclic shift + the
    flip is compared with L1, the best target view is returned.
    """
    target_2d = np.asarray(target, dtype=np.float32).squeeze()
    prediction_2d = np.asarray(prediction, dtype=np.float32).squeeze()

    if target_2d.ndim != 2 or prediction_2d.ndim != 2:
        raise ValueError(
            f"Expected 2D distance matrices, got {target_2d.shape} and {prediction_2d.shape}"
        )
    if target_2d.shape != prediction_2d.shape:
        raise ValueError(
            f"Target/prediction shape mismatch: {target_2d.shape} vs {prediction_2d.shape}"
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


def _find_log_path(config: dict[str, Any]) -> Path | None:
    name = config["dataset"]["name"]
    run_name = get_run_name(config)
    log_dir = Path(config["outputs"]["log_dir"])
    for path in (log_dir / f"{run_name}_train_log.csv", log_dir / f"{name}_train_log.csv"):
        if path.exists():
            return path
    return None


# =============================================================================
# Preprocessing visualization
# =============================================================================

def visualize_preprocessing(config: dict[str, Any]) -> None:
    name = config["dataset"]["name"]
    processed_dir = Path(config["dataset"]["processed_dir"])
    figure_dir = ensure_dir(config["outputs"]["figure_dir"])

    samples = torch.load(processed_dir / "train.pt", weights_only=False)

    n_show = min(6, len(samples))
    fig, axes = plt.subplots(n_show, 2, figsize=(6, 3 * n_show))
    if n_show == 1:
        axes = axes[None, :]

    for i in range(n_show):
        sample = samples[i]
        points = np.asarray(sample["points"])
        D = prepare_distance_batch(
            torch.tensor(sample["D"], dtype=torch.float32).view(1, 1, *sample["D"].shape),
            config,
        ).squeeze().numpy()
        label = sample["label"]

        ax0, ax1 = axes[i]
        _plot_outline(ax0, points, title=f"{name} contour | label={label}")

        im = ax1.imshow(D, cmap="viridis")
        ax1.set_title("Prepared distance matrix")
        ax1.axis("off")
        fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    out_path = figure_dir / f"{name}_preprocessing_debug.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[visualize] saved: {out_path}")


# =============================================================================
# Training curves
# =============================================================================

def visualize_training_curves(config: dict[str, Any]) -> None:
    run_name = get_run_name(config)
    figure_dir = ensure_dir(config["outputs"]["figure_dir"])

    log_path = _find_log_path(config)
    if log_path is None:
        print("[visualize] log file not found. Skipping training curves.")
        return

    df = pd.read_csv(log_path)
    required_cols = {"epoch", "train_total", "val_total", "train_rec", "val_rec"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[visualize] log file missing columns {missing}. Skipping curves.")
        return

    plt.figure(figsize=(7, 5))
    plt.plot(df["epoch"], df["train_total"], label="train total")
    plt.plot(df["epoch"], df["val_total"], label="val total")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{run_name} training curve")
    plt.legend()
    plt.tight_layout()
    out_path = figure_dir / f"{run_name}_training_curve.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[visualize] saved: {out_path}")

    plt.figure(figsize=(7, 5))
    plt.plot(df["epoch"], df["train_rec"], label="train rec")
    plt.plot(df["epoch"], df["val_rec"], label="val rec")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction loss")
    plt.title(f"{run_name} reconstruction loss")
    plt.legend()
    plt.tight_layout()
    out_path = figure_dir / f"{run_name}_reconstruction_curve.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[visualize] saved: {out_path}")

    if len(df) > 5:
        df_tail = df.iloc[1:].copy()
        plt.figure(figsize=(7, 5))
        plt.plot(df_tail["epoch"], df_tail["train_rec"], label="train rec")
        plt.plot(df_tail["epoch"], df_tail["val_rec"], label="val rec")
        plt.xlabel("Epoch")
        plt.ylabel("Reconstruction loss")
        plt.title(f"{run_name} reconstruction loss, excluding epoch 1")
        plt.legend()
        plt.tight_layout()
        out_path = figure_dir / f"{run_name}_reconstruction_curve_no_epoch1.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[visualize] saved: {out_path}")


# =============================================================================
# Reconstructions
# =============================================================================

@torch.no_grad()
def _reconstruct_with_any_model(model, config, D):
    """
    Returns (D_hat, latent, D_model) where D_model is the distance matrix the
    model actually saw (preproc_x for the VAE, prepared D for the AE/MLP).
    Latent = mu for the VAE/MLP, z for the AE.
    """
    model.eval()
    if get_model_type(config) in ["shapeembed", "vae"]:
        preproc_x, _, _, z_mean, _, _ = model(D)
        D_hat = model.decode(z_mean)
        if D_hat.shape != preproc_x.shape:
            raise RuntimeError(
                f"Model output shape {tuple(D_hat.shape)} != preproc target {tuple(preproc_x.shape)}"
            )
        return D_hat, z_mean, preproc_x

    D = prepare_distance_batch(D, config)
    latent = extract_latent(model, config, D)
    D_hat = model.decode(latent)
    if D_hat.shape != D.shape:
        raise RuntimeError(
            f"Model output shape {tuple(D_hat.shape)} != input shape {tuple(D.shape)}"
        )
    return D_hat, latent, D


@torch.no_grad()
def select_reconstruction_examples(model, config, n_examples: int,
                                   max_eval: int = 800, seed: int = 42):
    """Seeded class-balanced examples; selection never uses model errors."""
    dataset = DistanceMatrixDataset(Path(config["dataset"]["processed_dir"]) / "test.pt")
    device = next(model.parameters()).device
    n = min(len(dataset), max_eval)
    if n == 0 or n_examples < 1:
        raise ValueError("Need a nonempty dataset and positive n_examples")
    labels = np.array([int(dataset[i]["label"]) for i in range(n)])
    rng = np.random.default_rng(seed)
    queues = [list(rng.permutation(np.flatnonzero(labels == c))) for c in np.unique(labels)]
    chosen = []
    while any(queues) and len(chosen) < n_examples:
        for queue in queues:
            if queue and len(chosen) < n_examples:
                chosen.append(queue.pop())
    batch = torch.stack([dataset[i]["D"] for i in chosen]).to(device)
    out, _, inputs = _reconstruct_with_any_model(model, config, batch)
    return (list(inputs.cpu().numpy()[:, 0]), list(out.cpu().numpy()[:, 0]),
            [dataset[i]["points"].numpy() for i in chosen], labels[chosen].tolist())


def visualize_reconstructions(config: dict[str, Any], n_show: int | None = None) -> None:
    run_name = get_run_name(config)
    figure_dir = ensure_dir(config["outputs"]["figure_dir"])
    matrix_size = int(config["dataset"]["distance_matrix_size"])
    model_type = get_model_type(config)

    device = get_device(config["training"]["device"])
    print(f"[visualize] device: {device}")
    model = load_trained_model(config, device)

    if n_show is None:
        n_show = 5 if config["dataset"]["name"].lower() == "mnist" else 6

    in_mats, out_mats, contours, labels = select_reconstruction_examples(
        model, config, n_show
    )
    k = len(in_mats)
    H = recon_harmonics_for(model_type, matrix_size)

    panel_w = 2.5 if k <= 7 else 1.9
    fig, axes = plt.subplots(4, k, figsize=(max(panel_w * k, 6.0), 9.6 if k <= 7 else 8.6))
    if k == 1:
        axes = np.array(axes).reshape(4, 1)

    row_labels = [
        f"Input contour\n({matrix_size} pts)",
        "Target D\n(aligned)",
        "Reconstructed D",
        "Recon contour\n(MDS)",
    ]

    for c in range(k):
        D_in = np.asarray(in_mats[c], dtype=np.float64)
        D_out = np.asarray(out_mats[c], dtype=np.float64)
        orig_pts = np.asarray(contours[c], dtype=np.float64)

        D_in_aligned, align_info = align_target_to_prediction(D_in, D_out)
        try:
            recon_pts = distance_matrix_to_points(D_out, n_harmonics=H, n_out=matrix_size)
            recon_pts = align_points_for_display(recon_pts, orig_pts)
        except Exception as exc:
            print(f"[visualize] MDS/align failed at example {c}: {exc}")
            recon_pts = np.zeros_like(orig_pts)

        ax_o, ax_t, ax_r, ax_rc = axes[:, c]
        _plot_outline(ax_o, orig_pts)
        ax_t.imshow(D_in_aligned, cmap="viridis")
        ax_t.set_title(
            f"shift={align_info['shift']}, {'flip' if align_info['flipped'] else 'no flip'}",
            fontsize=7,
        )
        ax_t.axis("off")
        ax_r.imshow(D_out, cmap="viridis")
        ax_r.axis("off")
        _plot_outline(ax_rc, recon_pts)

    for r in range(4):
        axes[r, 0].text(
            -0.12, 0.5, row_labels[r], transform=axes[r, 0].transAxes,
            rotation=90, va="center", ha="right", fontsize=8.5,
        )

    fig.tight_layout()
    out_path = figure_dir / f"{run_name}_reconstructions.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] saved: {out_path}")


# =============================================================================
# Latent t-SNE
# =============================================================================

def visualize_latent(config: dict[str, Any]) -> None:
    run_name = get_run_name(config)
    embedding_dir = Path(config["outputs"]["embedding_dir"])
    figure_dir = ensure_dir(config["outputs"]["figure_dir"])

    emb_path = embedding_dir / f"{run_name}_embeddings.npy"
    label_path = embedding_dir / f"{run_name}_labels.npy"
    if not emb_path.exists() or not label_path.exists():
        print("[visualize] embeddings not found. Encoding first...")
        encode_dataset(config)
    if not emb_path.exists() or not label_path.exists():
        raise FileNotFoundError(f"Embeddings still not found:\n{emb_path}\n{label_path}")

    X = np.load(emb_path)
    y = np.load(label_path)

    vc = config.get("visualization", {})
    seed = int(vc.get("tsne_seed", 42))
    classes = np.unique(y)
    paper_look = len(classes) > 3
    budget = 7500 if paper_look else int(vc.get("max_points", 3000))

    rng = np.random.default_rng(seed)
    if len(X) > budget:
        per = max(1, budget // len(classes))
        picks = [rng.choice(np.where(y == c)[0], min(per, int((y == c).sum())), replace=False)
                 for c in classes]
        sel = np.concatenate(picks)
        rng.shuffle(sel)
        X = X[sel]
        y = y[sel]
    n = len(X)

    Xr = X
    if Xr.shape[1] > 50 and n > 3:
        Xr = PCA(n_components=min(50, Xr.shape[1], n - 1), random_state=seed).fit_transform(Xr)

    perplexity = float(min(5.0, max(2.0, n - 1.0)))      # ShapeEmbed paper value
    emb = TSNE(
        n_components=2, perplexity=perplexity, init="pca",
        learning_rate="auto", random_state=seed,
    ).fit_transform(Xr)

    cmap = plt.cm.get_cmap("tab10", max(len(classes), 1))

    if paper_look:
        s = float(np.clip(72000.0 / max(n, 1), 9.0, 16.0))
        fig, ax = plt.subplots(figsize=(6.6, 5.6))
        for i, c in enumerate(classes):
            m = y == c
            ax.scatter(emb[m, 0], emb[m, 1], s=s, alpha=0.9, color=cmap(i),
                       label=str(c), linewidths=0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{run_name} latent space (t-SNE)")
        ax.legend(title="Class", markerscale=2.5, frameon=False,
                  ncol=1 if len(classes) <= 5 else 2,
                  loc="center left", bbox_to_anchor=(1.0, 0.5))
    else:
        s = 50.0 if n < 250 else float(np.clip(38000.0 / max(n, 1), 10.0, 36.0))
        fig, ax = plt.subplots(figsize=(7.4, 5.2))
        for i, c in enumerate(classes):
            m = y == c
            ax.scatter(emb[m, 0], emb[m, 1], s=s, alpha=0.85, color=cmap(i),
                       label=str(c), linewidths=0)
        ax.set_title(f"{run_name} latent space (t-SNE)")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(title="Class", frameon=False,
                  ncol=1 if len(classes) <= 5 else 2, loc="best")

    fig.tight_layout()
    out_path = figure_dir / f"{run_name}_latent_tsne.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] saved: {out_path}")


# =============================================================================
# Latent interpolation (VAE only)
# =============================================================================

def _find_different_label_pair(dataset: DistanceMatrixDataset) -> tuple[int, int]:
    if len(dataset) < 2:
        raise ValueError("Need at least two test samples for interpolation.")
    first_label = int(dataset[0]["label"].item())
    for idx in range(1, len(dataset)):
        if int(dataset[idx]["label"].item()) != first_label:
            return 0, idx
    raise ValueError("Could not find two test samples with different labels.")


@torch.no_grad()
def visualize_latent_interpolation(config: dict[str, Any], num_steps: int = 9) -> None:
    model_type = get_model_type(config)
    if model_type not in ["shapeembed", "vae"]:
        raise ValueError("Latent interpolation is only supported for the original VAE.")

    run_name = get_run_name(config)
    processed_dir = Path(config["dataset"]["processed_dir"])
    figure_dir = ensure_dir(config["outputs"]["figure_dir"])
    matrix_size = int(config["dataset"]["distance_matrix_size"])
    H = recon_harmonics_for(model_type, matrix_size)

    device = get_device(config["training"]["device"])
    dataset = DistanceMatrixDataset(processed_dir / "test.pt")
    model = load_trained_model(config, device, checkpoint="best")
    model.eval()

    idx_a, idx_b = _find_different_label_pair(dataset)
    sa, sb = dataset[idx_a], dataset[idx_b]
    label_a, label_b = int(sa["label"].item()), int(sb["label"].item())
    orig_pts_a = sa["points"].numpy()

    D_a = sa["D"].unsqueeze(0).to(device)
    D_b = sb["D"].unsqueeze(0).to(device)
    z_a, _ = model.encode(D_a)
    z_b, _ = model.encode(D_b)

    ts = np.linspace(0.0, 1.0, num_steps, dtype=np.float32)
    mats, conts = [], []
    for col, t in enumerate(ts):
        z_t = (1.0 - float(t)) * z_a + float(t) * z_b
        D_hat = model.decode(z_t)
        D_clean = D_hat.squeeze().detach().cpu().numpy()
        mats.append(D_clean)
        try:
            p = distance_matrix_to_points(D_clean, n_harmonics=H, n_out=matrix_size)
        except Exception as exc:
            print(f"[visualize] interpolation MDS failed at t={t:.2f}: {exc}")
            p = np.zeros((matrix_size, 2), dtype=np.float32)
        conts.append(p)

    # chain-align successive outlines (display only) so the morph reads smoothly
    aligned, ref = [], orig_pts_a
    for p in conts:
        ap = align_points_for_display(p, ref)
        aligned.append(ap)
        ref = ap

    fig, axes = plt.subplots(2, num_steps, figsize=(2.0 * num_steps, 4.6))
    for col, t in enumerate(ts):
        c = _close_curve(aligned[col])
        ax_c, ax_m = axes[0, col], axes[1, col]
        ax_c.plot(c[:, 0], c[:, 1], "-o", markersize=2)
        title = f"t={t:.2f}"
        if col == 0:
            title += f"\ny={label_a}"
        elif col == num_steps - 1:
            title += f"\ny={label_b}"
        ax_c.set_title(title, fontsize=8)
        ax_c.set_aspect("equal")
        ax_c.invert_yaxis()
        ax_c.axis("off")
        ax_m.imshow(mats[col], cmap="viridis")
        ax_m.axis("off")

    axes[0, 0].text(-0.2, 0.5, "Decoded\noutline", transform=axes[0, 0].transAxes,
                    rotation=90, va="center", ha="right", fontsize=8.5)
    axes[1, 0].text(-0.2, 0.5, "Decoded\nmatrix", transform=axes[1, 0].transAxes,
                    rotation=90, va="center", ha="right", fontsize=8.5)
    fig.suptitle(f"{run_name}: VAE latent-space interpolation", y=1.02, fontsize=10)
    fig.tight_layout()
    out_path = figure_dir / f"{run_name}_latent_interpolation.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] saved: {out_path}")


# =============================================================================

def visualize_all(config: dict[str, Any]) -> None:
    visualize_training_curves(config)
    visualize_reconstructions(config)
    visualize_latent(config)
