import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from config import (
    DATASET,
    MODEL_TYPE,
    MATRIX_SIZE,
    LATENT_DIM,
    CHECKPOINT_DIR,
    RESULT_DIR,
    FIGURE_DIR,
    DATA_PROCESSED,
    RUN_NAME,
    DEVICE,
    BATCH_SIZE,
)
from models import build_model
from dataset import DistanceMatrixDataset
from visualize import (
    save_preprocessing_figure,
    save_training_curves,
    save_reconstruction_grid,
    align_target_to_prediction,
    set_plot_style,
)
from preprocess import (
    load_mnist_raw,
    load_bbbc010_raw,
    binarize_image,
    extract_main_contour,
    ensure_counterclockwise,
    uniformly_sample_closed_contour,
    contour_to_distance_matrix,
)


def preprocess_single_example(image):
    """
    Apply the same visualization preprocessing pipeline:
        raw image -> binary mask -> contour -> sampled points -> distance matrix
    """
    clean_mask = binarize_image(image)

    contour = extract_main_contour(clean_mask)
    contour = ensure_counterclockwise(contour)

    sampled = uniformly_sample_closed_contour(contour, MATRIX_SIZE)
    D, _ = contour_to_distance_matrix(sampled)

    return clean_mask, contour, sampled, D


def select_preprocessing_examples(images, labels):
    """
    MNIST:
        select one valid sample per digit class: 0,...,9

    BBBC010:
        select 3 dead and 3 live samples if possible.
    """
    selected = []

    if DATASET == "mnist":
        wanted_labels = list(range(10))
        seen = set()

        for image, label in zip(images, labels):
            if label in seen:
                continue

            try:
                clean_mask, contour, sampled, D = preprocess_single_example(image)
                selected.append((image, clean_mask, contour, sampled, D, label))
                seen.add(label)
            except Exception:
                continue

            if len(seen) == len(wanted_labels):
                break

    elif DATASET == "bbbc010":
        counts = {0: 0, 1: 0}
        max_per_class = 3

        for image, label in zip(images, labels):
            if label not in counts:
                continue

            if counts[label] >= max_per_class:
                continue

            try:
                clean_mask, contour, sampled, D = preprocess_single_example(image)
                selected.append((image, clean_mask, contour, sampled, D, label))
                counts[label] += 1
            except Exception:
                continue

            if counts[0] >= max_per_class and counts[1] >= max_per_class:
                break

    else:
        raise ValueError(f"Unknown DATASET: {DATASET}")

    if len(selected) == 0:
        raise RuntimeError("Could not select any valid preprocessing examples.")

    return selected


def build_preprocessing_gallery():
    """
    Save a landscape multi-sample preprocessing figure: one COLUMN per sample
    (10 digit classes for MNIST, 3 live + 3 dead for BBBC010), five ROWS of
    panels: raw image, cleaned mask, contour, sampled N-point contour, distance
    matrix.
    """
    if DATASET == "mnist":
        images, labels = load_mnist_raw()
    elif DATASET == "bbbc010":
        images, labels = load_bbbc010_raw()
    else:
        raise ValueError(f"Unknown DATASET: {DATASET}")

    selected = select_preprocessing_examples(images, labels)

    set_plot_style()

    n_cols = len(selected)
    n_rows = 5

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(2.0 * n_cols, 6.0), 9.0),
    )

    if n_cols == 1:
        axes = np.array(axes).reshape(n_rows, 1)

    row_labels = ["Raw", "Mask", "Contour", f"Sampled\n{MATRIX_SIZE} points", "Distance\nmatrix"]

    for c, (image, clean_mask, contour, sampled, D, label) in enumerate(selected):
        axes[0, c].imshow(image, cmap="gray")
        axes[0, c].axis("off")

        axes[1, c].imshow(clean_mask, cmap="gray")
        axes[1, c].axis("off")

        contour_closed = np.concatenate([contour, contour[:1]], axis=0)
        axes[2, c].plot(contour_closed[:, 0], contour_closed[:, 1], "-")
        axes[2, c].set_aspect("equal")
        axes[2, c].axis("off")

        sampled_closed = np.concatenate([sampled, sampled[:1]], axis=0)
        axes[3, c].plot(sampled_closed[:, 0], sampled_closed[:, 1], "-o", markersize=2)
        axes[3, c].set_aspect("equal")
        axes[3, c].axis("off")

        axes[4, c].imshow(D, cmap="viridis")
        axes[4, c].axis("off")

        if DATASET == "mnist":
            col_label = f"digit {label}"
        else:
            col_label = "dead" if label == 0 else "live"
        axes[0, c].set_title(col_label, fontsize=9)

    for r in range(n_rows):
        axes[r, 0].text(
            -0.18, 0.5, row_labels[r],
            transform=axes[r, 0].transAxes,
            rotation=90, va="center", ha="right", fontsize=9,
        )

    fig.tight_layout()

    save_path = FIGURE_DIR / f"{DATASET}_preprocessing_gallery.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved preprocessing gallery: {save_path}")


def build_training_curve_figure():
    history_path = RESULT_DIR / f"{RUN_NAME}_training_history.json"
    save_path = FIGURE_DIR / f"{RUN_NAME}_training_curves.png"
    save_training_curves(history_path, save_path)
    print(f"Saved training curves: {save_path}")


def select_reconstruction_examples(model, num_examples: int = 6, max_eval: int = 800,
                                   model_type: str | None = None, skip: int = 0,
                                   seed: int = 42):
    """Seeded class-balanced test examples, independent of reconstruction error.

    Returns raw model inputs/outputs and the corresponding ground-truth contours.
    The same seed selects the same examples for each model. No output enhancement
    or smoothing is applied. ``skip`` selects the next group of examples.
    """
    dataset = DistanceMatrixDataset(split="test")
    contours = torch.load(DATA_PROCESSED / "contours_test.pt", weights_only=True).numpy()
    n = min(len(dataset), max_eval)
    if n == 0 or num_examples < 1 or skip < 0:
        raise ValueError("Need a nonempty dataset, positive num_examples and nonnegative skip")
    labels = np.array([int(dataset[i][1]) for i in range(n)])
    rng = np.random.default_rng(seed)
    queues = [list(rng.permutation(np.flatnonzero(labels == c))) for c in np.unique(labels)]
    chosen = []
    while any(queues) and len(chosen) < num_examples + skip:
        for queue in queues:
            if queue and len(chosen) < num_examples + skip:
                chosen.append(queue.pop())
    chosen = chosen[skip:skip + num_examples]
    if not chosen:
        raise ValueError("skip exceeds the available examples")
    x = torch.stack([dataset[i][0] for i in chosen]).to(DEVICE)
    model.eval()
    with torch.no_grad():
        outputs = model(x)
    return (outputs["x_pre"].cpu().numpy()[:, 0],
            outputs["recon"].cpu().numpy()[:, 0], contours[chosen])


def build_reconstruction_figure(num_examples: int | None = None):
    if num_examples is None:
        num_examples = 6

    model = build_model(
        model_type=MODEL_TYPE,
        latent_dim=LATENT_DIM,
        matrix_size=MATRIX_SIZE,
    ).to(DEVICE)

    ckpt_path = CHECKPOINT_DIR / f"{RUN_NAME}_best.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    input_mats, recon_mats, contour_list = select_reconstruction_examples(
        model, num_examples=num_examples, model_type=MODEL_TYPE
    )

    save_reconstruction_grid(
        input_matrices=input_mats,
        recon_matrices=recon_mats,
        true_contours=contour_list,
        save_path=FIGURE_DIR / f"{RUN_NAME}_reconstructions.png",
        model_type=MODEL_TYPE,
    )

    print(f"Saved reconstruction figure: {FIGURE_DIR / f'{RUN_NAME}_reconstructions.png'}")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    build_preprocessing_gallery()

    history_path = RESULT_DIR / f"{RUN_NAME}_training_history.json"
    ckpt_path = CHECKPOINT_DIR / f"{RUN_NAME}_best.pt"

    if history_path.exists():
        build_training_curve_figure()
    else:
        print(f"Skipping training curves; missing: {history_path}")

    if ckpt_path.exists():
        build_reconstruction_figure()
    else:
        print(f"Skipping reconstructions; missing: {ckpt_path}")


if __name__ == "__main__":
    main()
