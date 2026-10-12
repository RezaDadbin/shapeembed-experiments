import numpy as np
import torch
import matplotlib.pyplot as plt

from config import (
    DATASET,
    LATENT_DIM,
    MATRIX_SIZE,
    CHECKPOINT_DIR,
    FIGURE_DIR,
    DEVICE,
)

from dataset import DistanceMatrixDataset
from models import build_model
from visualize import (
    distance_matrix_to_points,
    set_plot_style,
    align_points_for_display,
    _harmonics_for,
)


def choose_two_different_shapes(dataset: DistanceMatrixDataset):
    """
    Pick two samples with different labels if possible.
    For MNIST: usually two different digits.
    For BBBC010: live vs dead if both exist.
    """

    labels = dataset.y.numpy()
    unique_labels = np.unique(labels)

    if len(unique_labels) >= 2:
        label_a = unique_labels[0]
        label_b = unique_labels[1]

        idx_a = int(np.where(labels == label_a)[0][0])
        idx_b = int(np.where(labels == label_b)[0][0])
    else:
        idx_a = 0
        idx_b = min(1, len(dataset) - 1)

    return idx_a, idx_b


def plot_interpolation(
    decoded_matrices,
    decoded_contours,
    save_path,
):
    """
    Save a report-style interpolation figure.

    Row 1:
        decoded contours from interpolated latent vectors

    Row 2:
        decoded distance matrices
    """

    set_plot_style()

    num_steps = len(decoded_matrices)
    fig, axes = plt.subplots(
        2,
        num_steps,
        figsize=(2.0 * num_steps, 4.2),
    )

    if num_steps == 1:
        axes = np.array(axes).reshape(2, 1)

    ts = np.linspace(0.0, 1.0, num_steps)

    # Chain-align successive decoded outlines (display only) so the
    # interpolation reads as a smooth morph instead of random flips/rotations.
    aligned_contours = []
    reference = None
    for contour in decoded_contours:
        contour = np.asarray(contour)
        if reference is not None:
            contour = align_points_for_display(contour, reference)
        aligned_contours.append(contour)
        reference = contour

    for i, (D, contour, t) in enumerate(zip(decoded_matrices, aligned_contours, ts)):
        pts = np.concatenate([contour, contour[:1]], axis=0)

        axes[0, i].plot(pts[:, 0], pts[:, 1], "-o", markersize=2)
        axes[0, i].set_aspect("equal")
        axes[0, i].axis("off")
        axes[0, i].set_title(f"t={t:.2f}")

        axes[1, i].imshow(D, cmap="viridis")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Decoded\noutline")
    axes[1, 0].set_ylabel("Decoded\nmatrix")

    fig.suptitle(f"{DATASET}: latent interpolation using original ShapeEmbed", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def run_interpolation(num_steps: int = 9):
    """
    Phase 2 interpolation experiment.

    Uses:
        MODEL_TYPE = original

    Reason:
        For a VAE, we interpolate between the latent means mu_A and mu_B
        rather than stochastic samples z.
    """

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    model_type = "original"
    run_name = f"{DATASET}_{model_type}"

    dataset = DistanceMatrixDataset(split="test")

    idx_a, idx_b = choose_two_different_shapes(dataset)

    x_a, y_a = dataset[idx_a]
    x_b, y_b = dataset[idx_b]

    x_pair = torch.stack([x_a, x_b], dim=0).to(DEVICE)

    model = build_model(
        model_type=model_type,
        latent_dim=LATENT_DIM,
        matrix_size=MATRIX_SIZE,
    ).to(DEVICE)

    ckpt_path = CHECKPOINT_DIR / f"{run_name}_best.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        outputs = model(x_pair)

        # For VAE interpolation, use mu, not sampled z.
        mu_a = outputs["mu"][0]
        mu_b = outputs["mu"][1]

        decoded_matrices = []
        decoded_contours = []

        for t in np.linspace(0.0, 1.0, num_steps):
            z_t = (1.0 - t) * mu_a + t * mu_b
            D_t = model.decoder(z_t.unsqueeze(0))

            D_np = D_t[0, 0].detach().cpu().numpy()
            contour_np = distance_matrix_to_points(
                D_np,
                n_harmonics=_harmonics_for("original", MATRIX_SIZE),
                n_out=MATRIX_SIZE,
            )

            decoded_matrices.append(D_np)
            decoded_contours.append(contour_np)

    save_path = FIGURE_DIR / f"{DATASET}_original_latent_interpolation.png"

    plot_interpolation(
        decoded_matrices=decoded_matrices,
        decoded_contours=decoded_contours,
        save_path=save_path,
    )

    print("Interpolation finished.")
    print(f"Sample A index: {idx_a}, label: {int(y_a)}")
    print(f"Sample B index: {idx_b}, label: {int(y_b)}")
    print(f"Saved figure to: {save_path}")


if __name__ == "__main__":
    run_interpolation(num_steps=9)
