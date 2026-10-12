import os
from pathlib import Path
from typing import List, Tuple
from scipy.interpolate import splprep, splev

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from skimage import measure, filters
from sklearn.model_selection import train_test_split
from torchvision.datasets import MNIST
from tqdm import tqdm

from config import (
    DATASET,
    DATA_RAW,
    DATA_PROCESSED,
    MATRIX_SIZE,
    TRAIN_SPLIT,
    SEED,
)


# =============================================================================
# Basic mask cleaning
# =============================================================================

def binarize_image(image: np.ndarray) -> np.ndarray:
    """
    Convert grayscale image/mask to binary mask.

    For MNIST:
        input is grayscale digit image.

    For BBBC010:
        input should already be a binary or near-binary mask.

    We use Otsu threshold when possible.
    """

    image = image.astype(np.float32)

    if image.max() > 1.0:
        image = image / 255.0

    if image.max() == image.min():
        return image > 0.0

    threshold = filters.threshold_otsu(image)
    mask = image > threshold

    return mask.astype(bool)


def keep_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """
    ShapeEmbed assumes one simply-connected contour.

    If a mask has multiple foreground components, we keep the largest one.
    This is important for messy masks.
    """

    labeled, num = ndimage.label(mask)

    if num == 0:
        raise ValueError("Mask has no foreground pixels.")

    component_sizes = ndimage.sum(mask, labeled, index=np.arange(1, num + 1))
    largest_label = int(np.argmax(component_sizes)) + 1

    return labeled == largest_label


def fill_binary_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill holes so that the object has a single outer contour.

    This is consistent with the paper's simply-connected contour assumption.
    Holes/topology are discussed separately in Phase 3.
    """

    return ndimage.binary_fill_holes(mask)


# =============================================================================
# Contour extraction
# =============================================================================

def extract_main_contour(mask: np.ndarray) -> np.ndarray:
    """
    Extract contour using marching squares.

    Paper:
        They use marching squares to extract the pixel outline.

    skimage.measure.find_contours returns coordinates as:
        [row, col]

    We convert to:
        [x, y] = [col, -row]

    The -row turns image coordinates into Cartesian-like coordinates.
    Distance matrices are invariant to rotation/reflection, but this makes
    orientation handling cleaner.
    """

    contours = measure.find_contours(mask.astype(np.float32), level=0.5)

    if len(contours) == 0:
        raise ValueError("No contour found.")

    # Use the longest contour as the main simply-connected outline.
    contour_rc = max(contours, key=len)

    # Convert [row, col] to [x, y].
    contour_xy = np.stack(
        [contour_rc[:, 1], -contour_rc[:, 0]],
        axis=1,
    ).astype(np.float32)

    return contour_xy


def polygon_signed_area(points: np.ndarray) -> float:
    """
    Signed polygon area.

    Positive/negative sign tells us traversal direction.
    """

    x = points[:, 0]
    y = points[:, 1]

    return 0.5 * np.sum(
        x * np.roll(y, -1) - np.roll(x, -1) * y
    )


def ensure_counterclockwise(points: np.ndarray) -> np.ndarray:
    """
    Paper says the contour is sampled counterclockwise.

    The model/loss is direction-invariant later, but we still enforce
    a consistent convention here.
    """

    area = polygon_signed_area(points)

    if area < 0:
        points = points[::-1].copy()

    return points


# =============================================================================
# Uniform contour sampling
# =============================================================================

def uniformly_sample_closed_contour(
    points: np.ndarray,
    n_points: int,
) -> np.ndarray:
    """
    Uniformly sample N points along a closed contour.

    Paper:
        The outline is interpolated with a parametric linear spline curve
        and uniformly sampled.

    This function implements linear-spline arc-length sampling.
    """

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected points with shape [M, 2], got {points.shape}")

    if len(points) < 3:
        raise ValueError("Need at least 3 contour points.")

    # Close the contour explicitly.
    closed = np.concatenate([points, points[:1]], axis=0)

    # Segment lengths.
    diffs = closed[1:] - closed[:-1]
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))

    perimeter = seg_lengths.sum()

    if perimeter <= 1e-8:
        raise ValueError("Degenerate contour with zero perimeter.")

    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])

    # endpoint=False: the contour is closed, so the last sample would duplicate the first.
    target_distances = np.linspace(
        0.0,
        perimeter,
        n_points,
        endpoint=False,
    )

    sampled = np.zeros((n_points, 2), dtype=np.float32)

    for i, d in enumerate(target_distances):
        seg_idx = np.searchsorted(cumulative, d, side="right") - 1
        seg_idx = min(seg_idx, len(seg_lengths) - 1)

        local_start = cumulative[seg_idx]
        local_length = seg_lengths[seg_idx]

        if local_length <= 1e-8:
            alpha = 0.0
        else:
            alpha = (d - local_start) / local_length

        sampled[i] = (
            (1.0 - alpha) * closed[seg_idx]
            + alpha * closed[seg_idx + 1]
        )

    return sampled


# =============================================================================
# Distance matrix
# =============================================================================

def contour_to_distance_matrix(points: np.ndarray):
    """
    Convert sampled contour points to Euclidean distance matrix.

    Paper:
        D[i, j] = ||x_i - x_j||_2

    For ShapeEmbed:
        use normalized D as model input.

    For ShapeEmbed+Sz:
        save the original Frobenius norm before normalization.
    """

    diff = points[:, None, :] - points[None, :, :]
    distance_matrix = np.sqrt((diff ** 2).sum(axis=-1))

    fro_norm = np.linalg.norm(distance_matrix, ord="fro")

    normalized_distance_matrix = distance_matrix / (fro_norm + 1e-8)

    return normalized_distance_matrix.astype(np.float32), np.float32(fro_norm)


def mask_to_distance_matrix(
    mask: np.ndarray,
    n_points: int,
):
    """
    Paper-faithful preprocessing for one object:

        mask
        -> marching-squares contour
        -> counterclockwise orientation
        -> uniform contour sampling
        -> normalized distance matrix

    Also returns the pre-normalization Frobenius norm for ShapeEmbed+Sz.
    """

    mask = binarize_image(mask)

    contour = extract_main_contour(mask)
    contour = ensure_counterclockwise(contour)

    sampled_points = uniformly_sample_closed_contour(
        contour,
        n_points=n_points,
    )

    distance_matrix, fro_norm = contour_to_distance_matrix(sampled_points)

    return distance_matrix, sampled_points.astype(np.float32), fro_norm


# =============================================================================
# MNIST loading
# =============================================================================

def load_mnist_raw() -> Tuple[List[np.ndarray], List[int]]:
    """
    Load all 70,000 MNIST images using torchvision.

    The paper treats MNIST as a dataset of 70,000 objects.
    We combine train and test, then create our own stratified 80/20 split.
    """

    root = DATA_RAW / "mnist"

    train_set = MNIST(
        root=str(root),
        train=True,
        download=True,
    )

    test_set = MNIST(
        root=str(root),
        train=False,
        download=True,
    )

    images = []
    labels = []

    for img, label in train_set:
        images.append(np.array(img))
        labels.append(int(label))

    for img, label in test_set:
        images.append(np.array(img))
        labels.append(int(label))

    return images, labels


# =============================================================================
# BBBC010 loading
# =============================================================================

def load_bbbc010_raw() -> Tuple[List[np.ndarray], List[int]]:
    """
    Load BBBC010 masks from local folders.

    Expected folder structure:

        data/raw/bbbc010/
            live/
                *.png
                *.tif
                ...
            dead/
                *.png
                *.tif
                ...

    Accepted class folder names:
        live, alive -> label 1
        dead        -> label 0

    This function does not download BBBC010 automatically.
    Put the binary masks in these folders first.
    """

    root = DATA_RAW / "bbbc010"

    if not root.exists():
        raise FileNotFoundError(
            f"BBBC010 folder not found: {root}\n"
            f"Expected data/raw/bbbc010/live and data/raw/bbbc010/dead"
        )

    class_name_to_label = {
        "dead": 0,
        "live": 1,
        "alive": 1,
    }

    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    images = []
    labels = []

    for class_dir in root.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name.lower()

        if class_name not in class_name_to_label:
            continue

        label = class_name_to_label[class_name]

        files = [
            p for p in class_dir.rglob("*")
            if p.suffix.lower() in valid_exts
        ]

        for path in files:
            img = Image.open(path).convert("L")
            images.append(np.array(img))
            labels.append(label)

    if len(images) == 0:
        raise RuntimeError(
            f"No BBBC010 mask images found under {root}.\n"
            f"Use folders named live/ and dead/."
        )

    return images, labels


# =============================================================================
# Dataset preprocessing
# =============================================================================

def preprocess_dataset() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    if DATASET == "mnist":
        images, labels = load_mnist_raw()
    elif DATASET == "bbbc010":
        images, labels = load_bbbc010_raw()
    else:
        raise ValueError(f"Unknown DATASET: {DATASET}")

    distance_matrices = []
    contours = []
    norms = []
    clean_labels = []

    skipped = 0

    for image, label in tqdm(
        list(zip(images, labels)),
        desc=f"Preprocessing {DATASET}",
    ):
        try:
            D, pts, fro_norm = mask_to_distance_matrix(
                mask=image,
                n_points=MATRIX_SIZE,
            )

            distance_matrices.append(D)
            contours.append(pts)
            norms.append(fro_norm)
            clean_labels.append(label)

        except Exception:
            skipped += 1
            continue

    if len(distance_matrices) == 0:
        raise RuntimeError("No valid objects were preprocessed.")

    X = np.stack(distance_matrices, axis=0)  # [B, N, N]
    P = np.stack(contours, axis=0)           # [B, N, 2]
    y = np.array(clean_labels, dtype=np.int64)
    S = np.array(norms, dtype=np.float32)

    # Add channel dimension for PyTorch Conv2d:
    # [B, N, N] -> [B, 1, N, N]
    X = X[:, None, :, :]

    indices = np.arange(len(X))

    train_idx, test_idx = train_test_split(
        indices,
        train_size=TRAIN_SPLIT,
        random_state=SEED,
        stratify=y,
    )

    S_train = torch.from_numpy(S[train_idx]).float()
    S_test = torch.from_numpy(S[test_idx]).float()

    X_train = torch.from_numpy(X[train_idx]).float()
    y_train = torch.from_numpy(y[train_idx]).long()
    P_train = torch.from_numpy(P[train_idx]).float()

    X_test = torch.from_numpy(X[test_idx]).float()
    y_test = torch.from_numpy(y[test_idx]).long()
    P_test = torch.from_numpy(P[test_idx]).float()

    torch.save(X_train, DATA_PROCESSED / "X_train.pt")
    torch.save(y_train, DATA_PROCESSED / "y_train.pt")
    torch.save(P_train, DATA_PROCESSED / "contours_train.pt")

    torch.save(X_test, DATA_PROCESSED / "X_test.pt")
    torch.save(y_test, DATA_PROCESSED / "y_test.pt")
    torch.save(P_test, DATA_PROCESSED / "contours_test.pt")

    torch.save(S_train, DATA_PROCESSED / "norms_train.pt")
    torch.save(S_test, DATA_PROCESSED / "norms_test.pt")

    print()
    print(f"Saved processed dataset to: {DATA_PROCESSED}")
    print(f"Dataset: {DATASET}")
    print(f"Matrix size: {MATRIX_SIZE} x {MATRIX_SIZE}")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Skipped objects: {skipped}")


if __name__ == "__main__":
    preprocess_dataset()
