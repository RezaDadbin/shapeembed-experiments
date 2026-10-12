from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from skimage.io import imread
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from tqdm import tqdm

from shapeembed.preprocessing import binarize_mask, mask_to_distance_matrix
from shapeembed.utils import ensure_dir, save_json


class DistanceMatrixDataset(Dataset):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.samples = torch.load(self.path, weights_only=False)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        D = torch.tensor(sample["D"], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(sample["label"], dtype=torch.long)
        norm = torch.tensor(sample["norm"], dtype=torch.float32)
        points = torch.tensor(sample["points"], dtype=torch.float32)

        return {
            "D": D,
            "label": label,
            "norm": norm,
            "points": points,
        }


def mask_to_sample(
    mask: np.ndarray,
    label: int,
    n_points: int,
    contour_extraction: str = "project",
    contour_resampling: str = "linear",
) -> dict[str, Any] | None:
    try:
        mask = binarize_mask(mask)
        points, D, norm, info = mask_to_distance_matrix(
            mask,
            n_points=n_points,
            contour_extraction=contour_extraction,
            resampling=contour_resampling,
            return_info=True,
        )
    except Exception:
        return None

    if not np.isfinite(D).all():
        return None

    return {
        "D": D.astype(np.float32),
        "points": points.astype(np.float32),
        "label": int(label),
        "norm": float(norm),
        "preprocessing_info": info,
    }


def prepare_mnist(config: dict[str, Any]) -> None:
    dataset_cfg = config["dataset"]

    raw_dir = Path(dataset_cfg["raw_dir"])
    processed_dir = ensure_dir(dataset_cfg["processed_dir"])
    n_points = int(dataset_cfg["distance_matrix_size"])
    contour_extraction = str(dataset_cfg.get("contour_extraction", "project"))
    contour_resampling = str(dataset_cfg.get("contour_resampling", "linear"))
    test_size = float(dataset_cfg["test_size"])
    seed = int(dataset_cfg["seed"])
    max_samples = dataset_cfg.get("max_samples", None)

    print("[MNIST] Loading MNIST through torchvision...")

    train_set = MNIST(root=str(raw_dir), train=True, download=True)
    test_set = MNIST(root=str(raw_dir), train=False, download=True)

    print(f"[MNIST] Total raw samples: {len(train_set) + len(test_set)}")
    print(f"[MNIST] Using contour points: {n_points}")
    print(f"[MNIST] contour_extraction: {contour_extraction}")
    print(f"[MNIST] contour_resampling: {contour_resampling}")

    def process_split(split_set: MNIST, desc: str) -> tuple[list[dict[str, Any]], int]:
        split_samples = []
        split_skipped = 0

        for img, label in tqdm(split_set, desc=desc):
            img_array = np.array(img).astype(np.float32)
            if img_array.max() > 1.0:
                img_array = img_array / 255.0
            mask = img_array > 0.2

            sample = mask_to_sample(
                mask,
                label=label,
                n_points=n_points,
                contour_extraction=contour_extraction,
                contour_resampling=contour_resampling,
            )

            if sample is None:
                split_skipped += 1
                continue

            split_samples.append(sample)

        return split_samples, split_skipped

    train_raw_samples, train_skipped = process_split(
        train_set,
        "[MNIST] Processing train raw",
    )
    test_raw_samples, test_skipped = process_split(
        test_set,
        "[MNIST] Processing test raw",
    )
    skipped = train_skipped + test_skipped

    samples = train_raw_samples + test_raw_samples

    if len(samples) == 0:
        raise RuntimeError("No valid MNIST samples were created.")

    sample_labels = np.array([s["label"] for s in samples])

    if max_samples is not None:
        max_samples = max(0, int(max_samples))

        if 0 < max_samples < len(samples):
            indices = np.arange(len(samples))
            subset_idx, _ = train_test_split(
                indices,
                train_size=max_samples,
                random_state=seed,
                stratify=sample_labels,
            )
            samples = [samples[i] for i in subset_idx]
            sample_labels = np.array([s["label"] for s in samples])

    fallback_count = sum(
        1
        for sample in samples
        if sample.get("preprocessing_info", {}).get("resampling_fallback")
    )

    indices = np.arange(len(samples))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=sample_labels,
    )

    train_samples = [samples[i] for i in train_idx]
    test_samples = [samples[i] for i in test_idx]

    torch.save(train_samples, processed_dir / "train.pt")
    torch.save(test_samples, processed_dir / "test.pt")

    metadata = {
        "dataset": "mnist",
        "preprocessing": (
            f"marching_squares_{contour_extraction}_{contour_resampling}"
        ),
        "contour_extraction": contour_extraction,
        "contour_resampling": contour_resampling,
        "contour_level": 0.8
        if contour_extraction.lower() in ["official", "longest"]
        else 0.5,
        "distance_matrix_on_disk": "frobenius_normalized_unscaled",
        "scale_distance_matrix_by_size": bool(
            dataset_cfg.get("scale_distance_matrix_by_size", True)
        ),
        "shapeembed_phase1_preprocessing_note": (
            "model.type=shapeembed applies official Normalize('fro') and "
            "Scale(matrix_size) inside MyNet.forward when model.normalize_input=fro"
        ),
        "n_points": n_points,
        "num_total": len(samples),
        "num_train": len(train_samples),
        "num_test": len(test_samples),
        "split_strategy": "stratified_train_test_split_over_all_70000_raw_samples",
        "test_size": test_size,
        "seed": seed,
        "num_skipped": skipped,
        "num_resampling_fallbacks": fallback_count,
        "labels": sorted(list(set(sample_labels.tolist()))),
    }

    save_json(metadata, processed_dir / "metadata.json")

    print(f"[MNIST] Valid samples: {len(samples)}")
    print(f"[MNIST] Skipped samples: {skipped}")
    print(f"[MNIST] Resampling fallbacks: {fallback_count}")
    print(f"[MNIST] Saved train set: {processed_dir / 'train.pt'}")
    print(f"[MNIST] Saved test set: {processed_dir / 'test.pt'}")
    print(f"[MNIST] Saved metadata: {processed_dir / 'metadata.json'}")


def infer_bbbc010_label_from_filename(path: Path) -> int | None:
    """
    Label convention:
        1 = alive
        0 = dead

    Columns 01-12 are treated as live.
    Columns 13-24 are treated as dead.
    """
    name = path.name

    match = re.search(r"([A-Pa-p])(\d{2})", name)
    if match is None:
        return None

    col = int(match.group(2))

    if 1 <= col <= 12:
        return 1

    if 13 <= col <= 24:
        return 0

    return None


def find_bbbc010_mask_files(raw_dir: Path) -> list[Path]:
    exts = ["*.png", "*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.bmp"]
    files: list[Path] = []

    for ext in exts:
        files.extend(raw_dir.rglob(ext))

    return sorted(files)


def prepare_bbbc010(config: dict[str, Any]) -> None:
    dataset_cfg = config["dataset"]

    raw_dir = Path(dataset_cfg["raw_dir"])
    processed_dir = ensure_dir(dataset_cfg["processed_dir"])
    n_points = int(dataset_cfg["distance_matrix_size"])
    contour_extraction = str(dataset_cfg.get("contour_extraction", "project"))
    contour_resampling = str(dataset_cfg.get("contour_resampling", "linear"))
    test_size = float(dataset_cfg["test_size"])
    seed = int(dataset_cfg["seed"])
    max_samples = dataset_cfg.get("max_samples", None)

    mask_files = find_bbbc010_mask_files(raw_dir)

    if len(mask_files) == 0:
        raise FileNotFoundError(
            f"No BBBC010 mask files found under {raw_dir}. "
            "Make sure the foreground-eachworm dataset is extracted."
        )

    if max_samples is not None:
        mask_files = mask_files[: int(max_samples)]

    print(f"[BBBC010] Found mask files: {len(mask_files)}")
    print(f"[BBBC010] Using contour points: {n_points}")
    print(f"[BBBC010] contour_extraction: {contour_extraction}")
    print(f"[BBBC010] contour_resampling: {contour_resampling}")

    samples = []
    skipped_invalid = 0
    skipped_label = 0

    for path in tqdm(mask_files, desc="[BBBC010] Processing"):
        label = infer_bbbc010_label_from_filename(path)

        if label is None:
            skipped_label += 1
            continue

        try:
            mask = imread(path)
        except Exception:
            try:
                mask = np.array(Image.open(path))
            except Exception:
                skipped_invalid += 1
                continue

        sample = mask_to_sample(
            mask,
            label=label,
            n_points=n_points,
            contour_extraction=contour_extraction,
            contour_resampling=contour_resampling,
        )

        if sample is None:
            skipped_invalid += 1
            continue

        sample["filename"] = str(path)
        samples.append(sample)

    if len(samples) == 0:
        raise RuntimeError("No valid BBBC010 samples were created.")

    sample_labels = np.array([s["label"] for s in samples])
    fallback_count = sum(
        1
        for sample in samples
        if sample.get("preprocessing_info", {}).get("resampling_fallback")
    )

    indices = np.arange(len(samples))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=sample_labels,
    )

    train_samples = [samples[i] for i in train_idx]
    test_samples = [samples[i] for i in test_idx]

    torch.save(train_samples, processed_dir / "train.pt")
    torch.save(test_samples, processed_dir / "test.pt")

    class_counts = {
        "dead_0": int((sample_labels == 0).sum()),
        "alive_1": int((sample_labels == 1).sum()),
    }
    expected_total = 1407
    expected_class_counts = {
        "dead_0": 639,
        "alive_1": 768,
    }
    metadata_warnings = []

    if max_samples is None:
        total_tolerance = max(10, int(round(0.05 * expected_total)))

        if abs(len(samples) - expected_total) > total_tolerance:
            metadata_warnings.append(
                f"valid sample count {len(samples)} differs from expected "
                f"approximately {expected_total} by more than {total_tolerance}"
            )

        for key, expected in expected_class_counts.items():
            tolerance = max(5, int(round(0.05 * expected)))
            actual = class_counts[key]

            if abs(actual - expected) > tolerance:
                metadata_warnings.append(
                    f"class count {key}={actual} differs from expected "
                    f"approximately {expected} by more than {tolerance}"
                )

    for warning in metadata_warnings:
        print(f"[BBBC010][warning] {warning}")

    metadata = {
        "dataset": "bbbc010",
        "preprocessing": (
            f"marching_squares_{contour_extraction}_{contour_resampling}"
        ),
        "contour_extraction": contour_extraction,
        "contour_resampling": contour_resampling,
        "contour_level": 0.8
        if contour_extraction.lower() in ["official", "longest"]
        else 0.5,
        "distance_matrix_on_disk": "frobenius_normalized_unscaled",
        "scale_distance_matrix_by_size": bool(
            dataset_cfg.get("scale_distance_matrix_by_size", True)
        ),
        "shapeembed_phase1_preprocessing_note": (
            "model.type=shapeembed applies official Normalize('fro') and "
            "Scale(matrix_size) inside MyNet.forward when model.normalize_input=fro"
        ),
        "n_points": n_points,
        "num_total": len(samples),
        "num_train": len(train_samples),
        "num_test": len(test_samples),
        "num_skipped_invalid_masks": skipped_invalid,
        "num_skipped_unknown_labels": skipped_label,
        "num_resampling_fallbacks": fallback_count,
        "class_counts": class_counts,
        "expected_num_total_approx": expected_total,
        "expected_class_counts_approx": expected_class_counts,
        "metadata_warnings": metadata_warnings,
        "label_mapping": {
            "0": "dead",
            "1": "alive",
        },
    }

    save_json(metadata, processed_dir / "metadata.json")

    print(f"[BBBC010] Valid samples: {len(samples)}")
    print(f"[BBBC010] Skipped invalid masks: {skipped_invalid}")
    print(f"[BBBC010] Skipped unknown labels: {skipped_label}")
    print(f"[BBBC010] Resampling fallbacks: {fallback_count}")
    print(f"[BBBC010] Dead: {int((sample_labels == 0).sum())}")
    print(f"[BBBC010] Alive: {int((sample_labels == 1).sum())}")
    print(f"[BBBC010] Saved train set: {processed_dir / 'train.pt'}")
    print(f"[BBBC010] Saved test set: {processed_dir / 'test.pt'}")
    print(f"[BBBC010] Saved metadata: {processed_dir / 'metadata.json'}")


def prepare_dataset(config: dict[str, Any]) -> None:
    name = config["dataset"]["name"].lower()

    if name == "mnist":
        prepare_mnist(config)
    elif name == "bbbc010":
        prepare_bbbc010(config)
    else:
        raise ValueError(f"Unknown dataset: {name}")
