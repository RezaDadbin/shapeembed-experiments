import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (
    DATASET,
    LATENT_DIM,
    MATRIX_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    ROOT,
    CHECKPOINT_DIR,
    RESULT_DIR,
    FIGURE_DIR,
    DATA_PROCESSED,
    DEVICE,
    BETA_KL,
    LAMBDA_DIAG,
    LAMBDA_NONNEG,
    LAMBDA_SYM,
)

from dataset import DistanceMatrixDataset
from models import build_model
from losses import compute_loss
from utils import count_parameters
from visualize import save_reconstruction_grid
from report_figures import select_reconstruction_examples


MODEL_TYPES = ["original", "ae", "mlp_decoder"]


def load_trained_model(model_type: str):
    run_name = f"{DATASET}_{model_type}"
    ckpt_path = CHECKPOINT_DIR / f"{run_name}_best.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {ckpt_path}\n"
            f"Train this model first."
        )

    model = build_model(
        model_type=model_type,
        latent_dim=LATENT_DIM,
        matrix_size=MATRIX_SIZE,
    ).to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, ckpt_path


def average_losses(model, model_type: str, split: str):
    dataset = DistanceMatrixDataset(split=split)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    total_logs = {
        "total": 0.0,
        "rec": 0.0,
        "kl": 0.0,
        "diag": 0.0,
        "nonneg": 0.0,
        "sym": 0.0,
    }

    total_samples = 0

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(DEVICE)
            batch_size = x.size(0)

            outputs = model(x)

            _, logs = compute_loss(
                model_type=model_type,
                outputs=outputs,
                beta_kl=BETA_KL,
                lambda_diag=LAMBDA_DIAG,
                lambda_nonneg=LAMBDA_NONNEG,
                lambda_sym=LAMBDA_SYM,
            )

            for key in total_logs:
                total_logs[key] += float(logs.get(key, 0.0)) * batch_size

            total_samples += batch_size

    avg_logs = {
        key: value / max(total_samples, 1)
        for key, value in total_logs.items()
    }

    return avg_logs


def load_classification_metrics(model_type: str):
    run_name = f"{DATASET}_{model_type}"
    metrics_path = RESULT_DIR / f"{run_name}_evaluation.json"

    if not metrics_path.exists():
        return {}

    with open(metrics_path, "r") as f:
        return json.load(f)


def save_model_reconstruction_figure(model, model_type: str, num_examples: int | None = None):
    if num_examples is None:
        num_examples = 6
    input_mats, recon_mats, contour_list = select_reconstruction_examples(
        model, num_examples=num_examples, model_type=model_type
    )

    save_path = FIGURE_DIR / f"{DATASET}_{model_type}_comparison_reconstructions.png"

    save_reconstruction_grid(
        input_matrices=input_mats,
        recon_matrices=recon_mats,
        true_contours=contour_list,
        save_path=save_path,
        model_type=model_type,
    )

    return save_path


def make_comparison():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for model_type in MODEL_TYPES:
        print(f"Processing {model_type}...")

        model, ckpt_path = load_trained_model(model_type)

        params = count_parameters(model)

        train_losses = average_losses(
            model=model,
            model_type=model_type,
            split="train",
        )

        test_losses = average_losses(
            model=model,
            model_type=model_type,
            split="test",
        )

        cls = load_classification_metrics(model_type)

        recon_fig = save_model_reconstruction_figure(
            model=model,
            model_type=model_type,
        )

        row = {
            "dataset": DATASET,
            "model_type": model_type,
            "checkpoint": ckpt_path.relative_to(ROOT).as_posix(),
            "num_parameters": params,

            "train_total_loss": train_losses["total"],
            "train_reconstruction_loss": train_losses["rec"],
            "train_kl_loss": train_losses["kl"],
            "train_diag_loss": train_losses["diag"],
            "train_nonneg_loss": train_losses["nonneg"],
            "train_sym_loss": train_losses["sym"],

            "test_total_loss": test_losses["total"],
            "test_reconstruction_loss": test_losses["rec"],
            "test_kl_loss": test_losses["kl"],
            "test_diag_loss": test_losses["diag"],
            "test_nonneg_loss": test_losses["nonneg"],
            "test_sym_loss": test_losses["sym"],

            "macro_f1_mean": cls.get("macro_f1_mean", ""),
            "macro_f1_std": cls.get("macro_f1_std", ""),
            "accuracy_mean": cls.get("accuracy_mean", ""),
            "accuracy_std": cls.get("accuracy_std", ""),
            "macro_precision_mean": cls.get("macro_precision_mean", ""),
            "macro_precision_std": cls.get("macro_precision_std", ""),
            "macro_recall_mean": cls.get("macro_recall_mean", ""),
            "macro_recall_std": cls.get("macro_recall_std", ""),
            "log_loss_mean": cls.get("log_loss_mean", ""),
            "log_loss_std": cls.get("log_loss_std", ""),

            "with_size_macro_f1_mean": cls.get("with_size_macro_f1_mean", ""),
            "with_size_macro_f1_std": cls.get("with_size_macro_f1_std", ""),
            "with_size_accuracy_mean": cls.get("with_size_accuracy_mean", ""),
            "with_size_accuracy_std": cls.get("with_size_accuracy_std", ""),

            "reconstruction_figure": recon_fig.relative_to(ROOT).as_posix(),
        }

        rows.append(row)

    csv_path = RESULT_DIR / f"{DATASET}_model_comparison.csv"
    json_path = RESULT_DIR / f"{DATASET}_model_comparison.json"

    fieldnames = list(rows[0].keys())

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    print()
    print("Comparison finished.")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")

    print()
    print("Short summary:")
    for row in rows:
        f1 = row["macro_f1_mean"]
        rec = row["test_reconstruction_loss"]
        params = row["num_parameters"]
        print(
            f"{row['model_type']:12s} | "
            f"params={params:,} | "
            f"test_rec={rec:.6f} | "
            f"macro_f1={f1}"
        )


if __name__ == "__main__":
    make_comparison()
