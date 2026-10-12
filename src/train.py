import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from config import (
    MODEL_TYPE,
    LATENT_DIM,
    MATRIX_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    EPOCHS,
    LR,
    BETA_KL,
    LAMBDA_DIAG,
    LAMBDA_NONNEG,
    LAMBDA_SYM,
    CHECKPOINT_DIR,
    RESULT_DIR,
    RUN_NAME,
    DEVICE,
    SEED,
)

from dataset import DistanceMatrixDataset
from models import build_model
from losses import compute_loss
from utils import set_seed, count_parameters


def train():
    set_seed(SEED)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset = DistanceMatrixDataset(split="train")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.startswith("cuda"),
        drop_last=False,
    )

    model = build_model(
        model_type=MODEL_TYPE,
        latent_dim=LATENT_DIM,
        matrix_size=MATRIX_SIZE,
    ).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=LR)

    print(f"Model type: {MODEL_TYPE}")
    print(f"Device: {DEVICE}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Parameters: {count_parameters(model):,}")

    history = []

    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        epoch_logs = {
            "total": 0.0,
            "rec": 0.0,
            "kl": 0.0,
            "diag": 0.0,
            "nonneg": 0.0,
            "sym": 0.0,
        }

        num_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:03d}/{EPOCHS}",
            leave=False,
        )

        for x, _ in pbar:
            x = x.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(x)

            loss, logs = compute_loss(
                model_type=MODEL_TYPE,
                outputs=outputs,
                beta_kl=BETA_KL,
                lambda_diag=LAMBDA_DIAG,
                lambda_nonneg=LAMBDA_NONNEG,
                lambda_sym=LAMBDA_SYM,
            )

            loss.backward()
            optimizer.step()

            for key, value in logs.items():
                epoch_logs[key] += value

            num_batches += 1

            pbar.set_postfix({
                "loss": f"{logs['total']:.6f}",
                "rec": f"{logs['rec']:.6f}",
            })

        for key in epoch_logs:
            epoch_logs[key] /= max(num_batches, 1)

        epoch_logs["epoch"] = epoch
        history.append(epoch_logs)

        print(
            f"Epoch {epoch:03d} | "
            f"total={epoch_logs['total']:.6f} | "
            f"rec={epoch_logs['rec']:.6f} | "
            f"kl={epoch_logs['kl']:.6f} | "
            f"diag={epoch_logs['diag']:.6f} | "
            f"nonneg={epoch_logs['nonneg']:.6f} | "
            f"sym={epoch_logs['sym']:.6f}"
        )

        latest_path = CHECKPOINT_DIR / f"{RUN_NAME}_latest.pt"

        torch.save(
            {
                "epoch": epoch,
                "model_type": MODEL_TYPE,
                "matrix_size": MATRIX_SIZE,
                "latent_dim": LATENT_DIM,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_logs["total"],
            },
            latest_path,
        )

        if epoch_logs["total"] < best_loss:
            best_loss = epoch_logs["total"]

            best_path = CHECKPOINT_DIR / f"{RUN_NAME}_best.pt"

            torch.save(
                {
                    "epoch": epoch,
                    "model_type": MODEL_TYPE,
                    "matrix_size": MATRIX_SIZE,
                    "latent_dim": LATENT_DIM,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                },
                best_path,
            )

    history_path = RESULT_DIR / f"{RUN_NAME}_training_history.json"

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print()
    print(f"Training finished.")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Saved checkpoint: {CHECKPOINT_DIR / f'{RUN_NAME}_best.pt'}")
    print(f"Saved history: {history_path}")


if __name__ == "__main__":
    train()
