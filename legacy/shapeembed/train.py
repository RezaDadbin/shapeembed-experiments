from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from shapeembed.datasets import DistanceMatrixDataset
from shapeembed.losses import shapeembed_ae_loss, shapeembed_vae_loss
from shapeembed.models.factory import (
    build_model,
    get_checkpoint_path,
    get_latest_checkpoint_path,
    get_model_type,
    get_run_name,
)
from shapeembed.utils import (
    count_module_parameters,
    count_parameters,
    ensure_dir,
    get_device,
    prepare_distance_batch,
    save_json,
)


def is_official_phase1(model_type: str) -> bool:
    return model_type in ["shapeembed", "vae"]


def _to_float_dict(loss_dict: dict[str, torch.Tensor]) -> dict[str, float]:
    return {k: float(v.detach().cpu().item()) for k, v in loss_dict.items()}


def get_parameter_summary(model: torch.nn.Module) -> dict[str, int | None]:
    encoder = getattr(model, "encoder", None)
    decoder = getattr(model, "decoder", None)

    return {
        "total_parameters": count_parameters(model),
        "encoder_parameters": count_module_parameters(encoder),
        "decoder_parameters": count_module_parameters(decoder),
    }


def compute_model_loss(
    model: torch.nn.Module,
    model_type: str,
    D: torch.Tensor,
    loss_weights: dict[str, Any],
) -> dict[str, torch.Tensor]:
    if is_official_phase1(model_type):
        preproc_x, recon_x, _, z_mean, z_log_var, _ = model(D)

        if recon_x.shape != preproc_x.shape:
            raise RuntimeError(
                f"Model output shape {tuple(recon_x.shape)} does not match "
                f"preprocessed target shape {tuple(preproc_x.shape)}"
            )

        loss_dict = shapeembed_vae_loss(
            D_hat=recon_x,
            D=preproc_x,
            mu=z_mean,
            logvar=z_log_var,
            beta=loss_weights["beta"],
            gamma=loss_weights["gamma"],
            delta=loss_weights["delta"],
            epsilon=loss_weights["epsilon"],
            reconstruction_loss=loss_weights["reconstruction_loss"],
        )
        loss_dict["beta_kl"] = loss_weights["beta"] * loss_dict["kl"]
        loss_dict["dm"] = (
            loss_weights["gamma"] * loss_dict["diag"]
            + loss_weights["delta"] * loss_dict["nonneg"]
            + loss_weights["epsilon"] * loss_dict["sym"]
        )
        _assert_finite_losses(loss_dict)
        return loss_dict

    if model_type == "ae":
        D_hat, _ = model(D)

        if D_hat.shape != D.shape:
            raise RuntimeError(
                f"Model output shape {tuple(D_hat.shape)} does not match "
                f"target shape {tuple(D.shape)}"
            )

        loss_dict = shapeembed_ae_loss(
            D_hat=D_hat,
            D=D,
            gamma=loss_weights["gamma"],
            delta=loss_weights["delta"],
            epsilon=loss_weights["epsilon"],
            reconstruction_loss=loss_weights["reconstruction_loss"],
        )
        loss_dict["beta_kl"] = loss_weights["beta"] * loss_dict["kl"]
        loss_dict["dm"] = (
            loss_weights["gamma"] * loss_dict["diag"]
            + loss_weights["delta"] * loss_dict["nonneg"]
            + loss_weights["epsilon"] * loss_dict["sym"]
        )
        _assert_finite_losses(loss_dict)
        return loss_dict

    D_hat, mu, logvar, _ = model(D)

    if D_hat.shape != D.shape:
        raise RuntimeError(
            f"Model output shape {tuple(D_hat.shape)} does not match "
            f"target shape {tuple(D.shape)}"
        )

    loss_dict = shapeembed_vae_loss(
        D_hat=D_hat,
        D=D,
        mu=mu,
        logvar=logvar,
        beta=loss_weights["beta"],
        gamma=loss_weights["gamma"],
        delta=loss_weights["delta"],
        epsilon=loss_weights["epsilon"],
        reconstruction_loss=loss_weights["reconstruction_loss"],
    )
    loss_dict["beta_kl"] = loss_weights["beta"] * loss_dict["kl"]
    loss_dict["dm"] = (
        loss_weights["gamma"] * loss_dict["diag"]
        + loss_weights["delta"] * loss_dict["nonneg"]
        + loss_weights["epsilon"] * loss_dict["sym"]
    )
    _assert_finite_losses(loss_dict)
    return loss_dict


def _assert_finite_losses(loss_dict: dict[str, torch.Tensor]) -> None:
    for name, value in loss_dict.items():
        if not torch.isfinite(value).all():
            raise RuntimeError(f"Non-finite loss component {name}: {value}")


def verify_official_phase1_model(model: torch.nn.Module) -> None:
    print(f"[train] model class: {model.__class__.__name__}")
    print(f"[train] model.preproc: {model.preproc}")
    print(f"[train] encoder is my_resnet18: {model.encoder.__class__.__name__}")
    print(f"[train] z_mean.in_features: {model.z_mean.in_features}")
    print(f"[train] decoder seed view: {model.decoder[1]}")
    print(f"[train] first conv: {model.encoder.conv1}")
    print(f"[train] maxpool: {model.encoder.maxpool}")
    print("[train] phase1 target: preproc_x from model.forward, not externally scaled D")

    if model.z_mean.in_features != 512:
        raise RuntimeError("Official ShapeEmbed z_mean must receive 512 features.")
    if model.z_log_var.in_features != 512:
        raise RuntimeError("Official ShapeEmbed z_log_var must receive 512 features.")
    if not isinstance(model.encoder.fc, torch.nn.Identity):
        raise RuntimeError("Official ShapeEmbed encoder fc must be nn.Identity.")


def train_one_epoch(
    model: torch.nn.Module,
    model_type: str,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_weights: dict[str, Any],
    config: dict[str, Any],
    gradient_clip_norm: float | None,
) -> dict[str, float]:
    model.train()

    running = {
        "total": 0.0,
        "rec": 0.0,
        "kl": 0.0,
        "beta_kl": 0.0,
        "dm": 0.0,
        "diag": 0.0,
        "nonneg": 0.0,
        "sym": 0.0,
    }

    n_batches = 0

    for batch in tqdm(loader, desc="train", leave=False):
        D = batch["D"].to(device)

        if not is_official_phase1(model_type):
            D = prepare_distance_batch(D, config)

        optimizer.zero_grad(set_to_none=True)

        loss_dict = compute_model_loss(
            model=model,
            model_type=model_type,
            D=D,
            loss_weights=loss_weights,
        )

        loss_dict["total"].backward()

        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip_norm,
            )

        optimizer.step()

        float_losses = _to_float_dict(loss_dict)

        for k in running:
            running[k] += float_losses.get(k, 0.0)

        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in running.items()}


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    model_type: str,
    loader: DataLoader,
    device: torch.device,
    loss_weights: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, float]:
    model.eval()

    running = {
        "total": 0.0,
        "rec": 0.0,
        "kl": 0.0,
        "beta_kl": 0.0,
        "dm": 0.0,
        "diag": 0.0,
        "nonneg": 0.0,
        "sym": 0.0,
    }

    n_batches = 0

    for batch in tqdm(loader, desc="val", leave=False):
        D = batch["D"].to(device)

        if not is_official_phase1(model_type):
            D = prepare_distance_batch(D, config)

        loss_dict = compute_model_loss(
            model=model,
            model_type=model_type,
            D=D,
            loss_weights=loss_weights,
        )

        float_losses = _to_float_dict(loss_dict)

        for k in running:
            running[k] += float_losses.get(k, 0.0)

        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in running.items()}


def get_early_stopping_config(config: dict[str, Any]) -> dict[str, Any]:
    es = config["training"].get("early_stopping", {})

    return {
        "enabled": bool(es.get("enabled", False)),
        "monitor": str(es.get("monitor", "val_total")),
        "min_epochs": int(es.get("min_epochs", 100)),
        "patience": int(es.get("patience", 30)),
        "min_delta": float(es.get("min_delta", 1e-8)),
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    epoch: int,
    best_val: float,
    best_epoch: int,
    model_type: str,
    run_name: str,
) -> None:
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config,
        "epoch": epoch,
        "best_val": best_val,
        "best_epoch": best_epoch,
        "model_type": model_type,
        "run_name": run_name,
    }

    required_keys = {
        "model_state",
        "optimizer_state",
        "epoch",
        "best_val",
        "best_epoch",
    }
    missing = required_keys - checkpoint.keys()

    if missing:
        raise RuntimeError(f"Checkpoint payload missing required keys: {sorted(missing)}")

    torch.save(checkpoint, path)


def _summarize_adam_buffers(optimizer: torch.optim.Optimizer) -> tuple[int, int]:
    """
    Returns:
        num_state_entries, num_entries_with_exp_avg_and_exp_avg_sq
    """
    num_state_entries = len(optimizer.state)
    num_complete_adam_entries = 0

    for state in optimizer.state.values():
        if "exp_avg" in state and "exp_avg_sq" in state:
            num_complete_adam_entries += 1

    return num_state_entries, num_complete_adam_entries


def _restore_optimizer_state(
    optimizer: torch.optim.Optimizer,
    checkpoint: dict[str, Any],
) -> None:
    if "optimizer_state" not in checkpoint:
        print(
            "Warning: optimizer state missing; resuming model weights with fresh Adam, "
            "not an exact continuation."
        )
        return

    fresh_optimizer_state = optimizer.state_dict()

    try:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    except Exception as exc:
        try:
            optimizer.load_state_dict(fresh_optimizer_state)
        except Exception:
            optimizer.state.clear()

        print(
            "Warning: optimizer state could not be loaded; resuming model weights "
            f"with fresh Adam, not an exact continuation. Error: {exc}"
        )
        return

    num_state_entries, num_complete_adam_entries = _summarize_adam_buffers(optimizer)

    if num_state_entries == 0:
        print(
            "Warning: optimizer state loaded but contains no Adam momentum buffers; "
            "this is not an exact continuation."
        )
        return

    if num_complete_adam_entries != num_state_entries:
        print(
            "Warning: optimizer state loaded, but some Adam exp_avg/exp_avg_sq "
            f"buffers are missing ({num_complete_adam_entries}/{num_state_entries})."
        )
        return

    print(
        "[train] restored Adam optimizer state: "
        f"{num_complete_adam_entries}/{num_state_entries} parameter states include "
        "exp_avg and exp_avg_sq."
    )


def maybe_resume(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    latest_path: Path,
    best_path: Path,
    device: torch.device,
    resume: bool,
) -> tuple[int, float, int]:
    """
    Returns:
        start_epoch, best_val, best_epoch
    """
    if not resume:
        return 1, float("inf"), 0

    resume_path = None

    if latest_path.exists():
        resume_path = latest_path
    elif best_path.exists():
        resume_path = best_path

    if resume_path is None:
        print("[train] --resume requested, but no checkpoint found. Starting from scratch.")
        return 1, float("inf"), 0

    checkpoint = torch.load(resume_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state"])
    _restore_optimizer_state(optimizer, checkpoint)

    last_epoch = int(checkpoint.get("epoch", 0))
    best_val = float(checkpoint.get("best_val", float("inf")))
    best_epoch = int(checkpoint.get("best_epoch", last_epoch))

    start_epoch = last_epoch + 1

    print(f"[train] resumed from: {resume_path}")
    print(f"[train] start epoch: {start_epoch}")
    print(f"[train] best epoch so far: {best_epoch}")
    print(f"[train] best val so far: {best_val:.8f}")

    return start_epoch, best_val, best_epoch


def train_model(
    config: dict[str, Any],
    debug: bool = False,
    resume: bool = False,
) -> None:
    dataset_name = config["dataset"]["name"]
    model_type = get_model_type(config)
    run_name = get_run_name(config)

    if debug:
        run_name = f"{run_name}_debug"
        resume = False

    processed_dir = Path(config["dataset"]["processed_dir"])

    train_path = processed_dir / "train.pt"
    # Model selection uses a held-out portion of training data, never test.pt.
    from sklearn.model_selection import train_test_split
    full_train = DistanceMatrixDataset(train_path)
    indices = list(range(len(full_train)))
    labels = [int(full_train[i]["label"]) for i in indices]
    fraction = float(config["training"].get("validation_fraction", 0.15))
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    train_indices, val_indices = train_test_split(
        indices, test_size=fraction, stratify=labels,
        random_state=int(config["dataset"]["seed"]),
    )
    train_dataset = Subset(full_train, train_indices)
    val_dataset = Subset(full_train, val_indices)

    if debug:
        train_dataset = Subset(train_dataset, list(range(min(512, len(train_dataset)))))
        val_dataset = Subset(val_dataset, list(range(min(128, len(val_dataset)))))
        epochs = 3
        print("[debug] using small subset and 3 epochs")
    else:
        epochs = int(config["training"]["epochs"])

    device = get_device(config["training"]["device"])

    print(f"[train] dataset: {dataset_name}")
    print(f"[train] model type: {model_type}")
    print(f"[train] run name: {run_name}")
    print(f"[train] device: {device}")

    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"]["num_workers"])

    if device.type == "mps":
        num_workers = 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=is_official_phase1(model_type),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = build_model(config).to(device)
    parameter_summary = get_parameter_summary(model)
    print(f"[train] parameters: {parameter_summary['total_parameters']:,}")

    if parameter_summary["encoder_parameters"] is not None:
        print(f"[train] encoder parameters: {parameter_summary['encoder_parameters']:,}")
    if parameter_summary["decoder_parameters"] is not None:
        print(f"[train] decoder parameters: {parameter_summary['decoder_parameters']:,}")

    if is_official_phase1(model_type):
        verify_official_phase1_model(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["lr"]),
    )

    loss_weights = {
        "beta": float(config["training"]["beta"]),
        "gamma": float(config["training"]["gamma"]),
        "delta": float(config["training"]["delta"]),
        "epsilon": float(config["training"]["epsilon"]),
        "reconstruction_loss": str(
            config["training"].get("reconstruction_loss", "l1")
        ),
        "indexation_invariant_loss": bool(
            config["training"].get("indexation_invariant_loss", True)
        ),
        "reflection_invariant_loss": bool(
            config["training"].get("reflection_invariant_loss", True)
        ),
    }

    gradient_clip_raw = config["training"].get("gradient_clip_norm", None)
    gradient_clip_norm = (
        None if gradient_clip_raw is None else float(gradient_clip_raw)
    )

    early_stopping = get_early_stopping_config(config)

    if debug:
        early_stopping["enabled"] = False

    print(f"[train] early stopping: {early_stopping}")
    print(f"[train] gradient_clip_norm: {gradient_clip_norm}")

    checkpoint_dir = ensure_dir(config["outputs"]["checkpoint_dir"])
    log_dir = ensure_dir(config["outputs"]["log_dir"])
    metric_dir = ensure_dir(config["outputs"]["metric_dir"])

    if debug:
        best_path = checkpoint_dir / f"{run_name}.pt"
        latest_path = checkpoint_dir / f"{run_name}_latest.pt"
    else:
        best_path = get_checkpoint_path(config)
        latest_path = get_latest_checkpoint_path(config)

    log_path = log_dir / f"{run_name}_train_log.csv"
    summary_path = metric_dir / f"{run_name}_train_summary.json"

    start_epoch, best_val, best_epoch = maybe_resume(
        model=model,
        optimizer=optimizer,
        latest_path=latest_path,
        best_path=best_path,
        device=device,
        resume=resume,
    )

    if start_epoch > epochs:
        print(f"[train] checkpoint already reached epoch {start_epoch - 1}. Nothing to do.")
        return

    bad_epochs = 0
    stopped_early = False
    stopped_epoch = epochs

    monitor_key = early_stopping["monitor"]
    min_epochs = early_stopping["min_epochs"]
    patience = early_stopping["patience"]
    min_delta = early_stopping["min_delta"]

    fieldnames = [
        "epoch",
        "train_total",
        "train_rec",
        "train_kl",
        "train_beta_kl",
        "train_dm",
        "train_diag",
        "train_nonneg",
        "train_sym",
        "val_total",
        "val_rec",
        "val_kl",
        "val_beta_kl",
        "val_dm",
        "val_diag",
        "val_nonneg",
        "val_sym",
        "best_val",
        "best_epoch",
        "bad_epochs",
    ]

    write_header = True

    if resume and log_path.exists() and start_epoch > 1:
        write_header = False
        log_mode = "a"
    else:
        log_mode = "w"

    with log_path.open(log_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        for epoch in range(start_epoch, epochs + 1):
            print(f"\n[{run_name}] epoch {epoch}/{epochs}")

            train_losses = train_one_epoch(
                model=model,
                model_type=model_type,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                loss_weights=loss_weights,
                config=config,
                gradient_clip_norm=gradient_clip_norm,
            )

            val_losses = validate_one_epoch(
                model=model,
                model_type=model_type,
                loader=val_loader,
                device=device,
                loss_weights=loss_weights,
                config=config,
            )

            monitored_name = monitor_key.replace("val_", "")
            monitored_value = val_losses[monitored_name]

            improved = monitored_value < (best_val - min_delta)

            if improved:
                best_val = monitored_value
                best_epoch = epoch
                bad_epochs = 0

                save_checkpoint(
                    path=best_path,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    epoch=epoch,
                    best_val=best_val,
                    best_epoch=best_epoch,
                    model_type=model_type,
                    run_name=run_name,
                )

                print(f"[train] saved best checkpoint: {best_path}")
            else:
                if epoch >= min_epochs:
                    bad_epochs += 1

            save_checkpoint(
                path=latest_path,
                model=model,
                optimizer=optimizer,
                config=config,
                epoch=epoch,
                best_val=best_val,
                best_epoch=best_epoch,
                model_type=model_type,
                run_name=run_name,
            )

            row = {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_losses.items()},
                **{f"val_{k}": v for k, v in val_losses.items()},
                "best_val": best_val,
                "best_epoch": best_epoch,
                "bad_epochs": bad_epochs,
            }

            writer.writerow(row)
            f.flush()

            print(
                f"train_total={train_losses['total']:.6f} "
                f"train_rec={train_losses['rec']:.6f} | "
                f"val_total={val_losses['total']:.6f} "
                f"val_rec={val_losses['rec']:.6f} | "
                f"best_val={best_val:.8f} "
                f"best_epoch={best_epoch}"
            )

            if (
                early_stopping["enabled"]
                and epoch >= min_epochs
                and bad_epochs >= patience
            ):
                stopped_early = True
                stopped_epoch = epoch
                print(
                    f"[train] early stopping triggered at epoch {epoch}. "
                    f"Best epoch: {best_epoch}, best {monitor_key}: {best_val:.8f}"
                )
                break

    save_json(
        {
            "dataset": dataset_name,
            "model_type": model_type,
            "run_name": run_name,
            "max_epochs": epochs,
            "stopped_epoch": stopped_epoch,
            "stopped_early": stopped_early,
            "best_epoch": best_epoch,
            "best_val_total": best_val,
            "best_checkpoint": str(best_path),
            "latest_checkpoint": str(latest_path),
            "log": str(log_path),
            "total_parameters": parameter_summary["total_parameters"],
            "encoder_parameters": parameter_summary["encoder_parameters"],
            "decoder_parameters": parameter_summary["decoder_parameters"],
            "gradient_clip_norm": gradient_clip_norm,
            "debug": debug,
            "early_stopping": early_stopping,
            "resume_used": resume,
        },
        summary_path,
    )

    print("[train] finished.")
    print(f"[train] best epoch: {best_epoch}")
    print(f"[train] best val total: {best_val:.8f}")
    print(f"[train] best checkpoint: {best_path}")
    print(f"[train] latest checkpoint: {latest_path}")
    print(f"[train] log saved to: {log_path}")
