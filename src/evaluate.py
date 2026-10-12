import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    log_loss,
)

from config import (
    DATASET,
    MODEL_TYPE,
    LATENT_DIM,
    MATRIX_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    ROOT,
    CHECKPOINT_DIR,
    LATENT_DIR,
    RESULT_DIR,
    FIGURE_DIR,
    DATA_PROCESSED,
    RUN_NAME,
    DEVICE,
)

from dataset import DistanceMatrixDataset
from models import build_model
from visualize import save_tsne_figure


def get_latent_representation(outputs: dict, model_type: str) -> torch.Tensor:
    """
    For the original model and the MLP-decoder variant, use mu as the descriptor.
    For the AE variant, use deterministic z.
    """

    if model_type in ("original", "mlp_decoder"):
        return outputs["mu"]

    if model_type == "ae":
        return outputs["z"]

    raise ValueError(f"Unknown model_type: {model_type}")


def extract_latents(model, loader, model_type: str, device: str):
    model.eval()

    latents = []
    labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            outputs = model(x)
            z = get_latent_representation(outputs, model_type)

            latents.append(z.cpu().numpy())
            labels.append(y.numpy())

    latents = np.concatenate(latents, axis=0)
    labels = np.concatenate(labels, axis=0)

    return latents, labels


def five_fold_logistic_regression(latents: np.ndarray, labels: np.ndarray):
    """
    Paper-style downstream evaluation:
        logistic regression classifier
        5-fold stratified cross-validation
        mean ± std metrics
    """

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    fold_results = {
        "accuracy": [],
        "macro_f1": [],
        "macro_precision": [],
        "macro_recall": [],
        "log_loss": [],
    }

    for train_idx, test_idx in skf.split(latents, labels):
        X_train, X_test = latents[train_idx], latents[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        clf = LogisticRegression(
            max_iter=3000,
        )

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)

        fold_results["accuracy"].append(
            accuracy_score(y_test, y_pred)
        )

        fold_results["macro_f1"].append(
            f1_score(y_test, y_pred, average="macro")
        )

        fold_results["macro_precision"].append(
            precision_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            )
        )

        fold_results["macro_recall"].append(
            recall_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            )
        )

        fold_results["log_loss"].append(
            log_loss(y_test, y_prob, labels=clf.classes_)
        )

    summary = {}

    for key, values in fold_results.items():
        values = np.asarray(values, dtype=np.float64)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std())

    summary["num_folds"] = 5
    summary["classifier"] = "LogisticRegression"

    return summary


def evaluate():
    LATENT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset = DistanceMatrixDataset(split="train")
    test_dataset = DistanceMatrixDataset(split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = build_model(
        model_type=MODEL_TYPE,
        latent_dim=LATENT_DIM,
        matrix_size=MATRIX_SIZE,
    ).to(DEVICE)

    ckpt_path = CHECKPOINT_DIR / f"{RUN_NAME}_best.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    z_train, y_train = extract_latents(
        model=model,
        loader=train_loader,
        model_type=MODEL_TYPE,
        device=DEVICE,
    )

    z_test, y_test = extract_latents(
        model=model,
        loader=test_loader,
        model_type=MODEL_TYPE,
        device=DEVICE,
    )

    np.savez(
        LATENT_DIR / f"{RUN_NAME}_latents.npz",
        z_train=z_train,
        y_train=y_train,
        z_test=z_test,
        y_test=y_test,
    )

    # -------------------------------------------------------------------------
    # This is the part from 5.3.
    # Replace the old one-split classifier with this 5-fold CV evaluation.
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # Paper-style evaluation:
    # Extract descriptors for all samples and evaluate them with
    # stratified 5-fold logistic regression.
    # -------------------------------------------------------------------------
    z_all = np.concatenate([z_train, z_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)

    metrics = five_fold_logistic_regression(z_all, y_all)

    metrics.update({
        "model_type": MODEL_TYPE,
        "checkpoint": ckpt_path.relative_to(ROOT).as_posix(),
        "protocol": "five-fold classifier CV on pooled frozen train/test embeddings; encoder is not refit per fold",
        "num_samples": int(len(z_all)),
        "latent_dim": int(z_all.shape[1]),
        "uses_size_feature": False,
    })

    # -------------------------------------------------------------------------
    # ShapeEmbed+Sz evaluation for BBBC010.
    #
    # This is NOT a new model.
    # It only concatenates the original pre-normalization distance-matrix norm
    # to the learned latent descriptor during downstream classification.
    # -------------------------------------------------------------------------
    if DATASET == "bbbc010":
        norms_train = torch.load(DATA_PROCESSED / "norms_train.pt").numpy()
        norms_test = torch.load(DATA_PROCESSED / "norms_test.pt").numpy()

        norms_all = np.concatenate([norms_train, norms_test], axis=0)
        norms_all = norms_all.reshape(-1, 1)

        z_all_with_size = np.concatenate([z_all, norms_all], axis=1)

        metrics_with_size = five_fold_logistic_regression(
            z_all_with_size,
            y_all,
        )

        for key, value in metrics_with_size.items():
            metrics[f"with_size_{key}"] = value

        metrics["with_size_latent_dim"] = int(z_all_with_size.shape[1])
        metrics["with_size_uses_size_feature"] = True

    # -------------------------------------------------------------------------
    # Save metrics.
    # -------------------------------------------------------------------------
    with open(RESULT_DIR / f"{RUN_NAME}_evaluation.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # -------------------------------------------------------------------------
    # t-SNE figure.
    # This is only for visualization, not for metric computation.
    # -------------------------------------------------------------------------
    save_tsne_figure(
        latents=z_all,
        labels=y_all,
        save_path=FIGURE_DIR / f"{RUN_NAME}_tsne.png",
        title=f"{RUN_NAME} latent space (t-SNE)",
    )

    print("Evaluation finished.")
    print()
    print("Latent-only evaluation:")
    print(
        f"Macro F1: "
        f"{metrics['macro_f1_mean']:.4f} ± {metrics['macro_f1_std']:.4f}"
    )
    print(
        f"Accuracy: "
        f"{metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}"
    )
    print(
        f"Precision: "
        f"{metrics['macro_precision_mean']:.4f} ± {metrics['macro_precision_std']:.4f}"
    )
    print(
        f"Recall: "
        f"{metrics['macro_recall_mean']:.4f} ± {metrics['macro_recall_std']:.4f}"
    )
    print(
        f"Log-loss: "
        f"{metrics['log_loss_mean']:.4f} ± {metrics['log_loss_std']:.4f}"
    )

    if DATASET == "bbbc010":
        print()
        print("Latent + size evaluation:")
        print(
            f"Macro F1 + Sz: "
            f"{metrics['with_size_macro_f1_mean']:.4f} ± "
            f"{metrics['with_size_macro_f1_std']:.4f}"
        )
        print(
            f"Accuracy + Sz: "
            f"{metrics['with_size_accuracy_mean']:.4f} ± "
            f"{metrics['with_size_accuracy_std']:.4f}"
        )
        print(
            f"Precision + Sz: "
            f"{metrics['with_size_macro_precision_mean']:.4f} ± "
            f"{metrics['with_size_macro_precision_std']:.4f}"
        )
        print(
            f"Recall + Sz: "
            f"{metrics['with_size_macro_recall_mean']:.4f} ± "
            f"{metrics['with_size_macro_recall_std']:.4f}"
        )
        print(
            f"Log-loss + Sz: "
            f"{metrics['with_size_log_loss_mean']:.4f} ± "
            f"{metrics['with_size_log_loss_std']:.4f}"
        )

    print()
    print(f"Saved metrics to: {RESULT_DIR / f'{RUN_NAME}_evaluation.json'}")
    print(f"Saved t-SNE to: {FIGURE_DIR / f'{RUN_NAME}_tsne.png'}")

if __name__ == "__main__":
    evaluate()
