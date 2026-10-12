from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from shapeembed.encode import encode_dataset
from shapeembed.models.factory import get_run_name
from shapeembed.utils import ensure_dir, save_json


def _classifier_classes(classifier):
    classes = getattr(classifier, "classes_", None)

    if classes is not None:
        return classes

    try:
        return classifier[-1].classes_
    except Exception:
        return None


def evaluate_classifier(classifier, name, X, y, idx_splits):
    all_accuracy = []
    micro_precision = []
    macro_precision = []
    weighted_precision = []
    micro_recall = []
    macro_recall = []
    weighted_recall = []
    micro_f1 = []
    macro_f1 = []
    weighted_f1 = []
    all_log_loss = []
    all_y_test = []
    all_y_pred = []

    for train_index, test_index in tqdm.tqdm(
        idx_splits,
        desc=name,
        position=1,
        leave=True,
    ):
        X_train, X_test = np.take(X, train_index, axis=0), np.take(X, test_index, axis=0)
        y_train, y_test = np.take(y, train_index, axis=0), np.take(y, test_index, axis=0)

        classifier.fit(X_train, y_train)
        all_y_test.extend(y_test)
        y_pred = classifier.predict(X_test)
        y_prob = classifier.predict_proba(X_test)
        all_y_pred.extend(y_pred)
        classes = _classifier_classes(classifier)

        all_accuracy.append(accuracy_score(y_test, y_pred))
        micro_precision.append(precision_score(y_test, y_pred, average="micro"))
        macro_precision.append(precision_score(y_test, y_pred, average="macro"))
        weighted_precision.append(precision_score(y_test, y_pred, average="weighted"))
        micro_recall.append(recall_score(y_test, y_pred, average="micro"))
        macro_recall.append(recall_score(y_test, y_pred, average="macro"))
        weighted_recall.append(recall_score(y_test, y_pred, average="weighted"))
        micro_f1.append(f1_score(y_test, y_pred, average="micro"))
        macro_f1.append(f1_score(y_test, y_pred, average="macro"))
        weighted_f1.append(f1_score(y_test, y_pred, average="weighted"))
        all_log_loss.append(log_loss(y_test, y_prob, labels=classes))

    rpt = classification_report(all_y_test, all_y_pred)
    cm = confusion_matrix(all_y_test, all_y_pred)

    return (
        all_accuracy,
        micro_precision,
        macro_precision,
        weighted_precision,
        micro_recall,
        macro_recall,
        weighted_recall,
        micro_f1,
        macro_f1,
        weighted_f1,
        all_log_loss,
        rpt,
        cm,
    )


def run_classification(X, y, n_splits=None, logreg_maxiter=200):
    classifiers = {
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                solver="lbfgs",
                max_iter=logreg_maxiter,
            ),
        ),
        "RandomForest": RandomForestClassifier(random_state=42),
    }
    kf = StratifiedKFold(
        n_splits=n_splits if n_splits else len(np.unique(y)),
        shuffle=True,
        random_state=42,
    )
    idx_split = list(kf.split(X, y))
    eval_results = {
        nm: evaluate_classifier(clf, nm, X, y, idx_split)
        for nm, clf in classifiers.items()
    }

    summary = {}

    for nm, values in eval_results.items():
        (
            accs,
            mi_precs,
            ma_precs,
            w_precs,
            mi_recs,
            ma_recs,
            w_recs,
            mi_f1s,
            ma_f1s,
            w_f1s,
            loglosses,
            rpt,
            cm,
        ) = values

        summary[nm] = {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "micro_precision_mean": float(np.mean(mi_precs)),
            "micro_precision_std": float(np.std(mi_precs)),
            "macro_precision_mean": float(np.mean(ma_precs)),
            "macro_precision_std": float(np.std(ma_precs)),
            "weighted_precision_mean": float(np.mean(w_precs)),
            "weighted_precision_std": float(np.std(w_precs)),
            "micro_recall_mean": float(np.mean(mi_recs)),
            "micro_recall_std": float(np.std(mi_recs)),
            "macro_recall_mean": float(np.mean(ma_recs)),
            "macro_recall_std": float(np.std(ma_recs)),
            "weighted_recall_mean": float(np.mean(w_recs)),
            "weighted_recall_std": float(np.std(w_recs)),
            "micro_f1_mean": float(np.mean(mi_f1s)),
            "micro_f1_std": float(np.std(mi_f1s)),
            "macro_f1_mean": float(np.mean(ma_f1s)),
            "macro_f1_std": float(np.std(ma_f1s)),
            "weighted_f1_mean": float(np.mean(w_f1s)),
            "weighted_f1_std": float(np.std(w_f1s)),
            "log_loss_mean": float(np.mean(loglosses)),
            "log_loss_std": float(np.std(loglosses)),
            "classification_report": rpt,
            "confusion_matrix": cm.tolist(),
        }

    return summary


def _legacy_logistic_metrics(official_results: dict[str, Any]) -> dict[str, Any]:
    logreg = official_results["LogisticRegression"]
    return {
        "accuracy": {
            "mean": logreg["accuracy_mean"],
            "std": logreg["accuracy_std"],
        },
        "f1_macro": {
            "mean": logreg["macro_f1_mean"],
            "std": logreg["macro_f1_std"],
        },
        "precision_macro": {
            "mean": logreg["macro_precision_mean"],
            "std": logreg["macro_precision_std"],
        },
        "recall_macro": {
            "mean": logreg["macro_recall_mean"],
            "std": logreg["macro_recall_std"],
        },
        "log_loss": {
            "mean": logreg["log_loss_mean"],
            "std": logreg["log_loss_std"],
        },
    }


def evaluate_embeddings(
    config: dict[str, Any],
    use_size: bool = False,
    force_reencode: bool = True,
    checkpoint: str | None = None,
) -> None:
    run_name = get_run_name(config)
    checkpoint = str(
        checkpoint or config.get("evaluation", {}).get("checkpoint", "best")
    ).lower()

    if checkpoint not in ["best", "latest"]:
        raise ValueError(f"Unknown evaluation checkpoint: {checkpoint}")

    embedding_dir = Path(config["outputs"]["embedding_dir"])
    metric_dir = ensure_dir(config["outputs"]["metric_dir"])

    emb_path = embedding_dir / f"{run_name}_embeddings.npy"
    label_path = embedding_dir / f"{run_name}_labels.npy"
    norm_path = embedding_dir / f"{run_name}_norms.npy"

    missing_embeddings = not emb_path.exists() or not label_path.exists()

    if force_reencode:
        print(f"[evaluate] re-encoding embeddings from {checkpoint} checkpoint...")
        encode_dataset(config, checkpoint=checkpoint)
    elif missing_embeddings:
        print("[evaluate] embeddings not found. Encoding first...")
        encode_dataset(config, checkpoint=checkpoint)

    X = np.load(emb_path)
    y = np.load(label_path)

    if use_size:
        norms = np.load(norm_path).reshape(-1, 1)
        X = np.concatenate([X, norms], axis=1)

    evaluation_cfg = config.get("evaluation", {})
    n_splits = int(evaluation_cfg.get("n_splits", 5))
    logreg_max_iter = int(evaluation_cfg.get("logreg_max_iter", 5000))

    official_results = run_classification(
        X,
        y,
        n_splits=n_splits,
        logreg_maxiter=logreg_max_iter,
    )

    suffix = "with_size" if use_size else "shape_only"
    out_path = metric_dir / f"{run_name}_classification_{suffix}.json"

    summary = {
        "run_name": run_name,
        "use_size": use_size,
        "force_reencode": force_reencode,
        "checkpoint": checkpoint,
        "embedding_source": config.get("evaluation", {}).get("embedding_source", "z"),
        "num_samples": int(len(y)),
        "num_features": int(X.shape[1]),
        "official_classification": official_results,
        "metrics": _legacy_logistic_metrics(official_results),
        "headline": {
            "classifier": "LogisticRegression",
            "metric": "macro_f1_mean",
            "value": official_results["LogisticRegression"]["macro_f1_mean"],
        },
    }

    save_json(summary, out_path)

    if not out_path.exists():
        raise RuntimeError(f"Evaluation JSON was not saved: {out_path}")

    logreg = official_results["LogisticRegression"]
    print("\n[evaluate] LogisticRegression summary")
    print(f"accuracy: {logreg['accuracy_mean']:.4f} +/- {logreg['accuracy_std']:.4f}")
    print(f"macro_f1: {logreg['macro_f1_mean']:.4f} +/- {logreg['macro_f1_std']:.4f}")
    print(f"log_loss: {logreg['log_loss_mean']:.4f} +/- {logreg['log_loss_std']:.4f}")
    print(f"[evaluate] saved: {out_path}")


def evaluate_size_only(
    config: dict[str, Any],
    force_reencode: bool = True,
    checkpoint: str | None = None,
) -> None:
    run_name = get_run_name(config)
    checkpoint = str(
        checkpoint or config.get("evaluation", {}).get("checkpoint", "best")
    ).lower()

    if checkpoint not in ["best", "latest"]:
        raise ValueError(f"Unknown evaluation checkpoint: {checkpoint}")

    embedding_dir = Path(config["outputs"]["embedding_dir"])
    metric_dir = ensure_dir(config["outputs"]["metric_dir"])

    label_path = embedding_dir / f"{run_name}_labels.npy"
    norm_path = embedding_dir / f"{run_name}_norms.npy"

    missing = not label_path.exists() or not norm_path.exists()

    if missing:
        print("[evaluate] saved norms not found. Loading norms from processed data...")
        processed_dir = Path(config["dataset"]["processed_dir"])
        samples = []

        for split_name in ["train.pt", "test.pt"]:
            samples.extend(
                torch.load(processed_dir / split_name, weights_only=False)
            )

        y = np.asarray([sample["label"] for sample in samples])
        norms = np.asarray([sample["norm"] for sample in samples], dtype=np.float32)
        np.save(label_path, y)
        np.save(norm_path, norms)
    else:
        y = np.load(label_path)
        norms = np.load(norm_path)

    X = norms.reshape(-1, 1)

    evaluation_cfg = config.get("evaluation", {})
    n_splits = int(evaluation_cfg.get("n_splits", 5))
    logreg_max_iter = int(evaluation_cfg.get("logreg_max_iter", 5000))

    official_results = run_classification(
        X,
        y,
        n_splits=n_splits,
        logreg_maxiter=logreg_max_iter,
    )

    out_path = metric_dir / f"{run_name}_classification_size_only.json"

    summary = {
        "run_name": run_name,
        "mode": "size_only",
        "force_reencode": force_reencode,
        "checkpoint": checkpoint,
        "num_samples": int(len(y)),
        "num_features": int(X.shape[1]),
        "official_classification": official_results,
        "metrics": _legacy_logistic_metrics(official_results),
        "headline": {
            "classifier": "LogisticRegression",
            "metric": "macro_f1_mean",
            "value": official_results["LogisticRegression"]["macro_f1_mean"],
        },
    }

    save_json(summary, out_path)

    logreg = official_results["LogisticRegression"]
    print("\n[evaluate] LogisticRegression size-only summary")
    print(f"accuracy: {logreg['accuracy_mean']:.4f} +/- {logreg['accuracy_std']:.4f}")
    print(f"macro_f1: {logreg['macro_f1_mean']:.4f} +/- {logreg['macro_f1_std']:.4f}")
    print(f"log_loss: {logreg['log_loss_mean']:.4f} +/- {logreg['log_loss_std']:.4f}")
    print(f"[evaluate] saved: {out_path}")
