#!/usr/bin/env python3
"""
Unified entry point for the ShapeEmbed reproduction.

One codebase, two internal implementations sharing a single ``outputs/`` and
``data/`` tree:

  * MNIST + the original VAE ("vae") or the standard AE ("ae")  -> the *legacy*
    implementation in ``legacy/shapeembed/`` (its config dicts live in
    ``src/config.py`` as ``LEGACY_MNIST_VAE_CONFIG`` / ``LEGACY_MNIST_AE_CONFIG``);
    its trained checkpoints (``mnist_shapeembed.pt`` / ``mnist_ae.pt``) were
    produced with this code, so its model architecture is retained.
  * everything else (BBBC010 with any variant, MNIST with the MLP decoder) ->
    the *main* implementation in ``src/`` (config = ``src/config.py``).

Usage:
    python run.py --dataset {mnist,bbbc010} --model {vae,ae,mlp_decoder} \
                  --mode {organize,preprocess,train,evaluate,figures,interpolate,compare} \
                  [--device {cpu,mps,cuda,auto}]

Examples:
    python run.py --dataset bbbc010 --model vae         --mode preprocess
    python run.py --dataset bbbc010 --model vae         --mode train
    python run.py --dataset bbbc010 --model vae         --mode evaluate
    python run.py --dataset bbbc010 --model vae         --mode figures
    python run.py --dataset bbbc010 --model vae         --mode interpolate
    python run.py --dataset bbbc010 --model ae          --mode figures
    python run.py --dataset bbbc010 --model mlp_decoder --mode figures
    python run.py --dataset bbbc010 --model vae         --mode compare        # cross-model (needs all 3 bbbc010 ckpts)
    python run.py --dataset mnist   --model mlp_decoder --mode figures
    python run.py --dataset mnist   --model vae         --mode preprocess     # -> legacy path
    python run.py --dataset mnist   --model vae         --mode figures        # -> legacy path (recon + latent t-SNE + curves)
    python run.py --dataset mnist   --model vae         --mode interpolate    # -> legacy path
    python run.py --dataset mnist   --model ae          --mode figures        # -> legacy path
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
LEGACY = HERE / "legacy"
sys.path.insert(0, str(SRC))                  # the main implementation (config.py, models.py, ...)
sys.path.insert(0, str(LEGACY))               # the `shapeembed` reference package (MNIST vae/ae path)

MODES = ["organize", "preprocess", "train", "evaluate", "figures", "interpolate", "compare"]


def _is_legacy(dataset: str, model: str) -> bool:
    return dataset == "mnist" and model in ("vae", "ae")


# ---------------------------------------------------------------------------
# Legacy implementation (legacy/shapeembed/); config dicts inlined in src/config.py
# ---------------------------------------------------------------------------
def run_legacy(dataset: str, model: str, mode: str, device: str, **overrides) -> None:
    import copy
    import config
    from shapeembed.utils import set_seed

    base = config.LEGACY_MNIST_VAE_CONFIG if model == "vae" else config.LEGACY_MNIST_AE_CONFIG
    cfg = copy.deepcopy(base)

    # redirect every path at the repository tree
    out = HERE / "outputs"
    cfg["outputs"]["checkpoint_dir"] = str(out / "checkpoints")
    cfg["outputs"]["log_dir"] = str(out / "logs")
    cfg["outputs"]["embedding_dir"] = str(out / "embeddings")
    cfg["outputs"]["figure_dir"] = str(out / "figures")
    cfg["outputs"]["metric_dir"] = str(out / "metrics")
    cfg["dataset"]["raw_dir"] = str(HERE / "data" / "raw" / "mnist")
    cfg["dataset"]["processed_dir"] = str(HERE / "data" / "processed" / "mnist_legacy")
    cfg["training"]["device"] = device
    cfg["training"].update({k: v for k, v in overrides.items() if v is not None})

    set_seed(int(cfg["dataset"]["seed"]))

    if mode == "organize":
        print("[run] nothing to organize for MNIST (auto-downloaded by torchvision).")
    elif mode == "preprocess":
        from shapeembed.datasets import prepare_dataset
        prepare_dataset(cfg)
    elif mode == "train":
        from shapeembed.train import train_model
        train_model(cfg)
    elif mode == "evaluate":
        from shapeembed.evaluate import evaluate_embeddings
        evaluate_embeddings(cfg, use_size=False, force_reencode=True, checkpoint=None)
    elif mode == "figures":
        from shapeembed.visualize import visualize_all
        visualize_all(cfg)                       # training curves + reconstructions + latent t-SNE
    elif mode == "interpolate":
        if model != "vae":
            print("[run] latent interpolation is only defined for the VAE; nothing to do for the AE.")
            return
        from shapeembed.visualize import visualize_latent_interpolation
        visualize_latent_interpolation(cfg)
    elif mode == "compare":
        print("[run] the cross-model comparison lives in the main code path "
              "(BBBC010, `--dataset bbbc010 --model vae --mode compare`); nothing to do here.")
    else:
        raise ValueError(mode)


# ---------------------------------------------------------------------------
# Main implementation (src/, config.py-globals driven)
# ---------------------------------------------------------------------------
def run_main(dataset: str, model: str, mode: str, device: str, **overrides) -> None:
    # patch config.py's globals BEFORE importing any of the main modules (they
    # capture config values at import time)
    import config
    config.DATASET = dataset
    config.MODEL_TYPE = "original" if model == "vae" else model     # "original" == the ShapeEmbed VAE
    config.MATRIX_SIZE = 32 if dataset == "mnist" else 64
    config.DATA_PROCESSED = config.ROOT / "data" / "processed" / dataset
    config.RUN_NAME = f"{dataset}_{config.MODEL_TYPE}"
    config.DEVICE = device
    for key, value in overrides.items():
        if value is not None:
            setattr(config, key.upper(), value)

    if mode == "organize":
        if dataset != "bbbc010":
            print("[run] nothing to organize for MNIST (auto-downloaded by torchvision).")
            return
        import organize_bbbc010 as org
        org.download_if_needed(); org.unzip_if_needed(); org.organize()
    elif mode == "preprocess":
        import preprocess
        preprocess.preprocess_dataset()
    elif mode == "train":
        import train
        train.train()
    elif mode == "evaluate":
        import evaluate
        evaluate.evaluate()                      # 5-fold logreg metrics + t-SNE figure
    elif mode == "figures":
        import report_figures
        report_figures.main()                    # preprocessing gallery + training curves + reconstruction grid
    elif mode == "interpolate":
        if config.MODEL_TYPE not in ("original",):
            print("[run] latent interpolation uses the VAE; pass `--model vae`.")
            return
        import interpolate
        interpolate.run_interpolation(num_steps=9)
    elif mode == "compare":
        if dataset != "bbbc010":
            print("[run] the cross-model comparison is set up for BBBC010 only.")
            return
        import compare_models
        compare_models.make_comparison()         # needs bbbc010_{original,ae,mlp_decoder}_best.pt
    else:
        raise ValueError(mode)


def main() -> None:
    p = argparse.ArgumentParser(
        description="ShapeEmbed reproduction — unified runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset", required=True, choices=["mnist", "bbbc010"])
    p.add_argument("--model", required=True, choices=["vae", "ae", "mlp_decoder"],
                   help="'vae' = original ShapeEmbed VAE")
    p.add_argument("--mode", required=True, choices=MODES)
    p.add_argument("--device", default="auto", choices=["cpu", "mps", "cuda", "auto"])
    p.add_argument("--epochs", type=int, help="override the configured training epochs")
    p.add_argument("--batch-size", type=int, help="override the training batch size")
    p.add_argument("--num-workers", type=int, help="DataLoader workers (default: 0)")
    args = p.parse_args()
    for name in ("epochs", "batch_size"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers is not None and args.num_workers < 0:
        p.error("--num-workers must be nonnegative")
    from utils import resolve_device
    try:
        args.device = resolve_device(args.device)
    except ValueError as exc:
        p.error(str(exc))
    overrides = dict(epochs=args.epochs, batch_size=args.batch_size, num_workers=args.num_workers)

    legacy = _is_legacy(args.dataset, args.model)
    print(f"[run] dataset={args.dataset} model={args.model} mode={args.mode} "
          f"device={args.device}  ->  {'legacy (legacy/shapeembed/)' if legacy else 'main (src/)'}")

    if legacy:
        run_legacy(args.dataset, args.model, args.mode, args.device, **overrides)
    else:
        run_main(args.dataset, args.model, args.mode, args.device, **overrides)


if __name__ == "__main__":
    main()
