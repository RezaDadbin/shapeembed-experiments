# Reproducibility

## Environment

Use Python **3.11** with `requirements.txt`. The direct dependencies are pinned to versions exercised locally; this is not a lock of every transitive dependency or a promise of byte-identical training on all platforms. Local verification used Python 3.11.7 on macOS arm64, PyTorch 2.6.0, torchvision 0.21.0, NumPy 1.26.4, SciPy 1.16.2, scikit-image 0.22.0, and scikit-learn 1.7.1.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` and use `python` instead of `python3.11` when creating the environment. The direct script paths work from the repository root on all supported platforms. Model/data/output paths are anchored to the repository location.

The full dependency installation has not been repeated in a fresh environment. Checks used an existing environment with the pinned versions. A GitHub Actions workflow is included for Linux CPU checks; a hosted run is pending publication.

## 2D commands

### MNIST

MNIST is downloaded by torchvision during preprocessing. Both VAE and AE use the legacy MNIST preprocessing cache.

```bash
python run.py --dataset mnist --model vae --mode preprocess
python run.py --dataset mnist --model vae --mode train
python run.py --dataset mnist --model vae --mode evaluate
python run.py --dataset mnist --model vae --mode figures
python run.py --dataset mnist --model vae --mode interpolate

python run.py --dataset mnist --model ae --mode train
python run.py --dataset mnist --model ae --mode evaluate
python run.py --dataset mnist --model ae --mode figures

# The main-pipeline MLP decoder needs its own preprocessing cache.
python run.py --dataset mnist --model mlp_decoder --mode preprocess
python run.py --dataset mnist --model mlp_decoder --mode train
python run.py --dataset mnist --model mlp_decoder --mode evaluate
python run.py --dataset mnist --model mlp_decoder --mode figures
```

### BBBC010

The organizer downloads the individual-worm masks from the Broad Bioimage Benchmark Collection. Control-well columns 1–12 are labeled live, and 13–24 dead. Organization copies masks into class folders with stable names and does not delete existing class folders.

```bash
python run.py --dataset bbbc010 --model vae --mode organize
python run.py --dataset bbbc010 --model vae --mode preprocess
python run.py --dataset bbbc010 --model vae --mode train
python run.py --dataset bbbc010 --model vae --mode evaluate
python run.py --dataset bbbc010 --model vae --mode figures
python run.py --dataset bbbc010 --model vae --mode interpolate

python run.py --dataset bbbc010 --model ae --mode train
python run.py --dataset bbbc010 --model ae --mode evaluate
python run.py --dataset bbbc010 --model mlp_decoder --mode train
python run.py --dataset bbbc010 --model mlp_decoder --mode evaluate
python run.py --dataset bbbc010 --model vae --mode compare
```

`compare` requires all three BBBC010 checkpoints. Preprocessing writes into `data/processed/`; training and evaluation write into `outputs/`. The default 2D training length is 350 epochs. Main-pipeline batch size is 128; legacy MNIST batch size is 64.

All 2D commands accept `--device auto|cpu|mps|cuda`. Automatic selection prefers CUDA, then MPS, then CPU. Training also accepts `--epochs`, `--batch-size`, and `--num-workers`, for example:

```bash
python run.py --dataset bbbc010 --model ae --mode train --epochs 1 --batch-size 8 --device cpu
```

This reduces training time but still requires the processed dataset. Epochs and batch size must be positive. The default worker count is zero for portability. Main settings are in `src/config.py`; MNIST VAE/AE dictionaries are in the same file.

## Checkpoint conventions

| Pipeline | Best-checkpoint filename | State key |
| --- | --- | --- |
| MNIST VAE, legacy | `outputs/checkpoints/mnist_shapeembed.pt` | `model_state` |
| MNIST AE, legacy | `outputs/checkpoints/mnist_ae.pt` | `model_state` |
| Main 2D | `outputs/checkpoints/<dataset>_<original\|ae\|mlp_decoder>_best.pt` | `model_state_dict` |
| 3D | `outputs/shapes3d/checkpoints/<run>_best.pt` | `state_dict` |

The public release does not bundle pretrained weights. Regenerate them with the training commands. Historical checkpoints have different metadata conventions and should not be interchanged across pipelines. Only load trusted PyTorch checkpoints: some legacy files include Python objects and are loaded with `weights_only=False`.

## Seeds and protocol changes

Defaults use seed 42. Seeds make data splits and sampling repeatable; hardware kernels, library changes, and stochastic training can still change results. Archived metrics came from earlier runs, with no full original environment lock.

The cleaned release fixes repository-relative 3D paths, automatic device handling in the 2D runner, a legacy RGB conversion typo, zero-input normalization, and a rigid-alignment helper. It also uses deterministic posterior-mean VAE evaluation, seeded reconstruction examples selected independently of model errors, unfiltered main t-SNE samples, and unsmoothed default MDS outlines. Older output-variance amplification and class-overlap filtering are removed.

New legacy MNIST training holds out 15% of the training partition for validation; historical runs selected checkpoints using the test partition. The main 2D pipeline still selects by training loss. New 3D training defaults to `--val-frac 0.15`, while archived experiments used zero. [Results](results.md) explains what comparisons the saved metrics support.

## Validation

```bash
python -m compileall -q run.py src legacy phase3 scripts tests
python -m unittest discover -s tests -v
python scripts/summarize_results.py --check
```

The CPU tests check similarity-normalized geometry, cyclic/reversal-invariant losses, arbitrary-permutation invariance of the set encoder, valid point-decoder geometry, rigid alignment, finite forward/backward passes for all five routed 2D variants and the 3D model, deterministic VAE evaluation, and faithful reconstruction/t-SNE presentation.

A small 3D run completed generation (20 samples per class, 16 points), one training epoch, held-out evaluation, and reconstruction/interpolation figure export. Its scores are smoke-test outputs and are not included in the archived results tables. Full 350-epoch training and a fresh-environment install were not repeated. GPU backends and the remote GitHub Actions workflow have not been validated for this release.

The legacy MNIST trainer also completed one CPU epoch on a temporary synthetic training set with no `test.pt` present, verifying that checkpoint selection now uses only a split of the training data. This is a software check, not an MNIST benchmark.
