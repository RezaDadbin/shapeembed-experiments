from pathlib import Path
from utils import resolve_device

# ---------------------------------------------------------------------
# Experiment selection
# ---------------------------------------------------------------------
DATASET = "mnist"          # "mnist" or "bbbc010"
MODEL_TYPE = "mlp_decoder"         # "original", "ae", or "mlp_decoder"

# ---------------------------------------------------------------------
# Paper settings
# ---------------------------------------------------------------------
if DATASET == "mnist":
    MATRIX_SIZE = 32
elif DATASET == "bbbc010":
    MATRIX_SIZE = 64
else:
    raise ValueError(f"Unknown DATASET: {DATASET}")

LATENT_DIM = 128

# Paper: Adam, lr=1e-3, 350 epochs.
LR = 1e-3
EPOCHS = 350
BATCH_SIZE = 128
NUM_WORKERS = 0

# Paper loss weights.
BETA_KL = 1e-10
LAMBDA_DIAG = 1e-5
LAMBDA_NONNEG = 1e-5
LAMBDA_SYM = 1e-5

# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------
TRAIN_SPLIT = 0.8
SEED = 42

# Paper normalizes distance matrices by Frobenius norm.
NORMALIZE_INPUT = True

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed" / DATASET

OUTPUTS = ROOT / "outputs"
CHECKPOINT_DIR = OUTPUTS / "checkpoints"
FIGURE_DIR = OUTPUTS / "figures"
LATENT_DIR = OUTPUTS / "latents"
RESULT_DIR = OUTPUTS / "results"

RUN_NAME = f"{DATASET}_{MODEL_TYPE}"

DEVICE = resolve_device("auto")


# ---------------------------------------------------------------------
# Legacy MNIST configs (original VAE + standard AE)
# ---------------------------------------------------------------------
# The MNIST runs for the original ShapeEmbed VAE ("vae") and the Phase-2
# standard AE ("ae") use the config-driven implementation in legacy/shapeembed/.
# Their configs (formerly the legacy_configs/ YAML files) are inlined here so
# there is no separate config folder. run.py's legacy path deep-copies the
# relevant dict and overrides the path-ish fields (outputs/*, dataset.raw_dir,
# dataset.processed_dir, training.device) to point at this codebase's
# outputs/ and data/ trees.

LEGACY_MNIST_VAE_CONFIG = {
    "dataset": {
        "name": "mnist",
        "raw_dir": "data/raw/mnist",
        "processed_dir": "data/processed/mnist_legacy",
        "distance_matrix_size": 32,
        "scale_distance_matrix_by_size": True,
        "contour_extraction": "project",
        "contour_resampling": "linear",
        "test_size": 0.2,
        "seed": 42,
        "max_samples": None,
    },
    "model": {
        "type": "shapeembed",
        "latent_dim": 128,
        "base_channels": 64,
        "decoder_channels": 512,
        "circular_padding": True,
        "normalize_input": "fro",
        "rescale_input": None,
        "augment_input": False,
    },
    "training": {
        "epochs": 350,
        "batch_size": 64,
        "validation_fraction": 0.15,
        "lr": 0.001,
        "beta": 1.0e-10,
        "gamma": 1.0e-5,
        "delta": 1.0e-5,
        "epsilon": 1.0e-5,
        "reconstruction_loss": "mse",
        "gradient_clip_norm": 5.0,
        "num_workers": 0,
        "device": "auto",
        "early_stopping": {"enabled": False},
    },
    "evaluation": {
        "embedding_source": "z_mean",
        "checkpoint": "best",
        "class_weight": None,
        "standardize": True,
        "logreg_max_iter": 5000,
        "n_splits": 5,
    },
    "visualization": {
        "tsne_perplexity": 5,
        "tsne_seed": 42,
        "max_points": 70000,
    },
    "outputs": {
        "checkpoint_dir": "outputs/checkpoints",
        "log_dir": "outputs/logs",
        "embedding_dir": "outputs/embeddings",
        "figure_dir": "outputs/figures",
        "metric_dir": "outputs/metrics",
    },
}

LEGACY_MNIST_AE_CONFIG = {
    "dataset": {
        "name": "mnist",
        "raw_dir": "data/raw/mnist",
        "processed_dir": "data/processed/mnist_legacy",
        "distance_matrix_size": 32,
        "scale_distance_matrix_by_size": True,
        "contour_extraction": "project",
        "contour_resampling": "linear",
        "test_size": 0.2,
        "seed": 42,
        "max_samples": None,
    },
    "model": {
        "type": "ae",
        "latent_dim": 128,
        "base_channels": 64,
        "decoder_channels": 512,
    },
    "training": {
        "epochs": 350,
        "batch_size": 64,
        "validation_fraction": 0.15,
        "lr": 0.001,
        "beta": 1.0e-10,
        "gamma": 1.0e-5,
        "delta": 1.0e-5,
        "epsilon": 1.0e-5,
        "reconstruction_loss": "mse",
        "gradient_clip_norm": 5.0,
        "num_workers": 0,
        "device": "auto",
        "early_stopping": {"enabled": False},
    },
    "evaluation": {
        "checkpoint": "best",
        "class_weight": None,
        "standardize": True,
        "logreg_max_iter": 5000,
        "n_splits": 5,
    },
    "visualization": {
        "tsne_perplexity": 5,
        "tsne_seed": 42,
        "max_points": 70000,
    },
    "outputs": {
        "checkpoint_dir": "outputs/checkpoints",
        "log_dir": "outputs/logs",
        "embedding_dir": "outputs/embeddings",
        "figure_dir": "outputs/figures",
        "metric_dir": "outputs/metrics",
    },
}
