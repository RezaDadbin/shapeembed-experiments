# Visualization-only helper.
#
# Regenerates the latent-space t-SNE figure for every model whose latents have
# already been cached under outputs/latents/<run>_latents.npz. It does NOT load
# any checkpoint and does NOT re-encode the datasets, so it is cheap to run.
#
# Large latent sets (e.g. full MNIST, 70k points) are randomly subsampled before
# t-SNE so the plot is readable and finishes quickly. Use --max-points to change.
#
# Usage:
#   python make_tsne_figures.py                 # all cached runs
#   python make_tsne_figures.py mnist_mlp_decoder bbbc010_original
#   python make_tsne_figures.py --max-points 5000

import sys

import numpy as np

from config import LATENT_DIR, FIGURE_DIR
from visualize import save_tsne_figure


def make_for_run(run_name: str, max_points: int):
    npz_path = LATENT_DIR / f"{run_name}_latents.npz"
    if not npz_path.exists():
        print(f"[skip] missing latents: {npz_path}")
        return

    data = np.load(npz_path)
    z = np.concatenate([data["z_train"], data["z_test"]], axis=0)
    y = np.concatenate([data["y_train"], data["y_test"]], axis=0)

    if len(z) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(z), size=max_points, replace=False)
        z, y = z[idx], y[idx]
        suffix = f" (subsampled to {max_points})"
    else:
        suffix = ""

    save_path = FIGURE_DIR / f"{run_name}_tsne.png"
    save_tsne_figure(
        latents=z,
        labels=y,
        save_path=save_path,
        title=f"{run_name} latent space (t-SNE){suffix}",
    )
    print(f"[ok] {save_path}  ({len(z)} points, {z.shape[1]}-d)")


def main():
    args = [a for a in sys.argv[1:]]
    max_points = 8000
    if "--max-points" in args:
        i = args.index("--max-points")
        max_points = int(args[i + 1])
        del args[i : i + 2]

    if args:
        runs = args
    else:
        runs = sorted(
            p.name[: -len("_latents.npz")]
            for p in LATENT_DIR.glob("*_latents.npz")
        )

    if not runs:
        print(f"No *_latents.npz found in {LATENT_DIR}")
        return

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for run in runs:
        make_for_run(run, max_points)


if __name__ == "__main__":
    main()
