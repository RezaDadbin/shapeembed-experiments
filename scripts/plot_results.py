#!/usr/bin/env python3
"""Plot the archived permutation experiment, without modifying model outputs."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    runs = ["cnn_both", "mlp_both", "set_both"]
    rows = [json.loads((ROOT / "results/archived/3d" / f"{r}_evaluation.json").read_text()) for r in runs]
    names = ["CNN", "MLP", "Set encoder"]
    colors = ["#8997ab", "#64748b", "#137c8b"]
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    drift = [r["latent_drift_under_permutation"] for r in rows]
    accuracy = [r["accuracy_with_permuted_inputs"] * 100 for r in rows]
    axes[0].bar(names, drift, color=colors, width=0.6)
    axes[0].set(yscale="log", ylim=(1e-8, 80), ylabel="Mean relative latent drift (log scale)",
                title="Does reordering points change the descriptor?")
    axes[0].grid(axis="y", alpha=0.15)
    for i, value in enumerate(drift):
        axes[0].text(i, value * 1.7, f"{value:.2e}", ha="center", fontsize=10)
    axes[1].bar(names, accuracy, color=colors, width=0.6)
    axes[1].axhline(25, color="#a44343", linestyle="--", linewidth=1, label="Chance (4 classes)")
    axes[1].set(ylim=(0, 115), ylabel="Held-out accuracy after point reordering (%)",
                title="Classifier trained on the original ordering")
    axes[1].legend(loc="upper left", frameon=False, fontsize=9)
    for i, value in enumerate(accuracy):
        axes[1].text(i, value + 2, f"{value:.1f}%", ha="center", fontsize=10)
    fig.savefig(ROOT / "docs/assets/permutation_robustness.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
