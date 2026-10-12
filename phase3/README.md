# SurfaceGraph-ShapeEmbed: 3D proof of concept

This extension replaces the natural cyclic order of a 2D contour with an unordered set of sampled surface points. Each shape has a normalized Euclidean distance matrix and, optionally, an approximate geodesic matrix computed by shortest paths on a dense k-nearest-neighbor graph.

The proposed encoder sorts each distance row, applies a shared node MLP, and pools with mean and maximum. It also uses a permutation-invariant distance-matrix eigenvalue signature. CNN and flattened-matrix MLP encoders provide order-sensitive baselines.

The default decoder emits an `N × 3` point cloud and computes its pairwise distances. A direct matrix decoder provides an ablation with a penalty on the centered Gram matrix: it should be positive semidefinite with rank at most three. The ordinary distance matrix itself is not constrained to rank three.

## Full experiment

From the repository root, after installing `requirements.txt`:

```bash
# 800 examples: 200 each of cube, sphere, pyramid, and cylinder; 64 points.
python phase3/generate_dataset.py --per-class 200 --n-points 64 --seed 42

# Proposed model; the default reserves 15% of the training partition for validation.
python phase3/train.py --encoder set --features both --epochs 350
python phase3/evaluate.py --encoder set --features both

# Encoder ablations.
python phase3/train.py --encoder cnn --features both --epochs 350
python phase3/train.py --encoder mlp --features both --epochs 350

# Feature and decoder ablations.
python phase3/train.py --encoder set --features euclidean --epochs 350
python phase3/train.py --encoder set --features geo --epochs 350
python phase3/train.py --encoder set --features both --ae --epochs 350
python phase3/train.py --encoder set --features both --decoder matrix --edm-weight 0.01 --epochs 350
python phase3/train.py --encoder set --features both --decoder matrix --edm-weight 0 --tag noedm --epochs 350

# Evaluate all local checkpoints, including any demo runs.
python phase3/evaluate.py --compare
```

Train and evaluation accept `--device auto|cpu|cuda|mps`. Automatic selection prefers CUDA, then MPS, then CPU. CPU eigendecomposition is used where MPS lacks the required kernel. Full MPS/CUDA training is not part of the cleaned release's CPU smoke verification.

The archived results used `--val-frac 0`, selecting checkpoints by training reconstruction loss. To recreate that historical configuration, append `--val-frac 0` to each train command. The new default uses a separate validation subset and therefore has a different training protocol.

The generator overwrites its local dataset files. Keep different point counts or seeds in separate checkouts, and generate matching data before evaluating a checkpoint. The split is stratified 80/20 with seed 42 by default. Surface sampling uses 3,000 dense points before farthest-point subsampling; shortest-path geodesics are approximate and may have shortcuts.

## Training objective

The reconstruction term compares sorted multisets of pairwise Euclidean distances. The point decoder additionally uses a `0.05`-weighted point loss after detached principal-axis alignment and Hungarian matching. Variational models add a KL term (`beta=1e-6`) and clamp log variance to `[-10, -4]`; deterministic AE mode disables sampling and KL. The direct matrix decoder adds the Gram-matrix validity penalty.

Sorted distance distributions do not uniquely identify arbitrary shapes. Principal-axis alignment can be unstable for symmetric objects. These are practical proof-of-concept choices, not a general correspondence solver.

## Output layout

```text
data/processed/shapes3d/{train,test}.npz
outputs/shapes3d/checkpoints/<run>_best.pt
outputs/shapes3d/results/<run>_history.json
outputs/shapes3d/results/<run>_evaluation.json
outputs/shapes3d/results/encoder_comparison.{csv,json}
outputs/shapes3d/figures/<run>_{reconstruction,interpolation}.png
```

Run names follow `<encoder>_<features>[_mat][_ae][_<tag>]`. Use `--run` to evaluate an exact name, for example `python phase3/evaluate.py --run set_both_mat_noedm`.

The saved metrics include frozen-embedding cross-validation on the test partition, relative latent drift after independent row/column permutations, and train-to-permuted-test classifier accuracy. These are separate protocols; details are in [the results note](../docs/results.md).

## Scope

The dataset contains four deliberately distinct convex primitives with small aspect jitter and random pose. Reconstructions are shown as convex hulls in a canonical orientation. This demonstrates point-order invariance and a working learning pipeline. It does not establish performance on real scans, nonconvex or arbitrary-topology surfaces, mesh reconstruction, or bending robustness. Geodesic features are implemented; a controlled bending experiment remains future work.
