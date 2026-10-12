# 3D experiments

A synthetic extension of ShapeEmbed to unordered surface points. Each shape has normalized Euclidean distances and optional approximate geodesic distances from shortest paths on a dense neighbor graph.

The set encoder sorts distance rows, applies a shared MLP, pools with mean/max, and adds an eigenvalue signature. This handles arbitrary point permutations. CNN and flattened-matrix MLP encoders are order-sensitive baselines. The default decoder predicts point coordinates and computes valid 3D Euclidean distances from them; an alternative matrix decoder uses a penalty on the centered Gram matrix's positive semidefiniteness and rank at most three.

## Commands

From the repository root, after installing `requirements.txt`:

```bash
# Full dataset: 200 examples per family, 64 points.
python phase3/generate_dataset.py --per-class 200 --n-points 64 --seed 42
python phase3/train.py --encoder set --features both --epochs 350
python phase3/evaluate.py --encoder set --features both

# Encoder comparisons.
python phase3/train.py --encoder cnn --features both --epochs 350
python phase3/train.py --encoder mlp --features both --epochs 350

# Feature/decoder comparisons.
python phase3/train.py --encoder set --features euclidean --epochs 350
python phase3/train.py --encoder set --features geo --epochs 350
python phase3/train.py --encoder set --features both --ae --epochs 350
python phase3/train.py --encoder set --features both --decoder matrix --edm-weight 0.01 --epochs 350
python phase3/train.py --encoder set --features both --decoder matrix --edm-weight 0 --tag noedm --epochs 350
python phase3/evaluate.py --compare
```

Training and evaluation accept `--device auto|cpu|cuda|mps`. Automatic selection prefers CUDA, then MPS, then CPU. New training holds out 15% of the training partition for validation. The saved experiments used `--val-frac 0`; append that flag to recreate their checkpoint-selection setup.

Data is saved in `data/processed/shapes3d/`; generation replaces previous dataset files. Use matching point counts when evaluating checkpoints. Checkpoints, metrics, and figures are written to `outputs/shapes3d/`. Run names follow `<encoder>_<features>[_mat][_ae][_<tag>]`; evaluate an exact name with `--run`, such as `--run set_both_mat_noedm`.

The loss compares sorted pairwise distances, plus a small principal-axis/Hungarian point-matching term for the point decoder. Variational models add KL regularization and clamp posterior variance. Sorted distance distributions are weaker than full correspondence, and principal-axis alignment can be unstable for symmetric objects.

Reconstructions are rendered as convex hulls. Distances identify a shape with its mirror image, so the representation cannot distinguish chirality. The prototype demonstrates a working pipeline and point-order invariance on four simple convex families; real scans, nonconvex surfaces, bending robustness, and topology preservation have not been established. [Saved results and evaluation details](../results/README.md).
