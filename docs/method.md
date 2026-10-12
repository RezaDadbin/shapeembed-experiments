# Methods

## 2D contours

A binary object mask is reduced to one outer contour, sampled at a fixed number of approximately uniform arc-length positions: 32 for MNIST and 64 for BBBC010. The pairwise Euclidean distances form an `N × N` matrix. Frobenius normalization removes global size; the original norm can be appended to the embedding as a separate size feature.

Distances remove translation, rotation, and reflection in image coordinates. Cyclic shifts of a contour's starting point and reversal of traversal still permute matrix entries. The loss compares a prediction with all `2N` valid target indexings and takes the minimum mean squared error. This loss invariance does not by itself prove exact cyclic-shift invariance of a strided convolutional encoder.

The main encoder is a modified ResNet-18 with circular padding. The encoders add features of the matrix and its doubly flipped version, giving traversal-reversal invariance in evaluation mode. Variational models then estimate latent mean and log variance; the AE estimates a deterministic code. The default latent has 128 dimensions.

Three variants are exposed by `run.py`:

- `vae`: convolutional variational autoencoder;
- `ae`: deterministic autoencoder;
- `mlp_decoder`: variational encoder with a multilayer perceptron in place of the convolutional decoder.

The main pipeline combines indexation-invariant MSE, KL weight `1e-10`, and diagonal, nonnegativity, and symmetry regularizers with weight `1e-5`. The main VAE clamps log variance to keep posterior noise small; this is a project-specific modification. Evaluation decodes the posterior mean deterministically.

## Two historical 2D implementations

The MNIST VAE/AE experiments use `legacy/shapeembed/`, which adapts the official model and keeps its checkpoint architecture. BBBC010 and the MNIST MLP-decoder experiment use `src/`, a separate implementation with a simpler convolutional decoder.

The legacy model scales normalized matrices by `N`; the main pipeline uses Frobenius norm one. Preprocessing and model wrappers also differ. Accordingly, the archived MNIST comparisons span different pipelines and are not a controlled decoder-only ablation. Shared command-line options do not erase those differences.

Both preprocessing paths combine the original MNIST train/test images and create a new stratified 80/20 partition. This project does not report the standard official MNIST test benchmark. BBBC010 uses the individual-worm foreground masks with live/dead labels derived from control-well columns. Split and classifier details are in [results](results.md).

## Unordered 3D surfaces

A sampled surface provides a set of `N` points. Euclidean distances describe ambient geometry; approximate geodesic distances use shortest paths on a dense surface-neighbor graph. Both are separately normalized by Frobenius norm.

For each distance channel, sorting each row removes sensitivity to a common column permutation. A shared node MLP followed by mean/max pooling removes row-permutation sensitivity. A distance-matrix eigenspectrum adds a global invariant. Together these operations make the set encoder invariant to simultaneous row/column permutation, up to floating-point error. They do not guarantee a unique descriptor for every geometry.

The point decoder outputs centered coordinates and rescales them so their Euclidean distance matrix has unit Frobenius norm. Such a matrix corresponds to a point cloud in three dimensions by construction. The matrix-decoder ablation enforces symmetry, nonnegativity, and a zero diagonal directly, then penalizes negative eigenvalues and eigenvalue mass beyond the top three in the centered squared-distance Gram matrix.

Training compares sorted upper-triangle distance values; the point decoder also receives an aligned point-matching loss. This objective is invariant to point ordering but weaker than full geometric correspondence. It can confuse shapes with the same distance distribution, and principal-axis matching is imperfect for symmetric shapes.

## Visualization conventions

2D matrix heatmaps show raw decoder values. To draw an outline, the code symmetrizes the matrix, clamps negative entries, clears its diagonal, and performs a classical MDS projection using two nonnegative Gram eigenvalues. Outlines are aligned by translation, rotation, optional reflection, and scale for display. MDS is a projection, not proof that the raw output is a valid Euclidean distance matrix.

The public default does not amplify reconstruction variance or smooth outlines differently for different models. Reconstruction examples use seeded class-balanced sampling, independently of reconstruction error. Main-pipeline t-SNE uses a seeded uniform subsample, keeps class-overlapping points, and writes selected indices and coordinates to a JSON sidecar. Legacy latent plots use balanced subsampling and retain overlaps.

3D figures show decoded point clouds and convex hulls after canonical principal-axis orientation. Hulls omit concavities and cannot demonstrate topology preservation. The [topology extension](toposhapeembed.md) is a separate design proposal.
