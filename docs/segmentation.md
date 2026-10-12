# A shape prior for segmentation

**Status: research proposal; no segmentation network is implemented here.**

ShapeEmbed starts from a segmented object mask and learns a global shape descriptor. A segmentation extension would pair a separate pixel-prediction network with a shape-consistency objective.

Let a segmentation network predict a mask `M_hat` from an image `I`. A shape encoder could compare the predicted and ground-truth mask descriptors:

```text
L = L_seg(M_hat, M) + λ ||E(D(M_hat)) − E(D(M))||²
```

`D` denotes contour extraction, sampling, and distance-matrix construction. The pixel loss supplies localization, appearance, and image context; the descriptor term supplies a global boundary-shape preference.

The current preprocessing contains thresholding, discrete contour extraction, and resampling. Calling it inside a training loop does not provide a differentiable path from the embedding loss to the segmentation network. An end-to-end implementation would need differentiable contour/occupancy representations or a learned differentiable surrogate, with gradients and failure cases tested explicitly.

A shape descriptor intentionally discards pose and size, so it cannot replace pixel-level supervision. The single-contour representation also misses holes and disconnected components. A topology-aware extension could address those representational limits but remains a proposal in this repository.

Nearer-term uses require less machinery: flag unusual predicted shapes for review, rank candidate segmentations, or use embeddings in downstream quality-control analysis. None of these uses has been benchmarked in this project.
