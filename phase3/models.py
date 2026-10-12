# SurfaceGraph-ShapeEmbed: a permutation-invariant metric-graph extension of
# ShapeEmbed to 3D surfaces.
#
# A 3D shape is represented as an *unordered* set of N surface points equipped
# with pairwise distances -- a "surface metric graph".  The edge features are the
# Euclidean distance matrix D^E (extrinsic) and, optionally, an approximate
# geodesic distance matrix D^G (intrinsic / bending-aware).  Both matrices are
# Frobenius-normalised on disk, so the representation is invariant to rigid
# motion (by construction), to global scale (normalisation), and -- crucially in
# 3D -- to the arbitrary ordering of the sampled points (handled by the encoder).
#
# Three encoders are provided so the report can run the "encoder ablation":
#   * SetEncoder  -- permutation-invariant (sorted distance rows + DeepSets pooling
#                    + permutation-invariant spectral global features).  The proposed model.
#   * CNNEncoder  -- a 2D CNN over the matrix treated as an image (NOT permutation
#                    invariant: reordering the points permutes rows/columns).
#   * MLPEncoder  -- an MLP over the flattened matrix (also NOT permutation invariant).
#
# The default MLP decoder emits an N x 3 point cloud and its Euclidean distances.
# The matrix-decoder ablation uses a penalty on the centered Gram matrix to
# encourage a positive semidefinite matrix of rank at most three.

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------

def _mlp(sizes, last_act=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_act:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def _offdiag_mask(n: int, device) -> torch.Tensor:
    return 1.0 - torch.eye(n, device=device)


def _eigvalsh(m: torch.Tensor) -> torch.Tensor:
    """torch.linalg.eigvalsh, with a CPU fallback (MPS lacks the kernel) and a
    tiny diagonal jitter for numerical robustness on near-degenerate spectra."""
    dev = m.device
    work = m if dev.type != "mps" else m.cpu()
    eye = torch.eye(work.shape[-1], device=work.device, dtype=work.dtype)
    try:
        w = torch.linalg.eigvalsh(work + 1e-6 * eye)
    except torch.linalg.LinAlgError:
        w = torch.linalg.eigvalsh(work.double() + 1e-5 * eye).to(work.dtype)
    return w.to(dev)


# ---------------------------------------------------------------------
# Encoders.  All take x of shape (B, C, N, N) and return (mu, logvar).
# ---------------------------------------------------------------------

class SetEncoder(nn.Module):
    """Permutation-invariant encoder over the surface metric graph.

    Node feature for point i = the *sorted* row D[c, i, :] for each channel c
    (sorting -> invariance to the column permutation).  A shared MLP embeds each
    node; mean+max pooling over nodes -> invariance to the row permutation.  We
    additionally append the sorted eigenvalues of D^E, which are a
    permutation-invariant global signature of the metric graph.
    """

    def __init__(self, channels: int, n_points: int, latent_dim: int, hidden: int = 384):
        super().__init__()
        self.c, self.n = channels, n_points
        self.in_norm = nn.LayerNorm(channels * n_points)      # the matrix entries are tiny (||D||_F = 1, N large)
        self.phi = _mlp([channels * n_points, hidden, hidden], last_act=True)
        self.eig_norm = nn.LayerNorm(n_points)
        self.rho = _mlp([2 * hidden + n_points, hidden, hidden], last_act=True)
        self.to_mu = nn.Linear(hidden, latent_dim)
        self.to_logvar = nn.Linear(hidden, latent_dim)

    def forward(self, x):
        b, c, n, _ = x.shape
        rows_sorted = torch.sort(x, dim=-1).values            # (B, C, N, N)
        node_feat = rows_sorted.permute(0, 2, 1, 3).reshape(b, n, c * n)
        h = self.phi(self.in_norm(node_feat))                 # (B, N, H)
        pooled = torch.cat([h.mean(dim=1), h.amax(dim=1)], dim=-1)   # (B, 2H)
        eig = self.eig_norm(_eigvalsh(x[:, 0]))               # (B, N) ascending, perm-invariant
        g = self.rho(torch.cat([pooled, eig], dim=-1))
        return self.to_mu(g), self.to_logvar(g)


class CNNEncoder(nn.Module):
    """2D CNN over the distance matrix treated as a C-channel image.

    NOT permutation invariant -- included as the ablation baseline."""

    def __init__(self, channels: int, n_points: int, latent_dim: int, width: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, width, 3, stride=2, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, 2 * width, 3, stride=2, padding=1), nn.BatchNorm2d(2 * width), nn.ReLU(inplace=True),
            nn.Conv2d(2 * width, 4 * width, 3, stride=2, padding=1), nn.BatchNorm2d(4 * width), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.to_mu = nn.Linear(4 * width, latent_dim)
        self.to_logvar = nn.Linear(4 * width, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.to_mu(h), self.to_logvar(h)


class MLPEncoder(nn.Module):
    """MLP over the flattened distance matrix.  NOT permutation invariant."""

    def __init__(self, channels: int, n_points: int, latent_dim: int, hidden: int = 512):
        super().__init__()
        self.body = _mlp([channels * n_points * n_points, hidden, hidden], last_act=True)
        self.to_mu = nn.Linear(hidden, latent_dim)
        self.to_logvar = nn.Linear(hidden, latent_dim)

    def forward(self, x):
        h = self.body(x.flatten(1))
        return self.to_mu(h), self.to_logvar(h)


ENCODERS = {"set": SetEncoder, "cnn": CNNEncoder, "mlp": MLPEncoder}


# ---------------------------------------------------------------------
# Decoders.  Both expose .forward(z) -> dict(D_hat=(B,N,N), P_hat=(B,N,3) or None).
# ---------------------------------------------------------------------

def _matrix_from_points(P):
    """Pairwise Euclidean distance matrix of a batch of point clouds (B,N,3)."""
    return torch.cdist(P, P)


class PointCloudDecoder(nn.Module):
    """latent -> N x 3 point cloud; the predicted distance matrix is cdist(P, P).

    Reconstructing *points* (then taking distances) makes the decoded matrix a
    valid 3D Euclidean distance matrix *by construction*: no validity penalty
    is needed and classical MDS at evaluation time is exact.  This is the default."""

    def __init__(self, latent_dim: int, n_points: int, hidden: int = 384):
        super().__init__()
        self.n = n_points
        self.in_norm = nn.LayerNorm(latent_dim)               # so the decoder uses the latent regardless of its scale
        self.net = _mlp([latent_dim, hidden, 2 * hidden, 2 * hidden, 3 * n_points], last_act=False)

    def forward(self, z):
        P = self.net(self.in_norm(z)).view(-1, self.n, 3)
        P = P - P.mean(dim=1, keepdim=True)                   # centre (distances are translation-invariant anyway)
        D = _matrix_from_points(P)
        scale = torch.linalg.norm(D, dim=(-1, -2), keepdim=True).clamp_min(1e-8)
        return {"P_hat": P / scale, "D_hat": D / scale}       # match the ||D||_F = 1 normalisation of the targets


class MatrixDecoder(nn.Module):
    """latent -> N x N distance matrix directly (symmetric, non-negative, zero diag).

    Kept for the EDM-validity ablation: unlike the point decoder, nothing forces
    the output to be a realisable 3D distance matrix, so it relies on the rank-3
    penalty (see `edm_rank3_penalty`) to stay close to one."""

    def __init__(self, latent_dim: int, n_points: int, hidden: int = 256):
        super().__init__()
        self.n = n_points
        self.in_norm = nn.LayerNorm(latent_dim)
        self.net = _mlp([latent_dim, hidden, 2 * hidden, n_points * n_points], last_act=False)

    def forward(self, z):
        m = self.net(self.in_norm(z)).view(-1, self.n, self.n)
        m = 0.5 * (m + m.transpose(1, 2))                     # symmetry
        m = F.softplus(m)                                     # non-negativity
        m = m * _offdiag_mask(self.n, m.device)               # zero diagonal
        return {"P_hat": None, "D_hat": m}


DECODERS = {"point": PointCloudDecoder, "matrix": MatrixDecoder}


# ---------------------------------------------------------------------
# Full (V)AE
# ---------------------------------------------------------------------

class SurfaceGraphVAE(nn.Module):
    def __init__(self, encoder: str, channels: int, n_points: int, latent_dim: int,
                 variational: bool = True, decoder: str = "point"):
        super().__init__()
        if encoder not in ENCODERS:
            raise ValueError(f"unknown encoder {encoder!r}; choose from {list(ENCODERS)}")
        if decoder not in DECODERS:
            raise ValueError(f"unknown decoder {decoder!r}; choose from {list(DECODERS)}")
        self.variational = variational
        self.decoder_type = decoder
        self.encoder = ENCODERS[encoder](channels, n_points, latent_dim)
        self.decoder = DECODERS[decoder](latent_dim, n_points)

    def reparameterize(self, mu, logvar):
        if not (self.variational and self.training):
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        if not self.variational:
            logvar = torch.zeros_like(logvar)
        else:
            logvar = logvar.clamp(-10.0, -4.0)        # keep the posterior std small so it doesn't drown mu
        z = self.reparameterize(mu, logvar)
        dec = self.decoder(z)
        return {"mu": mu, "logvar": logvar, "z": z, **dec}


def build_model(encoder="set", channels=1, n_points=64, latent_dim=64, variational=True, decoder="point"):
    return SurfaceGraphVAE(encoder, channels, n_points, latent_dim, variational, decoder)


# ---------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------

def recon_loss(D_hat, D_target):
    """Plain MSE on the off-diagonal entries (assumes a shared point ordering)."""
    mask = _offdiag_mask(D_hat.shape[-1], D_hat.device)
    diff2 = ((D_hat - D_target) ** 2) * mask
    return diff2.sum(dim=(-1, -2)).mean() / mask.sum()


def sorted_recon_loss(D_hat, D_target):
    """MSE between the *sorted* multisets of pairwise distances of the two shapes.

    This compares the distance *distributions* -- it is invariant to the point
    ordering (no canonical order exists for a 3D point set) and to rigid motion (a
    rotation does not change distances), the 3D counterpart of ShapeEmbed's
    invariance to the 2N contour re-indexings.  Crucially it gives a useful gradient
    even when the decoder currently outputs a featureless blob, so the model can
    escape the trivial mean-shape solution that a point-matching loss gets stuck in."""
    n = D_hat.shape[-1]
    iu = torch.triu_indices(n, n, offset=1, device=D_hat.device)
    a = torch.sort(D_hat[:, iu[0], iu[1]], dim=-1).values
    b = torch.sort(D_target[:, iu[0], iu[1]], dim=-1).values
    return ((a - b) ** 2).mean()


def _canon_rot(P: np.ndarray) -> np.ndarray:
    """Rotation R (numpy) such that (P - mean) @ R.T is in a canonical principal-axis
    frame, with a deterministic sign convention from the third moment (and det = +1)."""
    Pc = P - P.mean(0)
    _, _, Vt = np.linalg.svd(Pc, full_matrices=False)
    R = Vt.copy()
    for k in range(3):
        if np.sum((Pc @ R.T)[:, k] ** 3) < 0:
            R[k] = -R[k]
    if np.linalg.det(R) < 0:                                        # keep it a proper rotation
        m3 = np.abs([np.sum((Pc @ R.T)[:, j] ** 3) for j in range(3)])
        R[int(np.argmin(m3))] *= -1
    return R


def aligned_point_loss(P_hat, P_true):
    """Squared per-point error between the predicted and target point *sets* after
    aligning both -- the decoder need only produce the right set of points, in any
    order and any pose.  This is the 3D analogue of ShapeEmbed's "minimum over the 2N
    re-indexings": both clouds are put in a canonical principal-axis frame (handles
    the pose) and their points are matched by Hungarian assignment (handles the order).
    The alignment is detached, so gradients flow through P_hat only."""
    B, n, _ = P_hat.shape
    ph = P_hat.detach().cpu().numpy(); pt = P_true.detach().cpu().numpy()
    perm = np.empty((B, n), dtype=np.int64); Rh = np.empty((B, 3, 3)); Rt = np.empty((B, 3, 3))
    for b in range(B):
        rh, rt = _canon_rot(ph[b]), _canon_rot(pt[b])
        ah = (ph[b] - ph[b].mean(0)) @ rh.T
        at = (pt[b] - pt[b].mean(0)) @ rt.T
        _, p = linear_sum_assignment(((ah[:, None, :] - at[None, :, :]) ** 2).sum(-1))
        perm[b], Rh[b], Rt[b] = p, rh, rt
    perm_t = torch.from_numpy(perm).to(P_hat.device)
    Rh_t = torch.from_numpy(Rh).to(dtype=P_hat.dtype, device=P_hat.device)
    Rt_t = torch.from_numpy(Rt).to(dtype=P_hat.dtype, device=P_hat.device)
    ph_c = torch.bmm(P_hat - P_hat.mean(1, keepdim=True), Rh_t.transpose(1, 2))
    pt_c = torch.gather(P_true - P_true.mean(1, keepdim=True), 1, perm_t.unsqueeze(-1).expand(-1, -1, 3))
    pt_c = torch.bmm(pt_c, Rt_t.transpose(1, 2))
    return ((ph_c - pt_c) ** 2).sum(-1).mean()


def kl_loss(mu, logvar):
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(dim=-1).mean()


def edm_rank3_penalty(D_hat):
    """Penalise the decoded matrix for not being the EDM of a 3D point cloud.

    Classical-MDS Gram matrix B = -1/2 J (D^2) J must be PSD with rank <= 3.
    We penalise (i) negative eigenvalues and (ii) all eigenvalue mass beyond the
    three largest.  `eigvalsh` returns eigenvalues in ascending order.
    """
    n = D_hat.shape[-1]
    J = torch.eye(n, device=D_hat.device) - 1.0 / n
    B = -0.5 * J @ (D_hat ** 2) @ J
    B = 0.5 * (B + B.transpose(1, 2))
    w = _eigvalsh(B)                              # (B, N) ascending
    neg = F.relu(-w).pow(2).sum(dim=-1)                       # eigenvalues should be >= 0
    extra = w[:, :n - 3].pow(2).sum(dim=-1)                   # only the 3 largest may be nonzero
    return (neg + extra).mean()
