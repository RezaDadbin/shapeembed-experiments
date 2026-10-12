import torch
import torch.nn.functional as F


# =============================================================================
# Equivalent distance matrices
# =============================================================================

def generate_equivalent_matrices(x: torch.Tensor) -> torch.Tensor:
    """
    Generate all 2N equivalent versions of the target distance matrix.

    Paper equation:
        d^{k,o}_{i,j} = d_{(i o + k) mod N, (j o + k) mod N}

    where:
        k = starting point shift
        o = traversal direction, either +1 or -1

    Input:
        x: [B, 1, N, N]

    Output:
        equivalents: [B, 2N, N, N]
    """

    if x.ndim != 4:
        raise ValueError(f"Expected x shape [B, 1, N, N], got {x.shape}")

    batch_size, channels, height, width = x.shape

    if channels != 1:
        raise ValueError(f"Expected one-channel matrices, got {channels}")

    if height != width:
        raise ValueError(f"Expected square matrices, got {height}x{width}")

    n = height
    base = torch.arange(n, device=x.device)

    equivalents = []

    for direction in (1, -1):
        for shift in range(n):
            idx = (base * direction + shift) % n

            x_equiv = x[:, :, idx, :]
            x_equiv = x_equiv[:, :, :, idx]

            equivalents.append(x_equiv)

    return torch.cat(equivalents, dim=1)


# =============================================================================
# Reconstruction loss
# =============================================================================

def indexation_invariant_reconstruction_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    ShapeEmbed reconstruction loss.

    Instead of comparing recon with only one arbitrary contour indexing,
    compare it against all 2N valid equivalent target matrices and take
    the minimum MSE.

    recon:
        [B, 1, N, N]

    target:
        [B, 1, N, N]
    """

    equivalents = generate_equivalent_matrices(target)

    # [B, 1, N, N] -> [B, 2N, N, N]
    recon_expanded = recon.expand(-1, equivalents.size(1), -1, -1)

    mse_per_equivalent = ((recon_expanded - equivalents) ** 2).mean(dim=(-2, -1))

    min_mse_per_sample = mse_per_equivalent.min(dim=1).values

    return min_mse_per_sample.mean()


# =============================================================================
# Distance-matrix regularizers
# =============================================================================

def diagonal_loss(recon: torch.Tensor) -> torch.Tensor:
    """
    Distance matrix property:
        D[i, i] = 0
    """

    diagonal = torch.diagonal(recon, dim1=-2, dim2=-1)
    return (diagonal ** 2).mean()


def non_negative_loss(recon: torch.Tensor) -> torch.Tensor:
    """
    Distance matrix property:
        D[i, j] >= 0
    """

    return F.relu(-recon).mean()


def symmetry_loss(recon: torch.Tensor) -> torch.Tensor:
    """
    Distance matrix property:
        D = D^T
    """

    return F.mse_loss(recon, recon.transpose(-1, -2))


# =============================================================================
# KL loss
# =============================================================================

def kl_divergence(
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    """
    Standard KL divergence between q(z|x) and N(0, I).

    Output is averaged over the batch.
    """

    kl_per_sample = -0.5 * torch.sum(
        1.0 + logvar - mu.pow(2) - logvar.exp(),
        dim=1,
    )

    return kl_per_sample.mean()


# =============================================================================
# Total losses
# =============================================================================

def original_shapeembed_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta_kl: float = 1e-10,
    lambda_diag: float = 1e-5,
    lambda_nonneg: float = 1e-5,
    lambda_sym: float = 1e-5,
):
    """
    Loss for:
        MODEL_TYPE = "original"

    Also used for:
        MODEL_TYPE = "mlp_decoder"

    because Phase 2.2 changes the decoder architecture only, not the
    variational objective.
    """

    rec = indexation_invariant_reconstruction_loss(recon, target)
    kl = kl_divergence(mu, logvar)

    diag = diagonal_loss(recon)
    nonneg = non_negative_loss(recon)
    sym = symmetry_loss(recon)

    total = (
        rec
        + beta_kl * kl
        + lambda_diag * diag
        + lambda_nonneg * nonneg
        + lambda_sym * sym
    )

    logs = {
        "total": total.detach().item(),
        "rec": rec.detach().item(),
        "kl": kl.detach().item(),
        "diag": diag.detach().item(),
        "nonneg": nonneg.detach().item(),
        "sym": sym.detach().item(),
    }

    return total, logs


def ae_variant_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    lambda_diag: float = 1e-5,
    lambda_nonneg: float = 1e-5,
    lambda_sym: float = 1e-5,
):
    """
    Loss for:
        MODEL_TYPE = "ae"

    Same reconstruction and distance-matrix regularization terms,
    but no KL term.
    """

    rec = indexation_invariant_reconstruction_loss(recon, target)

    diag = diagonal_loss(recon)
    nonneg = non_negative_loss(recon)
    sym = symmetry_loss(recon)

    total = (
        rec
        + lambda_diag * diag
        + lambda_nonneg * nonneg
        + lambda_sym * sym
    )

    logs = {
        "total": total.detach().item(),
        "rec": rec.detach().item(),
        "diag": diag.detach().item(),
        "nonneg": nonneg.detach().item(),
        "sym": sym.detach().item(),
    }

    return total, logs


def compute_loss(
    model_type: str,
    outputs: dict,
    beta_kl: float = 1e-10,
    lambda_diag: float = 1e-5,
    lambda_nonneg: float = 1e-5,
    lambda_sym: float = 1e-5,
):
    """
    Convenience function for train.py.

    target is outputs["x_pre"], because the model normalizes the input
    internally and the reconstruction should match the normalized matrix.
    """

    recon = outputs["recon"]
    target = outputs["x_pre"]

    if model_type in ("original", "mlp_decoder"):
        return original_shapeembed_loss(
            recon=recon,
            target=target,
            mu=outputs["mu"],
            logvar=outputs["logvar"],
            beta_kl=beta_kl,
            lambda_diag=lambda_diag,
            lambda_nonneg=lambda_nonneg,
            lambda_sym=lambda_sym,
        )

    if model_type == "ae":
        return ae_variant_loss(
            recon=recon,
            target=target,
            lambda_diag=lambda_diag,
            lambda_nonneg=lambda_nonneg,
            lambda_sym=lambda_sym,
        )

    raise ValueError(
        f"Unknown model_type={model_type}. "
        f"Use 'original', 'ae', or 'mlp_decoder'."
    )
