from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _tensor_stats(name: str, value: torch.Tensor) -> str:
    value_detached = value.detach()
    finite_mask = torch.isfinite(value_detached)
    finite_count = int(finite_mask.sum().item())
    total_count = value_detached.numel()

    if finite_count > 0:
        finite_values = value_detached[finite_mask]
        min_value = float(finite_values.min().cpu().item())
        max_value = float(finite_values.max().cpu().item())
    else:
        min_value = float("nan")
        max_value = float("nan")

    return (
        f"{name}: shape={tuple(value_detached.shape)} "
        f"dtype={value_detached.dtype} device={value_detached.device} "
        f"finite={finite_count}/{total_count} min={min_value} max={max_value}"
    )


def _assert_finite_tensor(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise RuntimeError(f"Non-finite tensor before loss: {_tensor_stats(name, value)}")


def _assert_finite_component(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise RuntimeError(f"Non-finite loss component {name}: {_tensor_stats(name, value)}")


def diagonal_loss(distance_matrix):
    diag = torch.diagonal(distance_matrix, dim1=-2, dim2=-1)
    return F.mse_loss(
        diag,
        torch.zeros_like(diag),
    )


def symmetry_loss(distance_matrix):
    return F.mse_loss(distance_matrix, distance_matrix.transpose(-2, -1))


def non_negative_loss(distance_matrix):
    negative_values = torch.relu(-distance_matrix)
    return negative_values.mean()


def triangle_inequality_loss(distance_matrix, n=1000):
    n, m = distance_matrix.shape[-2:]

    i, j = torch.meshgrid(torch.arange(n), torch.arange(m))

    indices = torch.arange(n * m)
    indices = indices[torch.randperm(indices.size()[0])]
    indices = indices[:n]
    combinations = torch.combinations(indices, 3)

    i, j, k = combinations.unbind(1)
    i1, i2 = i // m, i % m
    j1, j2 = j // m, j % m
    k1, k2 = k // m, k % m

    violation = (
        distance_matrix[..., i1, i2]
        + distance_matrix[..., j1, j2]
        - distance_matrix[..., k1, k2]
    )

    return torch.relu(violation).nanmean()


def clockwise_order_loss(distance_matrix):
    n, m = distance_matrix.shape[-2:]

    expected_order = torch.arange(n * m).reshape(1, 1, n, m).to(distance_matrix.device)

    expected_diff = (
        expected_order.unsqueeze(-1).unsqueeze(-1)
        - expected_order.unsqueeze(-3).unsqueeze(-3)
    ) % (n * m)

    distance_matrix_exp = distance_matrix.unsqueeze(-2).unsqueeze(-2)

    observed_diff = (distance_matrix_exp > distance_matrix_exp.transpose(-1, -2)).long()

    expected_diff = expected_diff.expand_as(observed_diff)

    loss = (observed_diff != expected_diff).float().mean()

    return loss


losses = {
    "diagonal": diagonal_loss,
    "symmetry": symmetry_loss,
    "non_negative": non_negative_loss,
    "triangle_inequality": triangle_inequality_loss,
}


def vae_loss(
    recon_x,
    x,
    z_mean,
    z_log_var,
    beta,
    use_index_invariant_loss=True,
    use_reflection_invariant_loss=True,
    alpha_dm_loss_diagonal=None,
    alpha_dm_loss_symmetry=None,
    alpha_dm_loss_non_negative=None,
):
    agg_loss = torch.Tensor([0.0]).to(recon_x.device)
    losses = {"total": 0.0}

    x_ = x.detach().clone()

    if use_index_invariant_loss:
        _, _, _, sz = x_.shape
        x_ = torch.cat(
            tuple(torch.roll(x_, (i, i), dims=(2, 3)) for i in range(sz)),
            dim=1,
        )

    if use_reflection_invariant_loss:
        x_ = torch.cat((x_, torch.flip(x_, dims=(2, 3))), dim=1)

    if use_reflection_invariant_loss or use_index_invariant_loss:
        _, sz, _, _ = x_.shape
        diffs = torch.sub(recon_x.repeat(1, sz, 1, 1), x_)
        means = torch.mean(torch.abs(diffs), dim=(2, 3))
        minimums = torch.min(means, dim=1).values
        mean_minimum = torch.mean(minimums)
        recon_loss = mean_minimum
    else:
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction="mean")
    losses["recon"] = recon_loss.item()
    agg_loss += recon_loss

    z_log_var = torch.clamp(z_log_var, min=-10.0, max=10.0)
    kl_loss = -0.5 * torch.sum(
        1 + z_log_var - z_mean.pow(2) - z_log_var.exp(),
        dim=1,
    ).mean()
    beta_kl_loss = beta * kl_loss
    losses["kl"] = kl_loss.item()
    losses["beta_kl"] = beta_kl_loss.item()
    agg_loss += beta_kl_loss

    if alpha_dm_loss_diagonal or alpha_dm_loss_symmetry or alpha_dm_loss_non_negative:
        dm_loss = 0.0
        if alpha_dm_loss_diagonal:
            dm_loss += alpha_dm_loss_diagonal * diagonal_loss(recon_x)
        if alpha_dm_loss_symmetry:
            dm_loss += alpha_dm_loss_symmetry * symmetry_loss(recon_x)
        if alpha_dm_loss_non_negative:
            dm_loss += alpha_dm_loss_non_negative * non_negative_loss(recon_x)
        losses["dm"] = dm_loss.item()
        agg_loss += dm_loss

    losses["total"] = agg_loss.item()
    return agg_loss, losses


def indexation_invariant_reconstruction_loss(
    D_hat: torch.Tensor,
    D: torch.Tensor,
    loss_type: str = "l1",
) -> torch.Tensor:
    if D_hat.ndim != 4 or D.ndim != 4:
        raise ValueError(f"Expected [B,1,N,N], got {D_hat.shape}, {D.shape}")

    D_target = D.detach().clone()
    _, _, _, matrix_size = D_target.shape
    candidates = torch.cat(
        tuple(
            torch.roll(D_target, (shift, shift), dims=(2, 3))
            for shift in range(matrix_size)
        ),
        dim=1,
    )
    candidates = torch.cat((candidates, torch.flip(candidates, dims=(2, 3))), dim=1)

    if loss_type.lower() in {"l1", "mae"}:
        diffs = torch.abs(D_hat.repeat(1, candidates.shape[1], 1, 1) - candidates)
    elif loss_type.lower() in {"mse", "l2"}:
        diffs = (D_hat.repeat(1, candidates.shape[1], 1, 1) - candidates) ** 2
    else:
        raise ValueError(f"Unknown reconstruction loss type: {loss_type}")

    means = torch.mean(diffs, dim=(2, 3))
    return torch.mean(torch.min(means, dim=1).values)


def kl_divergence_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    logvar = torch.clamp(logvar, min=-10.0, max=10.0)
    kl_per_sample = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp(),
        dim=1,
    )
    return kl_per_sample.mean()


def nonnegativity_loss(D_hat: torch.Tensor) -> torch.Tensor:
    return non_negative_loss(D_hat)


def shapeembed_vae_loss(
    D_hat: torch.Tensor,
    D: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
    gamma: float,
    delta: float,
    epsilon: float,
    reconstruction_loss: str = "l1",
) -> dict[str, torch.Tensor]:
    _assert_finite_tensor("D_hat", D_hat)
    _assert_finite_tensor("D", D)
    _assert_finite_tensor("mu", mu)
    _assert_finite_tensor("logvar", logvar)

    rec = indexation_invariant_reconstruction_loss(
        D_hat,
        D,
        loss_type=reconstruction_loss,
    )
    kl = kl_divergence_loss(mu, logvar)
    diag = diagonal_loss(D_hat)
    nonneg = non_negative_loss(D_hat)
    sym = symmetry_loss(D_hat)

    total = rec + beta * kl + gamma * diag + delta * nonneg + epsilon * sym

    components = {
        "total": total,
        "rec": rec,
        "kl": kl,
        "diag": diag,
        "nonneg": nonneg,
        "sym": sym,
    }

    for name, value in components.items():
        _assert_finite_component(name, value)

    return components


def shapeembed_ae_loss(
    D_hat: torch.Tensor,
    D: torch.Tensor,
    gamma: float,
    delta: float,
    epsilon: float,
    reconstruction_loss: str = "l1",
) -> dict[str, torch.Tensor]:
    _assert_finite_tensor("D_hat", D_hat)
    _assert_finite_tensor("D", D)

    rec = indexation_invariant_reconstruction_loss(
        D_hat,
        D,
        loss_type=reconstruction_loss,
    )
    diag = diagonal_loss(D_hat)
    nonneg = non_negative_loss(D_hat)
    sym = symmetry_loss(D_hat)

    total = rec + gamma * diag + delta * nonneg + epsilon * sym

    zero = torch.zeros((), device=D_hat.device, dtype=D_hat.dtype)

    components = {
        "total": total,
        "rec": rec,
        "kl": zero,
        "diag": diag,
        "nonneg": nonneg,
        "sym": sym,
    }

    for name, value in components.items():
        _assert_finite_component(name, value)

    return components
