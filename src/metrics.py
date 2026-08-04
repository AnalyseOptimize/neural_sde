from __future__ import annotations

from dataclasses import dataclass

import torch


def _as_paths(value, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if not torch.is_floating_point(tensor):
        tensor = tensor.float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(-1)
    if tensor.ndim != 3:
        raise ValueError(f"{name} must have shape (N, T, D) or (N, T), got {tuple(tensor.shape)}")
    if tensor.size(0) < 1 or tensor.size(1) < 2 or tensor.size(2) < 1:
        raise ValueError(f"{name} must have non-empty N, at least two time steps, and non-empty D")
    return tensor


def _validate_pair(real_paths, generated_paths) -> tuple[torch.Tensor, torch.Tensor]:
    real = _as_paths(real_paths, name="real_paths")
    generated = _as_paths(generated_paths, name="generated_paths")
    if real.shape != generated.shape:
        raise ValueError(
            "real_paths and generated_paths must have identical shape for a pathwise metric; "
            f"got {tuple(real.shape)} and {tuple(generated.shape)}"
        )
    generated = generated.to(device=real.device, dtype=real.dtype)
    return real, generated


def expected_supremum_squared_error(real_paths, generated_paths) -> torch.Tensor:
    """
    Estimate E sup_t ||S_t^real - S_t^generated||^2 from paired sample paths.

    Inputs have shape (N, T, D), or (N, T) for one-dimensional paths. For D > 1 the
    coordinate-wise squared error is summed before taking the supremum over time.
    """

    real, generated = _validate_pair(real_paths, generated_paths)
    squared_error = (real - generated).pow(2).sum(dim=-1)
    return squared_error.max(dim=1).values.mean()


def _quantile_w1_1d(
    real: torch.Tensor,
    generated: torch.Tensor,
    *,
    num_quantiles: int,
) -> torch.Tensor:
    if real.numel() == generated.numel():
        return (real.sort().values - generated.sort().values).abs().mean()

    q = torch.linspace(0.0, 1.0, num_quantiles, device=real.device, dtype=real.dtype)
    real_q = torch.quantile(real, q)
    generated_q = torch.quantile(generated, q)
    return (real_q - generated_q).abs().mean()


@dataclass(frozen=True)
class MarginalWasserstein1:
    by_time: torch.Tensor
    max_w1: torch.Tensor
    average_w1: torch.Tensor


def marginal_wasserstein1(
    real_paths,
    generated_paths,
    *,
    num_quantiles: int = 1024,
) -> MarginalWasserstein1:
    """
    Compute empirical W1 distances between marginal laws at each time point.

    Inputs have shape (N, T, D), or (N, T) for one-dimensional paths. The returned
    by_time tensor has shape (T, D). max_w1 and average_w1 aggregate over both time
    and coordinates.
    """

    if num_quantiles < 2:
        raise ValueError("num_quantiles must be >= 2")

    real = _as_paths(real_paths, name="real_paths")
    generated = _as_paths(generated_paths, name="generated_paths").to(
        device=real.device,
        dtype=real.dtype,
    )
    if real.shape[1:] != generated.shape[1:]:
        raise ValueError(
            "real_paths and generated_paths must have matching time and data dimensions; "
            f"got {tuple(real.shape[1:])} and {tuple(generated.shape[1:])}"
        )

    n_steps = real.size(1)
    data_size = real.size(2)
    values = torch.empty(n_steps, data_size, device=real.device, dtype=real.dtype)
    for time_idx in range(n_steps):
        for dim_idx in range(data_size):
            values[time_idx, dim_idx] = _quantile_w1_1d(
                real[:, time_idx, dim_idx],
                generated[:, time_idx, dim_idx],
                num_quantiles=num_quantiles,
            )

    return MarginalWasserstein1(
        by_time=values,
        max_w1=values.max(),
        average_w1=values.mean(),
    )


def compute_path_metrics(
    real_paths,
    generated_paths,
    *,
    num_quantiles: int = 1024,
) -> dict[str, torch.Tensor]:
    sup_error = expected_supremum_squared_error(real_paths, generated_paths)
    w1 = marginal_wasserstein1(
        real_paths,
        generated_paths,
        num_quantiles=num_quantiles,
    )
    return {
        "expected_supremum_squared_error": sup_error,
        "marginal_w1_by_time": w1.by_time,
        "marginal_w1_max": w1.max_w1,
        "marginal_w1_average": w1.average_w1,
    }
