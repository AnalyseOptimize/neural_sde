from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Protocol

import torch


class PathSimulator(Protocol):
    data_size: int

    def simulate(
        self,
        *,
        n_paths: int,
        n_steps: int,
        dt: float,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ts with shape (T,) and paths with shape (N, T, D)."""


def _as_1d_tensor(
    value: float | list[float] | tuple[float, ...] | torch.Tensor,
    *,
    size: int,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype).flatten()
    if tensor.numel() == 1:
        tensor = tensor.expand(size)
    if tensor.numel() != size:
        raise ValueError(f"{name} must be scalar or have length {size}, got {tensor.numel()}")
    return tensor


def _make_generator(device: torch.device, seed: Optional[int]) -> Optional[torch.Generator]:
    if seed is None:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


@dataclass(frozen=True)
class MultiDimensionalGBMSimulator:
    """Exact simulator for dS_i = mu_i S_i dt + sigma_i S_i dW_i."""

    s0: float | list[float] | tuple[float, ...] = 1.0
    mu: float | list[float] | tuple[float, ...] = 0.05
    sigma: float | list[float] | tuple[float, ...] = 0.2
    dim: int = 1
    corr: Optional[list[list[float]] | torch.Tensor] = None
    seed: Optional[int] = 0

    @property
    def data_size(self) -> int:
        return int(self.dim)

    def _correlation_cholesky(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.corr is None:
            return torch.eye(self.dim, device=device, dtype=dtype)

        corr = torch.as_tensor(self.corr, device=device, dtype=dtype)
        if corr.shape != (self.dim, self.dim):
            raise ValueError(f"corr must have shape ({self.dim}, {self.dim}), got {tuple(corr.shape)}")
        return torch.linalg.cholesky(corr)

    def simulate(
        self,
        *,
        n_paths: int,
        n_steps: int,
        dt: float,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.dim < 1:
            raise ValueError("dim must be >= 1")
        if n_paths < 1:
            raise ValueError("n_paths must be >= 1")
        if n_steps < 2:
            raise ValueError("n_steps must be >= 2")
        if dt <= 0:
            raise ValueError("dt must be positive")

        device = torch.device(device)
        generator = _make_generator(device, self.seed)
        ts = dt * torch.arange(n_steps, device=device, dtype=dtype)

        s0 = _as_1d_tensor(self.s0, size=self.dim, name="s0", device=device, dtype=dtype)
        mu = _as_1d_tensor(self.mu, size=self.dim, name="mu", device=device, dtype=dtype)
        sigma = _as_1d_tensor(self.sigma, size=self.dim, name="sigma", device=device, dtype=dtype)
        if bool(torch.any(sigma < 0)):
            raise ValueError("sigma must be non-negative")
        chol = self._correlation_cholesky(device=device, dtype=dtype)

        paths = torch.empty(n_paths, n_steps, self.dim, device=device, dtype=dtype)
        paths[:, 0, :] = s0

        drift = (mu - 0.5 * sigma.pow(2)) * dt
        sqrt_dt = math.sqrt(dt)
        for idx in range(1, n_steps):
            z = torch.randn(n_paths, self.dim, device=device, dtype=dtype, generator=generator)
            z = z @ chol.T
            paths[:, idx, :] = paths[:, idx - 1, :] * torch.exp(drift + sigma * sqrt_dt * z)

        return ts, paths


@dataclass(frozen=True)
class OUSimulator:
    """Exact one-dimensional OU simulator: dS = theta * (mu - S) dt + sigma dW."""

    s0: float = 1.0
    theta: float = 1.0
    mu: float = 0.0
    sigma: float = 0.3
    seed: Optional[int] = 0

    @property
    def data_size(self) -> int:
        return 1

    def simulate(
        self,
        *,
        n_paths: int,
        n_steps: int,
        dt: float,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if n_paths < 1:
            raise ValueError("n_paths must be >= 1")
        if n_steps < 2:
            raise ValueError("n_steps must be >= 2")
        if dt <= 0:
            raise ValueError("dt must be positive")
        if self.theta < 0:
            raise ValueError("theta must be non-negative")
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")

        device = torch.device(device)
        generator = _make_generator(device, self.seed)
        ts = dt * torch.arange(n_steps, device=device, dtype=dtype)
        paths = torch.empty(n_paths, n_steps, 1, device=device, dtype=dtype)
        paths[:, 0, 0] = float(self.s0)

        if abs(self.theta) < 1e-12:
            std = abs(self.sigma) * math.sqrt(dt)
            for idx in range(1, n_steps):
                eps = torch.randn(n_paths, 1, device=device, dtype=dtype, generator=generator)
                paths[:, idx, :] = paths[:, idx - 1, :] + std * eps
            return ts, paths

        exp_theta_dt = math.exp(-self.theta * dt)
        variance = self.sigma**2 * (1.0 - math.exp(-2.0 * self.theta * dt)) / (2.0 * self.theta)
        std = math.sqrt(max(variance, 0.0))

        for idx in range(1, n_steps):
            eps = torch.randn(n_paths, 1, device=device, dtype=dtype, generator=generator)
            mean = self.mu + (paths[:, idx - 1, :] - self.mu) * exp_theta_dt
            paths[:, idx, :] = mean + std * eps

        return ts, paths


@dataclass(frozen=True)
class DeterministicDriftSimulator:
    """One-dimensional deterministic process: dS = mu dt."""

    s0: float = 0.0
    mu: float = 0.1

    @property
    def data_size(self) -> int:
        return 1

    def simulate(
        self,
        *,
        n_paths: int,
        n_steps: int,
        dt: float,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if n_paths < 1:
            raise ValueError("n_paths must be >= 1")
        if n_steps < 2:
            raise ValueError("n_steps must be >= 2")
        if dt <= 0:
            raise ValueError("dt must be positive")

        device = torch.device(device)
        ts = dt * torch.arange(n_steps, device=device, dtype=dtype)
        path = self.s0 + self.mu * ts
        paths = path.view(1, n_steps, 1).expand(n_paths, -1, -1).clone()
        return ts, paths
