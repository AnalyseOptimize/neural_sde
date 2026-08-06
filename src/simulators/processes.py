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


def _validate_simulation_args(*, n_paths: int, n_steps: int, dt: float) -> None:
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")
    if n_steps < 2:
        raise ValueError("n_steps must be >= 2")
    if dt <= 0:
        raise ValueError("dt must be positive")


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


def sample_brownian_increments(
    *,
    n_paths: int,
    n_steps: int,
    dt: float,
    noise_size: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Return Brownian increments with shape (N, T - 1, M)."""

    _validate_simulation_args(n_paths=n_paths, n_steps=n_steps, dt=dt)
    if noise_size < 1:
        raise ValueError("noise_size must be >= 1")

    device = torch.device(device)
    generator = _make_generator(device, seed)
    return math.sqrt(dt) * torch.randn(
        n_paths,
        n_steps - 1,
        noise_size,
        device=device,
        dtype=dtype,
        generator=generator,
    )


def _prepare_brownian_increments(
    brownian_increments: Optional[torch.Tensor],
    *,
    n_paths: int,
    n_steps: int,
    dt: float,
    noise_size: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: Optional[int],
) -> torch.Tensor:
    if brownian_increments is None:
        return sample_brownian_increments(
            n_paths=n_paths,
            n_steps=n_steps,
            dt=dt,
            noise_size=noise_size,
            device=device,
            dtype=dtype,
            seed=seed,
        )

    increments = torch.as_tensor(brownian_increments, device=device, dtype=dtype)
    expected_shape = (n_paths, n_steps - 1, noise_size)
    if tuple(increments.shape) != expected_shape:
        raise ValueError(
            f"brownian_increments must have shape {expected_shape}, got {tuple(increments.shape)}"
        )
    return increments


@dataclass(frozen=True)
class ArithmeticBrownianMotionSimulator:
    """Exact simulator for dS_i = mu_i dt + sigma_i dW_i."""

    s0: float | list[float] | tuple[float, ...] = 0.0
    mu: float | list[float] | tuple[float, ...] = 0.0
    sigma: float | list[float] | tuple[float, ...] = 0.2
    dim: int = 1
    corr: Optional[list[list[float]] | torch.Tensor] = None
    seed: Optional[int] = 0

    @property
    def data_size(self) -> int:
        return int(self.dim)

    def constant_coefficients(self) -> dict[str, torch.Tensor]:
        mu = torch.as_tensor(self.mu, dtype=torch.float32).flatten()
        sigma = torch.as_tensor(self.sigma, dtype=torch.float32).flatten()
        if mu.numel() == 1:
            mu = mu.expand(self.dim)
        if sigma.numel() == 1:
            sigma = sigma.expand(self.dim)
        return {
            "drift": mu,
            "diffusion": sigma,
        }

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
        ts, paths, _ = self.simulate_with_brownian(
            n_paths=n_paths,
            n_steps=n_steps,
            dt=dt,
            device=device,
            dtype=dtype,
        )
        return ts, paths

    def simulate_with_brownian(
        self,
        *,
        n_paths: int,
        n_steps: int,
        dt: float,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        brownian_increments: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.dim < 1:
            raise ValueError("dim must be >= 1")
        _validate_simulation_args(n_paths=n_paths, n_steps=n_steps, dt=dt)

        device = torch.device(device)
        ts = dt * torch.arange(n_steps, device=device, dtype=dtype)
        s0 = _as_1d_tensor(self.s0, size=self.dim, name="s0", device=device, dtype=dtype)
        mu = _as_1d_tensor(self.mu, size=self.dim, name="mu", device=device, dtype=dtype)
        sigma = _as_1d_tensor(self.sigma, size=self.dim, name="sigma", device=device, dtype=dtype)
        if bool(torch.any(sigma < 0)):
            raise ValueError("sigma must be non-negative")

        base_increments = _prepare_brownian_increments(
            brownian_increments,
            n_paths=n_paths,
            n_steps=n_steps,
            dt=dt,
            noise_size=self.dim,
            device=device,
            dtype=dtype,
            seed=self.seed,
        )
        state_increments = base_increments @ self._correlation_cholesky(device=device, dtype=dtype).T

        paths = torch.empty(n_paths, n_steps, self.dim, device=device, dtype=dtype)
        paths[:, 0, :] = s0
        drift = mu * dt
        for idx in range(1, n_steps):
            paths[:, idx, :] = paths[:, idx - 1, :] + drift + sigma * state_increments[:, idx - 1, :]

        return ts, paths, base_increments


ABMSimulator = ArithmeticBrownianMotionSimulator


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
        ts, paths, _ = self.simulate_with_brownian(
            n_paths=n_paths,
            n_steps=n_steps,
            dt=dt,
            device=device,
            dtype=dtype,
        )
        return ts, paths

    def simulate_with_brownian(
        self,
        *,
        n_paths: int,
        n_steps: int,
        dt: float,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        brownian_increments: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.dim < 1:
            raise ValueError("dim must be >= 1")
        _validate_simulation_args(n_paths=n_paths, n_steps=n_steps, dt=dt)

        device = torch.device(device)
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
        base_increments = _prepare_brownian_increments(
            brownian_increments,
            n_paths=n_paths,
            n_steps=n_steps,
            dt=dt,
            noise_size=self.dim,
            device=device,
            dtype=dtype,
            seed=self.seed,
        )
        state_increments = base_increments @ chol.T
        for idx in range(1, n_steps):
            paths[:, idx, :] = paths[:, idx - 1, :] * torch.exp(
                drift + sigma * state_increments[:, idx - 1, :]
            )

        return ts, paths, base_increments


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

    def constant_coefficients(self) -> dict[str, torch.Tensor]:
        return {"diffusion": torch.tensor([float(self.sigma)], dtype=torch.float32)}

    def simulate(
        self,
        *,
        n_paths: int,
        n_steps: int,
        dt: float,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_simulation_args(n_paths=n_paths, n_steps=n_steps, dt=dt)
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

    def simulate_with_brownian(
        self,
        *,
        n_paths: int,
        n_steps: int,
        dt: float,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        brownian_increments: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Euler simulation driven by supplied Brownian grid increments."""

        _validate_simulation_args(n_paths=n_paths, n_steps=n_steps, dt=dt)
        if self.theta < 0:
            raise ValueError("theta must be non-negative")
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")

        device = torch.device(device)
        ts = dt * torch.arange(n_steps, device=device, dtype=dtype)
        increments = _prepare_brownian_increments(
            brownian_increments,
            n_paths=n_paths,
            n_steps=n_steps,
            dt=dt,
            noise_size=1,
            device=device,
            dtype=dtype,
            seed=self.seed,
        )

        paths = torch.empty(n_paths, n_steps, 1, device=device, dtype=dtype)
        paths[:, 0, 0] = float(self.s0)
        theta = float(self.theta)
        mu = float(self.mu)
        sigma = float(self.sigma)
        for idx in range(1, n_steps):
            prev = paths[:, idx - 1, :]
            paths[:, idx, :] = prev + theta * (mu - prev) * dt + sigma * increments[:, idx - 1, :]

        return ts, paths, increments


@dataclass(frozen=True)
class DeterministicDriftSimulator:
    """One-dimensional deterministic process: dS = mu dt."""

    s0: float = 0.0
    mu: float = 0.1

    @property
    def data_size(self) -> int:
        return 1

    def constant_coefficients(self) -> dict[str, torch.Tensor]:
        return {
            "drift": torch.tensor([float(self.mu)], dtype=torch.float32),
            "diffusion": torch.tensor([0.0], dtype=torch.float32),
        }

    def simulate(
        self,
        *,
        n_paths: int,
        n_steps: int,
        dt: float,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_simulation_args(n_paths=n_paths, n_steps=n_steps, dt=dt)

        device = torch.device(device)
        ts = dt * torch.arange(n_steps, device=device, dtype=dtype)
        path = self.s0 + self.mu * ts
        paths = path.view(1, n_steps, 1).expand(n_paths, -1, -1).clone()
        return ts, paths

    def simulate_with_brownian(
        self,
        *,
        n_paths: int,
        n_steps: int,
        dt: float,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        brownian_increments: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _validate_simulation_args(n_paths=n_paths, n_steps=n_steps, dt=dt)

        device = torch.device(device)
        ts, paths = self.simulate(
            n_paths=n_paths,
            n_steps=n_steps,
            dt=dt,
            device=device,
            dtype=dtype,
        )
        if brownian_increments is None:
            increments = torch.zeros(n_paths, n_steps - 1, 1, device=device, dtype=dtype)
        else:
            increments = _prepare_brownian_increments(
                brownian_increments,
                n_paths=n_paths,
                n_steps=n_steps,
                dt=dt,
                noise_size=1,
                device=device,
                dtype=dtype,
                seed=None,
            )
        return ts, paths, increments
