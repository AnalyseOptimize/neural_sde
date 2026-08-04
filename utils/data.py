from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    import torchcde
except ImportError as exc:  # pragma: no cover - exercised only in incomplete envs.
    raise ImportError("SDEGAN data pipeline requires torchcde.") from exc


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


@dataclass(frozen=True)
class PathDataConfig:
    dataset_size: int = 8192
    t_size: int = 64
    dt: float = 1.0
    normalize: bool = True
    shuffle: bool = True
    drop_last: bool = True


@dataclass(frozen=True)
class SDEGANData:
    ts: torch.Tensor
    paths: torch.Tensor
    y0: torch.Tensor
    coeffs: torch.Tensor
    dataloader: DataLoader

    @property
    def data_size(self) -> int:
        return int(self.paths.size(-1))


def validate_paths(paths: torch.Tensor) -> torch.Tensor:
    if paths.ndim == 2:
        paths = paths.unsqueeze(-1)
    if paths.ndim != 3:
        raise ValueError(f"paths must have shape (N, T, D), got {tuple(paths.shape)}")
    if paths.size(0) < 1 or paths.size(1) < 2 or paths.size(2) < 1:
        raise ValueError("paths must have non-empty N, at least two time steps, and non-empty D")
    return paths


def validate_ts(ts: torch.Tensor, *, expected_steps: int | None = None) -> torch.Tensor:
    if ts.ndim != 1:
        raise ValueError(f"ts must have shape (T,), got {tuple(ts.shape)}")
    if ts.numel() < 2:
        raise ValueError("ts must contain at least two time points")
    if expected_steps is not None and ts.numel() != expected_steps:
        raise ValueError(f"ts has T={ts.numel()}, expected T={expected_steps}")
    if not bool(torch.all(ts[1:] > ts[:-1])):
        raise ValueError("ts must be strictly increasing")
    return ts


def normalize_paths_by_initial(paths: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    paths = validate_paths(paths)
    y0 = paths[:, 0, :]
    mean = y0.mean(dim=0, keepdim=True).view(1, 1, -1)
    initial_std = y0.std(dim=0, keepdim=True, unbiased=False)
    path_std = paths.flatten(0, 1).std(dim=0, keepdim=True, unbiased=False)
    std = torch.where(initial_std > eps, initial_std, path_std).clamp_min(eps)
    std = std.view(1, 1, -1)
    return (paths - mean) / std


def add_time_channel(ts: torch.Tensor, paths: torch.Tensor) -> torch.Tensor:
    paths = validate_paths(paths)
    ts = validate_ts(ts, expected_steps=paths.size(1)).to(
        device=paths.device,
        dtype=paths.dtype,
    )
    time = ts.view(1, -1, 1).expand(paths.size(0), -1, 1)
    return torch.cat([time, paths], dim=-1)


def paths_to_coeffs(ts: torch.Tensor, paths: torch.Tensor) -> torch.Tensor:
    return torchcde.linear_interpolation_coeffs(add_time_channel(ts, paths))


def make_sdegan_dataset(
    *,
    simulator: PathSimulator,
    config: PathDataConfig,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> SDEGANData:
    if config.dataset_size < 1:
        raise ValueError("dataset_size must be >= 1")
    if config.t_size < 2:
        raise ValueError("t_size must be >= 2")
    if config.dt <= 0:
        raise ValueError("dt must be positive")

    ts, paths = simulator.simulate(
        n_paths=config.dataset_size,
        n_steps=config.t_size,
        dt=config.dt,
        device=device,
        dtype=dtype,
    )
    paths = validate_paths(paths)
    if config.normalize:
        paths = normalize_paths_by_initial(paths)

    coeffs = paths_to_coeffs(ts, paths)
    y0 = paths[:, 0, :]
    tensor_dataset = TensorDataset(coeffs, y0)
    dataloader = DataLoader(
        tensor_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return SDEGANData(ts=ts, paths=paths, y0=y0, coeffs=coeffs, dataloader=dataloader)
