from __future__ import annotations

"""Maximum-likelihood model for fully observed SDE paths.

The training objective is the conditional Gaussian Euler transition likelihood
of observed increments. Sampling can use either torchsde integration or the same
direct Euler scheme as the likelihood.
"""

from dataclasses import dataclass
import math
from typing import Literal, Optional

import torch
from torch import nn

try:
    import torchsde
except ImportError:  # pragma: no cover - direct Euler mode works without torchsde.
    torchsde = None

from src.nets.sdegan import LipSwish


Reduction = Literal["mean", "sum", "none"]
SamplingBackend = Literal["torchsde", "direct"]


def _validate_ts(ts: torch.Tensor, *, expected_steps: int | None = None) -> torch.Tensor:
    if ts.ndim != 1:
        raise ValueError(f"ts must have shape (T,), got {tuple(ts.shape)}")
    if ts.numel() < 2:
        raise ValueError("ts must contain at least two time points")
    if expected_steps is not None and ts.numel() != expected_steps:
        raise ValueError(f"ts has T={ts.numel()}, expected T={expected_steps}")
    if not bool(torch.all(ts[1:] > ts[:-1])):
        raise ValueError("ts must be strictly increasing")
    return ts


def _validate_paths(paths: torch.Tensor) -> torch.Tensor:
    if paths.ndim == 2:
        paths = paths.unsqueeze(-1)
    if paths.ndim != 3:
        raise ValueError(f"paths must have shape (B, T, D) or (B, T), got {tuple(paths.shape)}")
    if paths.size(0) < 1 or paths.size(1) < 2 or paths.size(2) < 1:
        raise ValueError("paths must have non-empty B, at least two time steps, and non-empty D")
    return paths


def _as_initial_value(
    value: float | list[float] | tuple[float, ...] | torch.Tensor,
    *,
    data_size: int,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).flatten()
    if tensor.numel() == 1:
        tensor = tensor.expand(data_size).clone()
    if tensor.numel() != data_size:
        raise ValueError(
            f"initial_value must be scalar or have length {data_size}, got {tensor.numel()}"
        )
    return tensor


def _as_1d_tensor(
    value: float | list[float] | tuple[float, ...] | torch.Tensor,
    *,
    size: int,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).flatten()
    if tensor.numel() == 1:
        tensor = tensor.expand(size).clone()
    if tensor.numel() != size:
        raise ValueError(f"{name} must be scalar or have length {size}, got {tensor.numel()}")
    return tensor


def _as_init_vector(
    value: float | list[float] | tuple[float, ...] | torch.Tensor,
    *,
    size: int,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).flatten()
    if tensor.numel() == 1:
        tensor = tensor.expand(size).clone()
    if tensor.numel() != size:
        raise ValueError(f"{name} must be scalar or have length {size}, got {tensor.numel()}")
    return tensor


def _general_diffusion_init(
    value: float | list[float] | tuple[float, ...] | torch.Tensor,
    *,
    data_size: int,
    noise_size: int,
) -> float | torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).flatten()
    if tensor.numel() != 1:
        return tensor

    matrix = torch.zeros(data_size, noise_size, dtype=torch.float32)
    diag_size = min(data_size, noise_size)
    diag_idx = torch.arange(diag_size)
    matrix[diag_idx, diag_idx] = tensor.item()
    return matrix.reshape(-1)


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp_min(1e-12)
    return torch.log(torch.expm1(value))


def _reduce(values: torch.Tensor, reduction: Reduction) -> torch.Tensor:
    if reduction == "none":
        return values
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    raise ValueError("reduction must be one of: mean, sum, none")


def _normalize_sampling_backend(backend: str) -> SamplingBackend:
    value = backend.lower()
    if value not in {"torchsde", "direct"}:
        raise ValueError("sampling_backend must be one of: torchsde, direct")
    return value  # type: ignore[return-value]


@dataclass(frozen=True)
class SDEMLConfig:
    data_size: int
    noise_size: int = 1
    noise_type: str = "diagonal"
    sde_type: str = "ito"
    drift_head: str = "simple"
    diffusion_head: str = "simple"
    drift_window_size: int = 1
    diffusion_window_size: int = 1
    hidden_size: int = 32
    num_layers: int = 2
    drift_init: float = 0.0
    diffusion_init: float | list[float] | tuple[float, ...] = 0.1
    drift_scale: float = 1.0
    diffusion_scale: float = 1.0
    final_tanh: bool = True
    variance_floor: float = 1e-6
    diffusion_min: float = 1e-4
    initial_value: float | tuple[float, ...] = 0.0
    learn_initial: bool = False
    time_origin: float = 0.0
    time_scale: float = 1.0
    state_center: float | tuple[float, ...] = 0.0
    state_scale: float | tuple[float, ...] = 1.0
    input_clip: Optional[float] = 20.0
    method: str = "euler"
    dt: Optional[float] = None
    adjoint: bool = True
    sampling_backend: SamplingBackend = "torchsde"


class ConstantCoefficientHead(nn.Module):
    def __init__(
        self,
        out_size: int,
        *,
        init_value: float | list[float] | tuple[float, ...] | torch.Tensor,
        output_scale: float,
        positive: bool = False,
        min_value: float = 0.0,
    ) -> None:
        super().__init__()
        self.output_scale = float(output_scale)
        self.positive = bool(positive)
        self.min_value = float(min_value)
        if self.positive and self.output_scale <= 0:
            raise ValueError("output_scale must be positive for positive coefficient heads")

        init = _as_init_vector(init_value, size=out_size, name="init_value")
        if self.positive:
            target = (init - self.min_value) / self.output_scale
            raw_init = _inverse_softplus(target)
        else:
            raw_init = init / self.output_scale
        self.raw_value = nn.Parameter(raw_init)

    def _transform(self, raw: torch.Tensor) -> torch.Tensor:
        if self.positive:
            return self.min_value + self.output_scale * torch.nn.functional.softplus(raw)
        return self.output_scale * raw

    def forward(self, t: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
        del t
        value = self._transform(self.raw_value)
        return value.view(1, -1).expand(window.size(0), -1)


class MLPCoefficientHead(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        hidden_size: int,
        num_layers: int,
        *,
        init_value: float | list[float] | tuple[float, ...] | torch.Tensor,
        output_scale: float,
        final_tanh: bool = False,
        positive: bool = False,
        min_value: float = 0.0,
        final_weight_std: float = 1e-3,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.output_scale = float(output_scale)
        self.positive = bool(positive)
        self.min_value = float(min_value)
        if self.positive and self.output_scale <= 0:
            raise ValueError("output_scale must be positive for positive coefficient heads")

        layers: list[nn.Module] = [nn.Linear(in_size, hidden_size), LipSwish()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), LipSwish()])
        self.features = nn.Sequential(*layers)
        self.final = nn.Linear(hidden_size, out_size)
        self.final_tanh = bool(final_tanh) and not self.positive

        nn.init.normal_(self.final.weight, mean=0.0, std=float(final_weight_std))
        init = _as_init_vector(init_value, size=out_size, name="init_value")
        if self.positive:
            target = (init - self.min_value) / self.output_scale
            raw_bias = _inverse_softplus(target)
        else:
            raw_bias = init / self.output_scale
        with torch.no_grad():
            self.final.bias.copy_(raw_bias)

    def _transform(self, raw: torch.Tensor) -> torch.Tensor:
        if self.final_tanh:
            raw = torch.tanh(raw)
        if self.positive:
            return self.min_value + self.output_scale * torch.nn.functional.softplus(raw)
        return self.output_scale * raw

    def forward(self, t: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
        batch_size = window.size(0)
        t = t.expand(batch_size, 1)
        features = torch.cat([t, window.reshape(batch_size, -1)], dim=1)
        raw = self.final(self.features(features))
        return self._transform(raw)


def _build_coefficient_head(
    head_type: str,
    *,
    data_size: int,
    out_size: int,
    hidden_size: int,
    num_layers: int,
    window_size: int,
    init_value: float | list[float] | tuple[float, ...] | torch.Tensor,
    final_tanh: bool,
    output_scale: float,
    positive: bool,
    min_value: float,
) -> nn.Module:
    head_type = head_type.lower()
    if head_type == "constant":
        return ConstantCoefficientHead(
            out_size,
            init_value=init_value,
            output_scale=output_scale,
            positive=positive,
            min_value=min_value,
        )
    if head_type == "simple":
        return MLPCoefficientHead(
            1 + data_size,
            out_size,
            hidden_size,
            num_layers,
            init_value=init_value,
            final_tanh=final_tanh,
            output_scale=output_scale,
            positive=positive,
            min_value=min_value,
        )
    if head_type == "window":
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        return MLPCoefficientHead(
            1 + data_size * window_size,
            out_size,
            hidden_size,
            num_layers,
            init_value=init_value,
            final_tanh=final_tanh,
            output_scale=output_scale,
            positive=positive,
            min_value=min_value,
        )
    raise ValueError("head_type must be one of: constant, simple, window")


class ObservableSDEFunc(nn.Module):
    def __init__(
        self,
        *,
        data_size: int,
        noise_size: int,
        noise_type: str,
        sde_type: str,
        drift_head: str,
        diffusion_head: str,
        drift_window_size: int,
        diffusion_window_size: int,
        hidden_size: int,
        num_layers: int,
        drift_init: float,
        diffusion_init: float | list[float] | tuple[float, ...] | torch.Tensor,
        drift_scale: float,
        diffusion_scale: float,
        final_tanh: bool,
        diffusion_min: float,
        time_origin: float,
        time_scale: float,
        state_center: float | list[float] | tuple[float, ...] | torch.Tensor,
        state_scale: float | list[float] | tuple[float, ...] | torch.Tensor,
        input_clip: Optional[float],
    ) -> None:
        super().__init__()
        self.data_size = int(data_size)
        self.noise_size = int(noise_size)
        self.noise_type = noise_type.lower()
        self.sde_type = sde_type.lower()
        self.drift_window_size = int(drift_window_size)
        self.diffusion_window_size = int(diffusion_window_size)
        self.diffusion_min = float(diffusion_min)
        self.input_clip = None if input_clip is None else float(input_clip)

        if self.data_size < 1:
            raise ValueError("data_size must be >= 1")
        if self.noise_size < 1:
            raise ValueError("noise_size must be >= 1")
        if self.noise_type not in {"diagonal", "general"}:
            raise ValueError("noise_type must be either diagonal or general")
        if self.sde_type not in {"ito", "stratonovich"}:
            raise ValueError("sde_type must be either ito or stratonovich")
        if self.diffusion_min < 0:
            raise ValueError("diffusion_min must be non-negative")
        if time_scale <= 0:
            raise ValueError("time_scale must be positive")
        if self.input_clip is not None and self.input_clip <= 0:
            raise ValueError("input_clip must be positive when provided")
        if self.noise_type == "diagonal" and self.noise_size != self.data_size:
            self.noise_size = self.data_size

        state_center_tensor = _as_1d_tensor(
            state_center,
            size=self.data_size,
            name="state_center",
        )
        state_scale_tensor = _as_1d_tensor(
            state_scale,
            size=self.data_size,
            name="state_scale",
        )
        if bool(torch.any(state_scale_tensor <= 0)):
            raise ValueError("state_scale entries must be positive")
        self.register_buffer("time_origin", torch.tensor(float(time_origin)))
        self.register_buffer("time_scale", torch.tensor(float(time_scale)))
        self.register_buffer("state_center", state_center_tensor.view(1, 1, -1))
        self.register_buffer("state_scale", state_scale_tensor.view(1, 1, -1))

        diffusion_out_size = (
            self.data_size if self.noise_type == "diagonal" else self.data_size * self.noise_size
        )
        self.drift = _build_coefficient_head(
            drift_head,
            data_size=self.data_size,
            out_size=self.data_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            window_size=self.drift_window_size,
            init_value=drift_init,
            final_tanh=final_tanh,
            output_scale=drift_scale,
            positive=False,
            min_value=0.0,
        )
        self.diffusion = _build_coefficient_head(
            diffusion_head,
            data_size=self.data_size,
            out_size=diffusion_out_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            window_size=self.diffusion_window_size,
            init_value=(
                _general_diffusion_init(
                    diffusion_init,
                    data_size=self.data_size,
                    noise_size=self.noise_size,
                )
                if self.noise_type == "general"
                else diffusion_init
            ),
            final_tanh=final_tanh,
            output_scale=diffusion_scale,
            positive=self.noise_type == "diagonal",
            min_value=self.diffusion_min if self.noise_type == "diagonal" else 0.0,
        )
        self._fixed_lags: Optional[torch.Tensor] = None

    @property
    def max_window_size(self) -> int:
        return max(self.drift_window_size, self.diffusion_window_size, 1)

    def set_fixed_lags(self, fixed_lags: Optional[torch.Tensor]) -> None:
        self._fixed_lags = fixed_lags

    def _window(self, y: torch.Tensor, *, window_size: int) -> torch.Tensor:
        if window_size == 1:
            return y.unsqueeze(1)

        if self._fixed_lags is None:
            lags = y.unsqueeze(1).expand(y.size(0), window_size - 1, y.size(1))
        else:
            lags = self._fixed_lags[:, : window_size - 1, :]
            if lags.size(1) < window_size - 1:
                pad = lags[:, -1:, :].expand(-1, window_size - 1 - lags.size(1), -1)
                lags = torch.cat([lags, pad], dim=1)
        return torch.cat([y.unsqueeze(1), lags], dim=1)

    def _observed_window(
        self,
        paths: torch.Tensor,
        time_idx: int,
        *,
        window_size: int,
    ) -> torch.Tensor:
        if window_size == 1:
            return paths[:, time_idx : time_idx + 1, :]

        values = []
        for lag_idx in range(window_size):
            source_idx = max(time_idx - lag_idx, 0)
            values.append(paths[:, source_idx, :])
        return torch.stack(values, dim=1)

    def _normalize_t(self, t: torch.Tensor) -> torch.Tensor:
        normalized = (t - self.time_origin.to(device=t.device, dtype=t.dtype)) / self.time_scale.to(
            device=t.device,
            dtype=t.dtype,
        )
        if self.input_clip is not None:
            normalized = normalized.clamp(-self.input_clip, self.input_clip)
        return normalized

    def _normalize_window(self, window: torch.Tensor) -> torch.Tensor:
        center = self.state_center.to(device=window.device, dtype=window.dtype)
        scale = self.state_scale.to(device=window.device, dtype=window.dtype)
        normalized = (window - center) / scale
        if self.input_clip is not None:
            normalized = normalized.clamp(-self.input_clip, self.input_clip)
        return normalized

    def _format_diffusion(self, diffusion: torch.Tensor) -> torch.Tensor:
        if self.noise_type == "general":
            return diffusion.view(diffusion.size(0), self.data_size, self.noise_size)
        return diffusion

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        window = self._normalize_window(self._window(y, window_size=self.drift_window_size))
        return self.drift(self._normalize_t(t), window)

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        window = self._normalize_window(self._window(y, window_size=self.diffusion_window_size))
        diffusion = self.diffusion(self._normalize_t(t), window)
        return self._format_diffusion(diffusion)

    def h(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        del t
        return torch.zeros_like(y)

    def f_and_g(self, t: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        drift = self.f(t, y)
        diffusion = self.g(t, y)
        return drift, diffusion

    def coefficients_on_path(
        self,
        ts: torch.Tensor,
        paths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ts = _validate_ts(ts, expected_steps=paths.size(1))
        paths = _validate_paths(paths)

        drifts = []
        diffusions = []
        for time_idx in range(paths.size(1) - 1):
            t = ts[time_idx]
            drift_window = self._observed_window(
                paths,
                time_idx,
                window_size=self.drift_window_size,
            )
            diffusion_window = self._observed_window(
                paths,
                time_idx,
                window_size=self.diffusion_window_size,
            )
            drift = self.drift(self._normalize_t(t), self._normalize_window(drift_window))
            diffusion = self.diffusion(
                self._normalize_t(t),
                self._normalize_window(diffusion_window),
            )
            drifts.append(drift)
            diffusions.append(self._format_diffusion(diffusion))

        return torch.stack(drifts, dim=1), torch.stack(diffusions, dim=1)


class SDEML(nn.Module):
    def __init__(
        self,
        data_size: int,
        noise_size: int = 1,
        *,
        noise_type: str = "diagonal",
        sde_type: str = "ito",
        drift_head: str = "simple",
        diffusion_head: str = "simple",
        drift_window_size: int = 1,
        diffusion_window_size: int = 1,
        hidden_size: int = 32,
        num_layers: int = 2,
        drift_init: float = 0.0,
        diffusion_init: float | list[float] | tuple[float, ...] | torch.Tensor = 0.1,
        drift_scale: float = 1.0,
        diffusion_scale: float = 1.0,
        final_tanh: bool = True,
        variance_floor: float = 1e-6,
        diffusion_min: float = 1e-4,
        initial_value: float | list[float] | tuple[float, ...] | torch.Tensor = 0.0,
        learn_initial: bool = False,
        time_origin: float = 0.0,
        time_scale: float = 1.0,
        state_center: float | list[float] | tuple[float, ...] | torch.Tensor = 0.0,
        state_scale: float | list[float] | tuple[float, ...] | torch.Tensor = 1.0,
        input_clip: Optional[float] = 20.0,
        method: str = "euler",
        dt: Optional[float] = None,
        adjoint: bool = True,
        sampling_backend: SamplingBackend = "torchsde",
    ) -> None:
        super().__init__()
        self.data_size = int(data_size)
        self.noise_size = int(noise_size)
        self.noise_type = noise_type.lower()
        self.sde_type = sde_type.lower()
        self.drift_head_type = drift_head.lower()
        self.diffusion_head_type = diffusion_head.lower()
        self.drift_window_size = int(drift_window_size)
        self.diffusion_window_size = int(diffusion_window_size)
        self.variance_floor = float(variance_floor)
        self.method = method
        self.dt = dt
        self.adjoint = bool(adjoint)
        self.sampling_backend = _normalize_sampling_backend(str(sampling_backend))

        if self.variance_floor <= 0:
            raise ValueError("variance_floor must be positive")

        initial = _as_initial_value(initial_value, data_size=self.data_size)
        if learn_initial:
            self.initial_value = nn.Parameter(initial)
        else:
            self.register_buffer("initial_value", initial)

        self.func = ObservableSDEFunc(
            data_size=self.data_size,
            noise_size=self.noise_size,
            noise_type=self.noise_type,
            sde_type=self.sde_type,
            drift_head=self.drift_head_type,
            diffusion_head=self.diffusion_head_type,
            drift_window_size=self.drift_window_size,
            diffusion_window_size=self.diffusion_window_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            drift_init=drift_init,
            diffusion_init=diffusion_init,
            drift_scale=drift_scale,
            diffusion_scale=diffusion_scale,
            final_tanh=final_tanh,
            diffusion_min=diffusion_min,
            time_origin=time_origin,
            time_scale=time_scale,
            state_center=state_center,
            state_scale=state_scale,
            input_clip=input_clip,
        )
        self.noise_size = self.func.noise_size

    @classmethod
    def from_config(cls, config: SDEMLConfig) -> "SDEML":
        return cls(**config.__dict__)

    @property
    def max_window_size(self) -> int:
        return self.func.max_window_size

    @property
    def needs_discrete_window(self) -> bool:
        return (
            self.drift_head_type == "window" and self.drift_window_size > 1
        ) or (
            self.diffusion_head_type == "window" and self.diffusion_window_size > 1
        )

    def initial_point(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        value = self.initial_value
        if device is not None or dtype is not None:
            value = value.to(device=device, dtype=dtype)
        return value.view(1, -1).expand(batch_size, -1)

    def _solver_dt(self, ts: torch.Tensor) -> float:
        if self.dt is not None:
            return float(self.dt)
        return float((ts[1:] - ts[:-1]).min().detach().cpu())

    def _sdeint(self, y0: torch.Tensor, ts: torch.Tensor, *, bm=None) -> torch.Tensor:
        if torchsde is None:
            raise ImportError("torchsde is required for sampling_backend='torchsde'.")
        kwargs = {
            "method": self.method,
            "dt": self._solver_dt(ts),
        }
        if bm is not None:
            kwargs["bm"] = bm
        if self.adjoint:
            if self.method == "reversible_heun":
                kwargs["adjoint_method"] = "adjoint_reversible_heun"
            return torchsde.sdeint_adjoint(self.func, y0, ts, **kwargs)
        return torchsde.sdeint(self.func, y0, ts, **kwargs)

    def _validate_brownian_increments(
        self,
        brownian_increments: torch.Tensor,
        *,
        batch_size: int,
        n_steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        increments = torch.as_tensor(brownian_increments, device=device, dtype=dtype)
        expected_shape = (batch_size, n_steps - 1, self.noise_size)
        if tuple(increments.shape) != expected_shape:
            raise ValueError(
                f"brownian_increments must have shape {expected_shape}, got {tuple(increments.shape)}"
            )
        return increments

    def _brownian_increments_from_bm(self, bm, ts: torch.Tensor, batch_size: int) -> torch.Tensor:
        increments = []
        for idx in range(ts.numel() - 1):
            left = float(ts[idx].detach().cpu().item())
            right = float(ts[idx + 1].detach().cpu().item())
            increments.append(bm(left, right))
        brownian_increments = torch.stack(increments, dim=1)
        return self._validate_brownian_increments(
            brownian_increments,
            batch_size=batch_size,
            n_steps=ts.numel(),
            device=ts.device,
            dtype=ts.dtype,
        )

    def _sample_direct(
        self,
        ts: torch.Tensor,
        y0: torch.Tensor,
        *,
        bm=None,
        brownian_increments: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if self.sde_type != "ito":
            raise ValueError("Direct Euler sampling is implemented for Ito SDEs only.")
        if bm is not None and brownian_increments is not None:
            raise ValueError("Pass either bm or brownian_increments, not both.")

        batch_size = y0.size(0)
        if bm is not None:
            increments = self._brownian_increments_from_bm(bm, ts, batch_size)
        elif brownian_increments is not None:
            increments = self._validate_brownian_increments(
                brownian_increments,
                batch_size=batch_size,
                n_steps=ts.numel(),
                device=ts.device,
                dtype=ts.dtype,
            )
        else:
            dt = (ts[1:] - ts[:-1]).sqrt().view(1, -1, 1)
            increments = dt * torch.randn(
                batch_size,
                ts.numel() - 1,
                self.noise_size,
                device=ts.device,
                dtype=ts.dtype,
                generator=generator,
            )

        paths = [y0]
        try:
            for idx in range(1, ts.numel()):
                if self.needs_discrete_window:
                    fixed_lags = self._make_fixed_lags(paths).to(device=y0.device, dtype=y0.dtype)
                    self.func.set_fixed_lags(fixed_lags)
                else:
                    self.func.set_fixed_lags(None)

                t = ts[idx - 1]
                dt = ts[idx] - ts[idx - 1]
                prev = paths[-1]
                drift, diffusion = self.func.f_and_g(t, prev)
                d_w = increments[:, idx - 1, :]
                if self.noise_type == "diagonal":
                    stochastic = diffusion * d_w
                else:
                    stochastic = torch.bmm(diffusion, d_w.unsqueeze(-1)).squeeze(-1)
                paths.append(prev + drift * dt + stochastic)
        finally:
            self.func.set_fixed_lags(None)

        return torch.stack(paths, dim=1)

    def _make_fixed_lags(self, paths: list[torch.Tensor]) -> torch.Tensor:
        values = []
        for lag_idx in range(1, self.max_window_size):
            source_idx = len(paths) - lag_idx
            if source_idx < 0:
                values.append(paths[0])
            else:
                values.append(paths[source_idx])
        return torch.stack(values, dim=1)

    def _sample_simple_or_constant(
        self,
        ts: torch.Tensor,
        y0: torch.Tensor,
        *,
        bm=None,
    ) -> torch.Tensor:
        self.func.set_fixed_lags(None)
        ys = self._sdeint(y0, ts, bm=bm)
        return ys.transpose(0, 1)

    def _sample_window(self, ts: torch.Tensor, y0: torch.Tensor, *, bm=None) -> torch.Tensor:
        paths = [y0]
        for idx in range(1, ts.numel()):
            fixed_lags = self._make_fixed_lags(paths).to(device=y0.device, dtype=y0.dtype)
            self.func.set_fixed_lags(fixed_lags)
            interval_ts = ts[idx - 1 : idx + 1]
            interval_ys = self._sdeint(paths[-1], interval_ts, bm=bm)
            paths.append(interval_ys[-1])
        self.func.set_fixed_lags(None)
        return torch.stack(paths, dim=1)

    def sample_paths(
        self,
        ts: torch.Tensor,
        y0: Optional[torch.Tensor] = None,
        *,
        batch_size: Optional[int] = None,
        bm=None,
        brownian_increments: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        backend: Optional[SamplingBackend] = None,
    ) -> torch.Tensor:
        ts = _validate_ts(ts)
        if y0 is None:
            if batch_size is None:
                raise ValueError("batch_size is required when y0 is not provided")
            y0 = self.initial_point(batch_size, device=ts.device, dtype=ts.dtype)
        elif y0.ndim == 1:
            y0 = y0.unsqueeze(0)

        if y0.ndim != 2:
            raise ValueError(f"y0 must have shape (B, D) or (D,), got {tuple(y0.shape)}")
        if y0.size(-1) != self.data_size:
            raise ValueError(f"y0 has D={y0.size(-1)}, expected D={self.data_size}")

        y0 = y0.to(device=ts.device, dtype=ts.dtype)
        backend = self.sampling_backend if backend is None else _normalize_sampling_backend(str(backend))
        if backend == "direct":
            return self._sample_direct(
                ts,
                y0,
                bm=bm,
                brownian_increments=brownian_increments,
                generator=generator,
            )
        if brownian_increments is not None:
            raise ValueError("brownian_increments can only be used with sampling_backend='direct'.")
        if self.needs_discrete_window:
            return self._sample_window(ts, y0, bm=bm)
        return self._sample_simple_or_constant(ts, y0, bm=bm)

    def coefficients_on_path(
        self,
        ts: torch.Tensor,
        paths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        paths = _validate_paths(paths)
        ts = _validate_ts(ts, expected_steps=paths.size(1)).to(
            device=paths.device,
            dtype=paths.dtype,
        )
        return self.func.coefficients_on_path(ts, paths)

    def transition_log_likelihood(self, ts: torch.Tensor, paths: torch.Tensor) -> torch.Tensor:
        paths = _validate_paths(paths)
        ts = _validate_ts(ts, expected_steps=paths.size(1)).to(
            device=paths.device,
            dtype=paths.dtype,
        )

        drift, diffusion = self.coefficients_on_path(ts, paths)
        dt = (ts[1:] - ts[:-1]).view(1, -1, 1)
        dy = paths[:, 1:, :] - paths[:, :-1, :]
        mean = drift * dt
        residual = dy - mean

        if self.noise_type == "diagonal":
            variance = diffusion.pow(2) * dt + self.variance_floor
            log_prob = -0.5 * (
                residual.pow(2) / variance + torch.log(variance) + math.log(2.0 * math.pi)
            )
            return log_prob.sum(dim=-1)

        covariance = diffusion @ diffusion.transpose(-1, -2)
        covariance = covariance * dt.unsqueeze(-1)
        eye = torch.eye(self.data_size, device=paths.device, dtype=paths.dtype).view(
            1,
            1,
            self.data_size,
            self.data_size,
        )
        covariance = covariance + self.variance_floor * eye

        batch_size, steps_minus_one, data_size = residual.shape
        flat_covariance = covariance.reshape(batch_size * steps_minus_one, data_size, data_size)
        flat_residual = residual.reshape(batch_size * steps_minus_one, data_size, 1)
        cholesky = torch.linalg.cholesky(flat_covariance)
        solved = torch.cholesky_solve(flat_residual, cholesky)
        quadratic = (flat_residual.transpose(-1, -2) @ solved).view(batch_size, steps_minus_one)
        log_det = (
            2.0
            * torch.log(torch.diagonal(cholesky, dim1=-2, dim2=-1)).sum(dim=-1)
        ).view(batch_size, steps_minus_one)
        return -0.5 * (data_size * math.log(2.0 * math.pi) + log_det + quadratic)

    def log_likelihood(
        self,
        ts: torch.Tensor,
        paths: torch.Tensor,
        *,
        reduction: Reduction = "mean",
        include_initial: bool = False,
        initial_std: Optional[float] = None,
    ) -> torch.Tensor:
        paths = _validate_paths(paths)
        transition_log_prob = self.transition_log_likelihood(ts, paths).sum(dim=1)

        if include_initial:
            if initial_std is None or initial_std <= 0:
                raise ValueError("initial_std must be positive when include_initial=True")
            y0_mean = self.initial_point(
                paths.size(0),
                device=paths.device,
                dtype=paths.dtype,
            )
            variance = float(initial_std) ** 2
            initial_residual = paths[:, 0, :] - y0_mean
            initial_log_prob = -0.5 * (
                initial_residual.pow(2) / variance
                + math.log(2.0 * math.pi * variance)
            ).sum(dim=-1)
            transition_log_prob = transition_log_prob + initial_log_prob

        return _reduce(transition_log_prob, reduction)

    def negative_log_likelihood(
        self,
        ts: torch.Tensor,
        paths: torch.Tensor,
        *,
        reduction: Reduction = "mean",
        include_initial: bool = False,
        initial_std: Optional[float] = None,
    ) -> torch.Tensor:
        log_likelihood = self.log_likelihood(
            ts,
            paths,
            reduction="none",
            include_initial=include_initial,
            initial_std=initial_std,
        )
        return _reduce(-log_likelihood, reduction)

    def direct_negative_log_likelihood(
        self,
        ts: torch.Tensor,
        paths: torch.Tensor,
        *,
        reduction: Reduction = "mean",
    ) -> torch.Tensor:
        """Conditional Euler NLL of observed paths given their initial points."""

        return self.negative_log_likelihood(
            ts,
            paths,
            reduction=reduction,
            include_initial=False,
            initial_std=None,
        )

    def forward(self, ts: torch.Tensor, paths: torch.Tensor) -> torch.Tensor:
        return self.negative_log_likelihood(ts, paths, reduction="mean")


SDEMaximumLikelihood = SDEML
