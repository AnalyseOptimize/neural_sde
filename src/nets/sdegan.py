from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

try:
    import torchcde
    import torchsde
except ImportError as exc:  # pragma: no cover - exercised only in incomplete envs.
    raise ImportError(
        "SDEGAN requires torch, torchcde and torchsde. Install them before training."
    ) from exc

from utils.data import paths_to_coeffs, validate_ts as _validate_ts


class LipSwish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.909 * torch.nn.functional.silu(x)


class MLP(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        hidden_size: int,
        num_layers: int,
        *,
        final_tanh: bool = False,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        activation = LipSwish() if activation is None else activation
        layers: list[nn.Module] = [nn.Linear(in_size, hidden_size), activation]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), LipSwish()])
        layers.append(nn.Linear(hidden_size, out_size))
        if final_tanh:
            layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConstantCoefficientHead(nn.Module):
    def __init__(self, out_size: int, *, init_value: float = 1.0) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.full((out_size,), float(init_value)))

    def forward(self, t: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
        return self.value.view(1, -1).expand(window.size(0), -1)


class MLPCoefficientHead(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        hidden_size: int,
        num_layers: int,
        *,
        final_tanh: bool = True,
        output_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.output_scale = float(output_scale)
        self.net = MLP(
            in_size,
            out_size,
            hidden_size,
            num_layers,
            final_tanh=final_tanh,
        )

    def forward(self, t: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
        batch_size = window.size(0)
        t = t.expand(batch_size, 1)
        features = torch.cat([t, window.reshape(batch_size, -1)], dim=1)
        return self.output_scale * self.net(features)


def _build_coefficient_head(
    head_type: str,
    *,
    data_size: int,
    out_size: int,
    hidden_size: int,
    num_layers: int,
    window_size: int,
    init_value: float,
    final_tanh: bool,
    output_scale: float,
) -> nn.Module:
    head_type = head_type.lower()
    if head_type == "constant":
        return ConstantCoefficientHead(out_size, init_value=init_value)
    if head_type == "simple":
        return MLPCoefficientHead(
            1 + data_size,
            out_size,
            hidden_size,
            num_layers,
            final_tanh=final_tanh,
            output_scale=output_scale,
        )
    if head_type == "window":
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        return MLPCoefficientHead(
            1 + data_size * window_size,
            out_size,
            hidden_size,
            num_layers,
            final_tanh=final_tanh,
            output_scale=output_scale,
        )
    raise ValueError("head_type must be one of: constant, simple, window")


class ConditionalGeneratorFunc(nn.Module):
    sde_type = "stratonovich"

    def __init__(
        self,
        *,
        data_size: int,
        noise_size: int,
        noise_type: str,
        drift_head: str,
        diffusion_head: str,
        drift_window_size: int,
        diffusion_window_size: int,
        hidden_size: int,
        num_layers: int,
        drift_init: float,
        diffusion_init: float,
        drift_scale: float,
        diffusion_scale: float,
        final_tanh: bool,
    ) -> None:
        super().__init__()
        self.data_size = int(data_size)
        self.noise_size = int(noise_size)
        self.noise_type = noise_type
        self.drift_window_size = int(drift_window_size)
        self.diffusion_window_size = int(diffusion_window_size)
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
        )
        self.diffusion = _build_coefficient_head(
            diffusion_head,
            data_size=self.data_size,
            out_size=diffusion_out_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            window_size=self.diffusion_window_size,
            init_value=diffusion_init,
            final_tanh=final_tanh,
            output_scale=diffusion_scale,
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

    def f_and_g(self, t: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        drift = self.drift(t, self._window(y, window_size=self.drift_window_size))
        diffusion = self.diffusion(t, self._window(y, window_size=self.diffusion_window_size))
        if self.noise_type == "general":
            diffusion = diffusion.view(y.size(0), self.data_size, self.noise_size)
        return drift, diffusion


@dataclass(frozen=True)
class SDEGeneratorConfig:
    data_size: int
    noise_size: int = 1
    noise_type: str = "diagonal"
    drift_head: str = "simple"
    diffusion_head: str = "simple"
    drift_window_size: int = 1
    diffusion_window_size: int = 1
    hidden_size: int = 32
    num_layers: int = 2
    drift_init: float = 0.0
    diffusion_init: float = 0.1
    drift_scale: float = 1.0
    diffusion_scale: float = 1.0
    final_tanh: bool = True
    method: str = "reversible_heun"
    dt: Optional[float] = None


class SDEGenerator(nn.Module):
    def __init__(
        self,
        data_size: int,
        noise_size: int = 1,
        *,
        noise_type: str = "diagonal",
        drift_head: str = "simple",
        diffusion_head: str = "simple",
        drift_window_size: int = 1,
        diffusion_window_size: int = 1,
        hidden_size: int = 32,
        num_layers: int = 2,
        drift_init: float = 0.0,
        diffusion_init: float = 0.1,
        drift_scale: float = 1.0,
        diffusion_scale: float = 1.0,
        final_tanh: bool = True,
        method: str = "reversible_heun",
        dt: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.data_size = int(data_size)
        self.noise_size = int(noise_size)
        self.noise_type = noise_type.lower()
        self.drift_head_type = drift_head.lower()
        self.diffusion_head_type = diffusion_head.lower()
        self.drift_window_size = int(drift_window_size)
        self.diffusion_window_size = int(diffusion_window_size)
        self.method = method
        self.dt = dt

        if self.data_size < 1:
            raise ValueError("data_size must be >= 1")
        if self.noise_size < 1:
            raise ValueError("noise_size must be >= 1")
        if self.noise_type not in {"diagonal", "general"}:
            raise ValueError("noise_type must be either diagonal or general")
        if self.noise_type == "diagonal" and self.noise_size != self.data_size:
            self.noise_size = self.data_size

        self.func = ConditionalGeneratorFunc(
            data_size=self.data_size,
            noise_size=self.noise_size,
            noise_type=self.noise_type,
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
        )

    @classmethod
    def from_config(cls, config: SDEGeneratorConfig) -> "SDEGenerator":
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

    def _solver_dt(self, ts: torch.Tensor) -> float:
        if self.dt is not None:
            return float(self.dt)
        return float((ts[1:] - ts[:-1]).min().detach().cpu())

    def _sdeint_adjoint(self, y0: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        kwargs = {
            "method": self.method,
            "dt": self._solver_dt(ts),
        }
        if self.method == "reversible_heun":
            kwargs["adjoint_method"] = "adjoint_reversible_heun"
        return torchsde.sdeint_adjoint(self.func, y0, ts, **kwargs)

    def _make_fixed_lags(
        self,
        paths: list[torch.Tensor],
    ) -> torch.Tensor:
        values = []
        for lag_idx in range(1, self.max_window_size):
            source_idx = len(paths) - lag_idx
            if source_idx < 0:
                values.append(paths[0])
            else:
                values.append(paths[source_idx])
        return torch.stack(values, dim=1)

    def _sample_simple_or_constant(self, ts: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
        self.func.set_fixed_lags(None)
        ys = self._sdeint_adjoint(y0, ts)
        return ys.transpose(0, 1)

    def _sample_window(self, ts: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
        paths = [y0]
        for idx in range(1, ts.numel()):
            fixed_lags = self._make_fixed_lags(paths).to(device=y0.device, dtype=y0.dtype)
            self.func.set_fixed_lags(fixed_lags)
            interval_ts = ts[idx - 1 : idx + 1]
            interval_ys = self._sdeint_adjoint(paths[-1], interval_ts)
            paths.append(interval_ys[-1])
        self.func.set_fixed_lags(None)
        return torch.stack(paths, dim=1)

    def sample_paths(self, ts: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
        ts = _validate_ts(ts)
        if y0.ndim == 1:
            y0 = y0.unsqueeze(0)
        if y0.ndim != 2:
            raise ValueError(f"y0 must have shape (B, D) or (D,), got {tuple(y0.shape)}")
        if y0.size(-1) != self.data_size:
            raise ValueError(f"y0 has D={y0.size(-1)}, expected D={self.data_size}")

        y0 = y0.to(device=ts.device, dtype=ts.dtype)
        if self.needs_discrete_window:
            return self._sample_window(ts, y0)
        return self._sample_simple_or_constant(ts, y0)

    def forward(self, ts: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
        return paths_to_coeffs(ts, self.sample_paths(ts, y0))

    def apply_initialization_scale(
        self,
        initial_scale: float = 1.0,
        func_scale: float = 1.0,
    ) -> None:
        del initial_scale
        with torch.no_grad():
            for param in self.func.parameters():
                param.mul_(func_scale)

    def simulate(
        self,
        nsteps: int,
        nsims: int,
        y0: torch.Tensor,
        *,
        t0: float = 0.0,
        dt: float = 1.0,
        device: Optional[torch.device | str] = None,
    ) -> torch.Tensor:
        device = device if device is not None else next(self.parameters()).device
        y0 = torch.as_tensor(y0, device=device, dtype=torch.float32)
        if y0.ndim == 1:
            y0 = y0.unsqueeze(0).expand(nsims, -1)
        ts = t0 + dt * torch.arange(nsteps, device=device, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            return self.sample_paths(ts, y0)


class CDEDiscriminatorFunc(nn.Module):
    def __init__(
        self,
        data_size: int,
        hidden_size: int,
        mlp_size: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.data_size = int(data_size)
        self.hidden_size = int(hidden_size)
        self.net = MLP(
            1 + hidden_size,
            hidden_size * (1 + data_size),
            mlp_size,
            num_layers,
            final_tanh=True,
        )

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        t = t.expand(h.size(0), 1)
        th = torch.cat([t, h], dim=1)
        return self.net(th).view(h.size(0), self.hidden_size, 1 + self.data_size)


@dataclass(frozen=True)
class CDEDiscriminatorConfig:
    data_size: int
    hidden_size: int = 16
    mlp_size: int = 16
    num_layers: int = 1
    method: str = "reversible_heun"
    adjoint: bool = True
    dt: float = 1.0


class CDEDiscriminator(nn.Module):
    def __init__(
        self,
        data_size: int,
        hidden_size: int = 16,
        mlp_size: int = 16,
        num_layers: int = 1,
        *,
        method: str = "reversible_heun",
        adjoint: bool = True,
        dt: float = 1.0,
    ) -> None:
        super().__init__()
        self.data_size = int(data_size)
        self.method = method
        self.adjoint = bool(adjoint)
        self.dt = float(dt)

        self.initial = MLP(
            1 + data_size,
            hidden_size,
            mlp_size,
            num_layers,
            final_tanh=False,
        )
        self.func = CDEDiscriminatorFunc(data_size, hidden_size, mlp_size, num_layers)
        self.readout = nn.Linear(hidden_size, 1)

    @classmethod
    def from_config(cls, config: CDEDiscriminatorConfig) -> "CDEDiscriminator":
        return cls(**config.__dict__)

    def forward(self, ys_coeffs: torch.Tensor) -> torch.Tensor:
        if ys_coeffs.ndim != 3:
            raise ValueError(
                f"ys_coeffs must have shape (B, T, 1 + D), got {tuple(ys_coeffs.shape)}"
            )

        path = torchcde.LinearInterpolation(ys_coeffs)
        h0 = self.initial(path.evaluate(path.interval[0]))
        kwargs = {
            "method": self.method,
            "backend": "torchsde",
            "dt": self.dt,
            "adjoint": self.adjoint,
        }
        if self.adjoint:
            kwargs["adjoint_params"] = (ys_coeffs,) + tuple(self.func.parameters())
            if self.method == "reversible_heun":
                kwargs["adjoint_method"] = "adjoint_reversible_heun"

        hs = torchcde.cdeint(path, self.func, h0, path.interval, **kwargs)
        return self.readout(hs[:, -1]).squeeze(-1)

    def mean_score(self, ys_coeffs: torch.Tensor) -> torch.Tensor:
        return self(ys_coeffs).mean()


class SDEGAN(nn.Module):
    def __init__(
        self,
        generator: SDEGenerator,
        discriminator: CDEDiscriminator,
    ) -> None:
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

    def sample_paths(self, ts: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
        return self.generator.sample_paths(ts, y0)

    def sample_coeffs(self, ts: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
        return self.generator(ts, y0)


# Backwards-compatible aliases for the naming used in the old repository.
Generator = SDEGenerator
Discriminator = CDEDiscriminator
Discrimator = CDEDiscriminator
