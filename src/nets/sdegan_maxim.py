from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional

import torch
from torch import nn

try:
    import torchsde
except ImportError as exc:  # pragma: no cover - exercised only in incomplete envs.
    raise ImportError("Maxim SDEGAN requires torchsde.") from exc


REFERENCE_ROOT = Path(__file__).resolve().parents[2] / "old_versions/neuralSDE(special)/GIT2/gensde"
if REFERENCE_ROOT.exists() and str(REFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(REFERENCE_ROOT))

try:
    from gensde.models.generative import sdegan as maxim_ref
except ImportError as exc:  # pragma: no cover - exercised only when old_versions is missing.
    raise ImportError(
        "Cannot import the reference Maxim SDEGAN code from "
        f"{REFERENCE_ROOT}. Check old_versions/neuralSDE(special)."
    ) from exc


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


class ConstantMap(nn.Module):
    """Reference-compatible constant map used as drift or diffusion head."""

    def __init__(
        self,
        out_size: int,
        *,
        init_value: float | list[float] | tuple[float, ...] | torch.Tensor,
    ) -> None:
        super().__init__()
        self.value = nn.Parameter(_as_init_vector(init_value, size=out_size, name="init_value"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value.view(1, -1).expand(x.size(0), -1)


class ReferenceHead(nn.Module):
    """Adapter around the old MLP/constant map with optional hidden-state windows."""

    def __init__(
        self,
        *,
        head_type: str,
        hidden_size: int,
        out_size: int,
        mlp_size: int,
        num_layers: int,
        window_size: int,
        init_value: float | list[float] | tuple[float, ...] | torch.Tensor,
        tanh: bool,
    ) -> None:
        super().__init__()
        self.head_type = head_type.lower()
        self.hidden_size = int(hidden_size)
        self.window_size = int(window_size)
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")

        if self.head_type == "constant":
            self.module = ConstantMap(out_size, init_value=init_value)
        elif self.head_type in {"simple", "window"}:
            in_size = 1 + self.hidden_size
            if self.head_type == "window":
                in_size = 1 + self.hidden_size * self.window_size
            self.module = maxim_ref.MLP(
                in_size=in_size,
                out_size=out_size,
                mlp_size=mlp_size,
                num_layers=num_layers,
                tanh=tanh,
            )
        else:
            raise ValueError("head_type must be one of: constant, simple, window")

    @property
    def needs_window(self) -> bool:
        return self.head_type == "window" and self.window_size > 1

    def forward(self, t: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
        batch_size = window.size(0)
        if self.head_type == "constant":
            return self.module(window[:, 0, :])
        t = t.expand(batch_size, 1)
        if self.head_type == "simple":
            features = torch.cat([t, window[:, 0, :]], dim=1)
        else:
            features = torch.cat([t, window.reshape(batch_size, -1)], dim=1)
        return self.module(features)


class MaximGeneratorFunc(nn.Module):
    """Hidden-state SDE function following the reference GeneratorFunc contract."""

    sde_type = "stratonovich"

    def __init__(
        self,
        *,
        noise_size: int,
        hidden_size: int,
        noise_type: str,
        drift: ReferenceHead,
        diffusion: ReferenceHead,
    ) -> None:
        super().__init__()
        self.noise_type = noise_type.lower()
        self._noise_size = int(noise_size)
        self._hidden_size = int(hidden_size)
        self.drift = drift
        self.diffusion = diffusion
        self._fixed_lags: Optional[torch.Tensor] = None
        if self.noise_type not in {"general", "diagonal"}:
            raise ValueError("noise_type must be either general or diagonal")

    @property
    def max_window_size(self) -> int:
        return max(self.drift.window_size, self.diffusion.window_size, 1)

    @property
    def needs_discrete_window(self) -> bool:
        return self.drift.needs_window or self.diffusion.needs_window

    def set_fixed_lags(self, fixed_lags: Optional[torch.Tensor]) -> None:
        self._fixed_lags = fixed_lags

    def _window(self, x: torch.Tensor, *, window_size: int) -> torch.Tensor:
        if window_size == 1:
            return x.unsqueeze(1)
        if self._fixed_lags is None:
            lags = x.unsqueeze(1).expand(x.size(0), window_size - 1, x.size(1))
        else:
            lags = self._fixed_lags[:, : window_size - 1, :]
            if lags.size(1) < window_size - 1:
                pad = lags[:, -1:, :].expand(-1, window_size - 1 - lags.size(1), -1)
                lags = torch.cat([lags, pad], dim=1)
        return torch.cat([x.unsqueeze(1), lags], dim=1)

    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.drift(t, self._window(x, window_size=self.drift.window_size))

    def g(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        values = self.diffusion(t, self._window(x, window_size=self.diffusion.window_size))
        if self.noise_type == "diagonal":
            return values.view(x.size(0), self._hidden_size)
        return values.view(x.size(0), self._hidden_size, self._noise_size)


class MaximConditionalGenerator(nn.Module):
    """Conditional generator with the same hidden-SDE/readout structure as the reference."""

    def __init__(
        self,
        *,
        initial: nn.Module,
        func: MaximGeneratorFunc,
        readout: nn.Module,
        method: str = "reversible_heun",
        dt: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.initial = initial
        self._func = func
        self._readout = readout
        self.method = method
        self.dt = dt

    @property
    def hidden_size(self) -> int:
        return self._func._hidden_size

    @property
    def noise_size(self) -> int:
        return self._func._noise_size

    @property
    def drift_head_type(self) -> str:
        return self._func.drift.head_type

    @property
    def diffusion_head_type(self) -> str:
        return self._func.diffusion.head_type

    def _solver_dt(self, ts: torch.Tensor) -> float:
        if self.dt is not None:
            return float(self.dt)
        return float((ts[1:] - ts[:-1]).min().detach().cpu())

    def _sdeint_adjoint(self, x0: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        kwargs = {
            "method": self.method,
            "dt": self._solver_dt(ts),
        }
        if self.method == "reversible_heun":
            kwargs["adjoint_method"] = "adjoint_reversible_heun"
        return torchsde.sdeint_adjoint(self._func, x0, ts, **kwargs)

    def _make_fixed_lags(self, paths: list[torch.Tensor]) -> torch.Tensor:
        values = []
        for lag_idx in range(1, self._func.max_window_size):
            source_idx = len(paths) - lag_idx
            values.append(paths[0] if source_idx < 0 else paths[source_idx])
        return torch.stack(values, dim=1)

    def _integrate(self, x0: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        if not self._func.needs_discrete_window:
            self._func.set_fixed_lags(None)
            xs = self._sdeint_adjoint(x0, ts)
            return xs.transpose(0, 1)

        paths = [x0]
        for idx in range(1, ts.numel()):
            fixed_lags = self._make_fixed_lags(paths).to(device=x0.device, dtype=x0.dtype)
            self._func.set_fixed_lags(fixed_lags)
            interval_xs = self._sdeint_adjoint(paths[-1], ts[idx - 1 : idx + 1])
            paths.append(interval_xs[-1])
        self._func.set_fixed_lags(None)
        return torch.stack(paths, dim=1)

    def forward(self, batch: dict) -> dict:
        x0 = self.initial(batch["valHistory"])
        ts = batch["tsTarget"][0, :] - batch["tsTarget"][0, 0]
        xs = self._integrate(x0, ts)
        ys = batch["valHistory"][..., -1:, :] * torch.ones_like(xs)[..., 0:1] + self._readout(xs)
        batch["valSampled"] = ys
        return batch

    def sample_paths(self, ts: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
        if y0.ndim == 1:
            y0 = y0.unsqueeze(0)
        y0 = y0.to(device=ts.device, dtype=ts.dtype)
        batch_size = y0.size(0)
        ts_batch = ts.view(1, -1).expand(batch_size, -1)
        batch = {
            "batch_size": batch_size,
            "valHistory": y0.unsqueeze(1),
            "tsTarget": ts_batch,
        }
        self.eval()
        with torch.no_grad():
            return self.forward(batch)["valSampled"]


class MaximSDEGAN(nn.Module):
    def __init__(
        self,
        *,
        generator: MaximConditionalGenerator,
        discriminator: nn.Module,
        data_size: int,
    ) -> None:
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.data_size = int(data_size)

    def sample_paths(self, ts: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
        return self.generator.sample_paths(ts, y0)


class MaximConditionalDiscriminator(maxim_ref.ConditionalDiscriminator):
    """Reference conditional discriminator with configurable CDE integration flags."""

    def __init__(
        self,
        *args,
        dt: float = 1.0,
        adjoint: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.dt = float(dt)
        self.adjoint = bool(adjoint)

    def performInt(
        self,
        Y,
        func,
        h0,
        interval,
        adjoint_params,
        method="reversible_heun",
        backend="torchsde",
        dt=1.0,
        adjoint_method="adjoint_reversible_heun",
    ):
        import torchcde

        kwargs = {
            "method": method,
            "backend": backend,
            "dt": self.dt,
            "adjoint": self.adjoint,
        }
        if self.adjoint:
            kwargs["adjoint_method"] = adjoint_method
            kwargs["adjoint_params"] = adjoint_params
        return torchcde.cdeint(Y, func, h0, interval, **kwargs)


@dataclass(frozen=True)
class MaximModelConfig:
    data_size: int
    hidden_size: int = 16
    noise_size: int = 3
    noise_type: str = "general"
    mlp_size: int = 16
    num_layers: int = 1
    drift_head: str = "simple"
    diffusion_head: str = "simple"
    drift_window_size: int = 1
    diffusion_window_size: int = 1
    drift_init: float = 0.0
    diffusion_init: float = 0.1
    coefficient_tanh: bool = False
    readout_bias: bool = False
    fusion_num_layers: int = 2
    fusion_last_bias: bool = False
    method: str = "reversible_heun"
    dt: Optional[float] = None
    discriminator_hidden_size: int = 16
    discriminator_mlp_size: int = 16
    discriminator_num_layers: int = 1
    discriminator_func_tanh: bool = False
    discriminator_initial_tanh: bool = False
    discriminator_dt: float = 1.0
    discriminator_adjoint: bool = True


def build_maxim_sdegan(config: MaximModelConfig) -> MaximSDEGAN:
    noise_type = config.noise_type.lower()
    if noise_type not in {"general", "diagonal"}:
        raise ValueError("noise_type must be either general or diagonal")
    noise_size = int(config.noise_size)
    diffusion_out_size = config.hidden_size * noise_size
    if noise_type == "diagonal":
        noise_size = config.hidden_size
        diffusion_out_size = config.hidden_size

    drift = ReferenceHead(
        head_type=config.drift_head,
        hidden_size=config.hidden_size,
        out_size=config.hidden_size,
        mlp_size=config.mlp_size,
        num_layers=config.num_layers,
        window_size=config.drift_window_size,
        init_value=config.drift_init,
        tanh=config.coefficient_tanh,
    )
    diffusion = ReferenceHead(
        head_type=config.diffusion_head,
        hidden_size=config.hidden_size,
        out_size=diffusion_out_size,
        mlp_size=config.mlp_size,
        num_layers=config.num_layers,
        window_size=config.diffusion_window_size,
        init_value=config.diffusion_init,
        tanh=config.coefficient_tanh,
    )
    func = MaximGeneratorFunc(
        noise_size=noise_size,
        hidden_size=config.hidden_size,
        noise_type=noise_type,
        drift=drift,
        diffusion=diffusion,
    )
    initial = maxim_ref.initHistoryEncoder(
        muEncoder=maxim_ref.MLP(
            in_size=config.data_size,
            out_size=config.hidden_size,
            mlp_size=config.mlp_size,
            num_layers=config.num_layers,
            tanh=False,
        ),
        noiseEncoder=maxim_ref.MLP(
            in_size=config.hidden_size,
            out_size=config.hidden_size,
            mlp_size=config.mlp_size,
            num_layers=config.num_layers,
            tanh=False,
        ),
        fusion=maxim_ref.MLP(
            in_size=2 * config.hidden_size,
            out_size=config.hidden_size,
            mlp_size=config.mlp_size,
            num_layers=config.fusion_num_layers,
            tanh=False,
            lastBias=config.fusion_last_bias,
        ),
    )
    generator = MaximConditionalGenerator(
        initial=initial,
        func=func,
        readout=nn.Linear(config.hidden_size, config.data_size, bias=config.readout_bias),
        method=config.method,
        dt=config.dt,
    )

    disc_hidden = config.discriminator_hidden_size
    discriminator = MaximConditionalDiscriminator(
        discFunc=maxim_ref.DiscriminatorFunc(
            discModule=maxim_ref.MLP(
                in_size=1 + disc_hidden,
                out_size=disc_hidden * (1 + config.data_size),
                mlp_size=config.discriminator_mlp_size,
                num_layers=config.discriminator_num_layers,
                tanh=config.discriminator_func_tanh,
            )
        ),
        discInitial=maxim_ref.initHistoryEncoderDisc(
            muEncoder=maxim_ref.MLP(
                in_size=1 + 2 * config.data_size,
                out_size=disc_hidden,
                mlp_size=config.discriminator_mlp_size,
                num_layers=config.discriminator_num_layers,
                tanh=config.discriminator_initial_tanh,
            )
        ),
        discReadout=nn.Linear(2 * disc_hidden, 1),
        dt=config.discriminator_dt,
        adjoint=config.discriminator_adjoint,
    )
    return MaximSDEGAN(generator=generator, discriminator=discriminator, data_size=config.data_size)
