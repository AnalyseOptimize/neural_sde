from __future__ import annotations

"""Conditional latent SDE with a small affine prior and shared diffusion.

Reference: Li et al. (2020), Scalable Gradients for Stochastic Differential
Equations; google-research/torchsde, examples/latent_sde{,_lorenz}.py.
The public path layout is (batch, time, channel), as in the other models here.
"""

from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import nn
from torch.distributions import Normal, kl_divergence
import torchsde

from src.nets.sde_ml import _validate_paths, _validate_ts


class Encoder(nn.Module):
    """Backward GRU context; time is an explicit input for irregular grids."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(inp)
        return self.lin(out)


@dataclass(frozen=True)
class LatentSDEConfig:
    data_size: int
    latent_size: int = 2
    context_size: int = 16
    hidden_size: int = 16
    encoder_type: str = "stochastic"
    decoder_type: str = "deterministic"
    diffusion_init: float = 0.3
    diffusion_min: float = 1e-3
    decoder_std_init: float = 0.05
    decoder_std_min: float = 1e-3


class ConditionalLatentSDE(nn.Module):
    """Learn p(x[1:] | x[0]) using a posterior SDE and a conditional prior.

    h: affine prior drift; f: h plus a context MLP; g: positive, constant,
    diagonal diffusion shared by both SDEs. torchsde integrates path KL.

    Stochastic encoder: Gaussian q(z0 | x[0:T]), p(z0 | x0), analytic KL0.
    Deterministic encoder: both start at the SAME learned embedding of x0,
    so KL0 is exactly zero (no invalid Dirac-versus-Gaussian KL).

    Deterministic decoder: linear readout, fixed Gaussian training likelihood.
    Stochastic decoder: linear mean and learned diagonal observation scale.
    Both modes still have Brownian noise in the latent dynamics.
    """

    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(
        self,
        data_size: int,
        latent_size: int = 2,
        context_size: int = 16,
        hidden_size: int = 16,
        *,
        encoder_type: str = "stochastic",
        decoder_type: str = "deterministic",
        diffusion_init: float = 0.3,
        diffusion_min: float = 1e-3,
        decoder_std_init: float = 0.05,
        decoder_std_min: float = 1e-3,
    ) -> None:
        super().__init__()
        self.config = LatentSDEConfig(
            data_size=data_size, latent_size=latent_size, context_size=context_size,
            hidden_size=hidden_size, encoder_type=encoder_type, decoder_type=decoder_type,
            diffusion_init=diffusion_init, diffusion_min=diffusion_min,
            decoder_std_init=decoder_std_init, decoder_std_min=decoder_std_min,
        )
        if min(data_size, latent_size, context_size, hidden_size) < 1:
            raise ValueError("Model dimensions must be positive")
        if encoder_type not in {"deterministic", "stochastic"}:
            raise ValueError("encoder_type must be deterministic or stochastic")
        if decoder_type not in {"deterministic", "stochastic"}:
            raise ValueError("decoder_type must be deterministic or stochastic")
        if not 0 < diffusion_min < diffusion_init:
            raise ValueError("Require 0 < diffusion_min < diffusion_init")
        if not 0 < decoder_std_min < decoder_std_init:
            raise ValueError("Require 0 < decoder_std_min < decoder_std_init")
        self.data_size, self.latent_size = int(data_size), int(latent_size)
        self.context_size, self.hidden_size = int(context_size), int(hidden_size)
        self.encoder_type, self.decoder_type = encoder_type, decoder_type
        self.diffusion_min, self.decoder_std_min = diffusion_min, decoder_std_min

        self.encoder = Encoder(self.data_size + 1, self.hidden_size, self.context_size)
        initial_outputs = self.latent_size * (2 if encoder_type == "stochastic" else 1)
        self.pz0_net = nn.Linear(self.data_size, initial_outputs)
        self.qz0_net = nn.Linear(self.context_size, initial_outputs) if encoder_type == "stochastic" else None
        self.h_net = nn.Linear(self.latent_size, self.latent_size)
        self.f_net = nn.Sequential(
            nn.Linear(self.latent_size + self.context_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.latent_size),
        )
        # Start posterior and prior drifts together, with a weak restoring prior.
        with torch.no_grad():
            self.h_net.weight.copy_(-0.1 * torch.eye(self.latent_size))
            self.h_net.bias.zero_()
            self.f_net[-1].weight.zero_()
            self.f_net[-1].bias.zero_()
        self.raw_diffusion = nn.Parameter(torch.full(
            (self.latent_size,), math.log(math.expm1(diffusion_init - diffusion_min))
        ))
        self.projector = nn.Linear(self.latent_size, self.data_size, bias=False)
        self.decoder_scale = None
        if decoder_type == "stochastic":
            self.decoder_scale = nn.Linear(self.latent_size, self.data_size)
            nn.init.zeros_(self.decoder_scale.weight)
            nn.init.constant_(self.decoder_scale.bias, math.log(math.expm1(decoder_std_init - decoder_std_min)))
        self._ctx: Optional[tuple[torch.Tensor, torch.Tensor]] = None

    @classmethod
    def from_config(cls, config: LatentSDEConfig) -> "ConditionalLatentSDE":
        return cls(**config.__dict__)

    def contextualize(self, ctx: tuple[torch.Tensor, torch.Tensor]) -> None:
        self._ctx = ctx

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self._ctx is None:
            raise RuntimeError("Posterior drift requires contextualize((ts, ctx)) first")
        ts, ctx = self._ctx
        index = torch.searchsorted(ts, t.detach()).clamp(max=ts.numel() - 1)
        return self.h(t, y) + self.f_net(torch.cat((y, ctx[index]), dim=-1))

    def h(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.h_net(y)

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (self.diffusion_min + nn.functional.softplus(self.raw_diffusion)).expand_as(y)

    @staticmethod
    def _normal(parameters: torch.Tensor) -> Normal:
        mean, logstd = parameters.chunk(2, dim=-1)
        return Normal(mean, logstd.clamp(-7.0, 3.0).exp())

    def _initial(self, y0: torch.Tensor, ctx0=None) -> tuple[torch.Tensor, torch.Tensor]:
        prior = self.pz0_net(y0)
        if self.encoder_type == "deterministic":
            return prior, y0.new_zeros(())
        pz0 = self._normal(prior)
        if ctx0 is None:
            return pz0.rsample(), y0.new_zeros(())
        qz0 = self._normal(self.qz0_net(ctx0))
        return qz0.rsample(), kl_divergence(qz0, pz0).sum(-1).mean()

    def _context(self, paths, ts):
        xs = paths.transpose(0, 1)
        time = ((ts - ts[0]) / (ts[-1] - ts[0]))[:, None, None].expand(-1, paths.size(0), 1)
        ctx = self.encoder(torch.cat((xs, time), dim=-1).flip(0)).flip(0)
        self.contextualize((ts, ctx))
        return ctx

    def decode(self, zs: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
        """Time-major conditional mean, anchored to the observed initial value."""
        return y0.unsqueeze(0) + self.projector(zs - zs[:1])

    def observation_distribution(self, zs, y0, noise_std=0.05):
        mean = self.decode(zs, y0)
        if self.decoder_scale is None:
            if noise_std <= 0:
                raise ValueError("noise_std must be positive")
            scale = float(noise_std)
        else:
            scale = self.decoder_std_min + nn.functional.softplus(self.decoder_scale(zs))
        return Normal(mean, scale)

    def _prepare(self, paths, ts):
        paths = _validate_paths(paths)
        parameter = next(self.parameters())
        paths = paths.to(parameter)
        if paths.size(-1) != self.data_size:
            raise ValueError(f"paths must have D={self.data_size}")
        ts = _validate_ts(ts, expected_steps=paths.size(1)).to(parameter)
        return paths, ts

    def forward(
        self, paths, ts, noise_std=0.05, adjoint=False, method="euler", dt=1e-2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        paths, ts = self._prepare(paths, ts)
        ctx = self._context(paths, ts)
        y0 = paths[:, 0]
        z0, kl0 = self._initial(y0, ctx[0])
        kwargs = dict(dt=dt, logqp=True, method=method)
        if adjoint:
            # Context is not a parameter: pass it explicitly to preserve GRU gradients.
            kwargs["adjoint_params"] = (ctx, self.raw_diffusion, *self.f_net.parameters(), *self.h_net.parameters())
        integrate = torchsde.sdeint_adjoint if adjoint else torchsde.sdeint
        zs, log_ratio = integrate(self, z0, ts, **kwargs)
        observations = self.observation_distribution(zs, y0, noise_std)
        # x0 is conditioned on, not an independent likelihood term.
        log_pxs = observations.log_prob(paths.transpose(0, 1))[1:].sum((0, 2)).mean()
        return log_pxs, kl0 + log_ratio.sum(0).mean()

    def _decode_sample(self, zs, y0, sample_observations):
        if sample_observations is None:
            sample_observations = self.decoder_type == "stochastic"
        if sample_observations and self.decoder_type == "stochastic":
            xs = self.observation_distribution(zs, y0).sample()
        else:
            xs = self.decode(zs, y0)
        xs[0] = y0  # The conditional initial observation is known exactly.
        return xs.transpose(0, 1)

    @torch.no_grad()
    def sample_paths(self, ts, y0, *, bm=None, dt=1e-2, method="euler", sample_observations=None):
        parameter = next(self.parameters())
        ts = _validate_ts(ts).to(parameter)
        if y0.ndim == 1:
            y0 = y0.unsqueeze(0)
        if y0.ndim != 2 or y0.size(-1) != self.data_size or y0.size(0) == 0:
            raise ValueError(f"y0 must have shape (B, {self.data_size})")
        y0 = y0.to(parameter)
        z0, _ = self._initial(y0)
        zs = torchsde.sdeint(self, z0, ts, names={"drift": "h"}, dt=dt, method=method or "euler", bm=bm)
        return self._decode_sample(zs, y0, sample_observations)

    @torch.no_grad()
    def reconstruct_paths(self, paths, ts, *, bm=None, dt=1e-2, method="euler", sample_observations=None):
        """Posterior reconstruction, distinct from prior generation in sample_paths."""
        paths, ts = self._prepare(paths, ts)
        ctx = self._context(paths, ts)
        y0 = paths[:, 0]
        z0, _ = self._initial(y0, ctx[0])
        zs = torchsde.sdeint(self, z0, ts, dt=dt, method=method or "euler", bm=bm)
        return self._decode_sample(zs, y0, sample_observations)
