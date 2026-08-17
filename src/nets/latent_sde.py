from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.distributions import Normal, kl_divergence

try:
    import torchsde
except ImportError as exc:  # pragma: no cover - exercised only in incomplete envs.
    raise ImportError("Latent SDE requires torchsde.") from exc


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


class Encoder(nn.Module):
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
    latent_size: int = 4
    context_size: int = 64
    hidden_size: int = 128


class ConditionalLatentSDE(nn.Module):
    """Latent SDE from torchsde examples with a Y0-conditional latent prior."""

    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(
        self,
        data_size: int,
        latent_size: int = 4,
        context_size: int = 64,
        hidden_size: int = 128,
    ) -> None:
        super().__init__()
        self.data_size = int(data_size)
        self.latent_size = int(latent_size)
        self.context_size = int(context_size)
        self.hidden_size = int(hidden_size)

        self.encoder = Encoder(
            input_size=self.data_size,
            hidden_size=self.hidden_size,
            output_size=self.context_size,
        )
        self.qz0_net = nn.Linear(self.context_size, self.latent_size + self.latent_size)

        self.f_net = nn.Sequential(
            nn.Linear(self.latent_size + self.context_size, self.hidden_size),
            nn.Softplus(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Softplus(),
            nn.Linear(self.hidden_size, self.latent_size),
        )
        self.h_net = nn.Sequential(
            nn.Linear(self.latent_size, self.hidden_size),
            nn.Softplus(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Softplus(),
            nn.Linear(self.hidden_size, self.latent_size),
        )
        self.g_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, self.hidden_size),
                    nn.Softplus(),
                    nn.Linear(self.hidden_size, 1),
                    nn.Sigmoid(),
                )
                for _ in range(self.latent_size)
            ]
        )
        self.projector = nn.Linear(self.latent_size, self.data_size)

        self.pz0_net = nn.Linear(self.data_size, self.latent_size + self.latent_size)
        self._ctx: Optional[tuple[torch.Tensor, torch.Tensor]] = None

    @classmethod
    def from_config(cls, config: LatentSDEConfig) -> "ConditionalLatentSDE":
        return cls(**config.__dict__)

    def contextualize(self, ctx: tuple[torch.Tensor, torch.Tensor]) -> None:
        self._ctx = ctx

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self._ctx is None:
            raise RuntimeError("Posterior drift f requires contextualize((ts, ctx)) first.")
        ts, ctx = self._ctx
        index = torch.searchsorted(ts, t.detach(), right=True).clamp(max=ts.numel() - 1)
        return self.f_net(torch.cat((y, ctx[int(index.item())]), dim=1))

    def h(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        del t
        return self.h_net(y)

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        del t
        y_split = torch.split(y, split_size_or_sections=1, dim=1)
        return torch.cat([g_net_i(y_i) for g_net_i, y_i in zip(self.g_nets, y_split)], dim=1)

    def _conditional_pz0(self, y0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pz0_net(y0).chunk(chunks=2, dim=1)

    def _posterior_context(
        self,
        xs: torch.Tensor,
        ts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx = self.encoder(torch.flip(xs, dims=(0,)))
        ctx = torch.flip(ctx, dims=(0,))
        self.contextualize((ts, ctx))
        qz0_mean, qz0_logstd = self.qz0_net(ctx[0]).chunk(chunks=2, dim=1)
        return ctx, qz0_mean, qz0_logstd, xs[0]

    def forward(
        self,
        paths: torch.Tensor,
        ts: torch.Tensor,
        noise_std: float,
        adjoint: bool = False,
        method: str = "euler",
        dt: float = 1e-2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        paths = _validate_paths(paths)
        ts = _validate_ts(ts, expected_steps=paths.size(1)).to(
            device=paths.device,
            dtype=paths.dtype,
        )
        xs = paths.transpose(0, 1)
        ctx, qz0_mean, qz0_logstd, y0 = self._posterior_context(xs, ts)
        z0 = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)

        if adjoint:
            adjoint_params = (
                (ctx,)
                + tuple(self.f_net.parameters())
                + tuple(self.g_nets.parameters())
                + tuple(self.h_net.parameters())
            )
            zs, log_ratio = torchsde.sdeint_adjoint(
                self,
                z0,
                ts,
                adjoint_params=adjoint_params,
                dt=dt,
                logqp=True,
                method=method,
            )
        else:
            zs, log_ratio = torchsde.sdeint(
                self,
                z0,
                ts,
                dt=dt,
                logqp=True,
                method=method,
            )

        reconstructed = self.projector(zs)
        xs_dist = Normal(loc=reconstructed, scale=float(noise_std))
        log_pxs = xs_dist.log_prob(xs).sum(dim=(0, 2)).mean(dim=0)

        pz0_mean, pz0_logstd = self._conditional_pz0(y0)
        qz0 = Normal(loc=qz0_mean, scale=qz0_logstd.exp())
        pz0 = Normal(loc=pz0_mean, scale=pz0_logstd.exp())
        logqp0 = kl_divergence(qz0, pz0).sum(dim=1).mean(dim=0)
        logqp_path = log_ratio.sum(dim=0).mean(dim=0)
        return log_pxs, logqp0 + logqp_path

    @torch.no_grad()
    def sample_paths(
        self,
        ts: torch.Tensor,
        y0: torch.Tensor,
        *,
        bm=None,
        dt: float = 1e-3,
        method: Optional[str] = None,
    ) -> torch.Tensor:
        ts = _validate_ts(ts)
        if y0.ndim == 1:
            y0 = y0.unsqueeze(0)
        if y0.ndim != 2:
            raise ValueError(f"y0 must have shape (B, D) or (D,), got {tuple(y0.shape)}")
        if y0.size(-1) != self.data_size:
            raise ValueError(f"y0 has D={y0.size(-1)}, expected D={self.data_size}")

        y0 = y0.to(device=ts.device, dtype=ts.dtype)
        pz0_mean, pz0_logstd = self._conditional_pz0(y0)
        eps = torch.randn_like(pz0_mean)
        z0 = pz0_mean + pz0_logstd.exp() * eps
        kwargs = {
            "names": {"drift": "h"},
            "dt": dt,
        }
        if method is not None:
            kwargs["method"] = method
        if bm is not None:
            kwargs["bm"] = bm
        zs = torchsde.sdeint(self, z0, ts, **kwargs)
        return self.projector(zs).transpose(0, 1)
