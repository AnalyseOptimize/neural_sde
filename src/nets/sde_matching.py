from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import Tensor, nn
from torch import distributions as D


def _validate_paths(paths: Tensor) -> Tensor:
    if paths.ndim == 2:
        paths = paths.unsqueeze(-1)
    if paths.ndim != 3:
        raise ValueError(f"paths must have shape (B, T, D) or (B, T), got {tuple(paths.shape)}")
    if paths.size(0) < 1 or paths.size(1) < 2 or paths.size(2) < 1:
        raise ValueError("paths must have non-empty B, at least two time steps, and non-empty D")
    return paths


def _validate_ts(ts: Tensor, *, expected_steps: int | None = None) -> Tensor:
    if ts.ndim != 1:
        raise ValueError(f"ts must have shape (T,), got {tuple(ts.shape)}")
    if ts.numel() < 2:
        raise ValueError("ts must contain at least two time points")
    if expected_steps is not None and ts.numel() != expected_steps:
        raise ValueError(f"ts has T={ts.numel()}, expected T={expected_steps}")
    if not bool(torch.all(ts[1:] > ts[:-1])):
        raise ValueError("ts must be strictly increasing")
    return ts


def _normalize_ts(ts: Tensor) -> Tensor:
    denominator = (ts[-1] - ts[0]).clamp_min(torch.finfo(ts.dtype).eps)
    return (ts - ts[0]) / denominator


def _expand_ts(ts: Tensor, *, batch_size: int, n_steps: int, device, dtype) -> Tensor:
    ts = _validate_ts(ts, expected_steps=n_steps).to(device=device, dtype=dtype)
    ts = _normalize_ts(ts)
    return ts.view(1, n_steps, 1).expand(batch_size, -1, -1)


def solve_sde(
    sde: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]],
    z: Tensor,
    ts: float,
    tf: float,
    n_steps: int,
) -> Tensor:
    tt = torch.linspace(ts, tf, n_steps + 1, device=z.device, dtype=z.dtype)[:-1]
    dt = (tf - ts) / n_steps
    dt_2 = abs(dt) ** 0.5

    path = [z]
    for t in tt:
        f, g = sde(z, t)
        w = torch.randn_like(z)
        z = z + f * dt + g * w * dt_2
        path.append(z)

    return torch.stack(path)


def jvp(f: Callable[[Tensor], Any], x: Tensor, v: Tensor) -> tuple[Any, ...]:
    return torch.autograd.functional.jvp(
        f,
        x,
        v,
        create_graph=torch.is_grad_enabled(),
    )


def t_dir(f: Callable[[Tensor], Any], t: Tensor) -> tuple[Any, ...]:
    return jvp(f, t, torch.ones_like(t))


def grad(f: Callable[[Tensor], Tensor], x: Tensor) -> tuple[Tensor, Tensor]:
    create_graph = torch.is_grad_enabled()
    with torch.enable_grad():
        x = x.clone()
        if not x.requires_grad:
            x.requires_grad = True
        y = f(x)
        (gradient,) = torch.autograd.grad(y.sum(), x, create_graph=create_graph)
    return y, gradient


class SDE(nn.Module, ABC):
    @abstractmethod
    def drift(self, z: Tensor, t: Tensor, *args: Any) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def vol(self, z: Tensor, t: Tensor, *args: Any) -> Tensor:
        raise NotImplementedError

    def forward(self, z: Tensor, t: Tensor, *args: Any) -> tuple[Tensor, Tensor]:
        drift = self.drift(z, t, *args)
        vol = self.vol(z, t, *args)
        return drift, vol


class ConditionalPriorInitDistribution(nn.Module):
    def __init__(self, data_size: int, latent_size: int) -> None:
        super().__init__()
        self.net = nn.Linear(data_size, 2 * latent_size)

    def forward(self, y0: Tensor) -> D.Distribution:
        m, log_s = self.net(y0).chunk(chunks=2, dim=1)
        s = torch.exp(log_s)
        return D.Independent(D.Normal(m, s), 1)


class PriorSDE(SDE):
    def __init__(self, latent_size: int, hidden_size: int) -> None:
        super().__init__()
        self.drift_net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )
        self.vol_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, hidden_size),
                    nn.Softplus(),
                    nn.Linear(hidden_size, 1),
                    nn.Sigmoid(),
                )
                for _ in range(latent_size)
            ]
        )

    def drift(self, z: Tensor, t: Tensor, *args: Any) -> Tensor:
        del t, args
        return self.drift_net(z)

    def vol(self, z: Tensor, t: Tensor, *args: Any) -> Tensor:
        del t, args
        z = torch.split(z, 1, dim=1)
        g = [net_i(z_i) for net_i, z_i in zip(self.vol_nets, z)]
        return torch.cat(g, dim=1)


class PriorObservation(nn.Module):
    def __init__(self, latent_size: int, data_size: int, noise_std: float) -> None:
        super().__init__()
        self.net = nn.Linear(latent_size, data_size)
        self.noise_std = float(noise_std)

    def get_coeffs(self, z: Tensor) -> tuple[Tensor, Tensor]:
        m = self.net(z)
        s = torch.ones_like(m) * self.noise_std
        return m, s

    def forward(self, z: Tensor) -> D.Distribution:
        m, s = self.get_coeffs(z)
        return D.Independent(D.Normal(m, s), 1)


class PosteriorEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        out, h = self.gru(x)
        return torch.cat([h[0, :, None], out], dim=1)


class PosteriorAffine(nn.Module):
    def __init__(self, latent_size: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * latent_size),
        )
        self.sm = nn.Softmax(dim=-1)

    def get_coeffs(self, ctx: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        l = ctx.shape[1] - 1
        h, out = ctx[:, 0], ctx[:, 1:]
        ts = torch.linspace(0, 1, l, device=ctx.device, dtype=ctx.dtype)[None, :]
        c = self.sm(-(l * (ts - t)) ** 2)
        out = (out * c[:, :, None]).sum(dim=1)
        ctx_t = torch.cat([h + out, t], dim=1)

        m, log_s = self.net(ctx_t).chunk(chunks=2, dim=1)
        s = torch.exp(log_s)
        return m, s

    def forward(
        self,
        ctx: Tensor,
        t: Tensor,
        return_t_dir: bool = False,
    ):
        if return_t_dir:
            def f(t_in: Tensor) -> tuple[Tensor, Tensor]:
                return self.get_coeffs(ctx, t_in)

            return t_dir(f, t)
        return self.get_coeffs(ctx, t)


@dataclass(frozen=True)
class SDEMatchingConfig:
    data_size: int
    latent_size: int = 4
    hidden_size: int = 100
    observation_noise_std: float = 0.01


class SDEMatching(nn.Module):
    def __init__(
        self,
        data_size: int,
        latent_size: int = 4,
        hidden_size: int = 100,
        observation_noise_std: float = 0.01,
    ) -> None:
        super().__init__()
        self.data_size = int(data_size)
        self.latent_size = int(latent_size)
        self.hidden_size = int(hidden_size)
        self.observation_noise_std = float(observation_noise_std)

        self.p_init_distr = ConditionalPriorInitDistribution(self.data_size, self.latent_size)
        self.p_sde = PriorSDE(self.latent_size, self.hidden_size)
        self.p_observe = PriorObservation(
            self.latent_size,
            self.data_size,
            self.observation_noise_std,
        )
        self.q_enc = PosteriorEncoder(self.data_size, self.hidden_size)
        self.q_affine = PosteriorAffine(self.latent_size, self.hidden_size)

    @classmethod
    def from_config(cls, config: SDEMatchingConfig) -> "SDEMatching":
        return cls(**config.__dict__)

    def loss_prior(self, ctx: Tensor, y0: Tensor) -> Tensor:
        bs = ctx.shape[0]
        t0 = torch.zeros(bs, 1, device=ctx.device, dtype=ctx.dtype)
        m0, s0 = self.q_affine(ctx, t0)
        q_z0 = D.Independent(D.Normal(m0, s0), 1)
        p_z0 = self.p_init_distr(y0)
        return D.kl_divergence(q_z0, p_z0)

    def loss_diff(self, ctx: Tensor, t: Tensor) -> Tensor:
        (m, s), (dm, ds) = self.q_affine(ctx, t, return_t_dir=True)
        eps = torch.randn_like(m)
        z = m + s * eps

        def g2_in(z_in: Tensor) -> Tensor:
            return self.p_sde.vol(z_in, t) ** 2

        g2, d_g2 = grad(g2_in, z)
        q_dz = dm + ds * eps
        q_score = -eps / s
        q_drift = q_dz + 0.5 * g2 * q_score + 0.5 * d_g2
        p_drift = self.p_sde.drift(z, t)
        loss_diff = 0.5 * (q_drift - p_drift) ** 2 / g2
        return loss_diff.sum(dim=1)

    def loss_recon(self, ctx: Tensor, x: Tensor, t: Tensor) -> Tensor:
        m, s = self.q_affine(ctx, t)
        eps = torch.randn_like(m)
        z = m + s * eps
        p_x = self.p_observe(z)
        return -p_x.log_prob(x)

    def loss_terms(self, paths: Tensor, ts: Tensor) -> dict[str, Tensor]:
        paths = _validate_paths(paths)
        bs = paths.shape[0]
        n = paths.shape[1]
        ts_batch = _expand_ts(
            ts,
            batch_size=bs,
            n_steps=n,
            device=paths.device,
            dtype=paths.dtype,
        )

        ctx = self.q_enc(paths)
        loss_prior = self.loss_prior(ctx, paths[:, 0, :])

        t = torch.rand(bs, 1, device=paths.device, dtype=paths.dtype) * (
            ts_batch[:, -1] - ts_batch[:, 0]
        ) + ts_batch[:, 0]
        loss_diff = self.loss_diff(ctx, t)

        rng = torch.arange(bs, device=paths.device)
        u = torch.randint(n, [bs], device=paths.device)
        t_u = ts_batch[rng, u]
        x_u = paths[rng, u]
        loss_recon = self.loss_recon(ctx, x_u, t_u)

        loss = loss_prior + loss_diff + loss_recon
        return {
            "loss": loss,
            "loss_prior": loss_prior,
            "loss_diff": loss_diff,
            "loss_recon": loss_recon,
        }

    def forward(self, paths: Tensor, ts: Tensor) -> Tensor:
        return self.loss_terms(paths, ts)["loss"]

    @torch.no_grad()
    def sample_paths(
        self,
        ts: Tensor,
        y0: Tensor,
        *,
        n_inner_steps: int = 1,
    ) -> Tensor:
        ts = _validate_ts(ts)
        if y0.ndim == 1:
            y0 = y0.unsqueeze(0)
        if y0.ndim != 2:
            raise ValueError(f"y0 must have shape (B, D) or (D,), got {tuple(y0.shape)}")
        if y0.size(-1) != self.data_size:
            raise ValueError(f"y0 has D={y0.size(-1)}, expected D={self.data_size}")
        if n_inner_steps < 1:
            raise ValueError("n_inner_steps must be >= 1")

        y0 = y0.to(device=ts.device, dtype=ts.dtype)
        z0 = self.p_init_distr(y0).rsample()
        total_steps = (ts.numel() - 1) * int(n_inner_steps)
        zs = solve_sde(self.p_sde, z0, 0.0, 1.0, n_steps=total_steps)
        zs = zs[:: int(n_inner_steps)].transpose(0, 1)

        flat_zs = zs.reshape(-1, self.latent_size)
        flat_xs, _ = self.p_observe.get_coeffs(flat_zs)
        return flat_xs.reshape(y0.size(0), ts.numel(), self.data_size)
