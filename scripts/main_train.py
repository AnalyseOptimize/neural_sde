from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nets.sdegan import CDEDiscriminator, SDEGenerator
from src.simulators import (
    DeterministicDriftSimulator,
    MultiDimensionalGBMSimulator,
    OUSimulator,
)
from src.train import SDEGANTrainConfig, train_sdegan
from utils.data import PathDataConfig, make_sdegan_dataset
from utils.logging import log_config, setup_logger
from utils.visual import plot_epoch_diagnostics, plot_real_generated_paths


def _to_container(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)


def _as_float_or_list(value, *, dim: int, name: str) -> float | list[float]:
    if isinstance(value, str):
        values = [float(part.strip()) for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        values = [float(item) for item in value]
    else:
        values = [float(value)]

    if len(values) == 1:
        return values[0]
    if len(values) != dim:
        raise ValueError(f"{name} must be scalar or contain {dim} values")
    return values


def _build_equicorrelation(dim: int, rho: float) -> list[list[float]] | None:
    if dim == 1:
        return None
    if not -1.0 / (dim - 1) < rho < 1.0:
        raise ValueError(f"rho must be in (-1/(dim-1), 1), got {rho}")
    return [[1.0 if i == j else rho for j in range(dim)] for i in range(dim)]


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_simulator(cfg: DictConfig):
    name = str(cfg.simulator.name)
    if name == "gbm":
        gbm = cfg.simulator.gbm
        dim = int(gbm.dim)
        corr = None
        if gbm.get("corr") is not None:
            corr = _to_container(gbm.corr)
        elif dim > 1:
            corr = _build_equicorrelation(dim, float(gbm.rho))
        return MultiDimensionalGBMSimulator(
            s0=_as_float_or_list(gbm.s0, dim=dim, name="gbm.s0"),
            mu=_as_float_or_list(gbm.mu, dim=dim, name="gbm.mu"),
            sigma=_as_float_or_list(gbm.sigma, dim=dim, name="gbm.sigma"),
            dim=dim,
            corr=corr,
            seed=int(cfg.seed),
        )

    if name == "deterministic":
        det = cfg.simulator.deterministic
        return DeterministicDriftSimulator(s0=float(det.s0), mu=float(det.mu))

    if name != "ou":
        raise ValueError(f"Unsupported simulator: {name}")

    ou = cfg.simulator.ou
    return OUSimulator(
        s0=float(ou.s0),
        theta=float(ou.theta),
        mu=float(ou.mu),
        sigma=float(ou.sigma),
        seed=int(cfg.seed),
    )


def maybe_save_plots(
    *,
    cfg: DictConfig,
    fig_dir: Path,
    history: dict[str, list[float]],
    generator: SDEGenerator,
    ts: torch.Tensor,
    real_paths: torch.Tensor,
    device: torch.device,
) -> None:
    if not bool(cfg.plots.enabled):
        return

    fig_dir.mkdir(parents=True, exist_ok=True)

    options = ["loss"]
    if history.get("marginal_w1_average") or history.get("marginal_w1_max"):
        options.append("w1")
    if history.get("expected_supremum_squared_error"):
        options.append("e_sup")

    for option in options:
        plot_epoch_diagnostics(
            history,
            option=option,
            save_path=fig_dir / f"{option}_by_epoch.pdf",
        )

    n_paths = min(int(cfg.plots.n_paths), real_paths.size(0))
    generator.eval()
    with torch.no_grad():
        y0 = real_paths[:n_paths, 0, :].to(device=device, dtype=ts.dtype)
        generated = generator.sample_paths(ts.to(device), y0).detach().cpu()
    plot_real_generated_paths(
        ts.detach().cpu(),
        real_paths[:n_paths].detach().cpu(),
        generated,
        n_paths=n_paths,
        align_initial=bool(cfg.plots.align_initial),
        save_path=fig_dir / "real_vs_generated_paths.pdf",
    )


@hydra.main(version_base=None, config_path="../configs", config_name="sdegan")
def main(cfg: DictConfig) -> dict[str, list[float]]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    set_seed(int(cfg.seed))

    run_logger, run_paths = setup_logger(cfg.paths.log_root, cfg.run_name)
    log_config(run_logger, _to_container(cfg))

    device = resolve_device(str(cfg.device))
    checkpoint_path = None
    if bool(cfg.train.save_checkpoint):
        checkpoint_path = str(run_paths.checkpoint_dir / "model.pt")

    data_config = PathDataConfig(
        dataset_size=int(cfg.data.dataset_size),
        t_size=int(cfg.data.t_size),
        dt=float(cfg.data.dt),
        normalize=bool(cfg.data.normalize),
        shuffle=bool(cfg.data.shuffle),
        drop_last=bool(cfg.data.drop_last),
    )
    simulator = build_simulator(cfg)
    data = make_sdegan_dataset(
        simulator=simulator,
        config=data_config,
        batch_size=int(cfg.train.batch_size),
        shuffle=data_config.shuffle,
        drop_last=data_config.drop_last,
        num_workers=int(cfg.data.num_workers),
    )
    run_logger.info(
        "Dataset prepared: simulator={}, data_size={}, t_size={}, batches={}",
        simulator.__class__.__name__,
        data.data_size,
        data.ts.numel(),
        len(data.dataloader),
    )

    generator = SDEGenerator(
        data_size=data.data_size,
        noise_size=int(cfg.model.generator.noise_size),
        noise_type=str(cfg.model.generator.noise_type),
        drift_head=str(cfg.model.generator.drift_head),
        diffusion_head=str(cfg.model.generator.diffusion_head),
        drift_window_size=int(cfg.model.generator.drift_window_size),
        diffusion_window_size=int(cfg.model.generator.diffusion_window_size),
        hidden_size=int(cfg.model.generator.hidden_size),
        num_layers=int(cfg.model.generator.num_layers),
        drift_init=float(cfg.model.generator.drift_init),
        diffusion_init=float(cfg.model.generator.diffusion_init),
        drift_scale=float(cfg.model.generator.drift_scale),
        diffusion_scale=float(cfg.model.generator.diffusion_scale),
        final_tanh=bool(cfg.model.generator.final_tanh),
        method=str(cfg.model.generator.method),
        dt=cfg.model.generator.dt,
    )
    discriminator = CDEDiscriminator(
        data_size=data.data_size,
        hidden_size=int(cfg.model.discriminator.hidden_size),
        mlp_size=int(cfg.model.discriminator.mlp_size),
        num_layers=int(cfg.model.discriminator.num_layers),
        method=str(cfg.model.discriminator.method),
        dt=float(cfg.model.discriminator.dt),
        adjoint=bool(cfg.model.discriminator.adjoint),
    )
    run_logger.info(
        "Models prepared: generator_params={}, discriminator_params={}",
        sum(param.numel() for param in generator.parameters()),
        sum(param.numel() for param in discriminator.parameters()),
    )

    train_config = SDEGANTrainConfig(
        epochs=int(cfg.train.epochs),
        steps=cfg.train.steps,
        steps_per_epoch=cfg.train.steps_per_epoch,
        batch_size=int(cfg.train.batch_size),
        generator_lr=float(cfg.train.generator_lr),
        discriminator_lr=float(cfg.train.discriminator_lr),
        weight_decay=float(cfg.train.weight_decay),
        optimizer=str(cfg.train.optimizer),
        n_critic=int(cfg.train.n_critic),
        clip_discriminator=bool(cfg.train.clip_discriminator),
        swa_start_step=cfg.train.swa_start_step,
        log_every=int(cfg.train.log_every),
        eval_every=int(cfg.train.eval_every),
        metrics_every_epoch=int(cfg.evaluation.every_epoch) if bool(cfg.evaluation.enabled) else 0,
        metrics_n_paths=int(cfg.evaluation.n_paths),
        metrics_num_quantiles=int(cfg.evaluation.num_quantiles),
        metrics_align_initial=bool(cfg.evaluation.align_initial),
        init_mult_initial=float(cfg.train.init_mult_initial),
        init_mult_func=float(cfg.train.init_mult_func),
        checkpoint_path=checkpoint_path,
    )
    history = train_sdegan(
        generator=generator,
        discriminator=discriminator,
        dataloader=data.dataloader,
        ts=data.ts,
        device=device,
        logger=run_logger,
        config=train_config,
        metric_real_paths=data.paths if bool(cfg.evaluation.enabled) else None,
    )

    maybe_save_plots(
        cfg=cfg,
        fig_dir=run_paths.fig_dir,
        history=history,
        generator=generator,
        ts=data.ts,
        real_paths=data.paths,
        device=device,
    )

    run_logger.info("Run finished. Run directory: {}", run_paths.run_dir)
    return history


if __name__ == "__main__":
    main()
