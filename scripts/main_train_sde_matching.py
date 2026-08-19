from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nets.sde_matching import SDEMatching
from src.simulators import (
    ArithmeticBrownianMotionSimulator,
    DeterministicDriftSimulator,
    MultiDimensionalGBMSimulator,
    OUSimulator,
    PerturbedPathSimulator,
)
from src.train import SDEMatchingTrainConfig, train_sde_matching
from utils.data import PathDataConfig, normalize_paths_by_initial, validate_paths
from utils.logging import log_config, setup_logger
from utils.visual import plot_epoch_diagnostics, plot_real_generated_paths


def _to_container(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)


def _as_float_or_list(value, *, dim: int, name: str) -> float | list[float]:
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)

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


def maybe_add_perturbations(simulator, cfg: DictConfig):
    perturbations = cfg.simulator.get("perturbations")
    if perturbations is None:
        return simulator

    data_size = int(simulator.data_size)
    jumps = perturbations.get("jumps", {})
    jump_enabled = bool(jumps.get("enabled", False))

    jump_intensity = 0.0
    jump_size = 1.0
    jump_seed = None
    if jump_enabled:
        jump_intensity = float(jumps.get("intensity", 0.0))
        jump_size = _as_float_or_list(
            jumps.get("size", 1.0),
            dim=data_size,
            name="simulator.perturbations.jumps.size",
        )
        jump_seed = jumps.get("seed")
        if jump_seed is None:
            jump_seed = int(cfg.seed) + 200_003

    perturbed = PerturbedPathSimulator(
        base=simulator,
        jump_intensity=jump_intensity,
        jump_size=jump_size,
        jump_seed=None if jump_seed is None else int(jump_seed),
    )
    return perturbed if perturbed.has_perturbations else simulator


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
    if name == "abm":
        abm = cfg.simulator.abm
        dim = int(abm.dim)
        corr = None
        if abm.get("corr") is not None:
            corr = _to_container(abm.corr)
        elif dim > 1:
            corr = _build_equicorrelation(dim, float(abm.rho))
        simulator = ArithmeticBrownianMotionSimulator(
            s0=_as_float_or_list(abm.s0, dim=dim, name="abm.s0"),
            mu=_as_float_or_list(abm.mu, dim=dim, name="abm.mu"),
            sigma=_as_float_or_list(abm.sigma, dim=dim, name="abm.sigma"),
            dim=dim,
            corr=corr,
            seed=int(cfg.seed),
        )
        return maybe_add_perturbations(simulator, cfg)

    if name == "gbm":
        gbm = cfg.simulator.gbm
        dim = int(gbm.dim)
        corr = None
        if gbm.get("corr") is not None:
            corr = _to_container(gbm.corr)
        elif dim > 1:
            corr = _build_equicorrelation(dim, float(gbm.rho))
        simulator = MultiDimensionalGBMSimulator(
            s0=_as_float_or_list(gbm.s0, dim=dim, name="gbm.s0"),
            mu=_as_float_or_list(gbm.mu, dim=dim, name="gbm.mu"),
            sigma=_as_float_or_list(gbm.sigma, dim=dim, name="gbm.sigma"),
            dim=dim,
            corr=corr,
            seed=int(cfg.seed),
        )
        return maybe_add_perturbations(simulator, cfg)

    if name == "deterministic":
        det = cfg.simulator.deterministic
        simulator = DeterministicDriftSimulator(s0=float(det.s0), mu=float(det.mu))
        return maybe_add_perturbations(simulator, cfg)

    if name != "ou":
        raise ValueError(f"Unsupported simulator: {name}")

    ou = cfg.simulator.ou
    simulator = OUSimulator(
        s0=float(ou.s0),
        theta=float(ou.theta),
        mu=float(ou.mu),
        sigma=float(ou.sigma),
        seed=int(cfg.seed),
    )
    return maybe_add_perturbations(simulator, cfg)


def make_path_dataloader(
    *,
    simulator,
    data_config: PathDataConfig,
    batch_size: int,
    num_workers: int,
) -> tuple[torch.Tensor, torch.Tensor, DataLoader]:
    ts, paths = simulator.simulate(
        n_paths=data_config.dataset_size,
        n_steps=data_config.t_size,
        dt=data_config.dt,
        device="cpu",
        dtype=torch.float32,
    )
    paths = validate_paths(paths)
    if data_config.normalize:
        paths = normalize_paths_by_initial(paths)

    dataloader = DataLoader(
        TensorDataset(paths),
        batch_size=batch_size,
        shuffle=data_config.shuffle,
        drop_last=data_config.drop_last,
        num_workers=num_workers,
    )
    return ts, paths, dataloader


def plot_matching_loss_components(history: dict, *, save_path: str | Path | None = None):
    epochs = np.asarray(history.get("epoch", []), dtype=float)
    if epochs.size == 0:
        epochs = np.arange(1, len(history.get("loss_prior_epoch", [])) + 1, dtype=float)

    with plt.rc_context(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "axes.linewidth": 0.8,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 1.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        for key, label, color in (
            ("loss_prior_epoch", "Initial KL", "#1f77b4"),
            ("loss_diff_epoch", "Diffusion matching", "#d62728"),
            ("loss_recon_epoch", "Reconstruction", "#2ca02c"),
        ):
            values = np.asarray(history.get(key, []), dtype=float)
            if values.size:
                xs = epochs if epochs.size == values.size else np.arange(1, values.size + 1)
                ax.plot(xs, values, label=label, color=color)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss component")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, color="0.88", linewidth=0.6)
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig


def maybe_save_plots(
    *,
    cfg: DictConfig,
    fig_dir: Path,
    history: dict[str, list[float]],
    model: SDEMatching,
    ts: torch.Tensor,
    real_paths: torch.Tensor,
    device: torch.device,
) -> None:
    if not bool(cfg.plots.enabled):
        return

    fig_dir.mkdir(parents=True, exist_ok=True)
    fig = plot_epoch_diagnostics(
        history,
        option="loss",
        save_path=fig_dir / "sde_matching_loss_by_epoch.pdf",
    )
    plt.close(fig)
    fig = plot_matching_loss_components(
        history,
        save_path=fig_dir / "sde_matching_loss_components_by_epoch.pdf",
    )
    plt.close(fig)
    if history.get("marginal_w1_average") or history.get("marginal_w1_max"):
        fig = plot_epoch_diagnostics(history, option="w1", save_path=fig_dir / "w1_by_epoch.pdf")
        plt.close(fig)
    if history.get("expected_supremum_squared_error"):
        fig = plot_epoch_diagnostics(history, option="e_sup", save_path=fig_dir / "e_sup_by_epoch.pdf")
        plt.close(fig)

    n_paths = min(int(cfg.plots.n_paths), real_paths.size(0))
    model.eval()
    with torch.no_grad():
        y0 = real_paths[:n_paths, 0, :].to(device=device, dtype=ts.dtype)
        generated = model.sample_paths(
            ts.to(device=device),
            y0,
            n_inner_steps=int(cfg.train.sample_inner_steps),
        ).detach().cpu()
    fig = plot_real_generated_paths(
        ts.detach().cpu(),
        real_paths[:n_paths].detach().cpu(),
        generated,
        n_paths=n_paths,
        align_initial=bool(cfg.plots.align_initial),
        save_path=fig_dir / "real_vs_generated_paths.pdf",
    )
    plt.close(fig)


@hydra.main(version_base=None, config_path="../configs", config_name="sde_matching")
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
    ts, paths, dataloader = make_path_dataloader(
        simulator=simulator,
        data_config=data_config,
        batch_size=int(cfg.train.batch_size),
        num_workers=int(cfg.data.num_workers),
    )
    run_logger.info(
        "Dataset prepared: simulator={}, data_size={}, t_size={}, batches={}",
        simulator.__class__.__name__,
        paths.size(-1),
        ts.numel(),
        len(dataloader),
    )

    model = SDEMatching(
        data_size=paths.size(-1),
        latent_size=int(cfg.model.latent_size),
        hidden_size=int(cfg.model.hidden_size),
        observation_noise_std=float(cfg.model.observation_noise_std),
    )
    run_logger.info(
        "SDEMatching prepared: params={}, data_size={}, latent_size={}, hidden_size={}, observation_noise_std={}",
        sum(param.numel() for param in model.parameters()),
        model.data_size,
        model.latent_size,
        model.hidden_size,
        model.observation_noise_std,
    )
    if bool(cfg.evaluation.get("coupled_brownian", False)):
        run_logger.warning(
            "evaluation.coupled_brownian is ignored for sde_matching: "
            "the learned latent SDE Brownian motion is not the simulator Brownian driver."
        )

    train_config = SDEMatchingTrainConfig(
        epochs=int(cfg.train.epochs),
        steps=cfg.train.steps,
        steps_per_epoch=cfg.train.steps_per_epoch,
        batch_size=int(cfg.train.batch_size),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
        grad_clip_norm=cfg.train.grad_clip_norm,
        log_every=int(cfg.train.log_every),
        eval_every=int(cfg.train.eval_every),
        sample_inner_steps=int(cfg.train.sample_inner_steps),
        metrics_every_epoch=int(cfg.evaluation.every_epoch) if bool(cfg.evaluation.enabled) else 0,
        metrics_n_paths=int(cfg.evaluation.n_paths),
        metrics_num_quantiles=int(cfg.evaluation.num_quantiles),
        metrics_align_initial=bool(cfg.evaluation.align_initial),
        checkpoint_path=checkpoint_path,
    )
    history = train_sde_matching(
        model=model,
        dataloader=dataloader,
        ts=ts,
        device=device,
        logger=run_logger,
        config=train_config,
        metric_real_paths=paths if bool(cfg.evaluation.enabled) else None,
    )

    maybe_save_plots(
        cfg=cfg,
        fig_dir=run_paths.fig_dir,
        history=history,
        model=model,
        ts=ts,
        real_paths=paths,
        device=device,
    )

    run_logger.info("Run finished. Run directory: {}", run_paths.run_dir)
    return history


if __name__ == "__main__":
    main()
