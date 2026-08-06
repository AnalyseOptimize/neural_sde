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
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nets.sde_ml import SDEML
from src.simulators import (
    ArithmeticBrownianMotionSimulator,
    DeterministicDriftSimulator,
    MultiDimensionalGBMSimulator,
    OUSimulator,
)
from src.train import SDEMLTrainConfig, train_sde_ml
from utils.data import PathDataConfig, normalize_paths_by_initial, validate_paths
from utils.logging import log_config, setup_logger
from utils.visual import (
    plot_constant_coefficient_history,
    plot_epoch_diagnostics,
    plot_real_generated_paths,
)


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
        return ArithmeticBrownianMotionSimulator(
            s0=_as_float_or_list(abm.s0, dim=dim, name="abm.s0"),
            mu=_as_float_or_list(abm.mu, dim=dim, name="abm.mu"),
            sigma=_as_float_or_list(abm.sigma, dim=dim, name="abm.sigma"),
            dim=dim,
            corr=corr,
            seed=int(cfg.seed),
        )

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


def true_constant_coefficients(simulator) -> dict[str, list[float]]:
    if not hasattr(simulator, "constant_coefficients"):
        return {}
    values = simulator.constant_coefficients()
    return {
        name: torch.as_tensor(value, dtype=torch.float32).flatten().tolist()
        for name, value in values.items()
    }


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


def resolve_initial_value(value, paths: torch.Tensor):
    if isinstance(value, str) and value.lower() == "auto":
        return paths[0, 0, :].detach().cpu().tolist()
    return _as_float_or_list(value, dim=paths.size(-1), name="model.initial_value")


def resolve_time_origin(value, ts: torch.Tensor) -> float:
    if isinstance(value, str) and value.lower() == "auto":
        return float(ts[0].detach().cpu().item())
    return float(value)


def resolve_time_scale(value, ts: torch.Tensor) -> float:
    if isinstance(value, str) and value.lower() == "auto":
        return max(float((ts[-1] - ts[0]).detach().cpu().item()), 1.0)
    return float(value)


def resolve_state_center(value, paths: torch.Tensor) -> float | list[float]:
    if isinstance(value, str) and value.lower() == "auto":
        return paths.flatten(0, 1).mean(dim=0).detach().cpu().tolist()
    return _as_float_or_list(value, dim=paths.size(-1), name="model.state_center")


def resolve_state_scale(value, paths: torch.Tensor) -> float | list[float]:
    if isinstance(value, str) and value.lower() == "auto":
        scale = paths.flatten(0, 1).abs().quantile(0.90, dim=0).clamp_min(1.0)
        return scale.detach().cpu().tolist()
    return _as_float_or_list(value, dim=paths.size(-1), name="model.state_scale")


def resolve_optional_float(value):
    if value is None:
        return None
    return float(value)


def maybe_save_plots(
    *,
    cfg: DictConfig,
    fig_dir: Path,
    history: dict[str, list[float]],
    model: SDEML,
    ts: torch.Tensor,
    real_paths: torch.Tensor,
    true_coefficients: dict[str, list[float]],
    device: torch.device,
) -> None:
    if not bool(cfg.plots.enabled):
        return

    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_epoch_diagnostics(
        history,
        option="loss",
        save_path=fig_dir / "negative_log_likelihood_by_epoch.pdf",
    )
    if history.get("marginal_w1_average") or history.get("marginal_w1_max"):
        plot_epoch_diagnostics(
            history,
            option="w1",
            save_path=fig_dir / "w1_by_epoch.pdf",
        )
    if history.get("expected_supremum_squared_error"):
        plot_epoch_diagnostics(
            history,
            option="e_sup",
            save_path=fig_dir / "e_sup_by_epoch.pdf",
        )
    if history.get("constant_drift_values"):
        plot_constant_coefficient_history(
            history,
            coefficient="drift",
            true_value=true_coefficients.get("drift"),
            save_path=fig_dir / "constant_drift_by_epoch.pdf",
        )
    if history.get("constant_diffusion_values"):
        plot_constant_coefficient_history(
            history,
            coefficient="diffusion",
            true_value=true_coefficients.get("diffusion"),
            save_path=fig_dir / "constant_diffusion_by_epoch.pdf",
        )

    n_paths = min(int(cfg.plots.n_paths), real_paths.size(0))
    sampling_backend = str(cfg.train.get("sampling_backend", cfg.model.get("sampling_backend", "torchsde")))
    model.eval()
    with torch.no_grad():
        generated = model.sample_paths(
            ts.to(device=device),
            batch_size=n_paths,
            backend=sampling_backend,
        ).detach().cpu()
    plot_real_generated_paths(
        ts.detach().cpu(),
        real_paths[:n_paths].detach().cpu(),
        generated,
        n_paths=n_paths,
        align_initial=bool(cfg.plots.align_initial),
        save_path=fig_dir / "real_vs_generated_paths.pdf",
    )


@hydra.main(version_base=None, config_path="../configs", config_name="sde_ml")
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

    model_cfg = cfg.model
    time_origin = resolve_time_origin(model_cfg.time_origin, ts)
    time_scale = resolve_time_scale(model_cfg.time_scale, ts)
    state_center = resolve_state_center(model_cfg.state_center, paths)
    state_scale = resolve_state_scale(model_cfg.state_scale, paths)
    input_clip = resolve_optional_float(model_cfg.input_clip)
    model = SDEML(
        data_size=paths.size(-1),
        noise_size=int(model_cfg.noise_size),
        noise_type=str(model_cfg.noise_type),
        sde_type=str(model_cfg.sde_type),
        drift_head=str(model_cfg.drift_head),
        diffusion_head=str(model_cfg.diffusion_head),
        drift_window_size=int(model_cfg.drift_window_size),
        diffusion_window_size=int(model_cfg.diffusion_window_size),
        hidden_size=int(model_cfg.hidden_size),
        num_layers=int(model_cfg.num_layers),
        drift_init=float(model_cfg.drift_init),
        diffusion_init=float(model_cfg.diffusion_init),
        drift_scale=float(model_cfg.drift_scale),
        diffusion_scale=float(model_cfg.diffusion_scale),
        final_tanh=bool(model_cfg.final_tanh),
        variance_floor=float(model_cfg.variance_floor),
        diffusion_min=float(model_cfg.diffusion_min),
        initial_value=resolve_initial_value(model_cfg.initial_value, paths),
        learn_initial=bool(model_cfg.learn_initial),
        time_origin=time_origin,
        time_scale=time_scale,
        state_center=state_center,
        state_scale=state_scale,
        input_clip=input_clip,
        method=str(model_cfg.method),
        dt=model_cfg.dt,
        adjoint=bool(model_cfg.adjoint),
        sampling_backend=str(model_cfg.get("sampling_backend", "torchsde")),
    )
    run_logger.info(
        "Model prepared: params={}, data_size={}, noise_type={}, sde_type={}, drift_head={}, diffusion_head={}",
        sum(param.numel() for param in model.parameters()),
        model.data_size,
        model.noise_type,
        model.sde_type,
        model.drift_head_type,
        model.diffusion_head_type,
    )
    run_logger.info(
        "Model input scaling: time_origin={}, time_scale={}, state_center={}, state_scale={}, input_clip={}",
        time_origin,
        time_scale,
        state_center,
        state_scale,
        input_clip,
    )

    train_config = SDEMLTrainConfig(
        epochs=int(cfg.train.epochs),
        steps=cfg.train.steps,
        steps_per_epoch=cfg.train.steps_per_epoch,
        batch_size=int(cfg.train.batch_size),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
        optimizer=str(cfg.train.optimizer),
        adam_beta1=float(cfg.train.get("adam_beta1", 0.9)),
        adam_beta2=float(cfg.train.get("adam_beta2", 0.999)),
        grad_clip_norm=cfg.train.grad_clip_norm,
        log_every=int(cfg.train.log_every),
        eval_every=int(cfg.train.eval_every),
        metrics_every_epoch=int(cfg.evaluation.every_epoch) if bool(cfg.evaluation.enabled) else 0,
        metrics_n_paths=int(cfg.evaluation.n_paths),
        metrics_num_quantiles=int(cfg.evaluation.num_quantiles),
        metrics_align_initial=bool(cfg.evaluation.align_initial),
        metrics_coupled_brownian=bool(cfg.evaluation.get("coupled_brownian", True)),
        metrics_brownian_seed=cfg.evaluation.get("brownian_seed"),
        likelihood_backend=str(cfg.train.get("likelihood_backend", "direct")),
        sampling_backend=str(cfg.train.get("sampling_backend", model.sampling_backend)),
        include_initial_likelihood=bool(cfg.train.include_initial_likelihood),
        initial_std=cfg.train.initial_std,
        checkpoint_path=checkpoint_path,
    )
    metric_simulator = None
    if bool(cfg.evaluation.enabled) and bool(cfg.evaluation.get("coupled_brownian", True)):
        if data_config.normalize:
            run_logger.warning(
                "Coupled Brownian E-sup is disabled because data.normalize=true. "
                "Use normalize=false for pathwise metrics in the simulator scale."
            )
        else:
            metric_simulator = simulator

    history = train_sde_ml(
        model=model,
        dataloader=dataloader,
        ts=ts,
        device=device,
        logger=run_logger,
        config=train_config,
        metric_real_paths=paths if bool(cfg.evaluation.enabled) else None,
        metric_simulator=metric_simulator,
    )

    maybe_save_plots(
        cfg=cfg,
        fig_dir=run_paths.fig_dir,
        history=history,
        model=model,
        ts=ts,
        real_paths=paths,
        true_coefficients=true_constant_coefficients(simulator),
        device=device,
    )

    run_logger.info("Run finished. Run directory: {}", run_paths.run_dir)
    return history


if __name__ == "__main__":
    main()
