from __future__ import annotations

from dataclasses import dataclass
import os
import random
import sys
from pathlib import Path
from typing import Any, Optional

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import compute_path_metrics
from src.nets.sdegan_maxim import MaximModelConfig, MaximSDEGAN, build_maxim_sdegan
from src.simulators import (
    ArithmeticBrownianMotionSimulator,
    DeterministicDriftSimulator,
    MultiDimensionalGBMSimulator,
    OUSimulator,
    PerturbedPathSimulator,
)
from src.train import (
    _cycle,
    _current_lr,
    _gradient_log_norm,
    _make_optimizer,
    _make_scheduler,
    clip_linear_weights,
)
from utils.data import PathDataConfig, normalize_paths_by_initial, validate_paths
from utils.logging import log_config, setup_logger
from utils.visual import (
    plot_constant_coefficient_history,
    plot_epoch_diagnostics,
    plot_real_generated_paths,
)


@dataclass(frozen=True)
class MaximSDEGANData:
    ts: torch.Tensor
    paths: torch.Tensor
    dataloader: DataLoader

    @property
    def data_size(self) -> int:
        return int(self.paths.size(-1))


@dataclass(frozen=True)
class MaximSDEGANTrainConfig:
    epochs: int = 150
    steps: Optional[int] = None
    steps_per_epoch: Optional[int] = None
    batch_size: int = 1024
    generator_lr: float = 1e-4
    discriminator_lr: float = 1e-1
    weight_decay: float = 0.0
    optimizer: str = "adam"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    scheduler: Optional[str] = "step"
    scheduler_step_size: int = 800
    scheduler_gamma: float = 0.8
    n_critic: int = 1
    clip_discriminator: bool = True
    log_every: int = 10
    eval_every: int = 0
    metrics_every_epoch: int = 1
    metrics_n_paths: int = 512
    metrics_num_quantiles: int = 1024
    metrics_align_initial: bool = True
    checkpoint_path: Optional[str] = None


def _to_container(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)


def _plain_config_value(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


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
    gaussian = perturbations.get("gaussian", {})
    jumps = perturbations.get("jumps", {})
    gaussian_enabled = bool(gaussian.get("enabled", False))
    jump_enabled = bool(jumps.get("enabled", False))

    gaussian_variance = 0.0
    gaussian_include_initial = True
    gaussian_seed = None
    if gaussian_enabled:
        gaussian_variance = _as_float_or_list(
            gaussian.get("variance", 0.0),
            dim=data_size,
            name="simulator.perturbations.gaussian.variance",
        )
        gaussian_include_initial = bool(gaussian.get("include_initial", True))
        gaussian_seed = gaussian.get("seed")
        if gaussian_seed is None:
            gaussian_seed = int(cfg.seed) + 100_003

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
        gaussian_variance=gaussian_variance,
        gaussian_include_initial=gaussian_include_initial,
        gaussian_seed=None if gaussian_seed is None else int(gaussian_seed),
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


def make_maxim_sdegan_dataset(
    *,
    simulator,
    config: PathDataConfig,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
) -> MaximSDEGANData:
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
        device="cpu",
        dtype=torch.float32,
    )
    paths = validate_paths(paths)
    if config.normalize:
        paths = normalize_paths_by_initial(paths)

    dataloader = DataLoader(
        TensorDataset(paths),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return MaximSDEGANData(ts=ts, paths=paths, dataloader=dataloader)


def _batch_from_paths(paths: torch.Tensor, ts: torch.Tensor, *, device: torch.device) -> dict:
    paths = paths.to(device=device, dtype=ts.dtype)
    ts = ts.to(device=device, dtype=paths.dtype)
    batch_size = paths.size(0)
    ts_target = ts.view(1, -1).expand(batch_size, -1)
    return {
        "batch_size": batch_size,
        "valTarget": paths,
        "valHistory": paths[:, :1, :],
        "tsTarget": ts_target,
        "tsHistory": ts[:1].view(1, 1).expand(batch_size, 1),
        "ts": ts_target,
    }


def _unpack_path_batch(batch, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(batch, torch.Tensor):
        paths = batch
    elif isinstance(batch, (tuple, list)) and len(batch) > 0:
        paths = batch[0]
    else:
        raise TypeError("Expected batch to contain paths.")
    return paths.to(device=device, dtype=dtype)


@torch.no_grad()
def evaluate_maxim_sdegan_loss(
    model: MaximSDEGAN,
    dataloader: DataLoader,
    ts: torch.Tensor,
    *,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> float:
    model.eval()
    total_loss = 0.0
    total_size = 0
    ts = ts.to(device)
    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        paths = _unpack_path_batch(batch, device=device, dtype=ts.dtype)
        gan_batch = _batch_from_paths(paths, ts, device=device)
        gan_batch = model.generator(gan_batch)
        gan_batch = model.discriminator(gan_batch)
        loss = gan_batch["generatedScore"] - gan_batch["realScore"]
        total_loss += float(loss.item()) * paths.size(0)
        total_size += paths.size(0)

    if total_size == 0:
        raise ValueError("Cannot evaluate on an empty dataloader.")
    return total_loss / total_size


@torch.no_grad()
def evaluate_maxim_path_metrics(
    model: MaximSDEGAN,
    real_paths: torch.Tensor,
    ts: torch.Tensor,
    *,
    n_paths: int,
    num_quantiles: int,
    align_initial: bool,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    model.eval()
    ts = ts.to(device)
    n_paths = min(int(n_paths), real_paths.size(0))
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")
    real = real_paths[:n_paths].to(device=device, dtype=ts.dtype)
    generated = model.sample_paths(ts, real[:, 0, :])
    if align_initial:
        generated = generated - generated[:, :1, :] + real[:, :1, :]
    return compute_path_metrics(real, generated, num_quantiles=num_quantiles)


@torch.no_grad()
def _maxim_constant_values(model: MaximSDEGAN, coefficient: str) -> Optional[list[float]]:
    head = getattr(model.generator._func, coefficient, None)
    if head is None or getattr(head, "head_type", None) != "constant":
        return None
    value = getattr(head.module, "value", None)
    if value is None:
        return None
    return value.detach().cpu().reshape(-1).tolist()


def _append_constant_history(history: dict[str, list], model: MaximSDEGAN, epoch: int) -> None:
    for coefficient in ("drift", "diffusion"):
        values = _maxim_constant_values(model, coefficient)
        if values is None:
            continue
        history[f"constant_{coefficient}_epoch"].append(epoch)
        history[f"constant_{coefficient}_values"].append(values)


def train_maxim_sdegan(
    *,
    model: MaximSDEGAN,
    dataloader: DataLoader,
    ts: torch.Tensor,
    device: torch.device,
    logger,
    config: MaximSDEGANTrainConfig,
    metric_real_paths: Optional[torch.Tensor] = None,
) -> dict[str, list[float]]:
    if config.epochs < 1:
        raise ValueError("epochs must be >= 1")
    if config.steps is not None and config.steps < 1:
        raise ValueError("steps must be >= 1 when provided")
    if config.steps_per_epoch is not None and config.steps_per_epoch < 1:
        raise ValueError("steps_per_epoch must be >= 1 when provided")
    if config.n_critic < 1:
        raise ValueError("n_critic must be >= 1")
    if hasattr(dataloader, "__len__") and len(dataloader) == 0:
        raise ValueError(
            "dataloader is empty; lower batch_size or disable drop_last for small datasets."
        )

    model = model.to(device)
    ts = ts.to(device)
    generator_optimizer = _make_optimizer(
        config.optimizer,
        model.generator.parameters(),
        lr=config.generator_lr,
        weight_decay=config.weight_decay,
        adam_betas=(config.adam_beta1, config.adam_beta2),
    )
    discriminator_optimizer = _make_optimizer(
        config.optimizer,
        model.discriminator.parameters(),
        lr=config.discriminator_lr,
        weight_decay=config.weight_decay,
        adam_betas=(config.adam_beta1, config.adam_beta2),
    )
    generator_scheduler = _make_scheduler(
        config.scheduler,
        generator_optimizer,
        step_size=config.scheduler_step_size,
        gamma=config.scheduler_gamma,
    )
    discriminator_scheduler = _make_scheduler(
        config.scheduler,
        discriminator_optimizer,
        step_size=config.scheduler_step_size,
        gamma=config.scheduler_gamma,
    )

    steps_per_epoch = config.steps_per_epoch or len(dataloader)
    total_steps = config.steps or (config.epochs * steps_per_epoch)
    total_epochs = (total_steps + steps_per_epoch - 1) // steps_per_epoch
    if config.steps is None and config.steps_per_epoch is None and hasattr(logger, "warning"):
        logger.warning(
            "SDEGAN Maxim steps_per_epoch is inferred from len(dataloader)={}. "
            "Changing dataset_size with a fixed batch_size changes the number of optimizer updates. "
            "Set train.steps or train.steps_per_epoch explicitly for comparable runs.",
            len(dataloader),
        )

    history: dict[str, list[float]] = {
        "loss_d": [],
        "loss_g": [],
        "loss_d_epoch": [],
        "loss_g_epoch": [],
        "critic_real": [],
        "critic_fake": [],
        "critic_real_epoch": [],
        "critic_fake_epoch": [],
        "lr_generator": [],
        "lr_discriminator": [],
        "grad_norm": [],
        "eval_loss": [],
        "eval_epoch": [],
        "epoch": [],
        "metrics_epoch": [],
        "expected_supremum_squared_error": [],
        "marginal_w1_max": [],
        "marginal_w1_average": [],
        "constant_drift_epoch": [],
        "constant_drift_values": [],
        "constant_diffusion_epoch": [],
        "constant_diffusion_values": [],
    }

    batches = _cycle(dataloader)
    logger.info(
        "Reference SDEGAN training started: epochs={}, steps={}, steps_per_epoch={}, batch_size={}, optimizer={}, adam_betas=({}, {}), scheduler={}, scheduler_step_size={}, scheduler_gamma={}, discriminator_steps={}, device={}",
        total_epochs,
        total_steps,
        steps_per_epoch,
        config.batch_size,
        config.optimizer,
        config.adam_beta1,
        config.adam_beta2,
        config.scheduler,
        config.scheduler_step_size,
        config.scheduler_gamma,
        config.n_critic,
        device,
    )

    epoch_loss_d: list[float] = []
    epoch_loss_g: list[float] = []
    epoch_real_score: list[float] = []
    epoch_fake_score: list[float] = []

    for step in range(1, total_steps + 1):
        epoch = (step - 1) // steps_per_epoch + 1
        loss = None
        real_score = None
        generated_score = None

        model.train()
        for _ in range(config.n_critic):
            paths = _unpack_path_batch(next(batches), device=device, dtype=ts.dtype)
            gan_batch = _batch_from_paths(paths, ts, device=device)

            generator_optimizer.zero_grad(set_to_none=True)
            discriminator_optimizer.zero_grad(set_to_none=True)
            gan_batch = model.generator(gan_batch)
            gan_batch = model.discriminator(gan_batch)
            loss = gan_batch["generatedScore"] - gan_batch["realScore"]
            real_score = gan_batch["realScore"]
            generated_score = gan_batch["generatedScore"]
            loss.backward()
            discriminator_optimizer.step()

        if loss is None or real_score is None or generated_score is None:
            raise RuntimeError("No SDEGAN discriminator step was executed.")

        for parameter in model.generator.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(-1.0)
        generator_optimizer.step()

        if generator_scheduler is not None:
            generator_scheduler.step()
        if discriminator_scheduler is not None:
            discriminator_scheduler.step()
        if config.clip_discriminator:
            clip_linear_weights(model.discriminator)

        loss_d_value = float(loss.detach().item())
        loss_g_value = float((-generated_score).detach().item())
        real_score_value = float(real_score.detach().item())
        fake_score_value = float(generated_score.detach().item())
        grad_norm = _gradient_log_norm(model.generator, model.discriminator)

        history["loss_d"].append(loss_d_value)
        history["loss_g"].append(loss_g_value)
        history["critic_real"].append(real_score_value)
        history["critic_fake"].append(fake_score_value)
        history["lr_generator"].append(_current_lr(generator_optimizer))
        history["lr_discriminator"].append(_current_lr(discriminator_optimizer))
        history["grad_norm"].append(grad_norm)
        epoch_loss_d.append(loss_d_value)
        epoch_loss_g.append(loss_g_value)
        epoch_real_score.append(real_score_value)
        epoch_fake_score.append(fake_score_value)

        is_epoch_end = step % steps_per_epoch == 0 or step == total_steps
        should_log = step == 1 or step % config.log_every == 0 or is_epoch_end
        should_eval = config.eval_every > 0 and (
            step % config.eval_every == 0 or step == total_steps
        )
        if should_eval:
            eval_loss = evaluate_maxim_sdegan_loss(
                model,
                dataloader,
                ts,
                device=device,
                max_batches=8,
            )
            history["eval_loss"].append(eval_loss)
            history["eval_epoch"].append(epoch)
        else:
            eval_loss = None

        if should_log:
            if eval_loss is None:
                logger.info(
                    "epoch={}/{} step={}/{} loss_d={:.6f} loss_g={:.6f} d_real={:.6f} d_fake={:.6f} lr_g={:.6g} lr_d={:.6g} grad_log_norm={:.6f}",
                    epoch,
                    total_epochs,
                    step,
                    total_steps,
                    loss_d_value,
                    loss_g_value,
                    real_score_value,
                    fake_score_value,
                    history["lr_generator"][-1],
                    history["lr_discriminator"][-1],
                    grad_norm,
                )
            else:
                logger.info(
                    "epoch={}/{} step={}/{} loss_d={:.6f} loss_g={:.6f} eval_loss={:.6f} d_real={:.6f} d_fake={:.6f} lr_g={:.6g} lr_d={:.6g} grad_log_norm={:.6f}",
                    epoch,
                    total_epochs,
                    step,
                    total_steps,
                    loss_d_value,
                    loss_g_value,
                    eval_loss,
                    real_score_value,
                    fake_score_value,
                    history["lr_generator"][-1],
                    history["lr_discriminator"][-1],
                    grad_norm,
                )

        if is_epoch_end:
            history["epoch"].append(epoch)
            history["loss_d_epoch"].append(float(torch.tensor(epoch_loss_d).mean().item()))
            history["loss_g_epoch"].append(float(torch.tensor(epoch_loss_g).mean().item()))
            history["critic_real_epoch"].append(float(torch.tensor(epoch_real_score).mean().item()))
            history["critic_fake_epoch"].append(float(torch.tensor(epoch_fake_score).mean().item()))
            _append_constant_history(history, model, epoch)

            should_compute_metrics = (
                metric_real_paths is not None
                and config.metrics_every_epoch > 0
                and (epoch % config.metrics_every_epoch == 0 or step == total_steps)
            )
            if should_compute_metrics:
                metrics = evaluate_maxim_path_metrics(
                    model,
                    metric_real_paths,
                    ts,
                    n_paths=config.metrics_n_paths,
                    num_quantiles=config.metrics_num_quantiles,
                    align_initial=config.metrics_align_initial,
                    device=device,
                )
                history["metrics_epoch"].append(epoch)
                history["expected_supremum_squared_error"].append(
                    float(metrics["expected_supremum_squared_error"].detach().cpu().item())
                )
                history["marginal_w1_max"].append(
                    float(metrics["marginal_w1_max"].detach().cpu().item())
                )
                history["marginal_w1_average"].append(
                    float(metrics["marginal_w1_average"].detach().cpu().item())
                )
                logger.info(
                    "epoch={}/{} metrics e_sup={:.6f} w1_max={:.6f} w1_avg={:.6f}",
                    epoch,
                    total_epochs,
                    history["expected_supremum_squared_error"][-1],
                    history["marginal_w1_max"][-1],
                    history["marginal_w1_average"][-1],
                )

            epoch_loss_d = []
            epoch_loss_g = []
            epoch_real_score = []
            epoch_fake_score = []

    if config.checkpoint_path:
        checkpoint_path = Path(config.checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "generator_state_dict": model.generator.state_dict(),
                "discriminator_state_dict": model.discriminator.state_dict(),
                "generator_optimizer_state_dict": generator_optimizer.state_dict(),
                "discriminator_optimizer_state_dict": discriminator_optimizer.state_dict(),
                "generator_scheduler_state_dict": (
                    None if generator_scheduler is None else generator_scheduler.state_dict()
                ),
                "discriminator_scheduler_state_dict": (
                    None if discriminator_scheduler is None else discriminator_scheduler.state_dict()
                ),
                "history": history,
                "train_config": config.__dict__,
                "ts": ts.detach().cpu(),
            },
            checkpoint_path,
        )
        logger.info("Checkpoint saved: {}", checkpoint_path)

    logger.info("Reference SDEGAN training finished")
    return history


def _save_reference_simple_slices(
    *,
    model: MaximSDEGAN,
    coefficient: str,
    fig_dir: Path,
    n_grid: int,
    n_levels: int,
    hidden_dimension: int,
) -> None:
    head = getattr(model.generator._func, coefficient, None)
    if head is None or getattr(head, "head_type", None) != "simple":
        return

    parameter = next(model.generator.parameters())
    device = parameter.device
    dtype = parameter.dtype
    hidden_size = model.generator.hidden_size
    hidden_dimension = int(hidden_dimension)
    if hidden_dimension < 0 or hidden_dimension >= hidden_size:
        raise ValueError(f"hidden_dimension must be in [0, {hidden_size - 1}]")

    t_grid = torch.linspace(0.0, 1.0, int(n_grid), device=device, dtype=dtype)
    h_grid = torch.linspace(-2.0, 2.0, int(n_grid), device=device, dtype=dtype)
    h_levels = torch.linspace(-2.0, 2.0, int(n_levels), device=device, dtype=dtype)
    t_levels = torch.linspace(0.0, 1.0, int(n_levels), device=device, dtype=dtype)

    def select_component(values: torch.Tensor) -> torch.Tensor:
        if coefficient == "drift":
            return values[:, 0]
        if model.generator._func.noise_type == "diagonal":
            return values[:, 0]
        return values.view(values.size(0), hidden_size, model.generator.noise_size)[:, 0, 0]

    time_slices = []
    state_slices = []
    model.eval()
    with torch.no_grad():
        for h_value in h_levels:
            window = torch.zeros(t_grid.numel(), 1, hidden_size, device=device, dtype=dtype)
            window[:, 0, hidden_dimension] = h_value
            values = []
            for idx in range(t_grid.numel()):
                values.append(head(t_grid[idx], window[idx : idx + 1]).squeeze(0))
            time_slices.append(select_component(torch.stack(values)).detach().cpu().numpy())

        for t_value in t_levels:
            window = torch.zeros(h_grid.numel(), 1, hidden_size, device=device, dtype=dtype)
            window[:, 0, hidden_dimension] = h_grid
            values = head(t_value, window)
            state_slices.append(select_component(values).detach().cpu().numpy())

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
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), squeeze=False)
        ax_t, ax_h = axes[0]
        if int(n_levels) == 1:
            color_values = np.asarray([0.5])
        else:
            color_values = np.linspace(0.12, 0.84, int(n_levels))
        colors = [plt.get_cmap("viridis")(float(value)) for value in color_values]
        pretty = "Drift" if coefficient == "drift" else "Diffusion"
        for idx, values in enumerate(time_slices):
            ax_t.plot(
                t_grid.detach().cpu().numpy(),
                values,
                color=colors[idx],
                label=rf"$X={float(h_levels[idx]):.3g}$",
            )
        ax_t.set_xlabel("Normalized time")
        ax_t.set_ylabel(f"{pretty} coefficient")
        ax_t.set_title(rf"{pretty} as a function of $t$")
        ax_t.spines["top"].set_visible(False)
        ax_t.spines["right"].set_visible(False)
        ax_t.grid(True, color="0.88", linewidth=0.6)
        ax_t.legend(frameon=False, loc="best")

        for idx, values in enumerate(state_slices):
            ax_h.plot(
                h_grid.detach().cpu().numpy(),
                values,
                color=colors[idx],
                label=rf"$t={float(t_levels[idx]):.3g}$",
            )
        ax_h.set_xlabel(r"Hidden state $X_t$")
        ax_h.set_ylabel(f"{pretty} coefficient")
        ax_h.set_title(rf"{pretty} as a function of $X_t$")
        ax_h.spines["top"].set_visible(False)
        ax_h.spines["right"].set_visible(False)
        ax_h.grid(True, color="0.88", linewidth=0.6)
        ax_h.legend(frameon=False, loc="best")
        fig.tight_layout()

    save_path = fig_dir / f"maxim_simple_{coefficient}_hidden_slices.pdf"
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def maybe_save_plots(
    *,
    cfg: DictConfig,
    fig_dir: Path,
    history: dict[str, list[float]],
    model: MaximSDEGAN,
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
        fig = plot_epoch_diagnostics(
            history,
            option=option,
            save_path=fig_dir / f"{option}_by_epoch.pdf",
        )
        plt.close(fig)

    if history.get("constant_drift_values"):
        fig = plot_constant_coefficient_history(
            history,
            coefficient="drift",
            true_value=None,
            save_path=fig_dir / "constant_hidden_drift_by_epoch.pdf",
        )
        plt.close(fig)
    if history.get("constant_diffusion_values"):
        fig = plot_constant_coefficient_history(
            history,
            coefficient="diffusion",
            true_value=None,
            save_path=fig_dir / "constant_hidden_diffusion_by_epoch.pdf",
        )
        plt.close(fig)

    simple_cfg = cfg.plots.get("simple_slices", {})
    if bool(simple_cfg.get("enabled", True)):
        for coefficient in ("drift", "diffusion"):
            _save_reference_simple_slices(
                model=model,
                coefficient=coefficient,
                fig_dir=fig_dir,
                n_grid=int(simple_cfg.get("n_grid", 128)),
                n_levels=int(simple_cfg.get("n_levels", 5)),
                hidden_dimension=int(simple_cfg.get("hidden_dimension", 0)),
            )

    n_paths = min(int(cfg.plots.n_paths), real_paths.size(0))
    model.eval()
    with torch.no_grad():
        y0 = real_paths[:n_paths, 0, :].to(device=device, dtype=ts.dtype)
        generated = model.sample_paths(ts.to(device), y0).detach().cpu()
    fig = plot_real_generated_paths(
        ts.detach().cpu(),
        real_paths[:n_paths].detach().cpu(),
        generated,
        n_paths=n_paths,
        align_initial=bool(cfg.plots.align_initial),
        save_path=fig_dir / "real_vs_generated_paths.pdf",
    )
    plt.close(fig)


@hydra.main(version_base=None, config_path="../configs", config_name="sdegan_maxim")
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
    data = make_maxim_sdegan_dataset(
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

    generator_cfg = cfg.model.generator
    discriminator_cfg = cfg.model.discriminator
    model = build_maxim_sdegan(
        MaximModelConfig(
            data_size=data.data_size,
            hidden_size=int(generator_cfg.hidden_size),
            noise_size=int(generator_cfg.noise_size),
            noise_type=str(generator_cfg.noise_type),
            mlp_size=int(generator_cfg.mlp_size),
            num_layers=int(generator_cfg.num_layers),
            drift_head=str(generator_cfg.drift_head),
            diffusion_head=str(generator_cfg.diffusion_head),
            drift_window_size=int(generator_cfg.drift_window_size),
            diffusion_window_size=int(generator_cfg.diffusion_window_size),
            drift_init=_plain_config_value(generator_cfg.drift_init),
            diffusion_init=_plain_config_value(generator_cfg.diffusion_init),
            coefficient_tanh=bool(generator_cfg.coefficient_tanh),
            readout_bias=bool(generator_cfg.readout_bias),
            fusion_num_layers=int(generator_cfg.fusion_num_layers),
            fusion_last_bias=bool(generator_cfg.fusion_last_bias),
            method=str(generator_cfg.method),
            dt=generator_cfg.dt,
            discriminator_hidden_size=int(discriminator_cfg.hidden_size),
            discriminator_mlp_size=int(discriminator_cfg.mlp_size),
            discriminator_num_layers=int(discriminator_cfg.num_layers),
            discriminator_func_tanh=bool(discriminator_cfg.func_tanh),
            discriminator_initial_tanh=bool(discriminator_cfg.initial_tanh),
            discriminator_dt=float(discriminator_cfg.dt),
            discriminator_adjoint=bool(discriminator_cfg.adjoint),
        )
    )
    run_logger.info(
        "Reference models prepared: generator_params={}, discriminator_params={}, data_size={}, hidden_size={}, noise_type={}, noise_size={}, drift_head={}, diffusion_head={}",
        sum(param.numel() for param in model.generator.parameters()),
        sum(param.numel() for param in model.discriminator.parameters()),
        data.data_size,
        model.generator.hidden_size,
        model.generator._func.noise_type,
        model.generator.noise_size,
        model.generator.drift_head_type,
        model.generator.diffusion_head_type,
    )
    if bool(cfg.evaluation.get("coupled_brownian", False)):
        run_logger.warning(
            "evaluation.coupled_brownian is ignored for sdegan_maxim: "
            "the reference generator is latent and does not expose the simulator Brownian driver."
        )

    train_config = MaximSDEGANTrainConfig(
        epochs=int(cfg.train.epochs),
        steps=cfg.train.steps,
        steps_per_epoch=cfg.train.steps_per_epoch,
        batch_size=int(cfg.train.batch_size),
        generator_lr=float(cfg.train.generator_lr),
        discriminator_lr=float(cfg.train.discriminator_lr),
        weight_decay=float(cfg.train.weight_decay),
        optimizer=str(cfg.train.optimizer),
        adam_beta1=float(cfg.train.get("adam_beta1", 0.9)),
        adam_beta2=float(cfg.train.get("adam_beta2", 0.999)),
        scheduler=cfg.train.get("scheduler", "step"),
        scheduler_step_size=int(cfg.train.get("scheduler_step_size", 800)),
        scheduler_gamma=float(cfg.train.get("scheduler_gamma", 0.8)),
        n_critic=int(cfg.train.n_critic),
        clip_discriminator=bool(cfg.train.clip_discriminator),
        log_every=int(cfg.train.log_every),
        eval_every=int(cfg.train.eval_every),
        metrics_every_epoch=int(cfg.evaluation.every_epoch) if bool(cfg.evaluation.enabled) else 0,
        metrics_n_paths=int(cfg.evaluation.n_paths),
        metrics_num_quantiles=int(cfg.evaluation.num_quantiles),
        metrics_align_initial=bool(cfg.evaluation.align_initial),
        checkpoint_path=checkpoint_path,
    )
    history = train_maxim_sdegan(
        model=model,
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
        model=model,
        ts=data.ts,
        real_paths=data.paths,
        device=device,
    )

    run_logger.info("Run finished. Run directory: {}", run_paths.run_dir)
    return history


if __name__ == "__main__":
    main()
