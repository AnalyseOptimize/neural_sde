from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch
try:
    import torchsde
except ImportError:  # pragma: no cover - SDE-ML direct mode does not require torchsde.
    torchsde = None
from torch import nn
from torch.optim import Optimizer
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader

from src.metrics import compute_path_metrics
from src.simulators import sample_brownian_increments

if TYPE_CHECKING:
    from src.nets.sde_ml import SDEML
    from src.nets.sdegan import CDEDiscriminator, SDEGenerator


@dataclass
class SDEGANTrainConfig:
    epochs: int = 100
    steps: Optional[int] = None
    steps_per_epoch: Optional[int] = None
    batch_size: int = 1024
    generator_lr: float = 2e-4
    discriminator_lr: float = 1e-3
    weight_decay: float = 1e-2
    optimizer: str = "adadelta"
    n_critic: int = 1
    clip_discriminator: bool = True
    swa_start_step: Optional[int] = 5_000
    log_every: int = 10
    eval_every: int = 0
    metrics_every_epoch: int = 1
    metrics_n_paths: int = 512
    metrics_num_quantiles: int = 1024
    metrics_align_initial: bool = True
    metrics_coupled_brownian: bool = True
    metrics_brownian_seed: Optional[int] = 0
    init_mult_initial: float = 1.0
    init_mult_func: float = 1.0
    checkpoint_path: Optional[str] = None


@dataclass
class SDEMLTrainConfig:
    epochs: int = 100
    steps: Optional[int] = None
    steps_per_epoch: Optional[int] = None
    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adam"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    grad_clip_norm: Optional[float] = None
    log_every: int = 10
    eval_every: int = 0
    metrics_every_epoch: int = 1
    metrics_n_paths: int = 512
    metrics_num_quantiles: int = 1024
    metrics_align_initial: bool = True
    metrics_coupled_brownian: bool = True
    metrics_brownian_seed: Optional[int] = 0
    likelihood_backend: str = "direct"
    sampling_backend: str = "torchsde"
    include_initial_likelihood: bool = False
    initial_std: Optional[float] = None
    checkpoint_path: Optional[str] = None


def _cycle(loader: Iterable):
    while True:
        yield from loader


def _first_tensor(batch) -> torch.Tensor:
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, (tuple, list)) and len(batch) > 0:
        return batch[0]
    raise TypeError("Expected a Tensor batch or a non-empty tuple/list batch.")


def _unpack_sdegan_batch(batch, *, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, torch.Tensor):
        coeffs = batch
        y0 = coeffs[:, 0, 1:]
    elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
        coeffs, y0 = batch[0], batch[1]
    elif isinstance(batch, (tuple, list)) and len(batch) == 1:
        coeffs = batch[0]
        y0 = coeffs[:, 0, 1:]
    else:
        raise TypeError("Expected batch to contain CDE coefficients and optional y0.")
    coeffs = coeffs.to(device)
    y0 = y0.to(device=device, dtype=coeffs.dtype)
    return coeffs, y0


def _unpack_path_batch(batch, *, device: torch.device | str) -> torch.Tensor:
    if isinstance(batch, torch.Tensor):
        paths = batch
    elif isinstance(batch, (tuple, list)) and len(batch) > 0:
        paths = batch[0]
    else:
        raise TypeError("Expected batch to contain paths.")
    return paths.to(device)


def _make_optimizer(
    name: str,
    parameters,
    *,
    lr: float,
    weight_decay: float,
    adam_betas: tuple[float, float] = (0.0, 0.9),
) -> Optimizer:
    name = name.lower()
    if name == "adadelta":
        return torch.optim.Adadelta(parameters, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, betas=adam_betas, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def _normalize_sde_ml_likelihood_backend(backend: str) -> str:
    value = backend.lower()
    if value not in {"direct", "torchsde"}:
        raise ValueError("SDE-ML likelihood_backend must be one of: direct, torchsde")
    return value


def _normalize_sde_ml_sampling_backend(backend: str) -> str:
    value = backend.lower()
    if value not in {"torchsde", "direct"}:
        raise ValueError("SDE-ML sampling_backend must be one of: torchsde, direct")
    return value


def clip_linear_weights(model: nn.Module) -> None:
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Linear):
                limit = 1.0 / module.out_features
                module.weight.clamp_(-limit, limit)


def _regular_grid_dt(ts: torch.Tensor) -> float:
    increments = (ts[1:] - ts[:-1]).to(dtype=torch.float64)
    dt = increments.mean()
    tolerance = max(1e-8, abs(float(dt.detach().cpu().item())) * 1e-4)
    if bool((increments - dt).abs().max() > tolerance):
        raise ValueError("Coupled Brownian metrics require an equidistant time grid.")
    return float(dt.detach().cpu().item())


def _make_brownian_interval(
    ts: torch.Tensor,
    *,
    batch_size: int,
    noise_size: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: Optional[int],
) -> torchsde.BrownianInterval:
    if torchsde is None:
        raise ImportError("torchsde is required for BrownianInterval-based metrics.")
    dt = _regular_grid_dt(ts)
    return torchsde.BrownianInterval(
        t0=float(ts[0].detach().cpu().item()),
        t1=float(ts[-1].detach().cpu().item()),
        size=(batch_size, noise_size),
        dtype=dtype,
        device=device,
        entropy=None if seed is None else int(seed),
        dt=dt,
    )


def _make_grid_brownian_increments(
    ts: torch.Tensor,
    *,
    batch_size: int,
    noise_size: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: Optional[int],
) -> torch.Tensor:
    dt = _regular_grid_dt(ts)
    return sample_brownian_increments(
        n_paths=batch_size,
        n_steps=ts.numel(),
        dt=dt,
        noise_size=noise_size,
        device=device,
        dtype=dtype,
        seed=seed,
    )


def _brownian_grid_increments(
    bm: torchsde.BrownianInterval,
    ts: torch.Tensor,
) -> torch.Tensor:
    increments = []
    for idx in range(ts.numel() - 1):
        left = float(ts[idx].detach().cpu().item())
        right = float(ts[idx + 1].detach().cpu().item())
        increments.append(bm(left, right))
    return torch.stack(increments, dim=1)


def _simulate_coupled_real_paths(
    simulator,
    *,
    ts: torch.Tensor,
    brownian_increments: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not hasattr(simulator, "simulate_with_brownian"):
        raise TypeError(
            f"{simulator.__class__.__name__} does not implement simulate_with_brownian."
        )
    dt = _regular_grid_dt(ts)
    _, paths, _ = simulator.simulate_with_brownian(
        n_paths=brownian_increments.size(0),
        n_steps=ts.numel(),
        dt=dt,
        device=device,
        dtype=dtype,
        brownian_increments=brownian_increments,
    )
    return paths


def _sde_ml_negative_log_likelihood(
    model: "SDEML",
    ts: torch.Tensor,
    paths: torch.Tensor,
    *,
    likelihood_backend: str,
    include_initial_likelihood: bool,
    initial_std: Optional[float],
) -> torch.Tensor:
    likelihood_backend = _normalize_sde_ml_likelihood_backend(likelihood_backend)
    if likelihood_backend == "direct":
        return model.direct_negative_log_likelihood(
            ts,
            paths,
            reduction="mean",
        )
    if likelihood_backend == "torchsde":
        # torchsde.sdeint does not expose the transition density of an observed path.
        # This keeps the legacy observed-transition likelihood while allowing torchsde
        # to be selected for sampling/metrics via sampling_backend.
        return model.negative_log_likelihood(
            ts,
            paths,
            reduction="mean",
            include_initial=include_initial_likelihood,
            initial_std=initial_std,
        )
    raise AssertionError(f"Unhandled likelihood_backend={likelihood_backend}")


@torch.no_grad()
def _constant_coefficient_values(model, coefficient_name: str) -> Optional[list[float]]:
    if getattr(model, f"{coefficient_name}_head_type", None) != "constant":
        return None
    func = getattr(model, "func", None)
    if func is None:
        return None
    head = getattr(func, coefficient_name, None)
    if head is None:
        return None

    parameter = next(iter(head.parameters()), None)
    if parameter is None:
        parameter = next(iter(model.parameters()), None)
    if parameter is None:
        device = torch.device("cpu")
        dtype = torch.float32
    else:
        device = parameter.device
        dtype = parameter.dtype

    data_size = int(getattr(model, "data_size"))
    window_size = int(getattr(model, f"{coefficient_name}_window_size", 1))
    window = torch.zeros(1, max(window_size, 1), data_size, device=device, dtype=dtype)
    t = torch.zeros((), device=device, dtype=dtype)
    value = head(t, window)
    if coefficient_name == "diffusion" and getattr(model, "noise_type", "diagonal") == "general":
        value = value.view(1, data_size, int(getattr(model, "noise_size")))
    return value.reshape(-1).detach().cpu().tolist()


def _append_constant_coefficient_history(
    history: dict[str, list],
    model,
    epoch: int,
) -> None:
    for coefficient_name in ("drift", "diffusion"):
        values = _constant_coefficient_values(model, coefficient_name)
        if values is None:
            continue
        history[f"constant_{coefficient_name}_epoch"].append(epoch)
        history[f"constant_{coefficient_name}_values"].append(values)


@torch.no_grad()
def evaluate_sdegan_loss(
    generator: SDEGenerator,
    discriminator: CDEDiscriminator,
    dataloader: DataLoader,
    ts: torch.Tensor,
    *,
    device: torch.device | str,
    max_batches: Optional[int] = None,
) -> float:
    generator.eval()
    discriminator.eval()

    total_loss = 0.0
    total_size = 0
    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        real, y0 = _unpack_sdegan_batch(batch, device=device)
        batch_size = real.size(0)
        fake = generator(ts, y0)
        loss = discriminator(fake).mean() - discriminator(real).mean()
        total_loss += float(loss.item()) * batch_size
        total_size += batch_size

    if total_size == 0:
        raise ValueError("Cannot evaluate on an empty dataloader.")
    return total_loss / total_size


@torch.no_grad()
def evaluate_path_metrics(
    generator: SDEGenerator,
    real_paths: torch.Tensor,
    ts: torch.Tensor,
    *,
    n_paths: int,
    num_quantiles: int,
    align_initial: bool,
    device: torch.device | str,
    coupled_brownian: bool = True,
    brownian_seed: Optional[int] = 0,
    simulator=None,
) -> dict[str, torch.Tensor]:
    generator.eval()
    device = torch.device(device)
    ts = ts.to(device)
    n_paths = min(int(n_paths), real_paths.size(0))
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")

    if coupled_brownian and simulator is not None:
        if generator.noise_size != getattr(simulator, "data_size", generator.noise_size):
            raise ValueError(
                "Coupled Brownian metrics require generator.noise_size to match "
                f"simulator.data_size; got {generator.noise_size} and {simulator.data_size}."
            )
        bm = _make_brownian_interval(
            ts,
            batch_size=n_paths,
            noise_size=generator.noise_size,
            device=device,
            dtype=ts.dtype,
            seed=brownian_seed,
        )
        brownian_increments = _brownian_grid_increments(bm, ts)
        real = _simulate_coupled_real_paths(
            simulator,
            ts=ts,
            brownian_increments=brownian_increments,
            device=device,
            dtype=ts.dtype,
        )
        generated = generator.sample_paths(ts, real[:, 0, :], bm=bm)
    else:
        real = real_paths[:n_paths].to(device=device, dtype=ts.dtype)
        generated = generator.sample_paths(ts, real[:, 0, :])

    if align_initial:
        generated = generated - generated[:, :1, :] + real[:, :1, :]
    return compute_path_metrics(
        real,
        generated,
        num_quantiles=num_quantiles,
    )


def train_sdegan(
    *,
    generator: SDEGenerator,
    discriminator: CDEDiscriminator,
    dataloader: DataLoader,
    ts: torch.Tensor,
    device: torch.device | str,
    logger,
    config: Optional[SDEGANTrainConfig] = None,
    metric_real_paths: Optional[torch.Tensor] = None,
    metric_simulator=None,
) -> dict[str, list[float]]:
    config = SDEGANTrainConfig() if config is None else config
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

    device = torch.device(device)
    ts = ts.to(device)
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    generator.apply_initialization_scale(
        initial_scale=config.init_mult_initial,
        func_scale=config.init_mult_func,
    )

    generator_optimizer = _make_optimizer(
        config.optimizer,
        generator.parameters(),
        lr=config.generator_lr,
        weight_decay=config.weight_decay,
    )
    discriminator_optimizer = _make_optimizer(
        config.optimizer,
        discriminator.parameters(),
        lr=config.discriminator_lr,
        weight_decay=config.weight_decay,
    )

    averaged_generator = AveragedModel(generator)
    averaged_discriminator = AveragedModel(discriminator)
    steps_per_epoch = config.steps_per_epoch or len(dataloader)
    total_steps = config.steps or (config.epochs * steps_per_epoch)
    total_epochs = (total_steps + steps_per_epoch - 1) // steps_per_epoch
    use_swa = config.swa_start_step is not None and config.swa_start_step < total_steps
    swa_updates = 0

    history: dict[str, list[float]] = {
        "loss_d": [],
        "loss_g": [],
        "loss_d_epoch": [],
        "loss_g_epoch": [],
        "critic_real": [],
        "critic_fake": [],
        "critic_real_epoch": [],
        "critic_fake_epoch": [],
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
        "SDEGAN training started: epochs={}, steps={}, steps_per_epoch={}, batch_size={}, optimizer={}, n_critic={}, device={}",
        total_epochs,
        total_steps,
        steps_per_epoch,
        config.batch_size,
        config.optimizer,
        config.n_critic,
        device,
    )

    epoch_loss_d: list[float] = []
    epoch_loss_g: list[float] = []
    epoch_real_score: list[float] = []
    epoch_fake_score: list[float] = []

    for step in range(1, total_steps + 1):
        epoch = (step - 1) // steps_per_epoch + 1
        real, y0 = _unpack_sdegan_batch(next(batches), device=device)
        batch_size = real.size(0)

        discriminator.train()
        generator.eval()
        discriminator_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            fake = generator(ts, y0)
        real_score = discriminator(real)
        fake_score = discriminator(fake)
        loss_d = fake_score.mean() - real_score.mean()
        loss_d.backward()
        discriminator_optimizer.step()
        if config.clip_discriminator:
            clip_linear_weights(discriminator)

        history["loss_d"].append(float(loss_d.detach().item()))
        history["critic_real"].append(float(real_score.mean().detach().item()))
        history["critic_fake"].append(float(fake_score.mean().detach().item()))
        epoch_loss_d.append(history["loss_d"][-1])
        epoch_real_score.append(history["critic_real"][-1])
        epoch_fake_score.append(history["critic_fake"][-1])

        loss_g_value = history["loss_g"][-1] if history["loss_g"] else float("nan")
        if step % config.n_critic == 0:
            generator.train()
            discriminator.eval()
            generator_optimizer.zero_grad(set_to_none=True)
            fake = generator(ts, y0)
            loss_g = -discriminator(fake).mean()
            loss_g.backward()
            generator_optimizer.step()
            loss_g_value = float(loss_g.detach().item())
            history["loss_g"].append(loss_g_value)
            epoch_loss_g.append(loss_g_value)

        if use_swa and step > int(config.swa_start_step):
            averaged_generator.update_parameters(generator)
            averaged_discriminator.update_parameters(discriminator)
            swa_updates += 1

        is_epoch_end = step % steps_per_epoch == 0 or step == total_steps
        should_log = step == 1 or step % config.log_every == 0 or is_epoch_end
        should_eval = config.eval_every > 0 and (
            step % config.eval_every == 0 or step == total_steps
        )
        if should_eval:
            eval_loss = evaluate_sdegan_loss(
                generator,
                discriminator,
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
                    "epoch={}/{} step={}/{} loss_d={:.6f} loss_g={:.6f} d_real={:.6f} d_fake={:.6f}",
                    epoch,
                    total_epochs,
                    step,
                    total_steps,
                    history["loss_d"][-1],
                    loss_g_value,
                    history["critic_real"][-1],
                    history["critic_fake"][-1],
                )
            else:
                logger.info(
                    "epoch={}/{} step={}/{} loss_d={:.6f} loss_g={:.6f} eval_loss={:.6f} d_real={:.6f} d_fake={:.6f}",
                    epoch,
                    total_epochs,
                    step,
                    total_steps,
                    history["loss_d"][-1],
                    loss_g_value,
                    eval_loss,
                    history["critic_real"][-1],
                    history["critic_fake"][-1],
                )

        if is_epoch_end:
            history["epoch"].append(epoch)
            history["loss_d_epoch"].append(float(torch.tensor(epoch_loss_d).mean().item()))
            history["loss_g_epoch"].append(
                float(torch.tensor(epoch_loss_g).mean().item()) if epoch_loss_g else float("nan")
            )
            history["critic_real_epoch"].append(float(torch.tensor(epoch_real_score).mean().item()))
            history["critic_fake_epoch"].append(float(torch.tensor(epoch_fake_score).mean().item()))
            _append_constant_coefficient_history(history, generator, epoch)

            should_compute_metrics = (
                metric_real_paths is not None
                and config.metrics_every_epoch > 0
                and (epoch % config.metrics_every_epoch == 0 or step == total_steps)
            )
            if should_compute_metrics:
                metrics = evaluate_path_metrics(
                    generator,
                    metric_real_paths,
                    ts,
                    n_paths=config.metrics_n_paths,
                    num_quantiles=config.metrics_num_quantiles,
                    align_initial=config.metrics_align_initial,
                    device=device,
                    coupled_brownian=config.metrics_coupled_brownian,
                    brownian_seed=config.metrics_brownian_seed,
                    simulator=metric_simulator,
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

    if swa_updates > 0:
        generator.load_state_dict(averaged_generator.module.state_dict())
        discriminator.load_state_dict(averaged_discriminator.module.state_dict())
        logger.info("Loaded stochastic weight averaged parameters: updates={}", swa_updates)

    if config.checkpoint_path:
        checkpoint_path = Path(config.checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "generator_state_dict": generator.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "generator_optimizer_state_dict": generator_optimizer.state_dict(),
                "discriminator_optimizer_state_dict": discriminator_optimizer.state_dict(),
                "history": history,
                "train_config": config.__dict__,
                "ts": ts.detach().cpu(),
            },
            checkpoint_path,
        )
        logger.info("Checkpoint saved: {}", checkpoint_path)

    logger.info("SDEGAN training finished")
    return history


@torch.no_grad()
def evaluate_sde_ml_loss(
    model: SDEML,
    dataloader: DataLoader,
    ts: torch.Tensor,
    *,
    device: torch.device | str,
    likelihood_backend: str = "direct",
    include_initial_likelihood: bool = False,
    initial_std: Optional[float] = None,
    max_batches: Optional[int] = None,
) -> float:
    model.eval()
    device = torch.device(device)
    ts = ts.to(device)

    total_loss = 0.0
    total_size = 0
    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        paths = _unpack_path_batch(batch, device=device)
        paths = paths.to(dtype=ts.dtype)
        batch_size = paths.size(0)
        loss = _sde_ml_negative_log_likelihood(
            model,
            ts,
            paths,
            likelihood_backend=likelihood_backend,
            include_initial_likelihood=include_initial_likelihood,
            initial_std=initial_std,
        )
        total_loss += float(loss.item()) * batch_size
        total_size += batch_size

    if total_size == 0:
        raise ValueError("Cannot evaluate on an empty dataloader.")
    return total_loss / total_size


@torch.no_grad()
def evaluate_sde_ml_path_metrics(
    model: SDEML,
    real_paths: torch.Tensor,
    ts: torch.Tensor,
    *,
    n_paths: int,
    num_quantiles: int,
    align_initial: bool,
    device: torch.device | str,
    coupled_brownian: bool = True,
    brownian_seed: Optional[int] = 0,
    sampling_backend: str = "torchsde",
    simulator=None,
) -> dict[str, torch.Tensor]:
    model.eval()
    device = torch.device(device)
    ts = ts.to(device)
    sampling_backend = _normalize_sde_ml_sampling_backend(sampling_backend)
    n_paths = min(int(n_paths), real_paths.size(0))
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")

    if coupled_brownian and simulator is not None:
        if model.noise_size != getattr(simulator, "data_size", model.noise_size):
            raise ValueError(
                "Coupled Brownian metrics require model.noise_size to match "
                f"simulator.data_size; got {model.noise_size} and {simulator.data_size}."
            )
        if sampling_backend == "direct":
            bm = None
            brownian_increments = _make_grid_brownian_increments(
                ts,
                batch_size=n_paths,
                noise_size=model.noise_size,
                device=device,
                dtype=ts.dtype,
                seed=brownian_seed,
            )
        else:
            bm = _make_brownian_interval(
                ts,
                batch_size=n_paths,
                noise_size=model.noise_size,
                device=device,
                dtype=ts.dtype,
                seed=brownian_seed,
            )
            brownian_increments = _brownian_grid_increments(bm, ts)
        real = _simulate_coupled_real_paths(
            simulator,
            ts=ts,
            brownian_increments=brownian_increments,
            device=device,
            dtype=ts.dtype,
        )
        generated = model.sample_paths(
            ts,
            y0=real[:, 0, :],
            bm=bm,
            brownian_increments=brownian_increments if sampling_backend == "direct" else None,
            backend=sampling_backend,
        )
    else:
        real = real_paths[:n_paths].to(device=device, dtype=ts.dtype)
        generated = model.sample_paths(ts, batch_size=n_paths, backend=sampling_backend)

    if align_initial:
        generated = generated - generated[:, :1, :] + real[:, :1, :]
    return compute_path_metrics(
        real,
        generated,
        num_quantiles=num_quantiles,
    )


def train_sde_ml(
    *,
    model: SDEML,
    dataloader: DataLoader,
    ts: torch.Tensor,
    device: torch.device | str,
    logger,
    config: Optional[SDEMLTrainConfig] = None,
    metric_real_paths: Optional[torch.Tensor] = None,
    metric_simulator=None,
) -> dict[str, list[float]]:
    config = SDEMLTrainConfig() if config is None else config
    if config.epochs < 1:
        raise ValueError("epochs must be >= 1")
    if config.steps is not None and config.steps < 1:
        raise ValueError("steps must be >= 1 when provided")
    if config.steps_per_epoch is not None and config.steps_per_epoch < 1:
        raise ValueError("steps_per_epoch must be >= 1 when provided")
    if not (0.0 <= config.adam_beta1 < 1.0):
        raise ValueError("adam_beta1 must be in [0, 1)")
    if not (0.0 <= config.adam_beta2 < 1.0):
        raise ValueError("adam_beta2 must be in [0, 1)")
    if config.grad_clip_norm is not None and config.grad_clip_norm <= 0:
        raise ValueError("grad_clip_norm must be positive when provided")
    if hasattr(dataloader, "__len__") and len(dataloader) == 0:
        raise ValueError(
            "dataloader is empty; lower batch_size or disable drop_last for small datasets."
        )
    likelihood_backend = _normalize_sde_ml_likelihood_backend(config.likelihood_backend)
    sampling_backend = _normalize_sde_ml_sampling_backend(config.sampling_backend)
    if likelihood_backend == "torchsde" and hasattr(logger, "warning"):
        logger.warning(
            "likelihood_backend=torchsde uses the observed Euler transition likelihood; "
            "torchsde.sdeint is used only for sampling/metrics via sampling_backend=torchsde."
        )

    device = torch.device(device)
    ts = ts.to(device)
    model = model.to(device)
    optimizer = _make_optimizer(
        config.optimizer,
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        adam_betas=(config.adam_beta1, config.adam_beta2),
    )

    steps_per_epoch = config.steps_per_epoch or len(dataloader)
    total_steps = config.steps or (config.epochs * steps_per_epoch)
    total_epochs = (total_steps + steps_per_epoch - 1) // steps_per_epoch
    if config.steps is None and config.steps_per_epoch is None and hasattr(logger, "warning"):
        logger.warning(
            "SDE-ML steps_per_epoch is inferred from len(dataloader)={}. "
            "Changing dataset_size with a fixed batch_size changes the number of optimizer updates. "
            "Set train.steps or train.steps_per_epoch explicitly for comparable runs.",
            len(dataloader),
        )
    history: dict[str, list[float]] = {
        "loss": [],
        "negative_log_likelihood": [],
        "negative_log_likelihood_per_step": [],
        "log_likelihood": [],
        "loss_epoch": [],
        "negative_log_likelihood_epoch": [],
        "negative_log_likelihood_per_step_epoch": [],
        "log_likelihood_epoch": [],
        "eval_loss": [],
        "eval_loss_per_step": [],
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
        "SDE-ML training started: epochs={}, steps={}, steps_per_epoch={}, batch_size={}, optimizer={}, adam_betas=({}, {}), likelihood_backend={}, sampling_backend={}, device={}",
        total_epochs,
        total_steps,
        steps_per_epoch,
        config.batch_size,
        config.optimizer,
        config.adam_beta1,
        config.adam_beta2,
        likelihood_backend,
        sampling_backend,
        device,
    )

    epoch_losses: list[float] = []
    epoch_losses_per_step: list[float] = []
    epoch_log_likelihoods: list[float] = []
    for step in range(1, total_steps + 1):
        epoch = (step - 1) // steps_per_epoch + 1
        paths = _unpack_path_batch(next(batches), device=device).to(dtype=ts.dtype)
        normalizer = max((ts.numel() - 1) * paths.size(-1), 1)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = _sde_ml_negative_log_likelihood(
            model,
            ts,
            paths,
            likelihood_backend=likelihood_backend,
            include_initial_likelihood=config.include_initial_likelihood,
            initial_std=config.initial_std,
        )
        loss.backward()
        if config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()

        loss_value = float(loss.detach().item())
        loss_per_step_value = loss_value / normalizer
        log_likelihood_value = -loss_value
        history["loss"].append(loss_value)
        history["negative_log_likelihood"].append(loss_value)
        history["negative_log_likelihood_per_step"].append(loss_per_step_value)
        history["log_likelihood"].append(log_likelihood_value)
        epoch_losses.append(loss_value)
        epoch_losses_per_step.append(loss_per_step_value)
        epoch_log_likelihoods.append(log_likelihood_value)

        is_epoch_end = step % steps_per_epoch == 0 or step == total_steps
        should_log = step == 1 or step % config.log_every == 0 or is_epoch_end
        should_eval = config.eval_every > 0 and (
            step % config.eval_every == 0 or step == total_steps
        )
        if should_eval:
            eval_loss = evaluate_sde_ml_loss(
                model,
                dataloader,
                ts,
                device=device,
                likelihood_backend=likelihood_backend,
                include_initial_likelihood=config.include_initial_likelihood,
                initial_std=config.initial_std,
                max_batches=8,
            )
            history["eval_loss"].append(eval_loss)
            history["eval_loss_per_step"].append(eval_loss / normalizer)
            history["eval_epoch"].append(epoch)
        else:
            eval_loss = None

        if should_log:
            if eval_loss is None:
                logger.info(
                    "epoch={}/{} step={}/{} nll={:.6f} nll_per_step={:.6f} log_likelihood={:.6f}",
                    epoch,
                    total_epochs,
                    step,
                    total_steps,
                    loss_value,
                    loss_per_step_value,
                    log_likelihood_value,
                )
            else:
                logger.info(
                    "epoch={}/{} step={}/{} nll={:.6f} nll_per_step={:.6f} eval_nll={:.6f} eval_nll_per_step={:.6f} log_likelihood={:.6f}",
                    epoch,
                    total_epochs,
                    step,
                    total_steps,
                    loss_value,
                    loss_per_step_value,
                    eval_loss,
                    eval_loss / normalizer,
                    log_likelihood_value,
                )

        if is_epoch_end:
            history["epoch"].append(epoch)
            history["loss_epoch"].append(float(torch.tensor(epoch_losses).mean().item()))
            history["negative_log_likelihood_epoch"].append(history["loss_epoch"][-1])
            history["negative_log_likelihood_per_step_epoch"].append(
                float(torch.tensor(epoch_losses_per_step).mean().item())
            )
            history["log_likelihood_epoch"].append(
                float(torch.tensor(epoch_log_likelihoods).mean().item())
            )
            _append_constant_coefficient_history(history, model, epoch)

            should_compute_metrics = (
                metric_real_paths is not None
                and config.metrics_every_epoch > 0
                and (epoch % config.metrics_every_epoch == 0 or step == total_steps)
            )
            if should_compute_metrics:
                metrics = evaluate_sde_ml_path_metrics(
                    model,
                    metric_real_paths,
                    ts,
                    n_paths=config.metrics_n_paths,
                    num_quantiles=config.metrics_num_quantiles,
                    align_initial=config.metrics_align_initial,
                    device=device,
                    coupled_brownian=config.metrics_coupled_brownian,
                    brownian_seed=config.metrics_brownian_seed,
                    sampling_backend=sampling_backend,
                    simulator=metric_simulator,
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

            epoch_losses = []
            epoch_losses_per_step = []
            epoch_log_likelihoods = []

    if config.checkpoint_path:
        checkpoint_path = Path(config.checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "train_config": config.__dict__,
                "ts": ts.detach().cpu(),
            },
            checkpoint_path,
        )
        logger.info("Checkpoint saved: {}", checkpoint_path)

    logger.info("SDE-ML training finished")
    return history
