from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader

from src.metrics import compute_path_metrics
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
    init_mult_initial: float = 1.0
    init_mult_func: float = 1.0
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


def _make_optimizer(
    name: str,
    parameters,
    *,
    lr: float,
    weight_decay: float,
) -> Optimizer:
    name = name.lower()
    if name == "adadelta":
        return torch.optim.Adadelta(parameters, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, betas=(0.0, 0.9), weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def clip_linear_weights(model: nn.Module) -> None:
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Linear):
                limit = 1.0 / module.out_features
                module.weight.clamp_(-limit, limit)


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
) -> dict[str, torch.Tensor]:
    generator.eval()
    device = torch.device(device)
    n_paths = min(int(n_paths), real_paths.size(0))
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")

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
