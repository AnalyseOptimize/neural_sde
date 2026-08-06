from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.nets.sdegan import CDEDiscriminator, SDEGenerator
from src.simulators import DeterministicDriftSimulator
from src.train import SDEGANTrainConfig, train_sdegan
from utils.data import PathDataConfig, make_sdegan_dataset


class ConsoleLogger:
    def info(self, message: str, *args) -> None:
        print(message.format(*args))


def _print_shape(stage: str, tensor: torch.Tensor) -> None:
    print(f"{stage}: shape={tuple(tensor.shape)}")


def _make_time_grid_0_100(n_steps: int = 6) -> torch.Tensor:
    ts = torch.linspace(0.0, 100.0, n_steps)
    _print_shape("[grid] ts on [0, 100]", ts)
    return ts


def _make_constant_generator() -> SDEGenerator:
    generator = SDEGenerator(
        data_size=1,
        noise_size=1,
        noise_type="diagonal",
        drift_head="constant",
        diffusion_head="constant",
        hidden_size=8,
        num_layers=1,
        drift_init=0.01,
        diffusion_init=0.0,
        method="reversible_heun",
        dt=20.0,
    )
    print(
        "[constant-generator] data_size=1, drift_head=constant, "
        "diffusion_head=constant, method=reversible_heun, dt=20.0"
    )
    return generator


def _make_tiny_discriminator() -> CDEDiscriminator:
    discriminator = CDEDiscriminator(
        data_size=1,
        hidden_size=4,
        mlp_size=8,
        num_layers=1,
        method="reversible_heun",
        adjoint=True,
        dt=1.0,
    )
    print(
        "[discriminator] data_size=1, hidden_size=4, "
        "method=reversible_heun, adjoint=True, dt=1.0"
    )
    return discriminator


def test_constant_generator_simulates_on_0_100_and_reports_shapes() -> None:
    torch.manual_seed(0)

    ts = _make_time_grid_0_100(n_steps=6)
    y0 = torch.full((4, 1), 1.0)
    _print_shape("[input] y0", y0)

    generator = _make_constant_generator()

    paths = generator.sample_paths(ts, y0)
    _print_shape("[generator.sample_paths] generated paths", paths)

    coeffs = generator(ts, y0)
    _print_shape("[generator.forward] generated CDE coefficients", coeffs)

    discriminator = _make_tiny_discriminator()
    scores = discriminator(coeffs)
    _print_shape("[discriminator.forward] generated scores", scores)

    assert paths.shape == (4, 6, 1)
    assert coeffs.shape[:2] == (4, 6)
    assert coeffs.shape[-1] == 2
    assert scores.shape == (4,)
    assert torch.allclose(paths[:, 0, :], y0, atol=1e-6)


def test_train_sdegan_constant_model_smoke_on_0_100_and_reports_shapes() -> None:
    torch.manual_seed(1)

    simulator = DeterministicDriftSimulator(s0=1.0, mu=0.01)
    data_config = PathDataConfig(
        dataset_size=8,
        t_size=6,
        dt=20.0,
        normalize=False,
        shuffle=False,
        drop_last=True,
    )
    data = make_sdegan_dataset(
        simulator=simulator,
        config=data_config,
        batch_size=4,
        shuffle=False,
        drop_last=True,
        device="cpu",
    )

    print("[dataset] deterministic process dS_t = mu dt on [0, 100]")
    _print_shape("[dataset] ts", data.ts)
    _print_shape("[dataset] paths", data.paths)
    _print_shape("[dataset] initial points y0", data.y0)
    _print_shape("[dataset] CDE coefficients", data.coeffs)

    real_coeffs, batch_y0 = next(iter(data.dataloader))
    _print_shape("[dataloader] batch real coefficients", real_coeffs)
    _print_shape("[dataloader] batch y0", batch_y0)

    generator = _make_constant_generator()
    discriminator = _make_tiny_discriminator()

    with torch.no_grad():
        generated_before_training = generator(data.ts, batch_y0)
    _print_shape("[pre-train generator.forward] generated coefficients", generated_before_training)

    with torch.no_grad():
        fake_for_discriminator = generator(data.ts, batch_y0)
        real_scores = discriminator(real_coeffs)
        fake_scores = discriminator(fake_for_discriminator)
        discriminator_loss = fake_scores.mean() - real_scores.mean()

        fake_for_generator = generator(data.ts, batch_y0)
        generator_scores = discriminator(fake_for_generator)
        generator_loss = -generator_scores.mean()

    _print_shape("[train-step trace] real coefficients", real_coeffs)
    _print_shape("[train-step trace] y0 conditioning", batch_y0)
    _print_shape("[train-step trace] fake coefficients for discriminator", fake_for_discriminator)
    _print_shape("[train-step trace] discriminator(real)", real_scores)
    _print_shape("[train-step trace] discriminator(fake)", fake_scores)
    _print_shape("[train-step trace] discriminator loss", discriminator_loss)
    _print_shape("[train-step trace] fake coefficients for generator", fake_for_generator)
    _print_shape("[train-step trace] discriminator(fake_for_generator)", generator_scores)
    _print_shape("[train-step trace] generator loss", generator_loss)

    train_config = SDEGANTrainConfig(
        epochs=2,
        steps=2,
        steps_per_epoch=1,
        batch_size=4,
        generator_lr=1e-3,
        discriminator_lr=1e-3,
        weight_decay=0.0,
        optimizer="adam",
        n_critic=1,
        clip_discriminator=True,
        swa_start_step=None,
        log_every=1,
        eval_every=0,
        metrics_every_epoch=0,
        checkpoint_path=None,
    )

    history = train_sdegan(
        generator=generator,
        discriminator=discriminator,
        dataloader=data.dataloader,
        ts=data.ts,
        device="cpu",
        logger=ConsoleLogger(),
        config=train_config,
        metric_real_paths=None,
    )

    print("[history] scalar values are not printed, only list lengths")
    print(f"[history] loss_d: length={len(history['loss_d'])}")
    print(f"[history] loss_g: length={len(history['loss_g'])}")
    print(f"[history] loss_d_epoch: length={len(history['loss_d_epoch'])}")
    print(f"[history] loss_g_epoch: length={len(history['loss_g_epoch'])}")

    assert len(history["loss_d"]) == 2
    assert len(history["loss_g"]) == 2
    assert len(history["loss_d_epoch"]) == 2
    assert len(history["loss_g_epoch"]) == 2


def run_smoke_tests() -> None:
    print("=" * 80)
    print("[run] Constant SDEGenerator smoke test")
    test_constant_generator_simulates_on_0_100_and_reports_shapes()

    print("=" * 80)
    print("[run] SDEGAN training smoke test")
    test_train_sdegan_constant_model_smoke_on_0_100_and_reports_shapes()

    print("=" * 80)
    print("[run] all SDEGAN smoke tests passed")


if __name__ == "__main__":
    run_smoke_tests()
