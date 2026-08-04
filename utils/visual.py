from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch


PlotOption = Literal["loss", "w1", "e_sup"]


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_paths(value, *, name: str) -> np.ndarray:
    paths = _to_numpy(value).astype(float, copy=False)
    if paths.ndim == 2:
        paths = paths[..., None]
    if paths.ndim != 3:
        raise ValueError(f"{name} must have shape (N, T, D) or (N, T), got {paths.shape}")
    if paths.shape[0] < 1 or paths.shape[1] < 2 or paths.shape[2] < 1:
        raise ValueError(f"{name} must have non-empty N, at least two time steps, and non-empty D")
    return paths


def _as_ts(ts, *, expected_steps: int) -> np.ndarray:
    ts = _to_numpy(ts).astype(float, copy=False)
    if ts.ndim != 1:
        raise ValueError(f"ts must have shape (T,), got {ts.shape}")
    if ts.shape[0] != expected_steps:
        raise ValueError(f"ts has T={ts.shape[0]}, expected T={expected_steps}")
    return ts


def _save_or_return(fig, save_path):
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig


def _scientific_style():
    return plt.rc_context(
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
    )


def _format_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="0.88", linewidth=0.6)
    ax.set_axisbelow(True)


def _select_dimensions(paths: np.ndarray, dimensions: Iterable[int] | None) -> list[int]:
    if dimensions is None:
        return list(range(paths.shape[-1]))
    dims = [int(dim) for dim in dimensions]
    for dim in dims:
        if dim < 0 or dim >= paths.shape[-1]:
            raise ValueError(f"dimension index {dim} is outside [0, {paths.shape[-1] - 1}]")
    return dims


def plot_real_generated_paths(
    ts,
    real_paths,
    generated_paths,
    *,
    n_paths: int = 16,
    dimensions: Iterable[int] | None = None,
    align_initial: bool = True,
    real_label: str = "Real",
    generated_label: str = "Generated",
    save_path: str | Path | None = None,
):
    """
    Plot real and generated trajectories in a paper-ready style.

    If align_initial=True, each generated path is shifted so that it starts from
    the corresponding real initial point. If generated trajectories are identical
    across samples, only one generated curve is drawn.
    """

    real = _as_paths(real_paths, name="real_paths")
    generated = _as_paths(generated_paths, name="generated_paths")
    if real.shape[1:] != generated.shape[1:]:
        raise ValueError(
            "real_paths and generated_paths must have matching time and data dimensions; "
            f"got {real.shape[1:]} and {generated.shape[1:]}"
        )

    n = min(int(n_paths), real.shape[0], generated.shape[0])
    if n < 1:
        raise ValueError("n_paths must be >= 1")

    real = real[:n]
    generated = generated[:n].copy()
    if align_initial:
        generated = generated - generated[:, :1, :] + real[:, :1, :]

    ts = _as_ts(ts, expected_steps=real.shape[1])
    dims = _select_dimensions(real, dimensions)
    generated_is_constant = bool(np.allclose(generated, generated[:1], rtol=1e-8, atol=1e-10))

    with _scientific_style():
        fig, axes = plt.subplots(
            len(dims),
            1,
            figsize=(6.4, max(2.7, 2.3 * len(dims))),
            sharex=True,
            squeeze=False,
        )
        axes = axes[:, 0]
        real_color = "#1f77b4"
        generated_color = "#d62728"

        for ax, dim in zip(axes, dims):
            for idx in range(n):
                label = real_label if idx == 0 else None
                ax.plot(ts, real[idx, :, dim], color=real_color, alpha=0.34, linewidth=0.9, label=label)

            generated_count = 1 if generated_is_constant else n
            for idx in range(generated_count):
                label = generated_label if idx == 0 else None
                linewidth = 1.4 if generated_is_constant else 0.95
                alpha = 0.95 if generated_is_constant else 0.42
                ax.plot(
                    ts,
                    generated[idx, :, dim],
                    color=generated_color,
                    alpha=alpha,
                    linewidth=linewidth,
                    label=label,
                )

            ax.set_ylabel(r"$S_t$" if real.shape[-1] == 1 else rf"$S_t^{{({dim + 1})}}$")
            _format_axes(ax)
            ax.legend(frameon=False, loc="best")

        axes[-1].set_xlabel("Time")
        fig.tight_layout()
    return _save_or_return(fig, save_path)


def plot_generator_real_paths(
    generator,
    ts,
    real_paths,
    *,
    n_paths: int = 16,
    device: torch.device | str | None = None,
    dimensions: Iterable[int] | None = None,
    align_initial: bool = True,
    save_path: str | Path | None = None,
):
    """
    Sample generated paths from generator.sample_paths and compare them to real paths.
    """

    if device is None:
        try:
            device = next(generator.parameters()).device
        except StopIteration:
            device = "cpu"

    real = _as_paths(real_paths, name="real_paths")
    n = min(int(n_paths), real.shape[0])
    ts_tensor = torch.as_tensor(ts, device=device, dtype=torch.float32)
    y0 = torch.as_tensor(real[:n, 0, :], device=device, dtype=torch.float32)
    generator.eval()
    with torch.no_grad():
        generated = generator.sample_paths(ts_tensor, y0).detach().cpu()

    return plot_real_generated_paths(
        _to_numpy(ts),
        real[:n],
        generated,
        n_paths=n,
        dimensions=dimensions,
        align_initial=align_initial,
        save_path=save_path,
    )


def _history_array(history: dict, keys: tuple[str, ...]) -> np.ndarray:
    for key in keys:
        if key in history and len(history[key]) > 0:
            return np.asarray(history[key], dtype=float)
    return np.asarray([], dtype=float)


def _epoch_axis(history: dict, values: np.ndarray, *, preferred_key: str = "epoch") -> np.ndarray:
    epochs = _history_array(history, (preferred_key, "epoch"))
    if epochs.size == values.size:
        return epochs
    return np.arange(1, values.size + 1, dtype=float)


def plot_epoch_diagnostics(
    history: dict,
    *,
    option: PlotOption,
    save_path: str | Path | None = None,
):
    """
    Plot one diagnostic family over epochs.

    option="loss" plots generator and discriminator epoch losses.
    option="w1" plots average and maximum marginal Wasserstein-1 distances.
    option="e_sup" plots E sup_t ||S_t^real - S_t^generated||^2.
    """

    if option not in {"loss", "w1", "e_sup"}:
        raise ValueError("option must be one of: loss, w1, e_sup")

    with _scientific_style():
        fig, ax = plt.subplots(figsize=(6.4, 3.6))

        if option == "loss":
            loss_g = _history_array(history, ("loss_g_epoch", "loss_g"))
            loss_d = _history_array(history, ("loss_d_epoch", "loss_d"))
            if loss_g.size == 0 and loss_d.size == 0:
                raise ValueError("history does not contain loss values")
            if loss_d.size:
                ax.plot(_epoch_axis(history, loss_d), loss_d, color="#1f77b4", label="Discriminator")
            if loss_g.size:
                ax.plot(_epoch_axis(history, loss_g), loss_g, color="#d62728", label="Generator")
            ax.set_ylabel("Loss")
            ax.legend(frameon=False, loc="best")

        elif option == "w1":
            w1_avg = _history_array(history, ("marginal_w1_average", "w1_average", "average_w1"))
            w1_max = _history_array(history, ("marginal_w1_max", "w1_max", "max_w1"))
            if w1_avg.size == 0 and w1_max.size == 0:
                raise ValueError("history does not contain Wasserstein-1 values")
            if w1_avg.size:
                ax.plot(
                    _epoch_axis(history, w1_avg, preferred_key="metrics_epoch"),
                    w1_avg,
                    color="#1f77b4",
                    label="Average W1",
                )
            if w1_max.size:
                ax.plot(
                    _epoch_axis(history, w1_max, preferred_key="metrics_epoch"),
                    w1_max,
                    color="#d62728",
                    label="Maximum W1",
                )
            ax.set_ylabel("Wasserstein-1 distance")
            ax.legend(frameon=False, loc="best")

        else:
            e_sup = _history_array(
                history,
                ("expected_supremum_squared_error", "e_sup", "supremum_squared_error"),
            )
            if e_sup.size == 0:
                raise ValueError("history does not contain E sup values")
            ax.plot(
                _epoch_axis(history, e_sup, preferred_key="metrics_epoch"),
                e_sup,
                color="#1f77b4",
                label=r"$E \sup_t \|S_t^{real} - S_t^{generated}\|^2$",
            )
            ax.set_ylabel(r"$E \sup_t \|S_t^{real} - S_t^{generated}\|^2$")
            ax.legend(frameon=False, loc="best")

        ax.set_xlabel("Epoch")
        _format_axes(ax)
        fig.tight_layout()
    return _save_or_return(fig, save_path)


def plot_loss_history(history: dict, *, save_path: str | Path | None = None):
    return plot_epoch_diagnostics(history, option="loss", save_path=save_path)


def plot_w1_history(history: dict, *, save_path: str | Path | None = None):
    return plot_epoch_diagnostics(history, option="w1", save_path=save_path)


def plot_expected_sup_history(history: dict, *, save_path: str | Path | None = None):
    return plot_epoch_diagnostics(history, option="e_sup", save_path=save_path)
