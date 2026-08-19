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


def _gradient_colors(n: int, *, cmap_name: str = "viridis") -> list:
    if n < 1:
        return []
    if n == 1:
        values = np.asarray([0.5])
    else:
        values = np.linspace(0.12, 0.84, int(n))
    cmap = plt.get_cmap(cmap_name)
    return [cmap(float(value)) for value in values]


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


def plot_coupled_real_generated_paths(
    ts,
    real_paths,
    generated_paths,
    *,
    n_paths: int = 5,
    dimensions: Iterable[int] | None = None,
    real_label: str = "True process",
    generated_label: str = "Generator",
    save_path: str | Path | None = None,
):
    """
    Plot pathwise-coupled real and generated trajectories.

    Each pair is drawn in the same color: the real path is solid and the
    generated path is dashed.
    """

    real = _as_paths(real_paths, name="real_paths")
    generated = _as_paths(generated_paths, name="generated_paths")
    if real.shape[1:] != generated.shape[1:]:
        raise ValueError(
            "real_paths and generated_paths must have matching time and data dimensions; "
            f"got {real.shape[1:]} and {generated.shape[1:]}"
        )

    n = min(int(n_paths), real.shape[0], generated.shape[0], 5)
    if n < 1:
        raise ValueError("n_paths must be >= 1")

    real = real[:n]
    generated = generated[:n]
    ts = _as_ts(ts, expected_steps=real.shape[1])
    dims = _select_dimensions(real, dimensions)

    with _scientific_style():
        fig, axes = plt.subplots(
            len(dims),
            1,
            figsize=(6.4, max(2.7, 2.3 * len(dims))),
            sharex=True,
            squeeze=False,
        )
        axes = axes[:, 0]
        colors = _gradient_colors(n)

        for ax, dim in zip(axes, dims):
            for idx in range(n):
                ax.plot(
                    ts,
                    real[idx, :, dim],
                    color=colors[idx],
                    alpha=0.94,
                    linewidth=1.15,
                    linestyle="-",
                    label=real_label if idx == 0 else None,
                )
                ax.plot(
                    ts,
                    generated[idx, :, dim],
                    color=colors[idx],
                    alpha=0.94,
                    linewidth=1.15,
                    linestyle=(0, (4.0, 2.2)),
                    label=generated_label if idx == 0 else None,
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


def _history_matrix(history: dict, key: str) -> np.ndarray:
    if key not in history or len(history[key]) == 0:
        return np.asarray([], dtype=float).reshape(0, 0)
    values = np.asarray(history[key], dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError(f"history[{key!r}] must be one- or two-dimensional")
    return values


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
            loss_ml = _history_array(
                history,
                (
                    "negative_log_likelihood_per_step_epoch",
                    "negative_log_likelihood_per_step",
                    "negative_log_likelihood_epoch",
                    "loss_epoch",
                    "negative_log_likelihood",
                    "loss",
                ),
            )
            if loss_g.size == 0 and loss_d.size == 0 and loss_ml.size == 0:
                raise ValueError("history does not contain loss values")
            if loss_d.size:
                ax.plot(_epoch_axis(history, loss_d), loss_d, color="#1f77b4", label="Discriminator")
            if loss_g.size:
                ax.plot(_epoch_axis(history, loss_g), loss_g, color="#d62728", label="Generator")
            if loss_ml.size and loss_g.size == 0 and loss_d.size == 0:
                ax.plot(
                    _epoch_axis(history, loss_ml),
                    loss_ml,
                    color="#1f77b4",
                    label="Negative log likelihood",
                )
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


def plot_constant_coefficient_history(
    history: dict,
    *,
    coefficient: Literal["drift", "diffusion"],
    true_value=None,
    save_path: str | Path | None = None,
):
    """
    Plot constant drift or diffusion coefficient estimates over epochs.
    """

    if coefficient not in {"drift", "diffusion"}:
        raise ValueError("coefficient must be one of: drift, diffusion")

    values = _history_matrix(history, f"constant_{coefficient}_values")
    if values.size == 0:
        raise ValueError(f"history does not contain constant {coefficient} values")

    epochs = _history_array(history, (f"constant_{coefficient}_epoch", "epoch"))
    if epochs.size != values.shape[0]:
        epochs = np.arange(1, values.shape[0] + 1, dtype=float)

    target = None
    if true_value is not None:
        target = np.asarray(_to_numpy(true_value), dtype=float).reshape(-1)
        if target.size == 1 and values.shape[1] > 1:
            target = np.repeat(target, values.shape[1])

    with _scientific_style():
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        pretty = "Drift" if coefficient == "drift" else "Diffusion"
        for idx in range(values.shape[1]):
            color = colors[idx % len(colors)]
            label = pretty if values.shape[1] == 1 else f"{pretty} coefficient {idx + 1}"
            ax.plot(epochs, values[:, idx], color=color, label=label)
            if target is not None and idx < target.size:
                true_label = (
                    f"True {coefficient}"
                    if values.shape[1] == 1
                    else f"True {coefficient} {idx + 1}"
                )
                ax.axhline(
                    target[idx],
                    color=color,
                    linestyle="--",
                    linewidth=0.9,
                    alpha=0.75,
                    label=true_label,
                )

        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"{pretty} coefficient")
        _format_axes(ax)
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()
    return _save_or_return(fig, save_path)


def _diffusion_history_as_matrices(
    values: np.ndarray,
    *,
    matrix_shape: tuple[int, int],
    diagonal: bool,
) -> np.ndarray:
    rows, cols = matrix_shape
    if rows < 1 or cols < 1:
        raise ValueError("matrix_shape entries must be positive")

    if values.shape[1] == rows * cols:
        return values.reshape(values.shape[0], rows, cols)

    if diagonal and rows == cols and values.shape[1] == rows:
        matrices = np.zeros((values.shape[0], rows, cols), dtype=float)
        diag_idx = np.arange(rows)
        matrices[:, diag_idx, diag_idx] = values
        return matrices

    raise ValueError(
        "constant_diffusion_values cannot be reshaped into the requested matrix: "
        f"got {values.shape[1]} coefficients, expected {rows * cols}"
        + (f" or {rows} diagonal coefficients" if diagonal and rows == cols else "")
    )


def _target_as_matrix(
    true_value,
    *,
    matrix_shape: tuple[int, int],
    diagonal: bool,
) -> np.ndarray | None:
    if true_value is None:
        return None

    rows, cols = matrix_shape
    target = np.asarray(_to_numpy(true_value), dtype=float)
    if target.shape == (rows, cols):
        return target

    flat = target.reshape(-1)
    if flat.size == rows * cols:
        return flat.reshape(rows, cols)
    if diagonal and rows == cols and flat.size == rows:
        return np.diag(flat)
    if rows == cols and flat.size == 1:
        return np.eye(rows, dtype=float) * float(flat[0])

    raise ValueError(
        "true_value cannot be reshaped into the requested diffusion matrix: "
        f"got {flat.size} entries, expected {rows * cols}"
        + (f" or {rows} diagonal entries" if diagonal and rows == cols else "")
    )


def plot_constant_diffusion_matrix_history(
    history: dict,
    *,
    matrix_shape: tuple[int, int],
    diagonal: bool = False,
    true_value=None,
    save_path: str | Path | None = None,
):
    """
    Plot every entry of a constant diffusion matrix over epochs.
    """

    values = _history_matrix(history, "constant_diffusion_values")
    if values.size == 0:
        raise ValueError("history does not contain constant diffusion values")

    rows, cols = matrix_shape
    matrices = _diffusion_history_as_matrices(
        values,
        matrix_shape=(rows, cols),
        diagonal=diagonal,
    )
    target = _target_as_matrix(
        true_value,
        matrix_shape=(rows, cols),
        diagonal=diagonal,
    )

    epochs = _history_array(history, ("constant_diffusion_epoch", "epoch"))
    if epochs.size != matrices.shape[0]:
        epochs = np.arange(1, matrices.shape[0] + 1, dtype=float)

    with _scientific_style():
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(max(6.4, 2.15 * cols), max(3.6, 1.85 * rows)),
            sharex=True,
            squeeze=False,
        )
        learned_color = "#1f77b4"
        true_color = "#d62728"
        handles = []
        labels = []

        for row in range(rows):
            for col in range(cols):
                ax = axes[row, col]
                (learned_line,) = ax.plot(
                    epochs,
                    matrices[:, row, col],
                    color=learned_color,
                    label="Estimated",
                )
                if not handles:
                    handles.append(learned_line)
                    labels.append("Estimated")

                if target is not None:
                    true_line = ax.axhline(
                        target[row, col],
                        color=true_color,
                        linestyle="--",
                        linewidth=0.9,
                        alpha=0.8,
                        label="True",
                    )
                    if len(handles) == 1:
                        handles.append(true_line)
                        labels.append("True")

                ax.set_title(rf"$\sigma_{{{row + 1},{col + 1}}}$")
                if row == rows - 1:
                    ax.set_xlabel("Epoch")
                if col == 0:
                    ax.set_ylabel("Diffusion coefficient")
                _format_axes(ax)

        if handles:
            fig.legend(handles, labels, frameon=False, loc="upper center", ncol=len(handles))
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return _save_or_return(fig, save_path)


def _target_as_covariance(
    true_value,
    *,
    diffusion_shape: tuple[int, int],
    diagonal: bool,
    true_value_kind: Literal["diffusion", "covariance"],
) -> np.ndarray | None:
    if true_value is None:
        return None
    if true_value_kind not in {"diffusion", "covariance"}:
        raise ValueError("true_value_kind must be one of: diffusion, covariance")

    rows, cols = diffusion_shape
    target = np.asarray(_to_numpy(true_value), dtype=float)
    flat = target.reshape(-1)

    if true_value_kind == "covariance":
        if target.shape == (rows, rows):
            return target
        if flat.size == rows * rows:
            return flat.reshape(rows, rows)
        if diagonal and flat.size == rows:
            return np.diag(flat)
        if flat.size == 1:
            return np.eye(rows, dtype=float) * float(flat[0])
        raise ValueError(
            "true_value cannot be interpreted as a covariance target: "
            f"got {flat.size} entries, expected {rows * rows}"
            + (f", or {rows} diagonal covariance entries" if diagonal else "")
        )

    if target.shape == (rows, cols):
        return target @ target.T
    if flat.size == rows * cols:
        sigma = flat.reshape(rows, cols)
        return sigma @ sigma.T
    if diagonal and flat.size == rows:
        return np.diag(flat**2)
    if flat.size == 1:
        return np.eye(rows, dtype=float) * float(flat[0]) ** 2

    raise ValueError(
        "true_value cannot be interpreted as a covariance target: "
        f"got {flat.size} entries, expected {rows * rows}, {rows * cols}"
        + (f", or {rows} diagonal coefficients" if diagonal else "")
    )


def plot_constant_diffusion_covariance_history(
    history: dict,
    *,
    diffusion_shape: tuple[int, int],
    diagonal: bool = False,
    true_value=None,
    true_value_kind: Literal["diffusion", "covariance"] = "diffusion",
    save_path: str | Path | None = None,
):
    """
    Plot every entry of Sigma Sigma^T for a constant diffusion coefficient.
    """

    values = _history_matrix(history, "constant_diffusion_values")
    if values.size == 0:
        raise ValueError("history does not contain constant diffusion values")

    rows, cols = diffusion_shape
    sigma = _diffusion_history_as_matrices(
        values,
        matrix_shape=(rows, cols),
        diagonal=diagonal,
    )
    covariance = sigma @ np.swapaxes(sigma, 1, 2)
    target = _target_as_covariance(
        true_value,
        diffusion_shape=(rows, cols),
        diagonal=diagonal,
        true_value_kind=true_value_kind,
    )

    epochs = _history_array(history, ("constant_diffusion_epoch", "epoch"))
    if epochs.size != covariance.shape[0]:
        epochs = np.arange(1, covariance.shape[0] + 1, dtype=float)

    with _scientific_style():
        fig, axes = plt.subplots(
            rows,
            rows,
            figsize=(max(6.4, 2.15 * rows), max(3.6, 1.85 * rows)),
            sharex=True,
            squeeze=False,
        )
        learned_color = "#1f77b4"
        true_color = "#d62728"
        handles = []
        labels = []

        for row in range(rows):
            for col in range(rows):
                ax = axes[row, col]
                (learned_line,) = ax.plot(
                    epochs,
                    covariance[:, row, col],
                    color=learned_color,
                    label="Estimated",
                )
                if not handles:
                    handles.append(learned_line)
                    labels.append("Estimated")

                if target is not None:
                    true_line = ax.axhline(
                        target[row, col],
                        color=true_color,
                        linestyle="--",
                        linewidth=0.9,
                        alpha=0.8,
                        label="True",
                    )
                    if len(handles) == 1:
                        handles.append(true_line)
                        labels.append("True")

                ax.set_title(rf"$(\Sigma \Sigma^\top)_{{{row + 1},{col + 1}}}$")
                if row == rows - 1:
                    ax.set_xlabel("Epoch")
                if col == 0:
                    ax.set_ylabel("Noise covariance")
                _format_axes(ax)

        if handles:
            fig.legend(handles, labels, frameon=False, loc="upper center", ncol=len(handles))
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return _save_or_return(fig, save_path)


def _parameter_device_dtype(model) -> tuple[torch.device, torch.dtype]:
    parameter = next(iter(model.parameters()), None)
    if parameter is None:
        return torch.device("cpu"), torch.float32
    return parameter.device, parameter.dtype


def _state_slice_range(real_paths: np.ndarray, state_dimension: int) -> tuple[float, float]:
    values = real_paths[..., state_dimension].reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -1.0, 1.0

    low, high = np.quantile(values, [0.02, 0.98])
    if not np.isfinite(low) or not np.isfinite(high):
        return -1.0, 1.0
    if high <= low + 1e-12:
        center = float(0.5 * (low + high))
        radius = max(0.5, 0.2 * abs(center))
        return center - radius, center + radius
    padding = 0.05 * (high - low)
    return float(low - padding), float(high + padding)


def _baseline_state(real_paths: np.ndarray) -> np.ndarray:
    flat = real_paths.reshape(-1, real_paths.shape[-1])
    baseline = np.nanmedian(flat, axis=0)
    baseline = np.where(np.isfinite(baseline), baseline, 0.0)
    return baseline.astype(float, copy=False)


def _select_coefficient_component(
    values: torch.Tensor,
    *,
    coefficient: Literal["drift", "diffusion"],
    component: int | tuple[int, int],
) -> torch.Tensor:
    if coefficient == "drift":
        if values.ndim != 2:
            raise ValueError(f"drift values must have shape (B, D), got {tuple(values.shape)}")
        if isinstance(component, tuple):
            component = component[0]
        return values[:, int(component)]

    if values.ndim == 2:
        if isinstance(component, tuple):
            component = component[0]
        return values[:, int(component)]
    if values.ndim == 3:
        if isinstance(component, tuple):
            row, col = component
        else:
            row = int(component) // values.size(2)
            col = int(component) % values.size(2)
        return values[:, int(row), int(col)]
    raise ValueError(f"diffusion values must have shape (B, D) or (B, D, M), got {tuple(values.shape)}")


def _evaluate_coefficient_component(
    model,
    coefficient: Literal["drift", "diffusion"],
    *,
    t: torch.Tensor,
    states: torch.Tensor,
    component: int | tuple[int, int],
) -> torch.Tensor:
    func = getattr(model, "func", None)
    if func is None:
        raise ValueError("model must expose a .func object with SDE coefficients")

    drift, diffusion = func.f_and_g(t, states)
    values = drift if coefficient == "drift" else diffusion
    return _select_coefficient_component(values, coefficient=coefficient, component=component)


def _component_label(
    *,
    coefficient: Literal["drift", "diffusion"],
    component: int | tuple[int, int],
) -> str:
    if coefficient == "drift":
        if isinstance(component, tuple):
            component = component[0]
        return rf"$\mu_{{{int(component) + 1}}}$"
    if isinstance(component, tuple):
        row, col = component
    else:
        row, col = int(component), None
    if col is None:
        return rf"$\sigma_{{{row + 1}}}$"
    return rf"$\sigma_{{{row + 1},{col + 1}}}$"


def plot_simple_coefficient_slices(
    model,
    ts,
    real_paths,
    *,
    coefficient: Literal["drift", "diffusion"],
    component: int | tuple[int, int] = 0,
    state_dimension: int = 0,
    n_grid: int = 128,
    n_levels: int = 5,
    save_path: str | Path | None = None,
):
    """
    Plot one-dimensional slices of a simple coefficient head.

    The left panel shows coefficient(t, S) as a function of t for several fixed
    state levels. The right panel shows coefficient(t, S) as a function of S for
    several fixed time levels.
    """

    if coefficient not in {"drift", "diffusion"}:
        raise ValueError("coefficient must be one of: drift, diffusion")
    if n_grid < 2:
        raise ValueError("n_grid must be >= 2")
    if n_levels < 1:
        raise ValueError("n_levels must be >= 1")

    paths = _as_paths(real_paths, name="real_paths")
    if state_dimension < 0 or state_dimension >= paths.shape[-1]:
        raise ValueError(
            f"state_dimension must be in [0, {paths.shape[-1] - 1}], got {state_dimension}"
        )
    ts_np = _as_ts(ts, expected_steps=paths.shape[1])
    t_grid_np = np.linspace(float(ts_np[0]), float(ts_np[-1]), int(n_grid))
    state_low, state_high = _state_slice_range(paths, int(state_dimension))
    state_grid_np = np.linspace(state_low, state_high, int(n_grid))
    state_levels_np = np.linspace(state_low, state_high, int(n_levels))
    time_levels_np = np.linspace(float(ts_np[0]), float(ts_np[-1]), int(n_levels))
    baseline_np = _baseline_state(paths)

    device, dtype = _parameter_device_dtype(model)
    baseline = torch.as_tensor(baseline_np, device=device, dtype=dtype)
    t_grid = torch.as_tensor(t_grid_np, device=device, dtype=dtype)
    state_grid = torch.as_tensor(state_grid_np, device=device, dtype=dtype)

    time_slices = []
    state_slices = []
    model.eval()
    with torch.no_grad():
        for state_value in state_levels_np:
            states = baseline.view(1, -1).expand(t_grid.numel(), -1).clone()
            states[:, int(state_dimension)] = float(state_value)
            values = []
            for idx in range(t_grid.numel()):
                value = _evaluate_coefficient_component(
                    model,
                    coefficient,
                    t=t_grid[idx],
                    states=states[idx : idx + 1],
                    component=component,
                )
                values.append(value.squeeze(0))
            time_slices.append(torch.stack(values).detach().cpu().numpy())

        for time_value in time_levels_np:
            states = baseline.view(1, -1).expand(state_grid.numel(), -1).clone()
            states[:, int(state_dimension)] = state_grid
            values = _evaluate_coefficient_component(
                model,
                coefficient,
                t=torch.as_tensor(float(time_value), device=device, dtype=dtype),
                states=states,
                component=component,
            )
            state_slices.append(values.detach().cpu().numpy())

    with _scientific_style():
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), squeeze=False)
        ax_t, ax_s = axes[0]
        time_colors = _gradient_colors(len(time_slices))
        state_colors = _gradient_colors(len(state_slices))
        pretty = "Drift" if coefficient == "drift" else "Diffusion"
        component_text = _component_label(coefficient=coefficient, component=component)

        for idx, values in enumerate(time_slices):
            ax_t.plot(
                t_grid_np,
                values,
                color=time_colors[idx],
                label=rf"$S_t={state_levels_np[idx]:.3g}$",
            )
        ax_t.set_xlabel("Time")
        ax_t.set_ylabel(f"{pretty} coefficient")
        ax_t.set_title(rf"{component_text} as a function of $t$")
        _format_axes(ax_t)
        ax_t.legend(frameon=False, loc="best")

        for idx, values in enumerate(state_slices):
            ax_s.plot(
                state_grid_np,
                values,
                color=state_colors[idx],
                label=rf"$t={time_levels_np[idx]:.3g}$",
            )
        ax_s.set_xlabel(r"$S_t$")
        ax_s.set_ylabel(f"{pretty} coefficient")
        ax_s.set_title(rf"{component_text} as a function of $S_t$")
        _format_axes(ax_s)
        ax_s.legend(frameon=False, loc="best")

        fig.tight_layout()
    return _save_or_return(fig, save_path)


def simple_coefficient_components(model, coefficient: Literal["drift", "diffusion"]) -> list[int | tuple[int, int]]:
    """Return component identifiers for plotting simple coefficient heads."""

    if coefficient not in {"drift", "diffusion"}:
        raise ValueError("coefficient must be one of: drift, diffusion")

    data_size = int(getattr(model, "data_size"))
    if coefficient == "drift":
        return list(range(data_size))

    noise_type = str(getattr(model, "noise_type", "diagonal")).lower()
    if noise_type == "general":
        noise_size = int(getattr(model, "noise_size"))
        return [(row, col) for row in range(data_size) for col in range(noise_size)]
    return list(range(data_size))


def plot_loss_history(history: dict, *, save_path: str | Path | None = None):
    return plot_epoch_diagnostics(history, option="loss", save_path=save_path)


def plot_w1_history(history: dict, *, save_path: str | Path | None = None):
    return plot_epoch_diagnostics(history, option="w1", save_path=save_path)


def plot_expected_sup_history(history: dict, *, save_path: str | Path | None = None):
    return plot_epoch_diagnostics(history, option="e_sup", save_path=save_path)
