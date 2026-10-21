"""OU coefficient recovery in observation coordinates, on independent paths."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.visual import _scientific_style, _state_slice_range


@torch.no_grad()
def save_ou_calibration(model, simulator, ts, paths, output_dir, *, n_grid=128, n_levels=40):
    """Save function grids, support-aware errors, and contour maps.

    `paths` must be an independent, unnormalised OU sample. Grid errors include
    extrapolation; path errors only evaluate states visited by this sample.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figs").mkdir(exist_ok=True)
    parameter = next(model.parameters())
    ts = ts.to(parameter)
    paths = paths.to(parameter)
    was_training = model.training
    model.eval()
    time_grid = torch.linspace(float(ts[0]), float(ts[-1]), n_grid, device=ts.device, dtype=ts.dtype)
    state_low, state_high = _state_slice_range(paths.cpu().numpy(), 0)
    state_grid = torch.linspace(state_low, state_high, n_grid, device=ts.device, dtype=ts.dtype)
    states = state_grid[:, None]
    grid_values = [model.func.f_and_g(t, states) for t in time_grid]
    drift_grid = torch.stack([f[:, 0] for f, _ in grid_values], dim=1)
    # In one observed dimension the likelihood identifies ||g||, not its sign
    # or the individual columns of a general diffusion matrix.
    diffusion_grid = torch.stack([g.reshape(n_grid, -1).norm(dim=1) for _, g in grid_values], dim=1)
    truth_drift = (simulator.theta * (simulator.mu - states)).expand_as(drift_grid)
    truth_diffusion = torch.full_like(diffusion_grid, simulator.sigma)

    drift, diffusion = model.coefficients_on_path(ts, paths)
    diffusion = diffusion.norm(dim=-1) if diffusion.ndim == 4 else diffusion.abs()
    truth_path_drift = simulator.theta * (simulator.mu - paths[:, :-1])
    dt = float(ts[1] - ts[0])
    theta_euler = -math.expm1(-simulator.theta * dt) / dt
    sigma_euler = simulator.sigma * math.sqrt(
        -math.expm1(-2 * simulator.theta * dt) / (2 * simulator.theta * dt)
    ) if simulator.theta > 0 else simulator.sigma

    def rmse(value):
        return float(value.square().mean().sqrt())

    # Descriptive linear projection of the learned drift; the fitted model is an MLP.
    x = paths[:, :-1].reshape(-1).double()
    f = drift.reshape(-1).double()
    slope = ((x - x.mean()) * (f - f.mean())).mean() / x.var(unbiased=False).clamp_min(1e-12)
    intercept = f.mean() - slope * x.mean()
    metrics = {
        "n_validation_paths": paths.size(0),
        "true_parameters": {"theta": simulator.theta, "mu": simulator.mu, "sigma": simulator.sigma},
        "euler_likelihood_targets": {"theta": theta_euler, "mu": simulator.mu, "sigma": sigma_euler},
        "learned_linear_projection": {
            "theta": float(-slope),
            "mu": float(-intercept / slope) if abs(float(slope)) > 1e-10 else None,
            "mean_diffusion": float(diffusion.mean()),
        },
        "on_validation_paths": {
            "drift_rmse": rmse(drift - truth_path_drift),
            "diffusion_rmse": rmse(diffusion - simulator.sigma),
            "drift_rmse_to_euler_target": rmse(drift - theta_euler * (simulator.mu - paths[:, :-1])),
            "diffusion_rmse_to_euler_target": rmse(diffusion - sigma_euler),
        },
        "full_rectangle_including_extrapolation": {
            "state_min": state_low, "state_max": state_high,
            "drift_rmse": rmse(drift_grid - truth_drift),
            "diffusion_rmse": rmse(diffusion_grid - truth_diffusion),
        },
    }
    (output_dir / "coefficient_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    t, x = time_grid.cpu().numpy(), state_grid.cpu().numpy()
    arrays = {
        "time": t, "state": x,
        "drift_learned": drift_grid.cpu().numpy(), "drift_true": truth_drift.cpu().numpy(),
        "diffusion_learned": diffusion_grid.cpu().numpy(), "diffusion_true": truth_diffusion.cpu().numpy(),
        "support_time": ts.cpu().numpy(),
        "support_quantiles": paths[..., 0].quantile(torch.tensor([0.02, 0.98], device=ts.device, dtype=ts.dtype), dim=0).cpu().numpy(),
    }
    np.savez_compressed(output_dir / "coefficient_grids.npz", **arrays)
    with _scientific_style():
        fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), layout="constrained")
        for row, name in enumerate(("drift", "diffusion")):
            truth, learned = arrays[f"{name}_true"], arrays[f"{name}_learned"]
            low, high = min(truth.min(), learned.min()), max(truth.max(), learned.max())
            padding = max(1e-6, 0.01 * max(abs(low), abs(high)))
            levels = np.linspace(low - padding, high + padding, n_levels + 1)
            error = learned - truth
            limit = max(float(np.abs(error).max()), 1e-6)
            for col, (values, title) in enumerate(((truth, "True"), (learned, "Learned"), (error, "Learned - true"))):
                ax = axes[row, col]
                contour = ax.contourf(t, x, values, levels=levels if col < 2 else np.linspace(-limit, limit, n_levels + 1), cmap="viridis" if col < 2 else "RdBu_r")
                fig.colorbar(contour, ax=ax, label=name)
                for boundary in arrays["support_quantiles"]:
                    ax.plot(arrays["support_time"], boundary, "k--", linewidth=0.8)
                ax.set(xlabel="Time", ylabel="State", title=f"{name.capitalize()}: {title}", ylim=(state_low, state_high))
        fig.suptitle("OU coefficient recovery; dashed: 2-98% of independent paths")
        fig.savefig(output_dir / "figs" / "ou_coefficient_maps.pdf", bbox_inches="tight")
        fig.savefig(output_dir / "figs" / "ou_coefficient_maps.png", dpi=160)
        plt.close(fig)
    model.train(was_training)
    return metrics
