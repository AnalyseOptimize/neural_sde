import pytest
import torch

from src.nets.sde_ml import SDEML
from src.simulators import OUSimulator


@pytest.mark.parametrize('noise_type', ['diagonal', 'general'])
@pytest.mark.parametrize('head', ['simple', 'constant', 'window'])
def test_vectorized_coefficients_match_stepwise_values_and_gradients(noise_type, head):
    torch.manual_seed(1)
    model = SDEML(data_size=2, noise_size=2, hidden_size=4, num_layers=1,
                  noise_type=noise_type, drift_head=head, diffusion_head=head,
                  time_origin=0.1, time_scale=2., state_center=[1., 2.], state_scale=[2., 3.])
    ts = torch.tensor([0.1, 0.13, 0.3, 0.5])
    paths = torch.randn(3, 4, 2)
    drift, diffusion = model.coefficients_on_path(ts, paths)
    sequential = [model.func.f_and_g(t, paths[:, i]) for i, t in enumerate(ts[:-1])]
    expected_drift = torch.stack([f for f, _ in sequential], dim=1)
    expected_diffusion = torch.stack([g for _, g in sequential], dim=1)
    torch.testing.assert_close(drift, expected_drift)
    torch.testing.assert_close(diffusion, expected_diffusion)
    params = tuple(model.func.parameters())
    actual_grad = torch.autograd.grad(drift.square().sum() + diffusion.square().sum(), params)
    expected_grad = torch.autograd.grad(expected_drift.square().sum() + expected_diffusion.square().sum(), params)
    for actual, expected in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_ou_starts_away_from_mean_and_relaxes_with_nonzero_variance():
    simulator = OUSimulator(seed=5)
    ts, paths = simulator.simulate(n_paths=4000, n_steps=201, dt=0.05)
    torch.testing.assert_close(paths[:, 0], torch.full((4000, 1), 2.))
    assert abs(float(paths[:, -1].mean()) - simulator.mu) < 0.02
    assert abs(float(paths[:, -1].var()) - simulator.sigma**2 / (2 * simulator.theta)) < 0.004
    drift, diffusion = simulator.f_and_g(ts[0], paths[:, 0])
    torch.testing.assert_close(drift, -2 * torch.ones_like(drift))
    torch.testing.assert_close(diffusion, 0.3 * torch.ones_like(diffusion))


def test_calibration_compares_effective_general_diffusion(tmp_path):
    from utils.calibration import save_ou_calibration

    simulator = OUSimulator(theta=0.0)
    ts, paths = simulator.simulate(n_paths=8, n_steps=5, dt=0.05)
    model = SDEML(data_size=1, noise_size=2, noise_type="general",
                  drift_head="constant", diffusion_head="constant",
                  diffusion_init=[0.18, -0.24])
    metrics = save_ou_calibration(model, simulator, ts, paths, tmp_path, n_grid=8)
    assert metrics['on_validation_paths']['drift_rmse'] == 0
    assert metrics['on_validation_paths']['diffusion_rmse'] < 1e-6
    assert metrics['full_rectangle_including_extrapolation']['diffusion_rmse'] < 1e-6
    assert (tmp_path / 'coefficient_grids.npz').is_file()
