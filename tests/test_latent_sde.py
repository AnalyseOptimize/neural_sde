from itertools import product

import pytest
import torch
import torchsde

from src.nets.latent_sde import ConditionalLatentSDE, LatentSDEConfig


def tiny_model(**kwargs):
    return ConditionalLatentSDE(1, latent_size=2, hidden_size=4, context_size=3, **kwargs)


@pytest.mark.parametrize('encoder_type,decoder_type,adjoint', list(product(
    ['deterministic', 'stochastic'], ['deterministic', 'stochastic'], [False, True],
)))
def test_loss_gradients_and_conditional_samples(encoder_type, decoder_type, adjoint):
    torch.manual_seed(4)
    model = tiny_model(encoder_type=encoder_type, decoder_type=decoder_type)
    # At the default zero residual initialization, GRU gradients start at zero.
    torch.nn.init.normal_(model.f_net[-1].weight, std=0.1)
    ts = torch.tensor([0.0, 0.04, 0.11, 0.2])
    paths = torch.randn(3, 4, 1) * 0.1 + 2
    log_pxs, kl = model(paths, ts, noise_std=0.2, adjoint=adjoint, dt=0.02)
    loss = -log_pxs + kl
    assert torch.isfinite(loss) and kl >= 0
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert model.encoder.gru.weight_ih_l0.grad.abs().sum() > 0
    assert model.raw_diffusion.grad.abs().sum() > 0
    for result in (
        model.sample_paths(ts, paths[:, 0], dt=0.02),
        model.reconstruct_paths(paths, ts, dt=0.02),
    ):
        assert result.shape == paths.shape
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result[:, 0], paths[:, 0], atol=0, rtol=0)


def test_path_kl_matches_girsanov_integral():
    model = tiny_model(encoder_type='deterministic')
    with torch.no_grad():
        model.h_net.weight.zero_()
        model.h_net.bias.zero_()
        model.f_net[-1].bias.fill_(0.15)
    ts = torch.linspace(0, 0.4, 5)
    paths = torch.ones(2, 5, 1)
    _, kl = model(paths, ts, dt=0.05)
    # Two latent coordinates, drift difference 0.15, diffusion 0.3.
    expected = 0.5 * 2 * (0.15 / 0.3)**2 * 0.4
    torch.testing.assert_close(kl, torch.tensor(expected), rtol=1e-5, atol=1e-6)
    z_prior, kl_prior = model._initial(paths[:, 0])
    z_posterior, kl0 = model._initial(paths[:, 0], torch.randn(2, 3))
    torch.testing.assert_close(z_prior, z_posterior)
    assert kl0 == kl_prior == 0


def test_gaussian_initial_kl_and_positive_diffusion():
    model = tiny_model()
    y0, ctx0 = torch.ones(3, 1), torch.randn(3, 3)
    _, actual = model._initial(y0, ctx0)
    q = model._normal(model.qz0_net(ctx0))
    p = model._normal(model.pz0_net(y0))
    expected = (torch.log(p.scale / q.scale) + (q.scale**2 + (q.loc - p.loc)**2) / (2 * p.scale**2) - 0.5).sum(-1).mean()
    torch.testing.assert_close(actual, expected)
    assert (model.g(torch.tensor(0.), torch.ones(3, 2)) >= model.diffusion_min).all()


def test_deterministic_readout_is_repeatable_with_fixed_brownian_motion():
    model = tiny_model(encoder_type='deterministic', decoder_type='deterministic')
    ts = torch.linspace(0, 0.2, 5)
    def sample():
        bm = torchsde.BrownianInterval(ts[0], ts[-1], size=(3, 2), entropy=123)
        return model.sample_paths(ts, torch.ones(3, 1), bm=bm, dt=0.05)
    first, second = sample(), sample()
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert first[:, -1].std() > 0  # Deterministic encoder/decoder still has SDE noise.


@torch.no_grad()
def test_stochastic_decoder_samples_learned_observation_distribution():
    torch.manual_seed(1)
    model = tiny_model(encoder_type='deterministic', decoder_type='stochastic')
    zs = torch.zeros(3, 10000, 2)
    y0 = torch.ones(10000, 1)
    means = model._decode_sample(zs, y0, False)
    draws = model._decode_sample(zs, y0, True)
    torch.testing.assert_close(draws[:, 0], y0)
    residual = draws[:, 1:] - means[:, 1:]
    assert abs(float(residual.mean())) < 0.002
    assert abs(float(residual.std()) - 0.05) < 0.002


def test_checkpoint_config_roundtrip(tmp_path):
    config = LatentSDEConfig(data_size=1, encoder_type='deterministic', decoder_type='stochastic')
    original = ConditionalLatentSDE.from_config(config)
    path = tmp_path / 'model.pt'
    torch.save({'model_config': original.config.__dict__, 'model_state_dict': original.state_dict()}, path)
    checkpoint = torch.load(path, weights_only=True)
    restored = ConditionalLatentSDE.from_config(LatentSDEConfig(**checkpoint['model_config']))
    restored.load_state_dict(checkpoint['model_state_dict'])
    assert restored.config == original.config


@pytest.mark.parametrize('kwargs', [
    {'encoder_type': 'invalid'}, {'decoder_type': 'invalid'},
    {'diffusion_min': 0}, {'decoder_std_init': 0}, {'latent_size': 0},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        ConditionalLatentSDE(data_size=1, **kwargs)
