"""Tests for TrajDiffusion (CPU, no GPU required)."""

import torch
import pytest

from mctd_maze.diffusion import TrajDiffusion, linear_beta_schedule


def test_beta_schedule_shape():
    betas = linear_beta_schedule(200)
    assert betas.shape == (200,)
    assert betas[0] < betas[-1]


def make_model(obs_dim=4, act_dim=2, small=True):
    kwargs = dict(
        hidden_dim=32, n_layers=2, n_heads=2, ffn_dim=64,
        n_diffusion_steps=20,
    ) if small else {}
    return TrajDiffusion(obs_dim=obs_dim, act_dim=act_dim, **kwargs)


def test_forward_shape():
    model = make_model()
    B, L, D = 2, 10, 6
    x = torch.randn(B, L, D)
    t = torch.randint(0, 20, (B,))
    out = model.model(x, t)
    assert out.shape == (B, L, D)


def test_loss_runs():
    model = make_model()
    x0 = torch.randn(4, 10, 6)
    loss = model.loss(x0)
    assert loss.item() > 0


def test_partial_denoise_shape():
    model = make_model()
    x = torch.randn(1, 10, 6)
    out = model.partial_denoise(x, t_start=15, n_steps=3)
    assert out.shape == x.shape


def test_jumpy_denoise_shape():
    model = make_model()
    x = torch.randn(1, 10, 6)
    out = model.jumpy_denoise(x, t_start=20, jumpy_interval=5)
    assert out.shape == x.shape
