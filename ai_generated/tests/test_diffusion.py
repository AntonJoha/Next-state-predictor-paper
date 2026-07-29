"""Tests for the DiffusionPredictor module."""

from __future__ import annotations

import pytest
import torch
from next_state_predictor.diffusion import DiffusionPredictor

DEVICE = torch.device("cpu")

INPUT_DIM = 4
HIDDEN_SIZE = 8
OUTPUT_DIM = 4
SEQ_LEN = 5
BATCH = 6
T = 20  # small T for fast tests


@pytest.fixture()
def model():
    return DiffusionPredictor(
        input_dim=INPUT_DIM,
        hidden_size=HIDDEN_SIZE,
        output_dim=OUTPUT_DIM,
        T=T,
        t_emb_dim=8,
        device=DEVICE,
    ).to(DEVICE)


@pytest.fixture()
def data():
    x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
    y = torch.randn(BATCH, 1, OUTPUT_DIM)
    x_1 = torch.cat((x, y), dim=1)[:, 1:, :]
    return x, x_1, y


# ── constructor guard ─────────────────────────────────────────────────────────


def test_diffusion_odd_t_emb_dim_raises():
    with pytest.raises(ValueError, match="t_emb_dim must be even for sinusoidal embedding"):
        DiffusionPredictor(t_emb_dim=7)


# ── forward ───────────────────────────────────────────────────────────────────


def test_diffusion_forward_shape(model, data):
    x, _x_1, _y = data
    out = model(x)
    assert out.shape == (BATCH, OUTPUT_DIM)


# ── get_loss / train_step ─────────────────────────────────────────────────────


def test_diffusion_get_loss_returns_float(model, data):
    x, x_1, y = data
    loss = model.get_loss(x, x_1, y)
    assert isinstance(loss, float)
    assert loss >= 0.0


def test_diffusion_train_step_returns_float(model, data):
    x, x_1, y = data
    optimizer = torch.optim.Adam(model.get_parameters(), lr=1e-3)
    loss = model.train_step(x, x_1, y, optimizer)
    assert isinstance(loss, float)


# ── loss decreases with training ──────────────────────────────────────────────


def test_diffusion_loss_decreases(model, data):
    # Average several stochastic loss evaluations to reduce variance from the
    # random timestep sampled inside get_loss / _ddpm_loss.
    x, x_1, y = data
    n_samples = 20
    optimizer = torch.optim.Adam(model.get_parameters(), lr=1e-2)
    before = sum(model.get_loss(x, x_1, y) for _ in range(n_samples)) / n_samples
    for _ in range(400):
        model.train_step(x, x_1, y, optimizer)
    after = sum(model.get_loss(x, x_1, y) for _ in range(n_samples)) / n_samples
    assert after < before, f"Loss did not decrease: {before:.4f} -> {after:.4f}"


# ── get_parameters ────────────────────────────────────────────────────────────


def test_diffusion_get_parameters_non_empty(model):
    params = list(model.get_parameters())
    assert len(params) > 0
    assert all(isinstance(p, torch.nn.Parameter) for p in params)
