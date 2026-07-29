"""Tests for the tDLGM module."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from next_state_predictor.tdlgm import (
    tDLGM,
    tDLGMCrossEntropy,
)

DEVICE = torch.device("cpu")

# Small model dimensions to keep tests fast.
INPUT_DIM = 4
HIDDEN_SIZE = 8
LATENT_DIM = 3
OUTPUT_DIM = 4
LAYERS = 2
SEQ_LEN = 3
BATCH = 10


@pytest.fixture()
def mse_model():
    return tDLGM(
        input_dim=INPUT_DIM,
        hidden_size=HIDDEN_SIZE,
        latent_dim=LATENT_DIM,
        output_dim=OUTPUT_DIM,
        layers=LAYERS,
        seq_len=SEQ_LEN,
        device=DEVICE,
    ).to(DEVICE)


@pytest.fixture()
def ce_model():
    return tDLGMCrossEntropy(
        input_dim=INPUT_DIM,
        hidden_size=HIDDEN_SIZE,
        latent_dim=LATENT_DIM,
        output_dim=OUTPUT_DIM,
        layers=LAYERS,
        seq_len=SEQ_LEN,
        device=DEVICE,
    ).to(DEVICE)


@pytest.fixture()
def mse_data():
    x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
    y = torch.randn(BATCH, 1, OUTPUT_DIM)
    x_1 = torch.cat((x, y), dim=1)[:, 1:, :]
    return x, x_1, y


@pytest.fixture()
def ce_data():
    x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
    y = torch.randint(0, OUTPUT_DIM, (BATCH, 1))
    y_one_hot = F.one_hot(y.squeeze(), num_classes=OUTPUT_DIM).float().unsqueeze(1)
    x_1 = torch.cat((x, y_one_hot), dim=1)[:, 1:, :]
    return x, x_1, y


# ── forward ───────────────────────────────────────────────────────────────────


def test_tdlgm_forward_shape(mse_model, mse_data):
    x, _x_1, _y = mse_data
    out = mse_model(x)
    assert out.shape == (BATCH, OUTPUT_DIM)


def test_tdlgm_ce_forward_shape(ce_model, ce_data):
    x, _x_1, _y = ce_data
    out = ce_model(x)
    assert out.shape == (BATCH, OUTPUT_DIM)


# ── get_loss / train_step ─────────────────────────────────────────────────────


def test_tdlgm_get_loss_returns_float(mse_model, mse_data):
    x, x_1, y = mse_data
    loss = mse_model.get_loss(x, x_1, y)
    assert isinstance(loss, float)
    assert loss >= 0.0


def test_tdlgm_ce_get_loss_returns_float(ce_model, ce_data):
    x, x_1, y = ce_data
    loss = ce_model.get_loss(x, x_1, y)
    assert isinstance(loss, float)
    assert loss >= 0.0


# ── loss decreases with training ──────────────────────────────────────────────


def test_tdlgm_loss_decreases(mse_model, mse_data):
    x, x_1, y = mse_data
    optimizer = torch.optim.Adam(mse_model.get_parameters(), lr=0.1)
    before = mse_model.get_loss(x, x_1, y)
    for _ in range(200):
        mse_model.train_step(x, x_1, y, optimizer)
    after = mse_model.get_loss(x, x_1, y)
    assert after < before, f"Loss did not decrease: {before:.4f} -> {after:.4f}"


def test_tdlgm_ce_loss_decreases(ce_model, ce_data):
    x, x_1, y = ce_data
    optimizer = torch.optim.Adam(ce_model.get_parameters(), lr=0.1)
    before = ce_model.get_loss(x, x_1, y)
    for _ in range(200):
        ce_model.train_step(x, x_1, y, optimizer)
    after = ce_model.get_loss(x, x_1, y)
    assert after < before, f"Loss did not decrease: {before:.4f} -> {after:.4f}"


# ── get_parameters ────────────────────────────────────────────────────────────


def test_get_parameters_non_empty(mse_model):
    params = list(mse_model.get_parameters())
    assert len(params) > 0
    assert all(isinstance(p, torch.nn.Parameter) for p in params)
