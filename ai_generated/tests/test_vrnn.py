"""Tests for the VRNN module."""

from __future__ import annotations

import pytest
import torch
from next_state_predictor.vrnn import VRNN

DEVICE = torch.device("cpu")

INPUT_DIM = 4
HIDDEN_SIZE = 8
LATENT_DIM = 3
OUTPUT_DIM = 4
SEQ_LEN = 5
BATCH = 6


@pytest.fixture()
def model():
    return VRNN(
        input_dim=INPUT_DIM,
        hidden_size=HIDDEN_SIZE,
        latent_dim=LATENT_DIM,
        output_dim=OUTPUT_DIM,
        device=DEVICE,
    ).to(DEVICE)


@pytest.fixture()
def data():
    x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
    y = torch.randn(BATCH, 1, OUTPUT_DIM)
    x_1 = torch.cat((x, y), dim=1)[:, 1:, :]
    return x, x_1, y


# ── forward ───────────────────────────────────────────────────────────────────


def test_vrnn_forward_shape(model, data):
    x, _x_1, _y = data
    out = model(x)
    assert out.shape == (BATCH, OUTPUT_DIM)


# ── get_loss / train_step ─────────────────────────────────────────────────────


def test_vrnn_get_loss_returns_float(model, data):
    x, x_1, y = data
    loss = model.get_loss(x, x_1, y)
    assert isinstance(loss, float)


def test_vrnn_train_step_returns_float(model, data):
    x, x_1, y = data
    optimizer = torch.optim.Adam(model.get_parameters(), lr=1e-3)
    loss = model.train_step(x, x_1, y, optimizer)
    assert isinstance(loss, float)


# ── loss decreases with training ──────────────────────────────────────────────


def test_vrnn_loss_decreases(model, data):
    x, x_1, y = data
    optimizer = torch.optim.Adam(model.get_parameters(), lr=0.05)
    before = model.get_loss(x, x_1, y)
    for _ in range(200):
        model.train_step(x, x_1, y, optimizer)
    after = model.get_loss(x, x_1, y)
    assert after < before, f"Loss did not decrease: {before:.4f} -> {after:.4f}"


# ── get_parameters ────────────────────────────────────────────────────────────


def test_vrnn_get_parameters_non_empty(model):
    params = list(model.get_parameters())
    assert len(params) > 0
    assert all(isinstance(p, torch.nn.Parameter) for p in params)
