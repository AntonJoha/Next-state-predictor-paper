"""Tests for scripts/evaluate_models.py helpers."""

from __future__ import annotations

import numpy as np

from scripts.evaluate_models import (
    _select_test_episode_indices,
    prepare_datasets,
    split_episode_data,
)

EPISODE_ID_DIVISOR = 10


def test_split_episode_data_keeps_episodes_intact():
    episode_lengths = [2, 2, 2, 2]
    states = np.array([[0], [1], [10], [11], [20], [21], [30], [31]], dtype=np.float32)
    next_states = states + 0.5

    train_states, train_next_states, train_lengths, test_states, test_next_states, test_lengths = split_episode_data(
        states,
        next_states,
        episode_lengths,
        test_split=0.5,
        seed=0,
    )

    assert sorted(train_lengths + test_lengths) == sorted(episode_lengths)
    assert train_states.shape == train_next_states.shape
    assert test_states.shape == test_next_states.shape

    def episode_ids(values: np.ndarray) -> list[int]:
        return [int(v[0] // EPISODE_ID_DIVISOR) for v in values]

    def split_ids(values: np.ndarray, lengths: list[int]) -> list[int]:
        ids: list[int] = []
        start = 0
        for length in lengths:
            chunk = values[start : start + length]
            ids.append(episode_ids(chunk)[0])
            assert len(set(episode_ids(chunk))) == 1
            start += length
        return ids

    train_ids = split_ids(train_states, train_lengths)
    test_ids = split_ids(test_states, test_lengths)
    assert set(train_ids).isdisjoint(test_ids)


def test_prepare_datasets_uses_training_only_normalization():
    episode_lengths = [2, 2, 2, 2]
    seed = 0
    test_split = 0.5
    n_episodes = len(episode_lengths)
    test_episode_indices = _select_test_episode_indices(n_episodes, test_split, seed)

    states = []
    next_states = []
    for episode_index in range(n_episodes):
        base = 100.0 if episode_index in test_episode_indices else 0.0
        states.extend([[base], [base + 1.0]])
        next_states.extend([[base + 0.5], [base + 1.5]])

    x_train, x1_train, y_train, x_test, x1_test, y_test = prepare_datasets(
        np.array(states, dtype=np.float32),
        np.array(next_states, dtype=np.float32),
        episode_lengths,
        seq_len=1,
        test_split=test_split,
        seed=seed,
    )

    assert x_train.min() >= 0.0
    assert x_train.max() <= 1.0
    assert x1_train.min() >= 0.0
    assert y_train.min() >= 0.0
    assert x_test.max() > 1.0
    assert x1_test.max() > 1.0
    assert y_test.max() > 1.0
