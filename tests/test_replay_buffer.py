"""Tests for replay buffer persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from next_state_predictor.replay_buffer import ReplayBuffer, Transition


def _assert_transition_equal(left: Transition, right: Transition) -> None:
    assert np.array_equal(left.observation, right.observation)
    assert left.action == right.action
    assert left.reward == right.reward
    assert np.array_equal(left.next_observation, right.next_observation)
    assert left.terminated == right.terminated
    assert left.truncated == right.truncated
    assert left.info == right.info


def test_replay_buffer_save_and_load_sqlite_round_trip(tmp_path: Path):
    buffer = ReplayBuffer(capacity=4)
    buffer.add(
        observation=np.array([1.0, 2.0], dtype=np.float32),
        action={"move": "left", "values": (1, 2)},
        reward=1.5,
        next_observation=np.array([3.0, 4.0], dtype=np.float32),
        terminated=False,
        truncated=False,
        info={"episode": 7, "tags": ["train", "demo"]},
    )
    buffer.add(
        observation={"position": (1, 2)},
        action=1,
        reward=2.5,
        next_observation={"position": (3, 4)},
        terminated=True,
        truncated=False,
        info=None,
    )

    db_path = tmp_path / "replay_buffer.sqlite"
    buffer.save_to_sqlite(db_path)

    loaded = ReplayBuffer.load_from_sqlite(db_path)

    assert loaded.capacity == buffer.capacity
    assert len(loaded) == len(buffer)
    for expected, actual in zip(buffer.transitions, loaded.transitions, strict=True):
        _assert_transition_equal(expected, actual)


def test_replay_buffer_sample_returns_transitions():
    buffer = ReplayBuffer(capacity=3)
    buffer.add("obs-1", 1, 1.0, "obs-2", False, False)
    buffer.add("obs-3", 2, 2.0, "obs-4", False, False)
    buffer.add("obs-5", 3, 3.0, "obs-6", True, False)

    sample = buffer.sample(2, seed=0)

    assert len(sample) == 2
    assert all(isinstance(item, Transition) for item in sample)
