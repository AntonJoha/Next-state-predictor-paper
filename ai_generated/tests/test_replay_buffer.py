"""Tests for replay buffer persistence."""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest
from next_state_predictor.replay_buffer import ReplayBuffer


def test_save_to_sqlite_stores_state_trajectories(tmp_path):
    buffer = ReplayBuffer()
    for i in range(3):
        state = np.array([float(i)], dtype=np.float32)
        next_state = np.array([float(i + 1)], dtype=np.float32)
        buffer.push(state, 0, 1.0, next_state, False)

    db_path = tmp_path / "transitions.db"
    buffer.save_to_sqlite(str(db_path), trajectory_length=2)

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT state, state_trajectory FROM transitions ORDER BY id ASC"
        ).fetchall()

    states = [json.loads(row[0]) for row in rows]
    trajectories = [json.loads(row[1]) for row in rows]
    assert trajectories == [
        [[0.0], [0.0]],
        [[0.0], [1.0]],
        [[1.0], [2.0]],
    ]
    assert states == [[0.0], [1.0], [2.0]]


def test_save_to_sqlite_rejects_invalid_trajectory_length(tmp_path):
    buffer = ReplayBuffer()
    buffer.push(
        np.array([0.0], dtype=np.float32),
        0,
        1.0,
        np.array([1.0], dtype=np.float32),
        False,
    )
    db_path = tmp_path / "transitions.db"
    with pytest.raises(ValueError, match="trajectory_length must be >= 1"):
        buffer.save_to_sqlite(str(db_path), trajectory_length=0)


def test_save_to_sqlite_resets_trajectory_after_terminal_transition(tmp_path):
    buffer = ReplayBuffer()
    buffer.push(
        np.array([0.0], dtype=np.float32),
        0,
        1.0,
        np.array([1.0], dtype=np.float32),
        False,
    )
    buffer.push(
        np.array([1.0], dtype=np.float32),
        0,
        1.0,
        np.array([2.0], dtype=np.float32),
        True,
    )
    buffer.push(
        np.array([2.0], dtype=np.float32),
        0,
        1.0,
        np.array([3.0], dtype=np.float32),
        False,
    )

    db_path = tmp_path / "transitions.db"
    buffer.save_to_sqlite(str(db_path), trajectory_length=2)

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT state_trajectory FROM transitions ORDER BY id ASC"
        ).fetchall()

    trajectories = [json.loads(row[0]) for row in rows]
    assert trajectories == [
        [[0.0], [0.0]],
        [[0.0], [1.0]],
        [[2.0], [2.0]],
    ]
