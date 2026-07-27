"""Experience replay buffer with SQLite persistence."""

from __future__ import annotations

import json
import random
import sqlite3
from collections import deque
from typing import NamedTuple

import numpy as np


class Transition(NamedTuple):
    """A single environment transition."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-capacity circular buffer for storing experience transitions.

    Args:
        capacity: Maximum number of transitions to keep in memory.
    """

    def __init__(self, capacity: int = 10_000) -> None:
        self._buffer: deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition to the buffer."""
        self._buffer.append(
            Transition(
                np.array(state, dtype=np.float32),
                int(action),
                float(reward),
                np.array(next_state, dtype=np.float32),
                bool(done),
            )
        )

    def sample(self, batch_size: int) -> list[Transition]:
        """Sample *batch_size* transitions uniformly at random."""
        return random.sample(self._buffer, batch_size)

    def __len__(self) -> int:
        return len(self._buffer)

    # ------------------------------------------------------------------
    # SQLite persistence
    # ------------------------------------------------------------------

    def save_to_sqlite(self, db_path: str, table: str = "transitions") -> None:
        """Append all buffered transitions to a SQLite database.

        Args:
            db_path: Path to the SQLite database file (created if absent).
            table: Table name to write to (created if absent).
        """
        conn = sqlite3.connect(db_path)
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {table} (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                state       TEXT    NOT NULL,
                action      INTEGER NOT NULL,
                reward      REAL    NOT NULL,
                next_state  TEXT    NOT NULL,
                done        INTEGER NOT NULL
            )"""
        )
        conn.executemany(
            f"INSERT INTO {table} (state, action, reward, next_state, done)"
            " VALUES (?, ?, ?, ?, ?)",
            [
                (
                    json.dumps(t.state.tolist()),
                    t.action,
                    t.reward,
                    json.dumps(t.next_state.tolist()),
                    int(t.done),
                )
                for t in self._buffer
            ],
        )
        conn.commit()
        conn.close()

    def load_from_sqlite(self, db_path: str, table: str = "transitions") -> None:
        """Load transitions from a SQLite database into the buffer.

        Existing buffer contents are preserved; newly loaded rows are appended
        (subject to the capacity limit).

        Args:
            db_path: Path to the SQLite database file.
            table: Table name to read from.
        """
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            f"SELECT state, action, reward, next_state, done FROM {table}"
        ).fetchall()
        conn.close()
        for row in rows:
            self._buffer.append(
                Transition(
                    np.array(json.loads(row[0]), dtype=np.float32),
                    int(row[1]),
                    float(row[2]),
                    np.array(json.loads(row[3]), dtype=np.float32),
                    bool(row[4]),
                )
            )
