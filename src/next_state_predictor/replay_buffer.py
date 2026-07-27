"""Experience replay buffer with SQLite persistence."""

from __future__ import annotations

import json
import re
import random
import sqlite3
from collections import deque
from typing import NamedTuple

import numpy as np


def _validate_table_name(table: str) -> str:
    """Raise ValueError if *table* is not a safe SQL identifier."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
        msg = (
            f"Invalid table name {table!r}. "
            "Table names must start with a letter or underscore and contain "
            "only letters, digits, and underscores."
        )
        raise ValueError(msg)
    return table


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

    def save_to_sqlite(
        self,
        db_path: str,
        table: str = "transitions",
        trajectory_length: int = 1,
    ) -> None:
        """Append all buffered transitions to a SQLite database.

        Args:
            db_path: Path to the SQLite database file (created if absent).
            table: Table name to write to (created if absent).
            trajectory_length: Number of recent states to store per transition.
        """
        if trajectory_length < 1:
            msg = "trajectory_length must be >= 1"
            raise ValueError(msg)
        _validate_table_name(table)
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                f"""CREATE TABLE IF NOT EXISTS {table} (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    state       TEXT    NOT NULL,
                    state_trajectory TEXT NOT NULL,
                    action      INTEGER NOT NULL,
                    reward      REAL    NOT NULL,
                    next_state  TEXT    NOT NULL,
                    done        INTEGER NOT NULL
                )"""
            )
            original_row_factory = conn.row_factory
            conn.row_factory = sqlite3.Row
            columns = {
                row["name"]
                for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            conn.row_factory = original_row_factory
            if "state_trajectory" not in columns:
                # Existing rows in migrated tables keep this default until
                # they are rewritten by a future save.
                conn.execute(
                    f"ALTER TABLE {table} "
                    "ADD COLUMN state_trajectory TEXT DEFAULT '[]'"
                )

            state_window: deque[np.ndarray] = deque(maxlen=trajectory_length)
            insert_rows = []
            for transition in self._buffer:
                state_window.append(transition.state)
                # For early transitions, pad with the first available state so
                # every stored trajectory has a fixed length. At the start of
                # an episode that first state is the current transition state.
                padded_window = [transition.state] * (
                    trajectory_length - len(state_window)
                ) + list(state_window)
                insert_rows.append(
                    (
                        json.dumps(transition.state.tolist()),
                        json.dumps([state.tolist() for state in padded_window]),
                        transition.action,
                        transition.reward,
                        json.dumps(transition.next_state.tolist()),
                        int(transition.done),
                    )
                )
                if transition.done:
                    state_window.clear()

            conn.executemany(
                f"INSERT INTO {table} "
                "(state, state_trajectory, action, reward, next_state, done)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                insert_rows,
            )

    def load_from_sqlite(self, db_path: str, table: str = "transitions") -> None:
        """Load transitions from a SQLite database into the buffer.

        Existing buffer contents are preserved; newly loaded rows are appended
        (subject to the capacity limit).

        Args:
            db_path: Path to the SQLite database file.
            table: Table name to read from.
        """
        _validate_table_name(table)
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                f"SELECT state, action, reward, next_state, done FROM {table}"
            ).fetchall()
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
