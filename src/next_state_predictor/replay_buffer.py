"""Replay buffer storage and SQLite persistence helpers."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
import json
import random
import sqlite3
import re
from typing import Any

import numpy as np


_TYPE_KEY = "__next_state_predictor_type__"
_METADATA_TABLE_NAME = "__next_state_predictor_replay_buffer_metadata"
_TABLE_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(slots=True)
class Transition:
    """A single replay buffer transition."""

    observation: Any
    action: Any
    reward: float
    next_observation: Any
    terminated: bool
    truncated: bool
    info: dict[str, Any] | None = None


def _validate_table_name(table_name: str) -> None:
    if not _TABLE_NAME_PATTERN.fullmatch(table_name):
        raise ValueError(
            "table_name must be a non-empty SQLite identifier made of letters, "
            "digits, and underscores, and must start with a letter or underscore"
        )
    if table_name.startswith("__next_state_predictor_"):
        raise ValueError("table_name must not use the reserved __next_state_predictor_ prefix")


def _encode_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            _TYPE_KEY: "ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": value.tolist(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return {_TYPE_KEY: "tuple", "items": [_encode_value(item) for item in value]}
    if isinstance(value, list):
        return [_encode_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _encode_value(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported value type for SQLite serialization: {type(value)!r}")


def _decode_value(value: Any) -> Any:
    if isinstance(value, dict):
        value_type = value.get(_TYPE_KEY)
        if value_type == "ndarray":
            try:
                dtype = np.dtype(value["dtype"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid dtype stored in replay buffer: {value['dtype']!r}") from exc

            raw_array = np.array(value["data"])
            stored_shape = tuple(int(dim) for dim in value["shape"])
            expected_size = 1
            for dimension in stored_shape:
                expected_size *= dimension
            if raw_array.size != expected_size:
                raise ValueError(
                    "Stored replay buffer array element count does not match the serialized data"
                )
            return raw_array.astype(dtype, copy=False).reshape(stored_shape)
        if value_type == "tuple":
            return tuple(_decode_value(item) for item in value["items"])
        return {key: _decode_value(item) for key, item in value.items() if key != _TYPE_KEY}
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    return value


def _dumps(value: Any) -> str:
    return json.dumps(_encode_value(value), separators=(",", ":"))


def _loads(value: str | None) -> Any:
    if value is None:
        return None
    return _decode_value(json.loads(value))


def _load_reward(value: str | None) -> float:
    loaded = _loads(value)
    try:
        return float(loaded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid reward stored in replay buffer: {loaded!r}") from exc


def _load_bool(value: str | None, field_name: str) -> bool:
    loaded = _loads(value)
    if isinstance(loaded, bool):
        return loaded
    if isinstance(loaded, int) and loaded in {0, 1}:
        return bool(loaded)
    if isinstance(loaded, str):
        normalized = loaded.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    raise ValueError(f"Invalid {field_name} stored in replay buffer: {loaded!r}")


class ReplayBuffer:
    """A finite replay buffer with SQLite persistence support."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self._buffer: deque[Transition] = deque(maxlen=capacity)

    def add(
        self,
        observation: Any,
        action: Any,
        reward: float,
        next_observation: Any,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any] | None = None,
    ) -> None:
        """Append a transition to the buffer."""
        self._buffer.append(
            Transition(
                observation=observation,
                action=action,
                reward=float(reward),
                next_observation=next_observation,
                terminated=bool(terminated),
                truncated=bool(truncated),
                info=info,
            )
        )

    def extend(self, transitions: Iterable[Transition]) -> None:
        """Append multiple transitions to the buffer."""
        for transition in transitions:
            self._buffer.append(transition)

    def sample(self, batch_size: int, seed: int | None = None) -> list[Transition]:
        """Sample a batch of transitions without replacement."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self._buffer):
            raise ValueError(
                f"batch_size ({batch_size}) cannot exceed the number of stored transitions "
                f"({len(self._buffer)})"
            )

        rng = random.Random(seed)
        return rng.sample(self._buffer, batch_size)

    def clear(self) -> None:
        """Remove all transitions from the buffer."""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)

    def __iter__(self) -> Iterator[Transition]:
        return iter(self._buffer)

    @property
    def transitions(self) -> tuple[Transition, ...]:
        """Return the stored transitions as an immutable snapshot."""
        return tuple(self._buffer)

    def save_to_sqlite(self, db_path: str | Path, table_name: str = "replay_buffer") -> None:
        """Persist the buffer to a SQLite database.

        Args:
            db_path: SQLite database path.
            table_name: Name of the table that will store the transitions.
        """
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _validate_table_name(table_name)

        table = table_name
        metadata_table = _METADATA_TABLE_NAME

        with sqlite3.connect(path) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {metadata_table} (
                    buffer_name TEXT PRIMARY KEY,
                    capacity INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    position INTEGER PRIMARY KEY,
                    observation TEXT NOT NULL,
                    action TEXT NOT NULL,
                    reward TEXT NOT NULL,
                    next_observation TEXT NOT NULL,
                    terminated TEXT NOT NULL,
                    truncated TEXT NOT NULL,
                    info TEXT
                )
                """
            )
            connection.execute(f"DELETE FROM {metadata_table} WHERE buffer_name = ?", (table_name,))
            connection.execute(f"DELETE FROM {table}")
            connection.execute(
                f"INSERT INTO {metadata_table} (buffer_name, capacity) VALUES (?, ?)",
                (table_name, self.capacity),
            )

            rows = [
                (
                    index,
                    _dumps(transition.observation),
                    _dumps(transition.action),
                    _dumps(transition.reward),
                    _dumps(transition.next_observation),
                    _dumps(transition.terminated),
                    _dumps(transition.truncated),
                    _dumps(transition.info),
                )
                for index, transition in enumerate(self._buffer)
            ]
            connection.executemany(
                f"""
                INSERT INTO {table} (
                    position,
                    observation,
                    action,
                    reward,
                    next_observation,
                    terminated,
                    truncated,
                    info
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    @classmethod
    def load_from_sqlite(cls, db_path: str | Path, table_name: str = "replay_buffer") -> ReplayBuffer:
        """Load a replay buffer from a SQLite database."""
        path = Path(db_path)
        _validate_table_name(table_name)
        table = table_name
        metadata_table = _METADATA_TABLE_NAME

        with sqlite3.connect(path) as connection:
            capacity_row = connection.execute(
                f"SELECT capacity FROM {metadata_table} WHERE buffer_name = ?",
                (table_name,),
            ).fetchone()
            if capacity_row is None:
                raise ValueError(
                    f"No replay buffer metadata found for table {table_name!r}. "
                    "Verify the database file and table name are correct."
                )

            buffer = cls(int(capacity_row[0]))
            rows = connection.execute(
                f"""
                SELECT observation, action, reward, next_observation, terminated, truncated, info
                FROM {table}
                ORDER BY position ASC
                """
            )
            for row in rows:
                observation, action, reward, next_observation, terminated, truncated, info = row
                buffer.add(
                    observation=_loads(observation),
                    action=_loads(action),
                    reward=_load_reward(reward),
                    next_observation=_loads(next_observation),
                    terminated=_load_bool(terminated, "terminated"),
                    truncated=_load_bool(truncated, "truncated"),
                    info=_loads(info),
                )

        return buffer
