"""Utility helpers."""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    """Set global random seeds for reproducibility and return a NumPy Generator.

    Sets the seed for Python's built-in :mod:`random` module and returns a
    seeded :class:`numpy.random.Generator` for use in NumPy operations.

    Args:
        seed: The integer seed value to use.

    Returns:
        A seeded :class:`numpy.random.Generator` instance.
    """
    random.seed(seed)
    return np.random.default_rng(seed)
