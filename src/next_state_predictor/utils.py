"""Utility helpers."""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Set global random seeds for reproducibility.

    Sets seeds for Python's built-in :mod:`random` module and NumPy.

    Args:
        seed: The integer seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
