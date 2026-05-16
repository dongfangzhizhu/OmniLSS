"""Phase 0 tensor shape protocol helpers."""

from __future__ import annotations

import numpy as np


def validate_vector(name: str, arr: np.ndarray, n: int | None = None) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1D (batch,), got shape={a.shape}")
    if n is not None and a.shape[0] != n:
        raise ValueError(f"{name} length mismatch: expected {n}, got {a.shape[0]}")
    return a


def validate_design_matrix(name: str, x: np.ndarray, n: int | None = None) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2D (batch, features), got shape={a.shape}")
    if n is not None and a.shape[0] != n:
        raise ValueError(f"{name} rows mismatch: expected {n}, got {a.shape[0]}")
    return a
