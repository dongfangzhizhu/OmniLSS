"""CPU-first weighted least squares solvers."""

from __future__ import annotations

import numpy as np


def solve_weighted_least_squares_cholesky(
    X: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
    ridge: float = 1e-10,
) -> np.ndarray:
    """Solve WLS via normal equations + Cholesky with ridge fallback."""
    X = np.asarray(X, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    sqrt_w = np.sqrt(np.clip(w, 1e-12, np.inf))
    Xw = X * sqrt_w[:, None]
    zw = z * sqrt_w

    XtX = Xw.T @ Xw
    Xtz = Xw.T @ zw
    p = XtX.shape[0]
    A = XtX + ridge * np.eye(p, dtype=np.float64)

    try:
        L = np.linalg.cholesky(A)
        y = np.linalg.solve(L, Xtz)
        beta = np.linalg.solve(L.T, y)
        return beta
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A, rcond=1e-10) @ Xtz
