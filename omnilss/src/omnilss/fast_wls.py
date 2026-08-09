"""CPU-first weighted least squares solvers."""

from __future__ import annotations

import numpy as np


def solve_weighted_least_squares_cholesky(
    X: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
    ridge: float | None = None,
) -> np.ndarray:
    """Solve WLS via numerically stable lstsq (QR/SVD).

    If ``ridge > 0``, Levenberg-style damping is applied by appending
    ``sqrt(ridge)*I`` rows (equivalent to ``X^T X + ridge*I``).  The old
    normal-equations + Cholesky path is replaced by lstsq to avoid squaring
    the condition number of ill-conditioned design matrices.
    """
    X = np.asarray(X, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    sqrt_w = np.sqrt(np.clip(w, 1e-12, np.inf))
    Xw = X * sqrt_w[:, None]
    zw = z * sqrt_w

    if ridge is not None and ridge > 0.0:
        # Levenberg-style damping: append sqrt(ridge)*I rows so the normal
        # equations are X^T X + ridge*I (numerically stable, no squaring of
        # the condition number beyond what lstsq already avoids).
        p = Xw.shape[1]
        Xw = np.vstack([Xw, np.sqrt(ridge) * np.eye(p)])
        zw = np.concatenate([zw, np.zeros(p)])

    # Stable path: weighted least squares via lstsq (QR/SVD).
    coef, _, _, _ = np.linalg.lstsq(Xw, zw, rcond=None)
    return coef
