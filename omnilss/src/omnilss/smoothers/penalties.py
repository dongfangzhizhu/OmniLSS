"""Penalty matrices for penalized regression splines.

R source: gamlss/R/pb.R
References:
- Eilers & Marx (1996). Flexible smoothing with B-splines and penalties.
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
import numpy as np
from scipy import linalg as sp_linalg


# ---------------------------------------------------------------------------
# Core penalty construction (pure numpy, device-agnostic)
# ---------------------------------------------------------------------------

def difference_penalty(n_basis: int, order: int = 2) -> np.ndarray:
    """Difference penalty matrix D of shape (n_basis-order, n_basis).

    R equivalent: diff(diag(r), diff=order)
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    D = np.eye(n_basis)
    for _ in range(order):
        D = np.diff(D, axis=0)
    return D


def penalty_matrix(n_basis: int, order: int = 2) -> np.ndarray:
    """Full penalty matrix P = D^T D of shape (n_basis, n_basis)."""
    D = difference_penalty(n_basis, order)
    return D.T @ D


# ---------------------------------------------------------------------------
# Effective df and lambda search — all numpy/scipy, no JAX device ops
# ---------------------------------------------------------------------------

def _edf_numpy(
    X: np.ndarray,
    P: np.ndarray,
    lambda_: float,
    w: Optional[np.ndarray] = None,
) -> float:
    """Compute edf = tr(H) using numpy.

    H = X (X^T W X + lambda P)^{-1} X^T W
    """
    if w is None:
        w = np.ones(X.shape[0])
    Xw = X * w[:, None]
    A = Xw.T @ X + lambda_ * P
    try:
        # tr(H) = tr(X A^{-1} X^T W) = sum_i (X_i^T A^{-1} X_i w_i)
        # = tr(A^{-1} X^T W X) = tr(A^{-1} (A - lambda P))
        #                       = p - lambda * tr(A^{-1} P)
        # Use the last form: cheaper than full hat matrix
        Ainv_P = sp_linalg.solve(A, P, assume_a="pos")
        edf = X.shape[1] - lambda_ * np.trace(Ainv_P)
    except sp_linalg.LinAlgError:
        edf = float("nan")
    return float(edf)


def effective_df(
    X: jnp.ndarray,
    penalty: jnp.ndarray,
    lambda_: float,
    weights: Optional[jnp.ndarray] = None,
) -> float:
    """Effective degrees of freedom for penalized regression.

    Accepts JAX arrays but converts to numpy internally so it works on
    both CPU and GPU without device-sync issues.
    """
    X_np = np.asarray(X)
    P_np = np.asarray(penalty)
    w_np = None if weights is None else np.asarray(weights)
    return _edf_numpy(X_np, P_np, float(lambda_), w_np)


def find_lambda_for_df(
    X: jnp.ndarray,
    penalty: jnp.ndarray,
    target_df: float,
    weights: Optional[jnp.ndarray] = None,
    lambda_min: float = 1e-10,
    lambda_max: float = 1e10,
    tol: float = 1e-4,
    max_iter: int = 60,
) -> float:
    """Bisection search for lambda that gives target edf.

    All computation in numpy — safe on GPU.
    """
    X_np = np.asarray(X)
    P_np = np.asarray(penalty)
    w_np = None if weights is None else np.asarray(weights)

    edf_lo = _edf_numpy(X_np, P_np, lambda_max, w_np)
    edf_hi = _edf_numpy(X_np, P_np, lambda_min, w_np)

    if target_df < edf_lo or target_df > edf_hi:
        raise ValueError(
            f"target_df={target_df:.4f} is outside achievable range "
            f"[{edf_lo:.4f}, {edf_hi:.4f}]"
        )

    lo, hi = lambda_min, lambda_max
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        edf_mid = _edf_numpy(X_np, P_np, mid, w_np)
        if abs(edf_mid - target_df) < tol:
            return mid
        if edf_mid > target_df:   # edf decreases as lambda increases
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0
