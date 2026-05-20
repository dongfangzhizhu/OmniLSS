"""Hessian stabilization utilities for iterative solvers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HessianStabilizationResult:
    matrix: np.ndarray
    lambda_value: float
    condition_number: float
    was_regularized: bool


def stabilize_hessian(
    hessian: np.ndarray,
    base_lambda: float = 1e-8,
    max_lambda: float = 1e3,
    condition_threshold: float = 1e10,
) -> HessianStabilizationResult:
    """Apply Levenberg-style ``H + λI`` regularization.

    The returned matrix is finite and positive-definite enough for stable solves.
    """
    H = np.asarray(hessian, dtype=np.float64)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("hessian must be a square matrix")

    if not np.all(np.isfinite(H)):
        H = np.nan_to_num(H, nan=0.0, posinf=1e12, neginf=-1e12)

    n = H.shape[0]
    identity = np.eye(n, dtype=np.float64)
    lambda_value = 0.0
    cond = float(np.linalg.cond(H)) if H.size else 1.0
    regularized = False

    if (not np.isfinite(cond)) or cond > condition_threshold:
        regularized = True
        lambda_value = base_lambda
        while lambda_value <= max_lambda:
            candidate = H + lambda_value * identity
            try:
                _ = np.linalg.cholesky(candidate)
                H = candidate
                cond = float(np.linalg.cond(H)) if H.size else 1.0
                break
            except np.linalg.LinAlgError:
                lambda_value *= 10.0
        else:
            lambda_value = max_lambda
            H = H + lambda_value * identity
            cond = float(np.linalg.cond(H)) if H.size else 1.0

    return HessianStabilizationResult(
        matrix=H,
        lambda_value=float(lambda_value),
        condition_number=cond,
        was_regularized=regularized,
    )
