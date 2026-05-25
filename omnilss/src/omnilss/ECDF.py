"""R-aligned empirical CDF helper.

R reference: `gamlss/R/ECDF.R::ECDF`.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ECDFResult:
    x: np.ndarray
    p: np.ndarray
    weights: np.ndarray


def ECDF(y, weights=None) -> ECDFResult:
    """Return sorted values and empirical cumulative probabilities."""
    arr = np.asarray(y, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("y must contain at least one value")

    if weights is None:
        w = np.ones_like(arr, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.shape != arr.shape:
            raise ValueError("weights must have same shape as y")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        if not np.isfinite(w).all():
            raise ValueError("weights must be finite")

    order = np.argsort(arr)
    x = arr[order]
    ws = w[order]
    total_w = float(np.sum(ws))
    if total_w <= 0.0:
        raise ValueError("sum of weights must be positive")
    p = np.cumsum(ws, dtype=np.float64) / total_w
    return ECDFResult(x=x, p=p, weights=ws)


__all__ = ["ECDF", "ECDFResult"]
