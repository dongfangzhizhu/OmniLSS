"""R-aligned quantile-sheet helper utilities.

R reference: `gamlss/R/quantSheets.R::{quantSheets, findPower, z.scoresQS}`.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class QuantSheetsResult:
    probs: np.ndarray
    quantiles: np.ndarray


def quantSheets(
    y, probs=(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99), weights=None
) -> QuantSheetsResult:
    """Return empirical quantile sheet for requested probabilities."""
    arr = np.asarray(y, dtype=np.float64).ravel()
    p = np.asarray(probs, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("y must contain at least one value")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probs must be in [0, 1]")
    if np.any(np.diff(p) < 0):
        raise ValueError("probs must be non-decreasing")

    if weights is None:
        q = np.quantile(arr, p)
        return QuantSheetsResult(probs=p, quantiles=np.asarray(q, dtype=np.float64))

    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.shape != arr.shape:
        raise ValueError("weights must have same shape as y")
    if np.any(w < 0) or not np.isfinite(w).all():
        raise ValueError("weights must be finite and non-negative")
    total_w = float(np.sum(w))
    if total_w <= 0.0:
        raise ValueError("sum of weights must be positive")

    order = np.argsort(arr)
    xs = arr[order]
    ws = w[order]
    cdf = np.cumsum(ws, dtype=np.float64) / total_w
    idx = np.searchsorted(cdf, p, side="left")
    idx = np.clip(idx, 0, xs.size - 1)
    q = xs[idx]
    return QuantSheetsResult(probs=p, quantiles=np.asarray(q, dtype=np.float64))


def findPower(x, y, weights=None) -> float:
    """Estimate Box-Cox power that best linearizes y against x on log-scale grid."""
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    if x_arr.size != y_arr.size or x_arr.size == 0:
        raise ValueError("x and y must be non-empty with equal length")

    if weights is None:
        w = np.ones_like(x_arr, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.shape != x_arr.shape:
            raise ValueError("weights must have same shape as x/y")
        if np.any(w < 0) or not np.isfinite(w).all():
            raise ValueError("weights must be finite and non-negative")
        if float(np.sum(w)) <= 0.0:
            raise ValueError("sum of weights must be positive")

    grid = np.linspace(-2.0, 2.0, 81)
    best_p, best_sse = 0.0, np.inf
    x_safe = np.maximum(x_arr, np.finfo(np.float64).eps)
    y_safe = np.maximum(y_arr, np.finfo(np.float64).eps)
    lx = np.log(x_safe)
    for p in grid:
        if abs(p) < 1e-8:
            ty = np.log(y_safe)
        else:
            ty = (np.power(y_safe, p) - 1.0) / p
        coef = np.polyfit(lx, ty, deg=1, w=np.sqrt(w))
        pred = np.polyval(coef, lx)
        sse = float(np.sum(w * (ty - pred) ** 2))
        if sse < best_sse:
            best_sse = sse
            best_p = float(p)
    return best_p


def z_scoresQS(y, mu, sigma, weights=None) -> np.ndarray:
    """Simple z-score utility used by quant-sheet diagnostics."""
    y_arr = np.asarray(y, dtype=np.float64)
    mu_arr = np.asarray(mu, dtype=np.float64)
    sigma_arr = np.maximum(
        np.asarray(sigma, dtype=np.float64), np.finfo(np.float64).eps
    )
    z = (y_arr - mu_arr) / sigma_arr
    if weights is None:
        return z
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != z.shape:
        raise ValueError("weights must have same shape as y")
    if np.any(w < 0) or not np.isfinite(w).all():
        raise ValueError("weights must be finite and non-negative")
    return w * z


__all__ = ["quantSheets", "findPower", "z_scoresQS", "QuantSheetsResult"]
