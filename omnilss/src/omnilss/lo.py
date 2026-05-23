"""Lightweight `lo(...)` smoother helper.

R reference: `gamlss/R/lo_new.R::lo`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LoResult:
    x: np.ndarray
    fitted: np.ndarray
    span: float


def lo(x, y, span: float = 0.5) -> LoResult:
    """Return a simple local-linear smoother approximation for diagnostics.

    This is a minimal data-path implementation intended for Phase-1 API
    completeness; parity-level behavior against R `lo()` is tracked separately.
    """
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    if x_arr.size == 0 or y_arr.size == 0:
        raise ValueError("x and y must be non-empty")
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same length")
    if not (0.0 < float(span) <= 1.0):
        raise ValueError("span must be in (0, 1]")

    n = x_arr.size
    k = max(2, int(np.ceil(span * n)))
    order = np.argsort(x_arr)
    xs = x_arr[order]
    ys = y_arr[order]
    fitted = np.empty_like(ys)

    for i in range(n):
        d = np.abs(xs - xs[i])
        h = np.partition(d, k - 1)[k - 1]
        if h <= 0:
            fitted[i] = ys[i]
            continue
        u = d / h
        w = np.where(u < 1.0, (1.0 - u**3) ** 3, 0.0)  # tricube
        sw = np.sum(w)
        fitted[i] = ys[i] if sw <= 0 else np.sum(w * ys) / sw

    return LoResult(x=xs, fitted=fitted, span=float(span))


__all__ = ["lo", "LoResult"]
