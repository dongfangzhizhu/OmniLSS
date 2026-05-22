"""R-aligned `vis.lo` style payload helper.

R reference: `gamlss/R/lo_new.R::vis.lo`.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class VisLoResult:
    x: np.ndarray
    y: np.ndarray


def vis_lo(x, y) -> VisLoResult:
    """Return x/y pairs sorted by x for downstream loess visualization."""
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    if x_arr.size == 0 or y_arr.size == 0:
        raise ValueError("x and y must be non-empty")
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same length")
    order = np.argsort(x_arr)
    return VisLoResult(x=x_arr[order], y=y_arr[order])


__all__ = ["vis_lo", "VisLoResult"]
