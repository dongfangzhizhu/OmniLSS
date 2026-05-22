"""R-aligned centile-by-skewness summary helper.

R reference: `gamlss.dist/R/momentSK.R::centileSK`.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CentileSKResult:
    centiles: np.ndarray
    values: np.ndarray


def centileSK(y, centiles=(1, 5, 10, 25, 50, 75, 90, 95, 99)) -> CentileSKResult:
    """Return empirical centiles for diagnostics traditionally grouped with SK tools."""
    arr = np.asarray(y, dtype=np.float64).ravel()
    c = np.asarray(centiles, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("y must contain at least one value")
    if np.any((c < 0) | (c > 100)):
        raise ValueError("centiles must be in [0, 100]")
    probs = c / 100.0
    vals = np.quantile(arr, probs)
    return CentileSKResult(centiles=c, values=np.asarray(vals, dtype=np.float64))


__all__ = ["centileSK", "CentileSKResult"]
