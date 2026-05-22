"""R-aligned flexible distribution summary helper.

R reference: `gamlss.dist/R/flexDist.R::flexDist`.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .momentSK import momentSK


@dataclass(frozen=True)
class FlexDistResult:
    support_min: float
    support_max: float
    mean: float
    variance: float
    skewness: float
    kurtosis: float
    centiles: np.ndarray


def flexDist(y, centiles=(0.01, 0.05, 0.5, 0.95, 0.99)) -> FlexDistResult:
    """Build a compact empirical distribution summary."""
    arr = np.asarray(y, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("y must contain at least one value")
    c = np.asarray(centiles, dtype=np.float64).ravel()
    if np.any((c < 0) | (c > 1)):
        raise ValueError("centiles must be in [0, 1]")
    m = momentSK(arr)
    q = np.quantile(arr, c)
    return FlexDistResult(
        support_min=float(np.min(arr)),
        support_max=float(np.max(arr)),
        mean=m.mean,
        variance=m.variance,
        skewness=m.skewness,
        kurtosis=m.kurtosis,
        centiles=np.asarray(q, dtype=np.float64),
    )


__all__ = ["flexDist", "FlexDistResult"]
