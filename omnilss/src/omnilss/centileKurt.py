"""R-aligned centile-kurtosis helper.

R reference: `gamlss.dist/R/momentSK.R::centileKurt`.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .centileSK import centileSK


@dataclass(frozen=True)
class CentileKurtResult:
    lower_tail_spread: float
    upper_tail_spread: float
    kurtosis_ratio: float


def centileKurt(y, centiles=(10, 25, 75, 90)) -> CentileKurtResult:
    """Compute a centile-based kurtosis-style spread ratio."""
    c = tuple(float(v) for v in centiles)
    if len(c) != 4:
        raise ValueError("centiles must contain exactly 4 values (e.g. 10,25,75,90)")
    q = centileSK(y, centiles=c).values
    lower = float(q[1] - q[0])
    upper = float(q[3] - q[2])
    mid = float(q[2] - q[1])
    eps = np.finfo(np.float64).eps
    ratio = (lower + upper) / max(mid, eps)
    return CentileKurtResult(lower_tail_spread=lower, upper_tail_spread=upper, kurtosis_ratio=ratio)


__all__ = ["centileKurt", "CentileKurtResult"]
