"""R-aligned theoretical centile helper.

R reference: `gamlss.dist/R/momentSK.R::theoCentileSK`.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class TheoCentileSKResult:
    centiles: np.ndarray
    z_values: np.ndarray


def theoCentileSK(centiles=(1, 5, 10, 25, 50, 75, 90, 95, 99)) -> TheoCentileSKResult:
    """Return theoretical normal-quantile z values for centiles."""
    c = np.asarray(centiles, dtype=np.float64).ravel()
    if np.any((c <= 0) | (c >= 100)):
        raise ValueError("centiles must be strictly between 0 and 100")
    p = c / 100.0
    z = norm.ppf(p)
    return TheoCentileSKResult(centiles=c, z_values=np.asarray(z, dtype=np.float64))


__all__ = ["theoCentileSK", "TheoCentileSKResult"]
