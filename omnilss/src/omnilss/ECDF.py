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


def ECDF(y) -> ECDFResult:
    """Return sorted values and empirical cumulative probabilities."""
    arr = np.asarray(y, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("y must contain at least one value")
    x = np.sort(arr)
    p = np.arange(1, x.size + 1, dtype=np.float64) / float(x.size)
    return ECDFResult(x=x, p=p)


__all__ = ["ECDF", "ECDFResult"]
