"""R-aligned moment/skewness/kurtosis helper.

R reference: `gamlss.dist/R/momentSK.R::{momentSK, momentSkew}`.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MomentSKResult:
    mean: float
    variance: float
    skewness: float
    kurtosis: float


def momentSK(y) -> MomentSKResult:
    """Compute empirical moments used by SK diagnostics."""
    arr = np.asarray(y, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("y must contain at least one value")
    mu = float(np.mean(arr))
    centered = arr - mu
    var = float(np.mean(centered ** 2))
    if var <= 0:
        return MomentSKResult(mean=mu, variance=var, skewness=0.0, kurtosis=0.0)
    sd = np.sqrt(var)
    skew = float(np.mean((centered / sd) ** 3))
    kurt = float(np.mean((centered / sd) ** 4))
    return MomentSKResult(mean=mu, variance=var, skewness=skew, kurtosis=kurt)


def momentSkew(y) -> float:
    """Return empirical skewness as convenience alias."""
    return momentSK(y).skewness


__all__ = ["momentSK", "momentSkew", "MomentSKResult"]
