"""R-aligned moment/skewness/kurtosis helper.

R reference: `gamlss.dist/R/Functions_for_SK_in_gamlss.R::momentSK`.

Weighted moments with the transformed skewness/kurtosis used by the bucket
plots (Jones and Pewsey 2009):

    g = w / sum(w)
    gamma1 = m3 / m2^1.5,  gamma2 = m4 / m2^2 - 3
    trans.skew = gamma1 / (1 + |gamma1|)
    trans.kurt = gamma2 / (1 + |gamma2|)
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
    excess_kurtosis: float
    trans_skew: float
    trans_kurt: float
    n: float
    jarque_bera: float


def momentSK(y, weights=None) -> MomentSKResult:
    """R momentSK: weighted sample moments and the transformed SK coordinates."""

    arr = np.asarray(y, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("y must contain at least one value")
    if weights is None:
        w = np.ones(arr.size, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.size != arr.size:
            raise ValueError("y and weights must have the same length")
    keep = ~(np.isnan(w) | (w == 0) | np.isnan(arr) | np.isinf(arr))
    arr = arr[keep]
    w = w[keep]
    if arr.size == 0:
        raise ValueError("no finite observations with positive weight")
    total = float(np.sum(w))
    g = w / total
    m1 = float(np.sum(g * arr))
    centered = arr - m1
    m2 = float(np.sum(g * centered**2))
    if m2 <= 0.0:
        return MomentSKResult(
            mean=m1,
            variance=m2,
            skewness=0.0,
            kurtosis=3.0,
            excess_kurtosis=0.0,
            trans_skew=0.0,
            trans_kurt=0.0,
            n=total,
            jarque_bera=0.0,
        )
    m3 = float(np.sum(g * centered**3))
    m4 = float(np.sum(g * centered**4))
    gamma1 = m3 / m2**1.5
    beta2 = m4 / m2**2
    gamma2 = beta2 - 3.0
    beta1 = gamma1**2
    return MomentSKResult(
        mean=m1,
        variance=m2,
        skewness=gamma1,
        kurtosis=beta2,
        excess_kurtosis=gamma2,
        trans_skew=gamma1 / (1.0 + abs(gamma1)),
        trans_kurt=gamma2 / (1.0 + abs(gamma2)),
        n=total,
        jarque_bera=(total / 6.0) * beta1 + (total / 24.0) * gamma2**2,
    )


def momentSkew(y) -> float:
    """Return empirical skewness as convenience alias."""

    return momentSK(y).skewness


__all__ = ["momentSK", "momentSkew", "MomentSKResult"]
