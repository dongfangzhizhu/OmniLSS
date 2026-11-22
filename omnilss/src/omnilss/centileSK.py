"""R-aligned centile-based skewness/kurtosis summary.

R reference: `gamlss.dist/R/Functions_for_SK_in_gamlss.R::centileSK`.

Using the 1%, 25%, 50%, 75% and 99% quantiles (weights honoured):

    S0.01 = ((Q1 + Q99)/2 - Q50) / ((Q99 - Q1)/2)
    S0.25 = ((Q25 + Q75)/2 - Q50) / ((Q75 - Q25)/2)
    K0.01 = (Q99 - Q1) / (Q75 - Q25)
    excess = K0.01 - 3.449,  trans.K = excess / (1 + |excess|)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CentileSKResult:
    centiles: np.ndarray
    values: np.ndarray
    sk_values: np.ndarray
    s025: float
    s001: float
    k001: float
    stand_k001: float
    excess_k001: float
    trans_k001: float


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, probs: np.ndarray
) -> np.ndarray:
    """R quantileW: type-7 interpolation on the weighted empirical CDF."""

    unique = np.unique(values)
    grouped = np.array([float(np.sum(weights[values == item])) for item in unique])
    cumulative = np.cumsum(grouped)
    total = float(cumulative[-1])
    order = 1.0 + (total - 1.0) * np.asarray(probs, dtype=np.float64)
    low = np.maximum(np.floor(order), 1.0)
    high = np.minimum(low + 1.0, total)
    fraction = order % 1.0

    def lookup(position: np.ndarray) -> np.ndarray:
        index = np.searchsorted(cumulative, position, side="left")
        index = np.clip(index, 0, unique.size - 1)
        return unique[index]

    lower_values = lookup(low)
    upper_values = lookup(high)
    return (1.0 - fraction) * lower_values + fraction * upper_values


def centileSK(y, centiles=(1, 25), weights=None) -> CentileSKResult:
    """R centileSK: centile-based skewness and kurtosis coordinates."""

    arr = np.asarray(y, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("y must contain at least one value")
    if weights is None:
        w = np.ones(arr.size, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.size != arr.size:
            raise ValueError("y and weights must have the same length")
    keep = ~(np.isnan(w) | (w == 0) | np.isnan(arr))
    arr = arr[keep]
    w = w[keep]
    if arr.size == 0:
        raise ValueError("no finite observations with positive weight")
    lower, upper = float(centiles[0]), float(centiles[1])
    if not (0.0 <= lower < upper <= 100.0):
        raise ValueError("centiles must satisfy 0 <= cent[0] < cent[1] <= 100")
    requested = np.asarray(centiles, dtype=np.float64) / 100.0
    values = _weighted_quantile(arr, w, requested)
    probs = np.array(
        [lower / 100.0, upper / 100.0, 0.5, 1.0 - upper / 100.0, 1.0 - lower / 100.0]
    )
    cent = _weighted_quantile(arr, w, probs)
    q_low, q_lower, q_mid, q_upper, q_high = cent
    s001 = ((q_low + q_high) / 2.0 - q_mid) / ((q_high - q_low) / 2.0)
    s025 = ((q_lower + q_upper) / 2.0 - q_mid) / ((q_upper - q_lower) / 2.0)
    k001 = (q_high - q_low) / (q_upper - q_lower)
    excess = k001 - 3.449
    return CentileSKResult(
        centiles=np.asarray(centiles, dtype=np.float64),
        values=np.asarray(values, dtype=np.float64),
        sk_values=np.asarray(cent, dtype=np.float64),
        s025=float(s025),
        s001=float(s001),
        k001=float(k001),
        stand_k001=float(k001 / 3.449),
        excess_k001=float(excess),
        trans_k001=float(excess / (1.0 + abs(excess))),
    )


__all__ = ["centileSK", "CentileSKResult"]
