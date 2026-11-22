"""R gamlss.dist sample centile skewness (centileSkew).

R source: R/gamlss.dist/R/Functions_for_SK_in_gamlss.R::centileSkew.

centileSkew always routes through R's local quantileW() with weights (default
all-ones), whose type-7-like quantiles use a right-continuous step function.
We reproduce that path exactly rather than calling numpy's quantile, so ties and
weighted data agree with R.
"""

from __future__ import annotations

import warnings

import numpy as np


def _quantile_weighted(
    y: np.ndarray, weights: np.ndarray, probs: np.ndarray
) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    if np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError("Probabilities must be between 0 and 1 inclusive")
    keep = ~(np.isnan(weights) | (weights == 0.0))
    y = y[keep]
    weights = weights[keep]
    ysort, inverse = np.unique(y, return_inverse=True)
    weights1 = np.zeros(ysort.shape[0], dtype=float)
    np.add.at(weights1, inverse, weights)
    cumu = np.cumsum(weights1)
    n = float(cumu[-1])
    order = 1.0 + (n - 1.0) * probs
    low = np.maximum(np.floor(order), 1).astype(int)
    high = np.minimum(low + 1, int(round(n))).astype(int)
    frac = order % 1.0

    def _step(xout: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(cumu, xout, side="left")
        return ysort[np.clip(idx, 0, ysort.shape[0] - 1)]

    allq = _step(np.concatenate([low, high]))
    k = probs.shape[0]
    return (1.0 - frac) * allq[:k] + frac * allq[k:]


def centile_skew(x, cent: float = 1, weights=None) -> dict[str, float]:
    """R's centileSkew: returns a dict with keys 'p' (cent/100) and 'Sp'."""

    x = np.asarray(x, dtype=float)
    if cent > 50:
        raise ValueError("cent should be less than 50")
    if weights is None:
        w = np.ones_like(x)
    else:
        w = np.broadcast_to(np.asarray(weights, dtype=float), x.shape).copy()
    if np.any(np.isnan(x)):
        warnings.warn("NA's will be removed from x", stacklevel=2)
    keep = ~(np.isnan(w) | (w == 0.0) | np.isnan(x))
    x = x[keep]
    w = w[keep]
    probs = np.array([cent / 100.0, 0.25, 0.5, 0.75, 1.0 - cent / 100.0])
    q = _quantile_weighted(x, w, probs)
    sp = ((q[0] + q[4]) / 2.0 - q[2]) / ((q[4] - q[0]) / 2.0)
    return {"p": float(cent) / 100.0, "Sp": float(sp)}


centileSkew = centile_skew

__all__ = ["centile_skew", "centileSkew"]
