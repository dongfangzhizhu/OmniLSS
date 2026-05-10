"""R-aligned residual autocorrelation interfaces.

R source reference:
- file: `gamlss/R/acfResid.R`
- functions: `acfResid`
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import GAMLSSModel
from .operations import is_gamlss, residuals


@dataclass(frozen=True)
class ResidualACFResult:
    """Compact residual autocorrelation summary for staged diagnostics."""

    lags: np.ndarray
    residual_acf: np.ndarray
    squared_acf: np.ndarray
    cubed_acf: np.ndarray
    quartic_acf: np.ndarray


def _require_gamlss(object: GAMLSSModel) -> None:
    if not is_gamlss(object):
        raise TypeError("This is not an gamlss object")


def _ordered_residuals(object: GAMLSSModel) -> np.ndarray:
    values = np.asarray(residuals(object, what="z-scores"), dtype=np.float64).ravel()
    if values.size == 0:
        return values
    return values[np.isfinite(values)]


def _autocorrelation(values: np.ndarray, lag: int) -> float:
    """R reference: `gamlss/R/acfResid.R::acfResid` (internal ACF computation)."""
    n = values.size
    if n == 0 or lag <= 0 or lag >= n:
        return float("nan")
    centered = values - np.mean(values)
    denom = np.sum(centered * centered)
    if denom <= 0.0:
        return float("nan")
    numer = np.sum(centered[:-lag] * centered[lag:])
    return float(numer / denom)


def acf_residuals(
    object: GAMLSSModel,
    max_lag: int | None = None,
) -> ResidualACFResult:
    """R reference: `gamlss/R/acfResid.R::acfResid`.

    Staged behavior:
    - Returns autocorrelation summaries instead of plotting.
    - Reports ACF for residuals, residuals^2, residuals^3, residuals^4.
    """

    _require_gamlss(object)
    z = _ordered_residuals(object)
    if z.size < 2:
        raise ValueError("need at least two residuals for autocorrelation diagnostics")
    max_allowed = z.size - 1
    lag_count = min(max_allowed, 10 if max_lag is None else int(max_lag))
    if lag_count < 1:
        raise ValueError("max_lag must be at least 1")
    lags = np.arange(1, lag_count + 1, dtype=np.int64)
    return ResidualACFResult(
        lags=lags,
        residual_acf=np.array([_autocorrelation(z, int(lag)) for lag in lags], dtype=np.float64),
        squared_acf=np.array([_autocorrelation(np.square(z), int(lag)) for lag in lags], dtype=np.float64),
        cubed_acf=np.array([_autocorrelation(np.power(z, 3.0), int(lag)) for lag in lags], dtype=np.float64),
        quartic_acf=np.array([_autocorrelation(np.power(z, 4.0), int(lag)) for lag in lags], dtype=np.float64),
    )


# R-style exact-name alias
acfResid = acf_residuals

__all__ = [
    "ResidualACFResult",
    "acfResid",
    "acf_residuals",
]
