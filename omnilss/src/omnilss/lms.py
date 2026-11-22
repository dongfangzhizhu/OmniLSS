"""R-aligned LMS/z-score utilities.

R reference: `gamlss/R/lms.R::{lms, z.scores}`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm as _scipy_norm


@dataclass(frozen=True)
class LMSResult:
    z: np.ndarray
    weighted_z: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    nu: np.ndarray
    weights: np.ndarray


def z_scores(y, mu, sigma, nu) -> np.ndarray:
    """Compute LMS z-scores using Box-Cox transform with log-limit at nu≈0."""
    y_arr = np.asarray(y, dtype=np.float64)
    mu_arr = np.asarray(mu, dtype=np.float64)
    sigma_arr = np.maximum(
        np.asarray(sigma, dtype=np.float64), np.finfo(np.float64).eps
    )
    nu_arr = np.asarray(nu, dtype=np.float64)

    ratio = np.maximum(
        y_arr / np.maximum(mu_arr, np.finfo(np.float64).eps), np.finfo(np.float64).eps
    )
    use_log = np.abs(nu_arr) < 1e-8
    nu_safe = np.where(use_log, 1.0, nu_arr)
    z_bc = (np.power(ratio, nu_safe) - 1.0) / (nu_safe * sigma_arr)
    z_log = np.log(ratio) / sigma_arr
    return np.where(use_log, z_log, z_bc)


def lms(y, mu, sigma, nu, weights=None) -> LMSResult:
    """Build LMS payload with computed z-scores."""
    y_arr = np.asarray(y, dtype=np.float64)
    mu_arr = np.asarray(mu, dtype=np.float64)
    sigma_arr = np.asarray(sigma, dtype=np.float64)
    nu_arr = np.asarray(nu, dtype=np.float64)
    z = z_scores(y=y_arr, mu=mu_arr, sigma=sigma_arr, nu=nu_arr)

    if weights is None:
        w = np.ones_like(z, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != z.shape:
            raise ValueError("weights must have same shape as y")
        if np.any(w < 0) or not np.isfinite(w).all():
            raise ValueError("weights must be finite and non-negative")

    return LMSResult(
        z=z, weighted_z=w * z, mu=mu_arr, sigma=sigma_arr, nu=nu_arr, weights=w
    )


@dataclass(frozen=True)
class CalibrationResult:
    """Structured result for R ``gamlss::calibration``."""

    target: np.ndarray
    calib: np.ndarray
    sample: np.ndarray
    table: pd.DataFrame
    fan_result: Any = None


def calibration(
    object: Any,
    xvar: str,
    cent: Sequence[float] = (0.4, 2.0, 10.0, 25.0, 50.0, 75.0, 90.0, 98.0, 99.6),
    legend: bool = False,
    fan: bool = False,
    **kwargs: Any,
) -> CalibrationResult:
    """R reference: ``gamlss/R/lms.R::calibration``.

    Calibrates centile curves using empirical quantiles of normalized quantile
    residuals (R type=7):

        z = quantile(resid(object), probs = cent / 100)
        percent = 100 * pNO(z, mu = 0, sigma = 1)

    Then evaluates ``centiles_coverage_data`` (or ``centiles_fan_data`` when
    ``fan=True``) at those calibrated percentages.
    """
    from .centilesFan import centiles_fan_data
    from .centilesPLOT import centiles_coverage_data
    from .operations import is_gamlss, residuals

    if not is_gamlss(object):
        raise TypeError("calibration() requires a fitted GAMLSSModel")

    target = np.asarray(tuple(float(c) for c in cent), dtype=np.float64)
    if target.size == 0 or np.any((target <= 0.0) | (target >= 100.0)):
        raise ValueError("cent must be non-empty percentages strictly in (0, 100)")

    res = np.asarray(residuals(object, what="z-scores"), dtype=np.float64).ravel()
    res = res[np.isfinite(res)]
    if res.size == 0:
        raise ValueError("object has no finite quantile residuals")

    # R quantile(..., type=7) matches NumPy's default method="linear"
    z = np.quantile(res, target / 100.0, method="linear")
    percent = 100.0 * _scipy_norm.cdf(z)
    # Clip strictly inside (0, 100) so downstream inverse-CDF centile calls stay finite
    percent_safe = np.clip(percent, 1e-6, 100.0 - 1e-6)

    if fan:
        fan_res = centiles_fan_data(
            object, xvar=xvar, cent=tuple(percent_safe.tolist()), **kwargs
        )
        cov = centiles_coverage_data(
            object, xvar=xvar, cent=tuple(percent_safe.tolist())
        )
        sample = np.asarray([row.percent_below for row in cov.rows], dtype=np.float64)
        table = pd.DataFrame({"target": target, "calib.": percent, "sample": sample})
        return CalibrationResult(
            target=target,
            calib=percent,
            sample=sample,
            table=table,
            fan_result=fan_res,
        )

    cov = centiles_coverage_data(
        object, xvar=xvar, cent=tuple(percent_safe.tolist()), **kwargs
    )
    calib_col = np.asarray([row.centile for row in cov.rows], dtype=np.float64)
    sample = np.asarray([row.percent_below for row in cov.rows], dtype=np.float64)
    table = pd.DataFrame({"target": target, "calib.": calib_col, "sample": sample})
    return CalibrationResult(
        target=target,
        calib=calib_col,
        sample=sample,
        table=table,
        fan_result=None,
    )


__all__ = ["CalibrationResult", "LMSResult", "calibration", "lms", "z_scores"]
