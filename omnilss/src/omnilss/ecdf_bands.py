"""ECDF non-parametric confidence bands from ``gamlss.ggplots::y_ecdf``.

R source: ``R/gamlss.ggplots/R/y_ecdf.R``.  The R function draws a ggplot with
``stat_ecdf`` plus a ``geom_ribbon`` of simultaneous confidence bands; this
port returns the numerical band / ECDF data that would have been plotted.

Two band types (Ross Darnell's SMIR ``NPL.bands``), both live-R verified to
<=1e-15 in ``omnilss/tests/test_ecdf_bands.py``:

* ``Owen`` (``NPOwen.bands``) and ``JW`` (``NPJagerWellner.bands``): over the
  distinct sorted values ``yi`` with cumulative counts ``cn`` and ``nn=n+1``,
  ``p=cn/nn``; a polynomial-in-``log(nn)`` ``lambda`` (different for 95/99 and
  for n<=100 vs >100) gives ``lambda=sqrt(2*kind_lambda)``;
  ``phi=pbeta(p, 1/3, 1/3)``, ``se=1/(5.3*sqrt(nn*(p*(1-p))^(1/3)))``;
  ``[lower, upper] = qbeta(clip(phi -/+ lambda*se, 0, 1), 1/3, 1/3)``.
  scipy ``scipy.stats.beta.cdf/ppf`` reproduces R's ``pbeta/qbeta`` exactly.

R quirks reproduced (do NOT "fix" these):
* the 99%-Owen polynomial goes **negative** for small n, so ``sqrt(2*lambda)``
  is NaN and ``lower``/``upper`` are NaN -- R emits ``NaNs produced`` and the
  band is all-NaN.  We keep NaN (verified against live R).
* the ECDF scores use ``gamlss::ECDF``, whose denominator is **n+1**
  (e.g. ``ECDF(c(1,2,2,3))(1:3) = c(0.2,0.6,0.8)``), i.e.
  ``scores = (#{y_j <= y_i}) / (n+1)`` -- NOT the usual n denominator.
* ``other_prep_data`` drops NAs by observation position, colours outliers via
  ``median +/- value*IQR`` (R ``IQR`` = type-7 quantile difference), and sets
  ``txt`` to ``obs`` for outliers else NA.
* ``R table(x)`` gives distinct values ascending with their counts; ``x`` of
  the band is those distinct values.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import beta


__all__ = ["EcdfBand", "EcdfData", "y_ecdf"]


@dataclass(frozen=True)
class EcdfBand:
    """Simultaneous confidence band rows (distinct x ascending)."""

    x: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    conf_level: str
    band_type: str


@dataclass(frozen=True)
class EcdfData:
    """``other_prep_data`` table that ``stat_ecdf`` would have drawn."""

    obs: np.ndarray
    y: np.ndarray
    scores: np.ndarray
    color: np.ndarray
    txt: np.ndarray


def _nonparam_bands(y: np.ndarray, conf_level: str, band_type: str) -> EcdfBand:
    vals, counts = np.unique(y, return_counts=True)
    cn = np.cumsum(counts).astype(np.float64)
    nn = np.full(cn.shape, float(y.size + 1))
    p = cn / nn
    log_nn = np.log(nn)
    if band_type == "Owen":
        if conf_level == "95":
            lam = np.where(
                nn <= 100,
                3.0123 + 0.4835 * log_nn - 0.00957 * log_nn**2 - 0.001488 * log_nn**3,
                3.0806 + 0.4894 * log_nn - 0.02086 * log_nn**2,
            )
        else:  # 99
            lam = np.where(
                nn <= 100,
                -4.626 - 0.541 * log_nn + 0.0242 * log_nn**2,
                -4.71 - 0.512 * log_nn + 0.0219 * log_nn**2,
            )
    else:  # Jager-Wellner
        if conf_level == "95":
            lam = np.where(
                nn <= 100,
                3.6792 + 0.5720 * log_nn - 0.0567 * log_nn**2 + 0.0027 * log_nn**3,
                3.7752 + 0.5062 * log_nn - 0.0417 * log_nn**2 + 0.0016 * log_nn**3,
            )
        else:  # 99
            lam = np.where(
                nn <= 100,
                5.3318 + 0.5539 * log_nn - 0.00370 * log_nn**2,
                5.6392 + 0.4018 * log_nn - 0.0183 * log_nn**2,
            )
    with np.errstate(invalid="ignore"):
        lam = np.sqrt(2.0 * lam)         # NaN for negative lambda (99%-Owen)
        phi = beta.cdf(p, 1.0 / 3.0, 1.0 / 3.0)
        se = 1.0 / (5.3 * np.sqrt(nn * (p * (1.0 - p)) ** (1.0 / 3.0)))
        phiu = np.clip(phi + lam * se, 0.0, 1.0)
        phil = np.clip(phi - lam * se, 0.0, 1.0)
    lower = beta.ppf(phil, 1.0 / 3.0, 1.0 / 3.0)
    upper = beta.ppf(phiu, 1.0 / 3.0, 1.0 / 3.0)
    return EcdfBand(x=vals, lower=lower, upper=upper,
                    conf_level=conf_level, band_type=band_type)


def _prep_data(y: np.ndarray, value: float) -> EcdfData:
    obs_all = np.arange(1, y.size + 1, dtype=np.float64)
    keep = ~np.isnan(y)
    obs = obs_all[keep]
    vals = y[keep]
    q25, q75 = np.quantile(vals, [0.25, 0.75], method="linear")
    iqr = q75 - q25
    med = float(np.median(vals))
    lo, hi = med - iqr * value, med + iqr * value
    # gamlss::ECDF uses denominator n+1
    n = vals.size
    scores = np.array([np.sum(vals <= v) / (n + 1.0) for v in vals],
                      dtype=np.float64)
    outlier = (vals >= hi) | (vals <= lo)
    color = np.where(outlier, "outlier", "normal")
    txt = np.where(outlier, obs.astype(np.int64).astype(object), None)
    return EcdfData(obs=obs, y=vals, scores=scores, color=color, txt=txt)


def y_ecdf(
    y: Any,
    type: str = "Owen",
    conf_level: str = "95",
    value: float = 2.0,
    **_: Any,
) -> tuple[EcdfBand, EcdfData]:
    """R ``gamlss.ggplots::y_ecdf`` numerical data (band + ECDF table).

    Returns ``(EcdfBand, EcdfData)`` -- the data behind ``geom_ribbon`` and
    ``stat_ecdf`` respectively.  ``type`` is ``"Owen"``/``"JW"``,
    ``conf_level`` is ``"95"``/``"99"``.  Plotting-only args (``points.col``,
    ``check_overlap``, ``show.outliers``, ``from``/``to``/``title``) are
    accepted and ignored, matching the R function's dead arguments.
    """
    if type not in ("Owen", "JW"):
        raise ValueError("type must be 'Owen' or 'JW'")
    if conf_level not in ("95", "99"):
        raise ValueError("conf_level must be '95' or '99'")
    values = np.asarray(y, dtype=np.float64).ravel()
    # R computes the bands from ``d$y`` (the NA-dropped vector), not raw y.
    kept = values[~np.isnan(values)]
    band = _nonparam_bands(kept, conf_level, type)
    data = _prep_data(values, value)
    return band, data
