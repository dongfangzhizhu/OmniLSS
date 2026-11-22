"""Single-variable diagnostics from ``gamlss.ggplots`` that return plot data.

R sources (vendored): ``R/gamlss.ggplots/R/{symmetry_plot.R, y_ecdf.R,
y_hist.R, y_acf.R, resid_index.R}``.  The R functions draw ggplot2 objects;
these ports return the numerical data that would have been plotted instead.

Live-R parity is asserted in ``omnilss/tests/test_ggplots_diag.py``: because
ggplot2 is not installed, the test sources the vendored R files after
stripping the ``ggplot2::`` prefix and stubbing the plotting calls, so the
*real R function bodies* run and ``ggplot()`` records the data frame.  R
quirks reproduced here:

* ``y_symmetry`` splits at the median with ``floor((n+1)/2)`` /
  ``ceiling((n+1)/2)`` and reverses the lower half;
* ``y_dots``/``resid_index`` use ``median +/- value * mad`` (1.4826 constant)
  for outlier flags and drop NAs by observation position;
* ``y_acf``/``y_pacf`` use R's biased ACF (denominator over all n terms),
  ``lag.max = floor(10*log10(n))``, Durbin-Levinson PACF, and a fixed 95%
  line ``qnorm(0.025)/sqrt(n)``;
* ``y_epdf`` rounds with R's ``format(digits=4)`` (``%.4g``) before tallying;
* ``y_epit`` is ``rank(y, ties.method="average") / (n + 1)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import norm, rankdata

__all__ = [
    "EmpiricalPdfTable",
    "AcfData",
    "SymmetryData",
    "ResidualIndexData",
    "y_epit",
    "y_epdf",
    "y_symmetry",
    "resid_symmetry",
    "y_dots",
    "y_acf",
    "y_pacf",
    "resid_index",
]


@dataclass(frozen=True)
class EmpiricalPdfTable:
    x: np.ndarray
    freq: np.ndarray


@dataclass(frozen=True)
class AcfData:
    lag: np.ndarray
    acf: np.ndarray
    ciline: float


@dataclass(frozen=True)
class SymmetryData:
    lower_half: np.ndarray
    upper_half: np.ndarray
    title: str


@dataclass(frozen=True)
class ResidualIndexData:
    obs: np.ndarray
    sdres: np.ndarray
    color: np.ndarray
    fct_color: np.ndarray
    txt: np.ndarray
    outliers: np.ndarray  # observation numbers flagged as outliers
    threshold: float


def _as_float_array(y: Any) -> np.ndarray:
    return np.asarray(y, dtype=np.float64).ravel()


def y_epit(y: Any) -> np.ndarray:
    """R y_epit: empirical PIT = rank(y) / (n + 1)."""
    values = _as_float_array(y)
    return rankdata(values, method="average") / (values.size + 1.0)


def y_epdf(y: Any, plot: bool = True) -> EmpiricalPdfTable:
    """R y_epdf(plot=FALSE): table of format(y, digits=4) values and counts."""
    values = _as_float_array(y)
    # R's format(y, digits=4) keeps 4 significant digits; as.numeric() then
    # parses them back.  For these diagnostics that is %.4g.
    rounded = np.array([float(f"{v:.4g}") for v in values], dtype=np.float64)
    unique, counts = np.unique(rounded, return_counts=True)
    return EmpiricalPdfTable(x=unique, freq=counts.astype(np.float64))


def _mad(values: np.ndarray) -> float:
    """R stats::mad: 1.4826 * median(|x - median(x)|)."""
    centre = float(np.median(values))
    return 1.4826 * float(np.median(np.abs(values - centre)))


def y_symmetry(y: Any, title: str | None = None) -> SymmetryData:
    """R y_symmetry: lower/upper half distances from the median."""
    values = _as_float_array(y)
    ordered = np.sort(values)
    n = ordered.size
    centre = float(np.median(values))
    half = (n + 1) // 2  # R floor((n+1)/2)
    lower = (centre - ordered[:half])[::-1]  # R rev(m - y1[1:half])
    start = int(np.ceil((n + 1) / 2.0)) - 1  # R y1[ceiling((n+1)/2):n]
    upper = ordered[start:] - centre
    return SymmetryData(
        lower_half=lower,
        upper_half=upper,
        title="Symmetry plot of variable y" if title is None else str(title),
    )


def resid_symmetry(model: Any, title: str | None = None) -> SymmetryData:
    """R resid_symmetry: symmetry data of a model's quantile residuals."""
    residuals = getattr(model, "residuals", model)
    data = y_symmetry(residuals)
    return SymmetryData(
        lower_half=data.lower_half,
        upper_half=data.upper_half,
        title="resid symmetry plot of model" if title is None else str(title),
    )


def y_dots(
    y: Any,
    value: float = 3,
    seed: int = 123,
    plot: bool = True,
) -> np.ndarray:
    """R y_dots(plot=FALSE): observation indices flagged as outliers.

    R draws a random vertical coordinate for the dot plot (set.seed/runif),
    but plot=FALSE only returns the outlier observation numbers, which depend
    solely on median +/- value * MAD -- no RNG is needed for parity.
    """
    values = _as_float_array(y)
    obs = np.arange(1, values.size + 1, dtype=np.float64)
    keep = ~np.isnan(values)
    obs = obs[keep]
    values = values[keep]
    centre = float(np.median(values))
    spread = _mad(values)
    low = centre - value * spread
    high = centre + value * spread
    outlier = (values >= high) | (values <= low)
    return obs[outlier]


def _acf_biased(x: np.ndarray, nlags: int) -> np.ndarray:
    """R stats::acf(plot=FALSE, demean=TRUE): denominator uses all n terms."""
    n = x.size
    centred = x - float(np.mean(x))
    denom = float(np.dot(centred, centred))
    return np.array(
        [1.0]
        + [
            float(np.dot(centred[: n - k], centred[k:])) / denom
            for k in range(1, nlags + 1)
        ],
        dtype=np.float64,
    )


def _pacf_durbin_levinson(r: np.ndarray, nlags: int) -> np.ndarray:
    """R stats::pacf(plot=FALSE) uses the Durbin-Levinson recursion."""
    phi = np.zeros((nlags + 1, nlags + 1), dtype=np.float64)
    phi[1, 1] = r[1]
    for k in range(2, nlags + 1):
        numerator = r[k] - sum(phi[k - 1, j] * r[k - j] for j in range(1, k))
        denominator = 1.0 - sum(phi[k - 1, j] * r[j] for j in range(1, k))
        phi[k, k] = numerator / denominator
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
    return np.array([phi[k, k] for k in range(1, nlags + 1)], dtype=np.float64)


def _lag_max(n: int) -> int:
    """R acf default lag.max = floor(10*log10(n)), capped at n-1."""
    return max(0, min(n - 1, int(math.floor(10.0 * math.log10(n)))))


def y_acf(x: Any, data: Any = None, title: str | None = None) -> AcfData:
    """R y_acf: the plotted ACF data (lag, acf) plus the 95% line."""
    values = _as_float_array(x)
    n = values.size
    lags = _lag_max(n)
    acf_values = _acf_biased(values, lags)
    return AcfData(
        lag=np.arange(0, lags + 1, dtype=np.float64),
        acf=acf_values,
        ciline=float(norm.ppf((1 - 0.95) / 2) / math.sqrt(n)),
    )


def y_pacf(x: Any, data: Any = None, title: str | None = None) -> AcfData:
    """R y_pacf: the plotted PACF data (lag, acf) plus the 95% line."""
    values = _as_float_array(x)
    n = values.size
    lags = _lag_max(n)
    r = _acf_biased(values, lags)
    pacf_values = _pacf_durbin_levinson(r, lags)
    return AcfData(
        lag=np.arange(1, lags + 1, dtype=np.float64),
        acf=pacf_values,
        ciline=float(norm.ppf((1 - 0.95) / 2) / math.sqrt(n)),
    )


def resid_index(
    obj: Any = None,
    resid: Any = None,
    value: float = 2,
    plot: bool = True,
) -> ResidualIndexData:
    """R resid_index(plot=FALSE): the plotted residual-index data frame."""
    if resid is None:
        if obj is None:
            raise ValueError(
                "A GAMLSS fitted object or the argument resid should be used"
            )
        resid = getattr(obj, "residuals", obj)
    sdres = _as_float_array(resid)
    obs = np.arange(1, sdres.size + 1, dtype=np.float64)
    keep = ~np.isnan(sdres)
    obs = obs[keep]
    sdres = sdres[keep]
    color = np.where((sdres >= value) | (sdres <= -value), "outlier", "normal")
    txt = np.where(color == "outlier", obs.astype(int).astype(object), None)
    flagged = obs[color == "outlier"]
    return ResidualIndexData(
        obs=obs,
        sdres=sdres,
        color=color,
        fct_color=color.copy(),
        txt=txt,
        outliers=flagged,
        threshold=float(value),
    )
