"""R gamlss tail diagnostics: loglogSurv1 / loglogSurv2 / loglogSurv3.

R reference: gamlss/R/FitTail.R (the three "method 2" tail plots).

All three take the upper tail of y (values above quantile(y, prob)), compute the
empirical survival at those points, transform it to log(-log(S)) and regress that
on a transformed covariate:

    type I   (log-Weibull)  x = log(log(y))
    type II  (Weibull)      x = log(y)
    type III (Gumbel)       x = y

R fits gamlss(newY ~ x), which for a normal linear model is ordinary least
squares, so the port solves the normal equations directly and returns the same
coefficients, residual sum of squares and fitted line.

R prints the coefficients and the error sum of squares and then draws the plot;
printing is reproduced (print=True) and the plot is not ported.
"""

from __future__ import annotations

import builtins
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LogLogSurvFit:
    """The fitted object returned by the R loglogSurv functions."""

    type: int
    prob: float
    threshold: float
    coefficients: np.ndarray
    ess: float
    sigma: float
    x: np.ndarray
    y: np.ndarray
    fitted: np.ndarray
    tail: np.ndarray

    @property
    def intercept(self) -> float:
        return float(self.coefficients[0])

    @property
    def slope(self) -> float:
        return float(self.coefficients[1])


@dataclass(frozen=True)
class TailCurveResult:
    """Points of an empirical survival curve (x, curve)."""

    x: np.ndarray
    curve: np.ndarray


def logSurv(y: Any) -> TailCurveResult:
    """R logSurv: the log empirical survival curve log(1 - F_n(x))."""

    from .ECDF import ECDF

    ecdf = ECDF(y)
    eps = np.finfo(np.float64).eps
    survival = np.clip(1.0 - ecdf.p, eps, 1.0)
    return TailCurveResult(x=ecdf.x, curve=np.log(survival))


def loglogSurv(y: Any) -> TailCurveResult:
    """R loglogSurv: log(-log(survival)) over the empirical support."""

    ls = logSurv(y)
    return TailCurveResult(x=ls.x, curve=np.log(-ls.curve))


def _ecdf_step(values: np.ndarray):
    """R ECDF(): unique sorted values with cumsum(tabulate(match(y, ysort)))/(n + 1)."""

    ordered = np.unique(np.sort(values))
    counts = np.bincount(np.searchsorted(ordered, values), minlength=ordered.size)
    cumulative = np.cumsum(counts) / (values.size + 1.0)
    return ordered, cumulative


def _tail_setup(y: Any, prob: float):
    values = np.asarray(y, dtype=float).ravel()
    if values.size == 0:
        raise ValueError("y must contain at least one value")
    if not 0.0 <= float(prob) <= 1.0:
        raise ValueError("prob must be between 0 and 1")
    threshold = float(np.quantile(values, float(prob), method="linear"))
    if threshold < 1.0:
        raise ValueError(
            "For the method for Type I to work the minimum value of the tail must be greater than 1"
        )
    tail = values[values > threshold]
    if tail.size == 0:
        raise ValueError("no observations above the requested quantile")
    ordered, cumulative = _ecdf_step(values)
    position = np.searchsorted(ordered, tail, side="right") - 1
    survival = cumulative[position]
    new_y = np.log(-np.log(1.0 - survival))
    return values, threshold, tail, new_y


def _fit_loglog(
    y: Any, prob: float, kind: int, print: bool, title, lcol, ltype, plot: bool
):
    _values, threshold, tail, new_y = _tail_setup(y, prob)
    if kind == 1:
        x = np.log(np.log(tail))
    elif kind == 2:
        x = np.log(tail)
    else:
        x = tail.copy()

    design = np.column_stack([np.ones(tail.size), x])
    coefficients, *_ = np.linalg.lstsq(design, new_y, rcond=None)
    fitted = design @ coefficients
    residuals = new_y - fitted
    ess = float(np.sum(residuals**2))
    sigma = float(np.sqrt(ess / tail.size))

    if print:
        builtins.print("coefficients " + str(coefficients))
        builtins.print("error sum of squares " + str(ess))
    if plot:
        warnings.warn(
            "the log-log survival plot is not ported; the fitted line is returned instead",
            UserWarning,
            stacklevel=3,
        )

    return LogLogSurvFit(
        type=int(kind),
        prob=float(prob),
        threshold=float(threshold),
        coefficients=np.asarray(coefficients, dtype=float),
        ess=ess,
        sigma=sigma,
        x=np.asarray(x, dtype=float),
        y=np.asarray(new_y, dtype=float),
        fitted=np.asarray(fitted, dtype=float),
        tail=np.asarray(tail, dtype=float),
    )


def loglogSurv1(
    y: Any,
    prob: float = 0.9,
    print: bool = True,
    title=None,
    lcol=None,
    ltype=1,
    plot: bool = False,
    **kwargs: Any,
):
    """R loglogSurv1: type I (log-Weibull) tail check, x = log(log(y))."""

    return _fit_loglog(y, prob, 1, print, title, lcol, ltype, plot)


def loglogSurv2(
    y: Any,
    prob: float = 0.9,
    print: bool = True,
    title=None,
    lcol=None,
    ltype=1,
    plot: bool = False,
    **kwargs: Any,
):
    """R loglogSurv2: type II (Weibull) tail check, x = log(y)."""

    return _fit_loglog(y, prob, 2, print, title, lcol, ltype, plot)


def loglogSurv3(
    y: Any,
    prob: float = 0.9,
    print: bool = True,
    title=None,
    lcol=None,
    ltype=1,
    plot: bool = False,
    **kwargs: Any,
):
    """R loglogSurv3: type III (Gumbel) tail check, x = y."""

    return _fit_loglog(y, prob, 3, print, title, lcol, ltype, plot)


__all__ = [
    "LogLogSurvFit",
    "logSurv",
    "loglogSurv",
    "loglogSurv1",
    "loglogSurv2",
    "loglogSurv3",
]
