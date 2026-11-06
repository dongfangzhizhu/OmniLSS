"""R gamlss power-polynomial smoother: pp() / gamlss.pp, plus scs().

R sources: R/gamlss/R/powerpoly.R (pp, gamlss.pp) and
R/gamlss/R/cubicSplines-10-08-12.R (scs).

pp() tags a covariate for a power polynomial: it shifts/scales x, records the
number of powers and their starting values, and gamlss.pp() fits

    y = b0 + b1 * x^p1 [+ b2 * x^p2 [+ b3 * x^p3]]

with nls(..., algorithm = "plinear"), i.e. the powers are found by
Gauss-Newton while the coefficients are profiled out by linear least squares.
The same Golub-Peyrera scheme is used here, so the fitted powers and
coefficients agree with R up to nls' convergence tolerance.

R's npoly = 3 branch evaluates the second power with coef(fit)[5] instead of
coef(fit)[6] (powerpoly.R:88), so its fitted values do not match its own
coefficients.  The bug is reproduced deliberately: the port returns R's fitted
values, and the fitted coefficients are kept in the coef field.

scs() is a thin variant of cs() that forwards spar/df to gamlss.cs and records
the NA positions; the spar branch of gamlss.cs is not ported (it depends on
smooth.spline's internal spar scaling), so only the df form can be fitted.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.optimize import least_squares

from .smoother_specs import CSSpline


@dataclass(frozen=True)
class PPSpec:
    """The object R's pp() returns: shifted/scaled covariate plus the start powers."""

    values: np.ndarray
    npoly: int
    start: list[float]
    shift: float
    scale: float
    name: str = "pp"


@dataclass(frozen=True)
class PPFit:
    """What R's gamlss.pp() returns for one backfitting step."""

    fitted_values: np.ndarray
    residuals: np.ndarray
    nl_df: float
    coef: np.ndarray
    lambda_: float
    var: np.ndarray
    name: str = "pp"


def _fp_scale(values: np.ndarray) -> tuple[float, float]:
    """R's pp() local fp.scale(): shift when x has non-positive values, decade scale."""

    if float(values.min()) <= 0.0:
        ordered = np.sort(values)
        gaps = np.diff(ordered)
        gaps = gaps[gaps > 0]
        if gaps.size == 0:
            raise ValueError("the covariate has no positive gap; pp() cannot shift it")
        shift = float(gaps.min() - values.min())
    else:
        shift = 0.0
    span = float(values.max() - values.min())
    if span <= 0:
        raise ValueError("the covariate is constant; pp() cannot scale it")
    scale = 10.0 ** (np.sign(np.log10(span)) * np.trunc(np.abs(np.log10(span))))
    return shift, float(scale)


def pp(
    x: Any,
    start: Sequence[float] | None = None,
    shift: float | None = None,
    scale: float | None = None,
) -> PPSpec:
    """R pp(): power-polynomial smoother specification."""

    values = np.asarray(x, dtype=float)
    if values.ndim != 1:
        raise ValueError("The default smoother is bivariate; you gave a matrix as an argument")
    if start is None or len(list(start)) == 0:
        raise ValueError("no starting values for the power parameter(s) is set")
    powers = [float(value) for value in start]
    if len(powers) not in (1, 2, 3):
        raise ValueError("the number of power polynomials can be 1,2 or 3")

    if shift is None or scale is None:
        auto_shift, auto_scale = _fp_scale(values)
        shift = auto_shift if shift is None else shift
        scale = auto_scale if scale is None else scale

    if shift > 0:
        warnings.warn(
            f"The origing of the x variable has been shifted by {shift}",
            UserWarning,
            stacklevel=2,
        )

    shifted = (values + float(shift)) / float(scale)
    return PPSpec(
        values=shifted,
        npoly=len(powers),
        start=powers,
        shift=float(shift),
        scale=float(scale),
    )


def _basis(values: np.ndarray, powers: np.ndarray, w: np.ndarray) -> np.ndarray:
    root_w = np.sqrt(w)
    columns = [root_w]
    for power in powers:
        with np.errstate(invalid="ignore"):
            term = np.power(values, power)
        if not np.all(np.isfinite(term)):
            raise ValueError("a power of the covariate is not finite")
        columns.append(term * root_w)
    return np.column_stack(columns)


def gamlss_pp(x: Any, y: Any, w: Any, **kwargs: Any) -> PPFit:
    """R gamlss.pp: one power-polynomial backfitting step."""

    spec = x
    if not isinstance(spec, PPSpec):
        raise TypeError("gamlss_pp needs the object returned by pp()")

    values = np.asarray(spec.values, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    if values.shape[0] != y.shape[0]:
        raise ValueError("y and the covariate have different lengths")

    target = np.sqrt(w) * y
    npoly = spec.npoly

    def residual(powers: np.ndarray) -> np.ndarray:
        design = _basis(values, powers, w)
        beta, *_ = np.linalg.lstsq(design, target, rcond=None)
        return target - design @ beta

    solution = least_squares(
        residual,
        np.asarray(spec.start, dtype=float),
        method="lm",
        xtol=1e-15,
        ftol=1e-15,
        gtol=1e-15,
        max_nfev=5000,
    )
    powers = np.asarray(solution.x, dtype=float)
    design = _basis(values, powers, w)
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    coef = np.concatenate([powers, beta])

    # R's fitted-value expressions, including the npoly = 3 coefficient typo.
    if npoly == 1:
        fitted = coef[1] + coef[2] * values ** coef[0]
    elif npoly == 2:
        fitted = coef[2] + coef[3] * values ** coef[0] + coef[4] * values ** coef[1]
    else:
        fitted = (
            coef[3]
            + coef[4] * values ** coef[0]
            + coef[4] * values ** coef[1]
            + coef[6] * values ** coef[2]
        )

    fitted = np.asarray(fitted, dtype=float)
    return PPFit(
        fitted_values=fitted,
        residuals=y - fitted,
        nl_df={1: 2.0, 2: 4.0, 3: 6.0}[npoly],
        coef=coef,
        lambda_=float("nan"),
        var=fitted,
    )


def scs(
    x: Any,
    df: float | None = None,
    spar: float | None = None,
    control: dict[str, Any] | None = None,
    **kwargs: Any,
) -> CSSpline:
    """R scs(): cubic smoothing spline tagged for gamlss.cs().

    R forwards df/spar to gamlss.cs() and records the NA positions; the spec is
    otherwise identical to cs(), so this returns a CSSpline named "scs".
    """

    from .smoother_controls import cs_control

    if np.ndim(x) == 2:
        raise ValueError("The default smoother is bivariate; you gave a matrix as an argument")
    values = np.asarray(x, dtype=float)
    if values.ndim != 1:
        raise ValueError("The default smoother is bivariate; you gave a matrix as an argument")

    settings = dict(control) if control is not None else cs_control(**kwargs)
    return CSSpline(
        name="scs",
        values=values,
        control=settings,
        df=None if df is None else float(df),
        spar=None if spar is None else float(spar),
        na_positions=tuple(int(index) for index in np.flatnonzero(np.isnan(values))),
    )


__all__ = ["PPFit", "PPSpec", "gamlss_pp", "pp", "scs"]
