"""R gamlss fractional-polynomial smoother: bfp(), fp(), gamlss.fp().

R source: R/gamlss/R/fracpoly.R.

fp() tags a covariate with the shifted/scaled values and the number of
fractional-polynomial terms; bfp() builds the basis for a given power vector;
gamlss.fp() searches the power grid [-2, -1, -0.5, 0, 0.5, 1, 2, 3] for the
combination with the smallest weighted residual sum of squares, refits by
weighted least squares and returns the fitted values, residuals, leverage-based
variance, nl.df and the selected powers.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Any

import numpy as np

from .family_groups import _hat_WX

_POWERS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0)


def _fp_scale(x: np.ndarray) -> tuple[float, float]:
    """R's fp.scale(): shift for non-positive x and a power-of-ten scale."""

    if np.min(x) <= 0:
        sorted_x = np.sort(x)
        gaps = sorted_x[1:] - sorted_x[:-1]
        positive = gaps[gaps > 0]
        shift = float(np.min(positive) - np.min(x)) if positive.size else 0.0
    else:
        shift = 0.0
    spread = float(np.max(x) - np.min(x))
    if spread == 0:
        scale = 1.0
    else:
        log10 = np.log10(spread)
        scale = float(10.0 ** (np.sign(log10) * np.trunc(np.abs(log10))))
    return shift, scale


def _basis(x: np.ndarray, powers: Any) -> np.ndarray:
    """R's bfp(): columns x^p, with log(x) for p=0 and log(x)*prev for repeats."""

    powers = np.atleast_1d(np.asarray(powers, dtype=float))
    values = np.asarray(x, dtype=float)
    columns = np.empty((values.shape[0], powers.shape[0]), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        first = np.log(values) if powers[0] == 0 else values ** powers[0]
    columns[:, 0] = first
    previous = first
    for index in range(1, powers.shape[0]):
        if powers[index] == powers[index - 1]:
            with np.errstate(divide="ignore", invalid="ignore"):
                column = np.log(values) * previous
        elif powers[index] == 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                column = np.log(values)
        else:
            column = values ** powers[index]
        columns[:, index] = column
        previous = column
    return columns


def bfp(x: Any, powers: Any = (1.0, 2.0), shift: float | None = None, scale: float | None = None) -> np.ndarray:
    """R bfp(): fractional-polynomial basis, optionally shifting/scaling x first.

    When shift or scale is not supplied, R computes them from x (fp.scale).
    """

    values = np.asarray(x, dtype=float)
    if shift is None or scale is None:
        computed_shift, computed_scale = _fp_scale(values)
        shift = computed_shift if shift is None else float(shift)
        scale = computed_scale if scale is None else float(scale)
    transformed = (values + shift) / scale
    return _basis(transformed, powers)


@dataclass(frozen=True)
class FPSpec:
    """The object R's fp() returns: the tagged covariate and its settings."""

    name: str
    values: np.ndarray
    npoly: int
    shift: float
    scale: float
    df: float
    smooth_class: str = "smooth"
    na_positions: tuple[int, ...] = ()


def fp(x: Any, npoly: int = 2, shift: float | None = None, scale: float | None = None, name: str | None = None) -> FPSpec:
    """R fp(): tag a covariate for fractional-polynomial fitting."""

    values = np.asarray(x, dtype=float)
    if values.ndim != 1:
        raise ValueError("The default smoother is bivariate; you gave a matrix as an argument")
    if npoly not in (1, 2, 3):
        raise ValueError("the number of fractional polynomials can be 1,2 or 3")
    if shift is None or scale is None:
        computed_shift, computed_scale = _fp_scale(values)
        shift = computed_shift if shift is None else float(shift)
        scale = computed_scale if scale is None else float(scale)
    if shift > 0:
        import warnings

        warnings.warn(f"The origing of the x variable has been shifted by  {shift}\n", UserWarning, stacklevel=2)
    transformed = (values + shift) / scale
    return FPSpec(
        name=name or "fp",
        values=transformed,
        npoly=int(npoly),
        shift=float(shift),
        scale=float(scale),
        df=float(2 * npoly),
        na_positions=tuple(int(index) for index in np.flatnonzero(np.isnan(transformed))),
    )


@dataclass(frozen=True)
class FPFit:
    """What R's gamlss.fp() returns for one backfitting step."""

    fitted_values: np.ndarray
    residuals: np.ndarray
    var: np.ndarray
    nl_df: float
    powers: np.ndarray
    coefficients: np.ndarray
    x: np.ndarray
    deviance: float


def gamlss_fp(x: Any, y: Any, w: Any, xeval: Any = None, **kwargs: Any):
    """R gamlss.fp: one fractional-polynomial backfitting step.

    Searches all non-decreasing power combinations of length npoly, picks the
    one with the smallest weighted residual sum of squares, then refits by
    weighted least squares with an intercept.

    Documented deviation: R predicts from rows appended to the stored values;
    the Python version evaluates the fitted polynomial at the supplied grid
    using the same shift/scale transformation.
    """

    if not isinstance(x, FPSpec):
        raise TypeError("gamlss_fp needs the object returned by fp()")
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    values = np.asarray(x.values, dtype=float)
    if xeval is not None:
        values = values[: y.shape[0]]
    if y.shape[0] != values.shape[0]:
        raise ValueError("y and the covariate have different lengths")

    npoly = x.npoly
    if npoly not in (1, 2, 3):
        raise ValueError("the number of fractional polynomials can be 1, 2 or 3")

    best_deviance = np.inf
    best_powers: tuple[float, ...] | None = None
    for combo in combinations_with_replacement(_POWERS, npoly):
        design = np.column_stack([np.ones(values.shape[0]), _basis(values, combo)])
        weighted = design * np.sqrt(w)[:, None]
        beta = np.linalg.lstsq(weighted, np.sqrt(w) * y, rcond=None)[0]
        deviance = float(np.sum(w * (y - design @ beta) ** 2))
        if deviance < best_deviance:
            best_deviance = deviance
            best_powers = combo

    if best_powers is None:  # pragma: no cover - defensive
        raise ValueError("no fractional polynomial could be fitted")

    design = np.column_stack([np.ones(values.shape[0]), _basis(values, best_powers)])
    weighted = design * np.sqrt(w)[:, None]
    beta = np.linalg.lstsq(weighted, np.sqrt(w) * y, rcond=None)[0]
    fitted = design @ beta
    residuals = y - fitted

    if xeval is not None:
        grid = (np.asarray(xeval, dtype=float) + x.shift) / x.scale
        prediction_design = np.column_stack([np.ones(grid.shape[0]), _basis(grid, best_powers)])
        return np.asarray(prediction_design @ beta, dtype=float)

    hat_matrix = np.sum(
        (weighted @ np.linalg.pinv(weighted.T @ weighted)) * weighted, axis=1
    )
    var = (hat_matrix - _hat_WX(w, np.ones_like(w))) / w

    return FPFit(
        fitted_values=fitted,
        residuals=residuals,
        var=var,
        nl_df=float(x.df),
        powers=np.asarray(best_powers, dtype=float),
        coefficients=beta,
        x=values,
        deviance=best_deviance,
    )


__all__ = ["FPFit", "FPSpec", "bfp", "fp", "gamlss_fp"]
