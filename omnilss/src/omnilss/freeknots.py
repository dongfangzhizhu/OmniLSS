# SPDX-License-Identifier: GPL-3.0-or-later
"""gamlss.add free-knot terms: fk / gamlss.fk / fitFixedKnots / fitFreeKnots /
centilesTwo (D2.250).

R source: R/gamlss.add/R/fk-gamlss.R, freeKnots-18-06-13.R, CentilesTwo.R.

The basis is a truncated-power spline:

    X = [1, x, ..., x^degree, (x - k1)^degree * I(x > k1), ...]

fitFixedKnots() is a weighted least-squares fit on that basis with the knots
held fixed (base="trun").  fitFreeKnots() optimises the knot positions to
minimise the residual sum of squares, then refits.  fk()/gamlss.fk() are the
backfitting interface R uses inside gamlss(); this package's joint design-matrix
fitter cannot re-optimise knots per outer cycle, so the term is offered as a
standalone callback (gamlss_fk) and refused inside gamlss() with a clear error.

Not ported: base="Bbase" (B-spline design matrix), lambda-penalised variants,
and the R methods (print/summary/vcov).  Each is rejected explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

_BASE_CHOICES = ("trun", "Bbase")


@dataclass(frozen=True)
class FkControl:
    """R fk.control defaults (degree 1, all.fixed FALSE, base trun)."""

    degree: int = 1
    all_fixed: bool = False
    fixed: tuple[float, ...] = ()
    base: str = "trun"


def fk_control(
    degree: int = 1,
    all_fixed: bool = False,
    fixed: Any = None,
    base: str = "trun",
    **kwargs: Any,
) -> FkControl:
    if base not in _BASE_CHOICES:
        raise ValueError(f"base must be one of {list(_BASE_CHOICES)}, got {base!r}")
    if base == "Bbase":
        raise NotImplementedError('base="Bbase" (B-spline design matrix) is not ported')
    return FkControl(
        degree=int(degree),
        all_fixed=bool(all_fixed),
        fixed=() if fixed is None else tuple(float(v) for v in np.ravel(fixed)),
        base=base,
    )


@dataclass(frozen=True)
class FkTerm:
    """R fk(): the covariate values plus start knots and control."""

    x: np.ndarray
    start: Any
    control: FkControl


def fk(x: Any = None, start: Any = None, control: Any = None, **kwargs: Any) -> FkTerm:
    """R fk(): declare a free-knot term for one covariate."""

    if x is None:
        raise ValueError("fk() needs the covariate values x")
    values = np.asarray(x, dtype=np.float64).ravel()
    if values.size == 0:
        raise ValueError("fk() needs at least one covariate value")
    if not np.all(np.isfinite(values)):
        raise ValueError("fk() rejects missing covariate values")
    if control is None:
        settings = fk_control(**dict(kwargs))
    elif isinstance(control, FkControl):
        settings = control
    elif isinstance(control, Mapping):
        settings = fk_control(**dict(control))
    else:
        raise TypeError("control must be an FkControl or a mapping")
    return FkTerm(x=values, start=start, control=settings)


def _design(x: np.ndarray, knots: np.ndarray, degree: int, fixed: tuple[float, ...]) -> np.ndarray:
    columns = [np.ones_like(x)]
    for power in range(1, degree + 1):
        columns.append(x**power)
    all_knots = np.sort(np.concatenate([np.asarray(fixed, dtype=np.float64), knots]))
    for knot in all_knots:
        columns.append(np.where(x > knot, (x - knot) ** degree, 0.0))
    return np.column_stack(columns)


@dataclass(frozen=True)
class FixedKnotFit:
    """R fitFixedKnots result (FixBreakPointsReg), reduced to the used fields."""

    fitted_values: np.ndarray
    residuals: np.ndarray
    df: int
    rss: float
    coef: np.ndarray
    sigma: float
    break_points: np.ndarray
    degree: int
    base: str
    design: np.ndarray
    deviance: float


def _gaussian_deviance(y: np.ndarray, fitted: np.ndarray, sigma: float) -> float:
    n = y.size
    rss = float(np.sum((y - fitted) ** 2))
    return float(n * np.log(2.0 * np.pi) + 2.0 * n * np.log(sigma) + rss / (sigma * sigma))


def fitFixedKnots(
    y: Any = None,
    x: Any = None,
    weights: Any = None,
    knots: Any = None,
    degree: int = 3,
    fixed: Any = None,
    base: str = "trun",
    **kwargs: Any,
) -> FixedKnotFit:
    """R fitFixedKnots: weighted least squares on a truncated-power basis."""

    if y is None or x is None:
        raise ValueError("fitFixedKnots needs y and x")
    if base not in _BASE_CHOICES:
        raise ValueError(f"base must be one of {list(_BASE_CHOICES)}, got {base!r}")
    if base == "Bbase":
        raise NotImplementedError('base="Bbase" (B-spline design matrix) is not ported')
    response = np.asarray(y, dtype=np.float64).ravel()
    covariate = np.asarray(x, dtype=np.float64).ravel()
    if response.size != covariate.size:
        raise ValueError("y and x must have the same length")
    if knots is None:
        raise ValueError("fitFixedKnots needs at least one knot")
    knot_values = np.asarray(knots, dtype=np.float64).ravel()
    fixed_values = () if fixed is None else tuple(float(v) for v in np.ravel(fixed))
    weights = (
        np.ones(response.size)
        if weights is None
        else np.asarray(weights, dtype=np.float64).ravel()
    )
    design = _design(covariate, knot_values, int(degree), fixed_values)
    root_w = np.sqrt(weights)
    coef, *_ = np.linalg.lstsq(design * root_w[:, None], response * root_w, rcond=None)
    fitted = design @ coef
    residuals = response - fitted
    rss = float(np.sum(residuals**2))
    sigma = float(rss / response.size)
    if sigma <= 0.0:
        sigma = float(np.finfo(np.float64).eps)
    return FixedKnotFit(
        fitted_values=fitted,
        residuals=residuals,
        df=int(np.linalg.matrix_rank(design)),
        rss=rss,
        coef=np.asarray(coef, dtype=np.float64),
        sigma=sigma,
        break_points=knot_values,
        degree=int(degree),
        base=base,
        design=design,
        deviance=_gaussian_deviance(response, fitted, sigma),
    )


@dataclass(frozen=True)
class FreeKnotFit(FixedKnotFit):
    """R fitFreeKnots result: the fixed-knot fit at the optimised knots."""


def fitFreeKnots(
    y: Any = None,
    x: Any = None,
    weights: Any = None,
    knots: Any = None,
    degree: int = 3,
    fixed: Any = None,
    base: str = "trun",
    **kwargs: Any,
) -> FreeKnotFit:
    """R fitFreeKnots: optimise knot positions, then refit the fixed-knot model.

    One knot uses a bounded scalar search; several use L-BFGS-B with the same
    (xmin + eps^0.25, xmax - eps^0.25) bounds as R.
    """

    if y is None or x is None:
        raise ValueError("fitFreeKnots needs y and x")
    response = np.asarray(y, dtype=np.float64).ravel()
    covariate = np.asarray(x, dtype=np.float64).ravel()
    if knots is None or np.asarray(knots).size == 0:
        raise ValueError("fitFreeKnots needs at least one starting knot")
    start = np.asarray(knots, dtype=np.float64).ravel()
    lower = float(covariate.min()) + np.finfo(np.float64).eps**0.25
    upper = float(covariate.max()) - np.finfo(np.float64).eps**0.25

    scale = max(upper - lower, np.finfo(np.float64).eps)

    def loss(candidate: Any) -> float:
        values = np.sort(lower + np.atleast_1d(np.asarray(candidate, dtype=np.float64)) * scale)
        if np.unique(values).size < values.size:
            return 1e12
        return fitFixedKnots(
            y=response, x=covariate, weights=weights, knots=values,
            degree=degree, fixed=fixed, base=base,
        ).rss

    from scipy.optimize import minimize, minimize_scalar

    if start.size == 1:
        result = minimize_scalar(loss, bounds=(0.0, 1.0), method="bounded")
        optimised = np.array([lower + float(result.x) * scale])
    else:
        # The objective is identical to R's (our rss at R's optimum matches to
        # 1e-12), but a single L-BFGS-B run from the starting knots can stop in
        # a shallower basin: on the sin(x) fixture it returned rss 10.0108 while
        # R's optim reached 9.9877.  A deterministic multi-start removes that
        # sensitivity; the reported optimum is the best of the runs.
        base_grid = np.arange(1, start.size + 1) / (start.size + 1)
        candidates = []
        for shift in (-0.15, 0.0, 0.15):
            candidate = np.clip(base_grid + shift, 0.02, 0.98)
            if np.all(np.diff(candidate) > 0):
                candidates.append(candidate)
        best_value = None
        best_point = None
        for candidate in candidates:
            result = minimize(
                loss, candidate, method="L-BFGS-B",
                bounds=[(1e-3, 1.0 - 1e-3)] * start.size,
                options={"eps": 5e-4, "ftol": 1e-14, "maxiter": 200},
            )
            if best_value is None or result.fun < best_value:
                best_value = float(result.fun)
                best_point = np.asarray(result.x, dtype=np.float64)
        optimised = np.sort(lower + best_point * scale)
    optimised = np.sort(optimised)
    fit = fitFixedKnots(
        y=response, x=covariate, weights=weights, knots=optimised,
        degree=degree, fixed=fixed, base=base,
    )
    return FreeKnotFit(
        fitted_values=fit.fitted_values,
        residuals=fit.residuals,
        df=fit.df + int(optimised.size),
        rss=fit.rss,
        coef=fit.coef,
        sigma=fit.sigma,
        break_points=optimised,
        degree=fit.degree,
        base=fit.base,
        design=fit.design,
        deviance=fit.deviance,
    )


@dataclass(frozen=True)
class FkFit:
    """R gamlss.fk's return list, reduced to the fields gamlss consumes."""

    fitted_values: np.ndarray
    residuals: np.ndarray
    nl_df: int
    lam: np.ndarray
    coef_smo: Any
    var: Any = None


def gamlss_fk(
    x: FkTerm = None,
    y: Any = None,
    w: Any = None,
    xeval: Any = None,
    first_iter: bool = True,
    **kwargs: Any,
) -> FkFit:
    """R gamlss.fk: fit the fk term for the working response y.

    R keeps an iteration counter in the gamlss frame: the first call uses fixed
    knots, later calls re-optimise them.  Without that frame the caller passes
    first_iter explicitly (default True, the first backfitting pass).
    """

    if not isinstance(x, FkTerm):
        raise TypeError("x must be an fk() term")
    if y is None:
        raise ValueError("y is required")
    if xeval is not None:
        raise NotImplementedError("the xeval (prediction) branch of gamlss.fk is not ported")
    yv = np.asarray(y, dtype=np.float64).ravel()
    if yv.size != x.x.size:
        raise ValueError("the fk term and y must have the same length")
    control = x.control
    start = x.start if x.start is not None else np.array([float(np.median(x.x))])
    if control.all_fixed or first_iter:
        fit = fitFixedKnots(
            y=yv, x=x.x, weights=w, knots=start,
            degree=control.degree, fixed=control.fixed, base=control.base,
        )
    else:
        fit = fitFreeKnots(
            y=yv, x=x.x, weights=w, knots=start,
            degree=control.degree, fixed=control.fixed, base=control.base,
        )
    return FkFit(
        fitted_values=fit.fitted_values,
        residuals=yv - fit.fitted_values,
        nl_df=fit.df - 1,
        lam=fit.break_points,
        coef_smo=fit,
    )


@dataclass(frozen=True)
class CentileSurface:
    """centilesTwo result: the centile value on a two-dimensional grid."""

    grid1: np.ndarray
    grid2: np.ndarray
    values: np.ndarray
    mask: np.ndarray
    cent: float
    figure: Any = None


def centilesTwo(
    object: Any = None,
    grid_x1: Any = None,
    grid_x2: Any = None,
    x1_name: str = None,
    x2_name: str = None,
    cent: float = 0.05,
    dist: float = 0.01,
    points: bool = True,
    image: bool = False,
    **kwargs: Any,
) -> CentileSurface:
    """R centilesTwo: centile surface of one centile over a two-covariate grid.

    The mask follows mgcv's exclude.too.far *intent* (scaled distance from the
    observed covariate cloud) with a nearest-neighbour distance in the unit
    square, not mgcv's triangulation; the difference is documented rather than
    hidden.  Plotting is optional and uses matplotlib when available.
    """

    from .prediction import predict_params

    if object is None or not hasattr(object, "family") or not hasattr(object, "call"):
        raise ValueError("the object should be a gamlss object")
    call = object.call or {}
    data = call.get("data")
    if data is None:
        raise ValueError("the gamlss object needs a data argument")
    if x1_name is None or x2_name is None:
        raise ValueError("x1_name and x2_name are required")
    if x1_name not in data:
        raise ValueError(f"{x1_name} is not in the data")
    if x2_name not in data:
        raise ValueError(f"{x2_name} is not in the data")
    if grid_x1 is None or grid_x2 is None:
        raise ValueError("grid_x1 and grid_x2 are required")

    g1 = np.asarray(grid_x1, dtype=np.float64).ravel()
    g2 = np.asarray(grid_x2, dtype=np.float64).ravel()
    mesh = np.meshgrid(g1, g2, indexing="ij")
    newdata = {x1_name: mesh[0].ravel(), x2_name: mesh[1].ravel()}
    params = predict_params(object, newdata)
    fam = object.family
    args = [np.asarray(params[name], dtype=np.float64) for name in fam.parameters]
    values = np.asarray(fam.q(float(cent), *args), dtype=np.float64).reshape(mesh[0].shape)

    observed1 = np.asarray(data[x1_name], dtype=np.float64)
    observed2 = np.asarray(data[x2_name], dtype=np.float64)
    scale1 = max(float(observed1.max() - observed1.min()), np.finfo(float).eps)
    scale2 = max(float(observed2.max() - observed2.min()), np.finfo(float).eps)
    scaled1 = (observed1 - observed1.min()) / scale1
    scaled2 = (observed2 - observed2.min()) / scale2
    query1 = (mesh[0].ravel() - observed1.min()) / scale1
    query2 = (mesh[1].ravel() - observed2.min()) / scale2
    nearest = np.full(query1.size, np.inf)
    for a, b in zip(scaled1, scaled2):
        nearest = np.minimum(nearest, np.hypot(query1 - a, query2 - b))
    mask = (nearest > float(dist)).reshape(mesh[0].shape)
    values = np.where(mask, np.nan, values)

    figure = None
    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt

        figure = plt.figure()
        axes = figure.add_subplot(111)
        if image:
            axes.imshow(
                values, origin="lower", aspect="auto",
                extent=[g1.min(), g1.max(), g2.min(), g2.max()],
            )
        else:
            axes.contour(g1, g2, values.T)
        if points:
            axes.plot(observed1, observed2, ".", color="black", markersize=2)
        axes.set_xlabel(x1_name)
        axes.set_ylabel(x2_name)
    except Exception:  # pragma: no cover - matplotlib is optional
        figure = None
    return CentileSurface(
        grid1=g1, grid2=g2, values=values, mask=mask, cent=float(cent), figure=figure
    )


__all__ = [
    "CentileSurface",
    "FkControl",
    "FkFit",
    "FkTerm",
    "FixedKnotFit",
    "FreeKnotFit",
    "centilesTwo",
    "fitFixedKnots",
    "fitFreeKnots",
    "fk",
    "fk_control",
    "gamlss_fk",
]
