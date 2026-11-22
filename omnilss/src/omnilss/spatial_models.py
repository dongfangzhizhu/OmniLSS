# SPDX-License-Identifier: GPL-3.0-or-later
"""gamlss.spatial: MRF / MRFA / gmrf / gamlss.gmrf / draw.polys / polys2polys
(D2.252).

R source: R/gamlss.spatial/R/MRF.R, MRFA.R, gmrf-for-gamlss.R,
extraFunctions.R.

The spatial model is a normal mixed model on an area factor:

    y = X beta + b + e,   b ~ N(0, sig2b * G^-1),   e ~ N(0, sig2e * W^-1)

where X is the area indicator matrix and G is the GMRF precision built from the
neighbour list (or supplied directly / via polygons).

  MRF   maximises the marginal (Q-function) likelihood over sig2e and sig2b
        with a bounded optimiser.
  MRFA  is the alternating route: iterate lambda = sig2e / sig2b, solve for a
        fixed lambda, or find the lambda matching a requested df.
  gmrf  / gamlss.gmrf expose the same two fits as a backfitting term; this
        package fits terms jointly, so the term is offered as a standalone
        callback and refused inside gamlss() with a clear error.

Verification is against live R with the spam calls stubbed out (spam is not
installed here): as.spam/diag.spam/spam are identity wrappers, and the R code
only uses them for storage.  draw.polys returns the polygon geometry and colour
values; pixel parity with R graphics is out of scope (DEVELOPMENT_PLAN P4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .spatial import polys2nb


def _levels_of(x: Any, area: Any = None) -> list[str]:
    data_levels = sorted({str(value) for value in np.asarray(x).ravel()})
    if area is None:
        return data_levels
    area_levels = sorted({str(value) for value in np.asarray(area).ravel()})
    if len(data_levels) > len(area_levels):
        raise ValueError("MRF basis dimension set too high")
    if any(level not in area_levels for level in data_levels):
        raise ValueError(
            "data contain regions that are not contained in the area specification"
        )
    return area_levels


def _codes(x: Any, levels: Sequence[str]) -> np.ndarray:
    position = {level: index for index, level in enumerate(levels)}
    return np.array(
        [position[str(value)] for value in np.asarray(x).ravel()], dtype=int
    )


def _indicator(codes: np.ndarray, levels: Sequence[str]) -> np.ndarray:
    matrix = np.zeros((codes.size, len(levels)))
    matrix[np.arange(codes.size), codes] = 1.0
    return matrix


def _precision_from_neighbour(neighbour: Any, levels: Sequence[str]) -> np.ndarray:
    mapping = (
        neighbour["nb"]
        if isinstance(neighbour, Mapping) and "nb" in neighbour
        else neighbour
    )
    names = [str(name) for name in mapping]
    if sorted(names) != sorted(levels):
        raise ValueError(
            "mismatch between neighbour/polys supplied area names and data area names"
        )
    size = len(levels)
    matrix = np.zeros((size, size))
    for name in names:
        row = list(levels).index(name)
        neighbours = list(mapping[name])
        matrix[row, row] = float(len(neighbours))
        for neighbour_index in neighbours:
            column = list(levels).index(names[int(neighbour_index) - 1])
            matrix[row, column] = -1.0
    if np.any(matrix != matrix.T):
        raise ValueError("Something wrong with auto- penalty construction")
    return matrix


def _as_precision(
    precision: Any,
    neighbour: Any,
    polys: Any,
    levels: Sequence[str],
) -> np.ndarray:
    if precision is not None:
        matrix = np.asarray(precision, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("the precision matrix is not suitable")
        if matrix.shape[0] != len(levels):
            raise ValueError("the precision matrix is not suitable")
        return matrix
    if neighbour is not None:
        return _precision_from_neighbour(neighbour, levels)
    if polys is not None:
        nb = polys2nb(polys)
        return _precision_from_neighbour(
            {"nb": dict(zip(nb.region_id, nb.neighbours))}, levels
        )
    raise ValueError(
        "precision matrix, boundary polygons and/or neighbours list must be supplied"
    )


def _gaussian_deviance(
    y: np.ndarray, fitted: np.ndarray, weights: np.ndarray, sigma: float
) -> float:
    residual = y - fitted
    return float(
        np.sum(
            weights
            * (np.log(2.0 * np.pi) + 2.0 * np.log(sigma) + residual**2 / sigma**2)
        )
    )


def _numerical_hessian(function, point: np.ndarray, step: float = 1e-3) -> np.ndarray:
    point = np.asarray(point, dtype=np.float64)
    size = point.size
    hessian = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            e_i = np.zeros(size)
            e_j = np.zeros(size)
            e_i[i] = step
            e_j[j] = step
            hessian[i, j] = (
                function(point + e_i + e_j)
                - function(point + e_i - e_j)
                - function(point - e_i + e_j)
                + function(point - e_i - e_j)
            ) / (4.0 * step * step)
    return hessian


@dataclass(frozen=True)
class SpatialFit:
    """R MRF / MRFA fitted object (both carry class MRF)."""

    fitted: np.ndarray
    df: float
    par: Mapping[str, Any]
    sig2e: float
    sig2b: float
    lam: float
    sige: float
    sigb: float
    y: np.ndarray
    x: np.ndarray
    G: np.ndarray
    var: np.ndarray
    nobs: int
    beta: np.ndarray
    weights: np.ndarray
    N: int
    method: str
    rss: float
    aic: float
    sbc: float
    deviance: float
    value_of_q: float | None = None


def MRF(
    y: Any = None,
    x: Any = None,
    precision: Any = None,
    neighbour: Any = None,
    polys: Any = None,
    area: Any = None,
    weights: Any = None,
    sig2e: float = 1.0,
    sig2b: float = 1.0,
    sig2e_fix: bool = False,
    sig2b_fix: bool = False,
    penalty: bool = False,
    delta: Sequence[float] = (0.01, 0.01),
    shift: Sequence[float] = (0.0, 0.0),
) -> SpatialFit:
    """R MRF: maximise the marginal likelihood (Q-function) over the variances."""

    if y is None or x is None:
        raise ValueError("MRF needs y and x")
    response = np.asarray(y, dtype=np.float64).ravel()
    levels = _levels_of(x, area)
    codes = _codes(x, levels)
    nfv = len(levels)
    N = response.size
    weights = (
        np.ones(N)
        if weights is None
        else np.nan_to_num(np.asarray(weights, dtype=np.float64).ravel())
    )
    nobs = int(np.sum(weights != 0))
    G = _as_precision(precision, neighbour, polys, levels)
    X = _indicator(codes, levels)
    XWX = np.diag(np.sum(weights[:, None] * X, axis=0))
    XWy = np.sum(weights[:, None] * X * response[:, None], axis=0)

    def qf(par: np.ndarray) -> float:
        first = float(np.exp(-par[0]))
        second = float(np.exp(-par[1]))
        system = first * XWX + second * G
        beta = np.linalg.solve(system, first * XWy)
        fitted = beta[codes]
        sign, logdet = np.linalg.slogdet(system)
        value = (
            -(nobs / 2.0) * np.log(2.0 * np.pi * np.exp(par[0]))
            - 0.5 * float(np.sum(np.log(np.where(weights == 0, 1e-4, weights))))
            - float(np.sum(weights * (response - fitted) ** 2)) / (2.0 * np.exp(par[0]))
            + (nfv / 2.0) * np.log(2.0 * np.pi)
            - (nfv / 2.0) * np.log(2.0 * np.pi * np.exp(par[1]))
            - float(beta @ G @ beta) / (2.0 * np.exp(par[1]))
            - 0.5 * float(logdet)
        )
        penal = (
            delta[0] * (par[0] - shift[0]) ** 2 + delta[1] * (par[1] - shift[1]) ** 2
            if penalty
            else 0.0
        )
        return -value + penal

    log_sig2e = float(np.log(sig2e))
    log_sig2b = float(np.log(sig2b))
    beta0 = np.linalg.solve(
        np.exp(-log_sig2e) * XWX + np.exp(-log_sig2b) * G,
        np.exp(-log_sig2e) * XWy,
    )
    fitted0 = beta0[codes]
    start = np.array(
        [
            np.log(float(np.sum(weights * (response - fitted0) ** 2)) / N),
            np.log(float(np.sum(beta0**2)) / N),
        ]
    )

    from scipy.optimize import minimize

    def _bounded(fun, x0, bounds):
        return minimize(
            fun, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500}
        )

    def _one_dim(fun, x0):
        return minimize(
            fun,
            np.atleast_1d(x0),
            method="L-BFGS-B",
            bounds=[(-20.0, 20.0)],
            options={"maxiter": 500},
        )

    if sig2e_fix and sig2b_fix:
        par = np.array([log_sig2e, log_sig2b])
        value = -qf(par)
        se = np.array([np.nan, np.nan])
    elif sig2e_fix:
        result = _one_dim(lambda value: qf(np.array([log_sig2e, value[0]])), start[1])
        hessian = _numerical_hessian(
            lambda value: qf(np.array([log_sig2e, value[0]])), result.x
        )
        se1 = float(np.sqrt(1.0 / hessian[0, 0])) if hessian[0, 0] > 0 else np.nan
        par = np.array([log_sig2e, float(result.x[0])])
        value = -float(result.fun)
        se = np.array([np.nan, se1])
    elif sig2b_fix:
        result = _one_dim(lambda value: qf(np.array([value[0], log_sig2b])), start[0])
        hessian = _numerical_hessian(
            lambda value: qf(np.array([value[0], log_sig2b])), result.x
        )
        se1 = float(np.sqrt(1.0 / hessian[0, 0])) if hessian[0, 0] > 0 else np.nan
        par = np.array([float(result.x[0]), log_sig2b])
        value = -float(result.fun)
        se = np.array([se1, np.nan])
    else:
        result = _bounded(qf, start, [(-20.0, 20.0), (-20.0, 20.0)])
        hessian = _numerical_hessian(qf, result.x)
        try:
            covariance = np.linalg.inv(hessian)
            se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        except np.linalg.LinAlgError:
            se = np.array([np.nan, np.nan])
        par = np.asarray(result.x, dtype=np.float64)
        value = -float(result.fun)

    beta = np.linalg.solve(
        np.exp(-par[0]) * XWX + np.exp(-par[1]) * G, np.exp(-par[0]) * XWy
    )
    fitted = beta[codes]
    sig2e_value = float(np.exp(par[0]))
    sig2b_value = float(np.exp(par[1]))
    sige = float(np.sqrt(sig2e_value))
    sigb = float(np.sqrt(sig2b_value))
    tr1 = float(beta @ G @ beta) / sig2b_value
    lam = sig2e_value / sig2b_value
    var = np.diag(np.linalg.solve(XWX + lam * G, XWX))[codes]
    rss = float(np.sum(weights * (response - fitted) ** 2))
    deviance = _gaussian_deviance(response, fitted, weights, sige)
    return SpatialFit(
        fitted=fitted,
        df=tr1,
        par={"par": par, "se": se},
        sig2e=sig2e_value,
        sig2b=sig2b_value,
        lam=lam,
        sige=sige,
        sigb=sigb,
        y=response,
        x=np.asarray(x),
        G=G,
        var=var,
        nobs=nobs,
        beta=beta,
        weights=weights,
        N=N,
        method="Q-function",
        rss=rss,
        aic=deviance + 2.0 * (tr1 + 1.0),
        sbc=deviance + np.log(nobs) * (tr1 + 1.0),
        deviance=deviance,
        value_of_q=value,
    )


def MRFA(
    y: Any = None,
    x: Any = None,
    precision: Any = None,
    neighbour: Any = None,
    polys: Any = None,
    area: Any = None,
    weights: Any = None,
    lambda_: float | None = None,
    df: float | None = None,
    start: float = 10.0,
) -> SpatialFit:
    """R MRFA: alternating fit of lambda = sig2e / sig2b (or a requested df)."""

    if y is None or x is None:
        raise ValueError("MRFA needs y and x")
    response = np.asarray(y, dtype=np.float64).ravel()
    levels = _levels_of(x, area)
    codes = _codes(x, levels)
    nfv = len(levels)
    N = response.size
    weights = (
        np.ones(N)
        if weights is None
        else np.nan_to_num(np.asarray(weights, dtype=np.float64).ravel())
    )
    nobs = int(np.sum(weights != 0))
    G = _as_precision(precision, neighbour, polys, levels)
    X = _indicator(codes, levels)
    XWX = np.diag(np.sum(weights[:, None] * X, axis=0))
    XWy = np.sum(weights[:, None] * X * response[:, None], axis=0)

    def regpen(value: float):
        system = XWX + value * G
        beta = np.linalg.solve(system, XWy)
        fitted = beta[codes]
        H = np.linalg.solve(system, XWX)
        return beta, fitted, float(np.trace(H)), np.diag(H)[codes]

    if lambda_ is not None:
        lam = float(lambda_)
        beta, fitted, edf, var = regpen(lam)
    elif df is None:
        lam = float(start)
        beta, fitted, edf, var = regpen(lam)
        for _ in range(200):
            sig2e_iter = float(np.sum(weights * (response - fitted) ** 2)) / max(
                nobs - edf, 1e-12
            )
            sig2b_iter = float(beta @ G @ beta) / max(edf, 1e-12)
            new_lam = sig2e_iter / sig2b_iter if sig2b_iter > 0 else 1e7
            new_lam = min(max(new_lam, 1e-7), 1e7)
            if abs(new_lam - lam) < 1e-7:
                lam = new_lam
                break
            lam = new_lam
            beta, fitted, edf, var = regpen(lam)
    else:
        q, r = np.linalg.qr(np.sqrt(weights)[:, None] * X)
        r_inv = np.linalg.inv(r)
        eigenvalues = np.linalg.eigvalsh(r_inv.T @ G @ r_inv)

        def edf_gap(value: float) -> float:
            return float(np.sum(1.0 / (1.0 + value * eigenvalues)) - df)

        from scipy.optimize import brentq

        low_gap = edf_gap(0.0)
        high_gap = edf_gap(1e5)
        lam = (
            1e5
            if np.sign(low_gap) == np.sign(high_gap)
            else float(brentq(edf_gap, 0.0, 1e5))
        )
        beta, fitted, edf, var = regpen(lam)

    sig2e_value = float(np.sum(weights * (response - fitted) ** 2)) / max(
        nobs - edf, 1e-12
    )
    sig2b_value = float(beta @ G @ beta) / max(edf, 1e-12)
    sige = float(np.sqrt(sig2e_value))
    sigb = float(np.sqrt(sig2b_value))
    rss = float(np.sum(weights * (response - fitted) ** 2))
    deviance = _gaussian_deviance(response, fitted, weights, sige)
    par = np.array([np.log(sig2e_value), np.log(sig2b_value)])
    return SpatialFit(
        fitted=fitted,
        df=edf,
        par={"par": par, "se": np.array([np.nan, np.nan])},
        sig2e=sig2e_value,
        sig2b=sig2b_value,
        lam=lam,
        sige=sige,
        sigb=sigb,
        y=response,
        x=np.asarray(x),
        G=G,
        var=var,
        nobs=nobs,
        beta=beta,
        weights=weights,
        N=N,
        method="altenating",
        rss=rss,
        aic=deviance + 2.0 * (edf + 1.0),
        sbc=deviance + np.log(nobs) * (edf + 1.0),
        deviance=deviance,
        value_of_q=None,
    )


@dataclass(frozen=True)
class GmrfControl:
    sig2e: float = 1.0
    sig2b: float = 1.0
    sig2e_fix: bool = False
    sig2b_fix: bool = False
    penalty: bool = False
    delta: tuple[float, float] = (0.01, 0.01)
    shift: tuple[float, float] = (0.0, 0.0)


def gmrf_control(
    sig2e: float = 1.0,
    sig2b: float = 1.0,
    sig2e_fix: bool = False,
    sig2b_fix: bool = False,
    penalty: bool = False,
    delta: Sequence[float] = (0.01, 0.01),
    shift: Sequence[float] = (0.0, 0.0),
    **kwargs: Any,
) -> GmrfControl:
    if sig2e < 0:
        sig2e = 1.0
    if sig2b < 0:
        sig2b = 1.0
    if len(delta) != 2 or any(value <= 0 for value in delta):
        delta = (0.01, 0.01)
    if len(shift) != 2:
        shift = (0.0, 0.0)
    return GmrfControl(
        sig2e=float(sig2e),
        sig2b=float(sig2b),
        sig2e_fix=bool(sig2e_fix),
        sig2b_fix=bool(sig2b_fix),
        penalty=bool(penalty),
        delta=(float(delta[0]), float(delta[1])),
        shift=(float(shift[0]), float(shift[1])),
    )


@dataclass(frozen=True)
class GmrfTerm:
    x: np.ndarray
    area: Any
    precision: np.ndarray
    adj_weight: float
    method: str
    df: float | None
    lam: float | None
    start: float
    control: GmrfControl


def gmrf(
    x: Any = None,
    precision: Any = None,
    neighbour: Any = None,
    polys: Any = None,
    area: Any = None,
    adj_weight: float = 1000.0,
    df: Any = None,
    lambda_: Any = None,
    start: float = 10.0,
    method: str = "Q",
    control: Any = None,
    **kwargs: Any,
) -> GmrfTerm:
    """R gmrf(): declare a Gaussian Markov random field term."""

    if x is None:
        raise ValueError("gmrf() needs the area factor x")
    if str(method).upper() not in {"Q", "A"}:
        raise ValueError('method must be "Q" or "A"')
    if str(method).upper() == "Q" and df is not None:
        raise ValueError('setting degrees of freedom works only with method="A"')
    levels = _levels_of(x, area)
    matrix = _as_precision(precision, neighbour, polys, levels)
    settings = (
        control if isinstance(control, GmrfControl) else gmrf_control(**dict(kwargs))
    )
    return GmrfTerm(
        x=np.asarray(x),
        area=area,
        precision=matrix,
        adj_weight=float(adj_weight),
        method=str(method).upper(),
        df=None if df is None else float(df),
        lam=None if lambda_ is None else float(lambda_),
        start=float(start),
        control=settings,
    )


@dataclass(frozen=True)
class GmrfFit:
    fitted_values: np.ndarray
    residuals: np.ndarray
    var: np.ndarray
    nl_df: float
    lam: float
    coef_smo: SpatialFit
    state: Any = None


def gamlss_gmrf(
    x: GmrfTerm = None,
    y: Any = None,
    w: Any = None,
    xeval: Any = None,
    state: Any = None,
    **kwargs: Any,
) -> GmrfFit:
    """R gamlss.gmrf: fit the gmrf term for the working response y.

    R keeps the starting variances in the gamlss frame; the caller passes them
    back through state.  The xeval prediction branch is not ported.
    """

    if not isinstance(x, GmrfTerm):
        raise TypeError("x must be a gmrf() term")
    if y is None:
        raise ValueError("y is required")
    if xeval is not None:
        raise NotImplementedError(
            "the xeval (prediction) branch of gamlss.gmrf is not ported"
        )
    response = np.asarray(y, dtype=np.float64).ravel()
    weights = (
        np.ones(response.size) if w is None else np.asarray(w, dtype=np.float64).ravel()
    )
    control = x.control
    if x.method == "Q":
        fit = MRF(
            y=response,
            x=x.x,
            area=x.area,
            precision=x.precision,
            weights=weights,
            sig2e=control.sig2e,
            sig2b=control.sig2b,
            sig2e_fix=control.sig2e_fix,
            sig2b_fix=control.sig2b_fix,
            penalty=control.penalty,
            delta=control.delta,
            shift=control.shift,
        )
        used_adjustment = False
        if np.any(np.abs(fit.beta) >= 20):
            fit = MRF(
                y=response,
                x=x.x,
                area=x.area,
                precision=x.precision,
                weights=weights * x.adj_weight,
                sig2e=control.sig2e,
                sig2b=control.sig2b,
                sig2e_fix=control.sig2e_fix,
                sig2b_fix=control.sig2b_fix,
                penalty=control.penalty,
                delta=control.delta,
                shift=control.shift,
            )
            used_adjustment = True
        new_state = (fit.sig2e, fit.sig2b, used_adjustment)
    else:
        start = float(state[0]) if state else x.start
        fit = MRFA(
            y=response,
            x=x.x,
            area=x.area,
            precision=x.precision,
            weights=weights,
            lambda_=x.lam,
            df=x.df,
            start=start,
        )
        used_adjustment = False
        if np.any(fit.beta < 1e-8):
            fit = MRFA(
                y=response,
                x=x.x,
                area=x.area,
                precision=x.precision,
                weights=weights * x.adj_weight,
                lambda_=x.lam,
                df=x.df,
                start=start,
            )
            used_adjustment = True
        new_state = (fit.lam, used_adjustment)
    return GmrfFit(
        fitted_values=fit.fitted,
        residuals=response - fit.fitted,
        var=fit.var,
        nl_df=fit.df - 2.0,
        lam=fit.sig2e / fit.sig2b if x.method == "Q" else fit.lam,
        coef_smo=fit,
        state=new_state,
    )


def polys2polys(object: Any = None, region_id: Any = None) -> dict[str, np.ndarray]:
    """R polys2polys: name polygon coordinate rings by region id.

    R takes an sp SpatialPolygons object.  sp is not a dependency here, so the
    port accepts either an already-named mapping/sequence of Nx2 arrays or a
    duck-typed object exposing polygons[i].coords (the sp shape).
    """

    if object is None:
        raise ValueError("polys2polys needs the polygon object")
    if isinstance(object, Mapping):
        return {
            str(name): np.asarray(value, dtype=np.float64)
            for name, value in object.items()
        }
    polygons = getattr(object, "polygons", None)
    if polygons is not None:
        names = region_id
        if names is None:
            names = getattr(object, "region_id", None)
        output = {}
        for index, polygon in enumerate(polygons):
            coords = np.asarray(getattr(polygon, "coords"), dtype=np.float64)
            name = str(names[index]) if names is not None else str(index + 1)
            output[name] = coords
        return output
    sequence = list(object)
    if region_id is None:
        return {
            str(index + 1): np.asarray(value, dtype=np.float64)
            for index, value in enumerate(sequence)
        }
    if len(region_id) != len(sequence):
        raise ValueError("region_id and the polygon list have different lengths")
    return {
        str(name): np.asarray(value, dtype=np.float64)
        for name, value in zip(region_id, sequence)
    }


@dataclass(frozen=True)
class PolygonPlot:
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    polygons: Mapping[str, np.ndarray]
    values: Mapping[str, float] | None
    colours: Mapping[str, tuple[float, float, float]] | None
    figure: Any = None


def _colour_map(
    values: Mapping[str, float], swap: bool
) -> dict[str, tuple[float, float, float]]:
    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    span = (high - low) or 1.0
    colours = {}
    for name, value in values.items():
        fraction = (value - low) / span
        if swap:
            fraction = 1.0 - fraction
        colours[name] = (fraction, 0.4, 1.0 - fraction)
    return colours


def draw_polys(
    polys: Any = None,
    object: Any = None,
    scheme: Any = None,
    swapcolors: bool = False,
    n_col: int = 100,
    **kwargs: Any,
) -> PolygonPlot:
    """R draw.polys: polygon map with MRF coefficients or a named value vector."""

    if not polys:
        raise ValueError("draw.polys needs the polygon list")
    bounds_x = []
    bounds_y = []
    for points in polys.values():
        array = np.asarray(points, dtype=np.float64)
        finite = array[np.all(np.isfinite(array), axis=1)]
        if finite.size:
            bounds_x.extend([finite[:, 0].min(), finite[:, 0].max()])
            bounds_y.extend([finite[:, 1].min(), finite[:, 1].max()])
    if not bounds_x:
        raise ValueError("draw.polys needs polygons with finite coordinates")
    values: dict[str, float] | None = None
    if object is not None:
        if isinstance(object, SpatialFit):
            values = {
                str(level): float(value)
                for level, value in zip(_levels_of(object.x), object.beta)
            }
        elif isinstance(object, Mapping):
            values = {str(name): float(value) for name, value in object.items()}
        elif isinstance(object, Sequence) and not isinstance(object, (str, bytes)):
            names = getattr(object, "names", None)
            if names is None:
                raise ValueError("the object do not have names")
            values = {str(name): float(value) for name, value in zip(names, object)}
        else:
            raise ValueError(
                "object class should be MRF or a vector with names matching the areas in the polys"
            )
        missing = sorted(set(values) - set(polys))
        if missing:
            raise ValueError(f"object names not present in polys: {missing}")
    colours = _colour_map(values, swapcolors) if values is not None else None
    figure = None
    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt

        figure = plt.figure()
        axes = figure.add_subplot(111)
        for name, points in polys.items():
            array = np.asarray(points, dtype=np.float64)
            colour = colours.get(name) if colours else None
            axes.fill(
                array[:, 0], array[:, 1], color=colour, edgecolor="black", linewidth=0.5
            )
        axes.set_xlim(min(bounds_x), max(bounds_x))
        axes.set_ylim(min(bounds_y), max(bounds_y))
        axes.set_aspect("equal")
    except Exception:  # pragma: no cover - matplotlib is optional
        figure = None
    return PolygonPlot(
        xlim=(float(min(bounds_x)), float(max(bounds_x))),
        ylim=(float(min(bounds_y)), float(max(bounds_y))),
        polygons={
            str(name): np.asarray(value, dtype=np.float64)
            for name, value in polys.items()
        },
        values=values,
        colours=colours,
        figure=figure,
    )


__all__ = [
    "GmrfControl",
    "GmrfFit",
    "GmrfTerm",
    "MRF",
    "MRFA",
    "PolygonPlot",
    "SpatialFit",
    "draw_polys",
    "gamlss_gmrf",
    "gmrf",
    "gmrf_control",
    "polys2polys",
]
