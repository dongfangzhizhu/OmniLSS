# SPDX-License-Identifier: GPL-3.0-or-later
"""R-aligned LOESS smoother (`gamlss::lo`, `gamlss::lo.control`, `gamlss::gamlss.lo`).

R source reference:
- file: ``gamlss/R/lo_new.R``
- functions: ``lo``, ``lo.control``, ``gamlss.lo``

Implements weighted local polynomial regression (degree 0, 1, or 2) with
tricube kernel weights, optional Tukey biweight robustness iterations
(``family="symmetric"``), exact hat-matrix trace (``trace_hat``,
``nl_df = trace_hat - 1``), ``enp_target`` span solving, and out-of-sample
evaluation via ``xeval``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class LoResult:
    """Backward-compatible result returned by ``lo(x, y, span=...)``."""

    x: np.ndarray
    fitted: np.ndarray
    span: float


@dataclass(frozen=True)
class LoSpec:
    """Specification returned by ``lo(...)`` when used as a GAMLSS smoother term."""

    x: np.ndarray
    formula: str
    control: dict[str, Any]
    xnames: tuple[str, ...] = ("x",)
    data: Mapping[str, Any] | None = field(default=None, repr=False)
    name: str = "lo"


@dataclass(frozen=True)
class GamlssLoFit:
    """Backfitting return payload matching R ``gamlss.lo()``."""

    fitted_values: np.ndarray
    residuals: np.ndarray
    nl_df: float
    trace_hat: float
    lambda_: float
    span: float
    degree: int
    coefSmo: dict[str, Any]
    var: np.ndarray
    name: str = "lo"


def lo_control(
    span: float = 0.75,
    enp_target: float | None = None,
    degree: int = 2,
    parametric: bool = False,
    drop_square: bool = False,
    normalize: bool = True,
    family: str = "gaussian",
    method: str = "loess",
    surface: str = "direct",
    statistics: str = "exact",
    trace_hat: str = "exact",
    cell: float = 0.2,
    iterations: int = 4,
    iterTrace: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """R ``gamlss::lo.control`` settings dictionary."""
    fam = str(family).lower()
    if fam not in {"gaussian", "symmetric"}:
        raise ValueError("family must be 'gaussian' or 'symmetric'")
    deg = int(degree)
    if deg not in {0, 1, 2}:
        raise ValueError("degree must be 0, 1, or 2")
    sp = float(span)
    if sp <= 0.0:
        raise ValueError("span must be positive")
    return {
        "span": sp,
        "enp.target": None if enp_target is None else float(enp_target),
        "degree": deg,
        "parametric": bool(parametric),
        "drop.square": bool(drop_square),
        "normalize": bool(normalize),
        "family": fam,
        "method": str(method),
        "surface": str(surface),
        "statistics": str(statistics),
        "trace.hat": str(trace_hat),
        "cell": float(cell),
        "iterations": int(iterations),
        "iterTrace": bool(iterTrace),
    }


def _trimmed_sd(col: np.ndarray) -> float:
    """R ``loess`` 10%-trimmed standard deviation used for multi-predictor normalization."""
    n = col.size
    if n <= 2:
        sd = float(np.std(col, ddof=1)) if n == 2 else 1.0
        return sd if sd > 0.0 else 1.0
    sorted_col = np.sort(col)
    trim = int(np.floor(0.1 * n))
    core = sorted_col[trim : n - trim] if n - 2 * trim >= 2 else sorted_col
    sd = float(np.std(core, ddof=1))
    return sd if sd > 0.0 else 1.0


def _to_design_2d(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"loess predictors must be 1D or 2D, got shape {arr.shape}")


def _local_poly_basis(dx: np.ndarray, degree: int) -> np.ndarray:
    """Local Taylor polynomial basis around dx = x - x0 for degree 0, 1, or 2."""
    n, d = dx.shape
    cols: list[np.ndarray] = [np.ones(n, dtype=np.float64)]
    if degree >= 1:
        for j in range(d):
            cols.append(dx[:, j])
    if degree >= 2:
        for j in range(d):
            for k in range(j, d):
                cols.append(dx[:, j] * dx[:, k])
    return np.column_stack(cols)


def _loess_eval_matrix(
    X_train: np.ndarray,
    X_query: np.ndarray,
    weights: np.ndarray,
    span: float,
    degree: int,
) -> np.ndarray:
    """Compute the smoother matrix ``S`` where ``y_hat = S @ y``."""
    n, d = X_train.shape
    m = X_query.shape[0]
    S = np.zeros((m, n), dtype=np.float64)

    # Number of neighbors q follows R/Cleveland's loess definition
    if span <= 1.0:
        q = min(n, max(1, int(np.floor(span * n))))
        scale_factor = 1.0
    else:
        q = n
        scale_factor = span ** (1.0 / d)

    for i in range(m):
        dx = X_train - X_query[i]
        dist = np.linalg.norm(dx, axis=1)
        if q < n:
            # q-th smallest distance (1-indexed -> q - 1)
            h = float(np.partition(dist, q - 1)[q - 1])
        else:
            h = float(np.max(dist)) * scale_factor

        if h <= 0.0:
            # Degenerate identical predictor values: weighted mean
            w_eff = np.where(dist == 0.0, weights, 0.0)
            sw = float(np.sum(w_eff))
            if sw > 0.0:
                S[i] = w_eff / sw
            else:
                S[i, np.argmin(dist)] = 1.0
            continue

        u = dist / h
        # R loess tricube kernel: (1 - u^3)^3 for u < 1 (exact zero at u >= 1)
        tricube = np.where(u < 1.0, (1.0 - u**3) ** 3, 0.0)
        eff_w = weights * tricube
        active = eff_w > 0.0
        if not np.any(active):
            S[i, np.argmin(dist)] = 1.0
            continue

        Phi = _local_poly_basis(dx[active], degree)
        w_act = eff_w[active]
        sqrt_w = np.sqrt(w_act)
        A = Phi * sqrt_w[:, None]
        # Solve min ||A beta - sqrt_w y|| via SVD pseudoinverse; row 0 of pinv(A)
        # gives the intercept (fitted value at dx = 0).
        u_svd, s_svd, vt_svd = np.linalg.svd(A, full_matrices=False)
        tol = (
            max(A.shape) * np.finfo(np.float64).eps * (s_svd[0] if s_svd.size else 1.0)
        )
        s_inv = np.where(s_svd > tol, 1.0 / s_svd, 0.0)
        # e_0^T V Sigma^{-1} U^T diag(sqrt_w)
        e0_v_sinv = vt_svd[:, 0] * s_inv
        row_weights = (u_svd @ e0_v_sinv) * sqrt_w
        S[i, active] = row_weights

    return S


def _fit_loess_core(
    X_raw: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    *,
    span: float,
    degree: int,
    normalize: bool,
    family: str,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Core weighted local polynomial fit returning (fitted, S, scales, robust_w, X_norm)."""
    n, d = X_raw.shape
    if normalize and d > 1:
        scales = np.array(
            [_trimmed_sd(X_raw[:, j]) for j in range(d)], dtype=np.float64
        )
        X_norm = X_raw / scales[None, :]
    else:
        scales = np.ones(d, dtype=np.float64)
        X_norm = X_raw

    robust_w = np.ones(n, dtype=np.float64)
    n_passes = max(1, int(iterations)) if family == "symmetric" else 1
    S = np.zeros((n, n), dtype=np.float64)
    fv = np.zeros(n, dtype=np.float64)

    for pass_idx in range(n_passes):
        eff_w = w * robust_w
        S = _loess_eval_matrix(X_norm, X_norm, eff_w, span=span, degree=degree)
        fv = S @ y
        if pass_idx + 1 < n_passes:
            abs_res = np.abs(y - fv)
            mad = float(np.median(abs_res))
            if mad <= 0.0:
                break
            u_r = abs_res / (6.0 * mad)
            robust_w = np.where(u_r < 1.0, (1.0 - u_r**2) ** 2, 0.0)

    return fv, S, scales, robust_w, X_norm


def _solve_span_for_enp(
    X_raw: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    *,
    enp_target: float,
    degree: int,
    normalize: bool,
) -> float:
    """Find span in (span_min, 1.0] whose trace(S) matches ``enp_target``."""
    n, d = X_raw.shape
    p = 1 + (d if degree >= 1 else 0) + (d * (d + 1) // 2 if degree >= 2 else 0)
    span_min = min(0.99, max((p + 2.0) / max(n, 1), 0.05))
    span_max = 1.0

    def _trace_diff(sp: float) -> float:
        _, S, _, _, _ = _fit_loess_core(
            X_raw,
            y,
            w,
            span=sp,
            degree=degree,
            normalize=normalize,
            family="gaussian",
            iterations=1,
        )
        return float(np.trace(S)) - float(enp_target)

    f_lo = _trace_diff(span_min)
    f_hi = _trace_diff(span_max)
    if f_lo <= 0.0:
        return span_min
    if f_hi >= 0.0:
        return span_max
    return float(brentq(_trace_diff, span_min, span_max, xtol=1e-4, maxiter=40))


def _extract_x_from_spec(x: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Unpack either a ``LoSpec``, a dict, or a raw covariate array."""
    if isinstance(x, LoSpec):
        return _to_design_2d(x.x), dict(x.control)
    if isinstance(x, Mapping) and "x" in x:
        ctrl = dict(x.get("control", {}))
        return _to_design_2d(x["x"]), ctrl
    return _to_design_2d(x), {}


def gamlss_lo(
    x: Any,
    y: Any,
    w: Any = None,
    xeval: Any = None,
    *,
    span: float | None = None,
    enp_target: float | None = None,
    degree: int | None = None,
    normalize: bool | None = None,
    family: str | None = None,
    iterations: int | None = None,
    control: Mapping[str, Any] | None = None,
) -> GamlssLoFit | np.ndarray:
    """R reference: ``gamlss/R/lo_new.R::gamlss.lo``.

    Backfitting smoother callback for LOESS terms. When ``xeval`` is ``None``
    returns a :class:`GamlssLoFit` (carrying ``fitted_values``, ``residuals``,
    ``nl_df = trace_hat - 1``, ``lambda_ = span``, ``coefSmo``); when ``xeval``
    is provided returns the predicted vector at ``xeval``.
    """
    X_raw, spec_ctrl = _extract_x_from_spec(x)
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    n = y_arr.size
    if X_raw.shape[0] != n:
        raise ValueError(f"x has {X_raw.shape[0]} rows but y has length {n}")
    if w is None:
        w_arr = np.ones(n, dtype=np.float64)
    else:
        w_arr = np.asarray(w, dtype=np.float64).ravel()
        if w_arr.size != n:
            raise ValueError(f"w has length {w_arr.size}, expected {n}")
        if np.any(w_arr < 0.0) or not np.all(np.isfinite(w_arr)):
            raise ValueError("weights w must be finite and non-negative")

    merged = lo_control()
    merged.update(spec_ctrl)
    if control is not None:
        merged.update(dict(control))
    if span is not None:
        merged["span"] = float(span)
    if enp_target is not None:
        merged["enp.target"] = float(enp_target)
    if degree is not None:
        merged["degree"] = int(degree)
    if normalize is not None:
        merged["normalize"] = bool(normalize)
    if family is not None:
        merged["family"] = str(family).lower()
    if iterations is not None:
        merged["iterations"] = int(iterations)

    deg = int(merged["degree"])
    norm_flag = bool(merged["normalize"])
    fam = str(merged["family"]).lower()
    iters = int(merged["iterations"])
    target_enp = merged.get("enp.target")
    eff_span = float(merged["span"])
    if target_enp is not None:
        eff_span = _solve_span_for_enp(
            X_raw,
            y_arr,
            w_arr,
            enp_target=float(target_enp),
            degree=deg,
            normalize=norm_flag,
        )

    fv, S, scales, robust_w, X_norm = _fit_loess_core(
        X_raw,
        y_arr,
        w_arr,
        span=eff_span,
        degree=deg,
        normalize=norm_flag,
        family=fam,
        iterations=iters,
    )

    if xeval is not None:
        X_eval_raw = _to_design_2d(xeval)
        if X_eval_raw.shape[1] != X_raw.shape[1]:
            raise ValueError(
                f"xeval has {X_eval_raw.shape[1]} columns, expected {X_raw.shape[1]}"
            )
        X_eval_norm = X_eval_raw / scales[None, :]
        S_eval = _loess_eval_matrix(
            X_norm, X_eval_norm, w_arr * robust_w, span=eff_span, degree=deg
        )
        return S_eval @ y_arr

    trace_hat = float(np.trace(S))
    nl_df = trace_hat - 1.0
    residuals = y_arr - fv
    # Pointwise leverage-based variance proxy: sum_j S_{ij}^2 / max(w_j, eps)
    inv_w = np.where(
        w_arr > 0.0, 1.0 / np.maximum(w_arr, np.finfo(np.float64).eps), 0.0
    )
    var = np.sum((S**2) * inv_w[None, :], axis=1)

    coef_smo = {
        "x": X_raw,
        "y": y_arr,
        "weights": w_arr,
        "robust_weights": robust_w,
        "scales": scales,
        "fitted": fv,
        "residuals": residuals,
        "trace.hat": trace_hat,
        "enp": trace_hat - 1.0,
        "pars": {
            "span": eff_span,
            "degree": deg,
            "family": fam,
            "normalize": norm_flag,
        },
    }
    return GamlssLoFit(
        fitted_values=fv,
        residuals=residuals,
        nl_df=nl_df,
        trace_hat=trace_hat,
        lambda_=eff_span,
        span=eff_span,
        degree=deg,
        coefSmo=coef_smo,
        var=var,
    )


def lo(
    x: Any,
    y: Any = None,
    span: float = 0.5,
    *,
    data: Mapping[str, Any] | None = None,
    control: Mapping[str, Any] | None = None,
    degree: int = 2,
    df: float | None = None,
    **kwargs: Any,
) -> LoResult | LoSpec:
    """R reference: ``gamlss/R/lo_new.R::lo``.

    Dual-mode interface:
    - Diagnostic mode ``lo(x, y, span=0.5)`` (existing OmniLSS contract):
      returns :class:`LoResult` sorted by ``x``.
    - Specification mode ``lo("~x", data=df, span=0.75)`` or ``lo(x_array)``
      (when ``y is None``): returns :class:`LoSpec` suitable for ``gamlss_lo``.
    """
    if y is None:
        ctrl = lo_control(span=span, enp_target=df, degree=degree, **kwargs)
        if control is not None:
            ctrl.update(dict(control))
        if isinstance(x, str):
            formula_str = x.strip()
            if not formula_str.startswith("~"):
                raise ValueError("lo() formula must start with '~'")
            if data is None:
                raise ValueError(
                    "lo(formula, data=...) requires data when x is a formula"
                )
            rhs = formula_str[1:].strip()
            var_names = tuple(tok.strip() for tok in rhs.split("+") if tok.strip())
            if not var_names:
                raise ValueError("lo() formula has no predictor variables")
            cols = [
                np.asarray(data[name], dtype=np.float64).ravel() for name in var_names
            ]
            X_mat = np.column_stack(cols)
            return LoSpec(
                x=X_mat, formula=formula_str, control=ctrl, xnames=var_names, data=data
            )
        X_mat = _to_design_2d(x)
        names = tuple(f"x{i + 1}" for i in range(X_mat.shape[1]))
        return LoSpec(
            x=X_mat,
            formula="~" + "+".join(names),
            control=ctrl,
            xnames=names,
            data=data,
        )

    x_arr = np.asarray(x, dtype=np.float64).ravel()
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    if x_arr.size == 0 or y_arr.size == 0:
        raise ValueError("x and y must be non-empty")
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same length")
    if not (0.0 < float(span) <= 1.0):
        raise ValueError("span must be in (0, 1]")

    n = x_arr.size
    k = max(2, int(np.ceil(span * n)))
    order = np.argsort(x_arr)
    xs = x_arr[order]
    ys = y_arr[order]
    fitted = np.empty_like(ys)

    for i in range(n):
        d = np.abs(xs - xs[i])
        h = np.partition(d, k - 1)[k - 1]
        if h <= 0:
            fitted[i] = ys[i]
            continue
        u = d / h
        w_loc = np.where(u < 1.0, (1.0 - u**3) ** 3, 0.0)
        sw = np.sum(w_loc)
        fitted[i] = ys[i] if sw <= 0 else np.sum(w_loc * ys) / sw

    return LoResult(x=xs, fitted=fitted, span=float(span))


__all__ = [
    "GamlssLoFit",
    "LoResult",
    "LoSpec",
    "gamlss_lo",
    "lo",
    "lo_control",
]
