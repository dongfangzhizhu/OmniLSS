"""R gamlss P-spline variants: pbm / pbo / pbp / pbz / pbc / pvc.

R sources: R/gamlss/R/{pbm,pbo_original,pbp,pb_goingtozero,pbc,pvc}.R.

All six spec constructors build the same truncated-power B-spline basis as
pb() and a difference penalty; the differences are the inter bookkeeping, the
df convention, an extra penalty (pbz), a cyclic basis/penalty (pbc), the stored
covariate (zeros vs. the real x) and pvc's `by` handling.  The fitting helpers
share gamlss.pb's penalised system (X'WX + lambda D'D), with pbm's asymmetric
monotonicity penalty and pbz's extra first-difference penalty added where R
uses them.

Documented limits:
* pvc is implemented for by=None (R's varying-coefficient interaction with a
  by-variable/factor is not ported).
* lambda estimation by ML/GAIC/GCV/df delegates to gamlss_pb; the variants'
  GCV denominators and df root searches differ slightly from pb's, so those
  paths are documented as approximate.  The fixed-lambda path (R's case 1) is
  exact and verified against live R for all six smoothers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline

from .family_groups import _hat_WX
from .smoother_controls import (
    pbc_control,
    pbm_control,
    pbo_control,
    pbp_control,
    pbz_control,
    pvc_control,
)
from .smoother_fits import SmootherFit, _regpen, gamlss_pb
from .smoother_specs import PBSpline, bbase, cbase, difference_matrix

_LAMBDA_MIN = 1.0e-7
_LAMBDA_MAX = 1.0e7


@dataclass(frozen=True)
class PSplineVariant:
    """The covariate object a variant spec constructor returns."""

    name: str
    X: np.ndarray
    D: np.ndarray
    D1: np.ndarray | None
    control: dict[str, Any]
    df: float | None
    lambda_: float | None
    max_df: float
    values: np.ndarray
    knots: np.ndarray
    zero_covariate: bool
    subtract_linear_hat: bool = True
    nl_df_offset: float = 2.0
    mono: str | None = None
    by: Any = None


def _inter_rules(
    values: np.ndarray, control: dict[str, Any], *, distinct: bool, short_at: int = 99
) -> int:
    """R's inter bookkeeping: short samples -> 10, then optionally distinct."""

    inter = int(control.get("inter", 20))
    if values.shape[0] < short_at:
        inter = 10
    if distinct:
        n_distinct = int(np.unique(values).shape[0])
        if n_distinct <= inter:
            inter = n_distinct
    return inter


def _pb_basis(
    values: np.ndarray,
    control: dict[str, Any],
    *,
    distinct: bool,
    short_at: int = 99,
    expand: bool = True,
    cyclic: bool = False,
):
    inter = _inter_rules(values, control, distinct=distinct, short_at=short_at)
    control = {**control, "inter": inter}
    xl = float(np.min(values))
    xr = float(np.max(values))
    if expand:
        xmax = xr + 0.01 * (xr - xl)
        xmin = xl - 0.01 * (xr - xl)
    else:
        xmax, xmin = xr, xl
    degree = int(control.get("degree", 3))
    if cyclic:
        design, knots = cbase(values, xl, xr, inter, degree), np.asarray([])
    else:
        design, knots = bbase(values, xmin, xmax, inter, degree)
    order = int(control.get("order", 2))
    penalty = difference_matrix(design.shape[1], order)
    return design, penalty, control, knots


def _df_plus_two(df: float | None, columns: int):
    if df is None:
        return None
    import warnings

    if df > columns - 2:
        warnings.warn(
            "The df's exceed the number of columns of the design matrix\n   they are set to 3",
            UserWarning,
            stacklevel=3,
        )
        df = 3
    if df < 0:
        warnings.warn("the extra df's are set to 0", UserWarning, stacklevel=3)
        return 2.0
    return float(df) + 2.0


def _as_values(x: Any) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    if values.ndim != 1:
        raise ValueError("x is a matric declare it as a vector")
    return values


def pbm(
    x: Any,
    df: float | None = None,
    lambda_: float | None = None,
    mono: str = "up",
    control: dict[str, Any] | None = None,
    name: str | None = None,
    **kwargs: Any,
) -> PSplineVariant:
    """R pbm(): monotone P-spline spec (asymmetric penalty applied at fit time)."""

    values = _as_values(x)
    if mono not in ("up", "down"):
        raise ValueError("'arg' should be one of ['up', 'down']")
    settings = dict(control) if control is not None else pbm_control(**kwargs)
    for key, value in kwargs.items():
        settings[key] = value
    design, penalty, settings, knots = _pb_basis(values, settings, distinct=True)
    return PSplineVariant(
        name="pbm",
        X=design,
        D=penalty,
        D1=None,
        control=settings,
        df=_df_plus_two(df, design.shape[1]),
        lambda_=lambda_,
        max_df=float(design.shape[1]),
        values=values,
        knots=knots,
        zero_covariate=True,
        nl_df_offset=1.0,
        mono=mono,
    )


def pbo(
    x: Any,
    df: float | None = None,
    lambda_: float | None = None,
    control: dict[str, Any] | None = None,
    name: str | None = None,
    **kwargs: Any,
) -> PSplineVariant:
    """R pbo(): original P-spline spec (no distinct-value inter reduction)."""

    values = _as_values(x)
    settings = dict(control) if control is not None else pbo_control(**kwargs)
    for key, value in kwargs.items():
        settings[key] = value
    design, penalty, settings, knots = _pb_basis(values, settings, distinct=False)
    return PSplineVariant(
        name="pbo",
        X=design,
        D=penalty,
        D1=None,
        control=settings,
        df=_df_plus_two(df, design.shape[1]),
        lambda_=lambda_,
        max_df=float(design.shape[1]),
        values=values,
        knots=knots,
        zero_covariate=False,
    )


def pbp(
    x: Any,
    df: float | None = None,
    lambda_: float | None = None,
    control: dict[str, Any] | None = None,
    name: str | None = None,
    **kwargs: Any,
) -> PSplineVariant:
    """R pbp(): P-spline spec with positivity constraint metadata."""

    values = _as_values(x)
    settings = dict(control) if control is not None else pbp_control(**kwargs)
    for key, value in kwargs.items():
        settings[key] = value
    design, penalty, settings, knots = _pb_basis(values, settings, distinct=True)
    return PSplineVariant(
        name="pbp",
        X=design,
        D=penalty,
        D1=None,
        control=settings,
        df=_df_plus_two(df, design.shape[1]),
        lambda_=lambda_,
        max_df=float(design.shape[1]),
        values=values,
        knots=knots,
        zero_covariate=False,
    )


def pbz(
    x: Any,
    df: float | None = None,
    lambda_: float | None = None,
    control: dict[str, Any] | None = None,
    name: str | None = None,
    **kwargs: Any,
) -> PSplineVariant:
    """R pbz(): P-spline spec with an extra first-difference penalty D1."""

    values = _as_values(x)
    settings = dict(control) if control is not None else pbz_control(**kwargs)
    for key, value in kwargs.items():
        settings[key] = value
    design, penalty, settings, knots = _pb_basis(values, settings, distinct=True)
    d1 = difference_matrix(design.shape[1], 1)
    return PSplineVariant(
        name="pbz",
        X=design,
        D=penalty,
        D1=d1,
        control=settings,
        df=_df_plus_two(df, design.shape[1]),
        lambda_=lambda_,
        max_df=float(design.shape[1]),
        values=values,
        knots=knots,
        zero_covariate=True,
        nl_df_offset=1.0,
    )


def _Cdiff(n: int, order: int = 2, sin: bool = True) -> np.ndarray:
    """R's Cdiff(): cyclic difference penalty, order 1 or 2, with a sine term."""

    if order == 1:
        d2 = np.zeros((n, n + 1))
        p = np.array([1.0, -1.0])
        for k in range(n):
            d2[k, k : k + 2] = p
        d = d2[:, 1 : n + 1].copy()
        d[:, n - 1] += d2[:, 0]
        return d
    d2 = np.zeros((n, n + 2))
    p = np.array([-1.0, 2.0 * np.cos(2 * np.pi / n) if sin else 2.0, -1.0])
    for k in range(n):
        d2[k, k : k + 3] = p
    d = d2[:, 1 : n + 1].copy()
    d[:, 0] += d2[:, n + 1]
    d[:, n - 1] += d2[:, 0]
    return d


def pbc(
    x: Any,
    df: float | None = None,
    lambda_: float | None = None,
    max_df: float | None = None,
    control: dict[str, Any] | None = None,
    name: str | None = None,
    **kwargs: Any,
) -> PSplineVariant:
    """R pbc(): cyclic P-spline spec with a sine-aware cyclic penalty."""

    values = _as_values(x)
    settings = dict(control) if control is not None else pbc_control(**kwargs)
    for key, value in kwargs.items():
        settings[key] = value
    design, _unused, settings, knots = _pb_basis(
        values, settings, distinct=True, short_at=100, expand=False, cyclic=True
    )
    order = int(settings.get("order", 2))
    penalty = _Cdiff(design.shape[1], order, bool(settings.get("sin", True)))
    stored_df = None
    if df is not None:
        if df > design.shape[1] - 2:
            import warnings

            warnings.warn(
                "The df's exceed the number of columns of the design matrix\n   they are set to 3",
                UserWarning,
                stacklevel=3,
            )
            df = 3
        stored_df = 1.0 if df < 1 else float(df) + 1.0
    columns = design.shape[1]
    if max_df is None:
        max_df = float(columns - 2)
    if max_df > columns - 2:
        import warnings

        warnings.warn(
            f"The max.df's are set to {columns - 2}\n", UserWarning, stacklevel=3
        )
        max_df = float(columns - 2)
    return PSplineVariant(
        name="pbc",
        X=design,
        D=penalty,
        D1=None,
        control=settings,
        df=stored_df,
        lambda_=lambda_,
        max_df=float(max_df),
        values=values,
        knots=knots,
        zero_covariate=True,
        subtract_linear_hat=False,
        nl_df_offset=1.0,
    )


def pvc(
    x: Any,
    df: float | None = None,
    lambda_: float | None = None,
    by: Any = None,
    control: dict[str, Any] | None = None,
    name: str | None = None,
    **kwargs: Any,
) -> PSplineVariant:
    """R pvc(): varying-coefficient P-spline spec (by=None path)."""

    values = _as_values(x)
    if by is not None:
        raise NotImplementedError(
            "pvc(by=...) varying-coefficient interaction is not ported"
        )
    settings = dict(control) if control is not None else pvc_control(**kwargs)
    for key, value in kwargs.items():
        settings[key] = value
    design, penalty, settings, knots = _pb_basis(values, settings, distinct=True)
    stored_df = None
    if df is not None:
        if df > design.shape[1] - 2:
            import warnings

            warnings.warn(
                "The df's exceed the number of columns of the design matrix\n   they are set to 3",
                UserWarning,
                stacklevel=3,
            )
            df = 3
        stored_df = 1.0 if df < 1 else float(df)
    return PSplineVariant(
        name="pvc",
        X=design,
        D=penalty,
        D1=None,
        control=settings,
        df=stored_df,
        lambda_=lambda_,
        max_df=float(design.shape[1]),
        values=values,
        knots=knots,
        zero_covariate=False,
        subtract_linear_hat=False,
        by=by,
    )


def _to_pb_spec(
    spec: PSplineVariant, lambda_: float | None = None, df: float | None = None
) -> PBSpline:
    """Wrap a variant's X/D as a PBSpline so gamlss_pb can run estimation paths."""

    return PBSpline(
        name=spec.name,
        control=spec.control,
        X=spec.X,
        D=spec.D,
        knots=spec.knots,
        df=spec.df if df is None else df,
        max_df=spec.max_df,
        lambda_=spec.lambda_ if lambda_ is None else lambda_,
        values=spec.values,
    )


def _fit_variant(
    spec: PSplineVariant,
    y: Any,
    w: Any,
    xeval: Any = None,
    lambda_: float | None = None,
    df: float | None = None,
):
    """One P-spline variant backfitting step.

    The fixed-lambda path is computed here with the variant's own regpen; all
    lambda-estimation paths are delegated to gamlss_pb (documented above).
    """

    X = np.asarray(spec.X, dtype=float)
    D = np.asarray(spec.D, dtype=float)
    values = np.asarray(spec.values, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    if y.shape[0] != X.shape[0]:
        raise ValueError("y and the design matrix have different lengths")

    lam = spec.lambda_ if lambda_ is None else lambda_
    target_df = spec.df if df is None else df
    if lam is None:
        from dataclasses import replace

        estimate = _to_pb_spec(spec, df=target_df)
        fit = gamlss_pb(estimate, y, w)
        if xeval is not None:
            return np.asarray(
                CubicSpline(values, fit.fitted_values, bc_type="natural")(
                    np.asarray(xeval, dtype=float)
                )
            )
        return replace(
            fit,
            nl_df=fit.edf - spec.nl_df_offset,
            name=spec.name,
            method="delegated",
        )

    lam = min(max(float(lam), _LAMBDA_MIN), _LAMBDA_MAX)
    n, p = X.shape
    sqrt_w = np.sqrt(w)
    q, r = np.linalg.qr(sqrt_w[:, None] * X, mode="reduced")
    Qy = q.T @ (sqrt_w * y)

    if spec.name == "pbm":
        beta, edf = _regpen_pbm(r, Qy, D, lam, p, spec)
    elif spec.name == "pbz":
        beta, edf = _regpen_pbz(r, Qy, D, spec.D1, lam, p, spec)
    else:
        beta, edf = _regpen(r, Qy, D, lam, p)
    fv = X @ beta

    spline = CubicSpline(values, fv, bc_type="natural")
    if xeval is not None:
        return np.asarray(spline(np.asarray(xeval, dtype=float)), dtype=float)

    waug = np.concatenate([w, np.ones(D.shape[0])])
    xaug = np.vstack([X, np.sqrt(lam) * D])
    weighted = xaug * np.sqrt(waug)[:, None]
    leverage = np.sum(
        (weighted @ np.linalg.pinv(weighted.T @ weighted)) * weighted, axis=1
    )[:n]
    covariate = np.zeros_like(values) if spec.zero_covariate else values
    if spec.subtract_linear_hat:
        leverage = leverage - _hat_WX(w, covariate)
    var = leverage / w

    return SmootherFit(
        fitted_values=fv,
        residuals=y - fv,
        var=var,
        nl_df=edf - spec.nl_df_offset,
        lambda_=lam,
        coef=np.asarray(beta, dtype=float),
        edf=edf,
        sigb2=None,
        sige2=None,
        method="fixed",
        name=spec.name,
        knots=spec.knots if spec.knots.size else None,
        spline=spline,
    )


def _regpen_pbm(
    r: np.ndarray,
    Qy: np.ndarray,
    D: np.ndarray,
    lam: float,
    p: int,
    spec: PSplineVariant,
):
    """R gamlss.pbm's regpen: iteratively add sqrt(kappa)*(w2*D2) rows."""

    kappa = float(spec.control.get("kappa", 1e10))
    d2 = difference_matrix(D.shape[1], 1)
    w2 = np.zeros(D.shape[1] - 1)
    beta = edf = None
    for _ in range(20):
        augmented = np.vstack(
            [r, np.sqrt(lam) * D, np.sqrt(kappa) * (w2[:, None] * d2)]
        )
        u, d, vt = np.linalg.svd(augmented, full_matrices=False)
        rank = int(np.sum(d > d.max() * np.finfo(float).eps ** 0.8))
        u1 = u[:p, :rank]
        y1 = u1.T @ Qy
        beta = vt[:rank].T @ (y1 / d[:rank])
        edf = float(np.sum(u1**2))
        slope = d2 @ beta
        w2new = (
            (slope < 0).astype(float)
            if spec.mono == "up"
            else (slope > 0).astype(float)
        )
        if np.array_equal(w2new, w2):
            w2 = w2new
            break
        w2 = w2new
    return beta, edf


def _regpen_pbz(
    r: np.ndarray,
    Qy: np.ndarray,
    D: np.ndarray,
    D1: np.ndarray | None,
    lam: float,
    p: int,
    spec: PSplineVariant,
):
    """R gamlss.pbz's regpen: add the first-difference penalty when edf <= lim."""

    lim = float(spec.control.get("lim", 3))
    start = spec.control.get("start", (0.0001, 0.0001))
    lam2 = float(start[1]) if np.ndim(start) and len(start) > 1 else float(start)
    beta, edf = _regpen(r, Qy, D, lam, p)
    if edf <= lim and D1 is not None:
        augmented = np.vstack([r, np.sqrt(lam) * D, np.sqrt(lam2) * D1])
        u, d, vt = np.linalg.svd(augmented, full_matrices=False)
        rank = int(np.sum(d > d.max() * np.finfo(float).eps ** 0.8))
        u1 = u[:p, :rank]
        y1 = u1.T @ Qy
        beta = vt[:rank].T @ (y1 / d[:rank])
        edf = float(np.sum(u1**2))
    return beta, edf


def gamlss_pbm(x, y, w, xeval=None, **kwargs):
    """R gamlss.pbm: monotone P-spline backfitting step."""
    return _fit_variant(_require(x, "pbm"), y, w, xeval, **_lambda_kwargs(kwargs))


def gamlss_pbo(x, y, w, xeval=None, **kwargs):
    """R gamlss.pbo: original P-spline backfitting step."""
    return _fit_variant(_require(x, "pbo"), y, w, xeval, **_lambda_kwargs(kwargs))


def gamlss_pbp(x, y, w, xeval=None, **kwargs):
    """R gamlss.pbp: positivity P-spline backfitting step."""
    return _fit_variant(_require(x, "pbp"), y, w, xeval, **_lambda_kwargs(kwargs))


def gamlss_pbz(x, y, w, xeval=None, **kwargs):
    """R gamlss.pbz: going-to-zero P-spline backfitting step."""
    return _fit_variant(_require(x, "pbz"), y, w, xeval, **_lambda_kwargs(kwargs))


def gamlss_pbc(x, y, w, xeval=None, **kwargs):
    """R gamlss.pbc: cyclic P-spline backfitting step."""
    return _fit_variant(_require(x, "pbc"), y, w, xeval, **_lambda_kwargs(kwargs))


def gamlss_pvc(x, y, w, xeval=None, **kwargs):
    """R gamlss.pvc: varying-coefficient P-spline backfitting step."""
    return _fit_variant(_require(x, "pvc"), y, w, xeval, **_lambda_kwargs(kwargs))


def _lambda_kwargs(kwargs: dict) -> dict:
    out = {}
    if "lambda_" in kwargs:
        out["lambda_"] = kwargs["lambda_"]
    if "df" in kwargs:
        out["df"] = kwargs["df"]
    return out


def _require(x: Any, name: str) -> PSplineVariant:
    if not isinstance(x, PSplineVariant) or x.name != name:
        raise TypeError(f"gamlss_{name} needs the object returned by {name}()")
    return x


__all__ = [
    "PSplineVariant",
    "gamlss_pbc",
    "gamlss_pbm",
    "gamlss_pbo",
    "gamlss_pbp",
    "gamlss_pbz",
    "gamlss_pvc",
    "pbc",
    "pbm",
    "pbo",
    "pbp",
    "pbz",
    "pvc",
]
