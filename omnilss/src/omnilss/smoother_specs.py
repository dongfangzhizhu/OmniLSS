"""R gamlss smoother specification constructors (pb, and the shared B-spline basis).

R source: R/gamlss/R/pb.R (bbase + pb), R/gamlss/R/PS.R (ps),
R/gamlss/R/cubicSplines-10-08-12.R (cs).

R's pb() does not fit anything: it returns the covariate with attributes X (the
B-spline basis), D (the difference penalty), df, max.df, lambda and control.
gamlss() later hands those to gamlss.pb(), which solves the penalised system.

The Python port returns a small immutable object carrying the same fields, and
reproduces bbase() exactly (truncated-power basis, R's seq() spacing, R's
knots attribute subsetting), so X and D are numerically identical to R.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .smoother_controls import cy_control, pb_control


def _seq(from_: float, to: float, by: float) -> np.ndarray:
    """R's seq(from, to, by): n = floor((to - from) / by + 1e-10) + 1."""

    if by == 0:
        raise ValueError("'by' must be a non-zero number")
    count = int(np.floor((to - from_) / by + 1e-10)) + 1
    return from_ + np.arange(max(count, 0)) * by


def difference_matrix(size: int, order: int) -> np.ndarray:
    """diff(diag(size), diff=order) as a dense matrix."""

    identity = np.eye(size)
    for _ in range(order):
        identity = np.diff(identity, axis=0)
    return identity


def bbase(
    x: Any,
    xl: float,
    xr: float,
    ndx: int,
    deg: int,
    quantiles: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """R's bbase(): truncated-power B-spline basis and its knots.

    quantiles=TRUE is kept for signature parity; R's own comment says that
    branch "is not working and should be taken out", and the default path is
    the one gamlss() uses.
    """

    values = np.asarray(x, dtype=float)
    dx = (xr - xl) / ndx
    knots = _seq(xl - deg * dx, xr + deg * dx, dx)
    # P[i, j] = (x_i - knots_j)^deg * (x_i > knots_j)
    powers = values[:, None] - knots[None, :]
    basis = np.where(values[:, None] > knots[None, :], powers**deg, 0.0)
    n = basis.shape[1]
    penalty = difference_matrix(n, deg + 1) / (_gamma(deg + 1) * dx**deg)
    design = (-1.0) ** (deg + 1) * basis @ penalty.T
    # R keeps a subset of the knots as an attribute (n is ncol(P), not ncol(B)).
    keep = np.ones(n, dtype=bool)
    if deg >= 2:
        keep[: deg - 1] = False
    keep[n - (deg - 1) :] = False
    return design, knots[keep]


def _gamma(value: int) -> float:
    from math import gamma

    return float(gamma(value))


@dataclass(frozen=True)
class PBSpline:
    """The object R's pb() returns: basis, penalty and the df/lambda settings."""

    name: str
    control: dict[str, Any]
    X: np.ndarray
    D: np.ndarray
    knots: np.ndarray
    df: float | None
    max_df: float
    lambda_: float | None
    values: np.ndarray = field(repr=False, default=None)

    @property
    def r(self) -> int:
        return int(self.X.shape[1])


def pb(
    x: Any,
    df: float | None = None,
    lambda_: float | None = None,
    max_df: float | None = None,
    control: dict[str, Any] | None = None,
    name: str | None = None,
    **kwargs: Any,
) -> PBSpline:
    """R pb(): P-spline basis and penalty for a covariate.

    Mirrors R's bookkeeping: for a sample shorter than 99 the number of
    intervals drops to 10, then to the number of distinct values when that is
    smaller; df is stored as df + 2 (R's convention) with the "df exceeds the
    basis" guard; max.df defaults to the number of basis columns.
    """

    values = np.asarray(x, dtype=float)
    if values.ndim != 1:
        raise ValueError("x is a matrix declare it as a vector")
    settings = dict(control) if control is not None else pb_control(**kwargs)
    for key, value in kwargs.items():
        settings[key] = value

    length = values.shape[0]
    distinct = int(np.unique(values).shape[0])
    inter = settings.get("inter", 20)
    if length < 99:
        inter = 10
    if distinct <= inter:
        inter = distinct
    settings["inter"] = inter

    xl = float(np.min(values))
    xr = float(np.max(values))
    xmax = xr + 0.01 * (xr - xl)
    xmin = xl - 0.01 * (xr - xl)
    design, knots = bbase(
        values,
        xmin,
        xmax,
        inter,
        settings.get("degree", 3),
        bool(settings.get("quantiles", False)),
    )

    columns = design.shape[1]
    order = settings.get("order", 2)
    penalty = np.eye(columns) if order == 0 else difference_matrix(columns, order)

    stored_df: float | None = None
    if df is not None:
        if df > (columns - 2):
            import warnings

            warnings.warn(
                "The df's exceed the number of columns of the design matrix\n   they are set to 3",
                UserWarning,
                stacklevel=2,
            )
            df = 3
        if df < 0:
            import warnings

            warnings.warn("the extra df's are set to 0", UserWarning, stacklevel=2)
            stored_df = 2.0
        else:
            stored_df = float(df) + 2.0

    if max_df is None:
        max_df = float(columns)
    if max_df > columns:
        import warnings

        warnings.warn(f"The max.df's are set to {columns}\n", UserWarning, stacklevel=2)
        max_df = float(columns)

    return PBSpline(
        name=name or "pb",
        control=settings,
        X=design,
        D=penalty,
        knots=knots,
        df=stored_df,
        max_df=float(max_df),
        lambda_=lambda_,
        values=values,
    )


@dataclass(frozen=True)
class PSSpline:
    """The object R's ps() returns: B-spline design, penalty, knots, df, lambda."""

    name: str
    design: np.ndarray
    penalty: np.ndarray
    knots: np.ndarray
    df: float
    lambda_: float | None
    order: int
    values: np.ndarray = field(repr=False, default=None)

    @property
    def X(self) -> np.ndarray:
        return self.design


def ps(
    x: Any,
    df: float | None = 3,
    lambda_: float | None = None,
    ps_intervals: int = 20,
    degree: int = 3,
    order: int = 3,
    name: str | None = None,
) -> PSSpline:
    """R ps(): Eilers-Marx P-spline basis via splineDesign plus a difference penalty.

    Mirrors R's bookkeeping: knots span the 1%-expanded range, the penalty
    order is clipped when it exceeds the number of columns, df below 1 is set to
    1 and otherwise stored as df + 2, and a negative lambda is replaced by 0.
    """

    if lambda_ is None and df is None:
        raise ValueError("df or lambda should be set \n")

    values = np.asarray(x, dtype=float)
    if values.ndim != 1:
        raise ValueError("x is a matrix declare it as a vector")

    finite = values[~np.isnan(values)]
    xl = float(np.min(finite))
    xr = float(np.max(finite))
    xmax = xr + 0.01 * (xr - xl)
    xmin = xl - 0.01 * (xr - xl)
    dx = (xmax - xmin) / ps_intervals

    sorder = degree + 1
    if (ps_intervals - 1) < 1:
        import warnings

        warnings.warn(
            "ps.intervals was too small; have used 2", UserWarning, stacklevel=2
        )
    knots = _seq(xmin - degree * dx, xmax + degree * dx, dx)

    from scipy.interpolate import BSpline

    sparse_design = BSpline.design_matrix(finite, knots, degree)
    design = np.asarray(sparse_design.todense(), dtype=float)
    columns = design.shape[1]

    if (order - columns + 1) > 0:
        import warnings

        warnings.warn(
            f"order was too large; have used  {columns - 1}", UserWarning, stacklevel=2
        )
        order = columns - 1

    stored_df = float(df) + 2.0 if df is not None else None
    if df is not None and df < 1:
        import warnings

        warnings.warn("the df are set to 1", UserWarning, stacklevel=2)
        stored_df = 1.0

    if lambda_ is not None and lambda_ < 0:
        import warnings

        warnings.warn("lambda was negative; have used  0", UserWarning, stacklevel=2)
        lambda_ = 0.0

    penalty = difference_matrix(columns, order) if order != 0 else np.eye(columns)

    return PSSpline(
        name=name or "ps",
        design=design,
        penalty=penalty,
        knots=knots,
        df=stored_df,
        lambda_=lambda_,
        order=order,
        values=finite,
    )


@dataclass(frozen=True)
class CSSpline:
    """The object R's cs() returns: the covariate tagged for gamlss.cs()."""

    name: str
    values: np.ndarray
    control: dict[str, Any]
    df: float | None
    spar: float | None
    na_positions: tuple[int, ...]
    smooth_class: str = "smooth"


def cs(
    x: Any,
    df: float | None = 3,
    spar: float | None = None,
    c_spar: Any = None,
    control: dict[str, Any] | None = None,
    name: str | None = None,
    **kwargs: Any,
) -> CSSpline:
    """R cs(): tag a covariate for the cubic smoothing spline smoother.

    R does not build a basis here; gamlss.cs() calls smooth.spline() during
    fitting.  The constructor only validates the input, applies the df guard and
    records the control list (including c.spar's low/high window).
    """

    from .smoother_controls import cs_control

    values = np.asarray(x, dtype=float)
    if values.ndim != 1:
        raise ValueError(
            "The default smoother is bivariate; you gave a matrix as an argument"
        )

    settings = dict(control) if control is not None else cs_control(**kwargs)
    for key, value in kwargs.items():
        settings[key] = value

    if c_spar is None:
        settings["control.spar"] = {"low": -1.5, "high": 2}
    elif isinstance(c_spar, dict) and len(c_spar) == 2:
        settings["control.spar"] = {"low": c_spar["low"], "high": c_spar["high"]}
    elif isinstance(c_spar, (list, tuple)) and len(c_spar) == 2:
        settings["control.spar"] = {"low": c_spar[0], "high": c_spar[1]}
    else:
        raise ValueError("c.spar is not defined properly")

    if df is not None and df < 0:
        import warnings

        warnings.warn("the df are set to 3", UserWarning, stacklevel=2)
        df = 3.0

    return CSSpline(
        name=name or "cs",
        values=values,
        control=settings,
        df=df,
        spar=spar,
        na_positions=tuple(int(index) for index in np.flatnonzero(np.isnan(values))),
    )


@dataclass(frozen=True)
class CYSpline(PBSpline):
    """The object R's cy() returns: cyclic B-spline basis and cyclic penalty."""


def cbase(x: np.ndarray, xl: float, xr: float, ndx: int, deg: int) -> np.ndarray:
    """R's cbase(): wrap the last deg B-spline columns into the first deg."""

    b0, _ = bbase(x, xl, xr, ndx, deg)
    n = b0.shape[1] - deg
    wrapped = b0[:, :n].copy()
    wrapped[:, :deg] += b0[:, n : n + deg]
    return wrapped


def cdiff(n: int) -> np.ndarray:
    """R's cdiff(): cyclic second-difference penalty matrix (n x n)."""

    d2 = np.zeros((n, n + 2))
    p = np.array([-1.0, 2.0 * np.cos(2.0 * np.pi / n), -1.0])
    for k in range(n):
        d2[k, k : k + 3] = p
    d = d2[:, 1 : n + 1].copy()
    d[:, 0] += d2[:, n + 1]
    d[:, n - 1] += d2[:, 0]
    return d


def cy(
    x: Any,
    df: float | None = None,
    lambda_: float | None = None,
    control: dict[str, Any] | None = None,
    name: str | None = None,
    **kwargs: Any,
) -> CYSpline:
    """R cy(): cyclic P-spline basis and penalty for a covariate.

    Unlike pb(), cy() uses the raw min/max (no 1% expansion), builds a wrapped
    (cyclic) B-spline basis and a cyclic difference penalty, and stores df + 1
    (R's convention for cyclic smoothers).
    """

    values = np.asarray(x, dtype=float)
    if values.ndim != 1:
        raise ValueError("x must be a vector")
    settings = dict(control) if control is not None else cy_control(**kwargs)
    for key, value in kwargs.items():
        settings[key] = value

    length = values.shape[0]
    inter = settings.get("inter", 20)
    if length < 100:
        inter = 10
    settings["inter"] = inter

    xl = float(np.min(values))
    xr = float(np.max(values))
    design = cbase(values, xl, xr, inter, int(settings.get("degree", 3)))
    penalty = cdiff(design.shape[1])

    stored_df: float | None = None
    if df is not None:
        if df > design.shape[1] - 2:
            import warnings

            warnings.warn(
                "The df's exceed the number of columns of the design matrix\n   they are set to 3",
                UserWarning,
                stacklevel=2,
            )
            df = 3
        if df < 1:
            import warnings

            warnings.warn("the df are set to 1", UserWarning, stacklevel=2)
            stored_df = 1.0
        else:
            stored_df = float(df) + 1.0

    return CYSpline(
        name=name or "cy",
        control=settings,
        X=design,
        D=penalty,
        knots=np.asarray([]),
        df=stored_df,
        max_df=float(design.shape[1]),
        lambda_=lambda_,
        values=values,
    )


__all__ = [
    "CSSpline",
    "CYSpline",
    "PBSpline",
    "PSSpline",
    "bbase",
    "cbase",
    "cdiff",
    "cs",
    "cy",
    "difference_matrix",
    "pb",
    "ps",
]
