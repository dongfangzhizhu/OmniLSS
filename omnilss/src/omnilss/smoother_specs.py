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

from .smoother_controls import pb_control


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
    design, knots = bbase(values, xmin, xmax, inter, settings.get("degree", 3), bool(settings.get("quantiles", False)))

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


__all__ = ["PBSpline", "bbase", "difference_matrix", "pb"]
