"""R gamlss smoother fitting functions (gamlss.pb and friends).

R source: R/gamlss/R/pb.R::gamlss.pb.

gamlss.pb() is the backfitting step for a P-spline: given the tagged covariate
returned by pb() (carrying X, D, lambda, df, max.df and control), the partial
residual y and the weights w, it solves the penalised least squares problem and
returns the fitted values, the coefficient, the effective df, the smoothing
parameter, the per-observation variance of the smoother and a natural spline
through (x, fitted) used for prediction.

The linear algebra mirrors R exactly: QR of sqrt(w) X, the augmented system
[R; sqrt(lambda) D], its SVD with R's rank tolerance, and the same ML / GAIC /
GCV / df-target searches for lambda (including the 1e-7 .. 1e7 clipping).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar

from .family_groups import _hat_WX
from .smoother_specs import PBSpline, PSSpline

_RANK_TOL = np.finfo(float).eps**0.8
_LAMBDA_MIN = 1.0e-7
_LAMBDA_MAX = 1.0e7


@dataclass(frozen=True)
class SmootherFit:
    """What R's gamlss.<smoother>() returns for a single backfitting step."""

    fitted_values: np.ndarray
    residuals: np.ndarray
    var: np.ndarray
    nl_df: float
    lambda_: float
    coef: np.ndarray
    edf: float
    sigb2: float | None
    sige2: float | None
    method: str
    name: str
    knots: np.ndarray | None
    spline: Any = None

    def predict(self, newx: Any) -> np.ndarray:
        """Natural spline interpolation of the fitted values (R's coefSmo$fun)."""

        if self.spline is None:
            raise ValueError("this fit does not carry a prediction spline")
        return np.asarray(self.spline(np.asarray(newx, dtype=float)), dtype=float)


def _regpen(R: np.ndarray, Qy: np.ndarray, D: np.ndarray, lambda_: float, p: int):
    """R's regpen(): penalised regression through the SVD of [R; sqrt(lambda) D]."""

    augmented = np.vstack([R, np.sqrt(lambda_) * D])
    u, d, vt = np.linalg.svd(augmented, full_matrices=False)
    rank = int(np.sum(d > d.max() * _RANK_TOL))
    u1 = u[:p, :rank]
    y1 = u1.T @ Qy
    beta = vt[:rank].T @ (y1 / d[:rank])
    edf = float(np.sum(u1**2))
    return beta, edf


def _design_of(x: Any):
    if isinstance(x, PBSpline):
        return (
            np.asarray(x.X, dtype=float),
            np.asarray(x.D, dtype=float),
            np.asarray(x.values, dtype=float),
            x.df,
            float(x.max_df),
            dict(x.control),
            x.lambda_,
            x.knots,
        )
    if isinstance(x, PSSpline):
        return (
            np.asarray(x.design, dtype=float),
            np.asarray(x.penalty, dtype=float),
            np.asarray(x.values, dtype=float),
            x.df,
            float(x.design.shape[1]),
            {"order": x.order, "method": "ML", "k": 2},
            x.lambda_,
            x.knots,
        )
    raise TypeError("x must be the object returned by pb() or ps()")


def gamlss_pb(x: Any, y: Any, w: Any, xeval: Any = None, **kwargs: Any):
    """R gamlss.pb: one P-spline backfitting step, or a prediction for xeval.

    Parameters
    ----------
    x : PBSpline or PSSpline
        The tagged covariate returned by pb() or ps().
    y : array_like
        Partial residuals for this term.
    w : array_like
        Prior weights.
    xeval : array_like, optional
        When given, return the natural-spline prediction at these points
        instead of the fit object.
    """

    X, D, values, df, max_df, control, lambda_, knots = _design_of(x)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    if y.shape[0] != X.shape[0]:
        raise ValueError("y and the design matrix have different lengths")

    order = int(control.get("order", 2))
    method = str(control.get("method", "ML"))
    k = float(control.get("k", 2))

    N = float(np.sum(w != 0))
    n, p = X.shape
    sqrt_w = np.sqrt(w)
    q, r = np.linalg.qr(sqrt_w[:, None] * X, mode="reduced")
    Qy = q.T @ (sqrt_w * y)

    def fit_at(lam: float):
        beta, edf = _regpen(r, Qy, D, lam, p)
        return beta, edf, X @ beta

    sig2 = tau2 = None
    lam = None if lambda_ is None else float(lambda_)

    if lam is not None:
        lam = min(max(lam, _LAMBDA_MIN), _LAMBDA_MAX)
        beta, edf, fv = fit_at(lam)
    elif df is not None:
        rinv = np.linalg.pinv(r)
        eig = np.linalg.eigvalsh(rinv.T @ (D.T @ D) @ rinv)

        def edf_gap(log_lambda: float) -> float:
            return float(np.sum(1.0 / (1.0 + np.exp(log_lambda) * eig)) - df)

        if np.sign(edf_gap(-30.0)) == np.sign(edf_gap(30.0)):
            log_lambda = 30.0
        else:
            log_lambda = brentq(edf_gap, -30.0, 30.0)
        lam = float(np.exp(log_lambda))
        beta, edf, fv = fit_at(lam)
        if abs(edf - df) > 0.1:
            import warnings

            warnings.warn(
                "the target df's are not acheived, try to reduce the no. of knot intervals in pb(). eg. inter=10",
                UserWarning,
                stacklevel=2,
            )
    else:
        lam = 10.0
        if method == "ML":
            for _ in range(50):
                beta, edf, fv = fit_at(lam)
                gamma = D @ beta
                sig2 = float(np.sum(w * (y - fv) ** 2) / (N - edf))
                tau2 = float(np.sum(gamma**2) / (edf - order))
                if tau2 < 1e-7:
                    tau2 = 1e-7
                previous = lam
                lam = min(max(sig2 / tau2, _LAMBDA_MIN), _LAMBDA_MAX)
                if abs(lam - previous) < 1e-7 or lam > 1e10:
                    break
        elif method == "GAIC":

            def objective(log_lambda: float) -> float:
                _, e, f = fit_at(float(np.exp(log_lambda)))
                return float(np.sum(w * (y - f) ** 2) + k * e)

            result = minimize_scalar(
                objective,
                bounds=(np.log(_LAMBDA_MIN), np.log(_LAMBDA_MAX)),
                method="bounded",
                options={"xatol": 1e-10},
            )
            lam = float(np.exp(result.x))
            beta, edf, fv = fit_at(lam)
        elif method == "GCV":
            rinv = np.linalg.inv(r)
            eigenvalues, eigenvectors = np.linalg.eigh(rinv.T @ (D.T @ D) @ rinv)
            yy = eigenvectors.T @ Qy
            y_y = float(np.sum((sqrt_w * y) ** 2))

            def gcv(log_lambda: float) -> float:
                denom = 1.0 + np.exp(log_lambda) * eigenvalues
                e = float(np.sum(1.0 / denom))
                y_hy2 = (
                    y_y
                    - 2.0 * float(np.sum(yy**2 / denom))
                    + float(np.sum(yy**2 / denom**2))
                )
                return float(n * y_hy2 / (n - k * e) ** 2)

            result = minimize_scalar(
                gcv,
                bounds=(np.log(_LAMBDA_MIN), np.log(_LAMBDA_MAX)),
                method="bounded",
                options={"xatol": 1e-10},
            )
            lam = float(np.exp(result.x))
            beta, edf, fv = fit_at(lam)
        else:
            raise ValueError(f"unknown lambda selection method {method!r}")

    if edf > max_df:
        rinv = np.linalg.inv(r)
        eig = np.linalg.eigvalsh(rinv.T @ (D.T @ D) @ rinv)

        def max_df_gap(log_lambda: float) -> float:
            return float(np.sum(1.0 / (1.0 + np.exp(log_lambda) * eig)) - max_df)

        if np.sign(max_df_gap(-30.0)) == np.sign(max_df_gap(30.0)):
            log_lambda = 30.0
        else:
            log_lambda = brentq(max_df_gap, -30.0, 30.0)
        lam = float(np.exp(log_lambda))
        beta, edf, fv = fit_at(lam)

    spline = CubicSpline(values, fv, bc_type="natural")
    if xeval is not None:
        return np.asarray(spline(np.asarray(xeval, dtype=float)), dtype=float)

    waug = np.concatenate([w, np.ones(D.shape[0])])
    xaug = np.vstack([X, np.sqrt(lam) * D])
    weighted = xaug * np.sqrt(waug)[:, None]
    leverage = np.sum(
        (weighted @ np.linalg.pinv(weighted.T @ weighted)) * weighted, axis=1
    )[:n]
    var = (leverage - _hat_WX(w, values)) / w

    return SmootherFit(
        fitted_values=fv,
        residuals=y - fv,
        var=var,
        nl_df=edf - 2.0,
        lambda_=lam,
        coef=np.asarray(beta, dtype=float),
        edf=edf,
        sigb2=tau2,
        sige2=sig2,
        method=method,
        name="pb",
        knots=knots,
        spline=spline,
    )


def _ps_golden_lambda(dm: np.ndarray, aug: np.ndarray, w: np.ndarray, df: float) -> float:
    """R's find.lambda.from.df(): golden-section search on the edf-vs-lambda curve.

    R searches lambda in [1e-6, 1e5] with relative tolerance 1e-4; the same
    bracket and stopping rule are used here so the selected lambda matches.
    """

    waug = np.concatenate([w, np.ones(aug.shape[0])])

    def edf(lam: float) -> float:
        xaug = np.vstack([dm, np.sqrt(lam) * aug])
        weighted = xaug * np.sqrt(waug)[:, None]
        information = weighted.T @ weighted
        inner = dm.T @ (w[:, None] * dm)
        return float(np.trace(np.linalg.solve(information, inner)))

    r = 0.61803399
    z = 1.0 - r
    a, c = 1.0e-6, 1.0e5
    b = r * a + z * c
    l4, l3 = a, c
    l1 = b if (c - b) > (b - a) else b - z * (b - a)
    l2 = b + z * (c - b) if (c - b) > (b - a) else b
    f1 = (edf(l1) - df) ** 2
    f2 = (edf(l2) - df) ** 2
    while True:
        if f2 < f1:
            l4 = l1
            l1 = l2
            l2 = r * l1 + z * l3
            f1 = f2
            f2 = (edf(l2) - df) ** 2
        else:
            l3 = l2
            l2 = l1
            l1 = r * l2 + z * l4
            f2 = f1
            f1 = (edf(l1) - df) ** 2
        if not abs(l3 - l4) > 1.0e-4 * (abs(l1) + abs(l2)):
            break
    return float(l1 if f1 < f2 else l2)


def gamlss_ps(x: Any, y: Any, w: Any, xeval: Any = None, **kwargs: Any):
    """R gamlss.ps: one P-spline (Eilers-Marx) backfitting step.

    Differs from gamlss.pb in the penalty: ps() stores the raw difference matrix
    and the fit augments the design with sqrt(lambda) * penalty instead of using
    pb's QR/SVD route.
    """

    if not isinstance(x, PSSpline):
        raise TypeError("gamlss_ps needs the object returned by ps()")

    dm = np.asarray(x.design, dtype=float)
    aug = np.asarray(x.penalty, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    if dm.shape[0] != y.shape[0]:
        raise ValueError("y and the design matrix have different lengths")
    if dm.shape[1] != aug.shape[1]:
        raise ValueError("the dimensions of the augmented matrix and of the design matrix are incompatible")

    n = y.shape[0]
    N = float(np.sum(w != 0))
    df = x.df
    lam = x.lambda_
    if lam is None:
        if df is None:
            raise ValueError("df or lambda should be set")
        lam = _ps_golden_lambda(dm, aug, w, float(df))
    lam = float(lam)

    aN = aug.shape[0]
    xaug = np.vstack([dm, np.sqrt(lam) * aug])
    yaug = np.concatenate([y, np.zeros(aN)])
    waug = np.concatenate([w, np.ones(aN)])

    weighted = xaug * np.sqrt(waug)[:, None]
    coef = np.linalg.lstsq(weighted, np.sqrt(waug) * yaug, rcond=None)[0]
    fitted = dm @ coef
    residuals = y - fitted

    information = weighted.T @ weighted
    inner = dm.T @ (w[:, None] * dm)
    nl_df = float(np.trace(np.linalg.solve(information, inner))) - 2.0

    leverage = np.sum((weighted @ np.linalg.pinv(information)) * weighted, axis=1)[:n]
    var = (leverage - _hat_WX(w, x.values)) / w

    if xeval is not None:
        from scipy.interpolate import CubicSpline

        return np.asarray(
            CubicSpline(np.asarray(x.values, dtype=float), fitted, bc_type="natural")(
                np.asarray(xeval, dtype=float)
            ),
            dtype=float,
        )

    return SmootherFit(
        fitted_values=fitted,
        residuals=residuals,
        var=var,
        nl_df=nl_df,
        lambda_=lam,
        coef=np.asarray(coef, dtype=float),
        edf=nl_df + 2.0,
        sigb2=None,
        sige2=None,
        method="df" if x.lambda_ is None else "fixed",
        name="ps",
        knots=x.knots,
        spline=None,
    )


__all__ = ["SmootherFit", "gamlss_pb", "gamlss_ps"]
