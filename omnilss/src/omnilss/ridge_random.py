"""R gamlss ridge-type smoothers: ri() / gamlss.ri and random() / gamlss.random.

R sources: R/gamlss/R/ri.R and R/gamlss/R/random.R.

ri() is a ridge-type penalised fit with an Lp penalty: the design matrix is
centred/scaled, the penalty matrix is D = I (order 0) or a difference matrix,
and gamlss.ri() solves the penalised system by an SVD of [R; sqrt(lambda) *
sqrt(omega) * D] where the weights omega = 1 / (|beta|^(2 - Lp) + kappa^2) are
refreshed each iteration (so Lp = 2 is plain ridge, Lp < 2 approaches the lasso).

random() fits a random intercept for a factor: each level gets
beta_k = sum(w y) / (n_k + lambda) and the effective df is
sum(n_k / (n_k + lambda)) over the levels that are actually present.

Both are compared with live R; the lambda searches use the same brackets and
stopping rules as R (uniroot on [0, 1e5] for the df target, an ML fixed point,
and nlminb-like bounded optimisation for GAIC).

R's magic constant c.crit and the omega iteration are reproduced exactly, so a
fixed lambda fit agrees with R to machine precision.  The GAIC branch uses
scipy's L-BFGS-B instead of R's PORT nlminb and therefore only agrees to the
optimiser's tolerance (documented in the tests).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import brentq, minimize

from .smoother_specs import difference_matrix

_RANK_TOL = np.finfo(float).eps ** 0.8


@dataclass(frozen=True)
class RISpec:
    """The object R's ri() returns: tagged zero vector carrying the ridge design."""

    X: np.ndarray
    D: np.ndarray
    names: list[str]
    lambda_: float | None
    df: float | None
    method: str
    Lp: float
    kappa: float
    iterations: int
    c_crit: float
    order: int
    start: float
    k: float
    name: str = "ri"
    zero_covariate: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    knots: None = None


@dataclass(frozen=True)
class RIFit:
    """What R's gamlss.ri() returns for one backfitting step."""

    fitted_values: np.ndarray
    residuals: np.ndarray
    var: np.ndarray
    nl_df: float
    lambda_: float
    coef: np.ndarray
    edf: float
    sigb2: float | None
    sige2: float | None
    omega: np.ndarray
    Lp: float
    name: str = "ri"

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(self.var)


@dataclass(frozen=True)
class RandomSpec:
    """The object R's random() returns: zero vector plus the factor codes."""

    codes: np.ndarray
    levels: list[Any]
    df: float | None
    lambda_: float | None
    start: float
    name: str = "random"

    @property
    def X(self) -> np.ndarray:
        return np.zeros(self.codes.shape[0])


@dataclass(frozen=True)
class RandomFit:
    """What R's gamlss.random() returns for one backfitting step."""

    fitted_values: np.ndarray
    residuals: np.ndarray
    var: np.ndarray
    nl_df: float
    lambda_: float
    coef: np.ndarray
    edf: float
    sigb2: float | None
    sige2: float | None
    levels: list[Any]
    codes: np.ndarray
    name: str = "random"

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(self.var)


def _warn(message: str) -> None:
    warnings.warn(message, UserWarning, stacklevel=3)


def _uniroot(function, lower: float, upper: float) -> float:
    """R's uniroot() on [lower, upper], falling back to the upper end."""

    low_value = function(lower)
    high_value = function(upper)
    if np.sign(low_value) == np.sign(high_value):
        return float(upper)
    return float(brentq(function, lower, upper))


def ri(
    X: Any,
    df: float | None = None,
    lambda_: float | None = None,
    method: str = "ML",
    order: int = 0,
    start: float = 10.0,
    Lp: float = 2.0,
    kappa: float = 0.00001,
    iter: int = 100,
    c_crit: float = 1.0e-6,
    k: float = 2.0,
) -> RISpec:
    """R ri(): ridge / Lp-penalised design for gamlss.ri().

    R applies scale() to X (centre and divide by the n - 1 standard deviation),
    so the returned design is already standardised.
    """

    matrix = np.asarray(X, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("X must be a matrix")
    if df is None and lambda_ is None:
        raise ValueError("Specify at least one: df or lambda")
    if method not in ("ML", "GAIC"):
        raise ValueError("'arg' should be one of ['ML', 'GAIC']")

    center = matrix.mean(axis=0)
    spread = matrix.std(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        scaled = (matrix - center) / spread

    n, p = scaled.shape
    names = [str(index + 1) for index in range(p)]

    if lambda_ is not None and lambda_ < 0:
        lambda_ = 0.0
        _warn(f"lambda was negative; have used  {lambda_}")

    if df is not None:
        if df > p:
            _warn("the df are set to p")
            df = float(p)
        if df < 0:
            _warn("the df are set to 0")
            df = 0.0

    penalty = np.eye(p) if order == 0 else difference_matrix(p, order)
    return RISpec(
        X=scaled,
        D=penalty,
        names=names,
        lambda_=None if lambda_ is None else float(lambda_),
        df=None if df is None else float(df),
        method=method,
        Lp=float(Lp),
        kappa=float(kappa),
        iterations=int(iter),
        c_crit=float(c_crit),
        order=int(order),
        start=float(start),
        k=float(k),
        zero_covariate=np.zeros(n),
    )


def gamlss_ri(x: Any, y: Any, w: Any, xeval: Any = None, **kwargs: Any):
    """R gamlss.ri: one ridge-type backfitting step (or a prediction for xeval)."""

    spec = x
    if not isinstance(spec, RISpec):
        raise TypeError("gamlss_ri needs the object returned by ri()")

    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    design = spec.X
    if design.shape[0] < y.shape[0]:
        raise ValueError("the design matrix has fewer rows than y")
    X = design[: y.shape[0]]
    n, p = X.shape
    if p != spec.D.shape[1]:
        raise ValueError(
            "the dimensions of the penalty matrix and of the design matrix are incompatible"
        )

    non_zero = int(np.sum(w != 0))
    if non_zero < p:
        raise ValueError("ri() needs at least p observations with a non-zero weight")

    sqrt_w = np.sqrt(w)
    q, R = np.linalg.qr(sqrt_w[:, None] * X, mode="reduced")
    Qy = q.T @ (sqrt_w * y)

    def regpen(lambda_: float):
        omega = np.ones(p)
        beta = np.zeros(p)
        previous = np.zeros(p)
        u = d = vt = None
        rank = 0
        for _ in range(spec.iterations):
            penalised = np.vstack(
                [R, np.sqrt(lambda_) * np.sqrt(omega)[:, None] * spec.D]
            )
            u, d, vt = np.linalg.svd(penalised, full_matrices=False)
            rank = int(np.sum(d > d.max() * _RANK_TOL))
            u1 = u[:p, :rank]
            beta = vt[:rank].T @ ((u1.T @ Qy) / d[:rank])
            gap = float(np.max(np.abs(previous - beta)))
            previous = beta
            omega = 1.0 / (np.abs(beta) ** (2.0 - spec.Lp) + spec.kappa**2)
            if gap < spec.c_crit:
                break
        edf = float(np.sum(u[:p, :rank] ** 2))
        return beta, edf, omega

    lam = spec.lambda_
    sig2 = tau2 = None
    method = spec.method
    start = spec.start

    if lam is not None:
        beta, edf, omega = regpen(float(lam))
    elif spec.df is None:
        lam = min(max(start, 1.0e-7), 1.0e7)
        if method == "ML":
            for _ in range(20):
                beta, edf, omega = regpen(lam)
                gamma = spec.D @ beta * np.sqrt(omega)
                fitted = X @ beta
                sig2 = float(np.sum(w * (y - fitted) ** 2) / (non_zero - edf))
                tau2 = float(np.sum(gamma**2) / (edf - spec.order))
                previous = lam
                lam = sig2 / tau2
                if abs(lam - previous) < 1.0e-4 or lam > 100000:
                    break
        else:

            def gaic(candidate: float) -> float:
                b, e, _ = regpen(float(candidate))
                return float(np.sum(w * (y - X @ b) ** 2) + spec.k * e)

            result = minimize(
                lambda value: gaic(float(np.asarray(value).ravel()[0])),
                x0=np.array([start]),
                method="L-BFGS-B",
                bounds=[(1.0e-7, 1.0e7)],
                options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 500},
            )
            lam = float(result.x[0])
            beta, edf, omega = regpen(lam)
    else:
        df = float(spec.df)
        r_inverse = np.linalg.inv(R)
        s = spec.D.T @ spec.D
        scaled = r_inverse.T @ s @ r_inverse
        scaled = 0.5 * (scaled + scaled.T)
        eigenvalues = np.linalg.eigvalsh(scaled)

        def gap(candidate: float) -> float:
            return float(np.sum(1.0 / (1.0 + candidate * eigenvalues)) - df)

        lam = _uniroot(gap, 0.0, 100000.0)
        beta, edf, omega = regpen(lam)

    if xeval is not None:
        extra = design[y.shape[0] :]
        if extra.shape[0] == 0:
            raise ValueError("the design matrix has no rows for prediction")
        return np.asarray(extra @ beta, dtype=float)

    waug = np.concatenate([w, np.ones(spec.D.shape[0])])
    xaug = np.vstack([X, np.sqrt(lam) * spec.D])
    weighted = xaug * np.sqrt(waug)[:, None]
    leverage = np.sum(
        (weighted @ np.linalg.pinv(weighted.T @ weighted)) * weighted, axis=1
    )[:n]
    var = leverage / w

    return RIFit(
        fitted_values=np.asarray(X @ beta, dtype=float),
        residuals=y - X @ beta,
        var=var,
        nl_df=edf - 1.0,
        lambda_=float(lam),
        coef=np.asarray(beta, dtype=float),
        edf=edf,
        sigb2=tau2,
        sige2=sig2,
        omega=omega,
        Lp=spec.Lp,
    )


def random_spec(
    x: Any,
    df: float | None = None,
    lambda_: float | None = None,
    start: float = 10.0,
) -> RandomSpec:
    """R random(): tag a factor for a random-intercept smoother.

    R stores the factor codes as the "contrasts" attribute and a zero covariate;
    the Python object keeps the codes and the level labels.
    """

    import pandas as pd

    if isinstance(x, pd.Series):
        values = x
    else:
        values = pd.Series(x)
    if not isinstance(values.dtype, pd.CategoricalDtype):
        raise ValueError("random() expects a factor as its first argument")

    codes = values.cat.codes.to_numpy()
    if np.any(codes < 0):
        raise ValueError("random() does not accept missing factor values")

    levels = list(values.cat.categories)
    total = len(levels)
    if df is not None:
        df = float(min(df, total))
        df = float(max(df, 0.001))

    return RandomSpec(
        codes=codes + 1,
        levels=levels,
        df=df,
        lambda_=None if lambda_ is None else float(lambda_),
        start=float(start),
    )


def _df_inverse(
    n: np.ndarray, df: float, lambda_: float | None = None, iterations: int = 10
) -> float:
    """R's df.inv(): solve sum(n / (n + lambda)) = df for lambda."""

    if df > n.shape[0]:
        return 0.0
    if lambda_ is None:
        lambda_ = float(np.sum(n) / df - np.mean(n))
    current = float(np.sum(n / (n + lambda_)))
    if abs((df - current) / df) < 0.0001 or iterations == 1:
        return float(lambda_)
    updated = float(
        np.exp(
            np.log(lambda_)
            + (current - df) / np.sum((n * lambda_) / (n + lambda_) ** 2)
        )
    )
    return _df_inverse(n, df, updated, iterations - 1)


def gamlss_random(x: Any, y: Any, w: Any, xeval: Any = None, **kwargs: Any):
    """R gamlss.random: one random-intercept backfitting step."""

    spec = x
    if not isinstance(spec, RandomSpec):
        raise TypeError("gamlss_random needs the object returned by random_spec()")
    if xeval is not None:
        raise NotImplementedError(
            "the prediction branch of gamlss.random() reads the stored fitted "
            "levels from the predict environment; pass new data through the "
            "fitted coefficient vector instead"
        )

    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    codes = np.asarray(spec.codes, dtype=int)
    total = len(spec.levels)
    if codes.shape[0] != y.shape[0]:
        raise ValueError("y and the factor have different lengths")

    present = np.zeros(total, dtype=bool)
    present[codes - 1] = True
    nw = np.array([float(np.sum(w[codes == level])) for level in range(1, total + 1)])
    weighted_sum = np.array(
        [float(np.sum((w * y)[codes == level])) for level in range(1, total + 1)]
    )

    def regpen(lambda_: float):
        beta = weighted_sum / (nw + lambda_)
        edf = float(np.sum(nw[present] / (nw[present] + lambda_)))
        return beta, edf

    lam = spec.lambda_
    sig2 = tau2 = 0.0
    non_zero = float(np.sum(w != 0))

    if lam is not None:
        beta, edf = regpen(float(lam))
    elif spec.df is None:
        lam = spec.start
        beta, edf = regpen(lam)
        for _ in range(50):
            beta, edf = regpen(lam)
            fitted = beta[codes - 1]
            sig2 = float(np.sum(w * (y - fitted) ** 2) / (non_zero - edf))
            tau2 = float(np.sum(beta[present] ** 2) / edf)
            previous = lam
            lam = sig2 / tau2
            if lam < 1.0e-7:
                lam = 1.0e-7
            if abs(lam - previous) < 1.0e-7 or lam > 1.0e7:
                break
    else:
        lam = _df_inverse(nw[present], float(spec.df))
        beta, edf = regpen(lam)

    fitted = beta[codes - 1]
    var = w / (nw[codes - 1] + lam)

    return RandomFit(
        fitted_values=np.asarray(fitted, dtype=float),
        residuals=y - fitted,
        var=var,
        nl_df=edf,
        lambda_=float(lam),
        coef=np.asarray(beta, dtype=float),
        edf=edf,
        sigb2=tau2,
        sige2=sig2,
        levels=spec.levels,
        codes=codes,
    )


__all__ = [
    "RIFit",
    "RISpec",
    "RandomFit",
    "RandomSpec",
    "gamlss_random",
    "gamlss_ri",
    "random_spec",
    "ri",
]
