"""R gamlss pcat() / gamlss.pcat: penalised categorical smoother.

R source: R/gamlss/R/p-categorical.R.

The smoother merges factor levels: X is the one-hot model matrix and the penalty
D holds every pairwise difference (K * (K - 1) / 2 rows), so a large lambda
pulls all level coefficients together.  The penalty is applied through the same
iteratively reweighted SVD as ri():

    RD = [R; sqrt(lambda) * sqrt(omega) * D; 1e-3 * I]
    omega = 1 / (|D beta|^(2 - Lp) + kappa^2)

with a ridge term P0 = 1e-6 * I appended to keep the system full rank.  Lp = 0
(the default) is a fused-lasso-like penalty on the level differences.

Compared with live R for a fixed lambda, a df target (both the Lp = 2 eigen
shortcut and the Lp != 2 regpen path) and the ML lambda iteration.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import brentq, minimize

_RANK_TOL = np.finfo(float).eps ** 0.8


@dataclass(frozen=True)
class PCatSpec:
    """The object R's pcat() returns: one-hot design plus the pairwise penalty."""

    X: np.ndarray
    D: np.ndarray
    levels: list[Any]
    lambda_: float | None
    df: float | None
    method: str
    Lp: float
    kappa: float
    iterations: int
    c_crit: float
    start: float
    k: float
    name: str = "pcat"
    zero_covariate: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))


@dataclass(frozen=True)
class PCatFit:
    """What R's gamlss.pcat() returns for one backfitting step."""

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
    levels: list[Any]
    name: str = "pcat"

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(self.var)


def pairwise_differences(size: int) -> np.ndarray:
    """R's genD(): every pairwise difference, rows ordered (i = 2..K, j = 1..i-1)."""

    rows = []
    for i in range(1, size):
        for j in range(i):
            row = np.zeros(size)
            row[i] = 1.0
            row[j] = -1.0
            rows.append(row)
    return np.asarray(rows, dtype=float)


def pcat(
    fac: Any,
    df: float | None = None,
    lambda_: float | None = None,
    method: str = "ML",
    start: float = 0.001,
    Lp: float = 0.0,
    kappa: float = 1e-5,
    iter: int = 100,
    c_crit: float = 1e-4,
    k: float = 2.0,
) -> PCatSpec:
    """R pcat(): penalised categorical smoother specification."""

    import pandas as pd

    if method not in ("ML", "GAIC"):
        raise ValueError("'arg' should be one of ['ML', 'GAIC']")

    values = fac if isinstance(fac, pd.Series) else pd.Series(fac)
    if not isinstance(values.dtype, pd.CategoricalDtype):
        values = values.astype("category")
    codes = values.cat.codes.to_numpy()
    if np.any(codes < 0):
        raise ValueError("pcat() does not accept missing factor values")
    levels = list(values.cat.categories)
    size = len(levels)

    design = np.zeros((codes.shape[0], size))
    design[np.arange(codes.shape[0]), codes] = 1.0
    penalty = pairwise_differences(size)

    if lambda_ is not None and lambda_ < 0:
        lambda_ = 1.0
        warnings.warn(
            f"lambda was negative; have used  {lambda_}", UserWarning, stacklevel=2
        )

    if df is not None:
        if df > size:
            warnings.warn(f"the df are set to  {size}", UserWarning, stacklevel=2)
            df = float(size)
        if df < 0:
            warnings.warn("the df are set to 0", UserWarning, stacklevel=2)
            df = 0.0
        if df == size:
            lambda_ = 0.0

    return PCatSpec(
        X=design,
        D=penalty,
        levels=levels,
        lambda_=None if lambda_ is None else float(lambda_),
        df=None if df is None else float(df),
        method=method,
        Lp=float(Lp),
        kappa=float(kappa),
        iterations=int(iter),
        c_crit=float(c_crit),
        start=float(start),
        k=float(k),
        zero_covariate=np.zeros(codes.shape[0]),
    )


def gamlss_pcat(x: Any, y: Any, w: Any, xeval: Any = None, **kwargs: Any):
    """R gamlss.pcat: one penalised-categorical backfitting step."""

    spec = x
    if not isinstance(spec, PCatSpec):
        raise TypeError("gamlss_pcat needs the object returned by pcat()")

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

    non_zero = float(np.sum(w != 0))
    q, R = np.linalg.qr(np.sqrt(w)[:, None] * X, mode="reduced")
    Qy = q.T @ (np.sqrt(w) * y)
    ridge = np.eye(p) * 1e-6
    aN = spec.D.shape[0]

    def regpen(lambda_: float):
        omega = np.ones(aN)
        beta = np.zeros(p)
        previous = np.zeros(p)
        u = d = vt = None
        rank = 0
        for _ in range(spec.iterations):
            penalised = np.vstack(
                [
                    R,
                    np.sqrt(lambda_) * np.sqrt(omega)[:, None] * spec.D,
                    ridge,
                ]
            )
            u, d, vt = np.linalg.svd(penalised, full_matrices=False)
            rank = int(np.sum(d > d.max() * _RANK_TOL))
            u1 = u[:p, :rank]
            beta = vt[:rank].T @ ((u1.T @ Qy) / d[:rank])
            gap = float(np.max(np.abs(previous - beta)))
            previous = beta
            differences = spec.D @ beta
            omega = 1.0 / (np.abs(differences) ** (2.0 - spec.Lp) + spec.kappa**2)
            if gap < spec.c_crit:
                break
        return beta, float(np.sum(u[:p, :rank] ** 2)), omega

    lam = spec.lambda_
    sig2 = tau2 = None

    if lam is not None:
        beta, edf, omega = regpen(float(lam))
    elif spec.df is None:
        lam = min(max(spec.start, 1.0e-7), 1.0e7)
        if spec.method == "ML":
            for _ in range(20):
                beta, edf, omega = regpen(lam)
                gamma = (np.sqrt(omega)[:, None] * spec.D) @ beta
                fitted = X @ beta
                sig2 = float(np.sum(w * (y - fitted) ** 2) / (non_zero - edf))
                tau2 = float(np.sum(gamma**2) / edf)
                if tau2 < 1.0e-7:
                    tau2 = 1.0e-7
                previous = lam
                lam = sig2 / tau2
                if abs(lam - previous) < 0.0001 or lam > 100000:
                    break
        else:

            def gaic(candidate: float) -> float:
                b, e, _ = regpen(float(candidate))
                return float(np.sum(w * (y - X @ b) ** 2) + spec.k * e)

            result = minimize(
                lambda value: gaic(float(np.asarray(value).ravel()[0])),
                x0=np.array([spec.start]),
                method="L-BFGS-B",
                bounds=[(1.0e-7, 1.0e7)],
                options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 500},
            )
            lam = float(result.x[0])
            beta, edf, omega = regpen(lam)
    else:
        df = float(spec.df)
        if spec.Lp != 2.0:

            def gap_regpen(candidate: float) -> float:
                return float(regpen(candidate)[1] - df)

            lam = _uniroot(gap_regpen, 0.0, 100000.0)
        else:
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

    waug = np.concatenate([w, np.ones(aN)])
    xaug = np.vstack([X, np.sqrt(lam) * (np.sqrt(omega)[:, None] * spec.D)])
    weighted = xaug * np.sqrt(waug)[:, None]
    leverage = np.sum(
        (weighted @ np.linalg.pinv(weighted.T @ weighted)) * weighted, axis=1
    )[:n]
    fitted = np.asarray(X @ beta, dtype=float)

    return PCatFit(
        fitted_values=fitted,
        residuals=y - fitted,
        var=leverage / w,
        nl_df=edf - 1.0,
        lambda_=float(lam),
        coef=np.asarray(beta, dtype=float),
        edf=edf,
        sigb2=tau2,
        sige2=sig2,
        omega=omega,
        Lp=spec.Lp,
        levels=spec.levels,
    )


def _uniroot(function, lower: float, upper: float) -> float:
    """R's uniroot() on [lower, upper], falling back to the upper end."""

    if np.sign(function(lower)) == np.sign(function(upper)):
        return float(upper)
    return float(brentq(function, lower, upper))


__all__ = ["PCatFit", "PCatSpec", "gamlss_pcat", "pairwise_differences", "pcat"]
