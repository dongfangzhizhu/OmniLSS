"""R gamlss.foreach principal-component regression: fitPCR, pc(), pcr().

R sources: R/gamlss.foreach/R/fitPCR.R (fitPCR + the fitted/coef/vcov/predict
methods), R/gamlss.foreach/R/PCR-GAMLSS.R (pc, gamlss.pc, which.yX.Corr) and
R/gamlss.foreach/R/fitPCR-GAMLSS.R (pcr, gamlss.pcr).

fitPCR() fits the whole PC path: the covariates are weighted-centred/scaled,
SVD-ed once, and for each m = 1..M the fitted values are
    Fi[, m] = x_scaled B[, m] + mean.y,   B[, m] = P[, 1:m] gamma[1:m]
with gamma = T y / d^2.  The number of components is chosen by a local AIC
(a normal likelihood with the weighted residual SD, plus k (m + 2)) or fixed by
df.  pc()/pcr() wrap this into gamlss smoothers; pcr() additionally supports the
t-value and supervised (SPCR) variants.

Documented deviations:
* R draws the coefficient paths with plot=TRUE; plotting is not ported and the
  argument is accepted and ignored.
* The tol= branch of pc() is dead in R -- it compares against ncol(x), and x is
  NULL (or the raw argument) at that point, so any tol raises "argument is of
  length zero".  Python trims to the numerical rank and documents it.
* vcov.PCR in R reads object$loading (the field is called loadings), so
  summary.PCR(param = "beta") fails there; the port uses loadings and works.
* which.yX.Corr() subsets with model.matrix(...)[, -1], which drops to a plain
  vector when exactly one covariate survives -- and fitPCR(supervised = TRUE)
  then fails on dim(X)[2].  Python always returns a one-column DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .pcr import SVDResult

_LOG_2PI = float(np.log(2.0 * np.pi))


def weighted_mean(values: Any, weights: Any) -> np.ndarray:
    """R weighted.mean() applied column-wise."""

    matrix = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if matrix.ndim == 1:
        return np.asarray(float((w * matrix).sum() / w.sum()))
    return (w[:, None] * matrix).sum(axis=0) / w.sum()


def weighted_sd(values: Any, weights: Any) -> np.ndarray | float:
    """R fitPCR's local weighted.sd(): the (sum.w^2 - sum.w2) denominator."""

    matrix = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    sum_w = float(w.sum())
    sum_w2 = float((w**2).sum())
    if matrix.ndim == 1:
        numerator = (
            float((w * matrix**2).sum()) * sum_w - float((w * matrix).sum()) ** 2
        )
        return float(np.sqrt(numerator / (sum_w**2 - sum_w2)))
    numerator = (w[:, None] * matrix**2).sum(axis=0) * sum_w - (
        (w[:, None] * matrix).sum(axis=0)
    ) ** 2
    return np.sqrt(numerator / (sum_w**2 - sum_w2))


def _loglik_lm(residuals: np.ndarray, weights: np.ndarray, rank: int) -> float:
    """R logLik.lm(): the weighted normal log-likelihood of a least-squares fit.

    R writes the last term as log(sum(w * res^2)) -- without dividing by N, even
    though the first terms use log(N).  That quirk is what AIC() reports, so it
    is reproduced verbatim.
    """

    active = weights != 0.0
    n0 = float(active.sum())
    rss = float((weights * residuals**2).sum())
    return 0.5 * (
        float(np.log(weights[active]).sum())
        - n0 * (_LOG_2PI + 1.0 - np.log(n0) + np.log(rss))
    )


def _lm_aic(residuals: np.ndarray, weights: np.ndarray, rank: int, k: float) -> float:
    return float(-2.0 * _loglik_lm(residuals, weights, rank) + k * (rank + 1))


@dataclass(frozen=True)
class PCRFit:
    """The object R's fitPCR() returns (class "PCR")."""

    coefficients: np.ndarray
    scores: np.ndarray
    loadings: np.ndarray
    gamma: np.ndarray
    se_gamma: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    fitted_values: np.ndarray
    xvar: np.ndarray
    gaic: np.ndarray
    pc: int
    k: float
    M: int
    mean_y: float
    dim_x: tuple[int, int]
    sigma: np.ndarray
    residuals: np.ndarray
    names: tuple[str, ...] = field(default_factory=tuple)
    method: str = ""

    def fitted(self, pc: int | None = None) -> np.ndarray:
        """R fitted.PCR()."""

        column = self.pc if pc is None else int(pc)
        return self.fitted_values[:, column - 1]

    def residuals_at(self, pc: int | None = None) -> np.ndarray:
        """R residuals.PCR()."""

        column = self.pc if pc is None else int(pc)
        return self.residuals[:, column - 1]

    def coef(self, param: str = "beta", pc: int | None = None) -> np.ndarray:
        """R coef.PCR(): the gamma vector or the beta column at pc."""

        if param == "gamma":
            return self.gamma
        column = self.pc if pc is None else int(pc)
        return self.coefficients[:, column - 1]

    def vcov(self, pc: int | None = None, type: str = "vcov"):
        """R vcov.PCR(): loaded with the singular values, not the scores."""

        column = self.pc if pc is None else int(pc)
        sigma = float(self.sigma[column - 1])
        se_gamma = sigma / np.sqrt(np.asarray(self.xvar[:column], dtype=float))
        middle = np.diag(se_gamma**2) if column > 1 else np.array([[se_gamma[0] ** 2]])
        loading = self.loadings[:, :column]
        covariance = loading @ middle @ loading.T
        standard_errors = np.sqrt(np.diag(covariance))
        if type == "vcov":
            return covariance
        if type == "se":
            return standard_errors
        if type == "cor":
            scale = np.sqrt(np.diag(covariance))
            return covariance / scale[:, None] / scale[None, :]
        if type == "all":
            return {"se": standard_errors, "vcov": covariance}
        raise ValueError("'arg' should be one of ['vcov', 'cor', 'se', 'all']")

    def predict(self, newdata: Any = None, pc: int | None = None) -> np.ndarray:
        """R predict.PCR(): fitted values, or the scaling applied to newdata."""

        if newdata is None:
            return self.fitted(pc=pc)
        matrix = np.asarray(newdata, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        if matrix.shape[1] != self.dim_x[1]:
            raise ValueError(
                "the dimensions of the new data is not the same with the old"
            )
        scaled = (matrix - self.center) / self.scale
        return np.asarray(scaled @ self.coef(pc=pc), dtype=float) + self.mean_y


def which_yX_Corr(
    y: Any,
    x: Any,
    r: float = 0.5,
    plot: bool = True,
    hierarchical: bool = True,
    print: bool = True,
    digits: int = 3,
    names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """R which.yX.Corr: keep the covariates correlated with the response."""

    response = np.asarray(y, dtype=float).ravel()
    matrix = np.asarray(x, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("x must be a matrix")

    if names is None:
        names = list(
            getattr(x, "columns", [f"X{index + 1}" for index in range(matrix.shape[1])])
        )
    names = [str(name) for name in names]
    correlations = np.array(
        [
            np.corrcoef(response, matrix[:, index])[0, 1]
            for index in range(matrix.shape[1])
        ]
    )
    correlations = np.round(correlations, digits)
    selected = [name for name, value in zip(names, correlations) if abs(value) > r]

    # R builds model.matrix(~ n1 + n2 + ...) and drops the intercept.  For a
    # numeric matrix that is exactly the subset of the selected columns; with no
    # selection the formula has no terms, so R returns an n x 0 matrix.
    positions = [names.index(name) for name in selected]
    subset = matrix[:, positions]
    frame = pd.DataFrame(subset, columns=selected)
    if print:
        print(f"cup point for correlation {r}")
        print(f"dimesions of new matrix {frame.shape}")
    return frame


def fit_pcr(
    x: Any,
    y: Any,
    weights: Any = None,
    M: int | None = None,
    df: int | None = None,
    supervised: bool = False,
    k: float = 2.0,
    r: float = 0.2,
    plot: bool = False,
    names: Sequence[str] | None = None,
) -> PCRFit:
    """R fitPCR: the principal-component regression path."""

    if y is None:
        raise ValueError("the y is needed")
    matrix = np.asarray(x, dtype=float)
    response = np.asarray(y, dtype=float).ravel()
    if matrix.ndim != 2:
        raise ValueError("x must be a matrix")
    n, p = matrix.shape
    if response.shape[0] != n:
        raise ValueError("y and x have different lengths")
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)

    maximum = min(n, p) if M is None else int(M)
    if df is not None:
        maximum = int(df)
    if n >= p and maximum > p:
        maximum = p
    if n < p and maximum > n:
        maximum = n

    column_names = (
        list(names)
        if names is not None
        else [
            str(value)
            for value in getattr(x, "columns", [f"X{index + 1}" for index in range(p)])
        ]
    )
    if len(column_names) != p:
        column_names = [f"X{index + 1}" for index in range(p)]

    mean_y = float(weighted_mean(response, w))
    Y = (np.sqrt(w) * response)[:, None]

    center = weighted_mean(matrix, w)
    spread = np.asarray(weighted_sd(matrix, w), dtype=float)
    scaled = (matrix - center) / spread
    X = np.sqrt(w)[:, None] * scaled

    design = scaled
    if supervised:
        screened = which_yX_Corr(Y, X, r=r, plot=False, print=False, names=column_names)
        screened_names = list(screened.columns)
        positions = [column_names.index(name) for name in screened_names]
        design = scaled[:, positions]
        X = np.sqrt(w)[:, None] * design
        column_names = screened_names
        p = X.shape[1]
        maximum = p if n >= p else n

    # SVD of the weighted, scaled design.
    u, d, vt = np.linalg.svd(X, full_matrices=False)
    d = np.asarray(d[:maximum], dtype=float)
    T = u[:, :maximum] * d[None, :]
    P = vt[:maximum, :].T
    gamma_vector = (T.T @ Y).ravel() / d**2

    beta = np.zeros((p, maximum))
    fitted = np.zeros((n, maximum))
    for m in range(1, maximum + 1):
        beta[:, m - 1] = P[:, :m] @ gamma_vector[:m]
        fitted[:, m - 1] = design @ beta[:, m - 1] + mean_y

    residuals = response[:, None] - fitted
    sigma = np.asarray(weighted_sd(residuals, w), dtype=float)

    if df is None:
        gaic = np.array(
            [
                float(
                    -2.0
                    * np.sum(
                        -0.5 * _LOG_2PI
                        - np.log(sigma[m - 1])
                        - (response - fitted[:, m - 1]) ** 2
                        / (2.0 * sigma[m - 1] ** 2),
                    )
                    + k * (m + 2)
                )
                for m in range(1, maximum + 1)
            ]
        )
        chosen = int(np.argmin(gaic)) + 1
    else:
        gaic = np.zeros(0)
        chosen = int(df)

    se_gamma = float(sigma[maximum - 1]) / d
    return PCRFit(
        coefficients=beta,
        scores=T,
        loadings=P,
        gamma=np.asarray(gamma_vector, dtype=float),
        se_gamma=np.asarray(se_gamma, dtype=float),
        center=np.asarray(center, dtype=float),
        scale=spread,
        fitted_values=fitted,
        xvar=np.asarray(d**2, dtype=float),
        gaic=np.asarray(gaic, dtype=float),
        pc=chosen,
        k=float(k),
        M=int(maximum),
        mean_y=mean_y,
        dim_x=(n, int(np.asarray(x).shape[1])),
        sigma=sigma,
        residuals=residuals,
        names=tuple(column_names),
    )


def get_sig_terms(fit: PCRFit, x: Any, val: float = 1.96) -> dict[str, np.ndarray]:
    """R getSigTerms(): keep the components whose gamma t-value exceeds val."""

    t_values = fit.gamma / fit.se_gamma
    keep = (np.abs(t_values) > val).astype(float)
    gamma = fit.gamma * keep
    beta = (fit.loadings @ gamma).ravel()
    matrix = np.asarray(x, dtype=float)
    scaled = (matrix - fit.center) / fit.scale
    fitted = scaled @ beta + fit.mean_y
    return {"fitted_values": fitted, "coefficients": beta, "gamma": gamma}


def _wls(design: np.ndarray, response: np.ndarray, w: np.ndarray):
    """Weighted least squares without an intercept, plus R\'s standard errors."""

    XtW = design.T * w
    information = XtW @ design
    coefficients = np.linalg.solve(information, XtW @ response)
    fitted = design @ coefficients
    residuals = response - fitted
    rank = design.shape[1]
    dof = response.shape[0] - rank
    sigma2 = float((w * residuals**2).sum() / dof)
    inverse = np.linalg.inv(information)
    standard_errors = np.sqrt(np.diag(sigma2 * inverse))
    leverage = np.einsum("ij,jk,ik->i", design, inverse, design)
    se_fit = np.sqrt(leverage * sigma2)
    return coefficients, fitted, residuals, standard_errors, se_fit, sigma2


@dataclass(frozen=True)
class PCSpec:
    """The object R\'s pc() returns: a prcomp-like decomposition in an attribute."""

    components: dict[str, Any]
    names: tuple[str, ...]
    max_no: int
    df: float | None
    k: float
    method: str


@dataclass(frozen=True)
class PCFit:
    """What R\'s gamlss.pc() returns for one backfitting step."""

    fitted_values: np.ndarray
    residuals: np.ndarray
    var: np.ndarray
    nl_df: float
    lambda_: float
    coef: np.ndarray
    beta: np.ndarray
    edf: float
    aic: float
    method: str
    name: str = "pc"


def pc(
    x_vars: Sequence[str] | None = None,
    x: Any = None,
    x_svd: Any = None,
    df: float | None = None,
    center: bool = True,
    scale: bool = True,
    tol: float | None = None,
    max_number: int | None = None,
    k: float | None = None,
    method: str = "t-values",
    data: Mapping[str, Any] | None = None,
) -> PCSpec:
    """R pc(): principal-component smoother specification."""

    if method not in ("t-values", "GAIC", "k-fold"):
        raise ValueError("'arg' should be one of ['t-values', 'GAIC', 'k-fold']")

    supplied = sum(item is not None for item in (x_vars, x, x_svd))
    if supplied == 0:
        raise ValueError("x or x.vars or x.svd has to be set in pc()")
    if supplied > 1:
        raise ValueError("use only one of the arguments x, x.vars or x.svd")

    if x_vars is not None:
        if data is None:
            raise ValueError("data is required when x.vars is used")
        frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(dict(data))
        matrix = np.asarray(frame[list(x_vars)], dtype=float)
        center_values = matrix.mean(axis=0) if center else None
        spread = matrix.std(axis=0, ddof=1) if scale else None
        scaled = matrix.copy()
        if center_values is not None:
            scaled = scaled - center_values
        if spread is not None:
            scaled = scaled / spread
        names = [str(name) for name in x_vars]
        n, p = scaled.shape
        u, d, vt = np.linalg.svd(scaled, full_matrices=False)
        components = {
            "d": d,
            "u": u,
            "vt": vt,
            "X": scaled,
            "center": center_values,
            "scale": spread,
        }
    elif x is not None:
        matrix = np.asarray(x, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("x must be a matrix")
        if center and scale:
            center_values = matrix.mean(axis=0)
            spread = matrix.std(axis=0, ddof=1)
            scaled = (matrix - center_values) / spread
        elif center:
            center_values = matrix.mean(axis=0)
            spread = None
            scaled = matrix - center_values
        elif scale:
            center_values = None
            spread = matrix.std(axis=0, ddof=1)
            scaled = matrix / spread
        else:
            center_values = None
            spread = None
            scaled = matrix.copy()
        names = [
            str(name)
            for name in getattr(
                x, "columns", [f"X{index + 1}" for index in range(scaled.shape[1])]
            )
        ]
        n, p = scaled.shape
        u, d, vt = np.linalg.svd(scaled, full_matrices=False)
        components = {
            "d": d,
            "u": u,
            "vt": vt,
            "X": scaled,
            "center": center_values,
            "scale": spread,
        }
    else:
        if isinstance(x_svd, SVDResult):
            components = {
                "d": np.asarray(x_svd.d, float),
                "u": np.asarray(x_svd.u, float),
                "vt": np.asarray(x_svd.vt, float),
                "X": np.asarray(x_svd.X, float),
                "center": np.asarray(x_svd.center, float),
                "scale": np.asarray(x_svd.scale, float),
            }
        elif isinstance(x_svd, Mapping) and {
            "d",
            "u",
            "vt",
            "X",
            "center",
            "scale",
        } <= set(x_svd):
            components = {
                key: np.asarray(x_svd[key], float)
                for key in ("d", "u", "vt", "X", "center", "scale")
            }
        else:
            raise ValueError("x.svd should be a list of 6, created by GetSVD()")
        n = components["u"].shape[0]
        # R reads p from nrow(vt), which is max.number when the SVD was trimmed.
        p = components["vt"].shape[0]
        names = [
            str(name)
            for name in getattr(
                components["X"], "columns", [f"X{index + 1}" for index in range(p)]
            )
        ]
        if len(names) != components["vt"].shape[1]:
            names = [f"X{index + 1}" for index in range(components["vt"].shape[1])]

    if max_number is None:
        max_number = min(n, p)
    max_number = int(max_number)
    if df is not None and df > min(n, p):
        raise ValueError("the df have to be less than the number of variables")
    if df is not None:
        max_number = int(df)
    if k is None:
        k = float(np.log(n))

    d = np.asarray(components["d"], dtype=float)
    vt = np.asarray(components["vt"], dtype=float)
    if tol is not None:
        # R compares against ncol(x), which is NULL here, so any tol errors.
        rank = int(np.sum(d > d[0] * tol))
        vt = vt[:rank, :]
    vt = vt[:max_number, :]
    scores = np.asarray(components["X"], dtype=float) @ vt.T
    sdev = d / np.sqrt(max(1.0, float(n - 1)))

    return PCSpec(
        components={
            "sdev": sdev,
            "rotation": vt.T,
            "center": components["center"],
            "scale": components["scale"],
            "x": scores,
        },
        names=tuple(names),
        max_no=max_number,
        df=None if df is None else float(df),
        k=float(k),
        method=method,
    )


def gamlss_pc(x: Any, y: Any, w: Any, xeval: Any = None, **kwargs: Any) -> PCFit:
    """R gamlss.pc: one principal-component backfitting step."""

    spec = x
    if not isinstance(spec, PCSpec):
        raise TypeError("gamlss_pc needs the object returned by pc()")
    if xeval is not None:
        raise NotImplementedError(
            "the xeval branch of gamlss.pc() reads the stored newdata from the "
            "predict environment; use PCFit.beta with the stored centre/scale instead"
        )

    scores = np.asarray(spec.components["x"], dtype=float)
    rotation = np.asarray(spec.components["rotation"], dtype=float)
    response = np.asarray(y, dtype=float)
    weights = np.asarray(w, dtype=float)
    n, p = scores.shape
    edf = spec.df
    max_number = spec.max_no

    if edf is None:
        cap = int(min(n - 5, p, max_number))
        if spec.method == "GAIC":
            finite = []
            for index in range(1, cap + 1):
                design = scores[:, :index]
                coefs, fitted, residuals, _, _, _ = _wls(design, response, weights)
                finite.append(_lm_aic(residuals, weights, index, spec.k))
            values = np.asarray(finite, dtype=float)
            if not np.any(np.isfinite(values)):
                raise ValueError("AIC is not defined for this model")
            edf = int(np.nanargmin(np.where(np.isfinite(values), values, np.nan))) + 1
            design = scores[:, :edf]
            coefs, fitted, residuals, _, se_fit, _ = _wls(design, response, weights)
            beta = rotation[:, :edf] @ coefs
            aic = _lm_aic(residuals, weights, edf, spec.k)
        elif spec.method == "t-values":
            design = scores[:, :cap]
            coefs, fitted, residuals, se, _, _ = _wls(design, response, weights)
            keep = np.abs(coefs / se) > 2.0
            edf = int(keep.sum())
            if edf == 0:
                ones = np.ones((n, 1))
                coefs, fitted, residuals, _, se_fit, _ = _wls(ones, response, weights)
            else:
                design = scores[:, :cap][:, keep]
                coefs, fitted, residuals, _, se_fit, _ = _wls(design, response, weights)
            expanded = np.zeros(cap)
            expanded[keep] = coefs
            beta = rotation[:, :cap] @ expanded
            aic = _lm_aic(residuals, weights, max(edf, 1), spec.k)
        else:
            raise NotImplementedError(
                "pc(method = 'k-fold') is not implemented in R either"
            )
    else:
        count = int(edf)
        design = scores[:, :count]
        coefs, fitted, residuals, _, se_fit, _ = _wls(design, response, weights)
        beta = rotation[:, :count] @ coefs
        aic = _lm_aic(residuals, weights, count, spec.k)

    return PCFit(
        fitted_values=np.asarray(fitted, dtype=float),
        residuals=np.asarray(residuals, dtype=float),
        var=np.asarray(se_fit**2, dtype=float),
        nl_df=float(edf),
        lambda_=0.0,
        coef=np.asarray(coefs, dtype=float),
        beta=np.asarray(beta, dtype=float),
        edf=float(edf),
        aic=float(aic),
        method=spec.method,
    )


@dataclass(frozen=True)
class PCRSpec:
    """The object R\'s pcr() returns: the raw design matrix plus the settings."""

    X: np.ndarray
    names: tuple[str, ...]
    M: int
    df: float | None
    k: float
    r: float
    method: str
    state: dict = field(default_factory=lambda: {"lambda": 0})


@dataclass(frozen=True)
class PCRSmootherFit:
    """What R\'s gamlss.pcr() returns for one backfitting step."""

    fitted_values: np.ndarray
    residuals: np.ndarray
    var: np.ndarray
    nl_df: float
    lambda_: float
    edf: float
    gamma: np.ndarray | None
    coef_smo: PCRFit
    method: str
    name: str = "pcr"


def pcr(
    x_vars: Sequence[str] | None = None,
    x: Any = None,
    df: float | None = None,
    M: int | None = None,
    k: float | None = None,
    r: float = 0.2,
    method: str = "GAIC",
    data: Mapping[str, Any] | None = None,
) -> PCRSpec:
    """R pcr(): principal-component regression smoother specification."""

    if method not in ("GAIC", "t-values", "SPCR"):
        raise ValueError("'arg' should be one of ['GAIC', 't-values', 'SPCR']")
    if x_vars is not None and x is not None:
        raise ValueError("use only one of the arguments x or x.vars")
    if x_vars is None and x is None:
        raise ValueError("x or x.vars has to be set in pcr()")

    if x_vars is not None:
        if data is None:
            raise ValueError("data is required when x.vars is used")
        frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(dict(data))
        matrix = np.asarray(frame[list(x_vars)], dtype=float)
        names = [str(name) for name in x_vars]
    else:
        matrix = np.asarray(x, dtype=float)
        names = [
            str(name)
            for name in getattr(
                x, "columns", [f"X{index + 1}" for index in range(matrix.shape[1])]
            )
        ]

    n, p = matrix.shape
    if df is not None and df > min(n, p):
        raise ValueError("the df have to be less than the number of variables")
    maximum = min(p, n) if M is None else int(M)
    if df is not None:
        maximum = int(df)
    if k is None:
        k = float(np.log(n))

    return PCRSpec(
        X=matrix,
        names=tuple(names),
        M=int(maximum),
        df=None if df is None else float(df),
        k=float(k),
        r=float(r),
        method=method,
    )


def gamlss_pcr(
    x: Any, y: Any, w: Any, xeval: Any = None, **kwargs: Any
) -> PCRSmootherFit:
    """R gamlss.pcr: one PCR backfitting step."""

    spec = x
    if not isinstance(spec, PCRSpec):
        raise TypeError("gamlss_pcr needs the object returned by pcr()")
    if xeval is not None:
        raise NotImplementedError(
            "the xeval branch of gamlss.pcr() reads the stored newdata from the "
            "predict environment and a fit that is not in scope"
        )

    matrix = np.asarray(spec.X, dtype=float)
    response = np.asarray(y, dtype=float)
    weights = np.asarray(w, dtype=float)

    gamma = None
    newbeta = None
    if spec.df is not None:
        fit = fit_pcr(
            matrix,
            response,
            weights,
            M=spec.M,
            df=spec.df,
            k=spec.k,
            r=spec.r,
            names=spec.names,
        )
        fitted = fit.fitted(pc=int(spec.df))
        edf = float(fit.pc)
        residuals = fit.residuals_at(pc=int(spec.df))
    elif spec.method == "GAIC":
        previous = float(spec.state.get("lambda", 0))
        fit = fit_pcr(
            matrix, response, weights, M=spec.M, k=spec.k, r=spec.r, names=spec.names
        )
        target = float(fit.pc)
        tolerance = float(np.ceil(spec.M / 10.0))
        if previous != 0.0 and abs(previous - target) > tolerance:
            column = int(previous)
            fitted = fit.fitted(pc=column)
            edf = previous
            residuals = fit.residuals_at(pc=column)
            spec.state["lambda"] = previous
        else:
            fitted = fit.fitted()
            edf = target
            residuals = fit.residuals_at()
            spec.state["lambda"] = target
    elif spec.method == "t-values":
        edf = float(spec.M)
        fit = fit_pcr(
            matrix,
            response,
            weights,
            M=spec.M,
            df=spec.M,
            k=spec.k,
            r=spec.r,
            names=spec.names,
        )
        result = get_sig_terms(fit, matrix)
        fitted = result["fitted_values"]
        gamma = result["gamma"]
        newbeta = result["coefficients"]
        edf = float(np.sum(gamma != 0))
        residuals = response - fitted
    else:
        fit = fit_pcr(
            matrix,
            response,
            weights,
            M=spec.M,
            df=None,
            supervised=True,
            k=spec.k,
            r=spec.r,
            names=spec.names,
        )
        fitted = fit.fitted()
        edf = float(fit.pc)
        residuals = fit.residuals_at()

    return PCRSmootherFit(
        fitted_values=np.asarray(fitted, dtype=float),
        residuals=np.asarray(residuals, dtype=float),
        var=np.asarray(fitted, dtype=float),
        nl_df=float(edf),
        lambda_=float(edf),
        edf=float(edf),
        gamma=None if gamma is None else np.asarray(gamma, dtype=float),
        coef_smo=fit,
        method=spec.method,
    )


# R name kept explicit: the coverage tooling maps "fitPCR" to fitPCR.
fitPCR = fit_pcr


__all__ = [
    "PCFit",
    "PCRFit",
    "PCRSmootherFit",
    "PCRSpec",
    "PCSpec",
    "fitPCR",
    "fit_pcr",
    "gamlss_pc",
    "gamlss_pcr",
    "get_sig_terms",
    "pc",
    "pcr",
    "weighted_mean",
    "weighted_sd",
    "which_yX_Corr",
]
