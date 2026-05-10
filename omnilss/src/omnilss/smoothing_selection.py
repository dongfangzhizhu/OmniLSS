"""Automatic smoothing parameter selection for GAMLSS.

This module implements methods for automatic selection of smoothing parameters
(lambda) for smooth terms in GAMLSS models, including:
- GCV (Generalized Cross-Validation)
- REML (Restricted Maximum Likelihood)
- AIC/BIC based selection

References
----------
- Wood, S. N. (2011). Fast stable restricted maximum likelihood and marginal
  likelihood estimation of semiparametric generalized linear models.
  Journal of the Royal Statistical Society: Series B, 73(1), 3-36.

- Wood, S. N. (2004). Stable and efficient multiple smoothing parameter
  estimation for generalized additive models. Journal of the American
  Statistical Association, 99(467), 673-686.

- Craven, P., & Wahba, G. (1978). Smoothing noisy data with spline functions.
  Numerische Mathematik, 31(4), 377-403.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple

import jax.numpy as jnp
import numpy as np
from scipy.linalg import cho_factor, cho_solve, svd
from scipy.optimize import minimize, minimize_scalar

# Type aliases
Array = np.ndarray | jnp.ndarray
OptimizationMethod = Literal["GCV", "REML", "AIC", "BIC"]


@dataclass(frozen=True)
class SmoothingResult:
    """Result of smoothing parameter selection.

    Attributes
    ----------
    lambda_opt : float
        Optimal smoothing parameter
    edf : float
        Effective degrees of freedom
    criterion_value : float
        Value of the optimization criterion (GCV, REML, etc.)
    method : str
        Method used for selection
    converged : bool
        Whether optimization converged
    n_iterations : int
        Number of iterations
    """

    lambda_opt: float
    edf: float
    criterion_value: float
    method: str
    converged: bool
    n_iterations: int


# =============================================================================
# Utility Functions
# =============================================================================


def compute_hat_matrix(
    X: Array, S: Array, lambda_: float, weights: Optional[Array] = None
) -> Array:
    """Compute the hat (influence) matrix for penalized regression.

    The hat matrix H is defined such that fitted values = H @ y.
    For penalized regression: H = X @ (X^T W X + λS)^(-1) @ X^T W

    Parameters
    ----------
    X : array_like
        Design matrix (n x p)
    S : array_like
        Penalty matrix (p x p)
    lambda_ : float
        Smoothing parameter
    weights : array_like, optional
        Observation weights (n,)

    Returns
    -------
    H : ndarray
        Hat matrix (n x n)
    """
    X = np.asarray(X, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    n, p = X.shape

    if weights is None:
        weights = np.ones(n)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    # W^(1/2) X
    W_sqrt = np.sqrt(weights)
    XW = X * W_sqrt[:, None]

    # X^T W X + λS
    XtWX = XW.T @ XW
    penalized = XtWX + lambda_ * S

    # Solve for (X^T W X + λS)^(-1) X^T W
    try:
        # Use Cholesky decomposition for stability
        c, lower = cho_factor(penalized, lower=True)
        XtW = X.T * weights[None, :]
        inv_XtW = cho_solve((c, lower), XtW)

        # H = X @ (X^T W X + λS)^(-1) @ X^T W
        H = X @ inv_XtW

    except np.linalg.LinAlgError:
        # Fallback to SVD if Cholesky fails
        warnings.warn("Cholesky decomposition failed, using SVD", RuntimeWarning)
        U, s, Vt = svd(penalized, full_matrices=False)
        s_inv = 1.0 / (s + np.finfo(float).eps)
        inv_penalized = (Vt.T * s_inv) @ U.T
        H = X @ inv_penalized @ X.T * weights[None, :]

    return H


def compute_edf(H: Array) -> float:
    """Compute effective degrees of freedom from hat matrix.

    EDF = trace(H)

    Parameters
    ----------
    H : array_like
        Hat matrix (n x n)

    Returns
    -------
    edf : float
        Effective degrees of freedom
    """
    return float(np.trace(H))


def compute_edf_fast(
    X: Array, S: Array, lambda_: float, weights: Optional[Array] = None
) -> float:
    """Compute EDF without forming full hat matrix.

    More efficient for large n. Uses the fact that:
    trace(H) = trace(X @ (X^T W X + λS)^(-1) @ X^T W)
             = trace((X^T W X + λS)^(-1) @ X^T W X)

    Parameters
    ----------
    X : array_like
        Design matrix (n x p)
    S : array_like
        Penalty matrix (p x p)
    lambda_ : float
        Smoothing parameter
    weights : array_like, optional
        Observation weights

    Returns
    -------
    edf : float
        Effective degrees of freedom
    """
    X = np.asarray(X, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    n, p = X.shape

    if weights is None:
        weights = np.ones(n)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    # W^(1/2) X
    W_sqrt = np.sqrt(weights)
    XW = X * W_sqrt[:, None]

    # X^T W X
    XtWX = XW.T @ XW

    # X^T W X + λS
    penalized = XtWX + lambda_ * S

    try:
        # Solve (X^T W X + λS)^(-1) @ X^T W X
        c, lower = cho_factor(penalized, lower=True)
        inv_XtWX = cho_solve((c, lower), XtWX)

        # trace((X^T W X + λS)^(-1) @ X^T W X)
        edf = float(np.trace(inv_XtWX))

    except np.linalg.LinAlgError:
        # Fallback
        warnings.warn("Cholesky failed in EDF computation", RuntimeWarning)
        edf = float(p)  # Conservative estimate

    return edf


# =============================================================================
# GCV (Generalized Cross-Validation)
# =============================================================================


def gcv_criterion(
    lambda_: float, X: Array, y: Array, S: Array, weights: Optional[Array] = None
) -> float:
    """Compute GCV criterion for given smoothing parameter.

    GCV(λ) = (n * RSS(λ)) / (n - EDF(λ))²

    where RSS is the residual sum of squares and EDF is the effective
    degrees of freedom.

    Parameters
    ----------
    lambda_ : float
        Smoothing parameter (must be positive)
    X : array_like
        Design matrix (n x p)
    y : array_like
        Response vector (n,)
    S : array_like
        Penalty matrix (p x p)
    weights : array_like, optional
        Observation weights (n,)

    Returns
    -------
    gcv : float
        GCV criterion value

    References
    ----------
    Craven & Wahba (1978). Smoothing noisy data with spline functions.
    """
    if lambda_ <= 0:
        return np.inf

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    n, p = X.shape

    if weights is None:
        weights = np.ones(n)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    # Fit penalized regression
    W_sqrt = np.sqrt(weights)
    XW = X * W_sqrt[:, None]
    yW = y * weights

    XtWX = XW.T @ XW
    Xty = X.T @ yW

    penalized = XtWX + lambda_ * S

    try:
        # Solve for coefficients
        c, lower = cho_factor(penalized, lower=True)
        beta = cho_solve((c, lower), Xty)

        # Compute fitted values and residuals
        fitted = X @ beta
        residuals = y - fitted

        # Weighted RSS
        rss = float(np.sum(weights * residuals**2))

        # Compute EDF efficiently
        inv_XtWX = cho_solve((c, lower), XtWX)
        edf = float(np.trace(inv_XtWX))

        # GCV criterion
        if n - edf <= 0:
            return np.inf

        gcv = (n * rss) / ((n - edf) ** 2)

    except (np.linalg.LinAlgError, ValueError):
        return np.inf

    return gcv


def select_lambda_gcv(
    X: Array,
    y: Array,
    S: Array,
    weights: Optional[Array] = None,
    lambda_min: float = 1e-8,
    lambda_max: float = 1e8,
    method: str = "brent",
) -> SmoothingResult:
    """Select optimal smoothing parameter using GCV.

    Parameters
    ----------
    X : array_like
        Design matrix (n x p)
    y : array_like
        Response vector (n,)
    S : array_like
        Penalty matrix (p x p)
    weights : array_like, optional
        Observation weights (n,)
    lambda_min : float, default=1e-8
        Minimum lambda to consider
    lambda_max : float, default=1e8
        Maximum lambda to consider
    method : str, default="brent"
        Optimization method ("brent" or "golden")

    Returns
    -------
    result : SmoothingResult
        Optimization result with optimal lambda and EDF

    Examples
    --------
    >>> X = np.random.randn(100, 10)
    >>> y = np.random.randn(100)
    >>> S = np.eye(10)
    >>> result = select_lambda_gcv(X, y, S)
    >>> print(f"Optimal lambda: {result.lambda_opt:.4f}")
    >>> print(f"EDF: {result.edf:.2f}")
    """

    # Objective function (minimize GCV)
    def objective(log_lambda):
        return gcv_criterion(np.exp(log_lambda), X, y, S, weights)

    # Optimize in log-space for better numerical behavior
    # Use 'bounded' method when bounds are specified
    result = minimize_scalar(
        objective,
        bounds=(np.log(lambda_min), np.log(lambda_max)),
        method="bounded",  # Changed from 'brent' to 'bounded' for compatibility with bounds
    )

    lambda_opt = np.exp(result.x)
    gcv_value = result.fun

    # Compute EDF at optimal lambda
    edf = compute_edf_fast(X, S, lambda_opt, weights)

    return SmoothingResult(
        lambda_opt=lambda_opt,
        edf=edf,
        criterion_value=gcv_value,
        method="GCV",
        converged=result.success,
        n_iterations=result.nfev,
    )


# =============================================================================
# REML (Restricted Maximum Likelihood)
# =============================================================================


def reml_criterion(
    lambda_: float, X: Array, y: Array, S: Array, weights: Optional[Array] = None
) -> float:
    """Compute REML criterion for given smoothing parameter.

    REML(λ) = -0.5 * (log|V| + log|X^T V^(-1) X| + y^T P y)

    where V = I/σ² and P is the projection matrix.

    Parameters
    ----------
    lambda_ : float
        Smoothing parameter
    X : array_like
        Design matrix (n x p)
    y : array_like
        Response vector (n,)
    S : array_like
        Penalty matrix (p x p)
    weights : array_like, optional
        Observation weights

    Returns
    -------
    reml : float
        Negative REML criterion (for minimization)

    References
    ----------
    Wood (2011). Fast stable REML estimation for GAMs.
    """
    if lambda_ <= 0:
        return np.inf

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    n, p = X.shape

    if weights is None:
        weights = np.ones(n)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    # Weighted design matrix
    W_sqrt = np.sqrt(weights)
    XW = X * W_sqrt[:, None]
    yW = y * weights

    XtWX = XW.T @ XW
    Xty = X.T @ yW

    penalized = XtWX + lambda_ * S

    try:
        # Cholesky decomposition
        c, lower = cho_factor(penalized, lower=True)

        # Solve for coefficients
        beta = cho_solve((c, lower), Xty)

        # Fitted values and residuals
        fitted = X @ beta
        residuals = y - fitted
        rss = float(np.sum(weights * residuals**2))

        # Log determinant of penalized matrix
        log_det_penalized = 2.0 * np.sum(np.log(np.diag(c)))

        # Log determinant of X^T W X
        c_xtx, _ = cho_factor(XtWX, lower=True)
        log_det_xtx = 2.0 * np.sum(np.log(np.diag(c_xtx)))

        # REML criterion (negative for minimization)
        # Simplified version without constant terms
        reml = 0.5 * (log_det_penalized - log_det_xtx + n * np.log(rss))

    except (np.linalg.LinAlgError, ValueError):
        return np.inf

    return reml


def select_lambda_reml(
    X: Array,
    y: Array,
    S: Array,
    weights: Optional[Array] = None,
    lambda_min: float = 1e-8,
    lambda_max: float = 1e8,
) -> SmoothingResult:
    """Select optimal smoothing parameter using REML.

    Parameters
    ----------
    X : array_like
        Design matrix (n x p)
    y : array_like
        Response vector (n,)
    S : array_like
        Penalty matrix (p x p)
    weights : array_like, optional
        Observation weights
    lambda_min : float, default=1e-8
        Minimum lambda
    lambda_max : float, default=1e8
        Maximum lambda

    Returns
    -------
    result : SmoothingResult
        Optimization result

    Examples
    --------
    >>> result = select_lambda_reml(X, y, S)
    >>> print(f"Optimal lambda: {result.lambda_opt:.4f}")
    """

    # Objective function
    def objective(log_lambda):
        return reml_criterion(np.exp(log_lambda), X, y, S, weights)

    # Optimize
    result = minimize_scalar(
        objective, bounds=(np.log(lambda_min), np.log(lambda_max)), method="bounded"
    )

    lambda_opt = np.exp(result.x)
    reml_value = result.fun

    # Compute EDF
    edf = compute_edf_fast(X, S, lambda_opt, weights)

    return SmoothingResult(
        lambda_opt=lambda_opt,
        edf=edf,
        criterion_value=reml_value,
        method="REML",
        converged=result.success,
        n_iterations=result.nfev,
    )


# =============================================================================
# AIC/BIC Based Selection
# =============================================================================


def aic_criterion(
    lambda_: float, X: Array, y: Array, S: Array, weights: Optional[Array] = None
) -> float:
    """Compute AIC for given smoothing parameter.

    AIC = n * log(RSS/n) + 2 * EDF

    Parameters
    ----------
    lambda_ : float
        Smoothing parameter
    X, y, S, weights
        As in gcv_criterion

    Returns
    -------
    aic : float
        AIC value
    """
    if lambda_ <= 0:
        return np.inf

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    n = len(y)

    if weights is None:
        weights = np.ones(n)

    # Fit and compute RSS
    W_sqrt = np.sqrt(weights)
    XW = X * W_sqrt[:, None]
    yW = y * weights

    XtWX = XW.T @ XW
    Xty = X.T @ yW
    penalized = XtWX + lambda_ * S

    try:
        c, lower = cho_factor(penalized, lower=True)
        beta = cho_solve((c, lower), Xty)

        fitted = X @ beta
        residuals = y - fitted
        rss = float(np.sum(weights * residuals**2))

        # EDF
        inv_XtWX = cho_solve((c, lower), XtWX)
        edf = float(np.trace(inv_XtWX))

        # AIC
        aic = n * np.log(rss / n) + 2.0 * edf

    except (np.linalg.LinAlgError, ValueError):
        return np.inf

    return aic


def select_lambda_aic(
    X: Array,
    y: Array,
    S: Array,
    weights: Optional[Array] = None,
    lambda_min: float = 1e-8,
    lambda_max: float = 1e8,
) -> SmoothingResult:
    """Select optimal smoothing parameter using AIC.

    Parameters
    ----------
    X, y, S, weights
        As in select_lambda_gcv
    lambda_min, lambda_max
        Search bounds

    Returns
    -------
    result : SmoothingResult
        Optimization result
    """

    def objective(log_lambda):
        return aic_criterion(np.exp(log_lambda), X, y, S, weights)

    result = minimize_scalar(
        objective, bounds=(np.log(lambda_min), np.log(lambda_max)), method="bounded"
    )

    lambda_opt = np.exp(result.x)
    aic_value = result.fun
    edf = compute_edf_fast(X, S, lambda_opt, weights)

    return SmoothingResult(
        lambda_opt=lambda_opt,
        edf=edf,
        criterion_value=aic_value,
        method="AIC",
        converged=result.success,
        n_iterations=result.nfev,
    )


# =============================================================================
# BIC-based Selection（UBRE 风格，用 log(n) 替换常数 2）
# =============================================================================


def bic_criterion(
    lambda_: float,
    X: Array,
    y: Array,
    S: Array,
    weights: Optional[Array] = None,
    sigma2: Optional[float] = None,
) -> float:
    """计算给定平滑参数对应的 BIC 准则值（UBRE 风格）。

    BIC(λ) = RSS/n + log(n) * σ² * EDF / n - σ²

    其中 σ² 由无惩罚最小二乘的残差方差估计，
    该公式与 Wood (2011) 的 UBRE 形式相同，
    区别在于用 log(n) 取代常数 2（即经典 BIC 惩罚）。

    Parameters
    ----------
    lambda_ : float
        平滑参数（惩罚强度）
    X : array_like
        设计矩阵 (n x p)
    y : array_like
        响应向量 (n,)
    S : array_like
        惩罚矩阵 (p x p)
    weights : array_like, optional
        观测权重 (n,)
    sigma2 : float, optional
        残差方差估计；若为 None，则从无惩罚拟合中估计

    Returns
    -------
    bic : float
        BIC 准则值
    """
    if lambda_ <= 0:
        return np.inf

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    n = len(y)

    if weights is None:
        weights = np.ones(n)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    # 若未提供 σ²，由无惩罚 OLS 残差方差估计
    if sigma2 is None:
        beta0 = np.linalg.lstsq(X, y, rcond=None)[0]
        resid0 = y - X @ beta0
        sigma2 = max(float(np.mean(resid0**2)), 1e-10)

    log_n = float(np.log(n))

    # ── 惩罚最小二乘拟合（Cholesky 分解）──
    W_sqrt = np.sqrt(weights)
    XW = X * W_sqrt[:, None]
    yW = y * weights

    XtWX = XW.T @ XW
    Xty = X.T @ yW
    penalized = XtWX + lambda_ * S

    try:
        c, lower = cho_factor(penalized, lower=True)
        beta = cho_solve((c, lower), Xty)

        fitted = X @ beta
        residuals = y - fitted
        rss = float(np.sum(weights * residuals**2))

        # 有效自由度（EDF）= trace((X'WX + λS)^{-1} X'WX)
        inv_XtWX = cho_solve((c, lower), XtWX)
        edf = float(np.trace(inv_XtWX))

        # BIC 准则：RSS/n + log(n)*σ²*EDF/n - σ²
        bic = rss / n + log_n * sigma2 * edf / n - sigma2

    except (np.linalg.LinAlgError, ValueError):
        return np.inf

    return bic


def select_lambda_bic(
    X: Array,
    y: Array,
    S: Array,
    weights: Optional[Array] = None,
    lambda_min: float = 1e-8,
    lambda_max: float = 1e8,
) -> SmoothingResult:
    """使用 BIC 准则（UBRE 风格）选择最优平滑参数。

    在对数尺度上用有界最小化搜索最优 λ，
    与 select_lambda_aic 使用相同的优化框架，
    仅将 AIC 惩罚系数 2 替换为 log(n)。

    Parameters
    ----------
    X, y, S, weights
        同 select_lambda_gcv
    lambda_min, lambda_max : float
        λ 搜索范围

    Returns
    -------
    result : SmoothingResult
        包含最优 λ 及 EDF 的结果对象
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # 预先估计 σ²（无惩罚拟合），避免在内层循环中重复计算
    beta0 = np.linalg.lstsq(X, y, rcond=None)[0]
    resid0 = y - X @ beta0
    sigma2 = max(float(np.mean(resid0**2)), 1e-10)

    def objective(log_lambda):
        # 在对数尺度上搜索，保证 λ > 0
        return bic_criterion(np.exp(log_lambda), X, y, S, weights, sigma2=sigma2)

    result = minimize_scalar(
        objective,
        bounds=(np.log(lambda_min), np.log(lambda_max)),
        method="bounded",
    )

    lambda_opt = np.exp(result.x)
    bic_value = result.fun
    # 用最优 λ 重新计算 EDF
    edf = compute_edf_fast(X, S, lambda_opt, weights)

    return SmoothingResult(
        lambda_opt=lambda_opt,
        edf=edf,
        criterion_value=bic_value,
        method="BIC",
        converged=result.success,
        n_iterations=result.nfev,
    )


# =============================================================================
# Main Selection Function
# =============================================================================


def select_smoothing_parameter(
    X: Array,
    y: Array,
    S: Array,
    weights: Optional[Array] = None,
    method: OptimizationMethod = "GCV",
    lambda_min: float = 1e-8,
    lambda_max: float = 1e8,
    **kwargs,
) -> SmoothingResult:
    """Select optimal smoothing parameter using specified method.

    This is the main entry point for automatic smoothing parameter selection.

    Parameters
    ----------
    X : array_like
        Design matrix (n x p)
    y : array_like
        Response vector (n,)
    S : array_like
        Penalty matrix (p x p)
    weights : array_like, optional
        Observation weights (n,)
    method : {"GCV", "REML", "AIC", "BIC"}, default="GCV"
        Selection method
    lambda_min : float, default=1e-8
        Minimum smoothing parameter
    lambda_max : float, default=1e8
        Maximum smoothing parameter
    **kwargs
        Additional arguments passed to specific methods

    Returns
    -------
    result : SmoothingResult
        Optimization result with optimal lambda and EDF

    Examples
    --------
    >>> # Generate example data
    >>> n, p = 100, 10
    >>> X = np.random.randn(n, p)
    >>> y = X @ np.random.randn(p) + np.random.randn(n) * 0.1
    >>> S = np.eye(p)
    >>>
    >>> # Select lambda using GCV
    >>> result = select_smoothing_parameter(X, y, S, method="GCV")
    >>> print(f"Optimal lambda: {result.lambda_opt:.4f}")
    >>> print(f"EDF: {result.edf:.2f}")
    >>>
    >>> # Try REML
    >>> result_reml = select_smoothing_parameter(X, y, S, method="REML")
    >>> print(f"REML lambda: {result_reml.lambda_opt:.4f}")
    """
    if method == "GCV":
        return select_lambda_gcv(X, y, S, weights, lambda_min, lambda_max, **kwargs)
    elif method == "REML":
        return select_lambda_reml(X, y, S, weights, lambda_min, lambda_max)
    elif method == "AIC":
        return select_lambda_aic(X, y, S, weights, lambda_min, lambda_max)
    elif method == "BIC":
        # BIC is similar to AIC but with different penalty
        # BIC = n * log(RSS/n) + log(n) * EDF
        # We can implement this similarly to AIC
        raise NotImplementedError("BIC method not yet implemented")
    else:
        raise ValueError(f"Unknown method: {method}. Choose from: GCV, REML, AIC, BIC")


# =============================================================================
# Export
# =============================================================================

__all__ = [
    "SmoothingResult",
    "select_smoothing_parameter",
    "select_lambda_gcv",
    "select_lambda_reml",
    "select_lambda_aic",
    "gcv_criterion",
    "reml_criterion",
    "aic_criterion",
    "compute_hat_matrix",
    "compute_edf",
    "compute_edf_fast",
]
