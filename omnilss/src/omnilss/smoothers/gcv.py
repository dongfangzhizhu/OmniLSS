"""GCV (Generalized Cross-Validation) for smoothing parameter selection.

This module implements the GCV criterion for automatic selection of smoothing
parameters (λ) in penalized regression splines.

GCV is a computationally efficient approximation to leave-one-out cross-validation
that doesn't require refitting the model n times.

References
----------
Craven, P., & Wahba, G. (1978). Smoothing noisy data with spline functions.
    Numerische mathematik, 31(4), 377-403.
Wood, S. N. (2017). Generalized Additive Models: An Introduction with R (2nd ed.).
    Chapman and Hall/CRC.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize_scalar


def trace_hat_matrix_efficient(X: np.ndarray, S: np.ndarray, lam: float) -> float:
    """高效计算 tr(H)，不构建完整 n×n 帽子矩阵。

    tr(H) = ||L⁻¹ X'||_F²，其中 L 是 Cholesky 因子。
    复杂度 O(np²) 而非 O(n²p)。
    """
    A = X.T @ X + lam * S
    try:
        L = np.linalg.cholesky(A)
        LInv_Xt = np.linalg.solve(L, X.T)
        return float(np.sum(LInv_Xt**2))
    except np.linalg.LinAlgError:
        H = X @ np.linalg.solve(A, X.T)
        return float(np.trace(H))


def compute_hat_matrix(X: np.ndarray, S: np.ndarray, lambda_: float) -> np.ndarray:
    """Compute the hat matrix H = X(X'X + λS)^(-1)X'.

    The hat matrix maps observed values to fitted values: ŷ = Hy

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n, p)
    S : np.ndarray
        Penalty matrix of shape (p, p)
    lambda_ : float
        Smoothing parameter

    Returns
    -------
    H : np.ndarray
        Hat matrix of shape (n, n)

    Notes
    -----
    For numerical stability, we use the Cholesky decomposition when possible.
    """
    n, p = X.shape

    # Compute X'X + λS
    XtX = X.T @ X
    penalized_XtX = XtX + lambda_ * S

    try:
        # Try Cholesky decomposition for numerical stability
        L = np.linalg.cholesky(penalized_XtX)
        # Solve (X'X + λS)^(-1)X' using forward and backward substitution
        XtX_inv_Xt = np.linalg.solve(L.T, np.linalg.solve(L, X.T))
    except np.linalg.LinAlgError:
        # Fall back to direct inversion if Cholesky fails
        warnings.warn("Cholesky decomposition failed, using direct inversion")
        XtX_inv_Xt = np.linalg.solve(penalized_XtX, X.T)

    # H = X(X'X + λS)^(-1)X'
    H = X @ XtX_inv_Xt

    return H


def compute_effective_df(H: np.ndarray) -> float:
    """Compute effective degrees of freedom: tr(H).

    Parameters
    ----------
    H : np.ndarray
        Hat matrix of shape (n, n)

    Returns
    -------
    edf : float
        Effective degrees of freedom

    Notes
    -----
    The effective degrees of freedom measures the model complexity.
    For ordinary least squares, edf = p (number of parameters).
    For penalized regression, edf < p due to shrinkage.
    """
    return np.trace(H)


def compute_gcv_score(
    y: np.ndarray, X: np.ndarray, S: np.ndarray, lambda_: float
) -> float:
    """Compute the GCV score for a given smoothing parameter.

    GCV(λ) = n * RSS(λ) / (n - tr(H(λ)))²

    where:
    - RSS: residual sum of squares
    - H: hat matrix
    - tr(H): effective degrees of freedom

    Parameters
    ----------
    y : np.ndarray
        Response vector of shape (n,)
    X : np.ndarray
        Design matrix of shape (n, p)
    S : np.ndarray
        Penalty matrix of shape (p, p)
    lambda_ : float
        Smoothing parameter (must be positive)

    Returns
    -------
    gcv : float
        GCV score (lower is better)

    Notes
    -----
    The GCV score is an estimate of the prediction error.
    We want to minimize this score to find the optimal λ.

    For numerical stability:
    - We add a small constant to avoid division by zero
    - We handle the case where edf ≈ n (overfitting)
    """
    n = len(y)

    # Ensure lambda is positive
    if lambda_ <= 0:
        return np.inf

    # Compute hat matrix
    H = compute_hat_matrix(X, S, lambda_)

    # Compute fitted values
    y_hat = H @ y

    # Compute residual sum of squares
    residuals = y - y_hat
    rss = np.sum(residuals**2)

    # Compute effective degrees of freedom
    edf = compute_effective_df(H)

    # Compute GCV score
    # Add small constant to denominator for numerical stability
    denominator = n - edf
    if denominator <= 0:
        # Model is too complex (overfitting)
        return np.inf

    gcv = n * rss / (denominator**2)

    return gcv


def optimize_lambda_gcv(
    y: np.ndarray,
    X: np.ndarray,
    S: np.ndarray,
    lambda_range: Tuple[float, float] = (1e-6, 1e6),
    log_space: bool = True,
) -> Tuple[float, float]:
    """Find the optimal smoothing parameter by minimizing GCV.

    Parameters
    ----------
    y : np.ndarray
        Response vector of shape (n,)
    X : np.ndarray
        Design matrix of shape (n, p)
    S : np.ndarray
        Penalty matrix of shape (p, p)
    lambda_range : tuple of float, default=(1e-6, 1e6)
        Search range for λ
    log_space : bool, default=True
        If True, search in log10 space for better numerical stability

    Returns
    -------
    lambda_opt : float
        Optimal smoothing parameter
    gcv_opt : float
        GCV score at optimal λ

    Notes
    -----
    We search in log10 space by default because:
    1. λ can span many orders of magnitude
    2. The GCV curve is often smoother in log space
    3. It's more numerically stable

    Examples
    --------
    >>> import numpy as np
    >>> from omnilss.smoothers.gcv import optimize_lambda_gcv
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> x = np.linspace(0, 1, n)
    >>> y = np.sin(2 * np.pi * x) + np.random.randn(n) * 0.1
    >>>
    >>> # Create design matrix (B-spline basis)
    >>> from omnilss.smoothers.bspline import bspline_basis
    >>> knots = np.linspace(0, 1, 10)
    >>> X = np.array(bspline_basis(x, knots, degree=3))
    >>>
    >>> # Create penalty matrix
    >>> from omnilss.smoothers.penalty import penalty_matrix
    >>> p = X.shape[1]
    >>> S = penalty_matrix(p, order=2)
    >>>
    >>> # Find optimal λ
    >>> lambda_opt, gcv_opt = optimize_lambda_gcv(y, X, S)
    >>> print(f"Optimal λ: {lambda_opt:.4f}")
    >>> print(f"GCV score: {gcv_opt:.4f}")
    """
    if log_space:
        # Search in log10 space
        log_lambda_min = np.log10(lambda_range[0])
        log_lambda_max = np.log10(lambda_range[1])

        def objective(log_lambda):
            lambda_ = 10**log_lambda
            return compute_gcv_score(y, X, S, lambda_)

        # Minimize GCV
        result = minimize_scalar(
            objective, bounds=(log_lambda_min, log_lambda_max), method="bounded"
        )

        lambda_opt = 10**result.x
        gcv_opt = result.fun
    else:
        # Search in linear space
        def objective(lambda_):
            return compute_gcv_score(y, X, S, lambda_)

        # Minimize GCV
        result = minimize_scalar(objective, bounds=lambda_range, method="bounded")

        lambda_opt = result.x
        gcv_opt = result.fun

    return lambda_opt, gcv_opt


def gcv_multiple_smooths(
    y: np.ndarray,
    X_list: list[np.ndarray],
    S_list: list[np.ndarray],
    lambda_init: Optional[list[float]] = None,
    max_iter: int = 20,
    tol: float = 1e-4,
) -> list[float]:
    """用坐标下降 + GCV 优化多个平滑参数。

    每轮依次优化 λ_i，固定其余 λ_j（坐标下降）。
    修复原版：原版传入的是 S_full（全局惩罚矩阵），
    导致每轮坐标步优化的目标函数完全相同，坐标下降失效。
    """
    n_smooths = len(X_list)
    if lambda_init is None:
        lambdas = [1.0] * n_smooths
    else:
        lambdas = list(lambda_init)

    X_full = np.hstack(X_list)

    # 预计算各平滑项的列偏移
    col_offsets = []
    offset = 0
    for X in X_list:
        col_offsets.append(offset)
        offset += X.shape[1]
    total_cols = X_full.shape[1]

    def build_S_total(idx_vary: int, lam_i: float) -> np.ndarray:
        """构建完整块对角惩罚矩阵。第 idx_vary 块用 lam_i，其余用当前 lambdas。"""
        S_total = np.zeros((total_cols, total_cols))
        for j, S_j in enumerate(S_list):
            c = col_offsets[j]
            p = S_j.shape[0]
            lam_j = lam_i if j == idx_vary else lambdas[j]
            S_total[c : c + p, c : c + p] = lam_j * S_j
        return S_total

    def gcv_with_full_S(y, X, S_total):
        """用 λ=1 计算 GCV，因为 λ 已吸收进 S_total。
        用 lstsq + 自适应 ridge 提高数値稳定性。
        """
        n_obs = len(y)
        XtX = X.T @ X
        # 自适应尘脊正则化
        diag_scale = max(float(np.max(np.abs(np.diag(XtX)))), 1e-8)
        ridge = max(1e-8 * diag_scale, 1e-10)
        A = XtX + S_total + ridge * np.eye(XtX.shape[0])

        # 用 lstsq 高效计算 tr(H) 和拟合値（处理潜在奇异矩阵）
        LInv_Xt, _, _, _ = np.linalg.lstsq(A, X.T, rcond=None)
        edf = float(np.sum(X * LInv_Xt.T))  # tr(H) = sum(X * (A^-1 X')')
        Xy = X.T @ y
        beta_hat, _, _, _ = np.linalg.lstsq(A, Xy, rcond=None)
        y_hat = X @ beta_hat

        rss = float(np.sum((y - y_hat) ** 2))
        denom = n_obs - edf
        if denom <= 0:
            return np.inf
        return n_obs * rss / (denom**2)

    # 坐标下降主循环
    for iteration in range(max_iter):
        lambdas_old = lambdas.copy()

        for i in range(n_smooths):

            def score_i(log_lam_i, _i=i):
                lam_i = 10.0**log_lam_i
                S_total = build_S_total(_i, lam_i)
                return gcv_with_full_S(y, X_full, S_total)

            result = minimize_scalar(
                score_i, bounds=(-6.0, 6.0), method="bounded", options={"xatol": 1e-5}
            )
            lambdas[i] = 10.0**result.x

        # 检查收敛
        max_change = max(
            abs(lambdas[j] - lambdas_old[j]) / (lambdas_old[j] + 1e-8)
            for j in range(n_smooths)
        )
        if max_change < tol:
            break

    return lambdas
