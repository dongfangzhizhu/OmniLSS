"""REML (Restricted Maximum Likelihood) for smoothing parameter selection.

This module implements the REML criterion for automatic selection of smoothing
parameters (λ) in penalized regression splines.

REML is theoretically more principled than GCV and often performs better,
especially for small sample sizes. It's the default method in mgcv.

References
----------
Wood, S. N. (2011). Fast stable restricted maximum likelihood and marginal
    likelihood estimation of semiparametric generalized linear models.
    Journal of the Royal Statistical Society: Series B, 73(1), 3-36.
Wood, S. N. (2017). Generalized Additive Models: An Introduction with R (2nd ed.).
    Chapman and Hall/CRC.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize_scalar


def compute_reml_score(
    y: np.ndarray, X: np.ndarray, S: np.ndarray, lambda_: float
) -> float:
    """Compute the REML score for a given smoothing parameter.

    REML(λ) = log|X'X| + log|X'X + λS| + y'(I - H)y

    where:
    - X'X: information matrix
    - S: penalty matrix
    - H: hat matrix = X(X'X + λS)^(-1)X'
    - y'(I - H)y: penalized residual sum of squares

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
    reml : float
        REML score (lower is better)

    Notes
    -----
    The REML score is derived from the restricted likelihood, which
    integrates out the fixed effects. This makes it more robust than
    GCV for small sample sizes.

    For numerical stability:
    - We use log-determinants to avoid overflow
    - We add a small constant to avoid singularity
    - We use Cholesky decomposition when possible

    The REML criterion balances:
    1. Model fit: y'(I - H)y (smaller is better)
    2. Model complexity: log|X'X + λS| (penalizes complexity)
    3. Design information: log|X'X| (accounts for design)
    """
    n, p = X.shape

    # Ensure lambda is positive
    if lambda_ <= 0:
        return np.inf

    # Compute X'X
    XtX = X.T @ X

    # 数值稳定尘脊正则化：基于对角元素同比例，抵抗奇异贡献
    diag_scale = max(np.max(np.abs(np.diag(XtX))), 1e-8)
    ridge = max(1e-8 * diag_scale, 1e-10)
    XtX_stable = XtX + ridge * np.eye(p)

    # log|X'X|（用特征分解居童，需处理负特征値）
    eigenvalues_XtX = np.linalg.eigvalsh(XtX_stable)
    eigenvalues_XtX = np.maximum(eigenvalues_XtX, ridge)
    log_det_XtX = np.sum(np.log(eigenvalues_XtX))

    # X'X + λS：加入一个自适应 ridge 确保正定性
    penalized_XtX = XtX + lambda_ * S + ridge * np.eye(p)

    # log|X'X + λS|（用特征分解）
    eigenvalues_pen = np.linalg.eigvalsh(penalized_XtX)
    eigenvalues_pen = np.maximum(eigenvalues_pen, ridge)
    log_det_pen = np.sum(np.log(eigenvalues_pen))

    # (X'X + λS)^{-1} X'：用 lstsq 处理潜在奇异情况
    XtX_inv_Xt, _, _, _ = np.linalg.lstsq(penalized_XtX, X.T, rcond=None)

    # 帽子矩阵 H = X (X'X + λS)^{-1} X'
    H = X @ XtX_inv_Xt

    # 残差平方和
    residuals = y - H @ y
    prss = float(np.sum(residuals**2))

    # REML 准则（近似 Wood 2011 简化版）
    reml = log_det_XtX + log_det_pen + prss

    return float(reml)


def optimize_lambda_reml(
    y: np.ndarray,
    X: np.ndarray,
    S: np.ndarray,
    lambda_range: Tuple[float, float] = (1e-6, 1e6),
    log_space: bool = True,
) -> Tuple[float, float]:
    """Find the optimal smoothing parameter by minimizing REML.

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
    reml_opt : float
        REML score at optimal λ

    Notes
    -----
    REML is generally preferred over GCV because:
    1. More theoretically principled (derived from likelihood)
    2. Better for small sample sizes
    3. More robust to model misspecification
    4. Default method in mgcv

    Like GCV, we search in log10 space by default for numerical stability.

    Examples
    --------
    >>> import numpy as np
    >>> from omnilss.smoothers.reml import optimize_lambda_reml
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> x = np.linspace(0, 1, n)
    >>> y = np.sin(2 * np.pi * x) + np.random.randn(n) * 0.1
    >>>
    >>> # Create design matrix (B-spline basis)
    >>> from omnilss.smoothers.bsplines import bspline_basis
    >>> knots = np.linspace(0, 1, 10)
    >>> X = np.array(bspline_basis(x, knots, degree=3))
    >>>
    >>> # Create penalty matrix
    >>> from omnilss.smoothers.penalties import penalty_matrix
    >>> p = X.shape[1]
    >>> S = penalty_matrix(p, order=2)
    >>>
    >>> # Find optimal λ
    >>> lambda_opt, reml_opt = optimize_lambda_reml(y, X, S)
    >>> print(f"Optimal λ: {lambda_opt:.4f}")
    >>> print(f"REML score: {reml_opt:.4f}")
    """
    if log_space:
        # Search in log10 space
        log_lambda_min = np.log10(lambda_range[0])
        log_lambda_max = np.log10(lambda_range[1])

        def objective(log_lambda):
            lambda_ = 10**log_lambda
            return compute_reml_score(y, X, S, lambda_)

        # Minimize REML
        result = minimize_scalar(
            objective, bounds=(log_lambda_min, log_lambda_max), method="bounded"
        )

        lambda_opt = 10**result.x
        reml_opt = result.fun
    else:
        # Search in linear space
        def objective(lambda_):
            return compute_reml_score(y, X, S, lambda_)

        # Minimize REML
        result = minimize_scalar(objective, bounds=lambda_range, method="bounded")

        lambda_opt = result.x
        reml_opt = result.fun

    return lambda_opt, reml_opt


def compare_gcv_reml(
    y: np.ndarray,
    X: np.ndarray,
    S: np.ndarray,
    lambda_range: Tuple[float, float] = (1e-6, 1e6),
) -> dict:
    """Compare GCV and REML methods for λ selection.

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

    Returns
    -------
    comparison : dict
        Dictionary with keys:
        - 'gcv_lambda': optimal λ from GCV
        - 'gcv_score': GCV score at optimal λ
        - 'reml_lambda': optimal λ from REML
        - 'reml_score': REML score at optimal λ
        - 'lambda_ratio': reml_lambda / gcv_lambda

    Notes
    -----
    This function is useful for understanding the differences between
    GCV and REML. In practice:
    - REML often gives slightly larger λ (more smoothing)
    - REML is more stable for small sample sizes
    - Both methods usually give similar results for large n

    Examples
    --------
    >>> comparison = compare_gcv_reml(y, X, S)
    >>> print(f"GCV λ: {comparison['gcv_lambda']:.4f}")
    >>> print(f"REML λ: {comparison['reml_lambda']:.4f}")
    >>> print(f"Ratio: {comparison['lambda_ratio']:.2f}")
    """
    from .gcv import optimize_lambda_gcv

    # Optimize using GCV
    gcv_lambda, gcv_score = optimize_lambda_gcv(y, X, S, lambda_range)

    # Optimize using REML
    reml_lambda, reml_score = optimize_lambda_reml(y, X, S, lambda_range)

    # Compute ratio
    lambda_ratio = reml_lambda / gcv_lambda if gcv_lambda > 0 else np.inf

    return {
        "gcv_lambda": gcv_lambda,
        "gcv_score": gcv_score,
        "reml_lambda": reml_lambda,
        "reml_score": reml_score,
        "lambda_ratio": lambda_ratio,
    }


def reml_multiple_smooths(
    y: np.ndarray,
    X_list: list[np.ndarray],
    S_list: list[np.ndarray],
    lambda_init: Optional[list[float]] = None,
    max_iter: int = 20,
    tol: float = 1e-4,
) -> list[float]:
    """用坐标下降 + REML 优化多个平滑参数。

    每轮依次优化 λ_i，固定其余 λ_j（坐标下降）。
    修复原版：原版传入的是 S_full（全局惩罚矩阵），
    导致每轮坐标步优化的目标函数都完全相同，坐标下降失效。
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
        """构建完整块对角惩罚矩阵。

        第 idx_vary 块用 lam_i，其余块用当前 lambdas 中的值。
        """
        S_total = np.zeros((total_cols, total_cols))
        for j, S_j in enumerate(S_list):
            c = col_offsets[j]
            p = S_j.shape[0]
            lam_j = lam_i if j == idx_vary else lambdas[j]
            S_total[c : c + p, c : c + p] = lam_j * S_j
        return S_total

    # 坐标下降主循环
    for iteration in range(max_iter):
        lambdas_old = lambdas.copy()

        for i in range(n_smooths):
            # 构建只含第i个λ_i为自由变量的目标函数
            def score_i(log_lam_i, _i=i):
                lam_i = 10.0**log_lam_i
                S_total = build_S_total(_i, lam_i)
                # 传入 lambda_=1.0，因为 λ 已经被吸收进 S_total
                return compute_reml_score(y, X_full, S_total, 1.0)

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


def log_det_cholesky(A: np.ndarray) -> float:
    """用 Cholesky 分解稳定计算对数行列式 log|A|。

    比直接 np.log(np.linalg.det(A)) 数值上更稳定，避免上溢/下溢。

    Parameters
    ----------
    A : np.ndarray
        正定对称矩阵

    Returns
    -------
    float
        log|A|
    """
    try:
        L = np.linalg.cholesky(A)
        return 2.0 * np.sum(np.log(np.maximum(np.diag(L), 1e-300)))
    except np.linalg.LinAlgError:
        # 回退到特征值分解
        eigvals = np.linalg.eigvalsh(A)
        return np.sum(np.log(np.maximum(eigvals, 1e-300)))


def trace_hat_matrix(X: np.ndarray, S: np.ndarray, lam: float) -> float:
    """高效计算帽子矩阵的迹 tr(H)，不构建完整 n×n 矩阵。

    利用：
        tr(H) = tr(X(X'X + λS)⁻¹X')
               = tr((X'X + λS)⁻¹ X'X)
               = ||L⁻¹ X'||_F²
    其中 L 是 (X'X + λS) 的 Cholesky 因子。

    复杂度：O(np²) 而非 O(n²p)，大数据集时优势显著。

    Parameters
    ----------
    X : np.ndarray
        设计矩阵 (n, p)
    S : np.ndarray
        惩罚矩阵 (p, p)
    lam : float
        平滑参数 λ

    Returns
    -------
    float
        tr(H)
    """
    A = X.T @ X + lam * S
    try:
        L = np.linalg.cholesky(A)
        # 用前向代换求解：L \ X'  → (p, n) 矩阵
        Xt = X.T  # (p, n)
        LInv_Xt = np.linalg.solve(L, Xt)  # scipy 的 solve 底层用 LAPACK
        return float(np.sum(LInv_Xt**2))
    except np.linalg.LinAlgError:
        # 回退：直接构建帽子矩阵迹
        H = X @ np.linalg.solve(A, X.T)
        return float(np.trace(H))
