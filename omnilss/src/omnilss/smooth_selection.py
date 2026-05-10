"""平滑参数自动选择

实现 GCV (Generalized Cross-Validation) 和 REML (Restricted Maximum Likelihood) 方法
用于自动选择最优平滑参数。

这是解决 OmniLSS 最大痛点的关键模块：
- R gamlss: gamlss(y ~ pb(x))  # 自动选择 df
- OmniLSS 目标: gamlss("y ~ s(x)", ...)  # 自动选择 df

References:
    - Wood, S.N. (2017). Generalized Additive Models: An Introduction with R (2nd ed.)
    - Rigby, R.A. and Stasinopoulos, D.M. (2005). Generalized additive models for location, scale and shape
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional
from scipy.optimize import minimize_scalar
import warnings


def gcv_score(
    lambda_param: float,
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    S: np.ndarray,
    eps: float = 1e-10
) -> float:
    """计算 GCV (Generalized Cross-Validation) 分数
    
    GCV 准则用于选择最优平滑参数，平衡拟合质量和模型复杂度。
    
    GCV = RSS / (n * (1 - df/n)^2)
    
    其中:
    - RSS: 残差平方和
    - df: 有效自由度 (effective degrees of freedom)
    - n: 样本数量
    
    Parameters
    ----------
    lambda_param : float
        平滑参数 (λ)，控制平滑程度
        - λ → 0: 无平滑，插值
        - λ → ∞: 最大平滑，线性
    X : np.ndarray, shape (n, p)
        设计矩阵
    y : np.ndarray, shape (n,)
        响应变量
    w : np.ndarray, shape (n,)
        观测权重
    S : np.ndarray, shape (p, p)
        惩罚矩阵（通常是差分矩阵）
    eps : float, default=1e-10
        数值稳定性参数
    
    Returns
    -------
    gcv : float
        GCV 分数（越小越好）
        如果计算失败返回 np.inf
    
    Examples
    --------
    >>> import numpy as np
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.linspace(0, 1, n)])
    >>> y = np.sin(2 * np.pi * np.linspace(0, 1, n)) + np.random.randn(n) * 0.1
    >>> w = np.ones(n)
    >>> S = np.eye(2)
    >>> gcv = gcv_score(lambda_param=0.1, X=X, y=y, w=w, S=S)
    >>> print(f"GCV score: {gcv:.4f}")
    
    Notes
    -----
    GCV 是留一交叉验证 (LOOCV) 的高效近似，避免了重复拟合模型。
    """
    n = len(y)
    
    # 输入验证
    if lambda_param < 0:
        return np.inf
    
    try:
        # 加权设计矩阵: W^{1/2} X
        sqrt_w = np.sqrt(w)
        Xw = X * sqrt_w[:, None]
        yw = y * sqrt_w
        
        # 惩罚最小二乘: (X'WX + λS)β = X'Wy
        XtWX = Xw.T @ Xw
        penalty = lambda_param * S
        
        # 求解系数
        coef = np.linalg.solve(XtWX + penalty, Xw.T @ yw)
        
        # 拟合值
        fitted = X @ coef
        
        # 残差平方和 (RSS)
        residuals = y - fitted
        rss = np.sum(w * residuals ** 2)
        
        # 有效自由度 (effective degrees of freedom)
        # df = trace(H) where H = X(X'WX + λS)^{-1}X'W
        # 这是影响矩阵 (hat matrix) 的迹
        try:
            # 计算 (X'WX + λS)^{-1} X'W
            inv_term = np.linalg.solve(XtWX + penalty, Xw.T)
            # H = X @ inv_term
            # trace(H) = trace(inv_term @ X)
            edf = np.trace(inv_term @ Xw)
        except np.linalg.LinAlgError:
            # 如果求逆失败，使用近似
            edf = np.trace(XtWX @ np.linalg.pinv(XtWX + penalty))
        
        # 确保 edf 在合理范围内
        edf = np.clip(edf, eps, n - eps)
        
        # GCV 分数
        # 分母 (1 - df/n)^2 惩罚高自由度模型
        denominator = (1 - edf / n) ** 2
        
        if denominator < eps:
            return np.inf
        
        gcv = rss / (n * denominator)
        
        # 数值稳定性检查
        if not np.isfinite(gcv) or gcv < 0:
            return np.inf
        
        return float(gcv)
        
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return np.inf


def reml_score(
    lambda_param: float,
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    S: np.ndarray,
    eps: float = 1e-10
) -> float:
    """计算 REML (Restricted Maximum Likelihood) 分数
    
    REML 通过最大化边际似然来选择平滑参数，考虑了参数估计的不确定性。
    
    -2 * log(REML) = log|X'WX + λS| - log|X'WX| + n*log(RSS)
    
    Parameters
    ----------
    lambda_param : float
        平滑参数
    X : np.ndarray, shape (n, p)
        设计矩阵
    y : np.ndarray, shape (n,)
        响应变量
    w : np.ndarray, shape (n,)
        观测权重
    S : np.ndarray, shape (p, p)
        惩罚矩阵
    eps : float, default=1e-10
        数值稳定性参数
    
    Returns
    -------
    reml : float
        REML 分数（越小越好）
        如果计算失败返回 np.inf
    
    Examples
    --------
    >>> import numpy as np
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.linspace(0, 1, n)])
    >>> y = np.sin(2 * np.pi * np.linspace(0, 1, n)) + np.random.randn(n) * 0.1
    >>> w = np.ones(n)
    >>> S = np.eye(2)
    >>> reml = reml_score(lambda_param=0.1, X=X, y=y, w=w, S=S)
    >>> print(f"REML score: {reml:.4f}")
    
    Notes
    -----
    REML 通常比 GCV 更稳定，特别是在小样本情况下。
    它考虑了固定效应参数估计的不确定性。
    """
    n = len(y)
    
    # 输入验证
    if lambda_param < 0:
        return np.inf
    
    try:
        # 加权设计矩阵
        sqrt_w = np.sqrt(w)
        Xw = X * sqrt_w[:, None]
        yw = y * sqrt_w
        
        # 惩罚最小二乘
        XtWX = Xw.T @ Xw
        penalty = lambda_param * S
        
        # 求解系数
        coef = np.linalg.solve(XtWX + penalty, Xw.T @ yw)
        
        # 拟合值和残差
        fitted = X @ coef
        residuals = y - fitted
        rss = np.sum(w * residuals ** 2)
        
        # 确保 RSS > 0
        if rss < eps:
            rss = eps
        
        # REML 分数的三个组成部分
        # 1. log|X'WX + λS|
        _, logdet1 = np.linalg.slogdet(XtWX + penalty)
        
        # 2. log|X'WX|
        # 添加小的正则化以确保正定
        XtWX_reg = XtWX + eps * np.eye(XtWX.shape[0])
        _, logdet2 = np.linalg.slogdet(XtWX_reg)
        
        # 3. n * log(RSS)
        log_rss = n * np.log(rss)
        
        # REML 分数
        reml = logdet1 - logdet2 + log_rss
        
        # 数值稳定性检查
        if not np.isfinite(reml):
            return np.inf
        
        return float(reml)
        
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return np.inf


def select_lambda_gcv(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    S: np.ndarray,
    lambda_range: Tuple[float, float] = (1e-6, 1e6),
    verbose: bool = False
) -> Tuple[float, float]:
    """使用 GCV 选择最优平滑参数
    
    通过最小化 GCV 分数来自动选择平滑参数 λ。
    
    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        设计矩阵
    y : np.ndarray, shape (n,)
        响应变量
    w : np.ndarray, shape (n,)
        观测权重
    S : np.ndarray, shape (p, p)
        惩罚矩阵
    lambda_range : tuple of float, default=(1e-6, 1e6)
        搜索范围 (lambda_min, lambda_max)
        - 太小: 过拟合
        - 太大: 欠拟合
    verbose : bool, default=False
        是否打印优化过程
    
    Returns
    -------
    lambda_opt : float
        最优平滑参数
    gcv_opt : float
        最优 GCV 分数
    
    Examples
    --------
    >>> import numpy as np
    >>> from omnilss.smooth_selection import select_lambda_gcv
    >>> 
    >>> # 生成测试数据
    >>> n = 100
    >>> x = np.linspace(0, 1, n)
    >>> y = np.sin(2 * np.pi * x) + np.random.randn(n) * 0.1
    >>> 
    >>> # 构建 B-spline 基
    >>> from scipy.interpolate import BSpline
    >>> k = 3  # 三次样条
    >>> nknots = 10
    >>> knots = np.linspace(0, 1, nknots)
    >>> # ... 构建设计矩阵 X 和惩罚矩阵 S ...
    >>> 
    >>> # 自动选择 lambda
    >>> w = np.ones(n)
    >>> lambda_opt, gcv_opt = select_lambda_gcv(X, y, w, S, verbose=True)
    >>> print(f"Optimal lambda: {lambda_opt:.6f}")
    >>> print(f"Optimal GCV: {gcv_opt:.6f}")
    
    Notes
    -----
    使用 scipy.optimize.minimize_scalar 的 bounded 方法进行优化。
    搜索在对数空间进行以提高数值稳定性。
    """
    # 在对数空间搜索以提高数值稳定性
    log_lambda_range = (np.log(lambda_range[0]), np.log(lambda_range[1]))
    
    def objective(log_lambda):
        lambda_val = np.exp(log_lambda)
        gcv = gcv_score(lambda_val, X, y, w, S)
        if verbose:
            print(f"  λ = {lambda_val:.6e}, GCV = {gcv:.6f}")
        return gcv
    
    if verbose:
        print("GCV optimization:")
    
    # 使用 bounded 方法在指定范围内搜索
    result = minimize_scalar(
        objective,
        bounds=log_lambda_range,
        method='bounded',
        options={'xatol': 1e-6}
    )
    
    lambda_opt = np.exp(result.x)
    gcv_opt = result.fun
    
    if verbose:
        print(f"Optimal: λ = {lambda_opt:.6e}, GCV = {gcv_opt:.6f}")
    
    # 检查是否收敛到边界
    if abs(lambda_opt - lambda_range[0]) < 1e-8:
        warnings.warn(
            f"Optimal lambda ({lambda_opt:.2e}) is at lower bound. "
            "Consider decreasing lambda_range[0].",
            UserWarning
        )
    elif abs(lambda_opt - lambda_range[1]) < 1e-8:
        warnings.warn(
            f"Optimal lambda ({lambda_opt:.2e}) is at upper bound. "
            "Consider increasing lambda_range[1].",
            UserWarning
        )
    
    return lambda_opt, gcv_opt


def select_lambda_reml(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    S: np.ndarray,
    lambda_range: Tuple[float, float] = (1e-6, 1e6),
    verbose: bool = False
) -> Tuple[float, float]:
    """使用 REML 选择最优平滑参数
    
    通过最小化 REML 分数来自动选择平滑参数 λ。
    
    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        设计矩阵
    y : np.ndarray, shape (n,)
        响应变量
    w : np.ndarray, shape (n,)
        观测权重
    S : np.ndarray, shape (p, p)
        惩罚矩阵
    lambda_range : tuple of float, default=(1e-6, 1e6)
        搜索范围
    verbose : bool, default=False
        是否打印优化过程
    
    Returns
    -------
    lambda_opt : float
        最优平滑参数
    reml_opt : float
        最优 REML 分数
    
    Examples
    --------
    >>> lambda_opt, reml_opt = select_lambda_reml(X, y, w, S, verbose=True)
    
    Notes
    -----
    REML 通常比 GCV 更稳定，推荐用于小样本情况。
    """
    # 在对数空间搜索
    log_lambda_range = (np.log(lambda_range[0]), np.log(lambda_range[1]))
    
    def objective(log_lambda):
        lambda_val = np.exp(log_lambda)
        reml = reml_score(lambda_val, X, y, w, S)
        if verbose:
            print(f"  λ = {lambda_val:.6e}, REML = {reml:.6f}")
        return reml
    
    if verbose:
        print("REML optimization:")
    
    result = minimize_scalar(
        objective,
        bounds=log_lambda_range,
        method='bounded',
        options={'xatol': 1e-6}
    )
    
    lambda_opt = np.exp(result.x)
    reml_opt = result.fun
    
    if verbose:
        print(f"Optimal: λ = {lambda_opt:.6e}, REML = {reml_opt:.6f}")
    
    # 检查边界
    if abs(lambda_opt - lambda_range[0]) < 1e-8:
        warnings.warn(
            f"Optimal lambda ({lambda_opt:.2e}) is at lower bound. "
            "Consider decreasing lambda_range[0].",
            UserWarning
        )
    elif abs(lambda_opt - lambda_range[1]) < 1e-8:
        warnings.warn(
            f"Optimal lambda ({lambda_opt:.2e}) is at upper bound. "
            "Consider increasing lambda_range[1].",
            UserWarning
        )
    
    return lambda_opt, reml_opt


def select_lambda(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    S: np.ndarray,
    method: str = "GCV",
    lambda_range: Tuple[float, float] = (1e-6, 1e6),
    verbose: bool = False
) -> Tuple[float, float]:
    """统一接口：自动选择平滑参数
    
    Parameters
    ----------
    X : np.ndarray
        设计矩阵
    y : np.ndarray
        响应变量
    w : np.ndarray
        权重
    S : np.ndarray
        惩罚矩阵
    method : str, default="GCV"
        选择方法: "GCV" 或 "REML"
    lambda_range : tuple
        搜索范围
    verbose : bool
        是否打印过程
    
    Returns
    -------
    lambda_opt : float
        最优平滑参数
    score_opt : float
        最优分数
    
    Raises
    ------
    ValueError
        如果 method 不是 "GCV" 或 "REML"
    
    Examples
    --------
    >>> # 使用 GCV
    >>> lambda_opt, _ = select_lambda(X, y, w, S, method="GCV")
    >>> 
    >>> # 使用 REML
    >>> lambda_opt, _ = select_lambda(X, y, w, S, method="REML")
    """
    method = method.upper()
    
    if method == "GCV":
        return select_lambda_gcv(X, y, w, S, lambda_range, verbose)
    elif method == "REML":
        return select_lambda_reml(X, y, w, S, lambda_range, verbose)
    else:
        raise ValueError(f"Unknown method: {method}. Must be 'GCV' or 'REML'.")
