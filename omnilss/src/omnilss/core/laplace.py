"""Laplace 近似基础设施

提供数值稳定的 Laplace 近似计算，用于：
1. REML 平滑参数选择（等价于经验贝叶斯边际似然）
2. Bayesian GAMLSS 的近似后验推断
3. 模型证据（model evidence）计算

核心参考：
- Wood (2011) Fast stable REML
- Tierney & Kadane (1986) Accurate approximations for posterior moments
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from scipy import linalg as scipy_linalg

# ──────────────────────────────────────────────────────────────
# 数值线性代数基础工具
# ──────────────────────────────────────────────────────────────


def stable_cholesky(A: np.ndarray, max_attempts: int = 5) -> np.ndarray:
    """数值稳定的 Cholesky 分解，自动添加 jitter（扰动）。

    若直接 Cholesky 失败，逐步增大 jitter 重试，
    确保总能返回一个合理的下三角因子。

    Parameters
    ----------
    A : np.ndarray
        正定对称矩阵
    max_attempts : int
        最大重试次数

    Returns
    -------
    L : np.ndarray
        Cholesky 因子 L，满足 L @ L.T ≈ A
    """
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]

    # 先对称化
    A = (A + A.T) / 2

    # 基于矩阵 Frobenius 范数的初始 jitter
    jitter = 1e-10 * np.sqrt(np.mean(np.diag(A) ** 2))

    for attempt in range(max_attempts):
        try:
            L = np.linalg.cholesky(A + jitter * np.eye(n))
            return L
        except np.linalg.LinAlgError:
            jitter *= 10  # 每次失败扩大10倍

    # 终极回退：特征值修复
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, jitter)
    A_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return np.linalg.cholesky(A_pd)


def log_det_cholesky(A: np.ndarray) -> float:
    """用 Cholesky 因子稳定计算 log|A|。

    log|A| = 2 * sum(log(diag(L)))，其中 L 是 A 的 Cholesky 因子。

    Parameters
    ----------
    A : np.ndarray
        正定对称矩阵

    Returns
    -------
    float
        log|A|
    """
    L = stable_cholesky(A)
    return float(2.0 * np.sum(np.log(np.maximum(np.diag(L), 1e-300))))


def log_det_positive_semidefinite(A: np.ndarray, rank: Optional[int] = None) -> float:
    """计算正半定矩阵的秩受限对数行列式 log|A|_+。

    对惩罚矩阵（通常秩亏缺）使用此函数：
    log|A|_+ = sum(log(正特征值))

    Parameters
    ----------
    A : np.ndarray
        正半定矩阵（可以秩亏缺）
    rank : int, optional
        矩阵的秩，如果不指定则自动估计

    Returns
    -------
    float
        秩受限对数行列式 log|A|_+
    """
    eigvals = np.linalg.eigvalsh(A)

    if rank is None:
        # 自动估计秩：特征值 > 相对阈值的数量
        tol = max(eigvals.max() * len(eigvals) * np.finfo(float).eps, 1e-10)
        rank = int(np.sum(eigvals > tol))

    # 只取最大的 rank 个特征值
    positive_eigvals = np.sort(eigvals)[-rank:]
    return float(np.sum(np.log(np.maximum(positive_eigvals, 1e-300))))


def trace_hat_matrix(
    X: np.ndarray, S: np.ndarray, lam: float, W: Optional[np.ndarray] = None
) -> float:
    """高效计算加权帽子矩阵迹 tr(H_W)，不构建完整 n×n 矩阵。

    tr(H_W) = tr(X (X'WX + λS)⁻¹ X'W) = ||L⁻¹ (X'W)^{1/2}||_F²

    复杂度 O(np²) 而非 O(n²p)。

    Parameters
    ----------
    X : np.ndarray  (n, p)
        设计矩阵
    S : np.ndarray  (p, p)
        惩罚矩阵
    lam : float
        平滑参数 λ
    W : np.ndarray  (n,), optional
        观测权重，None 表示单位权重

    Returns
    -------
    float
        tr(H)
    """
    n, p = X.shape
    if W is None:
        W = np.ones(n)

    # XtWX = X'WX
    XtWX = X.T @ (X * W[:, None])
    A = XtWX + lam * S

    # L = Cholesky(A)
    L = stable_cholesky(A)

    # L⁻¹ X'W^{1/2}，然后 tr(H) = ||L⁻¹ X' W^{1/2}||_F²
    sqrtW_X = X * np.sqrt(W)[:, None]  # (n, p)
    LInv_XtW = scipy_linalg.solve_triangular(L, sqrtW_X.T, lower=True)  # (p, n)
    return float(np.sum(LInv_XtW**2))


# ──────────────────────────────────────────────────────────────
# Wood (2011) REML 准则
# ──────────────────────────────────────────────────────────────


def reml_wood2011(
    y: np.ndarray,
    X: np.ndarray,
    S_list: list,
    lambdas: list[float],
    W: Optional[np.ndarray] = None,
) -> float:
    """Wood (2011) 的 REML 准则（用于 GAM 平滑参数选择）。

    REML(λ) = -0.5 * [
        log|X'WX + S(λ)|     （广义自由度项）
        - log|X'WX|_+        （信息行列式项）
        + n log(RSS_W(λ))    （拟合优度项）
    ]

    其中 S(λ) = Σ_i λ_i S_i 是加权惩罚矩阵。

    Parameters
    ----------
    y : np.ndarray  (n,)
        响应变量（或 IRLS 工作响应）
    X : np.ndarray  (n, p)
        设计矩阵
    S_list : list of np.ndarray
        惩罚矩阵列表，每个 (p, p)
    lambdas : list of float
        对应的平滑参数列表
    W : np.ndarray  (n,), optional
        IRLS 工作权重，None 表示单位权重

    Returns
    -------
    float
        REML 准则值（越小越好）
    """
    n, p = X.shape
    if W is None:
        W = np.ones(n)

    # 构建加权惩罚矩阵 S(λ)
    S_total = sum(lam * S for lam, S in zip(lambdas, S_list))

    # XtWX = X'WX（加权信息矩阵）
    XtWX = X.T @ (X * W[:, None])

    # 数值稳定的 ridge
    diag_scale = max(float(np.max(np.abs(np.diag(XtWX)))), 1e-8)
    ridge = 1e-8 * diag_scale

    A = XtWX + S_total + ridge * np.eye(p)

    # log|X'WX + S(λ)|（用 Cholesky）
    log_det_A = log_det_cholesky(A)

    # log|X'WX|_+（秩受限，处理 X'WX 可能秩亏缺的情况）
    rank_XtWX = np.linalg.matrix_rank(XtWX + ridge * np.eye(p))
    log_det_XtWX = log_det_positive_semidefinite(
        XtWX + ridge * np.eye(p), rank=rank_XtWX
    )

    # 加权 RSS：RSS_W = y'W(I - H)y
    beta_hat, _, _, _ = np.linalg.lstsq(A, X.T @ (W * y), rcond=None)
    y_hat = X @ beta_hat
    rss_w = float(np.sum(W * (y - y_hat) ** 2))
    rss_w = max(rss_w, 1e-10)

    # REML 准则（Wood 2011，公式 (6)）
    reml = 0.5 * (log_det_A - log_det_XtWX + n * np.log(rss_w))
    return float(reml)


# ──────────────────────────────────────────────────────────────
# Laplace 近似边际似然
# ──────────────────────────────────────────────────────────────


def laplace_log_marginal(
    neg_log_joint: Callable,
    init_params: Any,
    data: Any,
    method: str = "lbfgs",
    n_samples: Optional[int] = None,
) -> float:
    """用 Laplace 近似计算对数边际似然 log p(y)。

    log p(y) ≈ log p(y|θ*) - 0.5 * log|H| + 0.5 * d * log(2π)

    其中：
    - θ* = argmax p(y|θ) p(θ)（MAP 估计）
    - H = -∇²log p(θ|y)|_{θ*}（负 Hessian，Fisher 信息矩阵的近似）
    - d = 参数维度

    使用场景：
    1. REML 平滑参数选择（等价于边际似然最大化）
    2. 模型比较（贝叶斯因子近似）
    3. Bayesian 超参数优化

    Parameters
    ----------
    neg_log_joint : Callable
        负对数联合概率函数 -log p(y, θ)，签名为 f(params, data)
    init_params : pytree
        参数初始值（JAX pytree）
    data : Any
        传递给 neg_log_joint 的数据
    method : str
        MAP 估计方法（"lbfgs" 或 "gradient"）
    n_samples : int, optional
        如果指定，用 IS 修正 Laplace 近似（暂未实现）

    Returns
    -------
    float
        近似对数边际似然
    """
    # ── MAP 估计 ──
    if method == "lbfgs":
        from ..core.lbfgs_optimizer import lbfgs_optimize

        result = lbfgs_optimize(
            loss_fn=neg_log_joint,
            init_params=init_params,
            data=data,
            max_iter=200,
            verbose=False,
        )
        theta_star = result.params
        map_val = -float(result.loss)  # log p(y, θ*)
    else:
        # 简单梯度下降 fallback
        import optax

        optimizer = optax.adam(1e-3)
        params = init_params
        opt_state = optimizer.init(params)
        for _ in range(1000):
            loss, grads = jax.value_and_grad(neg_log_joint)(params, data)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
        theta_star = params
        map_val = -float(neg_log_joint(theta_star, data))

    # ── Hessian（在 MAP 点评估）──
    flat_params, unravel = jax.flatten_util.ravel_pytree(theta_star)
    n_params = len(flat_params)

    def neg_log_joint_flat(flat_p, data_):
        return neg_log_joint(unravel(flat_p), data_)

    try:
        H_flat = jax.hessian(neg_log_joint_flat)(flat_params, data)
        H_np = np.asarray(H_flat, dtype=np.float64)

        # Hessian 应为正定（我们是 MAP，最小化负对数）
        log_det_H = log_det_cholesky(H_np)
    except Exception:
        # Hessian 计算失败，返回 MAP 估计作为 fallback
        return map_val

    # ── Laplace 近似 ──
    # log p(y) ≈ log p(y, θ*) - 0.5 * log|H| + 0.5 * d * log(2π)
    log_marginal = map_val - 0.5 * log_det_H + 0.5 * n_params * np.log(2 * np.pi)
    return float(log_marginal)


def laplace_posterior_variance(
    neg_log_joint: Callable,
    theta_star: Any,
    data: Any,
) -> np.ndarray:
    """在 MAP 点近似计算后验协方差矩阵（Laplace 近似）。

    Cov(θ|y) ≈ H⁻¹，其中 H = -∇²log p(θ|y)|_{θ*}

    Parameters
    ----------
    neg_log_joint : Callable
        负对数联合概率
    theta_star : pytree
        MAP 估计值
    data : Any
        数据

    Returns
    -------
    np.ndarray
        近似后验协方差矩阵（参数化空间）
    """
    flat_params, unravel = jax.flatten_util.ravel_pytree(theta_star)

    def f_flat(p, d):
        return neg_log_joint(unravel(p), d)

    H = np.asarray(jax.hessian(f_flat)(flat_params, data))

    # H 的逆 = 近似后验协方差
    try:
        L = stable_cholesky(H)
        cov = np.linalg.inv(H)
    except Exception:
        # Fallback：pseudo-inverse
        cov = np.linalg.pinv(H)

    return cov


# ──────────────────────────────────────────────────────────────
# 导出
# ──────────────────────────────────────────────────────────────
__all__ = [
    # 数值线性代数
    "stable_cholesky",
    "log_det_cholesky",
    "log_det_positive_semidefinite",
    "trace_hat_matrix",
    # REML
    "reml_wood2011",
    # Laplace 近似
    "laplace_log_marginal",
    "laplace_posterior_variance",
]
