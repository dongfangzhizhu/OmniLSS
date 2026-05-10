"""Link functions used by migrated GAMLSS families.

R source references:
- file: `gamlss/R/gamlssML.R`
- uses family link functions such as `mu.linkfun`, `sigma.linkinv`
"""

from __future__ import annotations

import jax.numpy as jnp


def identity_link(mu: jnp.ndarray) -> jnp.ndarray:
    return jnp.asarray(mu, dtype=jnp.float64)


def identity_inverse(eta: jnp.ndarray) -> jnp.ndarray:
    return jnp.asarray(eta, dtype=jnp.float64)


def identity_derivative(eta: jnp.ndarray) -> jnp.ndarray:
    eta = jnp.asarray(eta, dtype=jnp.float64)
    return jnp.ones_like(eta, dtype=jnp.float64)


def log_link(mu: jnp.ndarray) -> jnp.ndarray:
    mu = jnp.asarray(mu, dtype=jnp.float64)
    return jnp.log(jnp.maximum(mu, jnp.finfo(jnp.float64).eps))


def log_inverse(eta: jnp.ndarray) -> jnp.ndarray:
    eta = jnp.asarray(eta, dtype=jnp.float64)
    return jnp.maximum(jnp.exp(eta), jnp.finfo(jnp.float64).eps)


def log_derivative(eta: jnp.ndarray) -> jnp.ndarray:
    eta = jnp.asarray(eta, dtype=jnp.float64)
    return jnp.maximum(jnp.exp(eta), jnp.finfo(jnp.float64).eps)


def logit_link(mu: jnp.ndarray) -> jnp.ndarray:
    mu = jnp.asarray(mu, dtype=jnp.float64)
    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.clip(mu, eps, 1.0 - eps)
    return jnp.log(mu / (1.0 - mu))


def logit_inverse(eta: jnp.ndarray) -> jnp.ndarray:
    eta = jnp.asarray(eta, dtype=jnp.float64)
    eps = jnp.finfo(jnp.float64).eps
    mu = 1.0 / (1.0 + jnp.exp(-eta))
    return jnp.clip(mu, eps, 1.0 - eps)


def logit_derivative(eta: jnp.ndarray) -> jnp.ndarray:
    mu = logit_inverse(eta)
    return mu * (1.0 - mu)


def inverse_link(mu: jnp.ndarray) -> jnp.ndarray:
    mu = jnp.asarray(mu, dtype=jnp.float64)
    eps = jnp.finfo(jnp.float64).eps
    return 1.0 / jnp.maximum(mu, eps)


def inverse_inverse(eta: jnp.ndarray) -> jnp.ndarray:
    eta = jnp.asarray(eta, dtype=jnp.float64)
    eps = jnp.finfo(jnp.float64).eps
    eta = jnp.where(jnp.abs(eta) < eps, eps, eta)
    return 1.0 / eta


def inverse_derivative(eta: jnp.ndarray) -> jnp.ndarray:
    eta = jnp.asarray(eta, dtype=jnp.float64)
    eps = jnp.finfo(jnp.float64).eps
    eta = jnp.where(jnp.abs(eta) < eps, eps, eta)
    return -1.0 / jnp.square(eta)


def probit_link(mu: jnp.ndarray) -> jnp.ndarray:
    """Probit link function: Φ^(-1)(μ)

    Maps probability μ ∈ (0,1) to real line η ∈ (-∞,∞)

    Parameters
    ----------
    mu : jnp.ndarray
        Probability values in (0, 1)

    Returns
    -------
    jnp.ndarray
        Linear predictor values on real line

    Notes
    -----
    The probit link is the inverse of the standard normal CDF.
    It is commonly used for binary response models as an alternative to logit.
    """
    from jax.scipy.stats import norm

    mu = jnp.asarray(mu, dtype=jnp.float64)
    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.clip(mu, eps, 1.0 - eps)
    return norm.ppf(mu)


def probit_inverse(eta: jnp.ndarray) -> jnp.ndarray:
    """Inverse probit link: Φ(η)

    Maps real line η ∈ (-∞,∞) to probability μ ∈ (0,1)

    Parameters
    ----------
    eta : jnp.ndarray
        Linear predictor values on real line

    Returns
    -------
    jnp.ndarray
        Probability values in (0, 1)

    Notes
    -----
    This is the standard normal CDF.
    """
    from jax.scipy.stats import norm

    eta = jnp.asarray(eta, dtype=jnp.float64)
    mu = norm.cdf(eta)
    eps = jnp.finfo(jnp.float64).eps
    return jnp.clip(mu, eps, 1.0 - eps)


def probit_derivative(eta: jnp.ndarray) -> jnp.ndarray:
    """Derivative of inverse probit: φ(η)

    Where φ is the standard normal PDF

    Parameters
    ----------
    eta : jnp.ndarray
        Linear predictor values on real line

    Returns
    -------
    jnp.ndarray
        Derivative values (always positive)

    Notes
    -----
    This is the standard normal PDF: φ(η) = (1/√(2π)) * exp(-η²/2)
    """
    from jax.scipy.stats import norm

    eta = jnp.asarray(eta, dtype=jnp.float64)
    return norm.pdf(eta)


# ─────────────────────────────────────────────
# 新增：cloglog 链接函数（互补对数-对数）
# ─────────────────────────────────────────────


def cloglog_link(mu: jnp.ndarray) -> jnp.ndarray:
    """互补对数-对数链接函数：log(-log(1 - μ))

    常用于：生存分析（比例风险），以及 Poisson 分布的关联响应。
    域：μ ∈ (0, 1) → η ∈ (-∞, ∞)
    """
    mu = jnp.asarray(mu, dtype=jnp.float64)
    eps = jnp.finfo(jnp.float64).eps
    # 将 mu 限制在开区间 (eps, 1-eps) 内，避免 log(0) 的数值问题
    mu = jnp.clip(mu, eps, 1.0 - eps)
    return jnp.log(-jnp.log(1.0 - mu))


def cloglog_inverse(eta: jnp.ndarray) -> jnp.ndarray:
    """cloglog 逆链接：1 - exp(-exp(η))"""
    eta = jnp.asarray(eta, dtype=jnp.float64)
    eps = jnp.finfo(jnp.float64).eps
    # 数值稳定性：当 eta 很大时 exp(eta) 会溢出，截断到安全范围
    eta_clipped = jnp.clip(eta, -500.0, 50.0)
    mu = 1.0 - jnp.exp(-jnp.exp(eta_clipped))
    return jnp.clip(mu, eps, 1.0 - eps)


def cloglog_derivative(eta: jnp.ndarray) -> jnp.ndarray:
    """cloglog 逆链接的导数：dμ/dη = exp(η) * exp(-exp(η))"""
    eta = jnp.asarray(eta, dtype=jnp.float64)
    # 同样截断以保证数值稳定
    eta_clipped = jnp.clip(eta, -500.0, 50.0)
    exp_eta = jnp.exp(eta_clipped)
    return exp_eta * jnp.exp(-exp_eta)


# ─────────────────────────────────────────────
# 新增：sqrt 链接函数
# ─────────────────────────────────────────────


def sqrt_link(mu: jnp.ndarray) -> jnp.ndarray:
    """平方根链接：η = sqrt(μ)

    常用于：Poisson 分布（方差稳定变换）。
    域：μ > 0 → η > 0
    """
    mu = jnp.asarray(mu, dtype=jnp.float64)
    eps = jnp.finfo(jnp.float64).eps
    # 保证 mu 非负，避免对负数开方
    return jnp.sqrt(jnp.maximum(mu, eps))


def sqrt_inverse(eta: jnp.ndarray) -> jnp.ndarray:
    """sqrt 逆链接：μ = η²"""
    eta = jnp.asarray(eta, dtype=jnp.float64)
    eps = jnp.finfo(jnp.float64).eps
    # 保证返回值严格为正
    return jnp.maximum(eta**2, eps)


def sqrt_derivative(eta: jnp.ndarray) -> jnp.ndarray:
    """sqrt 逆链接的导数：dμ/dη = 2η"""
    eta = jnp.asarray(eta, dtype=jnp.float64)
    eps = jnp.finfo(jnp.float64).eps
    # 保证导数严格为正（eta 理论上 > 0）
    return 2.0 * jnp.maximum(eta, eps)


# ─────────────────────────────────────────────
# 新增：链接函数注册表
# ─────────────────────────────────────────────

#: 字符串名称 → (link_fn, inverse_fn, derivative_fn) 映射
#: 用于根据字符串名称动态获取链接函数三件套
LINK_REGISTRY: dict = {
    "identity": (identity_link, identity_inverse, identity_derivative),
    "log": (log_link, log_inverse, log_derivative),
    "logit": (logit_link, logit_inverse, logit_derivative),
    "probit": (probit_link, probit_inverse, probit_derivative),
    "inverse": (inverse_link, inverse_inverse, inverse_derivative),
    "cloglog": (cloglog_link, cloglog_inverse, cloglog_derivative),
    "sqrt": (sqrt_link, sqrt_inverse, sqrt_derivative),
    # 别名
    "log-link": (log_link, log_inverse, log_derivative),
    "logit-link": (logit_link, logit_inverse, logit_derivative),
    "1/mu^2": (inverse_link, inverse_inverse, inverse_derivative),
}


def get_link(name: str):
    """根据字符串名称获取链接函数三件套。

    Parameters
    ----------
    name : str
        链接函数名称，如 "log", "logit", "identity", "probit", "cloglog", "sqrt"

    Returns
    -------
    tuple
        (link_fn, inverse_fn, derivative_fn)

    Raises
    ------
    ValueError
        如果名称不在注册表中

    Examples
    --------
    >>> link_fn, inv_fn, deriv_fn = get_link("log")
    >>> eta = link_fn(mu)
    >>> mu_hat = inv_fn(eta)
    """
    name_lower = name.lower().strip()
    if name_lower not in LINK_REGISTRY:
        # 列出所有可用名称，方便用户排查
        available = ", ".join(sorted(LINK_REGISTRY.keys()))
        raise ValueError(f"未知链接函数：'{name}'。\n可用链接函数：{available}")
    return LINK_REGISTRY[name_lower]


def get_link_fn(name: str):
    """获取链接函数（正向）。"""
    return get_link(name)[0]


def get_inverse_link_fn(name: str):
    """获取逆链接函数。"""
    return get_link(name)[1]


def get_link_derivative_fn(name: str):
    """获取链接函数导数。"""
    return get_link(name)[2]


# 导出列表
__all__ = [
    # 链接函数三件套
    "identity_link",
    "identity_inverse",
    "identity_derivative",
    "log_link",
    "log_inverse",
    "log_derivative",
    "logit_link",
    "logit_inverse",
    "logit_derivative",
    "inverse_link",
    "inverse_inverse",
    "inverse_derivative",
    "probit_link",
    "probit_inverse",
    "probit_derivative",
    "cloglog_link",
    "cloglog_inverse",
    "cloglog_derivative",
    "sqrt_link",
    "sqrt_inverse",
    "sqrt_derivative",
    # 注册表和辅助函数
    "LINK_REGISTRY",
    "get_link",
    "get_link_fn",
    "get_inverse_link_fn",
    "get_link_derivative_fn",
]
