"""混合分布族（Mixture Distributions）

实现 GAMLSS 中的混合模型和零膨胀模型：
1. ZeroInflated - 零膨胀分布（计数数据常用）
2. Hurdle - Hurdle 模型（分开建模零和正值）
3. FiniteMixture - 有限混合分布

这些类都返回标准的 FamilyDefinition 对象，
可直接用于 gamlss() 函数。

参考：
- Rigby et al. (2019) GAMLSS 书籍第 11 章
- Stasinopoulos et al. R gamlss 包文档
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

# ──────────────────────────────────────────────────────────────
# 零膨胀分布（Zero-Inflated）
# ──────────────────────────────────────────────────────────────


def zero_inflated(
    base_family,
    name: Optional[str] = None,
) -> "FamilyDefinition":
    """构建零膨胀版本的分布族。

    零膨胀分布将零值与非零值分开建模：
    P(Y = 0) = π + (1 - π) * P_base(Y = 0)
    P(Y = y) = (1 - π) * P_base(Y = y)  for y > 0

    参数结构（在 base_family 基础上增加一个 π 参数）：
    - π: 零膨胀概率（额外的零值概率），约束 ∈ (0, 1)，logit 链接
    - 其余参数继承自 base_family

    Parameters
    ----------
    base_family : FamilyDefinition
        基础分布族（通常是计数分布：PO, NBI, BN 等）
    name : str, optional
        新族名称，None 则用 "ZI" + base_family.name

    Returns
    -------
    FamilyDefinition
        零膨胀分布族

    Examples
    --------
    >>> from omnilss import PO, NBI
    >>> ZIP_fam = zero_inflated(PO())   # 等价于 omnilss 内置的 ZIP
    >>> ZINBI_fam = zero_inflated(NBI())
    >>>
    >>> model = gamlss(
    ...     "count ~ x",
    ...     sigma_formula="~ x",     # NBI 的 sigma
    ...     nu_formula="~ x",         # 零膨胀概率 π
    ...     family=ZINBI_fam,
    ...     data=data
    ... )
    """
    from ..families import FamilyDefinition
    from ..links import LINK_REGISTRY

    fname = name or f"ZI{base_family.name}"

    # 新参数列表 = base 参数 + "pi"（零膨胀概率）
    # π 通常是最后一个参数
    base_params = list(base_family.parameters)
    all_params = base_params + ["pi"]

    # ── 零膨胀对数密度 ──
    base_logpdf = base_family.d  # base_family.d(x=y, log=True, **params)

    def logpdf(y, **kwargs):
        pi = jnp.asarray(kwargs.get("pi", 0.1))
        pi = jnp.clip(pi, 1e-7, 1.0 - 1e-7)

        base_kwargs = {k: v for k, v in kwargs.items() if k != "pi"}

        # base 分布的 log P(Y=y)
        log_p_base = base_logpdf(x=y, log=True, **base_kwargs)
        # base 分布的 log P(Y=0)
        log_p_base_0 = base_logpdf(x=jnp.zeros_like(y), log=True, **base_kwargs)

        # 零膨胀混合：
        # log P(Y=0) = log(π + (1-π)*P_base(0))
        #            = log(exp(log π) + exp(log(1-π) + log P_base(0)))
        log_pi = jnp.log(pi)
        log_1_minus_pi = jnp.log1p(-pi)

        # 数值稳定的 log-sum-exp
        log_p_zero = jnp.logaddexp(log_pi, log_1_minus_pi + log_p_base_0)
        # log P(Y>0) = log(1-π) + log P_base(y>0)
        log_p_pos = log_1_minus_pi + log_p_base

        # 按 y==0 / y>0 选择
        is_zero = (y == 0).astype(jnp.float64)
        return is_zero * log_p_zero + (1 - is_zero) * log_p_pos

    def g_dev_inc(y, **kwargs):
        """全局偏差增量：-2 * log 似然"""
        return -2.0 * logpdf(y, **kwargs)

    # ── Score 函数（数值微分，中心差分）──
    def make_score(param_name):
        def score(y, **kwargs):
            val = jnp.asarray(kwargs.get(param_name))
            eps = jnp.maximum(1e-5 * jnp.abs(val), 1e-7)
            kw_p = {**kwargs, param_name: val + eps}
            kw_m = {**kwargs, param_name: val - eps}
            return (logpdf(y, **kw_p) - logpdf(y, **kw_m)) / (2 * eps)

        return score

    def make_hessian(param_name):
        def hessian(y, **kwargs):
            val = jnp.asarray(kwargs.get(param_name))
            eps = jnp.maximum(1e-4 * jnp.abs(val), 1e-6)
            kw_p = {**kwargs, param_name: val + eps}
            kw_m = {**kwargs, param_name: val - eps}
            kw_c = kwargs
            return (logpdf(y, **kw_p) - 2 * logpdf(y, **kw_c) + logpdf(y, **kw_m)) / (
                eps**2
            )

        return hessian

    score_fns = {p: make_score(p) for p in all_params}
    hess_fns = {p: make_hessian(p) for p in all_params}

    # ── 链接函数：继承 base_family，π 使用 logit ──
    link_names = dict(base_family.links) if base_family.links else {}
    link_names["pi"] = "logit"

    link_fns = dict(base_family.link_functions) if base_family.link_functions else {}
    link_inv_fns = dict(base_family.link_inverses) if base_family.link_inverses else {}
    link_deriv_fns = (
        dict(base_family.link_derivatives) if base_family.link_derivatives else {}
    )

    logit, logit_inv, logit_deriv = LINK_REGISTRY["logit"]
    link_fns["pi"] = logit
    link_inv_fns["pi"] = logit_inv
    link_deriv_fns["pi"] = logit_deriv

    # ── d/p/q 函数 ──
    def d_fn(x, log=False, **kwargs):
        lp = logpdf(x, **kwargs)
        return lp if log else jnp.exp(lp)

    return FamilyDefinition(
        name=fname,
        parameters=tuple(all_params),
        type="Discrete",
        g_dev_inc=g_dev_inc,
        links=link_names,
        link_functions=link_fns,
        link_inverses=link_inv_fns,
        link_derivatives=link_deriv_fns,
        score_functions=score_fns,
        hessian_functions=hess_fns,
        d=d_fn,
        p=None,
        q=None,
        r=None,
    )


# ──────────────────────────────────────────────────────────────
# Hurdle 模型
# ──────────────────────────────────────────────────────────────


def hurdle(
    base_family,
    name: Optional[str] = None,
) -> "FamilyDefinition":
    """构建 Hurdle 模型的分布族。

    Hurdle 模型将零值和正值分别建模：
    P(Y = 0) = π（由 π 参数直接控制）
    P(Y = y | y > 0) = P_base(Y = y) / (1 - P_base(Y = 0))

    与 ZeroInflated 的区别：
    - ZI：零值 = 结构零 + 随机零
    - Hurdle：零值完全由 π 决定，正值由截断基础分布决定

    Parameters
    ----------
    base_family : FamilyDefinition
        基础分布族（计数分布）
    name : str, optional
        新族名称，默认 "Hurdle" + base_family.name
    """
    from ..families import FamilyDefinition
    from ..links import LINK_REGISTRY

    fname = name or f"Hurdle{base_family.name}"
    base_params = list(base_family.parameters)
    all_params = base_params + ["pi"]

    base_logpdf = base_family.d

    def logpdf(y, **kwargs):
        # 提取并裁剪零膨胀概率 π
        pi = jnp.clip(jnp.asarray(kwargs.get("pi", 0.1)), 1e-7, 1.0 - 1e-7)
        base_kwargs = {k: v for k, v in kwargs.items() if k != "pi"}

        log_p_base = base_logpdf(x=y, log=True, **base_kwargs)
        log_p_base_0 = base_logpdf(x=jnp.zeros_like(y), log=True, **base_kwargs)

        # 截断分布：P_trunc(y>0) = P_base(y) / (1 - P_base(0))
        log_1_minus_p0 = jnp.log1p(-jnp.exp(log_p_base_0))

        log_p_zero = jnp.log(pi)
        log_p_pos = jnp.log1p(-pi) + log_p_base - log_1_minus_p0

        # y==0 时使用 log_p_zero，y>0 时使用 log_p_pos
        is_zero = (y == 0).astype(jnp.float64)
        return is_zero * log_p_zero + (1 - is_zero) * log_p_pos

    def g_dev_inc(y, **kwargs):
        """全局偏差增量：-2 * log 似然"""
        return -2.0 * logpdf(y, **kwargs)

    def make_score(param_name):
        def score(y, **kwargs):
            val = jnp.asarray(kwargs.get(param_name))
            eps = jnp.maximum(1e-5 * jnp.abs(val), 1e-7)
            return (
                logpdf(y, **{**kwargs, param_name: val + eps})
                - logpdf(y, **{**kwargs, param_name: val - eps})
            ) / (2 * eps)

        return score

    def make_hessian(param_name):
        def hessian(y, **kwargs):
            val = jnp.asarray(kwargs.get(param_name))
            eps = jnp.maximum(1e-4 * jnp.abs(val), 1e-6)
            return (
                logpdf(y, **{**kwargs, param_name: val + eps})
                - 2 * logpdf(y, **kwargs)
                + logpdf(y, **{**kwargs, param_name: val - eps})
            ) / (eps**2)

        return hessian

    score_fns = {p: make_score(p) for p in all_params}
    hess_fns = {p: make_hessian(p) for p in all_params}

    link_names = dict(base_family.links or {})
    link_names["pi"] = "logit"
    link_fns = dict(base_family.link_functions or {})
    link_inv_fns = dict(base_family.link_inverses or {})
    link_deriv_fns = dict(base_family.link_derivatives or {})
    logit, logit_inv, logit_deriv = LINK_REGISTRY["logit"]
    link_fns["pi"] = logit
    link_inv_fns["pi"] = logit_inv
    link_deriv_fns["pi"] = logit_deriv

    def d_fn(x, log=False, **kwargs):
        lp = logpdf(x, **kwargs)
        return lp if log else jnp.exp(lp)

    return FamilyDefinition(
        name=fname,
        parameters=tuple(all_params),
        type="Discrete",
        g_dev_inc=g_dev_inc,
        links=link_names,
        link_functions=link_fns,
        link_inverses=link_inv_fns,
        link_derivatives=link_deriv_fns,
        score_functions=score_fns,
        hessian_functions=hess_fns,
        d=d_fn,
        p=None,
        q=None,
        r=None,
    )


# ──────────────────────────────────────────────────────────────
# 有限混合分布（Finite Mixture）
# ──────────────────────────────────────────────────────────────


def finite_mixture(
    components: list,
    name: Optional[str] = None,
) -> "FamilyDefinition":
    """构建 K 分量有限混合分布族。

    P(Y = y) = Σ_{k=1}^K π_k * P_k(Y = y)

    其中 π_k 满足 Σπ_k = 1（simplex 约束）。

    Parameters
    ----------
    components : list of FamilyDefinition
        分量分布族列表
    name : str, optional
        族名称

    Returns
    -------
    FamilyDefinition

    Notes
    -----
    当前实现为两分量混合（K=2），多分量为后续版本。

    Examples
    --------
    >>> from omnilss import NO
    >>> # 二分量高斯混合
    >>> mix = finite_mixture([NO(), NO()])
    >>> # 模型：y ~ 0.7*N(mu1, sigma1) + 0.3*N(mu2, sigma2)
    >>> model = gamlss("y ~ 1", ..., family=mix, data=data)
    """
    from ..families import FamilyDefinition
    from ..links import LINK_REGISTRY

    K = len(components)
    fname = name or f"Mix{K}_{components[0].name}"

    # 参数命名：每个分量的参数加上 k 后缀，混合权重 pi_1, ..., pi_{K-1}
    # 为简单起见，目前只支持相同结构的分量
    base_params = list(components[0].parameters)

    # 为每个分量生成带后缀的参数名
    all_params = []
    component_params = []
    for k in range(K):
        params_k = [f"{p}_{k + 1}" for p in base_params]
        all_params.extend(params_k)
        component_params.append(params_k)

    # 混合权重（K-1 个自由参数，最后一个由 1 - sum 计算）
    for k in range(K - 1):
        all_params.append(f"pi_{k + 1}")

    comp_logpdfs = [c.d for c in components]

    def logpdf(y, **kwargs):
        # 提取混合权重，最后一个权重由约束 Σπ_k=1 确定
        pis = []
        for k in range(K - 1):
            pi_k = jnp.clip(
                jnp.asarray(kwargs.get(f"pi_{k + 1}", 1.0 / K)), 1e-7, 1 - 1e-7
            )
            pis.append(pi_k)
        # 最后一个权重 = 1 - sum(others)
        pi_last = jnp.clip(1.0 - sum(pis), 1e-7, 1 - 1e-7)
        pis.append(pi_last)

        # 计算每个分量的 log P_k(y)，并加上混合权重的 log
        log_terms = []
        for k, (params_k, comp_logpdf) in enumerate(
            zip(component_params, comp_logpdfs)
        ):
            kwargs_k = {
                base_params[j]: kwargs.get(params_k[j]) for j in range(len(base_params))
            }
            log_p_k = comp_logpdf(x=y, log=True, **kwargs_k)
            log_terms.append(jnp.log(pis[k]) + log_p_k)

        # log-sum-exp 实现数值稳定的 log Σ π_k P_k(y)
        log_terms_stacked = jnp.stack(log_terms, axis=0)
        return jax.scipy.special.logsumexp(log_terms_stacked, axis=0)

    def g_dev_inc(y, **kwargs):
        """全局偏差增量：-2 * log 似然"""
        return -2.0 * logpdf(y, **kwargs)

    def make_score(param_name):
        def score(y, **kwargs):
            val = jnp.asarray(kwargs.get(param_name))
            eps = jnp.maximum(1e-5 * jnp.abs(val), 1e-7)
            return (
                logpdf(y, **{**kwargs, param_name: val + eps})
                - logpdf(y, **{**kwargs, param_name: val - eps})
            ) / (2 * eps)

        return score

    def make_hessian(param_name):
        def hessian(y, **kwargs):
            val = jnp.asarray(kwargs.get(param_name))
            eps = jnp.maximum(1e-4 * jnp.abs(val), 1e-6)
            return (
                logpdf(y, **{**kwargs, param_name: val + eps})
                - 2 * logpdf(y, **kwargs)
                + logpdf(y, **{**kwargs, param_name: val - eps})
            ) / (eps**2)

        return hessian

    score_fns = {p: make_score(p) for p in all_params}
    hess_fns = {p: make_hessian(p) for p in all_params}

    # 链接函数：分量参数继承各自分量的链接，混合权重用 logit
    link_names = {}
    link_fns, link_inv_fns, link_deriv_fns = {}, {}, {}

    for k, (params_k, comp) in enumerate(zip(component_params, components)):
        for j, (pk, bp) in enumerate(zip(params_k, base_params)):
            link_name = (comp.links or {}).get(bp, "identity")
            link_names[pk] = link_name
            link_fns[pk] = LINK_REGISTRY[link_name][0]
            link_inv_fns[pk] = LINK_REGISTRY[link_name][1]
            link_deriv_fns[pk] = LINK_REGISTRY[link_name][2]

    logit, logit_inv, logit_deriv = LINK_REGISTRY["logit"]
    for k in range(K - 1):
        pk = f"pi_{k + 1}"
        link_names[pk] = "logit"
        link_fns[pk] = logit
        link_inv_fns[pk] = logit_inv
        link_deriv_fns[pk] = logit_deriv

    def d_fn(x, log=False, **kwargs):
        lp = logpdf(x, **kwargs)
        return lp if log else jnp.exp(lp)

    return FamilyDefinition(
        name=fname,
        parameters=tuple(all_params),
        type=components[0].type,
        g_dev_inc=g_dev_inc,
        links=link_names,
        link_functions=link_fns,
        link_inverses=link_inv_fns,
        link_derivatives=link_deriv_fns,
        score_functions=score_fns,
        hessian_functions=hess_fns,
        d=d_fn,
        p=None,
        q=None,
        r=None,
    )


# ──────────────────────────────────────────────────────────────
# 删失分布（Censored）
# ──────────────────────────────────────────────────────────────


def censored(
    base_family,
    name: Optional[str] = None,
) -> "FamilyDefinition":
    """构建删失版本的分布族（用于生存分析）。

    对数似然：
    log L = Σ_i [δ_i * log f(t_i) + (1-δ_i) * log S(t_i)]
    其中 δ_i = 1 表示事件发生，S = 生存函数 = 1 - CDF。

    Parameters
    ----------
    base_family : FamilyDefinition
        基础连续分布族
    name : str, optional
        族名称

    Notes
    -----
    使用此族时，需要在数据中提供 "event" 变量（0/1 删失指示）。
    """
    from ..families import FamilyDefinition
    from ..links import LINK_REGISTRY

    fname = name or f"Censored{base_family.name}"
    base_params = list(base_family.parameters)

    base_d = base_family.d
    base_p = base_family.p

    def logpdf(t, event=None, **kwargs):
        # event=1 表示事件（失效），event=0 表示删失（截尾）
        if event is None:
            event = jnp.ones_like(t)
        event = jnp.asarray(event, dtype=jnp.float64)

        log_f = base_d(x=t, log=True, **kwargs)

        if base_p is not None:
            # 利用 CDF 计算生存函数 S(t) = 1 - F(t)
            cdf = base_p(t, **kwargs)
            log_S = jnp.log1p(-jnp.clip(cdf, 0, 1 - 1e-7))
        else:
            # 若无 CDF，用密度近似生存函数（仅供调试，不精确）
            log_S = jnp.log(jnp.maximum(1.0 - jnp.exp(log_f), 1e-10))

        # 完整观测：log f(t)；删失观测：log S(t)
        return event * log_f + (1.0 - event) * log_S

    def g_dev_inc(t, event=None, **kwargs):
        """全局偏差增量：-2 * log 似然"""
        return -2.0 * logpdf(t, event=event, **kwargs)

    def make_score(pname):
        def score(t, event=None, **kwargs):
            val = jnp.asarray(kwargs.get(pname))
            eps = jnp.maximum(1e-5 * jnp.abs(val), 1e-7)
            return (
                logpdf(t, event=event, **{**kwargs, pname: val + eps})
                - logpdf(t, event=event, **{**kwargs, pname: val - eps})
            ) / (2 * eps)

        return score

    def make_hessian(pname):
        def hessian(t, event=None, **kwargs):
            val = jnp.asarray(kwargs.get(pname))
            eps = jnp.maximum(1e-4 * jnp.abs(val), 1e-6)
            return (
                logpdf(t, event=event, **{**kwargs, pname: val + eps})
                - 2 * logpdf(t, event=event, **kwargs)
                + logpdf(t, event=event, **{**kwargs, pname: val - eps})
            ) / (eps**2)

        return hessian

    score_fns = {p: make_score(p) for p in base_params}
    hess_fns = {p: make_hessian(p) for p in base_params}

    link_names = dict(base_family.links or {})
    link_fns = dict(base_family.link_functions or {})
    link_inv_fns = dict(base_family.link_inverses or {})
    link_deriv_fns = dict(base_family.link_derivatives or {})

    def d_fn(x, log=False, event=None, **kwargs):
        lp = logpdf(x, event=event, **kwargs)
        return lp if log else jnp.exp(lp)

    return FamilyDefinition(
        name=fname,
        parameters=tuple(base_params),
        type="Continuous",
        g_dev_inc=g_dev_inc,
        links=link_names,
        link_functions=link_fns,
        link_inverses=link_inv_fns,
        link_derivatives=link_deriv_fns,
        score_functions=score_fns,
        hessian_functions=hess_fns,
        d=d_fn,
        p=None,
        q=None,
        r=None,
    )


__all__ = [
    "zero_inflated",
    "hurdle",
    "finite_mixture",
    "censored",
]
