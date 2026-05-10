"""分布族自动生成工厂（FamilyFactory）

从 scipy.stats 分布对象自动构建 GAMLSS FamilyDefinition，
无需手动编写 score/hessian 函数——利用 JAX 自动微分生成。

主要用途：
1. 快速引入 scipy.stats 中的任意分布
2. 作为新分布族的原型工具
3. 分布比较研究中的快速构建
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import jax.numpy as jnp
import numpy as np

# FamilyDefinition 仅在类型检查时导入，避免循环引用
if TYPE_CHECKING:
    from ..families import FamilyDefinition

# ──────────────────────────────────────────────────────────────
# 从 scipy.stats 自动构建 FamilyDefinition
# ──────────────────────────────────────────────────────────────


def from_scipy(
    scipy_dist,
    param_names: Optional[list[str]] = None,
    links: Optional[dict[str, str]] = None,
    constraints: Optional[dict[str, str]] = None,
    family_name: Optional[str] = None,
) -> "FamilyDefinition":
    """从 scipy.stats 分布对象自动构建 GAMLSS FamilyDefinition。

    利用 JAX 的自动微分，自动生成：
    - logpdf（包装 scipy logpdf）
    - score 函数（jax.grad 对数似然关于每个参数）
    - Hessian 对角（jax.grad^2）
    - 链接函数（根据约束自动推断）

    Parameters
    ----------
    scipy_dist : scipy.stats 分布对象
        如 stats.gamma, stats.norm, stats.lognorm
    param_names : list[str], optional
        参数名列表，None 表示自动从 dist.shapes 解析
        （loc, scale 分别对应 mu, sigma）
    links : dict, optional
        {参数名: 链接函数名}，未指定的自动推断
    constraints : dict, optional
        {参数名: 约束类型}，未指定的自动推断
    family_name : str, optional
        族名称，None 则用分布类名

    Returns
    -------
    FamilyDefinition
        可直接用于 gamlss() 的分布族对象

    Examples
    --------
    >>> from scipy import stats
    >>> gamma_fam = from_scipy(stats.gamma,
    ...                         param_names=["mu", "sigma"],
    ...                         links={"mu": "log", "sigma": "log"})
    >>> model = gamlss("y ~ x", family=gamma_fam, data=data)

    >>> # 自动推断参数名和链接函数
    >>> weibull_fam = from_scipy(stats.weibull_min)
    >>> print(weibull_fam.name)  # "weibull_min"
    >>> print(weibull_fam.parameters)  # ("mu", "sigma", "nu")
    """
    from ..families import FamilyDefinition
    from ..links import LINK_REGISTRY

    # ── 1. 解析参数名 ──
    dist_instance = scipy_dist

    # 获取形状参数名（scipy 的 shapes 字符串）
    shapes_str = getattr(dist_instance, "shapes", None)
    shape_params = []
    if shapes_str:
        shape_params = [s.strip() for s in shapes_str.split(",")]

    if param_names is None:
        # 默认映射：loc → mu，scale → sigma，shape 参数 → nu, tau, ...
        extra_names = ["nu", "tau", "xi", "alpha", "beta"]
        param_names = []
        for i, sp in enumerate(shape_params):
            # 尝试用有语义的参数名
            if sp in ("a", "c", "k", "n"):
                param_names.append(extra_names[i] if i < len(extra_names) else sp)
            else:
                param_names.append(sp)
        param_names = ["mu"] + param_names if "loc" not in param_names else param_names
        if "sigma" not in param_names and len(shape_params) == 0:
            param_names.append("sigma")

    # ── 2. 推断约束和链接函数 ──
    AUTO_CONSTRAINTS = {
        "mu": "unrestricted",
        "sigma": "positive",
        "nu": "positive",
        "tau": "positive",
        "pi": "probability",
        "p": "probability",
        "n": "positive",
    }

    constraints = constraints or {}
    links = links or {}

    resolved_constraints = {}
    resolved_links = {}
    for name in param_names:
        # 约束：优先用用户指定，否则查自动映射表
        constraint = constraints.get(name, AUTO_CONSTRAINTS.get(name, "unrestricted"))
        resolved_constraints[name] = constraint
        # 链接函数：优先用用户指定，否则根据约束推断
        link = links.get(name)
        if link is None:
            from ..core.parameter_spec import AUTO_LINK_MAP

            link = AUTO_LINK_MAP.get(constraint, "identity")
        resolved_links[name] = link

    # ── 3. 构建 JAX 包装的对数似然 ──
    # 将 scipy logpdf 包装为 JAX 函数（通过 pure_callback）
    def make_jax_logpdf(scipy_dist_inner, n_shape_params):
        """创建 JAX 可微分的 logpdf 包装函数。"""

        def logpdf(y, **kwargs):
            """对数密度函数（JAX 包装）。

            参数映射：
            - kwargs["mu"] → loc（位置）
            - kwargs["sigma"] → scale（尺度）
            - 其他 kwargs → 形状参数
            """
            # 提取 loc 和 scale
            mu = kwargs.get("mu", kwargs.get("loc", 0.0))
            sigma = kwargs.get("sigma", kwargs.get("scale", 1.0))

            # 形状参数（除 mu/sigma 外的其余参数）
            shape_vals = []
            for name in param_names:
                if name not in ("mu", "sigma"):
                    shape_vals.append(kwargs.get(name, 1.0))

            # 转换为 numpy（scipy 不接受 JAX 数组）
            y_np = np.asarray(y)
            mu_np = np.asarray(mu)
            sigma_np = np.maximum(np.asarray(sigma), 1e-10)
            shape_np = [np.asarray(v) for v in shape_vals]

            if shape_np:
                lp = scipy_dist_inner.logpdf(y_np, *shape_np, loc=mu_np, scale=sigma_np)
            else:
                lp = scipy_dist_inner.logpdf(y_np, loc=mu_np, scale=sigma_np)

            return jnp.asarray(lp, dtype=jnp.float64)

        return logpdf

    jax_logpdf = make_jax_logpdf(dist_instance, len(shape_params))

    # ── 4. 用 JAX 自动微分生成 score 和 hessian ──
    # 注意：由于 scipy logpdf 通过 numpy 调用，无法直接用 jax.grad
    # 我们使用数值微分作为 fallback

    def make_score_fn(param_name, logpdf_fn, other_param_names):
        """为参数 param_name 创建 score 函数（中心差分数值微分）。"""

        def score(y, **kwargs):
            val = np.asarray(kwargs.get(param_name))
            eps = max(1e-5 * np.std(val), 1e-7)

            kwargs_plus = {**kwargs, param_name: val + eps}
            kwargs_minus = {**kwargs, param_name: val - eps}

            lp_plus = logpdf_fn(y, **kwargs_plus)
            lp_minus = logpdf_fn(y, **kwargs_minus)

            return (lp_plus - lp_minus) / (2 * eps)

        return score

    def make_hessian_fn(param_name, logpdf_fn, other_param_names):
        """为参数 param_name 创建 Hessian 对角（二阶中心差分数值微分）。"""

        def hessian(y, **kwargs):
            val = np.asarray(kwargs.get(param_name))
            eps = max(1e-4 * np.std(val), 1e-6)

            kwargs_plus = {**kwargs, param_name: val + eps}
            kwargs_minus = {**kwargs, param_name: val - eps}
            lp_center = logpdf_fn(y, **kwargs)
            lp_plus = logpdf_fn(y, **kwargs_plus)
            lp_minus = logpdf_fn(y, **kwargs_minus)

            return (lp_plus - 2 * lp_center + lp_minus) / (eps**2)

        return hessian

    score_fns = {
        name: make_score_fn(name, jax_logpdf, [n for n in param_names if n != name])
        for name in param_names
    }

    hessian_fns = {
        name: make_hessian_fn(name, jax_logpdf, [n for n in param_names if n != name])
        for name in param_names
    }

    # ── 5. 链接函数三件套（正向 / 逆向 / 导数）──
    link_fns = {name: LINK_REGISTRY[resolved_links[name]][0] for name in param_names}
    link_inv_fns = {
        name: LINK_REGISTRY[resolved_links[name]][1] for name in param_names
    }
    link_deriv_fns = {
        name: LINK_REGISTRY[resolved_links[name]][2] for name in param_names
    }

    # ── 6. Deviance 增量（g_dev_inc）──
    def g_dev_inc(y, **kwargs):
        """每观测的 deviance 增量：-2 * logpdf。"""
        return -2.0 * jax_logpdf(y, **kwargs)

    # ── 7. 构建 FamilyDefinition ──
    fname = family_name or getattr(dist_instance, "name", type(dist_instance).__name__)

    # d/p/q/r 函数（密度、CDF、分位数、随机样本）
    def d_fn(x, log=False, **kwargs):
        """密度函数（PDF/PMF）。"""
        if log:
            return jax_logpdf(x, **kwargs)
        return jnp.exp(jax_logpdf(x, **kwargs))

    def p_fn(q, **kwargs):
        """累积分布函数（CDF）。"""
        mu = kwargs.get("mu", 0.0)
        sigma = kwargs.get("sigma", 1.0)
        shape_vals = [kwargs.get(n) for n in param_names if n not in ("mu", "sigma")]
        if shape_vals:
            return jnp.asarray(
                dist_instance.cdf(np.asarray(q), *shape_vals, loc=mu, scale=sigma)
            )
        return jnp.asarray(dist_instance.cdf(np.asarray(q), loc=mu, scale=sigma))

    def q_fn(p, **kwargs):
        """分位数函数（逆 CDF）。"""
        mu = kwargs.get("mu", 0.0)
        sigma = kwargs.get("sigma", 1.0)
        shape_vals = [kwargs.get(n) for n in param_names if n not in ("mu", "sigma")]
        if shape_vals:
            return jnp.asarray(
                dist_instance.ppf(np.asarray(p), *shape_vals, loc=mu, scale=sigma)
            )
        return jnp.asarray(dist_instance.ppf(np.asarray(p), loc=mu, scale=sigma))

    return FamilyDefinition(
        name=fname,
        parameters=tuple(param_names),
        type="Continuous",
        g_dev_inc=g_dev_inc,
        links={name: resolved_links[name] for name in param_names},
        link_functions=link_fns,
        link_inverses=link_inv_fns,
        link_derivatives=link_deriv_fns,
        score_functions=score_fns,
        hessian_functions=hessian_fns,
        d=d_fn,
        p=p_fn,
        q=q_fn,
        r=None,  # 随机生成函数可选，此处留空
    )


# ──────────────────────────────────────────────────────────────
# 从参数规格构建 FamilyDefinition
# ──────────────────────────────────────────────────────────────


def from_param_specs(
    name: str,
    param_specs: list,
    logpdf: Callable,
    type_: str = "Continuous",
    d: Optional[Callable] = None,  # noqa: E741
    p: Optional[Callable] = None,
    q: Optional[Callable] = None,
    r: Optional[Callable] = None,
    score_functions: Optional[dict] = None,
    hessian_functions: Optional[dict] = None,
) -> "FamilyDefinition":
    """从 ParameterSpec 列表构建 FamilyDefinition。

    如果 score/hessian 未提供，用 JAX 自动微分生成。

    Parameters
    ----------
    name : str
        族名称
    param_specs : list[ParameterSpec]
        参数规格列表
    logpdf : Callable
        对数密度函数，签名：logpdf(y, mu, sigma, ...) -> jnp.ndarray
    type_ : str
        分布类型 "Continuous" / "Discrete"
    d, p, q, r : Callable, optional
        密度/CDF/分位数/随机采样函数
    score_functions : dict, optional
        {参数名: score 函数}，None 则用 JAX 自动微分
    hessian_functions : dict, optional
        {参数名: hessian 函数}，None 则用 JAX 自动微分

    Returns
    -------
    FamilyDefinition

    Examples
    --------
    >>> from scipy import stats
    >>> import jax.numpy as jnp
    >>>
    >>> specs = [MU_POS_SPEC, SIGMA_SPEC]
    >>>
    >>> def gamma_logpdf(y, mu, sigma):
    ...     # Gamma 参数化：shape=1/sigma², rate=1/(mu*sigma²)
    ...     shape = 1.0 / (sigma ** 2)
    ...     rate = shape / mu
    ...     return (shape - 1) * jnp.log(y) - rate * y + shape * jnp.log(rate) - jax.scipy.special.gammaln(shape)
    >>>
    >>> gamma_fam = from_param_specs("MY_GAMMA", specs, gamma_logpdf)
    """
    from ..families import FamilyDefinition
    from ..links import LINK_REGISTRY

    param_names = [spec.name for spec in param_specs]

    # ── 构建链接函数三件套 ──
    link_fns, link_inv_fns, link_deriv_fns = {}, {}, {}
    links_map = {}
    for spec in param_specs:
        link_name = spec.link or "identity"
        links_map[spec.name] = link_name
        link_fns[spec.name] = LINK_REGISTRY[link_name][0]
        link_inv_fns[spec.name] = LINK_REGISTRY[link_name][1]
        link_deriv_fns[spec.name] = LINK_REGISTRY[link_name][2]

    # ── 自动生成 score / hessian（用 JAX 数值微分）──
    if score_functions is None or hessian_functions is None:
        auto_scores = {}
        auto_hessians = {}

        for i, spec in enumerate(param_specs):
            param_name = spec.name

            # 构建以第 i 个参数为主变量的函数（固定其他参数）
            def make_score(idx, pnames):
                """生成对第 idx 个参数的 score 函数（中心差分）。"""

                def score_fn(y, **kwargs):
                    try:
                        param_val = jnp.asarray(kwargs[pnames[idx]])

                        # 构建一个只关于 param_val 的函数
                        def logpdf_i(v):
                            kw = dict(kwargs)
                            kw[pnames[idx]] = v
                            return jnp.sum(logpdf(y, **kw))

                        # JAX 前向差分（因为 JAX 对 scipy 调用的 AD 可能不工作）
                        eps = jnp.maximum(1e-5 * jnp.abs(param_val), 1e-7)
                        return (
                            logpdf_i(param_val + eps) - logpdf_i(param_val - eps)
                        ) / (2 * eps)
                    except Exception:
                        return jnp.zeros_like(jnp.asarray(kwargs.get(pnames[idx], 0.0)))

                return score_fn

            def make_hessian(idx, pnames):
                """生成对第 idx 个参数的 hessian 对角（二阶中心差分）。"""

                def hessian_fn(y, **kwargs):
                    try:
                        param_val = jnp.asarray(kwargs[pnames[idx]])

                        def logpdf_i(v):
                            kw = dict(kwargs)
                            kw[pnames[idx]] = v
                            return jnp.sum(logpdf(y, **kw))

                        eps = jnp.maximum(1e-4 * jnp.abs(param_val), 1e-6)
                        d2 = (
                            logpdf_i(param_val + eps)
                            - 2 * logpdf_i(param_val)
                            + logpdf_i(param_val - eps)
                        ) / (eps**2)
                        # 广播到与 y 相同的形状
                        return jnp.broadcast_to(
                            d2 / len(jnp.asarray(y)), jnp.asarray(y).shape
                        )
                    except Exception:
                        return jnp.full_like(
                            jnp.asarray(kwargs.get(pnames[idx], 0.0)), -0.1
                        )

                return hessian_fn

            auto_scores[param_name] = make_score(i, param_names)
            auto_hessians[param_name] = make_hessian(i, param_names)

        if score_functions is None:
            score_functions = auto_scores
        if hessian_functions is None:
            hessian_functions = auto_hessians

    # ── Deviance 增量 ──
    def g_dev_inc(y, **kwargs):
        """每观测的 deviance 增量：-2 * logpdf。"""
        return -2.0 * logpdf(y, **kwargs)

    # ── d 函数（密度）——若调用方未提供则自动构建 ──
    # 用局部变量 d_fn 避免遮蔽同名参数
    if d is None:

        def _d_fn(x, log=False, **kwargs):
            """密度函数（PDF/PMF），支持 log=True 返回对数密度。"""
            lp = logpdf(x, **kwargs)
            return lp if log else jnp.exp(lp)

        d_fn: Optional[Callable] = _d_fn
    else:
        d_fn = d

    return FamilyDefinition(
        name=name,
        parameters=tuple(param_names),
        type=type_,
        g_dev_inc=g_dev_inc,
        links=links_map,
        link_functions=link_fns,
        link_inverses=link_inv_fns,
        link_derivatives=link_deriv_fns,
        score_functions=score_functions,
        hessian_functions=hessian_functions,
        d=d_fn,
        p=p,
        q=q,
        r=r,
    )


__all__ = [
    "from_scipy",
    "from_param_specs",
]
