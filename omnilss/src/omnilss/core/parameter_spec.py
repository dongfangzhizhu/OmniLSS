"""参数规格系统（ParameterSpec）

定义分布参数的元数据，包括：
- 参数名
- 约束类型（正值、概率、无约束、有序、有界）
- 链接函数（自动推断或手动指定）
- 默认公式（截距-only 或用户指定）
- 自动约束系统
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Tuple

import jax.numpy as jnp

# ──────────────────────────────────────────────────────────────
# 约束类型
# ──────────────────────────────────────────────────────────────
ConstraintType = Literal[
    "positive",  # θ > 0，例如 sigma, shape
    "probability",  # θ ∈ (0, 1)，例如 p, pi
    "unrestricted",  # θ ∈ (-∞, ∞)，例如 mu（实数响应）
    "ordered",  # θ₁ < θ₂ < ...，例如 ordinal thresholds
    "bounded",  # θ ∈ (lb, ub)，需配合 bounds 字段
    "non_negative",  # θ ≥ 0（含零），例如计数参数
]

# 约束类型 → 默认链接函数名称
AUTO_LINK_MAP: dict[str, str] = {
    "positive": "log",
    "probability": "logit",
    "unrestricted": "identity",
    "ordered": "logit",
    "bounded": "logit",
    "non_negative": "log",
}

# 约束类型 → 可行域检查函数
CONSTRAINT_VALIDATORS: dict[str, Callable] = {
    "positive": lambda x: jnp.all(x > 0),
    "probability": lambda x: jnp.all((x > 0) & (x < 1)),
    "unrestricted": lambda x: jnp.all(jnp.isfinite(x)),
    "ordered": lambda x: jnp.all(x[..., 1:] > x[..., :-1]),
    "non_negative": lambda x: jnp.all(x >= 0),
}


# ──────────────────────────────────────────────────────────────
# ParameterSpec
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ParameterSpec:
    """分布参数的完整规格描述。

    作为 FamilyDefinition 中参数描述的结构化替代，
    支持动态参数数量（突破固定的 mu/sigma/nu/tau 四参数限制）。

    Attributes
    ----------
    name : str
        参数名称，例如 "mu", "sigma", "shape"
    constraint : ConstraintType
        参数的可行域约束
    link : str | None
        链接函数名称（None 表示自动推断）
    default_formula : str
        该参数的默认公式，"~ 1" 表示只含截距
    bounds : tuple | None
        用于 "bounded" 约束的 (lower, upper)
    description : str
        参数的人类可读描述

    Examples
    --------
    >>> mu_spec = ParameterSpec("mu", constraint="unrestricted", link="identity")
    >>> sigma_spec = ParameterSpec("sigma", constraint="positive")  # link="log" 自动推断
    >>> pi_spec = ParameterSpec("pi", constraint="probability")     # link="logit" 自动推断
    >>> bounded_spec = ParameterSpec("p", constraint="bounded", bounds=(0.0, 1.0))
    """

    name: str
    constraint: ConstraintType = "unrestricted"
    link: Optional[str] = None  # None → 根据 constraint 自动推断
    default_formula: str = "~ 1"
    bounds: Optional[Tuple[float, float]] = None
    description: str = ""

    def __post_init__(self):
        # 自动推断链接函数
        if self.link is None:
            inferred_link = AUTO_LINK_MAP.get(self.constraint, "identity")
            object.__setattr__(self, "link", inferred_link)

        # 验证 bounded 约束必须有 bounds
        if self.constraint == "bounded" and self.bounds is None:
            raise ValueError(
                f"ParameterSpec(name={self.name!r}): "
                "constraint='bounded' 要求提供 bounds=(lower, upper)"
            )

    # ── 链接函数访问器 ──

    @property
    def link_fn(self) -> Callable:
        """正向链接函数 g(θ) = η。"""
        from ..links import get_link_fn

        return get_link_fn(self.link)  # type: ignore[arg-type]

    @property
    def inverse_link_fn(self) -> Callable:
        """逆链接函数 g⁻¹(η) = θ。"""
        from ..links import get_inverse_link_fn

        return get_inverse_link_fn(self.link)  # type: ignore[arg-type]

    @property
    def link_derivative_fn(self) -> Callable:
        """逆链接函数的导数 dθ/dη。"""
        from ..links import get_link_derivative_fn

        return get_link_derivative_fn(self.link)  # type: ignore[arg-type]

    # ── 约束检查 ──

    def check_constraint(self, values: jnp.ndarray) -> bool:
        """检查数组是否满足该参数的约束。

        Parameters
        ----------
        values : jnp.ndarray
            待检查的参数值数组

        Returns
        -------
        bool
            True 表示满足约束
        """
        if self.constraint == "bounded":
            lb, ub = self.bounds  # type: ignore[misc]
            return bool(jnp.all((values > lb) & (values < ub)))
        validator = CONSTRAINT_VALIDATORS.get(self.constraint)
        if validator is None:
            return True
        return bool(validator(values))

    def clip_to_constraint(self, values: jnp.ndarray, eps: float = 1e-7) -> jnp.ndarray:
        """将数组截断到可行域内（数值保护）。

        Parameters
        ----------
        values : jnp.ndarray
            待截断的参数值
        eps : float
            边界缓冲量

        Returns
        -------
        jnp.ndarray
            截断后的值
        """
        if self.constraint == "positive":
            return jnp.maximum(values, eps)
        elif self.constraint == "non_negative":
            return jnp.maximum(values, 0.0)
        elif self.constraint == "probability":
            return jnp.clip(values, eps, 1.0 - eps)
        elif self.constraint == "bounded":
            lb, ub = self.bounds  # type: ignore[misc]
            return jnp.clip(values, lb + eps, ub - eps)
        return values

    def __repr__(self) -> str:
        return (
            f"ParameterSpec(name={self.name!r}, "
            f"constraint={self.constraint!r}, "
            f"link={self.link!r})"
        )


# ──────────────────────────────────────────────────────────────
# 常用参数规格预设
# ──────────────────────────────────────────────────────────────

#: 位置参数（实数域）
MU_SPEC = ParameterSpec(
    "mu", constraint="unrestricted", link="identity", description="位置参数（均值）"
)

#: 尺度参数（正值域）
SIGMA_SPEC = ParameterSpec(
    "sigma",
    constraint="positive",
    link="log",
    description="尺度参数（标准差 / 离散度）",
)

#: 形状参数（正值域）
NU_SPEC = ParameterSpec(
    "nu", constraint="positive", link="log", description="形状参数（偏度 / 尾重）"
)

#: 第四参数（正值域）
TAU_SPEC = ParameterSpec(
    "tau", constraint="positive", link="log", description="第四参数（峰度 / 额外形状）"
)

#: 概率参数（0-1 域）
PI_SPEC = ParameterSpec(
    "pi", constraint="probability", link="logit", description="混合概率 / 零膨胀概率"
)

#: 位置参数（正值域，例如 Gamma/LogNormal 的均值）
MU_POS_SPEC = ParameterSpec(
    "mu", constraint="positive", link="log", description="位置参数（正值域均值）"
)

# 默认四参数规格（与经典 GAMLSS 对应）
CLASSIC_GAMLSS_PARAMS = (MU_SPEC, SIGMA_SPEC, NU_SPEC, TAU_SPEC)


# ──────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────


def infer_param_spec(
    name: str, constraint: ConstraintType = "unrestricted"
) -> ParameterSpec:
    """根据参数名和约束类型创建 ParameterSpec，自动推断链接函数。

    Parameters
    ----------
    name : str
        参数名称
    constraint : ConstraintType
        约束类型

    Returns
    -------
    ParameterSpec

    Examples
    --------
    >>> spec = infer_param_spec("sigma", "positive")
    >>> spec.link
    'log'
    >>> spec = infer_param_spec("pi", "probability")
    >>> spec.link
    'logit'
    """
    return ParameterSpec(name=name, constraint=constraint)


def specs_from_names(
    names: list[str],
    constraints: Optional[dict[str, ConstraintType]] = None,
    links: Optional[dict[str, str]] = None,
) -> list[ParameterSpec]:
    """从参数名列表批量创建 ParameterSpec 列表。

    Parameters
    ----------
    names : list[str]
        参数名称列表
    constraints : dict, optional
        {参数名: 约束类型}，未指定的用 "unrestricted"
    links : dict, optional
        {参数名: 链接函数名}，未指定的根据 constraint 自动推断

    Returns
    -------
    list[ParameterSpec]

    Examples
    --------
    >>> specs = specs_from_names(
    ...     ["mu", "sigma", "nu"],
    ...     constraints={"sigma": "positive", "nu": "positive"},
    ... )
    """
    constraints = constraints or {}
    links = links or {}
    specs = []
    for name in names:
        constraint = constraints.get(name, "unrestricted")
        link = links.get(name)
        specs.append(ParameterSpec(name=name, constraint=constraint, link=link))
    return specs


__all__ = [
    "ConstraintType",
    "ParameterSpec",
    "AUTO_LINK_MAP",
    "CONSTRAINT_VALIDATORS",
    # 预设
    "MU_SPEC",
    "SIGMA_SPEC",
    "NU_SPEC",
    "TAU_SPEC",
    "PI_SPEC",
    "MU_POS_SPEC",
    "CLASSIC_GAMLSS_PARAMS",
    # 工具函数
    "infer_param_spec",
    "specs_from_names",
]
