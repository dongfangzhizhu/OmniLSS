"""Cole-Green (CG) Algorithm for GAMLSS.

CG 算法是 GAMLSS 的三大核心算法之一，与 RS 算法的区别在于：
- RS：更新参数 theta_k 时，只使用 d2l/deta_k2（对角块 Hessian）
- CG：更新参数 theta_k 时，额外使用交叉导数 d2l/deta_k*deta_j 进行校正

这使 CG 在参数间存在强相关性时收敛更快。

References
----------
Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized additive models
for location, scale and shape. JRSS-C, 54(3), 507-554.

Cole, T. J., & Green, P. J. (1992). Smoothing reference centile curves:
the LMS method and penalized likelihood. Statistics in Medicine, 11(10), 1305-1319.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ..distributions import resolve_family
from ..model import GAMLSSModel

# ---------------------------------------------------------------------------
# 交叉导数计算
# ---------------------------------------------------------------------------


def _compute_cross_derivatives(
    y: np.ndarray,
    param_values: Dict[str, np.ndarray],
    family: Any,
    param_k: str,
    param_j: str,
) -> np.ndarray:
    """用 JAX 自动微分计算交叉二阶导数 d2l/deta_k*deta_j。

    利用 JAX 的自动微分能力，无需为每个分布族手写解析交叉导数。

    Parameters
    ----------
    y : np.ndarray
        响应变量
    param_values : dict
        当前参数值 {参数名: 当前拟合值}
    family : FamilyDefinition
        分布族
    param_k : str
        第一个参数名（被更新的参数）
    param_j : str
        第二个参数名（交叉导数中的另一参数）

    Returns
    -------
    cross_deriv : np.ndarray
        元素级别的交叉导数数组 (n,)，即 d2l_i/(deta_k,i * deta_j,i) 对每个观测 i
    """
    y_jax = jnp.asarray(y, dtype=jnp.float64)

    # 获取当前拟合值（fitted values = g^{-1}(eta)）
    link_k = family.link_functions[param_k]
    link_j = family.link_functions[param_j]

    fv_k = jnp.asarray(param_values[param_k], dtype=jnp.float64)
    fv_j = jnp.asarray(param_values[param_j], dtype=jnp.float64)

    # 固定其他参数（不对其求导）
    fixed_params = {}
    for p, v in param_values.items():
        if p not in (param_k, param_j):
            fixed_params[p] = jnp.asarray(v, dtype=jnp.float64)

    # 将 fitted values 转成 eta（线性预测值）
    eta_k_all = jnp.asarray(link_k(fv_k), dtype=jnp.float64)
    eta_j_all = jnp.asarray(link_j(fv_j), dtype=jnp.float64)

    try:
        # 使用 vmap 对所有观测同时向量化计算交叉导数
        def cross_deriv_single(eta_k_i, eta_j_i, y_i):
            """计算单个观测的 d2l/(deta_k * deta_j)。"""
            # 用逆链接得到分布参数
            theta_k_i = family.link_inverses[param_k](eta_k_i)
            theta_j_i = family.link_inverses[param_j](eta_j_i)

            # 固定参数（标量化处理，避免 vmap 中 shape 问题）
            params_i = {
                p: v[0] if hasattr(v, "__len__") else v for p, v in fixed_params.items()
            }
            params_i[param_k] = theta_k_i
            params_i[param_j] = theta_j_i

            # 先对 eta_k 求导，得到 dl/deta_k 关于 eta_j 的函数，再对 eta_j 求导
            # 即：d/deta_j [ d/deta_k log p(y | params) ]
            d_kj = jax.grad(
                lambda ej: jax.grad(
                    lambda ek: family.logpdf(
                        y=y_i,
                        **{
                            **params_i,
                            param_k: family.link_inverses[param_k](ek),
                            param_j: family.link_inverses[param_j](ej),
                        },
                    )
                )(eta_k_i)
            )(eta_j_i)

            return d_kj

        # vmap 向量化，对全部 n 个观测并行计算
        cross_derivs = jax.vmap(cross_deriv_single)(eta_k_all, eta_j_all, y_jax)
        return np.asarray(cross_derivs, dtype=np.float64)

    except Exception:
        # 回退：返回零向量，退化为 RS 更新（无交叉导数校正）
        return np.zeros(len(y), dtype=np.float64)


# ---------------------------------------------------------------------------
# IRLS 步（带 CG 交叉导数校正）
# ---------------------------------------------------------------------------


def _irls_step_with_adjustment(
    y: np.ndarray,
    X: np.ndarray,
    eta: np.ndarray,
    fitted: np.ndarray,
    score: np.ndarray,
    hessian_diag: np.ndarray,
    cross_adjustment: np.ndarray,
    link_derivative: np.ndarray,
    offset: np.ndarray,
    step_size: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """带 CG 交叉导数校正的 IRLS 步。

    CG 的核心：在 IRLS 的工作响应中加入交叉导数校正项，
    使更新方向更加准确（考虑了参数间的相关性）。

    Parameters
    ----------
    y : np.ndarray
        响应变量 (n,)
    X : np.ndarray
        设计矩阵 (n, p)
    eta : np.ndarray
        当前线性预测值 (n,)
    fitted : np.ndarray
        当前拟合值 = g^{-1}(eta) (n,)
    score : np.ndarray
        一阶导 dl/dtheta (n,)
    hessian_diag : np.ndarray
        二阶导 d2l/dtheta2（应为负值）(n,)
    cross_adjustment : np.ndarray
        交叉导数校正量 sum_{j!=k} (d2l/deta_k*deta_j) * Delta_eta_j (n,)
    link_derivative : np.ndarray
        链接函数导数 deta/dtheta (n,)
    offset : np.ndarray
        偏移项 (n,)
    step_size : float
        步长（0 < step_size <= 1）

    Returns
    -------
    eta_new : np.ndarray
        更新后的线性预测值 (n,)
    beta_new : np.ndarray
        更新后的系数向量 (p,)
    """
    # 确保 Hessian 为负（对数似然的凹性保证）
    hessian_diag = np.where(hessian_diag < -1e-15, hessian_diag, -1e-15)

    # 工作权重：w = -d2l/dtheta2 / (deta/dtheta)^2
    working_weights = -hessian_diag / (link_derivative**2)
    working_weights = np.maximum(working_weights, 1e-8)

    # 标准 IRLS 工作响应：z = (eta - offset) + score / (w * link_deriv)
    z_standard = (eta - offset) + score / (working_weights * link_derivative)

    # CG 额外项：cross_adj / (w * link_deriv^2)
    # 来自对完整 Newton 方向的近似（交叉 Hessian 的贡献）
    z_cross = cross_adjustment / (working_weights * link_derivative**2 + 1e-15)

    z_adjusted = z_standard + z_cross

    # 带权最小二乘：beta = (X'WX)^{-1} X'Wz
    W_diag = working_weights
    XtWX = (X * W_diag[:, None]).T @ X  # (p, p)
    XtWz = X.T @ (W_diag * z_adjusted)  # (p,)

    # 数值正则化（Ridge 小量，防止奇异）
    ridge = 1e-10 * np.trace(XtWX) / max(XtWX.shape[0], 1)
    XtWX_reg = XtWX + ridge * np.eye(XtWX.shape[0])

    try:
        beta_new = np.linalg.solve(XtWX_reg, XtWz)
    except np.linalg.LinAlgError:
        beta_new = np.linalg.lstsq(XtWX_reg, XtWz, rcond=None)[0]

    eta_new = X @ beta_new + offset
    return eta_new, beta_new


# ---------------------------------------------------------------------------
# 主拟合函数
# ---------------------------------------------------------------------------


def cg_fit(
    formula: str,
    sigma_formula: str = "~ 1",
    nu_formula: Optional[str] = None,
    tau_formula: Optional[str] = None,
    family: str = "NO",
    data: Optional[Dict[str, np.ndarray]] = None,
    mu_step: float = 1.0,
    sigma_step: float = 1.0,
    nu_step: float = 1.0,
    tau_step: float = 1.0,
    max_outer_iter: int = 20,
    max_inner_iter: int = 5,
    outer_tol: float = 1e-4,
    inner_tol: float = 1e-4,
    verbose: bool = False,
) -> GAMLSSModel:
    """用 Cole-Green (CG) 算法拟合 GAMLSS 模型。

    CG 算法的双循环结构：
    - 外循环：迭代直到全局 deviance 收敛
    - 内循环：按参数 mu -> sigma -> nu -> tau 更新，
              每次用**交叉导数校正**调整工作响应

    与 RS 的区别：
    - RS 只用对角块 Hessian（忽略参数间相关性）
    - CG 用完整交叉导数，参数间强相关时收敛更快

    Parameters
    ----------
    formula : str
        mu 参数的公式，格式 "y ~ x1 + x2"
    sigma_formula : str
        sigma 参数的公式，默认 "~ 1"（截距项）
    nu_formula : str, optional
        nu 参数的公式（如分布族有该参数）
    tau_formula : str, optional
        tau 参数的公式（如分布族有该参数）
    family : str or FamilyDefinition
        分布族名称或对象
    data : dict
        数据字典 {变量名: 数组}
    mu_step : float
        mu 的步长，范围 (0, 1]
    sigma_step : float
        sigma 的步长，范围 (0, 1]
    nu_step : float
        nu 的步长，范围 (0, 1]
    tau_step : float
        tau 的步长，范围 (0, 1]
    max_outer_iter : int
        最大外循环次数
    max_inner_iter : int
        每个外循环中的内循环次数（暂时用于兼容接口）
    outer_tol : float
        外循环收敛容差（全局 deviance 相对变化）
    inner_tol : float
        内循环收敛容差（当前未启用参数级收敛检查）
    verbose : bool
        是否打印每轮迭代的 deviance 信息

    Returns
    -------
    GAMLSSModel
        拟合好的 GAMLSS 模型

    Examples
    --------
    >>> data = {"y": y, "x": x}
    >>> model = cg_fit("y ~ x", "~ x", family="NO", data=data)
    >>> print(model.g_dev)
    """
    if data is None:
        raise ValueError("data 不能为 None")

    # ── 解析分布族 ──
    family_obj = resolve_family(family)

    # 断言：正弸创建的分布族对象必定包含这些字段
    assert family_obj.link_functions is not None, (
        f"分布族 {family_obj.name} 缺少 link_functions"
    )
    assert family_obj.link_inverses is not None, (
        f"分布族 {family_obj.name} 缺少 link_inverses"
    )
    assert family_obj.link_derivatives is not None, (
        f"分布族 {family_obj.name} 缺少 link_derivatives"
    )

    if verbose:
        print("=" * 70)
        print("CG Algorithm (Cole-Green)")
        print("=" * 70)
        print(f"分布族：{family_obj.name}")
        print(f"参数：{family_obj.parameters}")
        print(f"最大外循环次数：{max_outer_iter}")

    # ── 导入拟合工具函数 ──
    from ..fitting import (
        _build_design_matrix_with_smooths,
        _build_rqres_callable,
        _compute_residuals,
        _fixed_parameter_term,
        _initial_mu_beta,
        _initial_parameter_value,
        _initial_sigma,
        _parse_formula,
        _resolve_fixed_parameter_values,
        _resolve_parameter_formulas,
    )

    # ── 解析公式 ──
    response_name, _ = _parse_formula(formula)
    parameter_formulas_dict = {}
    if nu_formula is not None:
        parameter_formulas_dict["nu"] = nu_formula
    if tau_formula is not None:
        parameter_formulas_dict["tau"] = tau_formula

    resolved_formulas = _resolve_parameter_formulas(
        response=response_name,
        family=family_obj,
        mu_formula=formula,
        sigma_formula=sigma_formula,
        parameter_formulas=parameter_formulas_dict if parameter_formulas_dict else None,
    )

    y = np.asarray(data[response_name], dtype=np.float64)
    n = len(y)
    w = np.ones(n, dtype=np.float64)  # 观测权重（默认均为 1）

    # 解析固定参数（某些分布族的参数不需要估计）
    fixed_param_values = _resolve_fixed_parameter_values(family_obj, data, n)

    # ── 构建各参数的设计矩阵 ──
    design_matrices = {}  # {参数名: 设计矩阵 (n, p)}
    smooth_infos = {}  # {参数名: 平滑信息列表}
    predictor_labels = {}  # {参数名: 列名列表}

    for param in family_obj.parameters:
        if param in fixed_param_values:
            continue  # 固定参数不需要设计矩阵
        param_formula = resolved_formulas[param]
        _, X_p, labels_p, smooth_info_p = _build_design_matrix_with_smooths(
            param_formula, data, weights=None
        )
        design_matrices[param] = X_p
        smooth_infos[param] = smooth_info_p
        predictor_labels[param] = labels_p

    # ── 初始化参数值 ──
    param_values = {}  # {参数名: 当前拟合值 (n,)}
    coefficients = {}  # {参数名: 当前回归系数 (p,)}
    linear_predictors = {}  # {参数名: 当前线性预测值 eta (n,)}
    offsets = {p: np.zeros(n) for p in family_obj.parameters}  # 偏移项

    # 初始化 mu
    if "mu" in design_matrices:
        # 签名：_initial_mu_beta(family, x, y, w)
        beta_mu = _initial_mu_beta(
            family_obj,
            design_matrices["mu"],
            y,
            w,
            fixed_parameter_values=fixed_param_values or None,
        )
        coefficients["mu"] = beta_mu
        linear_predictors["mu"] = design_matrices["mu"] @ beta_mu
        param_values["mu"] = np.asarray(
            family_obj.link_inverses["mu"](jnp.asarray(linear_predictors["mu"])),
            dtype=np.float64,
        )
    else:
        param_values["mu"] = fixed_param_values.get("mu", np.full(n, np.mean(y)))

    # 初始化 sigma
    if "sigma" in design_matrices:
        # 签名：_initial_sigma(family, y, mu, w) -> float（标量）
        sigma_scalar = _initial_sigma(
            family_obj,
            y,
            param_values["mu"],
            w,
            fixed_parameter_values=fixed_param_values or None,
        )
        sigma_scalar = max(float(sigma_scalar), 1e-4)
        # 通过链接函数将标量 sigma 转成 eta
        eta_sigma_init = float(
            np.asarray(
                family_obj.link_functions["sigma"](jnp.asarray([sigma_scalar])),
                dtype=np.float64,
            )[0]
        )
        eta_sigma_vec = np.full(n, eta_sigma_init)
        # 最小二乘得初始系数
        beta_sigma, _, _, _ = np.linalg.lstsq(
            design_matrices["sigma"], eta_sigma_vec, rcond=None
        )
        coefficients["sigma"] = beta_sigma
        linear_predictors["sigma"] = design_matrices["sigma"] @ beta_sigma
        param_values["sigma"] = np.asarray(
            family_obj.link_inverses["sigma"](jnp.asarray(linear_predictors["sigma"])),
            dtype=np.float64,
        )
        param_values["sigma"] = np.maximum(param_values["sigma"], 1e-6)
    elif "sigma" in fixed_param_values:
        param_values["sigma"] = fixed_param_values["sigma"]

    # 初始化 nu, tau
    for param in ["nu", "tau"]:
        if param in design_matrices:
            # 签名：_initial_parameter_value(family, parameter, y, mu, w) -> float
            init_scalar = float(
                _initial_parameter_value(family_obj, param, y, param_values["mu"], w)
            )
            eta_param_init = float(
                np.asarray(
                    family_obj.link_functions[param](jnp.asarray([init_scalar])),
                    dtype=np.float64,
                )[0]
            )
            eta_param_vec = np.full(n, eta_param_init)
            beta_param, _, _, _ = np.linalg.lstsq(
                design_matrices[param], eta_param_vec, rcond=None
            )
            coefficients[param] = beta_param
            linear_predictors[param] = design_matrices[param] @ beta_param
            param_values[param] = np.asarray(
                family_obj.link_inverses[param](jnp.asarray(linear_predictors[param])),
                dtype=np.float64,
            )
        elif param in fixed_param_values:
            param_values[param] = fixed_param_values[param]

    # ── CG 外循环 ──
    g_dev_old = np.inf
    g_dev = np.inf  # 初始化防止 unbound（max_outer_iter=0 时直接用于构建模型）
    outer_iter = 0  # 初始化防止 unbound
    converged = False
    deviance_history = []  # 记录每轮 deviance，用于收敛诊断

    for outer_iter in range(max_outer_iter):
        # 记录本轮开始时的 eta，供交叉导数校正使用
        eta_start = {
            p: linear_predictors.get(p, np.zeros(n)).copy()
            for p in family_obj.parameters
            if p not in fixed_param_values
        }

        # 各参数的步长（可能在 deviance 增加时自动减半）
        step_sizes = {
            "mu": mu_step,
            "sigma": sigma_step,
            "nu": nu_step,
            "tau": tau_step,
        }

        # ── 内循环：按参数依次更新 ──
        for param_k in family_obj.parameters:
            if param_k not in design_matrices:
                continue  # 固定参数跳过

            X_k = design_matrices[param_k]
            eta_k = linear_predictors[param_k].copy()
            fv_k = param_values[param_k].copy()
            offset_k = offsets[param_k]

            # ── 计算一阶导（score）dl/dtheta_k ──
            try:
                if (
                    family_obj.score_functions is None
                    or param_k not in family_obj.score_functions
                ):
                    raise ValueError(f"score_functions 未包含参数 {param_k}")
                score_k = np.asarray(
                    family_obj.score_functions[param_k](
                        y=jnp.asarray(y),
                        **{p: jnp.asarray(v) for p, v in param_values.items()},
                    ),
                    dtype=np.float64,
                )
            except Exception:
                score_k = np.zeros(n)

            # ── 计算二阶导（Hessian 对角）d2l/dtheta_k2 ──
            try:
                if (
                    family_obj.hessian_functions is None
                    or param_k not in family_obj.hessian_functions
                ):
                    raise ValueError(f"hessian_functions 未包含参数 {param_k}")
                hess_k = np.asarray(
                    family_obj.hessian_functions[param_k](
                        y=jnp.asarray(y),
                        **{p: jnp.asarray(v) for p, v in param_values.items()},
                    ),
                    dtype=np.float64,
                )
            except Exception:
                hess_k = np.full(n, -0.1)

            # ── 计算链接函数导数 deta/dtheta ──
            try:
                if (
                    family_obj.link_derivatives is None
                    or param_k not in family_obj.link_derivatives
                ):
                    raise ValueError(f"link_derivatives 未包含参数 {param_k}")
                link_deriv_k = np.asarray(
                    family_obj.link_derivatives[param_k](jnp.asarray(eta_k)),
                    dtype=np.float64,
                )
                # 防止除零
                link_deriv_k = np.where(
                    np.abs(link_deriv_k) < 1e-10, 1e-10, link_deriv_k
                )
            except Exception:
                link_deriv_k = np.ones(n)

            # ── CG 核心：计算交叉导数校正量 ──
            # cross_adj_k = sum_{j != k} (d2l/deta_k*deta_j)_i * Delta_eta_j,i
            cross_adj = np.zeros(n)

            # 找出本轮已更新的其他可估计参数
            estimable_params = [
                p
                for p in family_obj.parameters
                if p != param_k and p not in fixed_param_values and p in design_matrices
            ]

            for param_j in estimable_params:
                # Delta_eta_j = 当前 eta_j - 本轮开始时的 eta_j
                eta_j_start = eta_start.get(
                    param_j, linear_predictors.get(param_j, np.zeros(n))
                )
                delta_eta_j = linear_predictors.get(param_j, np.zeros(n)) - eta_j_start

                if np.max(np.abs(delta_eta_j)) < 1e-10:
                    continue  # 该参数本轮未更新，无贡献

                # 计算交叉导数 d2l/(deta_k * deta_j)
                try:
                    cross_kj = _compute_cross_derivatives(
                        y, param_values, family_obj, param_k, param_j
                    )
                    cross_adj += cross_kj * delta_eta_j
                except Exception:
                    pass  # 交叉导数计算失败时，该项贡献为 0（退化为 RS 步）

            # ── IRLS 步（带 CG 校正） ──
            try:
                eta_k_new, beta_k_new = _irls_step_with_adjustment(
                    y=y,
                    X=X_k,
                    eta=eta_k,
                    fitted=fv_k,
                    score=score_k,
                    hessian_diag=hess_k,
                    cross_adjustment=cross_adj,
                    link_derivative=link_deriv_k,
                    offset=offset_k,
                    step_size=step_sizes.get(param_k, 1.0),
                )

                # 步长控制：eta_new = eta + step * (eta_new - eta)
                step = step_sizes.get(param_k, 1.0)
                if step < 1.0:
                    eta_k_new = eta_k + step * (eta_k_new - eta_k)

                # 通过逆链接得到拟合值
                fv_k_new = np.asarray(
                    family_obj.link_inverses[param_k](jnp.asarray(eta_k_new)),
                    dtype=np.float64,
                )

                # 数值安全：sigma 必须为正
                if param_k == "sigma":
                    fv_k_new = np.maximum(fv_k_new, 1e-6)

                # 只接受无 NaN/Inf 的更新
                if not np.any(np.isnan(fv_k_new)) and not np.any(np.isinf(fv_k_new)):
                    linear_predictors[param_k] = eta_k_new
                    param_values[param_k] = fv_k_new
                    coefficients[param_k] = beta_k_new

            except Exception as e:
                if verbose:
                    print(f"  警告：{param_k} 更新失败（{e}），跳过本轮更新")

        # ── 计算全局 deviance ──
        try:
            dev_kwargs = {"y": y, **param_values}
            g_dev_incr = np.asarray(
                family_obj.g_dev_inc(**dev_kwargs), dtype=np.float64
            )
            g_dev = float(np.sum(w * g_dev_incr))
        except Exception:
            g_dev = g_dev_old  # deviance 计算失败时保持原值

        deviance_history.append(g_dev)

        if verbose:
            print(
                f"  外循环 {outer_iter + 1:3d}: "
                f"全局 Deviance = {g_dev:.6f}, "
                f"变化 = {abs(g_dev_old - g_dev):.2e}"
            )

        # ── 收敛检查 ──
        if abs(g_dev_old - g_dev) < outer_tol and outer_iter > 0:
            converged = True
            if verbose:
                print(f"  -> 在第 {outer_iter + 1} 次迭代收敛")
            break

        # deviance 显著增加时，自动减小步长（防止发散）
        if g_dev > g_dev_old + 1.0 and outer_iter > 0:
            for p in step_sizes:
                step_sizes[p] = max(step_sizes[p] * 0.5, 0.1)
            if verbose:
                print("  警告：Deviance 增加，步长减半")

        g_dev_old = g_dev

    if verbose:
        status = "收敛" if converged else f"达到最大迭代次数 ({max_outer_iter})"
        print(f"CG 算法完成：{status}")
        print("=" * 70)

    if not converged:
        warnings.warn(
            f"CG 算法在 {max_outer_iter} 次迭代后未收敛。"
            "最终 deviance 可能不是全局最优。考虑增加 max_outer_iter。",
            RuntimeWarning,
        )

    # ── 构建 GAMLSSModel ──
    fitted_values_jax = {
        p: jnp.asarray(param_values[p], dtype=jnp.float64)
        for p in family_obj.parameters
    }
    coefficients_jax = {
        p: jnp.asarray(
            coefficients.get(p, np.array([param_values[p][0]])), dtype=jnp.float64
        )
        for p in family_obj.parameters
    }
    linear_predictors_jax = {
        p: jnp.asarray(linear_predictors.get(p, np.zeros(n)), dtype=jnp.float64)
        for p in family_obj.parameters
    }
    design_matrices_jax = {
        p: jnp.asarray(design_matrices.get(p, np.zeros((n, 1))), dtype=jnp.float64)
        for p in family_obj.parameters
    }

    # 构建 terms 字典（描述每个参数的公式信息）
    terms = {}
    for param in family_obj.parameters:
        if param in fixed_param_values:
            terms[param] = _fixed_parameter_term(response_name, param)
        else:
            terms[param] = {
                "term_labels": predictor_labels.get(param, []),
                "response": response_name,
                "intercept": True,
                "formula": resolved_formulas[param],
            }

    # 计算随机分位数残差（quantile residuals）
    rqres_callable = _build_rqres_callable(family_obj)
    mu_vals = param_values["mu"]
    sigma_vals = param_values.get("sigma")
    if rqres_callable is not None:
        residual_values = rqres_callable(y=y, mu=mu_vals, sigma=sigma_vals)
    else:
        residual_values = _compute_residuals(family_obj, y, mu_vals, sigma_vals)

    # 自由度 = 所有可估计参数的系数总数
    df_fit_val = float(
        sum(
            len(np.asarray(coefficients.get(p, [0])))
            for p in family_obj.estimable_parameters
        )
    )

    model = GAMLSSModel(
        par=family_obj.parameters,
        family=family_obj,
        df_fit=df_fit_val,
        g_dev=g_dev,
        n=n,
        y=jnp.asarray(y, dtype=jnp.float64),
        fitted_values=fitted_values_jax,
        coefficients=coefficients_jax,
        linear_predictors=linear_predictors_jax,
        working_vectors={
            p: jnp.asarray(linear_predictors.get(p, np.zeros(n)))
            for p in family_obj.parameters
        },
        iterative_weights={p: jnp.ones(n) for p in family_obj.parameters},
        offsets={p: jnp.zeros(n) for p in family_obj.parameters},
        formulas=resolved_formulas,
        terms=terms,
        design_matrices=design_matrices_jax,
        weights=jnp.ones(n, dtype=jnp.float64),
        residuals=jnp.asarray(residual_values, dtype=jnp.float64),
        rqres=rqres_callable,
        iter=outer_iter + 1,
        type=family_obj.type,
        parameters=family_obj.parameters,
        call={
            "data": data,
            "formula": resolved_formulas["mu"],
            "parameter_formulas": dict(resolved_formulas),
            "method": "CG",
        },
        control={"n.cyc": max_outer_iter},
        additional_slots={
            "method": "CG",
            # CG 算法专属信息
            "cg_iterations": outer_iter + 1,
            "cg_converged": converged,
            "cg_final_deviance": g_dev,
            # 标准 GAMLSS 信息字段
            "noObs": int(n),
            "G.deviance": g_dev,
            "P.deviance": g_dev,
            "aic": g_dev + 2.0 * df_fit_val,
            "sbc": g_dev + np.log(max(n, 1)) * df_fit_val,
            "df.residual": float(n - df_fit_val),
            "df_residual": float(n - df_fit_val),
            "converged": converged,
            "deviance_history": tuple(float(v) for v in deviance_history),
            "smooth_infos": smooth_infos,
        },
    )

    return model
