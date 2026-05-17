"""预测功能模块

为 GAMLSSModel 提供预测功能，包括：
- 参数预测 (predict_params)
- 分位数预测 (predict_quantiles)
- Centile curves (centiles)

这是 GAMLSS 的核心价值：预测完整分布，而不仅仅是均值。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import jax.numpy as jnp
import numpy as np
import pandas as pd


class PredictionSchemaError(ValueError):
    """Raised when prediction data cannot reproduce the training design schema."""


def predict_params(
    model: Any, newdata: Dict[str, np.ndarray], which: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """预测分布参数

    对新数据预测所有分布参数（μ, σ, ν, τ）。

    Parameters
    ----------
    model : GAMLSSModel
        拟合的 GAMLSS 模型
    newdata : dict
        新数据，字典格式 {变量名: 值数组}
    which : list of str, optional
        要预测的参数列表，默认预测所有可估计参数

    Returns
    -------
    params : dict
        预测的参数值 {参数名: 预测值数组}

    Examples
    --------
    >>> model = gamlss("y ~ x1 + x2", family="NO", data=train_data)
    >>> newdata = {"x1": np.array([1, 2, 3]), "x2": np.array([4, 5, 6])}
    >>> params = predict_params(model, newdata)
    >>> print(params["mu"])  # 预测的 μ 值
    >>> print(params["sigma"])  # 预测的 σ 值

    Notes
    -----
    这个函数是分位数预测和 centile curves 的基础。
    """
    if which is None:
        which = list(model.family.estimable_parameters)

    params = {}

    for param in which:
        if param not in model.coefficients:
            continue

        # 获取系数
        beta = np.asarray(model.coefficients[param], dtype=np.float64)

        # 构建设计矩阵
        X_new = _build_design_matrix_for_prediction(model, param, newdata)

        # 线性预测: η = X @ β
        eta = X_new @ beta

        # 应用逆链接函数: param = g^{-1}(η)
        param_values = np.asarray(
            model.family.link_inverses[param](jnp.asarray(eta, dtype=jnp.float64)),
            dtype=np.float64,
        )

        params[param] = param_values

    # 添加固定参数
    for param in model.family.fixed_parameters or []:
        if param in newdata:
            params[param] = np.asarray(newdata[param], dtype=np.float64)

    return params


def _build_design_matrix_for_prediction(
    model: Any, param: str, newdata: Dict[str, np.ndarray]
) -> np.ndarray:
    """Build a prediction design matrix without silent schema fallbacks."""

    import re

    import jax.numpy as jnp

    from ._fitting_utils import _eval_linear_term

    if newdata:
        n = len(next(iter(newdata.values())))
    else:
        n = int(getattr(model, "n", 0))
        if n <= 0:
            raise PredictionSchemaError(
                f"Cannot infer prediction row count for parameter {param!r}"
            )

    additional_slots = getattr(model, "additional_slots", {}) or {}
    smooth_infos = additional_slots.get("smooth_infos", {})
    param_smooth_info = smooth_infos.get(param)
    design_schema = (additional_slots.get("design_matrix_schema", {}) or {}).get(
        "parameters", {}
    ).get(param, {})
    formula = model.formulas.get(param, "")
    if not formula or "~" not in formula:
        raise PredictionSchemaError(
            f"Missing formula schema for parameter {param!r}; cannot build prediction matrix"
        )

    rhs = formula.split("~", 1)[1].strip()
    columns: list[np.ndarray] = []
    has_intercept = not ("-1" in rhs or rhs == "0")
    if has_intercept:
        columns.append(np.ones(n, dtype=np.float64))

    def split_terms(rhs_str: str):
        terms = []
        depth = 0
        current = []
        for ch in rhs_str:
            if ch in "([":
                depth += 1
                current.append(ch)
            elif ch in ")]":
                depth -= 1
                current.append(ch)
            elif ch == "+" and depth == 0:
                term = "".join(current).strip()
                if term and term not in {"1", "-1", "0"}:
                    terms.append(term)
                current = []
            else:
                current.append(ch)
        term = "".join(current).strip()
        if term and term not in {"1", "-1", "0"}:
            terms.append(term)
        return terms

    smooth_funcs = {"pb", "ps", "cs", "s", "lo", "te", "ti"}
    for term_str in split_terms(rhs):
        smooth_match = re.match(r"^(\w+)\((.+)\)$", term_str, re.DOTALL)
        if smooth_match and smooth_match.group(1) in smooth_funcs:
            inner = smooth_match.group(2)
            var_name = inner.split(",")[0].strip()
            smooth_info_for_var = None
            if param_smooth_info and isinstance(param_smooth_info, list):
                for si in param_smooth_info:
                    if si.get("variable") == var_name or si.get("var") == var_name:
                        smooth_info_for_var = si
                        break
            elif param_smooth_info and isinstance(param_smooth_info, dict):
                smooth_info_for_var = param_smooth_info.get(var_name)

            if smooth_info_for_var is None:
                raise PredictionSchemaError(
                    f"Missing smooth metadata for term {term_str!r} in parameter {param!r}"
                )
            if var_name not in newdata:
                raise PredictionSchemaError(
                    f"Missing variable {var_name!r} required by smooth term {term_str!r}"
                )

            smoother_type = smooth_info_for_var.get("smoother", "pb")
            if smoother_type not in {"pb", "ps"}:
                raise PredictionSchemaError(
                    f"Prediction for smoother {smoother_type!r} is not schema-safe yet"
                )
            try:
                from .smoothers.bsplines import bspline_basis

                x_new = np.asarray(newdata[var_name], dtype=np.float64)
                knots = np.asarray(smooth_info_for_var["knots"])
                degree = int(smooth_info_for_var.get("degree", 3))
                basis_new = np.array(
                    bspline_basis(jnp.array(x_new), jnp.array(knots), degree=degree)
                )
            except Exception as exc:
                raise PredictionSchemaError(
                    f"Failed to rebuild smooth term {term_str!r}: {exc}"
                ) from exc
            columns.append(basis_new)
        else:
            try:
                columns.append(_eval_linear_term(term_str, newdata, n).reshape(-1))
            except Exception as exc:
                raise PredictionSchemaError(
                    f"Failed to evaluate prediction term {term_str!r}: {exc}"
                ) from exc

    if not columns:
        raise PredictionSchemaError(
            f"Formula for parameter {param!r} produced no prediction columns"
        )

    result_cols = []
    for col in columns:
        arr = np.asarray(col, dtype=np.float64)
        if arr.ndim == 1:
            if len(arr) != n:
                raise PredictionSchemaError(
                    f"Prediction column for parameter {param!r} has length {len(arr)}, expected {n}"
                )
            result_cols.append(arr.reshape(n, 1))
        else:
            if arr.shape[0] != n:
                raise PredictionSchemaError(
                    f"Prediction matrix block for parameter {param!r} has {arr.shape[0]} rows, expected {n}"
                )
            result_cols.append(arr)

    try:
        X_new = np.hstack(result_cols)
    except Exception as exc:
        raise PredictionSchemaError(
            f"Failed to assemble prediction matrix for parameter {param!r}: {exc}"
        ) from exc

    expected_columns = design_schema.get("n_columns")
    if expected_columns is not None and X_new.shape[1] != int(expected_columns):
        raise PredictionSchemaError(
            f"Prediction design for parameter {param!r} has {X_new.shape[1]} columns, "
            f"but saved schema expects {int(expected_columns)} columns"
        )

    if param in model.coefficients:
        n_coefs = len(np.asarray(model.coefficients[param]))
        if X_new.shape[1] != n_coefs:
            raise PredictionSchemaError(
                f"Prediction design for parameter {param!r} has {X_new.shape[1]} columns, "
                f"but model has {n_coefs} coefficients"
            )

    return X_new


def predict_quantiles(
    model: Any,
    newdata: Dict[str, np.ndarray],
    quantiles: List[float] = [0.05, 0.25, 0.5, 0.75, 0.95],
) -> Dict[float, np.ndarray]:
    """预测条件分位数

    这是 GAMLSS 的核心功能：预测完整的条件分布，而不仅仅是条件均值。

    Parameters
    ----------
    model : GAMLSSModel
        拟合的 GAMLSS 模型
    newdata : dict
        新数据 {变量名: 值数组}
    quantiles : list of float, default=[0.05, 0.25, 0.5, 0.75, 0.95]
        要预测的分位数列表，取值范围 (0, 1)

    Returns
    -------
    results : dict
        {分位数: 预测值数组}

    Examples
    --------
    >>> # 拟合模型
    >>> model = gamlss("y ~ x", family="NO", data=data)
    >>>
    >>> # 预测新数据的分位数
    >>> newdata = {"x": np.array([1, 2, 3])}
    >>> quantiles = predict_quantiles(model, newdata, quantiles=[0.05, 0.5, 0.95])
    >>>
    >>> # 查看结果
    >>> print("5% quantile:", quantiles[0.05])
    >>> print("50% quantile (median):", quantiles[0.5])
    >>> print("95% quantile:", quantiles[0.95])
    >>>
    >>> # 预测区间
    >>> lower = quantiles[0.05]
    >>> upper = quantiles[0.95]
    >>> print(f"90% prediction interval: [{lower}, {upper}]")

    Notes
    -----
    这个函数使用分布的分位数函数 (q function) 来计算条件分位数。
    对于每个新观测，它首先预测分布参数，然后计算相应的分位数。

    与传统回归的区别：
    - 传统回归：只预测 E[Y|X]
    - GAMLSS：预测完整分布 F(Y|X)，可以得到任意分位数
    """
    # 验证分位数范围
    for q in quantiles:
        if not 0 < q < 1:
            raise ValueError(f"Quantiles must be in (0, 1), got {q}")

    # 预测分布参数
    params = predict_params(model, newdata)

    # 检查分布是否有分位数函数
    if not hasattr(model.family, "q") or model.family.q is None:
        raise ValueError(
            f"Family {model.family.name} does not have a quantile function (q). "
            "Cannot compute quantiles."
        )

    # 计算每个分位数
    results = {}
    for q in quantiles:
        # 使用分布的 q 函数
        # q(p, mu, sigma, ...) 返回 P(Y <= y) = p 的 y 值
        q_values = model.family.q(q, **params)
        results[q] = np.asarray(q_values, dtype=np.float64)

    return results


def centiles(
    model: Any,
    xvar: str,
    xvalues: Optional[np.ndarray] = None,
    cent: List[float] = [5, 25, 50, 75, 95],
    n_points: int = 100,
    **fixed_vars,
) -> pd.DataFrame:
    """生成 centile curves（百分位曲线）

    Centile curves 是 GAMLSS 最重要的可视化工具，展示条件分布如何随协变量变化。

    Parameters
    ----------
    model : GAMLSSModel
        拟合的 GAMLSS 模型
    xvar : str
        X 轴变量名（通常是年龄、时间等）
    xvalues : np.ndarray, optional
        X 轴的值，如果未指定则使用训练数据的范围
    cent : list of float, default=[5, 25, 50, 75, 95]
        百分位数列表（0-100）
    n_points : int, default=100
        生成的点数（当 xvalues 未指定时）
    **fixed_vars
        其他变量的固定值

    Returns
    -------
    df : pd.DataFrame
        包含 centile curves 的数据框
        列: xvar, C5, C25, C50, C75, C95, ...

    Examples
    --------
    >>> # 拟合生长曲线模型
    >>> model = gamlss("height ~ s(age)", family="NO", data=growth_data)
    >>>
    >>> # 生成 centile curves
    >>> curves = centiles(model, xvar="age", cent=[5, 50, 95])
    >>>
    >>> # 可视化
    >>> import matplotlib.pyplot as plt
    >>> plt.figure(figsize=(10, 6))
    >>> for c in [5, 50, 95]:
    >>>     plt.plot(curves["age"], curves[f"C{c}"], label=f"{c}%")
    >>> plt.xlabel("Age")
    >>> plt.ylabel("Height")
    >>> plt.title("Growth Centile Curves")
    >>> plt.legend()
    >>> plt.grid(True, alpha=0.3)
    >>> plt.show()

    >>> # 多变量情况：固定其他变量
    >>> model = gamlss("y ~ s(age) + sex", family="NO", data=data)
    >>> curves_male = centiles(model, xvar="age", sex=1)  # 男性
    >>> curves_female = centiles(model, xvar="age", sex=0)  # 女性

    Notes
    -----
    Centile curves 在以下领域广泛应用：
    - 儿童生长标准（WHO, CDC）
    - 医学参考范围
    - 环境监测标准
    - 金融风险管理

    与传统回归曲线的区别：
    - 传统回归：只有一条均值曲线
    - GAMLSS centiles：多条百分位曲线，展示完整分布
    """
    # 验证百分位数范围
    for c in cent:
        if not 0 < c < 100:
            raise ValueError(f"Centiles must be in (0, 100), got {c}")

    # 生成 X 值
    if xvalues is None:
        # 使用训练数据的范围
        if model.call is None or "data" not in model.call:
            raise ValueError(
                "Cannot determine xvalues automatically. "
                "Please provide xvalues explicitly."
            )

        x_train = model.call["data"].get(xvar)
        if x_train is None:
            raise ValueError(f"Variable {xvar} not found in training data")

        x_train = np.asarray(x_train)
        xvalues = np.linspace(np.min(x_train), np.max(x_train), n_points)

    # 构建 newdata
    newdata = {xvar: xvalues, **fixed_vars}

    # 转换百分位数为分位数
    quantiles = [c / 100 for c in cent]

    # 预测分位数
    pred_quantiles = predict_quantiles(model, newdata, quantiles)

    # 构建 DataFrame
    result = pd.DataFrame({xvar: xvalues})
    for c, q in zip(cent, quantiles):
        result[f"C{c}"] = pred_quantiles[q]

    return result


def predict_response(
    model: Any, newdata: Dict[str, np.ndarray], type: str = "response"
) -> np.ndarray:
    """预测响应变量

    Parameters
    ----------
    model : GAMLSSModel
        拟合的模型
    newdata : dict
        新数据
    type : str, default="response"
        预测类型:
        - "response": 预测响应变量的期望值 E[Y|X]
        - "link": 预测线性预测值 η
        - "terms": 预测各项的贡献

    Returns
    -------
    predictions : np.ndarray
        预测值

    Examples
    --------
    >>> model = gamlss("y ~ x", family="NO", data=data)
    >>> newdata = {"x": np.array([1, 2, 3])}
    >>>
    >>> # 预测期望值
    >>> y_pred = predict_response(model, newdata, type="response")
    >>>
    >>> # 预测线性预测值
    >>> eta_pred = predict_response(model, newdata, type="link")
    """
    if type == "response":
        # 预测 μ（对于大多数分布，E[Y] = μ）
        params = predict_params(model, newdata, which=["mu"])
        return params["mu"]

    elif type == "link":
        # 预测 η_μ
        beta = np.asarray(model.coefficients["mu"], dtype=np.float64)
        X_new = _build_design_matrix_for_prediction(model, "mu", newdata)
        return X_new @ beta

    elif type == "terms":
        raise NotImplementedError("type='terms' is not yet implemented")

    else:
        raise ValueError(f"Unknown type: {type}")


# 为 GAMLSSModel 添加方法的辅助函数
def add_prediction_methods(model_class):
    """为 GAMLSSModel 类添加预测方法

    这个函数将预测方法动态添加到 GAMLSSModel 类中。

    Parameters
    ----------
    model_class : type
        GAMLSSModel 类

    Examples
    --------
    >>> from omnilss.model import GAMLSSModel
    >>> from omnilss.prediction import add_prediction_methods
    >>> add_prediction_methods(GAMLSSModel)
    >>>
    >>> # 现在可以直接调用
    >>> model = gamlss(...)
    >>> quantiles = model.predict_quantiles(newdata, [0.05, 0.95])
    """

    def _predict_params(self, newdata, which=None):
        return predict_params(self, newdata, which)

    def _predict_quantiles(self, newdata, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]):
        return predict_quantiles(self, newdata, quantiles)

    def _centiles(
        self, xvar, xvalues=None, cent=[5, 25, 50, 75, 95], n_points=100, **fixed_vars
    ):
        return centiles(self, xvar, xvalues, cent, n_points, **fixed_vars)

    def _predict(self, newdata, type="response"):
        return predict_response(self, newdata, type)

    # 添加方法
    model_class.predict_params = _predict_params
    model_class.predict_quantiles = _predict_quantiles
    model_class.centiles = _centiles
    model_class.predict = _predict

    return model_class
