"""Tensor product smooth implementation

实现 tensor product smooth (te, ti) 用于多维平滑。

Tensor product smooth 是处理多变量交互的关键技术，允许对多个变量
的联合效应进行平滑建模。

Examples
--------
>>> from omnilss.smoothers.tensor_smooth import create_tensor_basis, te, ti
>>> import numpy as np
>>>
>>> # 创建 2D tensor product basis
>>> x1 = np.random.randn(100)
>>> x2 = np.random.randn(100)
>>> basis, penalty = create_tensor_basis(x1, x2, k1=10, k2=10)
>>>
>>> # 在公式中使用
>>> # gamlss("y ~ te(x1, x2)", family="NO", data=data)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TensorProductInfo:
    """Tensor product smooth 信息

    Attributes
    ----------
    variables : list of str
        变量名列表
    marginal_bases : list of np.ndarray
        每个变量的边际基函数
    marginal_penalties : list of np.ndarray
        每个变量的边际惩罚矩阵
    tensor_basis : np.ndarray
        Tensor product basis
    tensor_penalty : np.ndarray
        Tensor product penalty
    lambda_ : float, optional
        平滑参数
    edf : float
        有效自由度
    """

    variables: List[str]
    marginal_bases: List[np.ndarray]
    marginal_penalties: List[np.ndarray]
    tensor_basis: np.ndarray
    tensor_penalty: np.ndarray
    lambda_: Optional[float] = None
    edf: float = 0.0


def create_tensor_basis(
    *X_vars: np.ndarray, k: int = 10, k_list: Optional[List[int]] = None, bs: str = "ps"
) -> Tuple[np.ndarray, np.ndarray]:
    """创建 tensor product basis（te，包含全部主效应和交互效应）

    Parameters
    ----------
    *X_vars : np.ndarray
        变量数组，每个数组是一个变量的值 (n,)
    k : int, default=10
        每个维度的基函数数量（如果 k_list 未指定）
    k_list : list of int, optional
        每个维度的基函数数量列表
    bs : str, default="ps"
        基函数类型 ("ps" for P-splines, "cr" for cubic regression)

    Returns
    -------
    basis : np.ndarray
        Tensor product basis (n, k1*k2*...*kd)
    penalty : np.ndarray
        Tensor product penalty matrix (k1*k2*...*kd, k1*k2*...*kd)

    Notes
    -----
    Tensor product basis 是多个边际基的 Kronecker 积：
    B(x1, x2, ..., xd) = B1(x1) ⊗ B2(x2) ⊗ ... ⊗ Bd(xd)

    Tensor product penalty 是：
    P = P1 ⊗ I2 ⊗ ... ⊗ Id + I1 ⊗ P2 ⊗ ... ⊗ Id + ... + I1 ⊗ I2 ⊗ ... ⊗ Pd

    Examples
    --------
    >>> import numpy as np
    >>> x1 = np.random.randn(100)
    >>> x2 = np.random.randn(100)
    >>> basis, penalty = create_tensor_basis(x1, x2, k=10)
    >>> print(basis.shape)  # (100, 100)
    >>> print(penalty.shape)  # (100, 100)
    """
    from .bsplines import bspline_basis
    from .penalties import penalty_matrix

    n_vars = len(X_vars)
    if n_vars < 2:
        raise ValueError("Tensor product requires at least 2 variables")

    n = len(X_vars[0])

    # 确定每个维度的基函数数量
    if k_list is None:
        k_list = [k] * n_vars
    elif len(k_list) != n_vars:
        raise ValueError(
            f"k_list length ({len(k_list)}) must match number of variables ({n_vars})"
        )

    # 创建边际基和惩罚矩阵
    marginal_bases = []
    marginal_penalties = []
    actual_k_list = []  # 实际的基函数数量（可能因 knot 构造而与请求值略有差异）

    for i, (X, k_i) in enumerate(zip(X_vars, k_list)):
        if len(X) != n:
            raise ValueError(
                f"All variables must have the same length (got {len(X)} for variable {i})"
            )

        # ── 构建 knot 序列 ──
        x_min, x_max = np.min(X), np.max(X)
        x_range = x_max - x_min if x_max > x_min else 1.0
        # degree=3 时，内部 knots 数量 = k - 4
        n_interior = k_i - 4
        if n_interior > 0:
            interior_knots = np.linspace(x_min, x_max, n_interior + 2)[1:-1]
        else:
            interior_knots = np.array([])

        degree = 3
        knots = np.concatenate(
            [
                np.repeat(x_min - 0.01 * x_range, degree + 1),
                interior_knots,
                np.repeat(x_max + 0.01 * x_range, degree + 1),
            ]
        )

        # ── 创建边际 B 样条基 ──
        # 用 jnp.array 包装，确保传入 JAX Array （bspline_basis 类型注解期望 JAX ndarray）
        import jax.numpy as _jnp

        B_i = np.array(
            bspline_basis(
                _jnp.array(X, dtype=_jnp.float64),
                _jnp.array(knots, dtype=_jnp.float64),
                degree=degree,
            )
        )
        actual_k_i = B_i.shape[1]
        actual_k_list.append(actual_k_i)

        # ── 创建差分惩罚矩阵 P = D'D，order=2 ──
        P_i = penalty_matrix(actual_k_i, order=2)

        marginal_bases.append(B_i)
        marginal_penalties.append(P_i)

    # ── 构建 tensor product basis（逐行 Kronecker 积）──
    basis = marginal_bases[0]
    for B_i in marginal_bases[1:]:
        new_basis = np.zeros((n, basis.shape[1] * B_i.shape[1]))
        for obs_idx in range(n):
            new_basis[obs_idx, :] = np.kron(basis[obs_idx, :], B_i[obs_idx, :])
        basis = new_basis

    # ── 构建 tensor product penalty ──
    # P_te = P1⊗I2⊗...⊗Id + I1⊗P2⊗...⊗Id + ...
    total_dim = int(np.prod(actual_k_list))
    penalty = np.zeros((total_dim, total_dim))

    for i in range(n_vars):
        term = marginal_penalties[i]
        # 左侧 Kronecker：I_{k_0} ⊗ ... ⊗ I_{k_{i-1}} ⊗ P_i
        for j in range(i):
            term = np.kron(np.eye(actual_k_list[j]), term)
        # 右侧 Kronecker：P_i ⊗ I_{k_{i+1}} ⊗ ... ⊗ I_{k_{d-1}}
        for j in range(i + 1, n_vars):
            term = np.kron(term, np.eye(actual_k_list[j]))
        penalty += term

    return basis, penalty


def create_ti_basis(
    *X_vars: np.ndarray, k: int = 10, k_list: Optional[List[int]] = None, bs: str = "ps"
) -> Tuple[np.ndarray, np.ndarray]:
    """创建 tensor product interaction basis（去除主效应）。

    ti(x1, x2) = te(x1, x2) 的列空间投影到 span([B1, B2]) 的正交补。
    即：只保留"纯交互"部分，去掉 s(x1) 和 s(x2) 的贡献。

    实现方法：
    1. 先创建完整的 te() 基矩阵
    2. 用 QR 分解求边际基张成空间的正交基 Q_main
    3. 将 te 基投影到 Q_main 的正交补，得到 te_proj
    4. 对 te_proj 做 SVD，保留奇异值足够大的列（真正的纯交互方向）
       注：column-norm 过滤在标准 B 样条下无效，因为投影后所有列均非零。
       SVD 可以正确识别 te_proj 的秩（= k1*k2 - rank_main）
    5. 惩罚矩阵通过坐标变换 P_ti = Vt_keep @ te_penalty @ Vt_keep.T 得到

    Parameters
    ----------
    *X_vars : np.ndarray
        变量数组，每个数组是一个变量的值 (n,)
    k : int, default=10
        每个维度的基函数数量
    k_list : list of int, optional
        每个维度的基函数数量列表
    bs : str, default="ps"
        基函数类型

    Returns
    -------
    basis : np.ndarray
        Interaction-only basis (n, d)，其中 d < k1*k2
    penalty : np.ndarray
        对应的惩罚矩阵 (d, d)
    """
    from .bsplines import bspline_basis
    from .penalties import penalty_matrix

    n_vars = len(X_vars)
    if n_vars < 2:
        raise ValueError("ti() requires at least 2 variables")

    n = len(X_vars[0])

    if k_list is None:
        k_list = [k] * n_vars

    # ── 第一步：构建所有边际基 ──
    marginal_bases = []
    actual_k_list = []

    for i, (X, k_i) in enumerate(zip(X_vars, k_list)):
        x_min, x_max = np.min(X), np.max(X)
        x_range = x_max - x_min if x_max > x_min else 1.0
        n_interior = max(k_i - 4, 0)

        if n_interior > 0:
            interior_knots = np.linspace(x_min, x_max, n_interior + 2)[1:-1]
        else:
            interior_knots = np.array([])

        degree = 3
        knots = np.concatenate(
            [
                np.repeat(x_min - 0.01 * x_range, degree + 1),
                interior_knots,
                np.repeat(x_max + 0.01 * x_range, degree + 1),
            ]
        )

        import jax.numpy as _jnp

        B_i = np.array(
            bspline_basis(
                _jnp.array(X, dtype=_jnp.float64),
                _jnp.array(knots, dtype=_jnp.float64),
                degree=degree,
            )
        )
        actual_k_list.append(B_i.shape[1])
        marginal_bases.append(B_i)

    # ── 第二步：构建完整 te() 基（逐行 Kronecker 积）──
    te_basis = marginal_bases[0]
    for B_i in marginal_bases[1:]:
        new_basis = np.zeros((n, te_basis.shape[1] * B_i.shape[1]))
        for obs_idx in range(n):
            new_basis[obs_idx, :] = np.kron(te_basis[obs_idx, :], B_i[obs_idx, :])
        te_basis = new_basis

    # ── 第三步：构建所有主效应空间的联合正交基 ──
    # 主效应空间 = span(B1) ∪ span(B2) ∪ ... ∪ span(Bd)
    # 水平拼接所有边际基：B_main 的列张成所有主效应空间的并
    B_main = np.hstack(marginal_bases)  # (n, sum(k_i))

    # QR 分解得到主效应空间的规范正交基
    Q_main, R_main = np.linalg.qr(B_main, mode="reduced")

    # 只保留非退化列（|R_ii| 足够大的列）
    diag_abs = np.abs(np.diag(R_main))
    rank_tol = 1e-8 * max(diag_abs.max(), 1.0)
    rank = int(np.sum(diag_abs > rank_tol))
    Q_main = Q_main[:, :rank]  # (n, rank)

    # ── 第四步：将 te_basis 投影到主效应空间的正交补 ──
    # te_proj_i = te_basis_i - Q_main @ (Q_main' @ te_basis_i)  对每列
    proj_onto_main = Q_main @ (Q_main.T @ te_basis)
    te_proj = te_basis - proj_onto_main  # (n, k1*k2)

    # ── 第五步：用 SVD 提取 te_proj 的实际列空间（纯交互空间）──
    # te_proj 的理论秩 = k1*k2 - rank(B_main)。
    # 直接用列范数过滤在标准 B 样条下无效（所有投影列均非零），
    # 必须用 SVD 来正确识别有效维度。
    U_ti, s_ti, Vt_ti = np.linalg.svd(te_proj, full_matrices=False)

    # 保留奇异值足够大的奇异方向（截断阈值 = 1e-6 * sigma_max）
    s_tol = 1e-6 * s_ti[0] if s_ti[0] > 0 else 1e-10
    n_keep = int(np.sum(s_ti > s_tol))
    n_keep = max(n_keep, 1)  # 保底至少保留 1 列

    # ti_basis：列正交归一化，每列都与主效应空间正交
    ti_basis = U_ti[:, :n_keep]  # (n, d)，d = 纯交互自由度
    Vt_keep = Vt_ti[:n_keep, :]  # (d, k1*k2)，奇异向量（用于惩罚变换）

    # ── 第六步：构建对应的惩罚矩阵 ──
    # 先构建完整 te() 惩罚（在原始 te_basis 系数空间）
    total_dim = int(np.prod(actual_k_list))
    te_penalty = np.zeros((total_dim, total_dim))

    for i in range(n_vars):
        P_i = penalty_matrix(actual_k_list[i], order=2)
        term = P_i
        for j in range(i):
            term = np.kron(np.eye(actual_k_list[j]), term)
        for j in range(i + 1, n_vars):
            term = np.kron(term, np.eye(actual_k_list[j]))
        te_penalty += term

    # 坐标变换：从 te_basis 系数空间变换到 ti_basis 系数空间
    # 若 f = ti_basis @ c，则 beta_orig ≈ Vt_keep.T @ diag(1/s[:n_keep]) @ c
    # 惩罚变换：P_ti = Vt_keep @ te_penalty @ Vt_keep.T
    # （省略 1/s 的缩放，因为平滑参数 lambda 会在拟合时吸收尺度）
    ti_penalty = Vt_keep @ te_penalty @ Vt_keep.T

    # 对称化（消除浮点误差引入的非对称性）
    ti_penalty = (ti_penalty + ti_penalty.T) / 2.0

    return ti_basis, ti_penalty


def te(
    *variables: str, k: int = 10, k_list: Optional[List[int]] = None, bs: str = "ps"
) -> Dict[str, Any]:
    """Tensor product smooth (te)

    创建 tensor product smooth 规格，用于公式中。
    te() 包含全部主效应和交互效应。

    Parameters
    ----------
    *variables : str
        变量名列表
    k : int, default=10
        每个维度的基函数数量（如果 k_list 未指定）
    k_list : list of int, optional
        每个维度的基函数数量列表
    bs : str, default="ps"
        基函数类型 ("ps" for P-splines, "cr" for cubic regression)

    Returns
    -------
    spec : dict
        Tensor product smooth 规格

    Examples
    --------
    >>> from omnilss import gamlss
    >>> model = gamlss("y ~ te(x1, x2)", family="NO", data=data)
    >>> model = gamlss("y ~ te(x1, x2, x3)", family="NO", data=data)
    >>> model = gamlss("y ~ te(x1, x2, k=15)", family="NO", data=data)

    Notes
    -----
    te() 包含所有的主效应和交互效应。
    如果只需要纯交互效应，请使用 ti()。
    """
    if len(variables) < 2:
        raise ValueError("Tensor product requires at least 2 variables")

    return {
        "type": "te",
        "variables": list(variables),
        "k": k,
        "k_list": k_list,
        "bs": bs,
    }


def ti(
    *variables: str, k: int = 10, k_list: Optional[List[int]] = None, bs: str = "ps"
) -> Dict[str, Any]:
    """Tensor product interaction (ti)

    创建 tensor product interaction 规格，只包含**纯交互效应**，去除主效应。

    ti() 常与单变量平滑一起使用，实现可分解的交互模型：
        y ~ s(x1) + s(x2) + ti(x1, x2)
    这等价于（但参数更少、解释更清晰）：
        y ~ te(x1, x2)

    Parameters
    ----------
    *variables : str
        变量名列表（至少 2 个）
    k : int, default=10
        每个维度的基函数数量（如果 k_list 未指定）
    k_list : list of int, optional
        每个维度的基函数数量列表
    bs : str, default="ps"
        基函数类型

    Returns
    -------
    spec : dict
        Tensor product interaction 规格，包含 ``is_interaction_only: True`` 标记

    Examples
    --------
    >>> from omnilss import gamlss
    >>> # 分解为主效应和纯交互效应
    >>> model = gamlss("y ~ ti(x1) + ti(x2) + ti(x1, x2)", family="NO", data=data)
    >>> # 只加入交互项（主效应单独建模）
    >>> model = gamlss("y ~ x1 + x2 + ti(x1, x2)", family="NO", data=data)

    Notes
    -----
    ti() 的优势是可以分别控制主效应和交互效应的平滑程度。
    与 te() 的区别：te() 含主效应，ti() 不含主效应（投影去除）。
    """
    return {
        "type": "ti",
        "variables": list(variables),
        "k": k,
        "k_list": k_list,
        "bs": bs,
        "is_interaction_only": True,  # 标记：需要去除主效应（span(B1), span(B2) 等）
    }


def create_tensor_product_info(
    data: Dict[str, np.ndarray], spec: Dict[str, Any]
) -> TensorProductInfo:
    """从规格创建 TensorProductInfo

    Parameters
    ----------
    data : dict
        数据字典 {变量名: 值数组}
    spec : dict
        Tensor product 规格（来自 te() 或 ti()）

    Returns
    -------
    info : TensorProductInfo
        Tensor product 信息，包含 basis、penalty 及边际基

    Notes
    -----
    当 spec["type"] == "ti" 且 spec["is_interaction_only"] == True 时，
    自动调用 create_ti_basis() 去除主效应；否则调用 create_tensor_basis()。
    """
    variables = spec["variables"]
    k = spec["k"]
    k_list = spec.get("k_list")
    bs = spec.get("bs", "ps")

    # 提取变量数据
    X_vars = [data[var] for var in variables]

    # ── 根据类型选择 te 或 ti 基的创建函数 ──
    smooth_type = spec.get("type", "te")
    if smooth_type == "ti" and spec.get("is_interaction_only", False):
        # ti()：去除主效应，只保留纯交互部分
        basis, penalty = create_ti_basis(*X_vars, k=k, k_list=k_list, bs=bs)
    else:
        # te()：包含全部主效应和交互效应
        basis, penalty = create_tensor_basis(*X_vars, k=k, k_list=k_list, bs=bs)

    # ── 重新构建边际基（供后续信息记录使用）──
    from .bsplines import bspline_basis
    from .penalties import penalty_matrix

    k_list_for_marginal = k_list if k_list is not None else [k] * len(variables)

    marginal_bases = []
    marginal_penalties = []

    for X, k_i in zip(X_vars, k_list_for_marginal):
        x_min, x_max = np.min(X), np.max(X)
        x_range = x_max - x_min if x_max > x_min else 1.0
        n_interior = k_i - 4
        if n_interior > 0:
            interior_knots = np.linspace(x_min, x_max, n_interior + 2)[1:-1]
        else:
            interior_knots = np.array([])

        degree = 3
        knots = np.concatenate(
            [
                np.repeat(x_min - 0.01 * x_range, degree + 1),
                interior_knots,
                np.repeat(x_max + 0.01 * x_range, degree + 1),
            ]
        )

        import jax.numpy as _jnp

        B_i = np.array(
            bspline_basis(
                _jnp.array(X, dtype=_jnp.float64),
                _jnp.array(knots, dtype=_jnp.float64),
                degree=degree,
            )
        )
        actual_k_i = B_i.shape[1]
        P_i = penalty_matrix(actual_k_i, order=2)
        marginal_bases.append(B_i)
        marginal_penalties.append(P_i)

    return TensorProductInfo(
        variables=variables,
        marginal_bases=marginal_bases,
        marginal_penalties=marginal_penalties,
        tensor_basis=basis,
        tensor_penalty=penalty,
    )


def evaluate_tensor_smooth(
    info: TensorProductInfo,
    coefficients: np.ndarray,
) -> np.ndarray:
    """评估 tensor product smooth

    Parameters
    ----------
    info : TensorProductInfo
        Tensor product 信息
    coefficients : np.ndarray
        系数向量

    Returns
    -------
    fitted : np.ndarray
        拟合值（线性预测贡献）
    """
    return info.tensor_basis @ coefficients
