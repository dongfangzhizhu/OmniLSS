"""测试 Tensor Product Smooth 功能

测试 tensor product smooth (te, ti) 的实现。
"""

import pytest
import numpy as np
from omnilss.smoothers.tensor_smooth import (
    create_tensor_basis,
    te,
    ti,
    create_tensor_product_info,
    evaluate_tensor_smooth
)


def test_create_tensor_basis_2d():
    """测试 2D tensor product basis 创建"""
    np.random.seed(42)
    n = 100
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    
    basis, penalty = create_tensor_basis(x1, x2, k=5)
    
    # 检查维度
    assert basis.shape == (n, 25)  # 5 * 5
    assert penalty.shape == (25, 25)
    
    # 检查惩罚矩阵对称性
    assert np.allclose(penalty, penalty.T)
    
    # 检查惩罚矩阵半正定
    eigenvalues = np.linalg.eigvalsh(penalty)
    assert np.all(eigenvalues >= -1e-10)


def test_create_tensor_basis_3d():
    """测试 3D tensor product basis 创建"""
    np.random.seed(42)
    n = 50
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    x3 = np.random.randn(n)
    
    basis, penalty = create_tensor_basis(x1, x2, x3, k=4)
    
    # 检查维度
    assert basis.shape == (n, 64)  # 4 * 4 * 4
    assert penalty.shape == (64, 64)
    
    # 检查惩罚矩阵对称性
    assert np.allclose(penalty, penalty.T)


def test_create_tensor_basis_different_k():
    """测试不同维度使用不同的 k"""
    np.random.seed(42)
    n = 100
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    
    basis, penalty = create_tensor_basis(x1, x2, k_list=[5, 8])
    
    # 检查维度
    assert basis.shape == (n, 40)  # 5 * 8
    assert penalty.shape == (40, 40)


def test_create_tensor_basis_error_single_variable():
    """测试单变量错误"""
    x = np.random.randn(100)
    
    with pytest.raises(ValueError, match="at least 2 variables"):
        create_tensor_basis(x, k=5)


def test_create_tensor_basis_error_different_lengths():
    """测试不同长度变量错误"""
    x1 = np.random.randn(100)
    x2 = np.random.randn(50)
    
    with pytest.raises(ValueError, match="same length"):
        create_tensor_basis(x1, x2, k=5)


def test_te_specification():
    """测试 te() 规格"""
    spec = te("x1", "x2", k=10)
    
    assert spec["type"] == "te"
    assert spec["variables"] == ["x1", "x2"]
    assert spec["k"] == 10
    assert spec["bs"] == "ps"


def test_te_specification_3d():
    """测试 3D te() 规格"""
    spec = te("x1", "x2", "x3", k=8)
    
    assert spec["type"] == "te"
    assert spec["variables"] == ["x1", "x2", "x3"]
    assert spec["k"] == 8


def test_te_specification_error():
    """测试 te() 单变量错误"""
    with pytest.raises(ValueError, match="at least 2 variables"):
        te("x1", k=10)


def test_ti_specification():
    """测试 ti() 规格"""
    spec = ti("x1", "x2", k=10)
    
    assert spec["type"] == "ti"
    assert spec["variables"] == ["x1", "x2"]
    assert spec["k"] == 10


def test_create_tensor_product_info():
    """测试创建 TensorProductInfo"""
    np.random.seed(42)
    n = 100
    data = {
        "x1": np.random.randn(n),
        "x2": np.random.randn(n)
    }
    
    spec = te("x1", "x2", k=5)
    info = create_tensor_product_info(data, spec)
    
    # 检查属性
    assert info.variables == ["x1", "x2"]
    assert len(info.marginal_bases) == 2
    assert len(info.marginal_penalties) == 2
    assert info.tensor_basis.shape == (n, 25)
    assert info.tensor_penalty.shape == (25, 25)


def test_evaluate_tensor_smooth():
    """测试评估 tensor smooth"""
    np.random.seed(42)
    n = 100
    data = {
        "x1": np.random.randn(n),
        "x2": np.random.randn(n)
    }
    
    spec = te("x1", "x2", k=5)
    info = create_tensor_product_info(data, spec)
    
    # 随机系数
    coefficients = np.random.randn(25)
    
    # 评估
    fitted = evaluate_tensor_smooth(info, coefficients)
    
    # 检查维度
    assert fitted.shape == (n,)
    
    # 检查与手动计算一致
    expected = info.tensor_basis @ coefficients
    assert np.allclose(fitted, expected)


def test_tensor_smooth_fitting_simple():
    """测试简单的 tensor smooth 拟合"""
    np.random.seed(42)
    n = 200
    x1 = np.random.uniform(-2, 2, n)
    x2 = np.random.uniform(-2, 2, n)
    
    # 真实函数: f(x1, x2) = x1^2 + x2^2 + x1*x2
    y_true = x1**2 + x2**2 + x1*x2
    y = y_true + np.random.randn(n) * 0.1
    
    # 创建 tensor product basis
    basis, penalty = create_tensor_basis(x1, x2, k=10)
    
    # 简单的岭回归拟合
    lambda_ = 0.1
    XtX = basis.T @ basis + lambda_ * penalty
    Xty = basis.T @ y
    coefficients = np.linalg.solve(XtX, Xty)
    
    # 预测
    y_pred = basis @ coefficients
    
    # 检查拟合质量（R²）
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    
    # R² 应该比较高
    assert r2 > 0.9, f"R² = {r2:.3f} is too low"


def test_tensor_smooth_vs_additive():
    """测试 tensor smooth 与加性模型的对比"""
    np.random.seed(42)
    n = 200
    x1 = np.random.uniform(-2, 2, n)
    x2 = np.random.uniform(-2, 2, n)
    
    # 有交互效应的真实函数
    y_true = x1**2 + x2**2 + 2*x1*x2
    y = y_true + np.random.randn(n) * 0.2
    
    # Tensor product smooth
    basis_tensor, penalty_tensor = create_tensor_basis(x1, x2, k=8)
    lambda_ = 0.1
    XtX = basis_tensor.T @ basis_tensor + lambda_ * penalty_tensor
    Xty = basis_tensor.T @ y
    coef_tensor = np.linalg.solve(XtX, Xty)
    y_pred_tensor = basis_tensor @ coef_tensor
    
    # 加性模型（分别对 x1 和 x2 平滑）
    from omnilss.smoothers.bsplines import bspline_basis
    from omnilss.smoothers.penalties import penalty_matrix
    
    # 为 x1 创建基
    x1_min, x1_max = np.min(x1), np.max(x1)
    x1_range = x1_max - x1_min
    n_interior1 = 8 - 4
    interior_knots1 = np.linspace(x1_min, x1_max, n_interior1 + 2)[1:-1]
    knots1 = np.concatenate([
        np.repeat(x1_min - 0.01 * x1_range, 4),
        interior_knots1,
        np.repeat(x1_max + 0.01 * x1_range, 4)
    ])
    basis1 = np.array(bspline_basis(np.array(x1), np.array(knots1), degree=3))
    actual_k1 = basis1.shape[1]
    penalty1 = penalty_matrix(actual_k1, order=2)
    
    # 为 x2 创建基
    x2_min, x2_max = np.min(x2), np.max(x2)
    x2_range = x2_max - x2_min
    n_interior2 = 8 - 4
    interior_knots2 = np.linspace(x2_min, x2_max, n_interior2 + 2)[1:-1]
    knots2 = np.concatenate([
        np.repeat(x2_min - 0.01 * x2_range, 4),
        interior_knots2,
        np.repeat(x2_max + 0.01 * x2_range, 4)
    ])
    basis2 = np.array(bspline_basis(np.array(x2), np.array(knots2), degree=3))
    actual_k2 = basis2.shape[1]
    penalty2 = penalty_matrix(actual_k2, order=2)
    
    basis_additive = np.hstack([basis1, basis2])
    penalty_additive = np.block([
        [penalty1, np.zeros((actual_k1, actual_k2))],
        [np.zeros((actual_k2, actual_k1)), penalty2]
    ])
    
    XtX_add = basis_additive.T @ basis_additive + lambda_ * penalty_additive
    Xty_add = basis_additive.T @ y
    coef_add = np.linalg.solve(XtX_add, Xty_add)
    y_pred_add = basis_additive @ coef_add
    
    # 计算 R²
    r2_tensor = 1 - np.sum((y - y_pred_tensor) ** 2) / np.sum((y - np.mean(y)) ** 2)
    r2_additive = 1 - np.sum((y - y_pred_add) ** 2) / np.sum((y - np.mean(y)) ** 2)
    
    # Tensor product 应该更好（因为有交互效应）
    assert r2_tensor > r2_additive, f"Tensor R²={r2_tensor:.3f} should be > Additive R²={r2_additive:.3f}"
    assert r2_tensor > 0.85


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
