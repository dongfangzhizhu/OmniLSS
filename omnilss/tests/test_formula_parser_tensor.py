"""测试公式解析器的 tensor product smooth 功能

测试 te() 和 ti() 的解析和设计矩阵构建。
"""

import pytest
import numpy as np
from omnilss.formula_parser import (
    parse_formula,
    build_design_matrix,
    TensorProductTerm,
)


def test_parse_te_basic():
    """测试基本的 te() 解析"""
    formula = "y ~ te(x1, x2)"
    parsed = parse_formula(formula)
    
    assert parsed.response == "y"
    assert len(parsed.tensor_terms) == 1
    assert len(parsed.linear_terms) == 0
    assert len(parsed.smooth_terms) == 0
    
    term = parsed.tensor_terms[0]
    assert term.smoother == "te"
    assert term.variables == ["x1", "x2"]
    assert term.k is None  # Default will be used
    assert term.bs == "ps"


def test_parse_te_with_k():
    """测试带 k 参数的 te()"""
    formula = "y ~ te(x1, x2, k=15)"
    parsed = parse_formula(formula)
    
    term = parsed.tensor_terms[0]
    assert term.smoother == "te"
    assert term.variables == ["x1", "x2"]
    assert term.k == 15


def test_parse_te_with_k_list():
    """测试带 k_list 参数的 te()"""
    formula = "y ~ te(x1, x2, k_list=[5, 8])"
    parsed = parse_formula(formula)
    
    term = parsed.tensor_terms[0]
    assert term.smoother == "te"
    assert term.variables == ["x1", "x2"]
    assert term.k_list == [5, 8]


def test_parse_te_3d():
    """测试 3D te()"""
    formula = "y ~ te(x1, x2, x3)"
    parsed = parse_formula(formula)
    
    term = parsed.tensor_terms[0]
    assert term.smoother == "te"
    assert term.variables == ["x1", "x2", "x3"]


def test_parse_ti_basic():
    """测试基本的 ti() 解析"""
    formula = "y ~ ti(x1, x2)"
    parsed = parse_formula(formula)
    
    assert len(parsed.tensor_terms) == 1
    
    term = parsed.tensor_terms[0]
    assert term.smoother == "ti"
    assert term.variables == ["x1", "x2"]


def test_parse_ti_decomposition():
    """测试 ti() 分解
    
    注意：在我们的实现中，ti() 需要至少 2 个变量。
    在 R 中，ti(x1) 表示主效应，但我们简化为只支持交互效应。
    正确的用法是：s(x1) + s(x2) + ti(x1, x2)
    """
    # 正确的用法：使用 s() 表示主效应，ti() 表示交互
    formula = "y ~ s(x1) + s(x2) + ti(x1, x2)"
    parsed = parse_formula(formula)
    
    assert len(parsed.smooth_terms) == 2  # s(x1) 和 s(x2)
    assert len(parsed.tensor_terms) == 1  # ti(x1, x2)
    
    assert parsed.smooth_terms[0].variable == "x1"
    assert parsed.smooth_terms[1].variable == "x2"
    assert parsed.tensor_terms[0].smoother == "ti"
    assert parsed.tensor_terms[0].variables == ["x1", "x2"]


def test_parse_te_with_linear():
    """测试 te() 与线性项混合"""
    formula = "y ~ x1 + te(x2, x3)"
    parsed = parse_formula(formula)
    
    assert len(parsed.linear_terms) == 1
    assert len(parsed.tensor_terms) == 1
    assert parsed.linear_terms[0].variable == "x1"
    assert parsed.tensor_terms[0].variables == ["x2", "x3"]


def test_parse_te_with_smooth():
    """测试 te() 与 smooth 项混合"""
    formula = "y ~ s(x1) + te(x2, x3)"
    parsed = parse_formula(formula)
    
    assert len(parsed.smooth_terms) == 1
    assert len(parsed.tensor_terms) == 1
    assert parsed.smooth_terms[0].variable == "x1"
    assert parsed.tensor_terms[0].variables == ["x2", "x3"]


def test_parse_multiple_te():
    """测试多个 te() 项"""
    formula = "y ~ te(x1, x2) + te(x3, x4)"
    parsed = parse_formula(formula)
    
    assert len(parsed.tensor_terms) == 2
    assert parsed.tensor_terms[0].variables == ["x1", "x2"]
    assert parsed.tensor_terms[1].variables == ["x3", "x4"]


def test_parse_te_error_single_variable():
    """测试 te() 单变量错误"""
    formula = "y ~ te(x1)"
    
    with pytest.raises(ValueError, match="at least 2 variables"):
        parsed = parse_formula(formula)


def test_build_design_matrix_te():
    """测试构建包含 te() 的设计矩阵"""
    np.random.seed(42)
    n = 100
    
    data = {
        "y": np.random.randn(n),
        "x1": np.random.randn(n),
        "x2": np.random.randn(n),
    }
    
    formula = "y ~ te(x1, x2, k=5)"
    parsed = parse_formula(formula)
    
    X, smooth_info = build_design_matrix(parsed, data)
    
    # 检查设计矩阵维度
    # Intercept + tensor basis (5*5 = 25)
    assert X.shape == (n, 1 + 25)
    
    # 检查 smooth_info
    assert "tensor_0" in smooth_info
    tensor_info = smooth_info["tensor_0"]
    assert tensor_info["smoother"] == "te"
    assert tensor_info["variables"] == ["x1", "x2"]
    assert tensor_info["basis"].shape == (n, 25)
    assert tensor_info["penalty"].shape == (25, 25)


def test_build_design_matrix_te_3d():
    """测试构建 3D te() 的设计矩阵"""
    np.random.seed(42)
    n = 50
    
    data = {
        "y": np.random.randn(n),
        "x1": np.random.randn(n),
        "x2": np.random.randn(n),
        "x3": np.random.randn(n),
    }
    
    formula = "y ~ te(x1, x2, x3, k=4)"
    parsed = parse_formula(formula)
    
    X, smooth_info = build_design_matrix(parsed, data)
    
    # Intercept + tensor basis (4*4*4 = 64)
    assert X.shape == (n, 1 + 64)
    
    tensor_info = smooth_info["tensor_0"]
    assert tensor_info["basis"].shape == (n, 64)
    assert tensor_info["penalty"].shape == (64, 64)


def test_build_design_matrix_te_with_linear():
    """测试构建包含 te() 和线性项的设计矩阵"""
    np.random.seed(42)
    n = 100
    
    data = {
        "y": np.random.randn(n),
        "x1": np.random.randn(n),
        "x2": np.random.randn(n),
        "x3": np.random.randn(n),
    }
    
    formula = "y ~ x1 + te(x2, x3, k=5)"
    parsed = parse_formula(formula)
    
    X, smooth_info = build_design_matrix(parsed, data)
    
    # Intercept + x1 + tensor basis (5*5 = 25)
    assert X.shape == (n, 1 + 1 + 25)


def test_build_design_matrix_te_with_smooth():
    """测试构建包含 te() 和 smooth 项的设计矩阵"""
    np.random.seed(42)
    n = 100
    
    data = {
        "y": np.random.randn(n),
        "x1": np.random.randn(n),
        "x2": np.random.randn(n),
        "x3": np.random.randn(n),
    }
    
    formula = "y ~ s(x1) + te(x2, x3, k=5)"
    parsed = parse_formula(formula)
    
    X, smooth_info = build_design_matrix(parsed, data)
    
    # Intercept + s(x1) basis + tensor basis
    # s(x1) 默认会创建一些基函数，tensor 是 5*5=25
    assert X.shape[0] == n
    assert X.shape[1] > 1 + 25  # 至少有 intercept + tensor
    
    # 检查 smooth_info
    assert "smooth_0" in smooth_info
    assert "tensor_0" in smooth_info


def test_build_design_matrix_multiple_te():
    """测试构建包含多个 te() 的设计矩阵"""
    np.random.seed(42)
    n = 100
    
    data = {
        "y": np.random.randn(n),
        "x1": np.random.randn(n),
        "x2": np.random.randn(n),
        "x3": np.random.randn(n),
        "x4": np.random.randn(n),
    }
    
    formula = "y ~ te(x1, x2, k=5) + te(x3, x4, k=4)"
    parsed = parse_formula(formula)
    
    X, smooth_info = build_design_matrix(parsed, data)
    
    # Intercept + tensor1 (5*5=25) + tensor2 (4*4=16)
    assert X.shape == (n, 1 + 25 + 16)
    
    assert "tensor_0" in smooth_info
    assert "tensor_1" in smooth_info


def test_tensor_product_term_validation():
    """测试 TensorProductTerm 验证"""
    # 应该成功
    term = TensorProductTerm(
        smoother="te",
        variables=["x1", "x2"],
    )
    assert len(term.variables) == 2
    
    # 应该失败（少于 2 个变量）
    with pytest.raises(ValueError, match="at least 2 variables"):
        TensorProductTerm(
            smoother="te",
            variables=["x1"],
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
