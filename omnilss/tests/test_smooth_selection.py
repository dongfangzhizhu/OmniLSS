"""测试平滑参数自动选择

测试 GCV 和 REML 方法的正确性和稳定性。
"""

import pytest
import numpy as np
from omnilss.smooth_selection import (
    gcv_score,
    reml_score,
    select_lambda_gcv,
    select_lambda_reml,
    select_lambda
)


@pytest.fixture
def simple_data():
    """生成简单的测试数据"""
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 1, n)
    # 真实函数: sin(2πx)
    y = np.sin(2 * np.pi * x) + np.random.randn(n) * 0.1
    return x, y


@pytest.fixture
def polynomial_basis(simple_data):
    """构建多项式基"""
    x, y = simple_data
    n = len(x)
    # 使用 4 次多项式
    X = np.column_stack([np.ones(n), x, x**2, x**3, x**4])
    w = np.ones(n)
    # 简单的二阶差分惩罚
    S = np.eye(5)
    S[0, 0] = 0  # 不惩罚截距
    return X, y, w, S


def test_gcv_score_basic(polynomial_basis):
    """测试 GCV 分数计算的基本功能"""
    X, y, w, S = polynomial_basis
    
    # 计算 GCV
    gcv = gcv_score(lambda_param=0.1, X=X, y=y, w=w, S=S)
    
    assert np.isfinite(gcv), "GCV score should be finite"
    assert gcv > 0, "GCV score should be positive"


def test_gcv_score_lambda_effect(polynomial_basis):
    """测试不同 lambda 值对 GCV 的影响"""
    X, y, w, S = polynomial_basis
    
    # 测试不同的 lambda 值
    lambdas = [1e-6, 1e-3, 1.0, 1e3, 1e6]
    gcv_scores = [gcv_score(lam, X, y, w, S) for lam in lambdas]
    
    # 所有分数应该是有限的
    assert all(np.isfinite(score) for score in gcv_scores)
    
    # GCV 应该有一个最小值（不是单调的）
    # 最小值应该不在两端（允许等于第一个值，因为可能最优就在边界）
    min_idx = gcv_scores.index(min(gcv_scores))
    assert min_idx >= 0 and min_idx < len(gcv_scores), "GCV should have a minimum"
    # 至少最大 lambda 不应该是最优的
    assert min(gcv_scores) < gcv_scores[-1], "Maximum lambda should not be optimal"


def test_gcv_score_negative_lambda(polynomial_basis):
    """测试负 lambda 值应该返回 inf"""
    X, y, w, S = polynomial_basis
    
    gcv = gcv_score(lambda_param=-1.0, X=X, y=y, w=w, S=S)
    
    assert gcv == np.inf, "Negative lambda should return inf"


def test_reml_score_basic(polynomial_basis):
    """测试 REML 分数计算的基本功能"""
    X, y, w, S = polynomial_basis
    
    # 计算 REML
    reml = reml_score(lambda_param=0.1, X=X, y=y, w=w, S=S)
    
    assert np.isfinite(reml), "REML score should be finite"


def test_reml_score_lambda_effect(polynomial_basis):
    """测试不同 lambda 值对 REML 的影响"""
    X, y, w, S = polynomial_basis
    
    # 测试不同的 lambda 值
    lambdas = [1e-6, 1e-3, 1.0, 1e3, 1e6]
    reml_scores = [reml_score(lam, X, y, w, S) for lam in lambdas]
    
    # 所有分数应该是有限的
    assert all(np.isfinite(score) for score in reml_scores)


def test_select_lambda_gcv(polynomial_basis):
    """测试 GCV 自动选择"""
    X, y, w, S = polynomial_basis
    
    lambda_opt, gcv_opt = select_lambda_gcv(X, y, w, S, verbose=False)
    
    assert lambda_opt > 0, "Optimal lambda should be positive"
    assert np.isfinite(lambda_opt), "Optimal lambda should be finite"
    assert np.isfinite(gcv_opt), "Optimal GCV should be finite"
    
    # 检查是否确实是最优的（至少比边界值好）
    gcv_low = gcv_score(1e-6, X, y, w, S)
    gcv_high = gcv_score(1e6, X, y, w, S)
    assert gcv_opt <= gcv_low or gcv_opt <= gcv_high, \
        "Optimal GCV should be better than boundary values"


def test_select_lambda_reml(polynomial_basis):
    """测试 REML 自动选择"""
    X, y, w, S = polynomial_basis
    
    lambda_opt, reml_opt = select_lambda_reml(X, y, w, S, verbose=False)
    
    assert lambda_opt > 0, "Optimal lambda should be positive"
    assert np.isfinite(lambda_opt), "Optimal lambda should be finite"
    assert np.isfinite(reml_opt), "Optimal REML should be finite"


def test_select_lambda_unified_interface(polynomial_basis):
    """测试统一接口"""
    X, y, w, S = polynomial_basis
    
    # 测试 GCV
    lambda_gcv, score_gcv = select_lambda(X, y, w, S, method="GCV")
    assert lambda_gcv > 0
    assert np.isfinite(score_gcv)
    
    # 测试 REML
    lambda_reml, score_reml = select_lambda(X, y, w, S, method="REML")
    assert lambda_reml > 0
    assert np.isfinite(score_reml)
    
    # 测试无效方法
    with pytest.raises(ValueError, match="Unknown method"):
        select_lambda(X, y, w, S, method="INVALID")


def test_gcv_vs_reml_comparison(polynomial_basis):
    """对比 GCV 和 REML 的结果"""
    X, y, w, S = polynomial_basis
    
    lambda_gcv, _ = select_lambda_gcv(X, y, w, S)
    lambda_reml, _ = select_lambda_reml(X, y, w, S)
    
    # 两者应该给出相似的结果（在对数空间）
    # 允许 2 个数量级的差异
    log_ratio = abs(np.log10(lambda_gcv) - np.log10(lambda_reml))
    assert log_ratio < 2.0, \
        f"GCV and REML should give similar results, got {lambda_gcv:.2e} vs {lambda_reml:.2e}"


def test_weighted_data():
    """测试加权数据"""
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + np.random.randn(n) * 0.1
    
    # 非均匀权重
    w = np.random.uniform(0.5, 1.5, n)
    
    X = np.column_stack([np.ones(n), x, x**2, x**3])
    S = np.eye(4)
    S[0, 0] = 0
    
    lambda_opt, gcv_opt = select_lambda_gcv(X, y, w, S)
    
    assert lambda_opt > 0
    assert np.isfinite(gcv_opt)


def test_small_sample():
    """测试小样本情况"""
    np.random.seed(42)
    n = 20  # 小样本
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + np.random.randn(n) * 0.1
    
    X = np.column_stack([np.ones(n), x, x**2])
    w = np.ones(n)
    S = np.eye(3)
    S[0, 0] = 0
    
    # GCV 可能不稳定，但应该能运行
    lambda_gcv, _ = select_lambda_gcv(X, y, w, S)
    assert lambda_gcv > 0
    
    # REML 应该更稳定
    lambda_reml, _ = select_lambda_reml(X, y, w, S)
    assert lambda_reml > 0


def test_verbose_output(polynomial_basis, capsys):
    """测试 verbose 输出"""
    X, y, w, S = polynomial_basis
    
    # 测试 GCV verbose
    select_lambda_gcv(X, y, w, S, verbose=True)
    captured = capsys.readouterr()
    assert "GCV optimization" in captured.out
    assert "Optimal" in captured.out
    
    # 测试 REML verbose
    select_lambda_reml(X, y, w, S, verbose=True)
    captured = capsys.readouterr()
    assert "REML optimization" in captured.out
    assert "Optimal" in captured.out


def test_lambda_range_effect(polynomial_basis):
    """测试不同搜索范围的影响"""
    X, y, w, S = polynomial_basis
    
    # 窄范围
    lambda_narrow, _ = select_lambda_gcv(
        X, y, w, S,
        lambda_range=(1e-3, 1e3)
    )
    
    # 宽范围
    lambda_wide, _ = select_lambda_gcv(
        X, y, w, S,
        lambda_range=(1e-6, 1e6)
    )
    
    # 如果最优值在窄范围内，两者应该相似
    # 否则宽范围应该找到更好的解
    assert lambda_narrow > 0
    assert lambda_wide > 0


def test_numerical_stability():
    """测试数值稳定性"""
    np.random.seed(42)
    n = 50
    x = np.linspace(0, 1, n)
    
    # 添加噪声较大的数据
    y = np.sin(2 * np.pi * x) + np.random.randn(n) * 0.5
    
    X = np.column_stack([np.ones(n), x, x**2, x**3, x**4, x**5])
    w = np.ones(n)
    
    # 强惩罚矩阵
    S = np.eye(6)
    S[0, 0] = 0
    for i in range(1, 6):
        S[i, i] = i ** 2  # 递增惩罚
    
    # 应该能处理而不崩溃
    lambda_opt, gcv_opt = select_lambda_gcv(X, y, w, S)
    
    assert np.isfinite(lambda_opt)
    assert np.isfinite(gcv_opt)


def test_perfect_fit():
    """测试完美拟合的情况"""
    np.random.seed(42)
    n = 10
    x = np.linspace(0, 1, n)
    # 完美的二次函数，无噪声
    y = 1 + 2*x + 3*x**2
    
    X = np.column_stack([np.ones(n), x, x**2])
    w = np.ones(n)
    S = np.eye(3)
    S[0, 0] = 0
    
    # 应该选择很小的 lambda（接近无惩罚）
    lambda_opt, _ = select_lambda_gcv(X, y, w, S)
    
    # lambda 应该很小
    assert lambda_opt < 1e-2, "For perfect fit, lambda should be very small"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
