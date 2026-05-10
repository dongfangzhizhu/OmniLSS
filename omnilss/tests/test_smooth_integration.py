"""测试平滑参数自动选择的集成

验证 GCV/REML 自动选择与平滑拟合系统的集成。
"""

import pytest
import numpy as np
from omnilss.smooth_fitting import (
    fit_penalized_wls,
    SmoothFitInfo,
    build_smooth_design
)


@pytest.fixture
def simple_smooth_data():
    """生成简单的平滑数据"""
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 1, n)
    # 真实函数: sin(2πx)
    y = np.sin(2 * np.pi * x) + np.random.randn(n) * 0.1
    return {"x": x, "y": y}


def test_fit_penalized_wls_auto_lambda():
    """测试自动 lambda 选择"""
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + np.random.randn(n) * 0.1
    
    # 构建设计矩阵
    X = np.column_stack([np.ones(n), x, x**2, x**3, x**4])
    z = y  # 工作响应
    w = np.ones(n)
    
    # 创建平滑项（lambda 未指定）
    import jax.numpy as jnp
    penalty = jnp.eye(5)
    penalty = penalty.at[0, 0].set(0)  # 不惩罚截距
    
    smooth_fit = SmoothFitInfo(
        term_index=0,
        variable="x",
        smoother="pb",
        basis_columns=(0, 5),
        penalty=penalty,
        lambda_=None,  # 未指定，应该自动选择
        edf=0.0
    )
    
    # 拟合（自动选择 lambda）
    beta = fit_penalized_wls(
        X, z, w,
        smooth_fits=[smooth_fit],
        auto_lambda=True,
        lambda_method="GCV"
    )
    
    # 检查结果
    assert beta is not None
    assert len(beta) == 5
    assert np.all(np.isfinite(beta))
    
    # 检查 lambda 已被设置
    assert smooth_fit.lambda_ is not None
    assert smooth_fit.lambda_ > 0
    print(f"Auto-selected lambda: {smooth_fit.lambda_:.6e}")


def test_fit_penalized_wls_manual_lambda():
    """测试手动指定 lambda（向后兼容）"""
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + np.random.randn(n) * 0.1
    
    X = np.column_stack([np.ones(n), x, x**2, x**3, x**4])
    z = y
    w = np.ones(n)
    
    penalty = np.eye(5)
    penalty[0, 0] = 0
    
    # 手动指定 lambda
    smooth_fit = SmoothFitInfo(
        basis_columns=(0, 5),
        penalty=penalty,
        lambda_=0.1,  # 手动指定
        edf=None
    )
    
    # 拟合（不自动选择）
    beta = fit_penalized_wls(
        X, z, w,
        smooth_fits=[smooth_fit],
        auto_lambda=False  # 禁用自动选择
    )
    
    assert beta is not None
    assert len(beta) == 5
    
    # lambda 应该保持不变
    assert smooth_fit.lambda_ == 0.1


def test_fit_penalized_wls_gcv_vs_reml():
    """对比 GCV 和 REML 方法"""
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + np.random.randn(n) * 0.1
    
    X = np.column_stack([np.ones(n), x, x**2, x**3, x**4])
    z = y
    w = np.ones(n)
    
    penalty = np.eye(5)
    penalty[0, 0] = 0
    
    # GCV
    smooth_fit_gcv = SmoothFitInfo(
        basis_columns=(0, 5),
        penalty=penalty,
        lambda_=None,
        edf=None
    )
    
    beta_gcv = fit_penalized_wls(
        X, z, w,
        smooth_fits=[smooth_fit_gcv],
        auto_lambda=True,
        lambda_method="GCV"
    )
    
    # REML
    smooth_fit_reml = SmoothFitInfo(
        basis_columns=(0, 5),
        penalty=penalty,
        lambda_=None,
        edf=None
    )
    
    beta_reml = fit_penalized_wls(
        X, z, w,
        smooth_fits=[smooth_fit_reml],
        auto_lambda=True,
        lambda_method="REML"
    )
    
    # 两者应该给出相似的结果
    assert smooth_fit_gcv.lambda_ > 0
    assert smooth_fit_reml.lambda_ > 0
    
    print(f"GCV lambda: {smooth_fit_gcv.lambda_:.6e}")
    print(f"REML lambda: {smooth_fit_reml.lambda_:.6e}")
    
    # Lambda 应该在相似的数量级
    log_ratio = abs(np.log10(smooth_fit_gcv.lambda_) - np.log10(smooth_fit_reml.lambda_))
    assert log_ratio < 2.0, "GCV and REML should give similar lambdas"


def test_multiple_smooth_terms():
    """测试多个平滑项"""
    np.random.seed(42)
    n = 100
    x1 = np.linspace(0, 1, n)
    x2 = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x1) + np.cos(2 * np.pi * x2) + np.random.randn(n) * 0.1
    
    # 构建设计矩阵：截距 + x1 多项式 + x2 多项式
    X = np.column_stack([
        np.ones(n),      # 截距
        x1, x1**2, x1**3,  # x1 平滑项
        x2, x2**2, x2**3   # x2 平滑项
    ])
    z = y
    w = np.ones(n)
    
    # 第一个平滑项（x1）
    penalty1 = np.eye(3)
    smooth_fit1 = SmoothFitInfo(
        basis_columns=(1, 4),  # 列 1-3
        penalty=penalty1,
        lambda_=None,
        edf=None
    )
    
    # 第二个平滑项（x2）
    penalty2 = np.eye(3)
    smooth_fit2 = SmoothFitInfo(
        basis_columns=(4, 7),  # 列 4-6
        penalty=penalty2,
        lambda_=None,
        edf=None
    )
    
    # 拟合（两个平滑项都自动选择 lambda）
    beta = fit_penalized_wls(
        X, z, w,
        smooth_fits=[smooth_fit1, smooth_fit2],
        auto_lambda=True,
        lambda_method="GCV"
    )
    
    assert beta is not None
    assert len(beta) == 7
    
    # 两个 lambda 都应该被设置
    assert smooth_fit1.lambda_ is not None
    assert smooth_fit1.lambda_ > 0
    assert smooth_fit2.lambda_ is not None
    assert smooth_fit2.lambda_ > 0
    
    print(f"Lambda 1: {smooth_fit1.lambda_:.6e}")
    print(f"Lambda 2: {smooth_fit2.lambda_:.6e}")


def test_auto_lambda_with_weights():
    """测试加权数据的自动 lambda 选择"""
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + np.random.randn(n) * 0.1
    
    # 非均匀权重
    w = np.random.uniform(0.5, 1.5, n)
    
    X = np.column_stack([np.ones(n), x, x**2, x**3])
    z = y
    
    penalty = np.eye(4)
    penalty[0, 0] = 0
    
    smooth_fit = SmoothFitInfo(
        basis_columns=(0, 4),
        penalty=penalty,
        lambda_=None,
        edf=None
    )
    
    beta = fit_penalized_wls(
        X, z, w,
        smooth_fits=[smooth_fit],
        auto_lambda=True
    )
    
    assert beta is not None
    assert smooth_fit.lambda_ > 0


def test_auto_lambda_fallback():
    """测试自动选择失败时的回退机制"""
    np.random.seed(42)
    n = 10  # 非常小的样本
    x = np.linspace(0, 1, n)
    y = np.random.randn(n)
    
    # 构建可能导致数值问题的设计矩阵
    X = np.column_stack([np.ones(n), x, x**2, x**3, x**4, x**5])
    z = y
    w = np.ones(n)
    
    # 病态的惩罚矩阵
    penalty = np.zeros((6, 6))
    
    smooth_fit = SmoothFitInfo(
        basis_columns=(0, 6),
        penalty=penalty,
        lambda_=None,
        edf=None
    )
    
    # 应该能处理而不崩溃（使用默认 lambda=1.0）
    with pytest.warns(UserWarning, match="Automatic lambda selection failed"):
        beta = fit_penalized_wls(
            X, z, w,
            smooth_fits=[smooth_fit],
            auto_lambda=True
        )
    
    assert beta is not None
    # 应该使用默认值
    assert smooth_fit.lambda_ == 1.0


def test_backward_compatibility():
    """测试向后兼容性"""
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + np.random.randn(n) * 0.1
    
    X = np.column_stack([np.ones(n), x, x**2, x**3])
    z = y
    w = np.ones(n)
    
    penalty = np.eye(4)
    penalty[0, 0] = 0
    
    smooth_fit = SmoothFitInfo(
        basis_columns=(0, 4),
        penalty=penalty,
        lambda_=0.5,  # 手动指定
        edf=None
    )
    
    # 旧的调用方式（没有 auto_lambda 参数）应该仍然工作
    beta = fit_penalized_wls(X, z, w, smooth_fits=[smooth_fit])
    
    assert beta is not None
    # 默认 auto_lambda=True，但因为 lambda > 0，不会重新选择
    assert smooth_fit.lambda_ == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
