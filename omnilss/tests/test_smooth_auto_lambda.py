"""测试平滑参数自动选择的集成（简化版）

验证 GCV/REML 自动选择与平滑拟合系统的集成。
"""

import pytest
import numpy as np
import jax.numpy as jnp
from omnilss.smooth_fitting import fit_penalized_wls, SmoothFitInfo


def create_smooth_fit(lambda_val=None):
    """创建测试用的 SmoothFitInfo"""
    penalty = jnp.eye(5)
    penalty = penalty.at[0, 0].set(0)  # 不惩罚截距
    
    return SmoothFitInfo(
        term_index=0,
        variable="x",
        smoother="pb",
        lambda_=lambda_val,
        edf=0.0,
        penalty=penalty,
        basis_columns=(0, 5)
    )


def test_auto_lambda_selection_gcv():
    """测试 GCV 自动选择"""
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + np.random.randn(n) * 0.1
    
    X = np.column_stack([np.ones(n), x, x**2, x**3, x**4])
    z = y
    w = np.ones(n)
    
    # Lambda 未指定
    smooth_fit = create_smooth_fit(lambda_val=None)
    
    # 拟合（自动选择）
    beta = fit_penalized_wls(
        X, z, w,
        smooth_fits=[smooth_fit],
        auto_lambda=True,
        lambda_method="GCV"
    )
    
    assert beta is not None
    assert len(beta) == 5
    assert np.all(np.isfinite(beta))
    
    # Lambda 应该被自动设置
    assert smooth_fit.lambda_ is not None
    assert smooth_fit.lambda_ > 0
    print(f"[OK] GCV auto-selected lambda: {smooth_fit.lambda_:.6e}")


def test_auto_lambda_selection_reml():
    """测试 REML 自动选择"""
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + np.random.randn(n) * 0.1
    
    X = np.column_stack([np.ones(n), x, x**2, x**3, x**4])
    z = y
    w = np.ones(n)
    
    smooth_fit = create_smooth_fit(lambda_val=None)
    
    beta = fit_penalized_wls(
        X, z, w,
        smooth_fits=[smooth_fit],
        auto_lambda=True,
        lambda_method="REML"
    )
    
    assert beta is not None
    assert smooth_fit.lambda_ is not None
    assert smooth_fit.lambda_ > 0
    print(f"[OK] REML auto-selected lambda: {smooth_fit.lambda_:.6e}")


def test_manual_lambda_preserved():
    """测试手动指定的 lambda 被保留"""
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + np.random.randn(n) * 0.1
    
    X = np.column_stack([np.ones(n), x, x**2, x**3, x**4])
    z = y
    w = np.ones(n)
    
    # 手动指定 lambda
    smooth_fit = create_smooth_fit(lambda_val=0.5)
    
    beta = fit_penalized_wls(
        X, z, w,
        smooth_fits=[smooth_fit],
        auto_lambda=True  # 即使启用，也不应该覆盖正值
    )
    
    assert beta is not None
    # Lambda 应该保持不变
    assert smooth_fit.lambda_ == 0.5
    print(f"[OK] Manual lambda preserved: {smooth_fit.lambda_}")


def test_auto_lambda_disabled():
    """测试禁用自动选择"""
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + np.random.randn(n) * 0.1
    
    X = np.column_stack([np.ones(n), x, x**2, x**3, x**4])
    z = y
    w = np.ones(n)
    
    smooth_fit = create_smooth_fit(lambda_val=0.1)
    
    beta = fit_penalized_wls(
        X, z, w,
        smooth_fits=[smooth_fit],
        auto_lambda=False  # 禁用
    )
    
    assert beta is not None
    assert smooth_fit.lambda_ == 0.1
    print(f"[OK] Auto lambda disabled, using manual: {smooth_fit.lambda_}")


def test_zero_lambda_triggers_auto():
    """测试 lambda=0 触发自动选择"""
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + np.random.randn(n) * 0.1
    
    X = np.column_stack([np.ones(n), x, x**2, x**3, x**4])
    z = y
    w = np.ones(n)
    
    # Lambda = 0 应该触发自动选择
    smooth_fit = create_smooth_fit(lambda_val=0.0)
    
    beta = fit_penalized_wls(
        X, z, w,
        smooth_fits=[smooth_fit],
        auto_lambda=True
    )
    
    assert beta is not None
    # Lambda 应该被自动设置为正值
    assert smooth_fit.lambda_ is not None
    assert smooth_fit.lambda_ > 0
    print(f"[OK] Zero lambda replaced with auto: {smooth_fit.lambda_:.6e}")


def test_negative_lambda_triggers_auto():
    """测试负 lambda 触发自动选择"""
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + np.random.randn(n) * 0.1
    
    X = np.column_stack([np.ones(n), x, x**2, x**3, x**4])
    z = y
    w = np.ones(n)
    
    # 负 lambda 应该触发自动选择
    smooth_fit = create_smooth_fit(lambda_val=-1.0)
    
    beta = fit_penalized_wls(
        X, z, w,
        smooth_fits=[smooth_fit],
        auto_lambda=True
    )
    
    assert beta is not None
    assert smooth_fit.lambda_ > 0
    print(f"[OK] Negative lambda replaced with auto: {smooth_fit.lambda_:.6e}")


def test_gcv_vs_reml_comparison():
    """对比 GCV 和 REML 的结果"""
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + np.random.randn(n) * 0.1
    
    X = np.column_stack([np.ones(n), x, x**2, x**3, x**4])
    z = y
    w = np.ones(n)
    
    # GCV
    smooth_fit_gcv = create_smooth_fit(lambda_val=None)
    fit_penalized_wls(X, z, w, [smooth_fit_gcv], auto_lambda=True, lambda_method="GCV")
    
    # REML
    smooth_fit_reml = create_smooth_fit(lambda_val=None)
    fit_penalized_wls(X, z, w, [smooth_fit_reml], auto_lambda=True, lambda_method="REML")
    
    print(f"[OK] GCV lambda:  {smooth_fit_gcv.lambda_:.6e}")
    print(f"[OK] REML lambda: {smooth_fit_reml.lambda_:.6e}")
    
    # 应该在相似的数量级
    log_ratio = abs(np.log10(smooth_fit_gcv.lambda_) - np.log10(smooth_fit_reml.lambda_))
    assert log_ratio < 2.0, "GCV and REML should give similar results"


def test_with_weights():
    """测试加权数据"""
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + np.random.randn(n) * 0.1
    w = np.random.uniform(0.5, 1.5, n)  # 非均匀权重
    
    X = np.column_stack([np.ones(n), x, x**2, x**3, x**4])
    z = y
    
    smooth_fit = create_smooth_fit(lambda_val=None)
    
    beta = fit_penalized_wls(X, z, w, [smooth_fit], auto_lambda=True)
    
    assert beta is not None
    assert smooth_fit.lambda_ > 0
    print(f"[OK] With weights, lambda: {smooth_fit.lambda_:.6e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
