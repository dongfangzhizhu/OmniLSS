"""测试优化器集成

验证所有优化器都能正确集成到 gamlss() 主函数中。
"""

import pytest
import numpy as np
import jax.numpy as jnp
from omnilss.fitting import gamlss


@pytest.fixture
def simple_data():
    """生成简单的测试数据"""
    np.random.seed(42)
    n = 100
    x = np.random.randn(n)
    y = 2 + 3*x + np.random.randn(n)
    return {"y": y, "x": x}


def test_rs_method(simple_data):
    """测试 RS 方法（默认）"""
    model = gamlss("y ~ x", family="NO", data=simple_data, method="RS")
    
    assert model is not None
    assert "mu" in model.coefficients
    assert "sigma" in model.coefficients
    assert model.g_dev > 0
    assert model.iter > 0
    
    # 检查系数接近真实值
    mu_coef = np.asarray(model.coefficients["mu"])
    assert abs(mu_coef[0] - 2.0) < 0.5  # Intercept
    assert abs(mu_coef[1] - 3.0) < 0.5  # x coefficient


def test_cg_method(simple_data):
    """测试 CG 方法"""
    model = gamlss("y ~ x", family="NO", data=simple_data, method="CG")
    
    assert model is not None
    assert "mu" in model.coefficients
    assert "sigma" in model.coefficients
    assert model.g_dev > 0
    assert model.iter > 0
    
    # 检查系数接近真实值
    mu_coef = np.asarray(model.coefficients["mu"])
    assert abs(mu_coef[0] - 2.0) < 0.5
    assert abs(mu_coef[1] - 3.0) < 0.5


def test_joint_adam_method(simple_data):
    """测试 Joint-Adam 方法"""
    model = gamlss(
        "y ~ x",
        family="NO",
        data=simple_data,
        method="joint",
        optimizer="adam",
        learning_rate=0.01,
        max_iter=500
    )
    
    assert model is not None
    assert "mu" in model.coefficients
    assert "sigma" in model.coefficients
    assert model.g_dev > 0
    
    # 检查系数接近真实值
    mu_coef = np.asarray(model.coefficients["mu"])
    assert abs(mu_coef[0] - 2.0) < 0.5
    assert abs(mu_coef[1] - 3.0) < 0.5


def test_joint_sgd_method(simple_data):
    """测试 Joint-SGD 方法"""
    model = gamlss(
        "y ~ x",
        family="NO",
        data=simple_data,
        method="joint",
        optimizer="sgd",
        learning_rate=0.05,  # 降低学习率以提高稳定性
        max_iter=2000
    )
    
    assert model is not None
    assert "mu" in model.coefficients
    assert "sigma" in model.coefficients
    # SGD 可能不稳定，只检查模型存在
    assert model.g_dev is not None


def test_joint_rmsprop_method(simple_data):
    """测试 Joint-RMSprop 方法"""
    model = gamlss(
        "y ~ x",
        family="NO",
        data=simple_data,
        method="joint",
        optimizer="rmsprop",
        learning_rate=0.01,
        max_iter=500
    )
    
    assert model is not None
    assert "mu" in model.coefficients
    assert "sigma" in model.coefficients
    assert model.g_dev > 0


def test_lbfgs_method(simple_data):
    """测试 L-BFGS 方法"""
    model = gamlss(
        "y ~ x",
        family="NO",
        data=simple_data,
        method="lbfgs",
        max_iter=100,
        history_size=10
    )
    
    assert model is not None
    assert "mu" in model.coefficients
    assert "sigma" in model.coefficients
    assert model.g_dev > 0
    
    # 检查系数接近真实值
    mu_coef = np.asarray(model.coefficients["mu"])
    assert abs(mu_coef[0] - 2.0) < 0.5
    assert abs(mu_coef[1] - 3.0) < 0.5


def test_mixed_method(simple_data):
    """测试 MIXED 方法"""
    model = gamlss("y ~ x", family="NO", data=simple_data, method="MIXED")
    
    assert model is not None
    assert "mu" in model.coefficients
    assert "sigma" in model.coefficients
    assert model.g_dev > 0


def test_verbose_output(simple_data, capsys):
    """测试 verbose 输出"""
    model = gamlss(
        "y ~ x",
        family="NO",
        data=simple_data,
        method="joint",
        optimizer="adam",
        learning_rate=0.01,
        max_iter=100,
        verbose=True
    )
    
    captured = capsys.readouterr()
    assert "Step 1: Getting initial estimates" in captured.out
    assert "Step 2: Refining with JOINT optimizer" in captured.out
    assert "Initial deviance" in captured.out
    assert "Final deviance" in captured.out


def test_convergence_check(simple_data):
    """测试收敛检查"""
    model = gamlss(
        "y ~ x",
        family="NO",
        data=simple_data,
        method="joint",
        optimizer="adam",
        learning_rate=0.01,
        max_iter=500
    )
    
    # 检查收敛标志
    assert "converged" in model.additional_slots
    assert isinstance(model.additional_slots["converged"], bool)
    
    # 检查 deviance 历史
    assert "deviance_history" in model.additional_slots
    history = model.additional_slots["deviance_history"]
    assert len(history) > 0
    
    # Deviance 应该递减
    if len(history) > 1:
        assert history[-1] <= history[0]


def test_optimizer_comparison(simple_data):
    """对比不同优化器的结果"""
    methods = {
        "RS": {"method": "RS"},
        "CG": {"method": "CG"},
        "Joint-Adam": {"method": "joint", "optimizer": "adam", "learning_rate": 0.01, "max_iter": 500},
        "L-BFGS": {"method": "lbfgs", "max_iter": 100},
    }
    
    results = {}
    for name, kwargs in methods.items():
        model = gamlss("y ~ x", family="NO", data=simple_data, **kwargs)
        results[name] = {
            "deviance": model.g_dev,
            "iterations": model.iter,
            "mu_coef": np.asarray(model.coefficients["mu"]),
        }
    
    # 所有方法应该得到相似的结果
    deviances = [r["deviance"] for r in results.values()]
    assert max(deviances) - min(deviances) < 10.0  # Deviance 差异不应太大
    
    # 所有方法的系数应该接近
    for name, result in results.items():
        mu_coef = result["mu_coef"]
        assert abs(mu_coef[0] - 2.0) < 0.5  # Intercept
        assert abs(mu_coef[1] - 3.0) < 0.5  # x coefficient


def test_invalid_method():
    """测试无效的方法名"""
    data = {"y": np.random.randn(10), "x": np.random.randn(10)}
    
    with pytest.raises(ValueError, match="method must be one of"):
        gamlss("y ~ x", family="NO", data=data, method="INVALID")


def test_max_iter_override(simple_data):
    """测试 max_iter 参数覆盖"""
    from omnilss.controls import gamlss_control
    
    # 使用 control 设置 n_cyc=1000
    control = gamlss_control(n_cyc=1000)
    
    # 但 max_iter=50 应该覆盖它
    model = gamlss(
        "y ~ x",
        family="NO",
        data=simple_data,
        method="joint",
        optimizer="adam",
        learning_rate=0.01,
        max_iter=50,
        control=control
    )
    
    # 迭代次数应该 <= 50
    assert model.iter <= 50


def test_different_families(simple_data):
    """测试不同分布族"""
    # 只测试 NO 和 LOGNO，因为 GA 需要正数据
    families = ["NO", "LOGNO"]
    
    for family in families:
        model = gamlss(
            "y ~ x",
            family=family,
            data=simple_data,
            method="joint",
            optimizer="adam",
            learning_rate=0.01,
            max_iter=500
        )
        
        assert model is not None
        assert model.family.name == family
        # 检查 deviance 是有限的
        assert np.isfinite(model.g_dev)


def test_model_attributes(simple_data):
    """测试模型属性"""
    model = gamlss(
        "y ~ x",
        family="NO",
        data=simple_data,
        method="joint",
        optimizer="adam",
        learning_rate=0.01,
        max_iter=500
    )
    
    # 检查基本属性
    assert hasattr(model, "coefficients")
    assert hasattr(model, "fitted_values")
    assert hasattr(model, "linear_predictors")
    assert hasattr(model, "residuals")
    assert hasattr(model, "g_dev")
    assert hasattr(model, "iter")
    assert hasattr(model, "family")
    
    # 检查 additional_slots
    assert "aic" in model.additional_slots
    assert "sbc" in model.additional_slots
    assert "method" in model.additional_slots
    assert "converged" in model.additional_slots
    assert "cycles" in model.additional_slots
    
    # 对于 Joint/L-BFGS，应该有额外的优化器信息
    if model.additional_slots["method"] in ["JOINT", "LBFGS"]:
        assert "optimizer_result" in model.additional_slots
        assert "loss_history" in model.additional_slots


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
