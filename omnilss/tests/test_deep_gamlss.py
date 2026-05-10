"""测试 Deep GAMLSS 功能

测试神经网络建模分布参数的功能。
"""

import pytest
import numpy as np
import jax.numpy as jnp
from omnilss.deep import (
    ParameterNetwork,
    DeepGAMLSS,
    fit_deep_gamlss,
    predict_deep_gamlss
)
from omnilss import NO, GA


def test_parameter_network_creation():
    """测试参数网络创建"""
    import jax
    
    net = ParameterNetwork(hidden_dims=(32, 16))
    
    # 初始化
    key = jax.random.PRNGKey(0)
    X = jnp.ones((10, 5))
    params = net.init(key, X, training=False)
    
    # 前向传播
    output = net.apply(params, X, training=False)
    
    # 检查输出形状
    assert output.shape == (10,)


def test_parameter_network_with_dropout():
    """测试带 Dropout 的参数网络"""
    import jax
    
    net = ParameterNetwork(hidden_dims=(32, 16), dropout_rate=0.2)
    
    key = jax.random.PRNGKey(0)
    dropout_key = jax.random.PRNGKey(1)
    X = jnp.ones((10, 5))
    params = net.init({"params": key, "dropout": dropout_key}, X, training=True)
    
    # 训练模式（有 dropout）
    output_train = net.apply(params, X, training=True, rngs={"dropout": dropout_key})
    assert output_train.shape == (10,)
    
    # 推理模式（无 dropout）
    output_eval = net.apply(params, X, training=False)
    assert output_eval.shape == (10,)


def test_deep_gamlss_creation():
    """测试 Deep GAMLSS 模型创建"""
    import jax
    
    family = NO()
    model = DeepGAMLSS(family=family, hidden_dims=(32, 16))
    
    # 初始化
    key = jax.random.PRNGKey(0)
    X = jnp.ones((10, 5))
    params = model.init(key, X, training=False)
    
    # 前向传播
    pred_params = model.apply(params, X, training=False)
    
    # 检查输出
    assert "mu" in pred_params
    assert "sigma" in pred_params
    assert pred_params["mu"].shape == (10,)
    assert pred_params["sigma"].shape == (10,)


def test_deep_gamlss_with_shared_layers():
    """测试带共享层的 Deep GAMLSS"""
    import jax
    
    family = NO()
    model = DeepGAMLSS(family=family, hidden_dims=(32, 16), shared_layers=True)
    
    key = jax.random.PRNGKey(0)
    X = jnp.ones((10, 5))
    params = model.init(key, X, training=False)
    
    pred_params = model.apply(params, X, training=False)
    
    assert "mu" in pred_params
    assert "sigma" in pred_params


def test_fit_deep_gamlss_basic():
    """测试基本的 Deep GAMLSS 拟合"""
    np.random.seed(42)
    
    # 生成简单数据
    n = 100
    X = np.random.randn(n, 2)
    y = 2 + 3*X[:, 0] + np.random.randn(n) * 0.5
    
    # 拟合模型
    family = NO()
    model, params, history = fit_deep_gamlss(
        jnp.array(X),
        jnp.array(y),
        family=family,
        hidden_dims=(16, 8),
        learning_rate=0.01,
        n_epochs=50,
        verbose=False
    )
    
    # 检查返回值
    assert model is not None
    assert params is not None
    assert "loss" in history
    assert len(history["loss"]) > 0
    
    # 检查损失下降
    assert history["loss"][-1] < history["loss"][0]


def test_fit_deep_gamlss_with_validation():
    """测试带验证集的 Deep GAMLSS 拟合"""
    np.random.seed(42)
    
    n = 200
    X = np.random.randn(n, 2)
    y = 2 + 3*X[:, 0] + np.random.randn(n) * 0.5
    
    family = NO()
    model, params, history = fit_deep_gamlss(
        jnp.array(X),
        jnp.array(y),
        family=family,
        hidden_dims=(16, 8),
        learning_rate=0.01,
        n_epochs=50,
        validation_split=0.2,
        verbose=False
    )
    
    # 检查验证损失
    assert "val_loss" in history
    assert len(history["val_loss"]) > 0


def test_fit_deep_gamlss_with_batches():
    """测试小批次训练"""
    np.random.seed(42)
    
    n = 200
    X = np.random.randn(n, 2)
    y = 2 + 3*X[:, 0] + np.random.randn(n) * 0.5
    
    family = NO()
    model, params, history = fit_deep_gamlss(
        jnp.array(X),
        jnp.array(y),
        family=family,
        hidden_dims=(16, 8),
        learning_rate=0.01,
        n_epochs=30,
        batch_size=32,
        verbose=False
    )
    
    assert model is not None
    assert len(history["loss"]) > 0


def test_predict_deep_gamlss():
    """测试 Deep GAMLSS 预测"""
    np.random.seed(42)
    
    # 训练数据
    n_train = 100
    X_train = np.random.randn(n_train, 2)
    y_train = 2 + 3*X_train[:, 0] + np.random.randn(n_train) * 0.5
    
    # 拟合模型
    family = NO()
    model, params, _ = fit_deep_gamlss(
        jnp.array(X_train),
        jnp.array(y_train),
        family=family,
        hidden_dims=(16, 8),
        learning_rate=0.01,
        n_epochs=30,
        verbose=False
    )
    
    # 预测
    n_test = 20
    X_test = np.random.randn(n_test, 2)
    pred_params = predict_deep_gamlss(model, params, jnp.array(X_test))
    
    # 检查预测结果
    assert "mu" in pred_params
    assert "sigma" in pred_params
    assert pred_params["mu"].shape == (n_test,)
    assert pred_params["sigma"].shape == (n_test,)
    
    # 检查 sigma 为正
    assert jnp.all(pred_params["sigma"] > 0)


def test_deep_gamlss_nonlinear_relationship():
    """测试 Deep GAMLSS 学习非线性关系"""
    np.random.seed(42)
    
    # 生成非线性数据
    n = 500
    X = np.random.uniform(-3, 3, (n, 2))
    # 复杂的非线性关系
    mu_true = np.sin(X[:, 0]) + np.cos(X[:, 1])
    y = mu_true + np.random.randn(n) * 0.3
    
    # 拟合模型
    family = NO()
    model, params, history = fit_deep_gamlss(
        jnp.array(X),
        jnp.array(y),
        family=family,
        hidden_dims=(64, 32),
        learning_rate=0.001,
        n_epochs=100,
        verbose=False
    )
    
    # 预测
    pred_params = predict_deep_gamlss(model, params, jnp.array(X))
    
    # 检查预测质量（R²）
    mu_pred = np.array(pred_params["mu"])
    ss_res = np.sum((y - mu_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    
    # R² 应该比较高（至少 > 0.5）
    assert r2 > 0.5, f"R² = {r2:.3f} is too low"


def test_deep_gamlss_gamma_family():
    """测试 Deep GAMLSS 与 Gamma 分布"""
    np.random.seed(42)
    
    # 生成 Gamma 数据
    n = 200
    X = np.random.randn(n, 2)
    mu_true = np.exp(1 + 0.5*X[:, 0])
    y = np.random.gamma(shape=2, scale=mu_true/2)
    
    # 拟合模型
    family = GA()
    model, params, history = fit_deep_gamlss(
        jnp.array(X),
        jnp.array(y),
        family=family,
        hidden_dims=(32, 16),
        learning_rate=0.01,
        n_epochs=50,
        verbose=False
    )
    
    # 预测
    pred_params = predict_deep_gamlss(model, params, jnp.array(X))
    
    # 检查参数
    assert "mu" in pred_params
    assert "sigma" in pred_params
    assert jnp.all(pred_params["mu"] > 0)
    assert jnp.all(pred_params["sigma"] > 0)


def test_deep_gamlss_early_stopping():
    """测试早停功能"""
    np.random.seed(42)
    
    n = 200
    X = np.random.randn(n, 2)
    y = 2 + 3*X[:, 0] + np.random.randn(n) * 0.5
    
    family = NO()
    model, params, history = fit_deep_gamlss(
        jnp.array(X),
        jnp.array(y),
        family=family,
        hidden_dims=(16, 8),
        learning_rate=0.01,
        n_epochs=1000,  # 很多轮
        validation_split=0.2,
        early_stopping_patience=5,
        verbose=False
    )
    
    # 应该提前停止（不会训练 1000 轮）
    assert len(history["loss"]) < 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
