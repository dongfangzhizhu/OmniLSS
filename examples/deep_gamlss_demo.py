"""Deep GAMLSS 演示

展示如何使用神经网络建模分布参数，这是 OmniLSS 超越 R gamlss 的创新功能。

Deep GAMLSS 的优势：
1. 自动学习复杂的非线性关系
2. 不需要手动指定平滑项
3. 可以处理高维数据
4. 自动学习交互效应
"""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from omnilss.deep import fit_deep_gamlss, predict_deep_gamlss
from omnilss import NO, GA

print("="*70)
print("Deep GAMLSS 演示")
print("="*70)
print()

# ============================================================================
# 演示 1: 基本用法 - 简单线性关系
# ============================================================================
print("演示 1: 基本用法 - 简单线性关系")
print("-" * 70)

np.random.seed(42)
n = 200
X = np.random.randn(n, 2)
y = 2 + 3*X[:, 0] + 1.5*X[:, 1] + np.random.randn(n) * 0.5

print(f"数据: n={n}, p={X.shape[1]}")
print(f"真实关系: y = 2 + 3*x1 + 1.5*x2 + ε")
print()

# 拟合 Deep GAMLSS
model, params, history = fit_deep_gamlss(
    jnp.array(X),
    jnp.array(y),
    family=NO(),
    hidden_dims=(32, 16),
    learning_rate=0.01,
    n_epochs=100,
    verbose=False
)

print(f"训练完成！")
print(f"初始损失: {history['loss'][0]:.4f}")
print(f"最终损失: {history['loss'][-1]:.4f}")
print()

# 预测
pred_params = predict_deep_gamlss(model, params, jnp.array(X))
mu_pred = np.array(pred_params["mu"])
sigma_pred = np.array(pred_params["sigma"])

# 计算 R²
ss_res = np.sum((y - mu_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - ss_res / ss_tot

print(f"预测质量:")
print(f"  R² = {r2:.4f}")
print(f"  平均 σ = {np.mean(sigma_pred):.4f}")
print()

# ============================================================================
# 演示 2: 复杂非线性关系
# ============================================================================
print("演示 2: 复杂非线性关系")
print("-" * 70)

np.random.seed(42)
n = 500
X = np.random.uniform(-3, 3, (n, 2))
# 复杂的非线性关系
mu_true = np.sin(X[:, 0]) + np.cos(X[:, 1]) + 0.5 * X[:, 0] * X[:, 1]
y = mu_true + np.random.randn(n) * 0.3

print(f"数据: n={n}, p={X.shape[1]}")
print(f"真实关系: y = sin(x1) + cos(x2) + 0.5*x1*x2 + ε")
print()

# 拟合 Deep GAMLSS
model, params, history = fit_deep_gamlss(
    jnp.array(X),
    jnp.array(y),
    family=NO(),
    hidden_dims=(64, 32),
    learning_rate=0.001,
    n_epochs=200,
    validation_split=0.2,
    verbose=False
)

print(f"训练完成！")
print(f"训练损失: {history['loss'][-1]:.4f}")
print(f"验证损失: {history['val_loss'][-1]:.4f}")
print()

# 预测
pred_params = predict_deep_gamlss(model, params, jnp.array(X))
mu_pred = np.array(pred_params["mu"])

# 计算 R²
ss_res = np.sum((y - mu_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - ss_res / ss_tot

print(f"预测质量:")
print(f"  R² = {r2:.4f}")
print(f"  相关系数 = {np.corrcoef(mu_true, mu_pred)[0, 1]:.4f}")
print()

# ============================================================================
# 演示 3: Gamma 分布 - 异方差性
# ============================================================================
print("演示 3: Gamma 分布 - 异方差性")
print("-" * 70)

np.random.seed(42)
n = 300
X = np.random.randn(n, 2)
# μ 和 σ 都依赖于 X
mu_true = np.exp(1 + 0.5*X[:, 0])
sigma_true = 0.5 + 0.3 * np.abs(X[:, 1])
# 生成 Gamma 数据
shape = 1 / (sigma_true ** 2)
scale = mu_true / shape
y = np.random.gamma(shape=shape, scale=scale)

print(f"数据: n={n}, p={X.shape[1]}")
print(f"真实关系: μ = exp(1 + 0.5*x1), σ = 0.5 + 0.3*|x2|")
print()

# 拟合 Deep GAMLSS
model, params, history = fit_deep_gamlss(
    jnp.array(X),
    jnp.array(y),
    family=GA(),
    hidden_dims=(32, 16),
    learning_rate=0.01,
    n_epochs=150,
    verbose=False
)

print(f"训练完成！")
print(f"最终损失: {history['loss'][-1]:.4f}")
print()

# 预测
pred_params = predict_deep_gamlss(model, params, jnp.array(X))
mu_pred = np.array(pred_params["mu"])
sigma_pred = np.array(pred_params["sigma"])

print(f"预测质量:")
print(f"  μ 相关系数 = {np.corrcoef(mu_true, mu_pred)[0, 1]:.4f}")
print(f"  σ 相关系数 = {np.corrcoef(sigma_true, sigma_pred)[0, 1]:.4f}")
print()

# ============================================================================
# 演示 4: 与传统 GAMLSS 对比
# ============================================================================
print("演示 4: 与传统 GAMLSS 对比")
print("-" * 70)

from omnilss import gamlss

np.random.seed(42)
n = 200
x = np.random.randn(n)
# 非线性关系
y = 2 + 3*np.sin(x) + np.random.randn(n) * 0.5
data = {"y": y, "x": x}

print(f"数据: n={n}")
print(f"真实关系: y = 2 + 3*sin(x) + ε")
print()

# 传统 GAMLSS（需要手动指定平滑项）
print("拟合传统 GAMLSS (需要手动指定平滑项)...")
model_traditional = gamlss("y ~ pb(x, df=5)", family="NO", data=data)
mu_traditional = model_traditional.fitted_values["mu"]

# Deep GAMLSS（自动学习）
print("拟合 Deep GAMLSS (自动学习非线性关系)...")
X_deep = x.reshape(-1, 1)
model_deep, params_deep, _ = fit_deep_gamlss(
    jnp.array(X_deep),
    jnp.array(y),
    family=NO(),
    hidden_dims=(32, 16),
    learning_rate=0.01,
    n_epochs=100,
    verbose=False
)
pred_params_deep = predict_deep_gamlss(model_deep, params_deep, jnp.array(X_deep))
mu_deep = np.array(pred_params_deep["mu"])

# 比较
r2_traditional = 1 - np.sum((y - mu_traditional) ** 2) / np.sum((y - np.mean(y)) ** 2)
r2_deep = 1 - np.sum((y - mu_deep) ** 2) / np.sum((y - np.mean(y)) ** 2)

print()
print(f"预测质量比较:")
print(f"  传统 GAMLSS R² = {r2_traditional:.4f}")
print(f"  Deep GAMLSS R²  = {r2_deep:.4f}")
print()

# ============================================================================
# 可视化
# ============================================================================
print("生成可视化...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 演示 1: 训练曲线
ax = axes[0, 0]
ax.plot(history['loss'], label='Training Loss')
if 'val_loss' in history and len(history['val_loss']) > 0:
    ax.plot(history['val_loss'], label='Validation Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('演示 1: 训练曲线')
ax.legend()
ax.grid(True, alpha=0.3)

# 演示 2: 非线性拟合
ax = axes[0, 1]
# 使用演示2的数据
np.random.seed(42)
n_demo2 = 500
X_demo2 = np.random.uniform(-3, 3, (n_demo2, 2))
mu_true_demo2 = np.sin(X_demo2[:, 0]) + np.cos(X_demo2[:, 1]) + 0.5 * X_demo2[:, 0] * X_demo2[:, 1]
y_demo2 = mu_true_demo2 + np.random.randn(n_demo2) * 0.3

# 拟合
model_demo2, params_demo2, _ = fit_deep_gamlss(
    jnp.array(X_demo2),
    jnp.array(y_demo2),
    family=NO(),
    hidden_dims=(64, 32),
    learning_rate=0.001,
    n_epochs=200,
    verbose=False
)
pred_params_demo2 = predict_deep_gamlss(model_demo2, params_demo2, jnp.array(X_demo2))
mu_pred_demo2 = np.array(pred_params_demo2["mu"])
r2_demo2 = 1 - np.sum((y_demo2 - mu_pred_demo2) ** 2) / np.sum((y_demo2 - np.mean(y_demo2)) ** 2)

# 选择 x1 维度进行可视化
idx_sorted = np.argsort(X_demo2[:, 0])
ax.scatter(X_demo2[idx_sorted, 0], y_demo2[idx_sorted], alpha=0.3, s=10, label='Data')
ax.plot(X_demo2[idx_sorted, 0], mu_pred_demo2[idx_sorted], 'r-', linewidth=2, label='Deep GAMLSS')
ax.plot(X_demo2[idx_sorted, 0], mu_true_demo2[idx_sorted], 'g--', linewidth=2, label='True')
ax.set_xlabel('x1')
ax.set_ylabel('y')
ax.set_title(f'演示 2: 非线性拟合 (R²={r2_demo2:.3f})')
ax.legend()
ax.grid(True, alpha=0.3)

# 演示 3: Gamma 分布 - μ 预测
ax = axes[1, 0]
# 使用演示3的数据
np.random.seed(42)
n_demo3 = 300
X_demo3 = np.random.randn(n_demo3, 2)
mu_true_demo3 = np.exp(1 + 0.5*X_demo3[:, 0])
sigma_true_demo3 = 0.5 + 0.3 * np.abs(X_demo3[:, 1])
shape_demo3 = 1 / (sigma_true_demo3 ** 2)
scale_demo3 = mu_true_demo3 / shape_demo3
y_demo3 = np.random.gamma(shape=shape_demo3, scale=scale_demo3)

# 拟合
model_demo3, params_demo3, _ = fit_deep_gamlss(
    jnp.array(X_demo3),
    jnp.array(y_demo3),
    family=GA(),
    hidden_dims=(32, 16),
    learning_rate=0.01,
    n_epochs=150,
    verbose=False
)
pred_params_demo3 = predict_deep_gamlss(model_demo3, params_demo3, jnp.array(X_demo3))
mu_pred_demo3 = np.array(pred_params_demo3["mu"])

ax.scatter(mu_true_demo3, mu_pred_demo3, alpha=0.5, s=20)
ax.plot([mu_true_demo3.min(), mu_true_demo3.max()], [mu_true_demo3.min(), mu_true_demo3.max()], 
        'r--', linewidth=2, label='Perfect')
ax.set_xlabel('True μ')
ax.set_ylabel('Predicted μ')
ax.set_title('演示 3: μ 预测 (Gamma)')
ax.legend()
ax.grid(True, alpha=0.3)

# 演示 4: 传统 vs Deep GAMLSS
ax = axes[1, 1]
idx_sorted = np.argsort(x)
ax.scatter(x[idx_sorted], y[idx_sorted], alpha=0.3, s=10, label='Data')
ax.plot(x[idx_sorted], mu_traditional[idx_sorted], 'b-', linewidth=2, 
        label=f'Traditional (R²={r2_traditional:.3f})')
ax.plot(x[idx_sorted], mu_deep[idx_sorted], 'r-', linewidth=2, 
        label=f'Deep (R²={r2_deep:.3f})')
ax.plot(x[idx_sorted], 2 + 3*np.sin(x[idx_sorted]), 'g--', linewidth=2, 
        label='True')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('演示 4: 传统 vs Deep GAMLSS')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('deep_gamlss_demo.png', dpi=150, bbox_inches='tight')
print(f"图片已保存: deep_gamlss_demo.png")
print()

# ============================================================================
# 总结
# ============================================================================
print("="*70)
print("总结")
print("="*70)
print()
print("Deep GAMLSS 的优势:")
print("  1. ✅ 自动学习复杂的非线性关系")
print("  2. ✅ 不需要手动指定平滑项")
print("  3. ✅ 可以处理高维数据")
print("  4. ✅ 自动学习交互效应")
print("  5. ✅ 支持多种分布族")
print()
print("适用场景:")
print("  - 复杂的非线性关系")
print("  - 高维数据")
print("  - 不确定如何指定平滑项")
print("  - 需要自动学习交互效应")
print()
print("注意事项:")
print("  - 需要足够的数据（建议 n > 100）")
print("  - 需要调整超参数（hidden_dims, learning_rate）")
print("  - 可能需要更长的训练时间")
print("  - 解释性不如传统 GAMLSS")
print()
print("="*70)
print("演示完成！")
print("="*70)
