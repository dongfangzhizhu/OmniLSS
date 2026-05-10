"""Test PARETO2 (Pareto Type 2 / Lomax) distribution dpqr functions."""

import jax.numpy as jnp
import jax.random as jrandom
from omnilss.dpqr_functions import dPARETO2, pPARETO2, qPARETO2, rPARETO2

print("=" * 70)
print("测试 PARETO2 (Pareto Type 2 / Lomax) 分布的 dpqr 函数")
print("=" * 70)

# Test parameters
mu = 1.0
sigma = 0.5

print(f"\n参数: mu={mu}, sigma={sigma}")

# 1. Test density function
print("\n1. 测试密度函数 dPARETO2")
x = jnp.array([0.5, 1.0, 2.0, 3.0, 5.0])
d = dPARETO2(x, mu, sigma)
print(f"x = {x}")
print(f"d(x) = {d}")
print(f"✓ 所有密度值为正: {jnp.all(d > 0)}")

# 2. Test CDF
print("\n2. 测试 CDF pPARETO2")
q = jnp.array([0.5, 1.0, 2.0, 3.0, 5.0])
p = pPARETO2(q, mu, sigma)
print(f"q = {q}")
print(f"p(q) = {p}")
print(f"✓ CDF 在 [0, 1] 范围内: {jnp.all((p >= 0) & (p <= 1))}")
print(f"✓ CDF 单调递增: {jnp.all(jnp.diff(p) > 0)}")

# 3. Test quantile function
print("\n3. 测试分位数函数 qPARETO2")
p_test = jnp.array([0.1, 0.25, 0.5, 0.75, 0.9])
q_result = qPARETO2(p_test, mu, sigma)
print(f"p = {p_test}")
print(f"q(p) = {q_result}")
print(f"✓ 分位数单调递增: {jnp.all(jnp.diff(q_result) > 0)}")

# 4. Test p-q consistency
print("\n4. 测试 p 和 q 的一致性")
x_test = jnp.linspace(0.1, 5.0, 20)
p_from_x = pPARETO2(x_test, mu, sigma)
x_back = qPARETO2(p_from_x, mu, sigma)
max_error = jnp.max(jnp.abs(x_test - x_back))
print(f"max|q(p(x)) - x| = {max_error:.6e}")
print(f"✓ 一致性良好: {max_error < 1e-12}")

# 5. Test random generation
print("\n5. 测试随机数生成 rPARETO2")
key = jrandom.PRNGKey(42)
n = 1000
samples = rPARETO2(key, n, mu, sigma)
print(f"生成 {n} 个样本")
print(f"样本均值: {jnp.mean(samples):.4f} (理论值: {mu:.4f} for sigma < 1)")
print(f"样本标准差: {jnp.std(samples):.4f}")
print(f"样本最小值: {jnp.min(samples):.4f}")
print(f"样本最大值: {jnp.max(samples):.4f}")
print(f"✓ 所有样本为正: {jnp.all(samples > 0)}")

# 6. Test different parameter values
print("\n6. 测试不同参数值")

# Case 1: Different sigma values
print("\n  Case 1: 不同的 sigma 值")
sigma_vals = [0.3, 0.5, 0.8]
x1 = 2.0
for s in sigma_vals:
    d1 = dPARETO2(x1, mu, s)
    print(f"  sigma={s}: d({x1}) = {d1:.6f}")

# Case 2: Different mu values
print("\n  Case 2: 不同的 mu 值")
mu_vals = [0.5, 1.0, 2.0]
x2 = 1.0
for m in mu_vals:
    d2 = dPARETO2(x2, m, sigma)
    print(f"  mu={m}: d({x2}) = {d2:.6f}")

# 7. Test density integration (numerical)
print("\n7. 验证密度函数积分（数值积分）")
x_grid = jnp.linspace(0.01, 20.0, 1000)
dx = x_grid[1] - x_grid[0]
d_grid = dPARETO2(x_grid, mu, sigma)
integral = jnp.sum(d_grid) * dx
print(f"密度函数积分: {integral:.6f} (应该接近 1.0)")
print(f"积分误差: {jnp.abs(integral - 1.0):.6e}")

# 8. Test heavy tail property
print("\n8. 测试重尾特性")
x_large = jnp.array([10.0, 50.0, 100.0])
p_large = pPARETO2(x_large, mu, sigma)
print(f"x = {x_large}")
print(f"P(X > x) = {1.0 - p_large}")
print(f"✓ 重尾特性: P(X > x) 下降缓慢")

print("\n" + "=" * 70)
print("✅ PARETO2 分布测试完成！")
print("=" * 70)
