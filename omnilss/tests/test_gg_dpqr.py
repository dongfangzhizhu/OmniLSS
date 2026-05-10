"""Test GG (Generalized Gamma) distribution dpqr functions."""

import jax.numpy as jnp
import jax.random as jrandom
from omnilss.dpqr_functions import dGG, pGG, qGG, rGG

print("=" * 70)
print("测试 GG (Generalized Gamma) 分布的 dpqr 函数")
print("=" * 70)

# Test parameters
mu = 1.0
sigma = 0.5
nu = 2.0

print(f"\n参数: mu={mu}, sigma={sigma}, nu={nu}")

# 1. Test density function
print("\n1. 测试密度函数 dGG")
x = jnp.array([0.5, 1.0, 1.5, 2.0, 3.0])
d = dGG(x, mu, sigma, nu)
print(f"x = {x}")
print(f"d(x) = {d}")
print(f"✓ 所有密度值为正: {jnp.all(d > 0)}")

# 2. Test CDF
print("\n2. 测试 CDF pGG")
q = jnp.array([0.5, 1.0, 1.5, 2.0, 3.0])
p = pGG(q, mu, sigma, nu)
print(f"q = {q}")
print(f"p(q) = {p}")
print(f"✓ CDF 在 [0, 1] 范围内: {jnp.all((p >= 0) & (p <= 1))}")
print(f"✓ CDF 单调递增: {jnp.all(jnp.diff(p) > 0)}")

# 3. Test quantile function
print("\n3. 测试分位数函数 qGG")
p_test = jnp.array([0.1, 0.25, 0.5, 0.75, 0.9])
q_result = qGG(p_test, mu, sigma, nu)
print(f"p = {p_test}")
print(f"q(p) = {q_result}")
print(f"✓ 分位数单调递增: {jnp.all(jnp.diff(q_result) > 0)}")

# 4. Test p-q consistency
print("\n4. 测试 p 和 q 的一致性")
x_test = jnp.linspace(0.5, 3.0, 20)
p_from_x = pGG(x_test, mu, sigma, nu)
x_back = qGG(p_from_x, mu, sigma, nu)
max_error = jnp.max(jnp.abs(x_test - x_back))
print(f"max|q(p(x)) - x| = {max_error:.6e}")
print(f"✓ 一致性良好: {max_error < 1e-10}")

# 5. Test random generation
print("\n5. 测试随机数生成 rGG")
key = jrandom.PRNGKey(42)
n = 1000
samples = rGG(key, n, mu, sigma, nu)
print(f"生成 {n} 个样本")
print(f"样本均值: {jnp.mean(samples):.4f}")
print(f"样本标准差: {jnp.std(samples):.4f}")
print(f"样本最小值: {jnp.min(samples):.4f}")
print(f"样本最大值: {jnp.max(samples):.4f}")
print(f"✓ 所有样本为正: {jnp.all(samples > 0)}")

# 6. Test special cases
print("\n6. 测试特殊情况")

# Case 1: nu = 1 (should be close to Gamma)
print("\n  Case 1: nu = 1 (接近 Gamma 分布)")
nu1 = 1.0
x1 = jnp.array([0.5, 1.0, 2.0])
d1 = dGG(x1, mu, sigma, nu1)
print(f"  x = {x1}")
print(f"  d(x) = {d1}")

# Case 2: nu close to 0 (should be close to Log-Normal)
print("\n  Case 2: nu ≈ 0 (接近 Log-Normal)")
nu2 = 1e-7
x2 = jnp.array([0.5, 1.0, 2.0])
d2 = dGG(x2, mu, sigma, nu2)
print(f"  x = {x2}")
print(f"  d(x) = {d2}")

# Case 3: negative nu
print("\n  Case 3: nu < 0 (负形状参数)")
nu3 = -2.0
x3 = jnp.array([0.5, 1.0, 2.0])
d3 = dGG(x3, mu, sigma, nu3)
p3 = pGG(x3, mu, sigma, nu3)
print(f"  x = {x3}")
print(f"  d(x) = {d3}")
print(f"  p(x) = {p3}")

# 7. Test density integration (numerical)
print("\n7. 验证密度函数积分（数值积分）")
x_grid = jnp.linspace(0.01, 10.0, 1000)
dx = x_grid[1] - x_grid[0]
d_grid = dGG(x_grid, mu, sigma, nu)
integral = jnp.sum(d_grid) * dx
print(f"密度函数积分: {integral:.6f} (应该接近 1.0)")
print(f"积分误差: {jnp.abs(integral - 1.0):.6e}")

print("\n" + "=" * 70)
print("✅ GG 分布测试完成！")
print("=" * 70)
