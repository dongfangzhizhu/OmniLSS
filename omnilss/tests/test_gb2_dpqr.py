"""Test GB2 (Generalized Beta Type 2) distribution dpqr functions."""

import jax.numpy as jnp
import jax.random as jrandom
from omnilss.dpqr_functions import dGB2, pGB2, qGB2, rGB2

print("=" * 70)
print("测试 GB2 (Generalized Beta Type 2) 分布的 dpqr 函数")
print("=" * 70)

# Test parameters
mu = 1.0
sigma = 1.0
nu = 2.0
tau = 3.0

print(f"\n参数: mu={mu}, sigma={sigma}, nu={nu}, tau={tau}")

# 1. Test density function
print("\n1. 测试密度函数 dGB2")
x = jnp.array([0.5, 1.0, 1.5, 2.0, 3.0])
d = dGB2(x, mu, sigma, nu, tau)
print(f"x = {x}")
print(f"d(x) = {d}")
print(f"✓ 所有密度值为正: {jnp.all(d > 0)}")

# 2. Test CDF
print("\n2. 测试 CDF pGB2")
q = jnp.array([0.5, 1.0, 1.5, 2.0, 3.0])
p = pGB2(q, mu, sigma, nu, tau)
print(f"q = {q}")
print(f"p(q) = {p}")
print(f"✓ CDF 在 [0, 1] 范围内: {jnp.all((p >= 0) & (p <= 1))}")
print(f"✓ CDF 单调递增: {jnp.all(jnp.diff(p) > 0)}")

# 3. Test quantile function
print("\n3. 测试分位数函数 qGB2")
p_test = jnp.array([0.1, 0.25, 0.5, 0.75, 0.9])
q_result = qGB2(p_test, mu, sigma, nu, tau)
print(f"p = {p_test}")
print(f"q(p) = {q_result}")
print(f"✓ 分位数单调递增: {jnp.all(jnp.diff(q_result) > 0)}")

# 4. Test p-q consistency
print("\n4. 测试 p 和 q 的一致性")
x_test = jnp.linspace(0.3, 3.0, 20)
p_from_x = pGB2(x_test, mu, sigma, nu, tau)
x_back = qGB2(p_from_x, mu, sigma, nu, tau)
max_error = jnp.max(jnp.abs(x_test - x_back))
print(f"max|q(p(x)) - x| = {max_error:.6e}")
print(f"✓ 一致性良好: {max_error < 1e-10}")

# 5. Test random generation
print("\n5. 测试随机数生成 rGB2")
key = jrandom.PRNGKey(42)
n = 1000
samples = rGB2(key, n, mu, sigma, nu, tau)
print(f"生成 {n} 个样本")
print(f"样本均值: {jnp.mean(samples):.4f}")
print(f"样本标准差: {jnp.std(samples):.4f}")
print(f"样本最小值: {jnp.min(samples):.4f}")
print(f"样本最大值: {jnp.max(samples):.4f}")
print(f"✓ 所有样本为正: {jnp.all(samples > 0)}")

# 6. Test different parameter combinations
print("\n6. 测试不同参数组合")

# Case 1: Different sigma
print("\n  Case 1: 不同的 sigma 值")
sigma_vals = [0.5, 1.0, 2.0]
x1 = 1.5
for s in sigma_vals:
    d1 = dGB2(x1, mu, s, nu, tau)
    print(f"  sigma={s}: d({x1}) = {d1:.6f}")

# Case 2: Different nu and tau
print("\n  Case 2: 不同的 nu 和 tau 值")
nu_tau_pairs = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
x2 = 1.0
for n, t in nu_tau_pairs:
    d2 = dGB2(x2, mu, sigma, n, t)
    print(f"  nu={n}, tau={t}: d({x2}) = {d2:.6f}")

# 7. Test density integration (numerical)
print("\n7. 验证密度函数积分（数值积分）")
x_grid = jnp.linspace(0.01, 10.0, 1000)
dx = x_grid[1] - x_grid[0]
d_grid = dGB2(x_grid, mu, sigma, nu, tau)
integral = jnp.sum(d_grid) * dx
print(f"密度函数积分: {integral:.6f} (应该接近 1.0)")
print(f"积分误差: {jnp.abs(integral - 1.0):.6e}")

print("\n" + "=" * 70)
print("✅ GB2 分布测试完成！")
print("=" * 70)
