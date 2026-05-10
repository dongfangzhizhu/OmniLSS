"""Test NET (Normal-Exponential-t) distribution dpqr functions."""

import jax.numpy as jnp
import jax.random as jrandom
from omnilss.dpqr_functions import dNET, pNET, qNET, rNET

print("=" * 70)
print("测试 NET (Normal-Exponential-t) 分布的 dpqr 函数")
print("=" * 70)

# Test parameters
mu = 0.0
sigma = 1.0
nu = 2.0
tau = 4.0

print(f"\n参数: mu={mu}, sigma={sigma}, nu={nu}, tau={tau}")

# 1. Test density function
print("\n1. 测试密度函数 dNET")
x = jnp.array([-3.0, -1.0, 0.0, 1.0, 3.0, 5.0])
d = dNET(x, mu, sigma, nu, tau)
print(f"x = {x}")
print(f"d(x) = {d}")
print(f"✓ 所有密度值为正: {jnp.all(d > 0)}")

# 2. Test symmetry around mu
print("\n2. 测试对称性")
x_sym = jnp.array([-2.0, 2.0])
d_sym = dNET(x_sym, mu, sigma, nu, tau)
print(f"d(-2) = {d_sym[0]:.6f}")
print(f"d(2) = {d_sym[1]:.6f}")
print(f"✓ 对称性: |d(-2) - d(2)| = {jnp.abs(d_sym[0] - d_sym[1]):.6e}")

# 3. Test CDF
print("\n3. 测试 CDF pNET")
q = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
p = pNET(q, mu, sigma, nu, tau)
print(f"q = {q}")
print(f"p(q) = {p}")
print(f"✓ CDF 在 [0, 1] 范围内: {jnp.all((p >= 0) & (p <= 1))}")
print(f"✓ CDF 单调递增: {jnp.all(jnp.diff(p) > 0)}")

# 4. Test quantile function
print("\n4. 测试分位数函数 qNET")
p_test = jnp.array([0.1, 0.25, 0.5, 0.75, 0.9])
q_result = qNET(p_test, mu, sigma, nu, tau)
print(f"p = {p_test}")
print(f"q(p) = {q_result}")
print(f"✓ 分位数单调递增: {jnp.all(jnp.diff(q_result) > 0)}")

# 5. Test p-q consistency
print("\n5. 测试 p 和 q 的一致性")
x_test = jnp.linspace(-2.0, 2.0, 10)
p_from_x = pNET(x_test, mu, sigma, nu, tau)
x_back = qNET(p_from_x, mu, sigma, nu, tau)
max_error = jnp.max(jnp.abs(x_test - x_back))
print(f"max|q(p(x)) - x| = {max_error:.6e}")
print(f"✓ 一致性良好: {max_error < 1e-6}")

# 6. Test random generation
print("\n6. 测试随机数生成 rNET")
key = jrandom.PRNGKey(42)
n = 1000
samples = rNET(key, n, mu, sigma, nu, tau)
print(f"生成 {n} 个样本")
print(f"样本均值: {jnp.mean(samples):.4f} (理论值: {mu:.4f})")
print(f"样本标准差: {jnp.std(samples):.4f}")
print(f"样本最小值: {jnp.min(samples):.4f}")
print(f"样本最大值: {jnp.max(samples):.4f}")

# 7. Test different parameter values
print("\n7. 测试不同参数值")

# Case 1: Different nu (threshold k1)
print("\n  Case 1: 不同的 nu 值（中心区域大小）")
nu_vals = [1.0, 2.0, 3.0]
x1 = 0.0
for n_val in nu_vals:
    d1 = dNET(x1, mu, sigma, n_val, tau)
    print(f"  nu={n_val}: d({x1}) = {d1:.6f}")

# Case 2: Different tau (threshold k2)
print("\n  Case 2: 不同的 tau 值（尾部开始位置）")
tau_vals = [3.0, 4.0, 5.0]
x2 = 0.0
for t_val in tau_vals:
    d2 = dNET(x2, mu, sigma, nu, t_val)
    print(f"  tau={t_val}: d({x2}) = {d2:.6f}")

# 8. Test density integration (numerical)
print("\n8. 验证密度函数积分（数值积分）")
x_grid = jnp.linspace(-10.0, 10.0, 1000)
dx = x_grid[1] - x_grid[0]
d_grid = dNET(x_grid, mu, sigma, nu, tau)
integral = jnp.sum(d_grid) * dx
print(f"密度函数积分: {integral:.6f} (应该接近 1.0)")
print(f"积分误差: {jnp.abs(integral - 1.0):.6e}")

print("\n" + "=" * 70)
print("✅ NET 分布测试完成！")
print("=" * 70)
