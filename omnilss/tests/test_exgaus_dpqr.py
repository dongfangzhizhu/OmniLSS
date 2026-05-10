"""Test exGAUS (Exponentially-modified Gaussian) distribution dpqr functions."""

import jax.numpy as jnp
import jax.random as jrandom
from omnilss.dpqr_functions import dexGAUS, pexGAUS, qexGAUS, rexGAUS

print("=" * 70)
print("测试 exGAUS (Exponentially-modified Gaussian) 分布的 dpqr 函数")
print("=" * 70)

# Test parameters
mu = 0.0
sigma = 1.0
nu = 0.5

print(f"\n参数: mu={mu}, sigma={sigma}, nu={nu}")

# 1. Test density function
print("\n1. 测试密度函数 dexGAUS")
x = jnp.array([-1.0, 0.0, 1.0, 2.0, 3.0])
d = dexGAUS(x, mu, sigma, nu)
print(f"x = {x}")
print(f"d(x) = {d}")
print(f"✓ 所有密度值为正: {jnp.all(d > 0)}")

# 2. Test CDF
print("\n2. 测试 CDF pexGAUS")
q = jnp.array([-1.0, 0.0, 1.0, 2.0, 3.0])
p = pexGAUS(q, mu, sigma, nu)
print(f"q = {q}")
print(f"p(q) = {p}")
print(f"✓ CDF 在 [0, 1] 范围内: {jnp.all((p >= 0) & (p <= 1))}")
print(f"✓ CDF 单调递增: {jnp.all(jnp.diff(p) > 0)}")

# 3. Test quantile function
print("\n3. 测试分位数函数 qexGAUS")
p_test = jnp.array([0.1, 0.25, 0.5, 0.75, 0.9])
q_result = qexGAUS(p_test, mu, sigma, nu)
print(f"p = {p_test}")
print(f"q(p) = {q_result}")
print(f"✓ 分位数单调递增: {jnp.all(jnp.diff(q_result) > 0)}")

# 4. Test p-q consistency
print("\n4. 测试 p 和 q 的一致性")
x_test = jnp.linspace(-2.0, 4.0, 15)
p_from_x = pexGAUS(x_test, mu, sigma, nu)
x_back = qexGAUS(p_from_x, mu, sigma, nu)
max_error = jnp.max(jnp.abs(x_test - x_back))
print(f"max|q(p(x)) - x| = {max_error:.6e}")
print(f"✓ 一致性良好: {max_error < 1e-6}")

# 5. Test random generation (convolution method)
print("\n5. 测试随机数生成 rexGAUS")
key = jrandom.PRNGKey(42)
n = 1000
samples = rexGAUS(key, n, mu, sigma, nu)
print(f"生成 {n} 个样本")
print(f"样本均值: {jnp.mean(samples):.4f} (理论值: {mu + nu:.4f})")
print(f"样本标准差: {jnp.std(samples):.4f}")
print(f"样本最小值: {jnp.min(samples):.4f}")
print(f"样本最大值: {jnp.max(samples):.4f}")

# Theoretical mean and variance
theoretical_mean = mu + nu
theoretical_var = sigma**2 + nu**2
print(f"理论均值: {theoretical_mean:.4f}")
print(f"理论标准差: {jnp.sqrt(theoretical_var):.4f}")

# 6. Test different parameter values
print("\n6. 测试不同参数值")

# Case 1: Different nu values
print("\n  Case 1: 不同的 nu 值（指数尾部强度）")
nu_vals = [0.1, 0.5, 1.0]
x1 = 2.0
for n_val in nu_vals:
    d1 = dexGAUS(x1, mu, sigma, n_val)
    print(f"  nu={n_val}: d({x1}) = {d1:.6f}")

# Case 2: Very small nu (should approach normal)
print("\n  Case 2: nu 很小（接近正态分布）")
nu_small = 0.01
x2 = jnp.array([0.0, 1.0, 2.0])
d_exgaus = dexGAUS(x2, mu, sigma, nu_small)
from scipy.stats import norm as sp_norm
d_normal = sp_norm.pdf(x2, loc=mu, scale=sigma)
print(f"  x = {x2}")
print(f"  exGAUS d(x) = {d_exgaus}")
print(f"  Normal d(x) = {d_normal}")
print(f"  差异: {jnp.max(jnp.abs(d_exgaus - d_normal)):.6e}")

# 7. Test density integration (numerical)
print("\n7. 验证密度函数积分（数值积分）")
x_grid = jnp.linspace(-5.0, 10.0, 1000)
dx = x_grid[1] - x_grid[0]
d_grid = dexGAUS(x_grid, mu, sigma, nu)
integral = jnp.sum(d_grid) * dx
print(f"密度函数积分: {integral:.6f} (应该接近 1.0)")
print(f"积分误差: {jnp.abs(integral - 1.0):.6e}")

# 8. Test right skewness (characteristic of exGAUS)
print("\n8. 测试右偏特性")
key2 = jrandom.PRNGKey(123)
samples2 = rexGAUS(key2, 10000, mu, sigma, nu)
from scipy.stats import skew
skewness = skew(samples2)
print(f"样本偏度: {skewness:.4f} (应该为正，表示右偏)")
print(f"✓ 右偏特性: {skewness > 0}")

print("\n" + "=" * 70)
print("✅ exGAUS 分布测试完成！")
print("=" * 70)
