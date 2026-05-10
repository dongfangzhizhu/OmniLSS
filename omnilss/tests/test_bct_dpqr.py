"""
测试 BCT 的 dpqr 函数

这个脚本测试新实现的 BCT (Box-Cox t) 分布的 d/p/q/r 函数。
"""

import sys
sys.path.insert(0, 'omnilss/src')

import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
from omnilss.dpqr_functions import dBCT, pBCT, qBCT, rBCT

print("="*70)
print("测试 BCT 分布的 dpqr 函数")
print("="*70)

# 测试参数
mu = 1.0      # 位置参数
sigma = 0.5   # 尺度参数
nu = 1.0      # Box-Cox 变换参数
tau = 10.0    # t 分布自由度

print(f"\n参数: mu={mu}, sigma={sigma}, nu={nu}, tau={tau}")

# 测试 1: 密度函数 (d)
print("\n1. 测试密度函数 dBCT")
x_values = jnp.array([0.5, 1.0, 1.5, 2.0, 3.0])
densities = dBCT(x_values, mu, sigma, nu, tau)
print(f"x = {x_values}")
print(f"d(x) = {densities}")
print(f"✓ 所有密度值为正: {jnp.all(densities > 0)}")

# 测试 2: CDF 函数 (p)
print("\n2. 测试 CDF 函数 pBCT")
q_values = jnp.array([0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0])
cdfs = pBCT(q_values, mu, sigma, nu, tau)
print(f"q = {q_values}")
print(f"P(X <= q) = {cdfs}")
print(f"✓ CDF 在 [0, 1] 范围内: {jnp.all((cdfs >= 0) & (cdfs <= 1))}")
print(f"✓ CDF 单调递增: {np.all(np.diff(cdfs) >= -1e-10)}")

# 测试 3: 分位数函数 (q)
print("\n3. 测试分位数函数 qBCT")
p_values = jnp.array([0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
quantiles = qBCT(p_values, mu, sigma, nu, tau)
print(f"p = {p_values}")
print(f"q(p) = {quantiles}")
print(f"✓ 所有分位数为正: {jnp.all(quantiles > 0)}")
print(f"✓ 分位数单调递增: {np.all(np.diff(quantiles) >= -1e-10)}")

# 测试 4: p 和 q 的一致性
print("\n4. 测试 p 和 q 的一致性")
test_q = jnp.array([0.5, 1.0, 1.5, 2.0, 2.5])
p_from_q = pBCT(test_q, mu, sigma, nu, tau)
q_from_p = qBCT(p_from_q, mu, sigma, nu, tau)
print(f"原始 q: {test_q}")
print(f"p = P(X <= q): {p_from_q}")
print(f"q' = q(p): {q_from_p}")
print(f"差异: {jnp.abs(test_q - q_from_p)}")
print(f"✓ 一致性检查: max|q - q'| = {jnp.max(jnp.abs(test_q - q_from_p)):.6e}")

# 测试 5: 随机数生成 (r)
print("\n5. 测试随机数生成 rBCT")
key = jrandom.PRNGKey(42)
n_samples = 10000
samples = rBCT(key, n_samples, mu, sigma, nu, tau)
print(f"生成 {n_samples} 个样本")
print(f"样本均值: {jnp.mean(samples):.4f}")
print(f"样本中位数: {jnp.median(samples):.4f}")
print(f"样本标准差: {jnp.std(samples):.4f}")
print(f"样本范围: [{jnp.min(samples):.4f}, {jnp.max(samples):.4f}]")
print(f"✓ 所有样本为正: {jnp.all(samples > 0)}")

# 测试 6: 不同的 nu 值（Box-Cox 参数）
print("\n6. 测试不同的 nu 值")

# nu = 0 (对数变换)
nu_zero = 0.0
samples_nu0 = rBCT(key, n_samples, mu, sigma, nu_zero, tau)
print(f"nu={nu_zero} (对数变换):")
print(f"  样本均值: {jnp.mean(samples_nu0):.4f}")
print(f"  样本中位数: {jnp.median(samples_nu0):.4f}")

# nu = 0.5
nu_half = 0.5
samples_nu_half = rBCT(key, n_samples, mu, sigma, nu_half, tau)
print(f"nu={nu_half}:")
print(f"  样本均值: {jnp.mean(samples_nu_half):.4f}")
print(f"  样本中位数: {jnp.median(samples_nu_half):.4f}")

# nu = 2.0
nu_two = 2.0
samples_nu2 = rBCT(key, n_samples, mu, sigma, nu_two, tau)
print(f"nu={nu_two}:")
print(f"  样本均值: {jnp.mean(samples_nu2):.4f}")
print(f"  样本中位数: {jnp.median(samples_nu2):.4f}")

# 测试 7: 不同的 tau 值（自由度）
print("\n7. 测试不同的 tau 值（自由度）")

# tau = 3 (重尾)
tau_small = 3.0
samples_tau3 = rBCT(key, n_samples, mu, sigma, nu, tau_small)
print(f"tau={tau_small} (重尾):")
print(f"  样本标准差: {jnp.std(samples_tau3):.4f}")
print(f"  样本范围: [{jnp.min(samples_tau3):.4f}, {jnp.max(samples_tau3):.4f}]")

# tau = 30 (接近正态)
tau_large = 30.0
samples_tau30 = rBCT(key, n_samples, mu, sigma, nu, tau_large)
print(f"tau={tau_large} (接近正态):")
print(f"  样本标准差: {jnp.std(samples_tau30):.4f}")
print(f"  样本范围: [{jnp.min(samples_tau30):.4f}, {jnp.max(samples_tau30):.4f}]")

# 测试 8: 不同的 sigma 值
print("\n8. 测试不同的 sigma 值")

# sigma 小 (低方差)
sigma_small = 0.2
samples_sigma_small = rBCT(key, n_samples, mu, sigma_small, nu, tau)
print(f"sigma={sigma_small} (低方差):")
print(f"  样本标准差: {jnp.std(samples_sigma_small):.4f}")

# sigma 大 (高方差)
sigma_large = 1.0
samples_sigma_large = rBCT(key, n_samples, mu, sigma_large, nu, tau)
print(f"sigma={sigma_large} (高方差):")
print(f"  样本标准差: {jnp.std(samples_sigma_large):.4f}")

# 测试 9: 验证密度积分
print("\n9. 验证密度积分（数值积分）")
from scipy.integrate import quad

def integrand(x):
    return float(dBCT(jnp.array([x]), mu, sigma, nu, tau)[0])

# 积分范围：从接近0到一个大值
integral, error = quad(integrand, 0.01, 10.0)
print(f"密度函数积分: {integral:.6f} (应该接近 1.0)")
print(f"积分误差: {error:.6e}")

# 测试 10: 边界情况
print("\n10. 测试边界情况")

# 非常小的 x
x_small = jnp.array([0.01, 0.05, 0.1])
d_small = dBCT(x_small, mu, sigma, nu, tau)
print(f"x 很小时: x={x_small}, d(x)={d_small}")
print(f"✓ 密度值有限: {jnp.all(jnp.isfinite(d_small))}")

# 非常大的 x
x_large = jnp.array([10.0, 20.0, 50.0])
d_large = dBCT(x_large, mu, sigma, nu, tau)
print(f"x 很大时: x={x_large}, d(x)={d_large}")
print(f"✓ 密度值有限: {jnp.all(jnp.isfinite(d_large))}")

print("\n" + "="*70)
print("✅ 所有测试完成！")
print("="*70)
