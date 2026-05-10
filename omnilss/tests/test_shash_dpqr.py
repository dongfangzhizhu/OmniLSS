"""
测试 SHASH 的 dpqr 函数

这个脚本测试新实现的 SHASH (Sinh-Arcsinh) 分布的 d/p/q/r 函数。
"""

import sys
sys.path.insert(0, 'omnilss/src')

import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
from omnilss.dpqr_functions import dSHASH, pSHASH, qSHASH, rSHASH

print("="*70)
print("测试 SHASH 分布的 dpqr 函数")
print("="*70)

# 测试参数
mu = 0.0      # 位置参数
sigma = 1.0   # 尺度参数
nu = 0.0      # 偏度参数 (nu=0 时对称)
tau = 1.0     # 峰度参数

print(f"\n参数: mu={mu}, sigma={sigma}, nu={nu}, tau={tau}")
print(f"注意: nu=0 时 SHASH 对称")

# 测试 1: 密度函数 (d)
print("\n1. 测试密度函数 dSHASH")
x_values = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
densities = dSHASH(x_values, mu, sigma, nu, tau)
print(f"x = {x_values}")
print(f"d(x) = {densities}")
print(f"✓ 所有密度值为正: {jnp.all(densities > 0)}")

# 测试 2: CDF 函数 (p)
print("\n2. 测试 CDF 函数 pSHASH")
q_values = jnp.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
cdfs = pSHASH(q_values, mu, sigma, nu, tau)
print(f"q = {q_values}")
print(f"P(X <= q) = {cdfs}")
print(f"✓ CDF 在 [0, 1] 范围内: {jnp.all((cdfs >= 0) & (cdfs <= 1))}")
print(f"✓ CDF 单调递增: {np.all(np.diff(cdfs) >= -1e-10)}")

# 测试 3: 分位数函数 (q)
print("\n3. 测试分位数函数 qSHASH")
p_values = jnp.array([0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
quantiles = qSHASH(p_values, mu, sigma, nu, tau)
print(f"p = {p_values}")
print(f"q(p) = {quantiles}")
print(f"✓ 分位数单调递增: {np.all(np.diff(quantiles) >= -1e-10)}")

# 测试 4: p 和 q 的一致性
print("\n4. 测试 p 和 q 的一致性")
test_q = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
p_from_q = pSHASH(test_q, mu, sigma, nu, tau)
q_from_p = qSHASH(p_from_q, mu, sigma, nu, tau)
print(f"原始 q: {test_q}")
print(f"p = P(X <= q): {p_from_q}")
print(f"q' = q(p): {q_from_p}")
print(f"差异: {jnp.abs(test_q - q_from_p)}")
print(f"✓ 一致性检查: max|q - q'| = {jnp.max(jnp.abs(test_q - q_from_p)):.6e}")

# 测试 5: 随机数生成 (r)
print("\n5. 测试随机数生成 rSHASH")
key = jrandom.PRNGKey(42)
n_samples = 10000
samples = rSHASH(key, n_samples, mu, sigma, nu, tau)
print(f"生成 {n_samples} 个样本")
print(f"样本均值: {jnp.mean(samples):.4f} (应该接近 mu={mu})")
print(f"样本中位数: {jnp.median(samples):.4f}")
print(f"样本标准差: {jnp.std(samples):.4f}")
print(f"样本范围: [{jnp.min(samples):.4f}, {jnp.max(samples):.4f}]")

# 测试 6: 对称性（nu=0 时）
print("\n6. 测试对称性（nu=0 时）")
from scipy.stats import skew
sample_skew = skew(np.array(samples))
print(f"nu={nu}:")
print(f"  偏度: {sample_skew:.4f} (应该接近 0)")

# 测试 7: 不同的 nu 值（偏度参数）
print("\n7. 测试不同的 nu 值（偏度参数）")

# nu < 0 (左偏)
nu_neg = -0.5
samples_neg = rSHASH(key, n_samples, mu, sigma, nu_neg, tau)
skew_neg = skew(np.array(samples_neg))
print(f"nu={nu_neg} (左偏):")
print(f"  样本均值: {jnp.mean(samples_neg):.4f}")
print(f"  偏度: {skew_neg:.4f}")

# nu > 0 (右偏)
nu_pos = 0.5
samples_pos = rSHASH(key, n_samples, mu, sigma, nu_pos, tau)
skew_pos = skew(np.array(samples_pos))
print(f"nu={nu_pos} (右偏):")
print(f"  样本均值: {jnp.mean(samples_pos):.4f}")
print(f"  偏度: {skew_pos:.4f}")

# 测试 8: 不同的 tau 值（峰度参数）
print("\n8. 测试不同的 tau 值（峰度参数）")

# tau 小 (轻尾)
tau_small = 0.5
samples_tau_small = rSHASH(key, n_samples, mu, sigma, nu, tau_small)
from scipy.stats import kurtosis
kurt_small = kurtosis(np.array(samples_tau_small))
print(f"tau={tau_small} (轻尾):")
print(f"  峰度: {kurt_small:.4f}")

# tau 大 (重尾)
tau_large = 2.0
samples_tau_large = rSHASH(key, n_samples, mu, sigma, nu, tau_large)
kurt_large = kurtosis(np.array(samples_tau_large))
print(f"tau={tau_large} (重尾):")
print(f"  峰度: {kurt_large:.4f}")

# 测试 9: 验证密度积分
print("\n9. 验证密度积分（数值积分）")
from scipy.integrate import quad

def integrand(x):
    return float(dSHASH(jnp.array([x]), mu, sigma, nu, tau)[0])

# 积分范围：从一个大负值到一个大正值
integral, error = quad(integrand, -10.0, 10.0)
print(f"密度函数积分: {integral:.6f} (应该接近 1.0)")
print(f"积分误差: {error:.6e}")

print("\n" + "="*70)
print("✅ 所有测试完成！")
print("="*70)
