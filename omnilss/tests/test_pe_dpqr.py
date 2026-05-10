"""
测试 PE 的 dpqr 函数

这个脚本测试新实现的 PE (Power Exponential / Generalized Normal) 分布的 d/p/q/r 函数。
"""

import sys
sys.path.insert(0, 'omnilss/src')

import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
from omnilss.dpqr_functions import dPE, pPE, qPE, rPE

print("="*70)
print("测试 PE 分布的 dpqr 函数")
print("="*70)

# 测试参数
mu = 0.0      # 位置参数
sigma = 1.0   # 尺度参数
nu = 2.0      # 形状参数 (nu=2 是正态分布)

print(f"\n参数: mu={mu}, sigma={sigma}, nu={nu}")
print(f"注意: nu=2 时 PE 是正态分布")

# 测试 1: 密度函数 (d)
print("\n1. 测试密度函数 dPE")
x_values = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
densities = dPE(x_values, mu, sigma, nu)
print(f"x = {x_values}")
print(f"d(x) = {densities}")
print(f"✓ 所有密度值为正: {jnp.all(densities > 0)}")

# 测试 2: CDF 函数 (p)
print("\n2. 测试 CDF 函数 pPE")
q_values = jnp.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
cdfs = pPE(q_values, mu, sigma, nu)
print(f"q = {q_values}")
print(f"P(X <= q) = {cdfs}")
print(f"✓ CDF 在 [0, 1] 范围内: {jnp.all((cdfs >= 0) & (cdfs <= 1))}")
print(f"✓ CDF 单调递增: {np.all(np.diff(cdfs) >= -1e-10)}")

# 测试 3: 分位数函数 (q)
print("\n3. 测试分位数函数 qPE")
p_values = jnp.array([0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
quantiles = qPE(p_values, mu, sigma, nu)
print(f"p = {p_values}")
print(f"q(p) = {quantiles}")
print(f"✓ 分位数单调递增: {np.all(np.diff(quantiles) >= -1e-10)}")

# 测试 4: p 和 q 的一致性
print("\n4. 测试 p 和 q 的一致性")
test_q = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
p_from_q = pPE(test_q, mu, sigma, nu)
q_from_p = qPE(p_from_q, mu, sigma, nu)
print(f"原始 q: {test_q}")
print(f"p = P(X <= q): {p_from_q}")
print(f"q' = q(p): {q_from_p}")
print(f"差异: {jnp.abs(test_q - q_from_p)}")
print(f"✓ 一致性检查: max|q - q'| = {jnp.max(jnp.abs(test_q - q_from_p)):.6e}")

# 测试 5: 随机数生成 (r)
print("\n5. 测试随机数生成 rPE")
key = jrandom.PRNGKey(42)
n_samples = 10000
samples = rPE(key, n_samples, mu, sigma, nu)
print(f"生成 {n_samples} 个样本")
print(f"样本均值: {jnp.mean(samples):.4f} (应该接近 mu={mu})")
print(f"样本中位数: {jnp.median(samples):.4f}")
print(f"样本标准差: {jnp.std(samples):.4f}")
print(f"样本范围: [{jnp.min(samples):.4f}, {jnp.max(samples):.4f}]")

# 测试 6: 不同的 nu 值（形状参数）
print("\n6. 测试不同的 nu 值")

# nu = 1 (Laplace 分布)
nu_one = 1.0
samples_nu1 = rPE(key, n_samples, mu, sigma, nu_one)
print(f"nu={nu_one} (Laplace):")
print(f"  样本均值: {jnp.mean(samples_nu1):.4f}")
print(f"  样本标准差: {jnp.std(samples_nu1):.4f}")

# nu = 2 (Normal 分布)
nu_two = 2.0
samples_nu2 = rPE(key, n_samples, mu, sigma, nu_two)
print(f"nu={nu_two} (Normal):")
print(f"  样本均值: {jnp.mean(samples_nu2):.4f}")
print(f"  样本标准差: {jnp.std(samples_nu2):.4f}")

# nu = 4 (重尾)
nu_four = 4.0
samples_nu4 = rPE(key, n_samples, mu, sigma, nu_four)
print(f"nu={nu_four} (重尾):")
print(f"  样本均值: {jnp.mean(samples_nu4):.4f}")
print(f"  样本标准差: {jnp.std(samples_nu4):.4f}")

# 测试 7: 对称性
print("\n7. 测试对称性")
from scipy.stats import skew
sample_skew = skew(np.array(samples))
print(f"偏度: {sample_skew:.4f} (应该接近 0，因为 PE 是对称分布)")

# 测试 8: 验证密度积分
print("\n8. 验证密度积分（数值积分）")
from scipy.integrate import quad

def integrand(x):
    return float(dPE(jnp.array([x]), mu, sigma, nu)[0])

# 积分范围：从一个大负值到一个大正值
integral, error = quad(integrand, -10.0, 10.0)
print(f"密度函数积分: {integral:.6f} (应该接近 1.0)")
print(f"积分误差: {error:.6e}")

print("\n" + "="*70)
print("✅ 所有测试完成！")
print("="*70)
