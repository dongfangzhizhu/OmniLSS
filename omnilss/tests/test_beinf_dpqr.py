"""
测试 BEINF 的 dpqr 函数

这个脚本测试新实现的 BEINF (Beta Inflated) 分布的 d/p/q/r 函数。
"""

import sys
sys.path.insert(0, 'omnilss/src')

import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
from omnilss.dpqr_functions import dBEINF, pBEINF, qBEINF, rBEINF

print("="*70)
print("测试 BEINF 分布的 dpqr 函数")
print("="*70)

# 测试参数
mu = 0.5      # 连续部分的均值
sigma = 0.2   # 离散参数
nu = 0.1      # 在 0 处的膨胀参数
tau = 0.15    # 在 1 处的膨胀参数

print(f"\n参数: mu={mu}, sigma={sigma}, nu={nu}, tau={tau}")

# 计算理论概率
total = 1.0 + nu + tau
p0_theory = nu / total
p1_theory = tau / total
p_cont_theory = 1.0 / total

print(f"\n理论概率:")
print(f"  P(Y = 0) = {p0_theory:.4f}")
print(f"  P(Y = 1) = {p1_theory:.4f}")
print(f"  P(0 < Y < 1) = {p_cont_theory:.4f}")

# 测试 1: 密度函数 (d)
print("\n1. 测试密度函数 dBEINF")
x_values = jnp.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
densities = dBEINF(x_values, mu, sigma, nu, tau)
print(f"x = {x_values}")
print(f"d(x) = {densities}")
print(f"✓ 在 x=0 处密度 = {densities[0]:.6f} (理论值 = {p0_theory:.6f})")
print(f"✓ 在 x=1 处密度 = {densities[-1]:.6f} (理论值 = {p1_theory:.6f})")

# 测试 2: CDF 函数 (p)
print("\n2. 测试 CDF 函数 pBEINF")
q_values = jnp.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
cdfs = pBEINF(q_values, mu, sigma, nu, tau)
print(f"q = {q_values}")
print(f"P(X <= q) = {cdfs}")
print(f"✓ 在 q=0 处 CDF = {cdfs[0]:.6f} (理论值 = {p0_theory:.6f})")
print(f"✓ 在 q=1 处 CDF = {cdfs[-1]:.6f} (应该 = 1.0)")
print(f"✓ CDF 单调递增: {np.all(np.diff(cdfs) >= -1e-10)}")

# 测试 3: 分位数函数 (q)
print("\n3. 测试分位数函数 qBEINF")
p_values = jnp.array([0.05, 0.08, 0.2, 0.5, 0.75, 0.9, 0.95])
quantiles = qBEINF(p_values, mu, sigma, nu, tau)
print(f"p = {p_values}")
print(f"q(p) = {quantiles}")
print(f"✓ p <= p0 时分位数为 0:")
for i, (p, q) in enumerate(zip(p_values, quantiles)):
    if p <= p0_theory:
        print(f"    p={p:.4f} (p0={p0_theory:.4f}): q={q:.6f}")
print(f"✓ p >= p0 + p_cont 时分位数为 1:")
for i, (p, q) in enumerate(zip(p_values, quantiles)):
    if p >= p0_theory + p_cont_theory:
        print(f"    p={p:.4f} (p0+p_cont={p0_theory + p_cont_theory:.4f}): q={q:.6f}")

# 测试 4: p 和 q 的一致性
print("\n4. 测试 p 和 q 的一致性")
test_q = jnp.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
p_from_q = pBEINF(test_q, mu, sigma, nu, tau)
q_from_p = qBEINF(p_from_q, mu, sigma, nu, tau)
print(f"原始 q: {test_q}")
print(f"p = P(X <= q): {p_from_q}")
print(f"q' = q(p): {q_from_p}")
print(f"差异: {jnp.abs(test_q - q_from_p)}")
print(f"✓ 一致性检查: max|q - q'| = {jnp.max(jnp.abs(test_q - q_from_p)):.6e}")

# 测试 5: 随机数生成 (r)
print("\n5. 测试随机数生成 rBEINF")
key = jrandom.PRNGKey(42)
n_samples = 10000
samples = rBEINF(key, n_samples, mu, sigma, nu, tau)
print(f"生成 {n_samples} 个样本")

# 统计零和一的比例
eps = 1e-10
n_zeros = jnp.sum(samples <= eps)
n_ones = jnp.sum(samples >= 1.0 - eps)
n_cont = n_samples - n_zeros - n_ones

p0_empirical = n_zeros / n_samples
p1_empirical = n_ones / n_samples
p_cont_empirical = n_cont / n_samples

print(f"\n实证概率:")
print(f"  P(Y = 0) = {p0_empirical:.4f} (理论值 = {p0_theory:.4f})")
print(f"  P(Y = 1) = {p1_empirical:.4f} (理论值 = {p1_theory:.4f})")
print(f"  P(0 < Y < 1) = {p_cont_empirical:.4f} (理论值 = {p_cont_theory:.4f})")

# 连续部分的统计
cont_samples = samples[(samples > eps) & (samples < 1.0 - eps)]
if len(cont_samples) > 0:
    print(f"\n连续部分统计 (0 < Y < 1):")
    print(f"  样本数: {len(cont_samples)}")
    print(f"  均值: {jnp.mean(cont_samples):.4f} (应该接近 mu={mu})")
    print(f"  标准差: {jnp.std(cont_samples):.4f}")
    print(f"  范围: [{jnp.min(cont_samples):.4f}, {jnp.max(cont_samples):.4f}]")

# 测试 6: 边界情况
print("\n6. 测试边界情况")

# nu 和 tau 都很小 (几乎是纯 Beta)
nu_small = 0.01
tau_small = 0.01
samples_small = rBEINF(key, n_samples, mu, sigma, nu_small, tau_small)
n_zeros_small = jnp.sum(samples_small <= eps)
n_ones_small = jnp.sum(samples_small >= 1.0 - eps)
print(f"nu={nu_small}, tau={tau_small}:")
print(f"  零的比例 = {n_zeros_small / n_samples:.4f}")
print(f"  一的比例 = {n_ones_small / n_samples:.4f}")

# nu 很大 (大量的零)
nu_large = 0.8
tau_small = 0.1
samples_large_nu = rBEINF(key, n_samples, mu, sigma, nu_large, tau_small)
n_zeros_large = jnp.sum(samples_large_nu <= eps)
n_ones_large = jnp.sum(samples_large_nu >= 1.0 - eps)
total_large = 1.0 + nu_large + tau_small
p0_large = nu_large / total_large
print(f"nu={nu_large}, tau={tau_small}:")
print(f"  零的比例 = {n_zeros_large / n_samples:.4f} (理论值 = {p0_large:.4f})")
print(f"  一的比例 = {n_ones_large / n_samples:.4f}")

# tau 很大 (大量的一)
nu_small = 0.1
tau_large = 0.8
samples_large_tau = rBEINF(key, n_samples, mu, sigma, nu_small, tau_large)
n_zeros_tau = jnp.sum(samples_large_tau <= eps)
n_ones_tau = jnp.sum(samples_large_tau >= 1.0 - eps)
total_tau = 1.0 + nu_small + tau_large
p1_large = tau_large / total_tau
print(f"nu={nu_small}, tau={tau_large}:")
print(f"  零的比例 = {n_zeros_tau / n_samples:.4f}")
print(f"  一的比例 = {n_ones_tau / n_samples:.4f} (理论值 = {p1_large:.4f})")

# 测试 7: 不同的 mu 和 sigma
print("\n7. 测试不同的 mu 和 sigma")

# mu 接近 0
mu_low = 0.2
samples_mu_low = rBEINF(key, n_samples, mu_low, sigma, nu, tau)
cont_samples_mu_low = samples_mu_low[(samples_mu_low > eps) & (samples_mu_low < 1.0 - eps)]
if len(cont_samples_mu_low) > 0:
    print(f"mu={mu_low}: 连续部分均值 = {jnp.mean(cont_samples_mu_low):.4f}")

# mu 接近 1
mu_high = 0.8
samples_mu_high = rBEINF(key, n_samples, mu_high, sigma, nu, tau)
cont_samples_mu_high = samples_mu_high[(samples_mu_high > eps) & (samples_mu_high < 1.0 - eps)]
if len(cont_samples_mu_high) > 0:
    print(f"mu={mu_high}: 连续部分均值 = {jnp.mean(cont_samples_mu_high):.4f}")

# sigma 很小 (低方差)
sigma_small = 0.05
samples_sigma_small = rBEINF(key, n_samples, mu, sigma_small, nu, tau)
cont_samples_sigma_small = samples_sigma_small[(samples_sigma_small > eps) & (samples_sigma_small < 1.0 - eps)]
if len(cont_samples_sigma_small) > 0:
    print(f"sigma={sigma_small}: 连续部分标准差 = {jnp.std(cont_samples_sigma_small):.4f}")

# sigma 较大 (高方差)
sigma_large = 0.4
samples_sigma_large = rBEINF(key, n_samples, mu, sigma_large, nu, tau)
cont_samples_sigma_large = samples_sigma_large[(samples_sigma_large > eps) & (samples_sigma_large < 1.0 - eps)]
if len(cont_samples_sigma_large) > 0:
    print(f"sigma={sigma_large}: 连续部分标准差 = {jnp.std(cont_samples_sigma_large):.4f}")

print("\n" + "="*70)
print("✅ 所有测试完成！")
print("="*70)
