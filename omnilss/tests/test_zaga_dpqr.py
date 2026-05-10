"""
测试 ZAGA 和 ZAIG 的 dpqr 函数

这个脚本测试新实现的 ZAGA 和 ZAIG 分布的 d/p/q/r 函数。
"""

import sys
sys.path.insert(0, 'omnilss/src')

import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
from omnilss.dpqr_functions import dZAGA, pZAGA, qZAGA, rZAGA, dZAIG, pZAIG, qZAIG, rZAIG

print("="*70)
print("测试 ZAGA 分布的 dpqr 函数")
print("="*70)

# 测试参数
mu = 2.0
sigma = 0.5
nu = 0.2  # 20% 的零

print(f"\n参数: mu={mu}, sigma={sigma}, nu={nu}")

# 测试 1: 密度函数 (d)
print("\n1. 测试密度函数 dZAGA")
x_values = jnp.array([0.0, 0.5, 1.0, 2.0, 5.0])
densities = dZAGA(x_values, mu, sigma, nu)
print(f"x = {x_values}")
print(f"d(x) = {densities}")
print(f"✓ 在 x=0 处密度 = {densities[0]:.6f} (应该接近 nu={nu})")

# 测试 2: CDF 函数 (p)
print("\n2. 测试 CDF 函数 pZAGA")
q_values = jnp.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0])
cdfs = pZAGA(q_values, mu, sigma, nu)
print(f"q = {q_values}")
print(f"P(X <= q) = {cdfs}")
print(f"✓ 在 q=0 处 CDF = {cdfs[0]:.6f} (应该 = nu={nu})")
print(f"✓ CDF 单调递增: {np.all(np.diff(cdfs) >= 0)}")

# 测试 3: 分位数函数 (q)
print("\n3. 测试分位数函数 qZAGA")
p_values = jnp.array([0.1, 0.2, 0.5, 0.75, 0.9, 0.95])
quantiles = qZAGA(p_values, mu, sigma, nu)
print(f"p = {p_values}")
print(f"q(p) = {quantiles}")
print(f"✓ p <= nu 时分位数为 0: {quantiles[0]:.6f} (p={p_values[0]}, nu={nu})")
print(f"✓ p = nu 时分位数为 0: {quantiles[1]:.6f} (p={p_values[1]}, nu={nu})")

# 测试 4: p 和 q 的一致性
print("\n4. 测试 p 和 q 的一致性")
test_q = jnp.array([0.0, 1.0, 2.0, 3.0, 5.0])
p_from_q = pZAGA(test_q, mu, sigma, nu)
q_from_p = qZAGA(p_from_q, mu, sigma, nu)
print(f"原始 q: {test_q}")
print(f"p = P(X <= q): {p_from_q}")
print(f"q' = q(p): {q_from_p}")
print(f"差异: {jnp.abs(test_q - q_from_p)}")
print(f"✓ 一致性检查: max|q - q'| = {jnp.max(jnp.abs(test_q - q_from_p)):.6e}")

# 测试 5: 随机数生成 (r)
print("\n5. 测试随机数生成 rZAGA")
key = jrandom.PRNGKey(42)
n_samples = 1000
samples = rZAGA(key, n_samples, mu, sigma, nu)
print(f"生成 {n_samples} 个样本")
print(f"零的比例: {jnp.mean(samples == 0.0):.4f} (应该接近 nu={nu})")
print(f"非零样本的均值: {jnp.mean(samples[samples > 0]):.4f} (应该接近 mu={mu})")
print(f"非零样本的标准差: {jnp.std(samples[samples > 0]):.4f}")
print(f"样本范围: [{jnp.min(samples):.4f}, {jnp.max(samples):.4f}]")

# 测试 6: 边界情况
print("\n6. 测试边界情况")
# nu 接近 0 (几乎没有零)
nu_small = 0.01
samples_small_nu = rZAGA(key, n_samples, mu, sigma, nu_small)
print(f"nu={nu_small}: 零的比例 = {jnp.mean(samples_small_nu == 0.0):.4f}")

# nu 接近 1 (几乎全是零)
nu_large = 0.99
samples_large_nu = rZAGA(key, n_samples, mu, sigma, nu_large)
print(f"nu={nu_large}: 零的比例 = {jnp.mean(samples_large_nu == 0.0):.4f}")

print("\n" + "="*70)
print("测试 ZAIG 分布的 dpqr 函数")
print("="*70)

print(f"\n参数: mu={mu}, sigma={sigma}, nu={nu}")

# 测试 ZAIG
print("\n1. 测试密度函数 dZAIG")
densities_zaig = dZAIG(x_values, mu, sigma, nu)
print(f"x = {x_values}")
print(f"d(x) = {densities_zaig}")

print("\n2. 测试 CDF 函数 pZAIG")
cdfs_zaig = pZAIG(q_values, mu, sigma, nu)
print(f"q = {q_values}")
print(f"P(X <= q) = {cdfs_zaig}")

print("\n3. 测试分位数函数 qZAIG")
quantiles_zaig = qZAIG(p_values, mu, sigma, nu)
print(f"p = {p_values}")
print(f"q(p) = {quantiles_zaig}")

print("\n4. 测试随机数生成 rZAIG")
samples_zaig = rZAIG(key, n_samples, mu, sigma, nu)
print(f"生成 {n_samples} 个样本")
print(f"零的比例: {jnp.mean(samples_zaig == 0.0):.4f} (应该接近 nu={nu})")
print(f"非零样本的均值: {jnp.mean(samples_zaig[samples_zaig > 0]):.4f} (应该接近 mu={mu})")

print("\n" + "="*70)
print("✅ 所有测试完成！")
print("="*70)
