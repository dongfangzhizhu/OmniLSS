"""Test BB (Beta-Binomial) distribution dpqr functions."""

import jax.numpy as jnp
import jax.random as jrandom
from omnilss.dpqr_functions import dBB, pBB, qBB, rBB

print("=" * 70)
print("测试 BB (Beta-Binomial) 分布的 dpqr 函数")
print("=" * 70)

# Test parameters
mu = 0.5
sigma = 0.5
bd = 10

print(f"\n参数: mu={mu}, sigma={sigma}, bd={bd}")

# 1. Test density function
print("\n1. 测试密度函数 dBB")
x = jnp.array([0, 2, 5, 8, 10])
d = dBB(x, mu, sigma, bd)
print(f"x = {x}")
print(f"d(x) = {d}")
print(f"✓ 所有密度值为正: {jnp.all(d > 0)}")

# 2. Test PMF sums to 1
print("\n2. 验证 PMF 总和")
x_all = jnp.arange(0, bd + 1)
pmf_all = dBB(x_all, mu, sigma, bd)
pmf_sum = jnp.sum(pmf_all)
print(f"PMF 总和: {pmf_sum:.6f} (应该接近 1.0)")
print(f"✓ PMF 归一化: {jnp.abs(pmf_sum - 1.0) < 1e-6}")

# 3. Test CDF
print("\n3. 测试 CDF pBB")
q = jnp.array([0, 2, 5, 8, 10])
p = pBB(q, mu, sigma, bd)
print(f"q = {q}")
print(f"p(q) = {p}")
print(f"✓ CDF 在 [0, 1] 范围内: {jnp.all((p >= 0) & (p <= 1))}")
print(f"✓ CDF 单调递增: {jnp.all(jnp.diff(p) >= 0)}")
print(f"✓ CDF(bd) = 1.0: {jnp.abs(p[-1] - 1.0) < 1e-6}")

# 4. Test quantile function
print("\n4. 测试分位数函数 qBB")
p_test = jnp.array([0.1, 0.25, 0.5, 0.75, 0.9])
q_result = qBB(p_test, mu, sigma, bd)
print(f"p = {p_test}")
print(f"q(p) = {q_result}")
print(f"✓ 分位数单调递增: {jnp.all(jnp.diff(q_result) >= 0)}")

# 5. Test p-q consistency
print("\n5. 测试 p 和 q 的一致性")
x_test = jnp.array([0, 2, 4, 6, 8, 10])
p_from_x = pBB(x_test, mu, sigma, bd)
x_back = qBB(p_from_x, mu, sigma, bd)
print(f"x_test = {x_test}")
print(f"x_back = {x_back}")
print(f"✓ 一致性: {jnp.allclose(x_test, x_back)}")

# 6. Test random generation
print("\n6. 测试随机数生成 rBB")
key = jrandom.PRNGKey(42)
n = 1000
samples = rBB(key, n, mu, sigma, bd)
print(f"生成 {n} 个样本")
print(f"样本均值: {jnp.mean(samples):.4f} (理论值: {mu * bd:.4f})")
print(f"样本标准差: {jnp.std(samples):.4f}")
print(f"样本最小值: {jnp.min(samples):.0f}")
print(f"样本最大值: {jnp.max(samples):.0f}")
print(f"✓ 所有样本在 [0, bd] 范围内: {jnp.all((samples >= 0) & (samples <= bd))}")

# 7. Test different parameter values
print("\n7. 测试不同参数值")

# Case 1: Different sigma (overdispersion)
print("\n  Case 1: 不同的 sigma 值（过度离散）")
sigma_vals = [0.1, 0.5, 1.0]
x1 = 5
for s in sigma_vals:
    d1 = dBB(x1, mu, s, bd)
    print(f"  sigma={s}: d({x1}) = {d1:.6f}")

# Case 2: Different mu
print("\n  Case 2: 不同的 mu 值")
mu_vals = [0.3, 0.5, 0.7]
x2 = 5
for m in mu_vals:
    d2 = dBB(x2, m, sigma, bd)
    print(f"  mu={m}: d({x2}) = {d2:.6f}")

print("\n" + "=" * 70)
print("✅ BB 分布测试完成！")
print("=" * 70)
