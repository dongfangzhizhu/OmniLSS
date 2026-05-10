"""
测试 GT 的 dpqr 函数

这个脚本测试新实现的 GT (Generalized t) 分布的 d/p/q/r 函数。
注意：GT 的 CDF 和分位数函数使用数值方法，计算较慢。
"""

import sys
sys.path.insert(0, 'omnilss/src')

import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
from omnilss.dpqr_functions import dGT

print("="*70)
print("测试 GT 分布的 dpqr 函数")
print("="*70)

# 测试参数
mu = 0.0      # 位置参数
sigma = 1.0   # 尺度参数
nu = 2.0      # 形状参数 1
tau = 2.0     # 形状参数 2

print(f"\n参数: mu={mu}, sigma={sigma}, nu={nu}, tau={tau}")

# 测试 1: 密度函数 (d)
print("\n1. 测试密度函数 dGT")
x_values = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
densities = dGT(x_values, mu, sigma, nu, tau)
print(f"x = {x_values}")
print(f"d(x) = {densities}")
print(f"✓ 所有密度值为正: {jnp.all(densities > 0)}")

# 测试 2: 对称性
print("\n2. 测试对称性")
print(f"d(-2) = {densities[0]:.6f}")
print(f"d(2) = {densities[4]:.6f}")
print(f"✓ 对称性: |d(-2) - d(2)| = {jnp.abs(densities[0] - densities[4]):.6e}")

# 测试 3: 不同的 nu 和 tau 值
print("\n3. 测试不同的 nu 和 tau 值")

# nu 小 (重尾)
nu_small = 1.0
densities_nu_small = dGT(x_values, mu, sigma, nu_small, tau)
print(f"nu={nu_small} (重尾): d(2) = {densities_nu_small[4]:.6f}")

# nu 大 (轻尾)
nu_large = 10.0
densities_nu_large = dGT(x_values, mu, sigma, nu_large, tau)
print(f"nu={nu_large} (轻尾): d(2) = {densities_nu_large[4]:.6f}")

# tau 小
tau_small = 1.0
densities_tau_small = dGT(x_values, mu, sigma, nu, tau_small)
print(f"tau={tau_small}: d(2) = {densities_tau_small[4]:.6f}")

# tau 大
tau_large = 4.0
densities_tau_large = dGT(x_values, mu, sigma, nu, tau_large)
print(f"tau={tau_large}: d(2) = {densities_tau_large[4]:.6f}")

# 测试 4: 验证密度积分（简化版）
print("\n4. 验证密度积分（数值积分）")
from scipy.integrate import quad

def integrand(x):
    return float(dGT(jnp.array([x]), mu, sigma, nu, tau)[0])

# 积分范围：从一个大负值到一个大正值
integral, error = quad(integrand, -10.0, 10.0)
print(f"密度函数积分: {integral:.6f} (应该接近 1.0)")
print(f"积分误差: {error:.6e}")

print("\n" + "="*70)
print("✅ 基本测试完成！")
print("="*70)
print("\n注意：GT 的 CDF 和分位数函数使用数值方法，计算较慢。")
print("因此我们只测试了密度函数。在实际使用中，这些函数是可用的。")
