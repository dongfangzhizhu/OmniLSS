"""
演示 verbose 参数的使用

这个脚本展示了如何使用 gamlss() 函数的 verbose 参数来查看详细的拟合进度。
"""

import numpy as np
import sys
sys.path.insert(0, 'omnilss/src')

from omnilss import gamlss

# 设置随机种子以获得可重复的结果
np.random.seed(42)

print("="*70)
print("演示 gamlss() 函数的 verbose 参数")
print("="*70)

# 示例 1: 正态分布（NO），小数据集
print("\n" + "="*70)
print("示例 1: 正态分布（NO），n=100")
print("="*70)

n = 100
x = np.random.randn(n)
y = 2 + 3 * x + np.random.randn(n) * 0.5

data1 = {"y": y, "x": x}

print("\n1.1 verbose=False（默认，无详细输出）")
print("-" * 70)
model1 = gamlss("y ~ x", family="NO", data=data1, verbose=False)
print(f"✓ 拟合完成，deviance = {model1.g_dev:.4f}")

print("\n1.2 verbose=True（详细输出）")
print("-" * 70)
model2 = gamlss("y ~ x", family="NO", data=data1, verbose=True)

# 示例 2: Gamma 分布（GA），中等数据集
print("\n" + "="*70)
print("示例 2: Gamma 分布（GA），n=500")
print("="*70)

n = 500
x = np.random.randn(n)
mu = np.exp(1 + 0.5 * x)
y = np.random.gamma(shape=4, scale=mu/4)

data2 = {"y": y, "x": x}

print("\nverbose=True（详细输出）")
print("-" * 70)
model3 = gamlss("y ~ x", family="GA", data=data2, verbose=True)

# 示例 3: Poisson 分布（PO），大数据集
print("\n" + "="*70)
print("示例 3: Poisson 分布（PO），n=1000")
print("="*70)

n = 1000
x = np.random.randn(n)
mu = np.exp(1 + 0.3 * x)
y = np.random.poisson(mu)

data3 = {"y": y, "x": x}

print("\nverbose=True（详细输出）")
print("-" * 70)
model4 = gamlss("y ~ x", family="PO", data=data3, verbose=True)

# 总结
print("\n" + "="*70)
print("总结")
print("="*70)
print("\nverbose 参数的优势：")
print("1. 实时查看拟合进度")
print("2. 监控每次迭代的 deviance 变化")
print("3. 查看每次迭代的时间消耗")
print("4. 了解收敛状态")
print("5. 计算吞吐量（observations/second）")
print("\n推荐使用场景：")
print("- 大数据集拟合（n > 1000）")
print("- 复杂分布（BE, ZAGA, BEINF 等）")
print("- 调试收敛问题")
print("- 性能分析")
print("\n" + "="*70)
