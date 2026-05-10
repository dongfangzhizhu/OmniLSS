"""Test verbose parameter with complex distribution."""
import sys
sys.path.insert(0, 'omnilss/src')

import numpy as np
import pandas as pd
from omnilss import gamlss

# 创建 Gamma 分布测试数据
np.random.seed(42)
n = 500
x1 = np.random.randn(n)
mu = np.exp(1 + 0.5 * x1)
y = np.random.gamma(shape=2, scale=mu/2, size=n)
data = pd.DataFrame({'y': y, 'x1': x1})

print("="*70)
print("Testing Gamma distribution with verbose=True")
print("="*70)

model = gamlss('y ~ x1', family='GA', data=data, verbose=True)

print(f'\n✓ Model fitted successfully')
print(f'  Deviance: {model.g_dev:.4f}')
print(f'  Converged: {model.additional_slots.get("converged", False)}')
print(f'  Iterations: {model.additional_slots.get("cycles", 0)}')
print(f'  AIC: {model.additional_slots.get("aic", 0):.4f}')
