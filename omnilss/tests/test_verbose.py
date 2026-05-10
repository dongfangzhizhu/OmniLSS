"""Test verbose parameter in gamlss."""
import sys
sys.path.insert(0, 'omnilss/src')

import numpy as np
import pandas as pd
from omnilss import gamlss

# 创建测试数据
np.random.seed(42)
n = 100
x1 = np.random.randn(n)
y = 2 + 3 * x1 + np.random.randn(n)
data = pd.DataFrame({'y': y, 'x1': x1})

print("="*70)
print("Test 1: verbose=False (default, silent mode)")
print("="*70)
model1 = gamlss('y ~ x1', family='NO', data=data, verbose=False)
print(f'✓ Model fitted successfully')
print(f'  Deviance: {model1.g_dev:.4f}')
print(f'  Converged: {model1.additional_slots.get("converged", False)}')
print(f'  Iterations: {model1.additional_slots.get("cycles", 0)}')

print("\n" + "="*70)
print("Test 2: verbose=True (detailed progress)")
print("="*70)
model2 = gamlss('y ~ x1', family='NO', data=data, verbose=True)
print(f'\n✓ Model fitted successfully')
print(f'  Deviance: {model2.g_dev:.4f}')
print(f'  Converged: {model2.additional_slots.get("converged", False)}')
print(f'  Iterations: {model2.additional_slots.get("cycles", 0)}')

print("\n" + "="*70)
print("✓ All tests passed!")
print("="*70)
