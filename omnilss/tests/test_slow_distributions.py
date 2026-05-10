"""Test slow distributions with verbose output."""
import sys
sys.path.insert(0, 'omnilss/src')

import numpy as np
import pandas as pd
from omnilss import gamlss
from omnilss.fitting_jit import should_use_jit

# Test distributions
distributions = {
    'BE': 'Beta',
    'ZAGA': 'Zero-Adjusted Gamma',
    'BEINF': 'Beta Inflated',
    'ZIP': 'Zero-Inflated Poisson',
}

print("="*70)
print("Testing JIT Optimization Detection for Slow Distributions")
print("="*70)

for dist_name, dist_desc in distributions.items():
    # Check if JIT would be used
    use_jit_small = should_use_jit(dist_name, 50)
    use_jit_medium = should_use_jit(dist_name, 500)
    use_jit_large = should_use_jit(dist_name, 5000)
    
    print(f"\n{dist_name} ({dist_desc}):")
    print(f"  n=50:   JIT={'✓ Enabled' if use_jit_small else '✗ Disabled'}")
    print(f"  n=500:  JIT={'✓ Enabled' if use_jit_medium else '✗ Disabled'}")
    print(f"  n=5000: JIT={'✓ Enabled' if use_jit_large else '✗ Disabled'}")

print("\n" + "="*70)
print("Testing BE distribution with verbose output")
print("="*70)

# Generate Beta data
np.random.seed(42)
n = 200
x1 = np.random.randn(n)
mu = 1 / (1 + np.exp(-(0.5 + 0.3 * x1)))
y = np.random.beta(mu * 10, (1 - mu) * 10)
data = pd.DataFrame({'y': y, 'x1': x1})

model = gamlss('y ~ x1', family='BE', data=data, verbose=True)

print(f'\n✓ BE model fitted successfully')
print(f'  Deviance: {model.g_dev:.4f}')
print(f'  Converged: {model.additional_slots.get("converged", False)}')
print(f'  Iterations: {model.additional_slots.get("cycles", 0)}')

print("\n" + "="*70)
print("Testing ZAGA distribution with verbose output")
print("="*70)

# Generate ZAGA data
np.random.seed(42)
n = 200
x1 = np.random.randn(n)

pi = 1 / (1 + np.exp(-(-1 + 0.3 * x1)))
mu = np.exp(1.5 + 0.5 * x1)
sigma = 0.6

y = np.zeros(n)
for i in range(n):
    if np.random.rand() < pi[i]:
        y[i] = 0
    else:
        shape = 1 / sigma**2
        y[i] = np.random.gamma(shape, mu[i] / shape)

data = pd.DataFrame({'y': y, 'x1': x1})

model = gamlss('y ~ x1', family='ZAGA', data=data, verbose=True)

print(f'\n✓ ZAGA model fitted successfully')
print(f'  Deviance: {model.g_dev:.4f}')
print(f'  Converged: {model.additional_slots.get("converged", False)}')
print(f'  Iterations: {model.additional_slots.get("cycles", 0)}')

print("\n" + "="*70)
print("✓ All tests completed!")
print("="*70)
