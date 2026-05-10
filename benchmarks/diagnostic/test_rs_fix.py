"""Test the RS algorithm fix for NBI distribution."""

import numpy as np
import sys
sys.path.insert(0, 'omnilss/src')

from omnilss import gamlss

# Set random seed for reproducibility
np.random.seed(42)

# Generate test data (same as in R)
n = 100
x1 = np.random.normal(0, 1, n)
mu_true = np.exp(1 + 0.5 * x1)
sigma_true = 0.5
y = np.random.negative_binomial(n=1/sigma_true**2, p=1/(1 + mu_true * sigma_true**2), size=n)

data = {
    'y': y,
    'x1': x1
}

print("=" * 70)
print("Testing RS Algorithm Fix for NBI")
print("=" * 70)
print(f"Sample size: {n}")
print(f"Formula: y ~ x1")
print(f"Family: NBI")
print("=" * 70)

# Fit model with RS algorithm
model = gamlss(
    formula="y ~ x1",
    sigma_formula="~1",
    family="NBI",
    data=data,
    method="RS",
    verbose=True
)

print("\n" + "=" * 70)
print("Results:")
print("=" * 70)
print(f"Global Deviance: {model.g_dev:.6f}")
print(f"Iterations: {model.iter}")
print(f"Converged: {model.additional_slots.get('converged', False)}")
print("\nMu coefficients:")
print(model.coefficients['mu'])
print("\nSigma fitted value:")
print(f"  {np.exp(model.coefficients['sigma'][0]):.6f}")
print("=" * 70)

# Expected from R (approximately):
# Deviance: ~405.87
# Sigma: ~0.63
print("\nExpected from R:")
print("  Deviance: ~405.87")
print("  Sigma: ~0.63")
print("=" * 70)
