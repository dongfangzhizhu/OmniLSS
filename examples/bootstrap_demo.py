"""
Bootstrap Methods Demonstration
================================

This example demonstrates the efficient bootstrap implementation using
JAX vectorization and parallelization.

Key Features:
- Non-parametric bootstrap (most robust)
- Parametric bootstrap (more efficient when model is correct)
- Residual bootstrap (for heteroscedastic errors)
- Confidence intervals and uncertainty quantification
- Visualization of bootstrap distributions
"""

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from omnilss.bootstrap import (
    nonparametric_bootstrap,
    parametric_bootstrap,
    residual_bootstrap,
)

# Set random seeds
np.random.seed(42)
key = jax.random.PRNGKey(42)

print("=" * 70)
print("Bootstrap Methods Demonstration")
print("=" * 70)

# ============================================================================
# 1. Generate synthetic data
# ============================================================================
print("\n1. Generating synthetic data...")

n = 200
x = np.linspace(0, 10, n)

# True model: y = 2 + 3*x + noise
true_intercept = 2.0
true_slope = 3.0
true_sigma = 1.5

y_true = true_intercept + true_slope * x
noise = np.random.normal(0, true_sigma, n)
y = y_true + noise

print(f"   - Sample size: {n}")
print(f"   - True intercept: {true_intercept}")
print(f"   - True slope: {true_slope}")
print(f"   - True error std: {true_sigma}")

# ============================================================================
# 2. Define fit function (simple OLS)
# ============================================================================
print("\n2. Defining fit function...")

def fit_ols(data):
    """Ordinary least squares fit."""
    X = jnp.column_stack([jnp.ones(len(data['x'])), data['x']])
    y = data['y']
    
    # Solve normal equations: (X'X)beta = X'y
    XtX = X.T @ X
    Xty = X.T @ y
    beta = jnp.linalg.solve(XtX, Xty)
    
    return beta

# Prepare data
data = {
    'y': jnp.array(y),
    'x': jnp.array(x)
}

# Initial fit
initial_fit = fit_ols(data)
print(f"   - Initial intercept estimate: {initial_fit[0]:.4f}")
print(f"   - Initial slope estimate: {initial_fit[1]:.4f}")

# ============================================================================
# 3. Non-parametric Bootstrap
# ============================================================================
print("\n3. Running non-parametric bootstrap...")
print("   (This is the most robust method)")

key1, key = jax.random.split(key)

result_nonparam = nonparametric_bootstrap(
    fit_ols, data, key1, n_boots=1000, alpha=0.05, parallel=True
)

print(f"\n{result_nonparam.summary()}")

# ============================================================================
# 4. Parametric Bootstrap
# ============================================================================
print("\n4. Running parametric bootstrap...")
print("   (Assumes normal errors)")

# Define sampling function
def sample_normal(params, k):
    """Generate samples from fitted normal model."""
    intercept, slope = params
    X = jnp.column_stack([jnp.ones(n), jnp.array(x)])
    mu = X @ params
    
    # Estimate sigma from residuals
    fitted = X @ params
    residuals = data['y'] - fitted
    sigma = jnp.std(residuals)
    
    # Generate new y
    return mu + jax.random.normal(k, shape=(n,)) * sigma

key2, key = jax.random.split(key)

result_param = parametric_bootstrap(
    fit_ols, sample_normal, data, initial_fit,
    key2, n_boots=1000, alpha=0.05, parallel=True
)

print(f"\n{result_param.summary()}")

# ============================================================================
# 5. Residual Bootstrap
# ============================================================================
print("\n5. Running residual bootstrap...")
print("   (Resamples residuals)")

# Compute fitted values and residuals
X = jnp.column_stack([jnp.ones(n), data['x']])
fitted_values = X @ initial_fit
residuals = data['y'] - fitted_values

key3, key = jax.random.split(key)

result_residual = residual_bootstrap(
    fit_ols, data, fitted_values, residuals,
    key3, n_boots=1000, alpha=0.05, parallel=True
)

print(f"\n{result_residual.summary()}")

# ============================================================================
# 6. Visualize Results
# ============================================================================
print("\n6. Creating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Data and fits
ax = axes[0, 0]
ax.scatter(x, y, alpha=0.3, s=20, label='Data', color='gray')
ax.plot(x, y_true, 'k-', linewidth=2, label='True model', alpha=0.7)

# Plot bootstrap fits (sample 100 random fits)
for i in np.random.choice(1000, 100, replace=False):
    beta = result_nonparam.coefficients[i]
    y_fit = beta[0] + beta[1] * x
    ax.plot(x, y_fit, 'b-', alpha=0.02)

# Plot mean fit
mean_beta = result_nonparam.mean
y_mean = mean_beta[0] + mean_beta[1] * x
ax.plot(x, y_mean, 'r-', linewidth=2, label='Bootstrap mean')

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('Non-parametric Bootstrap Fits', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 2: Intercept distribution (non-parametric)
ax = axes[0, 1]
intercepts = result_nonparam.coefficients[:, 0]
ax.hist(intercepts, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
ax.axvline(true_intercept, color='red', linestyle='--', linewidth=2, label='True value')
ax.axvline(result_nonparam.mean[0], color='green', linestyle='-', linewidth=2, label='Bootstrap mean')
ax.axvline(result_nonparam.lower_ci[0], color='orange', linestyle=':', linewidth=2, label='95% CI')
ax.axvline(result_nonparam.upper_ci[0], color='orange', linestyle=':', linewidth=2)
ax.set_xlabel('Intercept', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Intercept Distribution (Non-parametric)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 3: Slope distribution (non-parametric)
ax = axes[0, 2]
slopes = result_nonparam.coefficients[:, 1]
ax.hist(slopes, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
ax.axvline(true_slope, color='red', linestyle='--', linewidth=2, label='True value')
ax.axvline(result_nonparam.mean[1], color='green', linestyle='-', linewidth=2, label='Bootstrap mean')
ax.axvline(result_nonparam.lower_ci[1], color='orange', linestyle=':', linewidth=2, label='95% CI')
ax.axvline(result_nonparam.upper_ci[1], color='orange', linestyle=':', linewidth=2)
ax.set_xlabel('Slope', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Slope Distribution (Non-parametric)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 4: Compare methods - Intercept
ax = axes[1, 0]
methods = ['Non-param', 'Parametric', 'Residual']
intercept_means = [
    result_nonparam.mean[0],
    result_param.mean[0],
    result_residual.mean[0]
]
intercept_stds = [
    result_nonparam.std[0],
    result_param.std[0],
    result_residual.std[0]
]
intercept_lower = [
    result_nonparam.lower_ci[0],
    result_param.lower_ci[0],
    result_residual.lower_ci[0]
]
intercept_upper = [
    result_nonparam.upper_ci[0],
    result_param.upper_ci[0],
    result_residual.upper_ci[0]
]

x_pos = np.arange(len(methods))
ax.errorbar(x_pos, intercept_means, 
            yerr=[np.array(intercept_means) - np.array(intercept_lower),
                  np.array(intercept_upper) - np.array(intercept_means)],
            fmt='o', markersize=10, capsize=10, capthick=2, linewidth=2)
ax.axhline(true_intercept, color='red', linestyle='--', linewidth=2, label='True value')
ax.set_xticks(x_pos)
ax.set_xticklabels(methods)
ax.set_ylabel('Intercept', fontsize=12)
ax.set_title('Intercept Estimates (All Methods)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Plot 5: Compare methods - Slope
ax = axes[1, 1]
slope_means = [
    result_nonparam.mean[1],
    result_param.mean[1],
    result_residual.mean[1]
]
slope_stds = [
    result_nonparam.std[1],
    result_param.std[1],
    result_residual.std[1]
]
slope_lower = [
    result_nonparam.lower_ci[1],
    result_param.lower_ci[1],
    result_residual.lower_ci[1]
]
slope_upper = [
    result_nonparam.upper_ci[1],
    result_param.upper_ci[1],
    result_residual.upper_ci[1]
]

ax.errorbar(x_pos, slope_means,
            yerr=[np.array(slope_means) - np.array(slope_lower),
                  np.array(slope_upper) - np.array(slope_means)],
            fmt='o', markersize=10, capsize=10, capthick=2, linewidth=2)
ax.axhline(true_slope, color='red', linestyle='--', linewidth=2, label='True value')
ax.set_xticks(x_pos)
ax.set_xticklabels(methods)
ax.set_ylabel('Slope', fontsize=12)
ax.set_title('Slope Estimates (All Methods)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Plot 6: Standard errors comparison
ax = axes[1, 2]
width = 0.35
x_pos = np.arange(2)

intercept_ses = [result_nonparam.std[0], result_param.std[0], result_residual.std[0]]
slope_ses = [result_nonparam.std[1], result_param.std[1], result_residual.std[1]]

x_methods = np.arange(len(methods))
width = 0.35

ax.bar(x_methods - width/2, intercept_ses, width, label='Intercept SE', alpha=0.8)
ax.bar(x_methods + width/2, slope_ses, width, label='Slope SE', alpha=0.8)

ax.set_xticks(x_methods)
ax.set_xticklabels(methods)
ax.set_ylabel('Standard Error', fontsize=12)
ax.set_title('Standard Errors Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('examples/bootstrap_demo.png', dpi=150, bbox_inches='tight')
print("   - Saved plot to 'examples/bootstrap_demo.png'")

# ============================================================================
# 7. Performance Comparison
# ============================================================================
print("\n7. Performance Metrics:")
print("-" * 70)

print(f"\n   Non-parametric Bootstrap:")
print(f"      - Success rate: {result_nonparam.n_successful/result_nonparam.n_boots*100:.1f}%")
print(f"      - Intercept: {result_nonparam.mean[0]:.4f} ± {result_nonparam.std[0]:.4f}")
print(f"      - Slope: {result_nonparam.mean[1]:.4f} ± {result_nonparam.std[1]:.4f}")
print(f"      - Intercept CI: [{result_nonparam.lower_ci[0]:.4f}, {result_nonparam.upper_ci[0]:.4f}]")
print(f"      - Slope CI: [{result_nonparam.lower_ci[1]:.4f}, {result_nonparam.upper_ci[1]:.4f}]")

print(f"\n   Parametric Bootstrap:")
print(f"      - Success rate: {result_param.n_successful/result_param.n_boots*100:.1f}%")
print(f"      - Intercept: {result_param.mean[0]:.4f} ± {result_param.std[0]:.4f}")
print(f"      - Slope: {result_param.mean[1]:.4f} ± {result_param.std[1]:.4f}")
print(f"      - Intercept CI: [{result_param.lower_ci[0]:.4f}, {result_param.upper_ci[0]:.4f}]")
print(f"      - Slope CI: [{result_param.lower_ci[1]:.4f}, {result_param.upper_ci[1]:.4f}]")

print(f"\n   Residual Bootstrap:")
print(f"      - Success rate: {result_residual.n_successful/result_residual.n_boots*100:.1f}%")
print(f"      - Intercept: {result_residual.mean[0]:.4f} ± {result_residual.std[0]:.4f}")
print(f"      - Slope: {result_residual.mean[1]:.4f} ± {result_residual.std[1]:.4f}")
print(f"      - Intercept CI: [{result_residual.lower_ci[0]:.4f}, {result_residual.upper_ci[0]:.4f}]")
print(f"      - Slope CI: [{result_residual.lower_ci[1]:.4f}, {result_residual.upper_ci[1]:.4f}]")

# ============================================================================
# 8. Key Insights
# ============================================================================
print("\n" + "=" * 70)
print("KEY INSIGHTS")
print("=" * 70)
print("""
1. JAX Vectorization Benefits:
   - All 1000 bootstrap iterations run in parallel
   - Much faster than R's sequential loops
   - Efficient memory usage with vmap

2. Bootstrap Method Comparison:
   - Non-parametric: Most robust, no distributional assumptions
   - Parametric: More efficient when model is correct
   - Residual: Good for heteroscedastic errors

3. Confidence Intervals:
   - All methods give similar results for this simple case
   - CI width reflects estimation uncertainty
   - True parameters should be within 95% CI

4. Practical Recommendations:
   - Use non-parametric for robustness
   - Use parametric when confident in model
   - Check convergence (n_successful should be high)
   - Increase n_boots for more stable CI

5. JAX Advantages:
   - Automatic differentiation ready
   - GPU acceleration possible
   - Reproducible with fixed random seeds
   - Clean functional programming style
""")

print("\n" + "=" * 70)
print("Demo completed successfully!")
print("=" * 70)

plt.show()
