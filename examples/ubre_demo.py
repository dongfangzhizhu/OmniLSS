"""
UBRE (Unbiased Risk Estimator) Demonstration
=============================================

This example demonstrates the use of UBRE for automatic smoothing parameter
selection in penalized regression splines.

UBRE is an alternative to GCV that requires knowledge (or estimation) of the
error variance σ². It's particularly useful when:
1. The error variance can be reliably estimated
2. You want more stable parameter selection than GCV
3. You're working with replicated data

Key Features Demonstrated:
- Basic UBRE score calculation
- Automatic lambda selection using UBRE
- Comparison of UBRE vs GCV
- Error variance estimation
"""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from omnilss.smoothers.ubre import (
    ubre_score,
    select_lambda_ubre,
    estimate_sigma2,
    ubre_vs_gcv,
)
from omnilss.smoothers.penalties import penalty_matrix

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("UBRE (Unbiased Risk Estimator) Demonstration")
print("=" * 70)

# ============================================================================
# 1. Generate synthetic data with known smooth function
# ============================================================================
print("\n1. Generating synthetic data...")

n = 200
x = np.linspace(0, 10, n)

# True underlying function (smooth)
y_true = np.sin(x) + 0.5 * np.cos(2 * x)

# Add Gaussian noise with known variance
true_sigma = 0.3
noise = np.random.normal(0, true_sigma, n)
y = y_true + noise

print(f"   - Sample size: {n}")
print(f"   - True error std: {true_sigma:.3f}")
print(f"   - True error variance (σ²): {true_sigma**2:.4f}")

# ============================================================================
# 2. Create design matrix (polynomial basis)
# ============================================================================
print("\n2. Creating design matrix...")

# Use polynomial basis for simplicity
p = 30  # Number of basis functions
X = np.column_stack([x**i for i in range(p)])
X = jnp.array(X)
y_jax = jnp.array(y)

# Create penalty matrix (penalize second derivatives)
P = jnp.array(penalty_matrix(p, order=2))

print(f"   - Number of basis functions: {p}")
print(f"   - Design matrix shape: {X.shape}")
print(f"   - Penalty matrix shape: {P.shape}")

# ============================================================================
# 3. Estimate error variance from preliminary fit
# ============================================================================
print("\n3. Estimating error variance...")

# Fit with moderate penalty to get initial estimate
lambda_init = 1.0
XtX = X.T @ X
Xty = X.T @ y_jax
XtX_pen = XtX + lambda_init * P

coef_init = jnp.linalg.solve(XtX_pen, Xty)
fitted_init = X @ coef_init

# Estimate variance
XtX_pen_inv = jnp.linalg.inv(XtX_pen)
hat_matrix_init = X @ XtX_pen_inv @ X.T
edf_init = jnp.trace(hat_matrix_init)

sigma2_est = estimate_sigma2(y_jax, fitted_init, float(edf_init))

print(f"   - Initial lambda: {lambda_init}")
print(f"   - Effective df: {edf_init:.2f}")
print(f"   - Estimated σ²: {sigma2_est:.4f}")
print(f"   - True σ²: {true_sigma**2:.4f}")
print(f"   - Estimation error: {abs(sigma2_est - true_sigma**2):.4f}")

# ============================================================================
# 4. Select optimal lambda using UBRE
# ============================================================================
print("\n4. Selecting optimal smoothing parameter using UBRE...")

# Use the true variance (in practice, use estimated variance)
sigma2_known = true_sigma**2

best_lambda, ubre_scores = select_lambda_ubre(
    X, y_jax, P, sigma2_known, return_scores=True
)

print(f"   - Optimal lambda: {best_lambda:.6f}")
print(f"   - Number of lambda values tested: {len(ubre_scores)}")

# Fit with optimal lambda
XtX_pen_opt = XtX + best_lambda * P
coef_opt = jnp.linalg.solve(XtX_pen_opt, Xty)
fitted_opt = X @ coef_opt

# Calculate effective df
XtX_pen_inv_opt = jnp.linalg.inv(XtX_pen_opt)
hat_matrix_opt = X @ XtX_pen_inv_opt @ X.T
edf_opt = jnp.trace(hat_matrix_opt)

print(f"   - Effective df with optimal lambda: {edf_opt:.2f}")

# Calculate UBRE score
ubre_opt = ubre_score(y_jax, fitted_opt, hat_matrix_opt, sigma2_known)
print(f"   - UBRE score: {ubre_opt:.6f}")

# ============================================================================
# 5. Compare UBRE vs GCV
# ============================================================================
print("\n5. Comparing UBRE and GCV...")

results = ubre_vs_gcv(X, y_jax, P, sigma2_known)

print(f"   - Optimal lambda (UBRE): {results['best_lambda_ubre']:.6f}")
print(f"   - Optimal lambda (GCV): {results['best_lambda_gcv']:.6f}")
print(f"   - Ratio (UBRE/GCV): {results['best_lambda_ubre']/results['best_lambda_gcv']:.3f}")

# Fit with GCV-selected lambda
XtX_pen_gcv = XtX + results['best_lambda_gcv'] * P
coef_gcv = jnp.linalg.solve(XtX_pen_gcv, Xty)
fitted_gcv = X @ coef_gcv

# ============================================================================
# 6. Visualize results
# ============================================================================
print("\n6. Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Data and fits
ax = axes[0, 0]
ax.scatter(x, y, alpha=0.3, s=20, label='Noisy data', color='gray')
ax.plot(x, y_true, 'k-', linewidth=2, label='True function', alpha=0.7)
ax.plot(x, np.asarray(fitted_opt), 'b-', linewidth=2, label=f'UBRE fit (λ={best_lambda:.2e})')
ax.plot(x, np.asarray(fitted_gcv), 'r--', linewidth=2, 
        label=f'GCV fit (λ={results["best_lambda_gcv"]:.2e})')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('UBRE vs GCV: Fitted Functions', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 2: UBRE scores across lambda values
ax = axes[0, 1]
lambda_grid = results['lambda_grid']
ax.semilogx(lambda_grid, results['ubre_scores'], 'b-', linewidth=2, label='UBRE')
ax.axvline(results['best_lambda_ubre'], color='b', linestyle='--', 
           label=f'Optimal λ (UBRE) = {results["best_lambda_ubre"]:.2e}')
ax.set_xlabel('Lambda (λ)', fontsize=12)
ax.set_ylabel('UBRE Score', fontsize=12)
ax.set_title('UBRE Score vs Smoothing Parameter', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 3: GCV scores across lambda values
ax = axes[1, 0]
ax.semilogx(lambda_grid, results['gcv_scores'], 'r-', linewidth=2, label='GCV')
ax.axvline(results['best_lambda_gcv'], color='r', linestyle='--',
           label=f'Optimal λ (GCV) = {results["best_lambda_gcv"]:.2e}')
ax.set_xlabel('Lambda (λ)', fontsize=12)
ax.set_ylabel('GCV Score', fontsize=12)
ax.set_title('GCV Score vs Smoothing Parameter', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 4: Residuals comparison
ax = axes[1, 1]
residuals_ubre = y - np.asarray(fitted_opt)
residuals_gcv = y - np.asarray(fitted_gcv)

ax.scatter(x, residuals_ubre, alpha=0.5, s=20, label='UBRE residuals', color='blue')
ax.scatter(x, residuals_gcv, alpha=0.5, s=20, label='GCV residuals', color='red', marker='x')
ax.axhline(0, color='k', linestyle='-', linewidth=1, alpha=0.5)
ax.axhline(true_sigma, color='k', linestyle='--', linewidth=1, alpha=0.5, label=f'±{true_sigma:.2f} (true σ)')
ax.axhline(-true_sigma, color='k', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Residuals', fontsize=12)
ax.set_title('Residuals Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('examples/ubre_demo.png', dpi=150, bbox_inches='tight')
print("   - Saved plot to 'examples/ubre_demo.png'")

# ============================================================================
# 7. Performance metrics
# ============================================================================
print("\n7. Performance Metrics:")
print("-" * 70)

# Mean Squared Error
mse_ubre = np.mean((y_true - np.asarray(fitted_opt))**2)
mse_gcv = np.mean((y_true - np.asarray(fitted_gcv))**2)

print(f"   UBRE fit:")
print(f"      - MSE: {mse_ubre:.6f}")
print(f"      - Effective df: {edf_opt:.2f}")
print(f"      - Lambda: {best_lambda:.6e}")

print(f"\n   GCV fit:")
print(f"      - MSE: {mse_gcv:.6f}")
XtX_pen_inv_gcv = jnp.linalg.inv(XtX_pen_gcv)
hat_matrix_gcv = X @ XtX_pen_inv_gcv @ X.T
edf_gcv = jnp.trace(hat_matrix_gcv)
print(f"      - Effective df: {edf_gcv:.2f}")
print(f"      - Lambda: {results['best_lambda_gcv']:.6e}")

print(f"\n   Comparison:")
print(f"      - MSE improvement (UBRE vs GCV): {(mse_gcv - mse_ubre)/mse_gcv * 100:.2f}%")

# ============================================================================
# 8. Key Insights
# ============================================================================
print("\n" + "=" * 70)
print("KEY INSIGHTS")
print("=" * 70)
print("""
1. UBRE vs GCV:
   - UBRE requires knowledge of error variance σ²
   - GCV estimates it implicitly from the data
   - When σ² is known accurately, UBRE can be more stable
   - Both methods typically give similar results

2. Smoothing Parameter Selection:
   - Lower λ → more flexible fit (higher edf)
   - Higher λ → smoother fit (lower edf)
   - UBRE/GCV automatically balance bias-variance tradeoff

3. Error Variance Estimation:
   - Can be estimated from a preliminary fit
   - More accurate with replicated data
   - Robust methods (e.g., MAD) can help with outliers

4. Practical Recommendations:
   - Use UBRE when you have good variance estimates
   - Use GCV when variance is unknown
   - Compare both methods for robustness
   - Consider cross-validation for small samples
""")

print("\n" + "=" * 70)
print("Demo completed successfully!")
print("=" * 70)

plt.show()
