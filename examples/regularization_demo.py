"""
Regularization Methods Demonstration
=====================================

This script demonstrates L1 (Lasso), L2 (Ridge), and Elastic Net regularization
for feature selection and coefficient shrinkage.

Key Features Demonstrated:
1. Lasso (L1) for feature selection
2. Ridge (L2) for coefficient shrinkage
3. Elastic Net combining L1 and L2
4. Cross-validation for lambda selection
5. Comparison of methods
6. Sparse vs dense solutions

Author: OmniLSS Development Team
Date: 2026-05-04
"""

import numpy as np
import matplotlib.pyplot as plt

from omnilss.regularization import (
    fit_lasso_coordinate_descent,
    fit_ridge,
    fit_elastic_net,
    cross_validate_lambda,
    fit_regularized,
)

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("Regularization Methods Demonstration")
print("=" * 70)
print()

# ============================================================================
# Generate Sparse Data
# ============================================================================
print("Generating sparse data...")
print("-" * 70)

n = 200  # Number of observations
p = 50   # Number of features
k = 5    # Number of true non-zero coefficients

# Design matrix
X = np.random.randn(n, p)

# True sparse coefficients
beta_true = np.zeros(p)
true_indices = np.random.choice(p, k, replace=False)
beta_true[true_indices] = np.random.randn(k) * 3

# Response with noise
y = X @ beta_true + np.random.randn(n) * 0.5

print(f"Number of observations: {n}")
print(f"Number of features: {p}")
print(f"True non-zero coefficients: {k}")
print(f"True coefficient indices: {sorted(true_indices)}")
print(f"Signal-to-noise ratio: {np.var(X @ beta_true) / np.var(y - X @ beta_true):.2f}")
print()

# ============================================================================
# Example 1: Lasso (L1) for Feature Selection
# ============================================================================
print("Example 1: Lasso (L1) for Feature Selection")
print("-" * 70)

# Fit Lasso with different lambda values
lambda_values = [0.01, 0.1, 0.5, 1.0, 2.0]
lasso_results = {}

for lam in lambda_values:
    result = fit_lasso_coordinate_descent(X, y, lambda_=lam)
    lasso_results[lam] = result
    
    # Calculate MSE
    mse = np.mean((y - result.fitted_values) ** 2)
    
    print(f"λ={lam:4.2f}: n_nonzero={result.n_nonzero:2d}, "
          f"n_iter={result.n_iter:3d}, MSE={mse:.4f}")

print()

# ============================================================================
# Example 2: Ridge (L2) for Coefficient Shrinkage
# ============================================================================
print("Example 2: Ridge (L2) for Coefficient Shrinkage")
print("-" * 70)

ridge_results = {}

for lam in lambda_values:
    result = fit_ridge(X, y, lambda_=lam)
    ridge_results[lam] = result
    
    # Calculate MSE
    mse = np.mean((y - result.fitted_values) ** 2)
    
    # L2 norm of coefficients
    l2_norm = np.sqrt(np.sum(result.coefficients ** 2))
    
    print(f"λ={lam:4.2f}: ||β||₂={l2_norm:6.2f}, MSE={mse:.4f}")

print()

# ============================================================================
# Example 3: Elastic Net (L1 + L2)
# ============================================================================
print("Example 3: Elastic Net (L1 + L2)")
print("-" * 70)

alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9]
lambda_fixed = 0.5
elastic_results = {}

for alpha in alpha_values:
    result = fit_elastic_net(X, y, lambda_=lambda_fixed, alpha=alpha)
    elastic_results[alpha] = result
    
    mse = np.mean((y - result.fitted_values) ** 2)
    
    print(f"α={alpha:.1f}: n_nonzero={result.n_nonzero:2d}, MSE={mse:.4f}")

print()

# ============================================================================
# Example 4: Cross-Validation for Lambda Selection
# ============================================================================
print("Example 4: Cross-Validation for Lambda Selection")
print("-" * 70)

# Lasso CV
best_lambda_lasso, cv_scores_lasso = cross_validate_lambda(
    X, y, method="lasso", n_folds=5
)
print(f"Lasso - Best λ: {best_lambda_lasso:.4e}")

# Ridge CV
best_lambda_ridge, cv_scores_ridge = cross_validate_lambda(
    X, y, method="ridge", n_folds=5
)
print(f"Ridge - Best λ: {best_lambda_ridge:.4e}")

# Elastic Net CV
best_lambda_en, cv_scores_en = cross_validate_lambda(
    X, y, alpha=0.5, method="elastic_net", n_folds=5
)
print(f"Elastic Net (α=0.5) - Best λ: {best_lambda_en:.4e}")

print()

# ============================================================================
# Example 5: Unified Interface
# ============================================================================
print("Example 5: Unified Interface with Auto Lambda Selection")
print("-" * 70)

# Lasso with auto lambda
result_lasso_auto = fit_regularized(X, y, method="lasso", cv=True, n_folds=5)
print(f"Lasso: λ={result_lasso_auto.lambda_:.4e}, "
      f"n_nonzero={result_lasso_auto.n_nonzero}")

# Ridge with auto lambda
result_ridge_auto = fit_regularized(X, y, method="ridge", cv=True, n_folds=5)
print(f"Ridge: λ={result_ridge_auto.lambda_:.4e}")

# Elastic Net with auto lambda
result_en_auto = fit_regularized(
    X, y, alpha=0.5, method="elastic_net", cv=True, n_folds=5
)
print(f"Elastic Net: λ={result_en_auto.lambda_:.4e}, "
      f"n_nonzero={result_en_auto.n_nonzero}")

print()

# ============================================================================
# Visualization
# ============================================================================
print("Creating visualizations...")

fig = plt.figure(figsize=(16, 12))

# Plot 1: Lasso coefficient paths
ax1 = plt.subplot(3, 3, 1)
for i in range(p):
    coefs = [lasso_results[lam].coefficients[i] for lam in lambda_values]
    if i in true_indices:
        ax1.plot(lambda_values, coefs, 'r-', linewidth=2, alpha=0.7)
    else:
        ax1.plot(lambda_values, coefs, 'b-', linewidth=0.5, alpha=0.3)

ax1.axhline(y=0, color='k', linestyle='--', linewidth=1)
ax1.set_xlabel('λ')
ax1.set_ylabel('Coefficient value')
ax1.set_title('Lasso Coefficient Paths')
ax1.set_xscale('log')
ax1.grid(True, alpha=0.3)
ax1.legend(['True non-zero', 'True zero'], loc='best')

# Plot 2: Ridge coefficient paths
ax2 = plt.subplot(3, 3, 2)
for i in range(p):
    coefs = [ridge_results[lam].coefficients[i] for lam in lambda_values]
    if i in true_indices:
        ax2.plot(lambda_values, coefs, 'r-', linewidth=2, alpha=0.7)
    else:
        ax2.plot(lambda_values, coefs, 'b-', linewidth=0.5, alpha=0.3)

ax2.axhline(y=0, color='k', linestyle='--', linewidth=1)
ax2.set_xlabel('λ')
ax2.set_ylabel('Coefficient value')
ax2.set_title('Ridge Coefficient Paths')
ax2.set_xscale('log')
ax2.grid(True, alpha=0.3)

# Plot 3: Number of non-zero coefficients vs lambda (Lasso)
ax3 = plt.subplot(3, 3, 3)
n_nonzero = [lasso_results[lam].n_nonzero for lam in lambda_values]
ax3.plot(lambda_values, n_nonzero, 'o-', linewidth=2, markersize=8)
ax3.axhline(y=k, color='r', linestyle='--', linewidth=2, label=f'True k={k}')
ax3.set_xlabel('λ')
ax3.set_ylabel('Number of non-zero coefficients')
ax3.set_title('Lasso Sparsity')
ax3.set_xscale('log')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Plot 4: MSE vs lambda (Lasso)
ax4 = plt.subplot(3, 3, 4)
mse_lasso = [np.mean((y - lasso_results[lam].fitted_values) ** 2) 
             for lam in lambda_values]
ax4.plot(lambda_values, mse_lasso, 'o-', linewidth=2, markersize=8, color='green')
ax4.set_xlabel('λ')
ax4.set_ylabel('MSE')
ax4.set_title('Lasso: MSE vs λ')
ax4.set_xscale('log')
ax4.grid(True, alpha=0.3)

# Plot 5: MSE vs lambda (Ridge)
ax5 = plt.subplot(3, 3, 5)
mse_ridge = [np.mean((y - ridge_results[lam].fitted_values) ** 2) 
             for lam in lambda_values]
ax5.plot(lambda_values, mse_ridge, 'o-', linewidth=2, markersize=8, color='orange')
ax5.set_xlabel('λ')
ax5.set_ylabel('MSE')
ax5.set_title('Ridge: MSE vs λ')
ax5.set_xscale('log')
ax5.grid(True, alpha=0.3)

# Plot 6: Elastic Net: MSE vs alpha
ax6 = plt.subplot(3, 3, 6)
mse_en = [np.mean((y - elastic_results[alpha].fitted_values) ** 2) 
          for alpha in alpha_values]
ax6.plot(alpha_values, mse_en, 'o-', linewidth=2, markersize=8, color='purple')
ax6.set_xlabel('α (1=Lasso, 0=Ridge)')
ax6.set_ylabel('MSE')
ax6.set_title(f'Elastic Net: MSE vs α (λ={lambda_fixed})')
ax6.grid(True, alpha=0.3)

# Plot 7: True vs Estimated coefficients (Lasso with best lambda)
ax7 = plt.subplot(3, 3, 7)
result_best = lasso_results[0.5]  # Use lambda=0.5
ax7.scatter(beta_true, result_best.coefficients, alpha=0.6, s=50)
ax7.plot([beta_true.min(), beta_true.max()], 
         [beta_true.min(), beta_true.max()], 
         'r--', linewidth=2, label='Perfect recovery')
ax7.set_xlabel('True coefficients')
ax7.set_ylabel('Estimated coefficients')
ax7.set_title(f'Lasso: True vs Estimated (λ=0.5)')
ax7.legend()
ax7.grid(True, alpha=0.3)

# Plot 8: CV scores for Lasso
ax8 = plt.subplot(3, 3, 8)
lambda_grid = np.logspace(-3, 1, len(cv_scores_lasso))
ax8.plot(lambda_grid, cv_scores_lasso, 'o-', linewidth=2, markersize=6)
ax8.axvline(x=best_lambda_lasso, color='r', linestyle='--', linewidth=2,
            label=f'Best λ={best_lambda_lasso:.2e}')
ax8.set_xlabel('λ')
ax8.set_ylabel('CV MSE')
ax8.set_title('Lasso: Cross-Validation')
ax8.set_xscale('log')
ax8.grid(True, alpha=0.3)
ax8.legend()

# Plot 9: Comparison of methods
ax9 = plt.subplot(3, 3, 9)
methods = ['Lasso', 'Ridge', 'Elastic Net']
n_nonzero_methods = [
    result_lasso_auto.n_nonzero,
    p,  # Ridge doesn't zero out
    result_en_auto.n_nonzero,
]
mse_methods = [
    np.mean((y - result_lasso_auto.fitted_values) ** 2),
    np.mean((y - result_ridge_auto.fitted_values) ** 2),
    np.mean((y - result_en_auto.fitted_values) ** 2),
]

x_pos = np.arange(len(methods))
ax9_twin = ax9.twinx()

bars1 = ax9.bar(x_pos - 0.2, n_nonzero_methods, 0.4, 
                label='Non-zero coefs', color='steelblue', alpha=0.7)
bars2 = ax9_twin.bar(x_pos + 0.2, mse_methods, 0.4, 
                     label='MSE', color='coral', alpha=0.7)

ax9.set_xlabel('Method')
ax9.set_ylabel('Number of non-zero coefficients', color='steelblue')
ax9_twin.set_ylabel('MSE', color='coral')
ax9.set_title('Method Comparison (Auto λ)')
ax9.set_xticks(x_pos)
ax9.set_xticklabels(methods)
ax9.tick_params(axis='y', labelcolor='steelblue')
ax9_twin.tick_params(axis='y', labelcolor='coral')
ax9.axhline(y=k, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax9.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('examples/regularization_demo.png', dpi=150, bbox_inches='tight')
print("Saved: examples/regularization_demo.png")

# ============================================================================
# Summary Statistics
# ============================================================================
print()
print("=" * 70)
print("Summary Statistics")
print("=" * 70)
print()

print("Best Models (Auto Lambda Selection):")
print("-" * 70)

# Lasso
mse_lasso_auto = np.mean((y - result_lasso_auto.fitted_values) ** 2)
recovered = np.sum((np.abs(result_lasso_auto.coefficients) > 0.01) & 
                   (np.abs(beta_true) > 0))
print(f"Lasso:")
print(f"  - λ: {result_lasso_auto.lambda_:.4e}")
print(f"  - Non-zero coefficients: {result_lasso_auto.n_nonzero}/{p}")
print(f"  - True non-zeros recovered: {recovered}/{k}")
print(f"  - MSE: {mse_lasso_auto:.4f}")
print()

# Ridge
mse_ridge_auto = np.mean((y - result_ridge_auto.fitted_values) ** 2)
l2_norm = np.sqrt(np.sum(result_ridge_auto.coefficients ** 2))
print(f"Ridge:")
print(f"  - λ: {result_ridge_auto.lambda_:.4e}")
print(f"  - ||β||₂: {l2_norm:.2f}")
print(f"  - MSE: {mse_ridge_auto:.4f}")
print()

# Elastic Net
mse_en_auto = np.mean((y - result_en_auto.fitted_values) ** 2)
recovered_en = np.sum((np.abs(result_en_auto.coefficients) > 0.01) & 
                      (np.abs(beta_true) > 0))
print(f"Elastic Net (α=0.5):")
print(f"  - λ: {result_en_auto.lambda_:.4e}")
print(f"  - Non-zero coefficients: {result_en_auto.n_nonzero}/{p}")
print(f"  - True non-zeros recovered: {recovered_en}/{k}")
print(f"  - MSE: {mse_en_auto:.4f}")
print()

# ============================================================================
# Recommendations
# ============================================================================
print("=" * 70)
print("Recommendations")
print("=" * 70)
print()
print("1. Use Lasso (L1) when:")
print("   - You want feature selection")
print("   - You believe the true model is sparse")
print("   - Interpretability is important")
print()
print("2. Use Ridge (L2) when:")
print("   - Features are correlated")
print("   - You want to shrink all coefficients")
print("   - Prediction accuracy is more important than sparsity")
print()
print("3. Use Elastic Net when:")
print("   - You want both feature selection and grouping")
print("   - Features are highly correlated")
print("   - You're unsure between Lasso and Ridge")
print()
print("4. Lambda selection:")
print("   - Always use cross-validation for lambda selection")
print("   - Use 5-10 folds for CV")
print("   - Consider nested CV for model selection")
print()
print("5. For GAMLSS models:")
print("   - Apply regularization to smooth term coefficients")
print("   - Use Lasso for automatic term selection")
print("   - Use Ridge for numerical stability")
print()

print("=" * 70)
print("Demonstration complete!")
print("=" * 70)
