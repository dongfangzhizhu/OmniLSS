"""
Thin Plate Spline (TPS) Demonstration
======================================

This script demonstrates the use of thin plate splines for multi-dimensional
smoothing in OmniLSS.

Key Features Demonstrated:
1. 1D TPS (equivalent to cubic spline)
2. 2D TPS for spatial smoothing
3. Comparison with true function
4. Effect of smoothing parameter
5. Knot selection methods
6. Prediction at new locations

Author: OmniLSS Development Team
Date: 2026-05-04
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from omnilss.smoothers.tps import fit_tps

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("Thin Plate Spline (TPS) Demonstration")
print("=" * 70)
print()

# ============================================================================
# Example 1: 1D TPS (Cubic Spline)
# ============================================================================
print("Example 1: 1D TPS (Cubic Spline)")
print("-" * 70)

n_1d = 50
X_1d = np.linspace(0, 1, n_1d).reshape(-1, 1)
y_true_1d = np.sin(2 * np.pi * X_1d.ravel())
y_1d = y_true_1d + np.random.normal(0, 0.2, n_1d)

# Fit TPS with automatic lambda selection
result_1d = fit_tps(X_1d, y_1d, k=15, method="GCV")

print(f"Number of observations: {n_1d}")
print(f"Number of knots: 15")
print(f"Selected lambda: {result_1d.lambda_:.4e}")
print(f"Effective degrees of freedom: {result_1d.edf:.2f}")
print(f"Selection method: {result_1d.selection_method}")

# Predict at fine grid
X_pred_1d = np.linspace(0, 1, 200).reshape(-1, 1)
y_pred_1d = result_1d.predict(X_pred_1d)

# Calculate R²
ss_res = np.sum((y_1d - result_1d.fitted_values) ** 2)
ss_tot = np.sum((y_1d - np.mean(y_1d)) ** 2)
r_squared_1d = 1 - ss_res / ss_tot
print(f"R² on training data: {r_squared_1d:.4f}")
print()

# ============================================================================
# Example 2: 2D TPS for Spatial Smoothing
# ============================================================================
print("Example 2: 2D TPS for Spatial Smoothing")
print("-" * 70)

n_2d = 200
x1 = np.random.uniform(0, 1, n_2d)
x2 = np.random.uniform(0, 1, n_2d)
X_2d = np.column_stack([x1, x2])

# True function: sin(2πx1) * cos(2πx2)
y_true_2d = np.sin(2 * np.pi * x1) * np.cos(2 * np.pi * x2)
y_2d = y_true_2d + np.random.normal(0, 0.15, n_2d)

# Fit TPS with different knot numbers
k_values = [10, 20, 40]
results_2d = {}

for k in k_values:
    result = fit_tps(X_2d, y_2d, k=k, method="GCV", knot_method="kmeans")
    results_2d[k] = result
    
    ss_res = np.sum((y_2d - result.fitted_values) ** 2)
    ss_tot = np.sum((y_2d - np.mean(y_2d)) ** 2)
    r_squared = 1 - ss_res / ss_tot
    
    print(f"k={k:2d}: lambda={result.lambda_:.4e}, edf={result.edf:5.2f}, R²={r_squared:.4f}")

print()

# ============================================================================
# Example 3: Effect of Smoothing Parameter
# ============================================================================
print("Example 3: Effect of Smoothing Parameter")
print("-" * 70)

# Generate simple 2D data
n_smooth = 100
x1_smooth = np.random.uniform(0, 1, n_smooth)
x2_smooth = np.random.uniform(0, 1, n_smooth)
X_smooth = np.column_stack([x1_smooth, x2_smooth])
y_smooth = x1_smooth + x2_smooth + np.random.normal(0, 0.1, n_smooth)

# Fit with different lambda values
lambda_values = [1e-6, 1e-3, 1e-1, 1.0]
results_smooth = {}

for lam in lambda_values:
    result = fit_tps(X_smooth, y_smooth, lambda_=lam, k=20, method="fixed")
    results_smooth[lam] = result
    
    print(f"lambda={lam:.1e}: edf={result.edf:5.2f}")

print()

# ============================================================================
# Example 4: Knot Selection Methods
# ============================================================================
print("Example 4: Knot Selection Methods")
print("-" * 70)

knot_methods = ["uniform", "kmeans", "all"]
results_knots = {}

for method in knot_methods:
    if method == "all":
        k_knot = None
    else:
        k_knot = 30
    
    result = fit_tps(X_2d, y_2d, k=k_knot, method="GCV", knot_method=method)
    results_knots[method] = result
    
    ss_res = np.sum((y_2d - result.fitted_values) ** 2)
    ss_tot = np.sum((y_2d - np.mean(y_2d)) ** 2)
    r_squared = 1 - ss_res / ss_tot
    
    n_knots = result.knots.shape[0]
    print(f"{method:8s}: n_knots={n_knots:3d}, edf={result.edf:5.2f}, R²={r_squared:.4f}")

print()

# ============================================================================
# Visualization
# ============================================================================
print("Creating visualizations...")

fig = plt.figure(figsize=(16, 12))

# Plot 1: 1D TPS
ax1 = plt.subplot(3, 3, 1)
ax1.scatter(X_1d, y_1d, alpha=0.5, s=20, label='Data')
ax1.plot(X_pred_1d, y_pred_1d, 'r-', linewidth=2, label='TPS fit')
ax1.plot(X_pred_1d, np.sin(2 * np.pi * X_pred_1d.ravel()), 'g--', 
         linewidth=1, label='True function')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title(f'1D TPS (k=15, λ={result_1d.lambda_:.2e})')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: 2D TPS - Data
ax2 = plt.subplot(3, 3, 2, projection='3d')
ax2.scatter(x1, x2, y_2d, c=y_2d, cmap='viridis', alpha=0.6, s=20)
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_zlabel('y')
ax2.set_title('2D Data (n=200)')

# Plot 3: 2D TPS - Fitted Surface (k=20)
ax3 = plt.subplot(3, 3, 3, projection='3d')
result_plot = results_2d[20]

# Create grid for surface
x1_grid = np.linspace(0, 1, 30)
x2_grid = np.linspace(0, 1, 30)
X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid)
X_grid = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])
y_grid = result_plot.predict(X_grid).reshape(X1_grid.shape)

ax3.plot_surface(X1_grid, X2_grid, y_grid, cmap='viridis', alpha=0.8)
ax3.scatter(x1, x2, y_2d, c='red', alpha=0.3, s=10)
ax3.set_xlabel('x1')
ax3.set_ylabel('x2')
ax3.set_zlabel('y')
ax3.set_title(f'2D TPS Fit (k=20, edf={result_plot.edf:.1f})')

# Plot 4: Comparison of k values
ax4 = plt.subplot(3, 3, 4)
k_list = list(results_2d.keys())
edf_list = [results_2d[k].edf for k in k_list]
r2_list = []
for k in k_list:
    result = results_2d[k]
    ss_res = np.sum((y_2d - result.fitted_values) ** 2)
    ss_tot = np.sum((y_2d - np.mean(y_2d)) ** 2)
    r2_list.append(1 - ss_res / ss_tot)

ax4.plot(k_list, edf_list, 'o-', linewidth=2, markersize=8, label='EDF')
ax4.set_xlabel('Number of knots (k)')
ax4.set_ylabel('Effective degrees of freedom')
ax4.set_title('EDF vs Number of Knots')
ax4.grid(True, alpha=0.3)
ax4.legend()

# Plot 5: R² vs k
ax5 = plt.subplot(3, 3, 5)
ax5.plot(k_list, r2_list, 's-', linewidth=2, markersize=8, color='green')
ax5.set_xlabel('Number of knots (k)')
ax5.set_ylabel('R²')
ax5.set_title('Model Fit vs Number of Knots')
ax5.grid(True, alpha=0.3)
ax5.set_ylim([0, 1])

# Plot 6: Effect of lambda
ax6 = plt.subplot(3, 3, 6)
lambda_list = list(results_smooth.keys())
edf_smooth = [results_smooth[lam].edf for lam in lambda_list]
ax6.semilogx(lambda_list, edf_smooth, 'o-', linewidth=2, markersize=8)
ax6.set_xlabel('Smoothing parameter (λ)')
ax6.set_ylabel('Effective degrees of freedom')
ax6.set_title('EDF vs Smoothing Parameter')
ax6.grid(True, alpha=0.3)

# Plot 7: True vs Fitted (k=20)
ax7 = plt.subplot(3, 3, 7)
result_compare = results_2d[20]
ax7.scatter(y_true_2d, result_compare.fitted_values, alpha=0.5, s=20)
ax7.plot([y_true_2d.min(), y_true_2d.max()], 
         [y_true_2d.min(), y_true_2d.max()], 
         'r--', linewidth=2, label='Perfect fit')
ax7.set_xlabel('True values')
ax7.set_ylabel('Fitted values')
ax7.set_title('True vs Fitted (k=20)')
ax7.legend()
ax7.grid(True, alpha=0.3)

# Plot 8: Residuals
ax8 = plt.subplot(3, 3, 8)
residuals = y_2d - result_compare.fitted_values
ax8.scatter(result_compare.fitted_values, residuals, alpha=0.5, s=20)
ax8.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax8.set_xlabel('Fitted values')
ax8.set_ylabel('Residuals')
ax8.set_title('Residual Plot (k=20)')
ax8.grid(True, alpha=0.3)

# Plot 9: Knot locations (k=20)
ax9 = plt.subplot(3, 3, 9)
ax9.scatter(x1, x2, c=y_2d, cmap='viridis', alpha=0.3, s=20, label='Data')
knots = result_compare.knots
ax9.scatter(knots[:, 0], knots[:, 1], c='red', marker='x', s=100, 
            linewidths=2, label='Knots')
ax9.set_xlabel('x1')
ax9.set_ylabel('x2')
ax9.set_title(f'Knot Locations (k={knots.shape[0]})')
ax9.legend()
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('examples/tps_demo.png', dpi=150, bbox_inches='tight')
print("Saved: examples/tps_demo.png")

# ============================================================================
# Summary Statistics
# ============================================================================
print()
print("=" * 70)
print("Summary Statistics")
print("=" * 70)
print()

print("1D TPS:")
print(f"  - R²: {r_squared_1d:.4f}")
print(f"  - RMSE: {np.sqrt(np.mean((y_1d - result_1d.fitted_values)**2)):.4f}")
print()

print("2D TPS (k=20):")
result_final = results_2d[20]
ss_res_final = np.sum((y_2d - result_final.fitted_values) ** 2)
ss_tot_final = np.sum((y_2d - np.mean(y_2d)) ** 2)
r2_final = 1 - ss_res_final / ss_tot_final
rmse_final = np.sqrt(np.mean((y_2d - result_final.fitted_values)**2))
print(f"  - R²: {r2_final:.4f}")
print(f"  - RMSE: {rmse_final:.4f}")
print(f"  - Lambda: {result_final.lambda_:.4e}")
print(f"  - EDF: {result_final.edf:.2f}")
print()

# ============================================================================
# Performance Comparison
# ============================================================================
print("Performance Comparison (2D TPS):")
print("-" * 70)
print(f"{'Method':<15} {'n_knots':<10} {'EDF':<10} {'R²':<10} {'RMSE':<10}")
print("-" * 70)

for method in knot_methods:
    result = results_knots[method]
    n_knots = result.knots.shape[0]
    ss_res = np.sum((y_2d - result.fitted_values) ** 2)
    ss_tot = np.sum((y_2d - np.mean(y_2d)) ** 2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((y_2d - result.fitted_values)**2))
    
    print(f"{method:<15} {n_knots:<10} {result.edf:<10.2f} {r2:<10.4f} {rmse:<10.4f}")

print()

# ============================================================================
# Recommendations
# ============================================================================
print("=" * 70)
print("Recommendations")
print("=" * 70)
print()
print("1. For small datasets (n < 100):")
print("   - Use knot_method='all' for best accuracy")
print("   - Let GCV select lambda automatically")
print()
print("2. For medium datasets (100 < n < 1000):")
print("   - Use k=20-50 with knot_method='kmeans'")
print("   - Balance between accuracy and speed")
print()
print("3. For large datasets (n > 1000):")
print("   - Use k=50-100 with knot_method='uniform'")
print("   - Consider subsampling for very large datasets")
print()
print("4. For spatial data:")
print("   - TPS is ideal for irregular spatial grids")
print("   - Use 2D TPS for geographic smoothing")
print()
print("5. For higher dimensions (d > 3):")
print("   - TPS can be computationally expensive")
print("   - Consider tensor product splines instead")
print()

print("=" * 70)
print("Demonstration complete!")
print("=" * 70)
