"""Demonstration of automatic smoothing parameter selection.

This script demonstrates the use of auto_smooth() for automatic
lambda selection using GCV and REML methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from omnilss.smoothers.smooth_parameter_selection import auto_smooth
from omnilss.fitting import gamlss_ml

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
print("=" * 70)
print("Automatic Smoothing Parameter Selection Demo")
print("=" * 70)

n = 150
x = np.linspace(0, 10, n)
true_mu = np.sin(x) + 0.5 * np.cos(2 * x)
y = true_mu + 0.3 * np.random.randn(n)

data = {"x": x, "y": y}

print(f"\nGenerated {n} observations")
print(f"True function: sin(x) + 0.5*cos(2x)")
print(f"Noise level: σ = 0.3")

# Example 1: Basic usage with REML (default)
print("\n" + "=" * 70)
print("Example 1: Basic usage with REML (default)")
print("=" * 70)

model_reml = auto_smooth(
    formula="y ~ pb(x, df=12)",
    family="NO",
    data=data,
    method="REML",
    verbose=True
)

print(f"\nModel summary:")
print(f"  Deviance: {model_reml.g_dev:.4f}")
print(f"  AIC: {model_reml.additional_slots['aic']:.4f}")
print(f"  Effective df: {model_reml.df_fit:.2f}")

# Extract smooth information
smooth_fits = model_reml.additional_slots["smooth_fits"]
smooth = smooth_fits["mu"][0]
print(f"\nSmooth term:")
print(f"  Variable: {smooth.variable}")
print(f"  Selected λ: {smooth.lambda_:.4e}")
print(f"  Effective df: {smooth.edf:.2f}")

# Example 2: Using GCV method
print("\n" + "=" * 70)
print("Example 2: Using GCV method")
print("=" * 70)

model_gcv = auto_smooth(
    formula="y ~ pb(x, df=12)",
    family="NO",
    data=data,
    method="GCV",
    verbose=False
)

smooth_gcv = model_gcv.additional_slots["smooth_fits"]["mu"][0]
print(f"\nGCV results:")
print(f"  Selected λ: {smooth_gcv.lambda_:.4e}")
print(f"  Effective df: {smooth_gcv.edf:.2f}")
print(f"  AIC: {model_gcv.additional_slots['aic']:.4f}")

# Example 3: Comparison with manual lambda
print("\n" + "=" * 70)
print("Example 3: Comparison with manual lambda")
print("=" * 70)

# Manual lambda (traditional approach)
model_manual = gamlss_ml(
    formula="y ~ pb(x, df=12, lambda_=1.0)",
    family="NO",
    data=data
)

smooth_manual = model_manual.additional_slots["smooth_fits"]["mu"][0]
print(f"\nManual λ = 1.0:")
print(f"  Effective df: {smooth_manual.edf:.2f}")
print(f"  AIC: {model_manual.additional_slots['aic']:.4f}")

print(f"\nAuto REML:")
print(f"  λ: {smooth.lambda_:.4e}")
print(f"  Effective df: {smooth.edf:.2f}")
print(f"  AIC: {model_reml.additional_slots['aic']:.4f}")

print(f"\nAuto GCV:")
print(f"  λ: {smooth_gcv.lambda_:.4e}")
print(f"  Effective df: {smooth_gcv.edf:.2f}")
print(f"  AIC: {model_gcv.additional_slots['aic']:.4f}")

# Example 4: Multiple smooth terms
print("\n" + "=" * 70)
print("Example 4: Multiple smooth terms")
print("=" * 70)

# Generate data with two predictors
x1 = np.linspace(0, 10, n)
x2 = np.linspace(0, 5, n)
true_mu_multi = np.sin(x1) + 0.5 * np.cos(2 * x2)
y_multi = true_mu_multi + 0.3 * np.random.randn(n)

data_multi = {"x1": x1, "x2": x2, "y": y_multi}

model_multi = auto_smooth(
    formula="y ~ pb(x1, df=10) + pb(x2, df=8)",
    family="NO",
    data=data_multi,
    method="REML",
    verbose=False
)

print(f"\nMultiple smooth terms:")
smooth_fits_multi = model_multi.additional_slots["smooth_fits"]
for i, smooth in enumerate(smooth_fits_multi["mu"]):
    print(f"  Smooth {i+1} ({smooth.variable}):")
    print(f"    λ: {smooth.lambda_:.4e}")
    print(f"    Effective df: {smooth.edf:.2f}")

print(f"\nModel AIC: {model_multi.additional_slots['aic']:.4f}")

# Example 5: Heteroscedastic model (mu and sigma smooths)
print("\n" + "=" * 70)
print("Example 5: Heteroscedastic model (mu and sigma smooths)")
print("=" * 70)

# Generate heteroscedastic data
x_hetero = np.linspace(0, 10, n)
mu_hetero = np.sin(x_hetero)
sigma_hetero = 0.1 + 0.3 * (x_hetero / 10)  # Increasing variance
y_hetero = mu_hetero + sigma_hetero * np.random.randn(n)

data_hetero = {"x": x_hetero, "y": y_hetero}

model_hetero = auto_smooth(
    formula="y ~ pb(x, df=10)",
    sigma_formula="~ pb(x, df=5)",
    family="NO",
    data=data_hetero,
    method="REML",
    verbose=False
)

print(f"\nHeteroscedastic model:")
smooth_fits_hetero = model_hetero.additional_slots["smooth_fits"]
print(f"  μ parameter:")
print(f"    λ: {smooth_fits_hetero['mu'][0].lambda_:.4e}")
print(f"    Effective df: {smooth_fits_hetero['mu'][0].edf:.2f}")
print(f"  σ parameter:")
print(f"    λ: {smooth_fits_hetero['sigma'][0].lambda_:.4e}")
print(f"    Effective df: {smooth_fits_hetero['sigma'][0].edf:.2f}")

print(f"\nModel AIC: {model_hetero.additional_slots['aic']:.4f}")

# Visualization
print("\n" + "=" * 70)
print("Creating visualization...")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: REML vs GCV vs Manual
ax = axes[0, 0]
ax.scatter(x, y, alpha=0.3, s=20, label="Data")
ax.plot(x, true_mu, 'k--', linewidth=2, label="True function")
ax.plot(x, np.asarray(model_reml.fitted_values["mu"]), 'r-', linewidth=2, label=f"REML (λ={smooth.lambda_:.2e})")
ax.plot(x, np.asarray(model_gcv.fitted_values["mu"]), 'b-', linewidth=2, label=f"GCV (λ={smooth_gcv.lambda_:.2e})")
ax.plot(x, np.asarray(model_manual.fitted_values["mu"]), 'g-', linewidth=2, label="Manual (λ=1.0)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Comparison: REML vs GCV vs Manual λ")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Multiple smooth terms
ax = axes[0, 1]
ax.scatter(x1, y_multi, alpha=0.3, s=20, label="Data")
ax.plot(x1, true_mu_multi, 'k--', linewidth=2, label="True function")
ax.plot(x1, np.asarray(model_multi.fitted_values["mu"]), 'r-', linewidth=2, label="Fitted (auto λ)")
ax.set_xlabel("x1")
ax.set_ylabel("y")
ax.set_title("Multiple Smooth Terms")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Heteroscedastic model - mu
ax = axes[1, 0]
ax.scatter(x_hetero, y_hetero, alpha=0.3, s=20, label="Data")
ax.plot(x_hetero, mu_hetero, 'k--', linewidth=2, label="True μ")
ax.plot(x_hetero, np.asarray(model_hetero.fitted_values["mu"]), 'r-', linewidth=2, label="Fitted μ")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Heteroscedastic Model: μ(x)")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Heteroscedastic model - sigma
ax = axes[1, 1]
ax.plot(x_hetero, sigma_hetero, 'k--', linewidth=2, label="True σ")
ax.plot(x_hetero, np.asarray(model_hetero.fitted_values["sigma"]), 'r-', linewidth=2, label="Fitted σ")
ax.set_xlabel("x")
ax.set_ylabel("σ")
ax.set_title("Heteroscedastic Model: σ(x)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("examples/auto_smooth_demo.png", dpi=150, bbox_inches="tight")
print("\nPlot saved to: examples/auto_smooth_demo.png")

# Summary
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print("\nKey takeaways:")
print("1. auto_smooth() automatically selects optimal λ values")
print("2. REML is the default method (more robust)")
print("3. GCV is faster for large datasets")
print("4. Automatic selection usually outperforms manual λ")
print("5. Works seamlessly with multiple smooth terms")
print("6. Supports heteroscedastic models (smooths in μ and σ)")

print("\n" + "=" * 70)
print("Demo complete!")
print("=" * 70)
