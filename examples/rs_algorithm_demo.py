"""Demonstration of RS (Rigby-Stasinopoulos) Algorithm.

This script demonstrates the complete RS algorithm implementation,
showing how it iteratively updates distribution parameters until
convergence.
"""

import numpy as np
import matplotlib.pyplot as plt
from omnilss.algorithms import rs_fit
from omnilss.fitting import gamlss_ml

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data with heteroscedastic noise
n = 200
x = np.linspace(0, 10, n)
true_mu = 2 + 3 * x
true_sigma = 0.5 + 0.3 * x  # Increasing variance
y = np.random.normal(true_mu, true_sigma)

data = {"y": y, "x": x}

print("=" * 70)
print("RS Algorithm Demonstration")
print("=" * 70)
print(f"Sample size: {n}")
print(f"True model: μ = 2 + 3*x, σ = 0.5 + 0.3*x")
print("=" * 70)

# Fit model using RS algorithm with verbose output
print("\n1. Fitting with RS Algorithm (verbose):")
print("-" * 70)
model_rs = rs_fit(
    formula="y ~ x",
    sigma_formula="~ x",
    family="NO",
    data=data,
    verbose=True
)

print("\n2. Model Summary:")
print("-" * 70)
print(f"Converged: {model_rs.additional_slots['rs_converged']}")
print(f"Iterations: {model_rs.additional_slots['rs_iterations']}")
print(f"Global Deviance: {model_rs.g_dev:.4f}")
print(f"AIC: {model_rs.additional_slots['aic']:.4f}")
print(f"SBC: {model_rs.additional_slots['sbc']:.4f}")

print("\n3. Parameter Estimates:")
print("-" * 70)
mu_coef = np.asarray(model_rs.coefficients["mu"])
sigma_coef = np.asarray(model_rs.coefficients["sigma"])
print(f"μ coefficients: {mu_coef}")
print(f"σ coefficients: {sigma_coef}")

# Compare with gamlss_ml
print("\n4. Comparison with gamlss_ml:")
print("-" * 70)
model_ml = gamlss_ml(
    formula="y ~ x",
    sigma_formula="~ x",
    family="NO",
    data=data
)
print(f"RS Global Deviance: {model_rs.g_dev:.4f}")
print(f"ML Global Deviance: {model_ml.g_dev:.4f}")
print(f"Difference: {abs(model_rs.g_dev - model_ml.g_dev):.6f}")

# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Fitted mean
ax = axes[0, 0]
ax.scatter(x, y, alpha=0.5, s=20, label="Data")
ax.plot(x, true_mu, "g--", linewidth=2, label="True μ")
ax.plot(x, np.asarray(model_rs.fitted_values["mu"]), "r-", linewidth=2, label="Fitted μ (RS)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Mean Function")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Fitted sigma
ax = axes[0, 1]
ax.plot(x, true_sigma, "g--", linewidth=2, label="True σ")
ax.plot(x, np.asarray(model_rs.fitted_values["sigma"]), "r-", linewidth=2, label="Fitted σ (RS)")
ax.set_xlabel("x")
ax.set_ylabel("σ")
ax.set_title("Scale Function")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Residuals
ax = axes[1, 0]
residuals = y - np.asarray(model_rs.fitted_values["mu"])
ax.scatter(x, residuals, alpha=0.5, s=20)
ax.axhline(y=0, color="r", linestyle="--")
ax.set_xlabel("x")
ax.set_ylabel("Residuals")
ax.set_title("Residuals vs x")
ax.grid(True, alpha=0.3)

# Plot 4: Q-Q plot
ax = axes[1, 1]
from scipy import stats
standardized_residuals = residuals / np.asarray(model_rs.fitted_values["sigma"])
stats.probplot(standardized_residuals, dist="norm", plot=ax)
ax.set_title("Q-Q Plot (Standardized Residuals)")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("examples/rs_algorithm_demo.png", dpi=150, bbox_inches="tight")
print("\n5. Plot saved to: examples/rs_algorithm_demo.png")

print("\n" + "=" * 70)
print("RS Algorithm demonstration complete!")
print("=" * 70)
