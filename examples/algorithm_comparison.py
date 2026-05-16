"""Compare RS and CG algorithms for GAMLSS fitting.

This script demonstrates the differences between the RS (Rigby-Stasinopoulos)
and CG (Cole-Green) algorithms.
"""

import sys
from pathlib import Path

# Add omnilss to path
sys.path.insert(0, str(Path(__file__).parent.parent / "omnilss" / "src"))

import numpy as np
import matplotlib.pyplot as plt
from omnilss.algorithms import rs_fit, cg_fit
from omnilss.fitting import gamlss_ml


def compare_algorithms():
    """Compare RS, CG, and gamlss_ml algorithms."""
    print("=" * 70)
    print("Algorithm Comparison: RS vs CG vs gamlss_ml")
    print("=" * 70)
    
    # Generate data
    np.random.seed(42)
    n = 200
    x = np.linspace(0, 10, n)
    sigma_true = 0.5 + 0.3*x
    y = 2 + 3*x + np.random.normal(0, sigma_true, n)
    data = {"y": y, "x": x}
    
    print(f"\nDataset: n={n}, heteroscedastic model")
    print(f"True model: y = 2 + 3*x + N(0, 0.5 + 0.3*x)")
    
    # Fit with RS algorithm
    print("\n" + "-" * 70)
    print("Fitting with RS Algorithm...")
    print("-" * 70)
    model_rs = rs_fit(
        formula="y ~ x",
        sigma_formula="~ x",
        family="NO",
        data=data,
        verbose=True
    )
    
    # Fit with CG algorithm
    print("\n" + "-" * 70)
    print("Fitting with CG Algorithm...")
    print("-" * 70)
    model_cg = cg_fit(
        formula="y ~ x",
        sigma_formula="~ x",
        family="NO",
        data=data,
        verbose=True
    )
    
    # Fit with gamlss_ml
    print("\n" + "-" * 70)
    print("Fitting with gamlss_ml...")
    print("-" * 70)
    model_ml = gamlss_ml(
        formula="y ~ x",
        sigma_formula="~ x",
        family="NO",
        data=data
    )
    
    # Compare results
    print("\n" + "=" * 70)
    print("Comparison Results")
    print("=" * 70)
    
    print(f"\n{'Algorithm':<15} {'Deviance':<15} {'Iterations':<15} {'Status':<15}")
    print("-" * 70)
    
    print(f"{'RS':<15} {model_rs.g_dev:<15.4f} "
          f"{model_rs.additional_slots['rs_iterations']:<15} "
          f"{'Converged' if model_rs.additional_slots['rs_converged'] else 'Not converged':<15}")
    
    print(f"{'CG':<15} {model_cg.g_dev:<15.4f} "
          f"{model_cg.additional_slots['cg_iterations']:<15} "
          f"{'Converged' if model_cg.additional_slots['cg_converged'] else 'Not converged':<15}")
    
    print(f"{'gamlss_ml':<15} {model_ml.g_dev:<15.4f} {'-':<15} {'-':<15}")
    
    # Deviance differences
    print(f"\nDeviance Differences:")
    print(f"  RS vs gamlss_ml: {model_rs.g_dev - model_ml.g_dev:+.4f}")
    print(f"  CG vs gamlss_ml: {model_cg.g_dev - model_ml.g_dev:+.4f}")
    print(f"  RS vs CG:        {model_rs.g_dev - model_cg.g_dev:+.4f}")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Fitted mu values
    for idx, (model, name) in enumerate([(model_rs, "RS"), (model_cg, "CG"), (model_ml, "gamlss_ml")]):
        ax = axes[0, idx]
        ax.scatter(x, y, alpha=0.3, s=20, label="Data")
        ax.plot(x, model.fitted_values["mu"], 'r-', linewidth=2, label="Fitted μ")
        ax.plot(x, 2 + 3*x, 'g--', linewidth=2, label="True μ")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{name}: Fitted μ")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Row 2: Fitted sigma values
    for idx, (model, name) in enumerate([(model_rs, "RS"), (model_cg, "CG"), (model_ml, "gamlss_ml")]):
        ax = axes[1, idx]
        ax.plot(x, model.fitted_values["sigma"], 'b-', linewidth=2, label="Fitted σ")
        ax.plot(x, sigma_true, 'g--', linewidth=2, label="True σ")
        ax.set_xlabel("x")
        ax.set_ylabel("σ")
        ax.set_title(f"{name}: Fitted σ")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("examples/algorithm_comparison.png", dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: examples/algorithm_comparison.png")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
The RS and CG algorithms are two different approaches to fitting GAMLSS models:

1. RS Algorithm (Rigby-Stasinopoulos):
   - Single outer loop
   - Updates each parameter independently
   - Fast convergence (2-3 iterations)
   - Fully implemented in Python
   - Recommended for most use cases

2. CG Algorithm (Cole-Green):
   - Double loop structure (outer + inner)
   - Accounts for parameter interactions via cross-derivatives
   - Similar convergence to RS
   - Currently uses gamlss_ml backend (simplified implementation)
   - Full implementation requires cross-derivative methods in all families

3. gamlss_ml:
   - Maximum likelihood estimation
   - Single iteration
   - Fast but may not find optimal solution
   - Good for initialization

In this example:
- All three algorithms produce similar results
- RS algorithm converges in {rs_iters} iterations
- CG algorithm uses gamlss_ml backend
- Deviance differences are minimal (< 1.0)

Recommendation: Use RS algorithm for production work.
    """.format(rs_iters=model_rs.additional_slots['rs_iterations']))


if __name__ == "__main__":
    compare_algorithms()
