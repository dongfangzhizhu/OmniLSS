"""Demonstration of Mixed algorithm for GAMLSS fitting.

This script shows how to use the Mixed algorithm which intelligently
selects between RS and CG algorithms.
"""

import sys
from pathlib import Path

# Add omnilss to path
sys.path.insert(0, str(Path(__file__).parent.parent / "omnilss" / "src"))

import numpy as np
import matplotlib.pyplot as plt
from omnilss.algorithms import mixed_fit, compare_algorithms


def demo_mixed_auto():
    """Demonstrate automatic algorithm selection."""
    print("=" * 70)
    print("Demo 1: Automatic Algorithm Selection")
    print("=" * 70)
    
    # Generate data
    np.random.seed(42)
    n = 200
    x = np.linspace(0, 10, n)
    sigma_true = 0.5 + 0.3*x
    y = 2 + 3*x + np.random.normal(0, sigma_true, n)
    data = {"y": y, "x": x}
    
    # Fit with automatic selection
    model = mixed_fit(
        formula="y ~ x",
        sigma_formula="~ x",
        family="NO",
        data=data,
        algorithm="auto",
        verbose=True
    )
    
    print(f"\nSelected algorithm: {model.additional_slots['mixed_algorithm_used'].upper()}")
    print(f"Auto-selected: {model.additional_slots['mixed_auto_selected']}")
    print(f"Final deviance: {model.g_dev:.4f}")


def demo_explicit_selection():
    """Demonstrate explicit algorithm selection."""
    print("\n" + "=" * 70)
    print("Demo 2: Explicit Algorithm Selection")
    print("=" * 70)
    
    # Generate data
    np.random.seed(42)
    n = 150
    x = np.linspace(0, 10, n)
    y = 2 + 3*x + np.random.normal(0, 1, n)
    data = {"y": y, "x": x}
    
    # Fit with RS
    print("\nUsing RS algorithm explicitly:")
    model_rs = mixed_fit(
        formula="y ~ x",
        sigma_formula="~ x",
        family="NO",
        data=data,
        algorithm="rs",
        verbose=False
    )
    print(f"  Deviance: {model_rs.g_dev:.4f}")
    print(f"  Iterations: {model_rs.additional_slots['rs_iterations']}")
    
    # Fit with CG
    print("\nUsing CG algorithm explicitly:")
    model_cg = mixed_fit(
        formula="y ~ x",
        sigma_formula="~ x",
        family="NO",
        data=data,
        algorithm="cg",
        verbose=False
    )
    print(f"  Deviance: {model_cg.g_dev:.4f}")
    print(f"  Note: {model_cg.additional_slots.get('cg_note', 'N/A')}")


def demo_comparison():
    """Demonstrate algorithm comparison."""
    print("\n" + "=" * 70)
    print("Demo 3: Algorithm Comparison")
    print("=" * 70)
    
    # Generate data
    np.random.seed(42)
    n = 200
    x = np.linspace(0, 10, n)
    sigma_true = 0.5 + 0.3*x
    y = 2 + 3*x + np.random.normal(0, sigma_true, n)
    data = {"y": y, "x": x}
    
    # Compare algorithms
    models = compare_algorithms(
        formula="y ~ x",
        sigma_formula="~ x",
        family="NO",
        data=data,
        verbose=True
    )
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot fitted mu
    ax = axes[0]
    ax.scatter(x, y, alpha=0.3, s=20, label="Data")
    ax.plot(x, models["rs"].fitted_values["mu"], 'r-', linewidth=2, label="RS")
    ax.plot(x, models["cg"].fitted_values["mu"], 'b--', linewidth=2, label="CG")
    ax.plot(x, 2 + 3*x, 'g:', linewidth=2, label="True")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Fitted μ Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot fitted sigma
    ax = axes[1]
    ax.plot(x, models["rs"].fitted_values["sigma"], 'r-', linewidth=2, label="RS")
    ax.plot(x, models["cg"].fitted_values["sigma"], 'b--', linewidth=2, label="CG")
    ax.plot(x, sigma_true, 'g:', linewidth=2, label="True")
    ax.set_xlabel("x")
    ax.set_ylabel("σ")
    ax.set_title("Fitted σ Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("examples/mixed_algorithm_demo.png", dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: examples/mixed_algorithm_demo.png")


def demo_summary():
    """Print summary of Mixed algorithm."""
    print("\n" + "=" * 70)
    print("Mixed Algorithm Summary")
    print("=" * 70)
    print("""
The Mixed algorithm provides a unified interface for GAMLSS fitting:

1. Automatic Selection (algorithm="auto"):
   - Currently selects RS algorithm (recommended)
   - Future versions may implement intelligent selection based on:
     * Data characteristics
     * Model complexity
     * Convergence behavior

2. Explicit RS (algorithm="rs"):
   - Uses the fully implemented RS algorithm
   - Fast convergence (2-3 iterations)
   - Excellent numerical stability
   - Recommended for production use

3. Explicit CG (algorithm="cg"):
   - Uses simplified CG implementation (gamlss_ml backend)
   - Useful for comparison
   - Full CG implementation planned for future

Usage Examples:

    # Automatic selection
    model = mixed_fit("y ~ x", "~ x", family="NO", data=data)
    
    # Explicit RS
    model = mixed_fit("y ~ x", "~ x", family="NO", data=data, algorithm="rs")
    
    # Explicit CG
    model = mixed_fit("y ~ x", "~ x", family="NO", data=data, algorithm="cg")
    
    # Compare algorithms
    models = compare_algorithms("y ~ x", "~ x", family="NO", data=data)

Recommendation: Use algorithm="auto" or algorithm="rs" for production work.
    """)


if __name__ == "__main__":
    demo_mixed_auto()
    demo_explicit_selection()
    demo_comparison()
    demo_summary()
    
    print("\n" + "=" * 70)
    print("All demos completed successfully!")
    print("=" * 70)
