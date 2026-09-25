"""Demonstration of the HYBRID algorithm dispatcher.

HYBRID routes to one RS or L-BFGS backend based on data characteristics. It is
separate from ``mixed_fit``, which implements R's sequential RS-to-CG
``mixed(n1, n2)`` algorithm.
"""

# ruff: noqa: E402

import importlib.util
import sys
from pathlib import Path

# Add omnilss to path
sys.path.insert(0, str(Path(__file__).parent.parent / "omnilss" / "src"))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import numpy as np
from omnilss.algorithms import hybrid_fit, compare_algorithms


def demo_mixed_auto():
    """Demonstrate automatic algorithm selection."""
    print("=" * 70)
    print("Demo 1: Automatic Algorithm Selection")
    print("=" * 70)

    # Generate data
    np.random.seed(42)
    n = 200
    x = np.linspace(0, 10, n)
    sigma_true = 0.5 + 0.3 * x
    y = 2 + 3 * x + np.random.normal(0, sigma_true, n)
    data = {"y": y, "x": x}

    # Fit with automatic selection
    model = hybrid_fit(
        formula="y ~ x",
        sigma_formula="~ x",
        family="NO",
        data=data,
        algorithm="auto",
        verbose=True,
    )

    print(
        f"\nSelected algorithm: {model.additional_slots['hybrid_algorithm_used'].upper()}"
    )
    print(f"Auto-selected: {model.additional_slots['hybrid_auto_selected']}")
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
    y = 2 + 3 * x + np.random.normal(0, 1, n)
    data = {"y": y, "x": x}

    # Fit with RS
    print("\nUsing RS algorithm explicitly:")
    model_rs = hybrid_fit(
        formula="y ~ x",
        sigma_formula="~ x",
        family="NO",
        data=data,
        algorithm="rs",
        verbose=False,
    )
    print(f"  Deviance: {model_rs.g_dev:.4f}")
    print(f"  Iterations: {model_rs.additional_slots['rs_iterations']}")

    # Fit with CG
    print("\nUsing CG algorithm explicitly:")
    model_cg = hybrid_fit(
        formula="y ~ x",
        sigma_formula="~ x",
        family="NO",
        data=data,
        algorithm="cg",
        verbose=False,
    )
    print(f"  Deviance: {model_cg.g_dev:.4f}")
    print(
        f"  Iterations: {model_cg.additional_slots.get('cg_iterations', model_cg.iter)}"
    )
    print(f"  Backend: {model_cg.additional_slots.get('cg_backend', 'N/A')}")
    print(
        "  Cross derivatives: "
        f"{model_cg.additional_slots.get('cg_cross_derivatives', 'N/A')}"
    )


def demo_comparison():
    """Demonstrate algorithm comparison."""
    print("\n" + "=" * 70)
    print("Demo 3: Algorithm Comparison")
    print("=" * 70)

    # Generate data
    np.random.seed(42)
    n = 200
    x = np.linspace(0, 10, n)
    sigma_true = 0.5 + 0.3 * x
    y = 2 + 3 * x + np.random.normal(0, sigma_true, n)
    data = {"y": y, "x": x}

    # Compare algorithms
    models = compare_algorithms(
        formula="y ~ x", sigma_formula="~ x", family="NO", data=data, verbose=True
    )

    # Plot comparison when matplotlib is available.
    if importlib.util.find_spec("matplotlib") is None:
        print("\nmatplotlib is not installed; skipping plot generation.")
    else:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot fitted mu
        ax = axes[0]
        ax.scatter(x, y, alpha=0.3, s=20, label="Data")
        ax.plot(x, models["rs"].fitted_values["mu"], "r-", linewidth=2, label="RS")
        ax.plot(x, models["cg"].fitted_values["mu"], "b--", linewidth=2, label="CG")
        ax.plot(x, 2 + 3 * x, "g:", linewidth=2, label="True")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Fitted μ Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot fitted sigma
        ax = axes[1]
        ax.plot(x, models["rs"].fitted_values["sigma"], "r-", linewidth=2, label="RS")
        ax.plot(x, models["cg"].fitted_values["sigma"], "b--", linewidth=2, label="CG")
        ax.plot(x, sigma_true, "g:", linewidth=2, label="True")
        ax.set_xlabel("x")
        ax.set_ylabel("σ")
        ax.set_title("Fitted σ Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("examples/mixed_algorithm_demo.png", dpi=150, bbox_inches="tight")
        print("\nPlot saved to: examples/mixed_algorithm_demo.png")


def demo_summary():
    """Print a summary of the HYBRID dispatcher."""
    print("\n" + "=" * 70)
    print("HYBRID Dispatcher Summary")
    print("=" * 70)
    print("""
HYBRID provides a single-backend dispatcher for GAMLSS fitting:

1. Automatic selection (algorithm="auto") chooses RS or L-BFGS from sample
   size, family and smooth-term constraints.
2. algorithm="rs" forces the standard RS implementation.
3. algorithm="lbfgs" forces the experimental L-BFGS compatibility backend.
   The legacy algorithm="cg" spelling is deprecated and maps to L-BFGS.

Use gamlss(..., method="MIXED", mixed_n1=1, mixed_n2=20) when you need
R-compatible sequential RS-to-CG fitting.
    """)


if __name__ == "__main__":
    demo_mixed_auto()
    demo_explicit_selection()
    demo_comparison()
    demo_summary()

    print("\n" + "=" * 70)
    print("All demos completed successfully!")
    print("=" * 70)
