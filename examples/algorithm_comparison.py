"""Compare RS and current CG backends for GAMLSS fitting.

This script demonstrates the current OmniLSS algorithm choices:

- RS: Rigby-Stasinopoulos IRLS, the default CPU workhorse.
- CG_FULL_HESSIAN: default Cole-Green backend for ``method="CG"``; it uses
  full observed-information Hessian blocks and records cross-derivative
  diagnostics.
- CG_IRLS_CROSS: experimental eta-scale IRLS cross-derivative backend selected
  with ``cg_backend="irls_cross"``.
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

from omnilss.controls import gamlss_control
from omnilss.fitting import gamlss


def _status(model):
    """Return a compact convergence/backend status for a fitted model."""
    slots = model.additional_slots
    converged = slots.get("cg_converged", slots.get("rs_converged", False))
    backend = slots.get("cg_backend", slots.get("method", "RS"))
    cross = slots.get("cg_cross_derivatives", "n/a")
    return backend, cross, "Converged" if converged else "Not converged"


def compare_algorithms():
    """Compare RS, CG_FULL_HESSIAN, and CG_IRLS_CROSS algorithms."""
    print("=" * 88)
    print("Algorithm Comparison: RS vs CG_FULL_HESSIAN vs CG_IRLS_CROSS")
    print("=" * 88)

    # Generate data
    np.random.seed(42)
    n = 200
    x = np.linspace(0, 10, n)
    sigma_true = 0.5 + 0.3 * x
    y = 2 + 3 * x + np.random.normal(0, sigma_true, n)
    data = {"y": y, "x": x}
    control = gamlss_control(n_cyc=12, c_crit=1e-4, trace=False)

    print(f"\nDataset: n={n}, heteroscedastic Normal model")
    print("True model: y = 2 + 3*x + N(0, 0.5 + 0.3*x)")

    fit_specs = [
        (
            "RS",
            {
                "method": "RS",
            },
        ),
        (
            "CG_FULL_HESSIAN",
            {
                "method": "CG",
                "cg_backend": "full_hessian",
            },
        ),
        (
            "CG_IRLS_CROSS",
            {
                "method": "CG",
                "cg_backend": "irls_cross",
            },
        ),
    ]

    models = {}
    for name, method_kwargs in fit_specs:
        print("\n" + "-" * 88)
        print(f"Fitting with {name}...")
        print("-" * 88)
        models[name] = gamlss(
            formula="y ~ x",
            sigma_formula="~ x",
            family="NO",
            data=data,
            control=control,
            verbose=False,
            **method_kwargs,
        )

    # Compare results
    print("\n" + "=" * 88)
    print("Comparison Results")
    print("=" * 88)
    print(
        f"\n{'Algorithm':<18} {'Backend':<18} {'Cross derivs':<15} "
        f"{'Deviance':<14} {'Iterations':<11} {'Status':<15}"
    )
    print("-" * 88)

    for name, model in models.items():
        slots = model.additional_slots
        backend, cross, status = _status(model)
        iterations = slots.get("cg_iterations", slots.get("rs_iterations", model.iter))
        print(
            f"{name:<18} {backend:<18} {cross:<15} "
            f"{model.g_dev:<14.4f} {iterations!s:<11} {status:<15}"
        )

    print("\nDeviance Differences:")
    rs_dev = models["RS"].g_dev
    for name in ["CG_FULL_HESSIAN", "CG_IRLS_CROSS"]:
        print(f"  RS - {name:<15}: {rs_dev - models[name].g_dev:+.4f}")

    # Plot comparison when matplotlib is available.
    if importlib.util.find_spec("matplotlib") is None:
        print("\nmatplotlib is not installed; skipping plot generation.")
    else:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Row 1: Fitted mu values
        for idx, (name, model) in enumerate(models.items()):
            ax = axes[0, idx]
            ax.scatter(x, y, alpha=0.3, s=20, label="Data")
            ax.plot(x, model.fitted_values["mu"], "r-", linewidth=2, label="Fitted μ")
            ax.plot(x, 2 + 3 * x, "g--", linewidth=2, label="True μ")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"{name}: Fitted μ")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Row 2: Fitted sigma values
        for idx, (name, model) in enumerate(models.items()):
            ax = axes[1, idx]
            ax.plot(
                x, model.fitted_values["sigma"], "b-", linewidth=2, label="Fitted σ"
            )
            ax.plot(x, sigma_true, "g--", linewidth=2, label="True σ")
            ax.set_xlabel("x")
            ax.set_ylabel("σ")
            ax.set_title(f"{name}: Fitted σ")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output = Path("examples/algorithm_comparison.png")
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to: {output}")

    # Summary
    print("\n" + "=" * 88)
    print("Summary")
    print("=" * 88)
    print(
        """
Current OmniLSS fitting guidance:

1. RS Algorithm (Rigby-Stasinopoulos)
   - Default NumPy IRLS path.
   - Recommended for routine production fitting on CPU.
   - Records rs_iterations / rs_converged diagnostics.

2. CG_FULL_HESSIAN (default method="CG")
   - Cole-Green correctness backend.
   - Uses full coefficient-level observed information with cross-parameter
     Hessian blocks instead of a block-diagonal RS approximation.
   - Records cg_backend="CG_FULL_HESSIAN" and
     cg_cross_derivatives="full_hessian" diagnostics.

3. CG_IRLS_CROSS (method="CG", cg_backend="irls_cross")
   - Experimental eta-scale IRLS backend with cross-derivative corrections.
   - Keeps an RS-style weighted least-squares structure while exposing
     cg_cross_derivatives="eta_correction" diagnostics.
   - Useful for experimentation and future large-design work; validate against
     CG_FULL_HESSIAN for correctness-sensitive workflows.

Recommendation: use RS for routine work, use CG_FULL_HESSIAN when you need an
auditable Cole-Green cross-derivative path, and use CG_IRLS_CROSS only when you
explicitly want the experimental eta-level backend.
        """.strip()
    )


if __name__ == "__main__":
    compare_algorithms()
