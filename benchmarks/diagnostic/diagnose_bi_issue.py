"""Diagnose BI deviance issue from performance report."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "omnilss" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "omnilss" / "tests"))
sys.path.insert(0, str(Path(__file__).parent.parent / "performance"))

from omnilss.distributions import resolve_family
from omnilss.fitting import gamlss
from rbus.r_bridge import RBridge
from benchmarks.data_generators import generate_data_for_distribution


def test_bi_with_benchmark_data(n_obs, n_predictors, formula):
    """Test BI using the same data generator as benchmarks."""
    print(f"\n{'='*80}")
    print(f"Testing BI: n={n_obs}, formula={formula}")
    print(f"{'='*80}")
    
    # Generate data using benchmark generator
    data = generate_data_for_distribution("BI", n_obs, n_predictors, seed=42)
    
    print(f"Data generated:")
    print(f"  y: min={data['y'].min():.3f}, max={data['y'].max():.3f}, mean={data['y'].mean():.3f}")
    print(f"  y unique values: {np.unique(data['y'])}")
    if 'bd' in data:
        print(f"  bd: min={data['bd'].min():.3f}, max={data['bd'].max():.3f}, mean={data['bd'].mean():.3f}")
    print()
    
    # Fit Python model
    print("Fitting Python model...")
    family = resolve_family("BI")
    try:
        py_model = gamlss(
            formula=formula,
            sigma_formula="~1",
            family=family,
            data=data,
        )
        print(f"  Python deviance: {py_model.deviance:.6f}")
        print(f"  Python converged: {py_model.converged}")
        print(f"  Python mu coefficients: {py_model.coefficients['mu']}")
        print(f"  Python fitted mu (first 5): {py_model.fitted_values['mu'][:5]}")
    except Exception as e:
        print(f"  Python ERROR: {e}")
        return
    
    print()
    
    # Fit R model
    print("Fitting R model...")
    bridge = RBridge()
    try:
        r_result = bridge.call_r_gamlss(
            data=data,
            formula=formula,
            family="BI",
            sigma_formula="~1",
        )
        print(f"  R deviance: {r_result['deviance']:.6f}")
        print(f"  R converged: {r_result.get('converged', 'unknown')}")
        print(f"  R mu coefficients: {r_result['coefficients']['mu']}")
        print(f"  R fitted mu (first 5): {r_result['fitted_values']['mu'][:5]}")
    except Exception as e:
        print(f"  R ERROR: {e}")
        return
    
    print()
    
    # Compare
    dev_diff = abs(py_model.deviance - r_result['deviance'])
    dev_rel_diff = dev_diff / r_result['deviance'] * 100
    
    print(f"{'='*80}")
    print("Comparison")
    print(f"{'='*80}")
    print(f"Deviance difference: {dev_diff:.6f} ({dev_rel_diff:.2f}%)")
    
    # Check coefficients
    py_coef = np.array(py_model.coefficients['mu'])
    r_coef = np.array(r_result['coefficients']['mu'])
    coef_diff = np.abs(py_coef - r_coef)
    print(f"Max coefficient difference: {coef_diff.max():.6e}")
    
    # Check fitted values
    py_fitted = np.array(py_model.fitted_values['mu'])
    r_fitted = np.array(r_result['fitted_values']['mu'])
    fitted_diff = np.abs(py_fitted - r_fitted)
    print(f"Max fitted value difference: {fitted_diff.max():.6e}")
    print()


def main():
    """Run BI diagnostics."""
    print("="*80)
    print("BI Deviance Diagnostic")
    print("="*80)
    
    # Test cases from the report
    test_cases = [
        (100, 0, "y ~ 1"),
        (100, 1, "y ~ x1"),
        (100, 2, "y ~ x1 + x2"),
        (500, 0, "y ~ 1"),
        (500, 1, "y ~ x1"),
        (500, 2, "y ~ x1 + x2"),
    ]
    
    for n_obs, n_predictors, formula in test_cases:
        test_bi_with_benchmark_data(n_obs, n_predictors, formula)


if __name__ == "__main__":
    main()
