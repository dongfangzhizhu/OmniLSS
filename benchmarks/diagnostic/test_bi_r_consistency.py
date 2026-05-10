"""Test BI distribution consistency with R GAMLSS."""

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
from benchmarks.data_generators import generate_binomial_data


def test_bi_consistency(n_obs, n_predictors, formula):
    """Test BI consistency with R using benchmark data generator."""
    print(f"\n{'='*80}")
    print(f"Testing BI: n={n_obs}, predictors={n_predictors}, formula={formula}")
    print(f"{'='*80}")
    
    # Generate data using benchmark generator (generates proportions)
    data = generate_binomial_data(n_obs, n_predictors, bd=10, seed=42)
    
    # Ensure y is an array
    if not isinstance(data['y'], np.ndarray):
        # When n_predictors=0, y might be a scalar, convert to array
        y_val = data['y']
        data['y'] = np.full(n_obs, y_val, dtype=float)
    
    print(f"Data info:")
    print(f"  y: min={data['y'].min():.3f}, max={data['y'].max():.3f}, mean={data['y'].mean():.3f}")
    print(f"  bd: {data['bd'][0]:.0f}")
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
        print(f"  Python mu coefficients: {py_model.coefficients['mu']}")
        print(f"  Python fitted mu (first 5): {py_model.fitted_values['mu'][:5]}")
        py_success = True
    except Exception as e:
        print(f"  Python ERROR: {e}")
        py_success = False
        py_model = None
    
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
        print(f"  R mu coefficients: {r_result['coefficients']['mu']}")
        print(f"  R fitted mu (first 5): {r_result['fitted_values']['mu'][:5]}")
        r_success = True
    except Exception as e:
        print(f"  R ERROR: {e}")
        r_success = False
        r_result = None
    
    print()
    
    # Compare
    if py_success and r_success:
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
        
        # Verdict
        if dev_rel_diff < 1.0:
            print("PASS: Deviance difference < 1%")
        else:
            print(f"FAIL: Deviance difference {dev_rel_diff:.2f}% > 1%")
        
        return dev_rel_diff
    else:
        print("FAIL: One or both models failed to fit")
        return None


def main():
    """Run BI consistency tests."""
    print("="*80)
    print("BI Distribution R Consistency Test")
    print("="*80)
    
    # Test cases matching the performance report
    test_cases = [
        (100, 0, "y ~ 1"),
        (100, 1, "y ~ x1"),
        (100, 2, "y ~ x1 + x2"),
        (500, 0, "y ~ 1"),
        (500, 1, "y ~ x1"),
        (500, 2, "y ~ x1 + x2"),
        (5000, 0, "y ~ 1"),
        (5000, 1, "y ~ x1"),
        (5000, 2, "y ~ x1 + x2"),
    ]
    
    results = []
    for n_obs, n_predictors, formula in test_cases:
        result = test_bi_consistency(n_obs, n_predictors, formula)
        results.append((n_obs, formula, result))
    
    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    
    passed = sum(1 for _, _, r in results if r is not None and r < 1.0)
    failed = sum(1 for _, _, r in results if r is None or r >= 1.0)
    
    print(f"Passed: {passed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    print()
    
    for n_obs, formula, result in results:
        if result is not None:
            status = "PASS" if result < 1.0 else "FAIL"
            print(f"{status} n={n_obs:5d}, {formula:20s}: {result:6.2f}%")
        else:
            print(f"FAIL n={n_obs:5d}, {formula:20s}: FAILED")
    
    print("="*80)


if __name__ == "__main__":
    main()
