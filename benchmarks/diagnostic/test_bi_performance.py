"""Quick test to verify BI distribution performance and deviance."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import time

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "omnilss" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "omnilss" / "tests"))
sys.path.insert(0, str(Path(__file__).parent.parent / "performance"))

from omnilss.distributions import resolve_family
from omnilss.fitting import gamlss
from rbus.r_bridge import RBridge
from benchmarks.data_generators import generate_binomial_data


def test_bi_performance():
    """Test BI performance and deviance consistency."""
    print("="*80)
    print("BI Distribution Performance Test")
    print("="*80)
    print()
    
    test_cases = [
        (100, 0, "y ~ 1"),
        (500, 1, "y ~ x1"),
        (5000, 2, "y ~ x1 + x2"),
    ]
    
    results = []
    
    for n_obs, n_predictors, formula in test_cases:
        print(f"\nTest: n={n_obs}, predictors={n_predictors}, formula={formula}")
        print("-" * 80)
        
        # Generate data
        data = generate_binomial_data(n_obs, n_predictors, bd=10, seed=42)
        
        # Ensure y is an array
        if not isinstance(data['y'], np.ndarray):
            y_val = data['y']
            data['y'] = np.full(n_obs, y_val, dtype=float)
        
        # Fit Python model
        family = resolve_family("BI")
        
        start_time = time.time()
        py_model = gamlss(
            formula=formula,
            sigma_formula="~1",
            family=family,
            data=data,
        )
        py_time = time.time() - start_time
        
        # Fit R model
        bridge = RBridge()
        
        start_time = time.time()
        r_result = bridge.call_r_gamlss(
            data=data,
            formula=formula,
            family="BI",
            sigma_formula="~1",
        )
        r_time = time.time() - start_time
        
        # Calculate metrics
        dev_diff = abs(py_model.deviance - r_result['deviance'])
        dev_rel_diff = dev_diff / r_result['deviance'] * 100
        speedup = r_time / py_time
        
        results.append({
            'n': n_obs,
            'formula': formula,
            'py_dev': py_model.deviance,
            'r_dev': r_result['deviance'],
            'dev_diff': dev_diff,
            'dev_rel_diff': dev_rel_diff,
            'py_time': py_time,
            'r_time': r_time,
            'speedup': speedup,
        })
        
        print(f"  Python deviance: {py_model.deviance:.6f}")
        print(f"  R deviance: {r_result['deviance']:.6f}")
        print(f"  Deviance diff: {dev_diff:.6f} ({dev_rel_diff:.4f}%)")
        print(f"  Python time: {py_time:.4f}s")
        print(f"  R time: {r_time:.4f}s")
        print(f"  Speedup: {speedup:.2f}x")
        
        if dev_rel_diff < 1.0:
            print(f"  ✓ PASS")
        else:
            print(f"  ✗ FAIL")
    
    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    
    passed = sum(1 for r in results if r['dev_rel_diff'] < 1.0)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")
    print()
    
    print("Deviance Differences:")
    for r in results:
        status = "✓" if r['dev_rel_diff'] < 1.0 else "✗"
        print(f"  {status} n={r['n']:5d}, {r['formula']:20s}: {r['dev_rel_diff']:8.4f}%")
    
    print()
    print("Performance (Speedup):")
    for r in results:
        print(f"  n={r['n']:5d}, {r['formula']:20s}: {r['speedup']:6.2f}x")
    
    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    print(f"\n  Average speedup: {avg_speedup:.2f}x")
    
    print("\n" + "="*80)
    
    if passed == total:
        print("✓ All tests PASSED!")
    else:
        print(f"✗ {total - passed} test(s) FAILED")
    
    print("="*80)


if __name__ == "__main__":
    test_bi_performance()
