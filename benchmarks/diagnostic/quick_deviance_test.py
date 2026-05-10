"""Quick deviance test for BI and ZAGA."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "omnilss" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "omnilss" / "tests"))

from omnilss.distributions import resolve_family
from omnilss.fitting import gamlss
from rbus.r_bridge import RBridge


def test_distribution(dist_name, data, formula="y ~ x1", sigma_formula="~1"):
    """Test a single distribution."""
    print(f"\nTesting {dist_name}...")
    
    # Fit Python model
    family = resolve_family(dist_name)
    py_model = gamlss(
        formula=formula,
        sigma_formula=sigma_formula,
        family=family,
        data=data,
    )
    
    # Fit R model
    bridge = RBridge()
    r_result = bridge.call_r_gamlss(
        data=data,
        formula=formula,
        family=dist_name,
        sigma_formula=sigma_formula,
    )
    
    # Compare
    dev_diff = abs(py_model.deviance - r_result['deviance'])
    dev_rel_diff = dev_diff / r_result['deviance'] * 100
    
    print(f"  Python deviance: {py_model.deviance:.6f}")
    print(f"  R deviance: {r_result['deviance']:.6f}")
    print(f"  Difference: {dev_diff:.6f} ({dev_rel_diff:.2f}%)")
    
    return dev_rel_diff


def main():
    """Run quick deviance tests."""
    print("=" * 80)
    print("Quick Deviance Test for BI and ZAGA")
    print("=" * 80)
    
    # Test BI
    print("\n" + "=" * 80)
    print("BI Distribution")
    print("=" * 80)
    
    np.random.seed(42)
    n = 500
    x1 = np.random.randn(n)
    true_mu = 1 / (1 + np.exp(-(0.5 + 0.8 * x1)))
    y_bi = (np.random.rand(n) < true_mu).astype(float)
    data_bi = {"y": y_bi, "x1": x1}
    
    bi_diff = test_distribution("BI", data_bi)
    
    # Test ZAGA
    print("\n" + "=" * 80)
    print("ZAGA Distribution")
    print("=" * 80)
    
    np.random.seed(42)
    n = 500
    x1 = np.random.randn(n)
    true_nu = 0.3
    true_mu = np.exp(0.5 + 0.5 * x1)
    true_sigma = 0.5
    is_zero = np.random.rand(n) < true_nu
    shape = 1 / (true_sigma ** 2)
    scale = true_mu * (true_sigma ** 2)
    y_zaga = np.where(is_zero, 0, np.random.gamma(shape, scale, n))
    data_zaga = {"y": y_zaga, "x1": x1}
    
    zaga_diff = test_distribution("ZAGA", data_zaga)
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"BI deviance difference: {bi_diff:.2f}%")
    print(f"ZAGA deviance difference: {zaga_diff:.2f}%")
    print()
    
    if bi_diff < 1.0 and zaga_diff < 5.0:
        print("✓ All tests passed! Deviance differences are within acceptable range.")
    else:
        print("✗ Some tests failed. Deviance differences are too large.")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
