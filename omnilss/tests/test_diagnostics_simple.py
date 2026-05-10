"""Simple tests for model diagnostics module (no pytest required)."""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '../src')

from omnilss.fitting import gamlss
from omnilss import diagnostics


def test_normal_model():
    """Test diagnostics with a simple normal model."""
    print("\n" + "="*70)
    print("TEST 1: Normal Model Diagnostics")
    print("="*70)
    
    # Generate data
    np.random.seed(42)
    n = 200
    x = np.random.uniform(0, 10, n)
    y = 2.0 + 0.5 * x + np.random.normal(0, 1, n)
    
    data = pd.DataFrame({'x': x, 'y': y})
    
    # Fit model
    print("\nFitting normal model...")
    model = gamlss(
        formula="y ~ x",
        family="NO",
        data=data
    )
    print("✓ Model fitted successfully")
    
    # Test quantile residuals
    print("\n1. Testing quantile residuals...")
    qres = diagnostics.quantile_residuals(model)
    print(f"   N observations: {qres.n}")
    print(f"   Mean: {qres.mean:.4f} (should be ≈ 0)")
    print(f"   Variance: {qres.variance:.4f} (should be ≈ 1)")
    print(f"   Skewness: {qres.skewness:.4f} (should be ≈ 0)")
    print(f"   Kurtosis: {qres.kurtosis:.4f} (should be ≈ 0)")
    
    assert abs(qres.mean) < 0.2, f"Mean too far from 0: {qres.mean}"
    assert abs(qres.variance - 1.0) < 0.3, f"Variance too far from 1: {qres.variance}"
    print("   ✓ Quantile residuals look good")
    
    # Test Q-Q plot
    print("\n2. Testing Q-Q plot...")
    qq = diagnostics.qq_plot_data(model)
    print(f"   Filliben correlation: {qq.correlation:.4f} (should be ≈ 1)")
    assert qq.correlation > 0.95, f"Correlation too low: {qq.correlation}"
    print("   ✓ Q-Q plot looks good")
    
    # Test worm plot
    print("\n3. Testing worm plot...")
    worm = diagnostics.worm_plot_data(model)
    outside = np.sum((worm.deviations < worm.lower_band) |
                     (worm.deviations > worm.upper_band))
    pct_outside = 100.0 * outside / worm.n
    print(f"   Points outside 95% CI: {outside}/{worm.n} ({pct_outside:.1f}%)")
    print(f"   Expected: ~{int(0.05 * worm.n)} (5%)")
    assert pct_outside < 15, f"Too many points outside bands: {pct_outside}%"
    print("   ✓ Worm plot looks good")
    
    # Test residual plot
    print("\n4. Testing residual plot...")
    resid = diagnostics.residual_plot_data(model)
    print(f"   N residuals: {len(resid.residuals)}")
    print(f"   N fitted values: {len(resid.fitted_values)}")
    assert len(resid.residuals) == len(resid.fitted_values)
    print("   ✓ Residual plot data looks good")
    
    # Test calibration
    print("\n5. Testing calibration check...")
    calib = diagnostics.calibration_check(model, n_bins=10)
    print(f"   N bins: {calib.n_bins}")
    assert calib.n_bins == 10
    print("   ✓ Calibration check looks good")
    
    # Test centile check
    print("\n6. Testing centile check...")
    centiles = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    centile = diagnostics.centile_check(model, centiles=centiles)
    print(f"   N centiles: {len(centile.centiles)}")
    if len(centile.coverage) > 0:
        print("   Centile  Expected  Observed  Difference")
        for i, c in enumerate(centiles):
            exp = centile.expected_coverage[i]
            obs = centile.coverage[i]
            diff = obs - exp
            print(f"   {c:6.2f}   {exp:7.3f}   {obs:7.3f}   {diff:+7.3f}")
    print("   ✓ Centile check looks good")
    
    # Test comprehensive diagnostics
    print("\n7. Testing comprehensive diagnostics...")
    comp = diagnostics.comprehensive_diagnostics(model)
    assert comp.quantile_residuals.n > 0
    assert len(comp.qq_plot.theoretical_quantiles) > 0
    print("   ✓ Comprehensive diagnostics work")
    
    # Test print summary
    print("\n8. Testing print summary...")
    diagnostics.print_diagnostic_summary(model)
    print("   ✓ Print summary works")
    
    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED FOR NORMAL MODEL")
    print("="*70)


def test_gamma_model():
    """Test diagnostics with a gamma model."""
    print("\n" + "="*70)
    print("TEST 2: Gamma Model Diagnostics")
    print("="*70)
    
    # Generate data
    np.random.seed(123)
    n = 200
    x = np.random.uniform(0, 5, n)
    
    # Generate gamma data
    mu = np.exp(0.5 + 0.3 * x)
    sigma = 0.5
    shape = 1.0 / (sigma ** 2)
    scale = mu / shape
    y = np.random.gamma(shape, scale, n)
    
    data = pd.DataFrame({'x': x, 'y': y})
    
    # Fit model
    print("\nFitting gamma model...")
    model = gamlss(
        formula="y ~ x",
        family="GA",
        data=data
    )
    print("✓ Model fitted successfully")
    
    # Test quantile residuals
    print("\n1. Testing quantile residuals...")
    qres = diagnostics.quantile_residuals(model)
    print(f"   N observations: {qres.n}")
    print(f"   Mean: {qres.mean:.4f}")
    print(f"   Variance: {qres.variance:.4f}")
    print(f"   Skewness: {qres.skewness:.4f}")
    print(f"   Kurtosis: {qres.kurtosis:.4f}")
    
    # Gamma model residuals should still be approximately standard normal
    assert abs(qres.mean) < 0.3, f"Mean too far from 0: {qres.mean}"
    print("   ✓ Quantile residuals look reasonable")
    
    # Test comprehensive diagnostics
    print("\n2. Testing comprehensive diagnostics...")
    comp = diagnostics.comprehensive_diagnostics(model)
    assert comp.quantile_residuals.n > 0
    print("   ✓ Comprehensive diagnostics work")
    
    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED FOR GAMMA MODEL")
    print("="*70)


def test_edge_cases():
    """Test edge cases."""
    print("\n" + "="*70)
    print("TEST 3: Edge Cases")
    print("="*70)
    
    # Small sample
    print("\n1. Testing with small sample (n=10)...")
    np.random.seed(42)
    n = 10
    x = np.random.uniform(0, 10, n)
    y = 2.0 + 0.5 * x + np.random.normal(0, 1, n)
    
    data = pd.DataFrame({'x': x, 'y': y})
    model = gamlss(formula="y ~ x", family="NO", data=data)
    
    qres = diagnostics.quantile_residuals(model)
    assert qres.n == n
    print(f"   ✓ Works with n={n}")
    
    # Different confidence levels
    print("\n2. Testing different confidence levels...")
    worm_90 = diagnostics.worm_plot_data(model, confidence_level=0.90)
    worm_95 = diagnostics.worm_plot_data(model, confidence_level=0.95)
    worm_99 = diagnostics.worm_plot_data(model, confidence_level=0.99)
    
    # Higher confidence should have wider bands
    assert np.all(np.abs(worm_90.upper_band) <= np.abs(worm_95.upper_band))
    assert np.all(np.abs(worm_95.upper_band) <= np.abs(worm_99.upper_band))
    print("   ✓ Confidence bands scale correctly")
    
    # Different bin counts
    print("\n3. Testing different bin counts...")
    for n_bins in [5, 10, 20]:
        calib = diagnostics.calibration_check(model, n_bins=n_bins)
        assert calib.n_bins == n_bins
    print("   ✓ Different bin counts work")
    
    print("\n" + "="*70)
    print("✓ ALL EDGE CASE TESTS PASSED")
    print("="*70)


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("RUNNING DIAGNOSTICS MODULE TESTS")
    print("="*70)
    
    try:
        test_normal_model()
        test_gamma_model()
        test_edge_cases()
        
        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("="*70)
        print("\nDiagnostics module is working correctly!")
        print()
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST FAILED")
        print("="*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
