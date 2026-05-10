"""Tests for automatic smoothing parameter selection."""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from omnilss import smoothing_selection


def test_gcv_basic():
    """Test basic GCV functionality."""
    print("\n" + "="*70)
    print("TEST 1: Basic GCV Selection")
    print("="*70)
    
    # Generate simple test data
    np.random.seed(42)
    n, p = 100, 10
    X = np.random.randn(n, p)
    beta_true = np.random.randn(p)
    y = X @ beta_true + np.random.randn(n) * 0.5
    S = np.eye(p)  # Identity penalty
    
    print(f"\nData: n={n}, p={p}")
    print(f"True signal strength: {np.linalg.norm(X @ beta_true):.2f}")
    print(f"Noise level: 0.5")
    
    # Select lambda using GCV
    result = smoothing_selection.select_lambda_gcv(X, y, S)
    
    print(f"\nGCV Results:")
    print(f"  Optimal lambda: {result.lambda_opt:.6f}")
    print(f"  EDF: {result.edf:.2f}")
    print(f"  GCV value: {result.criterion_value:.4f}")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.n_iterations}")
    
    # Sanity checks
    assert result.lambda_opt > 0, "Lambda should be positive"
    assert 0 < result.edf <= p, f"EDF should be in (0, {p}]"
    assert result.converged, "Optimization should converge"
    
    print("\n✓ GCV basic test passed")


def test_reml_basic():
    """Test basic REML functionality."""
    print("\n" + "="*70)
    print("TEST 2: Basic REML Selection")
    print("="*70)
    
    np.random.seed(123)
    n, p = 100, 10
    X = np.random.randn(n, p)
    beta_true = np.random.randn(p)
    y = X @ beta_true + np.random.randn(n) * 0.5
    S = np.eye(p)
    
    print(f"\nData: n={n}, p={p}")
    
    # Select lambda using REML
    result = smoothing_selection.select_lambda_reml(X, y, S)
    
    print(f"\nREML Results:")
    print(f"  Optimal lambda: {result.lambda_opt:.6f}")
    print(f"  EDF: {result.edf:.2f}")
    print(f"  REML value: {result.criterion_value:.4f}")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.n_iterations}")
    
    assert result.lambda_opt > 0
    assert 0 < result.edf <= p
    assert result.converged
    
    print("\n✓ REML basic test passed")


def test_aic_basic():
    """Test basic AIC functionality."""
    print("\n" + "="*70)
    print("TEST 3: Basic AIC Selection")
    print("="*70)
    
    np.random.seed(456)
    n, p = 100, 10
    X = np.random.randn(n, p)
    beta_true = np.random.randn(p)
    y = X @ beta_true + np.random.randn(n) * 0.5
    S = np.eye(p)
    
    print(f"\nData: n={n}, p={p}")
    
    # Select lambda using AIC
    result = smoothing_selection.select_lambda_aic(X, y, S)
    
    print(f"\nAIC Results:")
    print(f"  Optimal lambda: {result.lambda_opt:.6f}")
    print(f"  EDF: {result.edf:.2f}")
    print(f"  AIC value: {result.criterion_value:.4f}")
    print(f"  Converged: {result.converged}")
    
    assert result.lambda_opt > 0
    assert 0 < result.edf <= p
    
    print("\n✓ AIC basic test passed")


def test_compare_methods():
    """Compare different selection methods."""
    print("\n" + "="*70)
    print("TEST 4: Compare Selection Methods")
    print("="*70)
    
    np.random.seed(789)
    n, p = 150, 15
    X = np.random.randn(n, p)
    beta_true = np.random.randn(p) * 0.5
    y = X @ beta_true + np.random.randn(n) * 1.0
    S = np.eye(p)
    
    print(f"\nData: n={n}, p={p}")
    print(f"Comparing GCV, REML, and AIC...")
    
    # Try all methods
    result_gcv = smoothing_selection.select_lambda_gcv(X, y, S)
    result_reml = smoothing_selection.select_lambda_reml(X, y, S)
    result_aic = smoothing_selection.select_lambda_aic(X, y, S)
    
    print(f"\nResults Comparison:")
    print(f"  Method    Lambda      EDF     Criterion")
    print(f"  ------  ----------  ------  -----------")
    print(f"  GCV     {result_gcv.lambda_opt:10.6f}  {result_gcv.edf:6.2f}  {result_gcv.criterion_value:11.4f}")
    print(f"  REML    {result_reml.lambda_opt:10.6f}  {result_reml.edf:6.2f}  {result_reml.criterion_value:11.4f}")
    print(f"  AIC     {result_aic.lambda_opt:10.6f}  {result_aic.edf:6.2f}  {result_aic.criterion_value:11.4f}")
    
    # All should give reasonable results
    for result in [result_gcv, result_reml, result_aic]:
        assert result.lambda_opt > 0
        assert 0 < result.edf <= p
    
    print("\n✓ Method comparison test passed")


def test_with_weights():
    """Test with observation weights."""
    print("\n" + "="*70)
    print("TEST 5: Selection with Weights")
    print("="*70)
    
    np.random.seed(101)
    n, p = 100, 10
    X = np.random.randn(n, p)
    beta_true = np.random.randn(p)
    y = X @ beta_true + np.random.randn(n) * 0.5
    S = np.eye(p)
    
    # Random weights
    weights = np.random.uniform(0.5, 1.5, n)
    
    print(f"\nData: n={n}, p={p}")
    print(f"Weights: min={weights.min():.2f}, max={weights.max():.2f}")
    
    # Select with weights
    result_weighted = smoothing_selection.select_lambda_gcv(X, y, S, weights=weights)
    result_unweighted = smoothing_selection.select_lambda_gcv(X, y, S)
    
    print(f"\nResults:")
    print(f"  Weighted:   lambda={result_weighted.lambda_opt:.6f}, EDF={result_weighted.edf:.2f}")
    print(f"  Unweighted: lambda={result_unweighted.lambda_opt:.6f}, EDF={result_unweighted.edf:.2f}")
    
    # Results should be different
    assert result_weighted.lambda_opt != result_unweighted.lambda_opt
    
    print("\n✓ Weighted selection test passed")


def test_main_interface():
    """Test the main selection interface."""
    print("\n" + "="*70)
    print("TEST 6: Main Interface")
    print("="*70)
    
    np.random.seed(202)
    n, p = 100, 10
    X = np.random.randn(n, p)
    y = np.random.randn(n)
    S = np.eye(p)
    
    print(f"\nTesting main interface with different methods...")
    
    # Test each method through main interface
    for method in ["GCV", "REML", "AIC"]:
        result = smoothing_selection.select_smoothing_parameter(
            X, y, S, method=method
        )
        print(f"  {method:6s}: lambda={result.lambda_opt:.6f}, EDF={result.edf:.2f}")
        assert result.method == method
        assert result.lambda_opt > 0
    
    print("\n✓ Main interface test passed")


def test_edf_computation():
    """Test EDF computation methods."""
    print("\n" + "="*70)
    print("TEST 7: EDF Computation")
    print("="*70)
    
    np.random.seed(303)
    n, p = 50, 8
    X = np.random.randn(n, p)
    S = np.eye(p)
    lambda_ = 1.0
    
    print(f"\nData: n={n}, p={p}, lambda={lambda_}")
    
    # Compute EDF using different methods
    H = smoothing_selection.compute_hat_matrix(X, S, lambda_)
    edf_from_H = smoothing_selection.compute_edf(H)
    edf_fast = smoothing_selection.compute_edf_fast(X, S, lambda_)
    
    print(f"\nEDF Results:")
    print(f"  From hat matrix: {edf_from_H:.4f}")
    print(f"  Fast method:     {edf_fast:.4f}")
    print(f"  Difference:      {abs(edf_from_H - edf_fast):.6f}")
    
    # Should be very close
    assert abs(edf_from_H - edf_fast) < 1e-6
    
    # EDF should be in reasonable range
    assert 0 < edf_from_H <= p
    assert 0 < edf_fast <= p
    
    print("\n✓ EDF computation test passed")


def test_edge_cases():
    """Test edge cases."""
    print("\n" + "="*70)
    print("TEST 8: Edge Cases")
    print("="*70)
    
    # Small sample
    print("\n1. Small sample (n=10, p=5)")
    np.random.seed(404)
    n, p = 10, 5
    X = np.random.randn(n, p)
    y = np.random.randn(n)
    S = np.eye(p)
    
    result = smoothing_selection.select_lambda_gcv(X, y, S)
    print(f"   Lambda: {result.lambda_opt:.6f}, EDF: {result.edf:.2f}")
    assert result.lambda_opt > 0
    
    # Large lambda (heavy smoothing)
    print("\n2. Testing with large lambda")
    lambda_large = 1e6
    edf_large = smoothing_selection.compute_edf_fast(X, S, lambda_large)
    print(f"   Lambda: {lambda_large:.0e}, EDF: {edf_large:.2f}")
    assert edf_large < p  # Should be heavily penalized
    
    # Small lambda (little smoothing)
    print("\n3. Testing with small lambda")
    lambda_small = 1e-6
    edf_small = smoothing_selection.compute_edf_fast(X, S, lambda_small)
    print(f"   Lambda: {lambda_small:.0e}, EDF: {edf_small:.2f}")
    assert edf_small > edf_large  # Should be less penalized
    
    print("\n✓ Edge cases test passed")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("RUNNING SMOOTHING SELECTION TESTS")
    print("="*70)
    
    try:
        test_gcv_basic()
        test_reml_basic()
        test_aic_basic()
        test_compare_methods()
        test_with_weights()
        test_main_interface()
        test_edf_computation()
        test_edge_cases()
        
        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("="*70)
        print("\nSmoothing selection module is working correctly!")
        print()
        
        return 0
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST FAILED")
        print("="*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
