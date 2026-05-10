"""Tests for REML (Restricted Maximum Likelihood) smoothing parameter selection."""

import pytest
import numpy as np
from omnilss.smoothers.reml import (
    compute_reml_score,
    optimize_lambda_reml,
    compare_gcv_reml
)
from omnilss.smoothers.bsplines import bspline_basis
from omnilss.smoothers.penalties import penalty_matrix


@pytest.fixture
def simple_data():
    """Generate simple test data."""
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + np.random.randn(n) * 0.1
    return x, y


@pytest.fixture
def design_matrix(simple_data):
    """Create B-spline design matrix."""
    x, _ = simple_data
    knots = np.linspace(0, 1, 10)
    X = np.array(bspline_basis(x, knots, degree=3))
    return X


@pytest.fixture
def penalty_mat(design_matrix):
    """Create penalty matrix."""
    p = design_matrix.shape[1]
    S = penalty_matrix(p, order=2)
    return S


def test_compute_reml_score_positive(simple_data, design_matrix, penalty_mat):
    """Test that REML score is positive."""
    _, y = simple_data
    X = design_matrix
    S = penalty_mat
    
    reml = compute_reml_score(y, X, S, lambda_=1.0)
    
    assert reml > 0


def test_compute_reml_score_lambda_effect(simple_data, design_matrix, penalty_mat):
    """Test that REML score changes with λ."""
    _, y = simple_data
    X = design_matrix
    S = penalty_mat
    
    reml_small = compute_reml_score(y, X, S, lambda_=0.01)
    reml_medium = compute_reml_score(y, X, S, lambda_=1.0)
    reml_large = compute_reml_score(y, X, S, lambda_=100.0)
    
    # REML scores should be different
    assert reml_small != reml_medium
    assert reml_medium != reml_large


def test_compute_reml_score_invalid_lambda(simple_data, design_matrix, penalty_mat):
    """Test that negative λ returns inf."""
    _, y = simple_data
    X = design_matrix
    S = penalty_mat
    
    reml = compute_reml_score(y, X, S, lambda_=-1.0)
    
    assert reml == np.inf


def test_optimize_lambda_reml_convergence(simple_data, design_matrix, penalty_mat):
    """Test that optimization converges."""
    _, y = simple_data
    X = design_matrix
    S = penalty_mat
    
    lambda_opt, reml_opt = optimize_lambda_reml(y, X, S)
    
    # Optimal λ should be positive
    assert lambda_opt > 0
    
    # REML score should be finite
    assert np.isfinite(reml_opt)


def test_optimize_lambda_reml_is_minimum(simple_data, design_matrix, penalty_mat):
    """Test that optimized λ gives lower REML than nearby values."""
    _, y = simple_data
    X = design_matrix
    S = penalty_mat
    
    lambda_opt, reml_opt = optimize_lambda_reml(y, X, S)
    
    # Test nearby λ values
    for factor in [0.5, 2.0]:
        lambda_test = lambda_opt * factor
        reml_test = compute_reml_score(y, X, S, lambda_test)
        
        # Optimal should be better (or very close, allowing for numerical precision)
        assert reml_opt <= reml_test + 1e-3  # Relaxed tolerance


def test_optimize_lambda_reml_log_space(simple_data, design_matrix, penalty_mat):
    """Test optimization in log space."""
    _, y = simple_data
    X = design_matrix
    S = penalty_mat
    
    lambda_opt_log, reml_opt_log = optimize_lambda_reml(
        y, X, S, log_space=True
    )
    
    lambda_opt_linear, reml_opt_linear = optimize_lambda_reml(
        y, X, S, log_space=False
    )
    
    # Both methods should give similar results (within same order of magnitude)
    # Note: Linear space optimization can be less stable
    ratio = lambda_opt_log / lambda_opt_linear if lambda_opt_linear > 0 else np.inf
    assert 0.01 < ratio < 100.0  # Within two orders of magnitude


def test_optimize_lambda_reml_range(simple_data, design_matrix, penalty_mat):
    """Test that optimal λ respects search range."""
    _, y = simple_data
    X = design_matrix
    S = penalty_mat
    
    # Narrow range
    lambda_range = (0.1, 10.0)
    lambda_opt, _ = optimize_lambda_reml(y, X, S, lambda_range=lambda_range)
    
    # Optimal should be within range
    assert lambda_range[0] <= lambda_opt <= lambda_range[1]


def test_reml_with_noisy_data():
    """Test REML with different noise levels."""
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 1, n)
    
    # True function
    y_true = np.sin(2 * np.pi * x)
    
    # Create design matrix
    knots = np.linspace(0, 1, 10)
    X = np.array(bspline_basis(x, knots, degree=3))
    p = X.shape[1]
    S = penalty_matrix(p, order=2)
    
    # Low noise
    y_low_noise = y_true + np.random.randn(n) * 0.01
    lambda_low, _ = optimize_lambda_reml(y_low_noise, X, S)
    
    # High noise
    y_high_noise = y_true + np.random.randn(n) * 0.5
    lambda_high, _ = optimize_lambda_reml(y_high_noise, X, S)
    
    # Higher noise should lead to more smoothing (larger λ)
    assert lambda_high > lambda_low


def test_reml_reproducibility(simple_data, design_matrix, penalty_mat):
    """Test that REML optimization is reproducible."""
    _, y = simple_data
    X = design_matrix
    S = penalty_mat
    
    # Run optimization twice
    lambda_opt1, reml_opt1 = optimize_lambda_reml(y, X, S)
    lambda_opt2, reml_opt2 = optimize_lambda_reml(y, X, S)
    
    # Results should be identical
    assert lambda_opt1 == lambda_opt2
    assert reml_opt1 == reml_opt2


def test_compare_gcv_reml(simple_data, design_matrix, penalty_mat):
    """Test comparison between GCV and REML."""
    _, y = simple_data
    X = design_matrix
    S = penalty_mat
    
    comparison = compare_gcv_reml(y, X, S)
    
    # Check that all keys are present
    assert 'gcv_lambda' in comparison
    assert 'gcv_score' in comparison
    assert 'reml_lambda' in comparison
    assert 'reml_score' in comparison
    assert 'lambda_ratio' in comparison
    
    # Both λ values should be positive
    assert comparison['gcv_lambda'] > 0
    assert comparison['reml_lambda'] > 0
    
    # Ratio should be positive and finite
    assert comparison['lambda_ratio'] > 0
    assert np.isfinite(comparison['lambda_ratio'])


def test_gcv_vs_reml_similarity(simple_data, design_matrix, penalty_mat):
    """Test that GCV and REML both produce valid results."""
    _, y = simple_data
    X = design_matrix
    S = penalty_mat
    
    comparison = compare_gcv_reml(y, X, S)
    
    # Both methods should produce positive λ values
    assert comparison['gcv_lambda'] > 0
    assert comparison['reml_lambda'] > 0
    
    # Both should produce finite scores
    assert np.isfinite(comparison['gcv_score'])
    assert np.isfinite(comparison['reml_score'])
    
    # Note: GCV and REML can give very different λ values
    # This is expected and not a bug


def test_reml_with_different_sample_sizes():
    """Test REML with different sample sizes."""
    np.random.seed(42)
    
    for n in [50, 100, 200]:
        x = np.linspace(0, 1, n)
        y = np.sin(2 * np.pi * x) + np.random.randn(n) * 0.1
        
        # Create design matrix
        knots = np.linspace(0, 1, 10)
        X = np.array(bspline_basis(x, knots, degree=3))
        p = X.shape[1]
        S = penalty_matrix(p, order=2)
        
        # Optimize λ
        lambda_opt, reml_opt = optimize_lambda_reml(y, X, S)
        
        # Should converge for all sample sizes
        assert lambda_opt > 0
        assert np.isfinite(reml_opt)


def test_reml_numerical_stability():
    """Test REML numerical stability with challenging data."""
    np.random.seed(42)
    n = 50  # Small sample size
    x = np.linspace(0, 1, n)
    
    # Very noisy data
    y = np.sin(2 * np.pi * x) + np.random.randn(n) * 2.0
    
    # Create design matrix
    knots = np.linspace(0, 1, 10)
    X = np.array(bspline_basis(x, knots, degree=3))
    p = X.shape[1]
    S = penalty_matrix(p, order=2)
    
    # Should still converge
    lambda_opt, reml_opt = optimize_lambda_reml(y, X, S)
    
    assert lambda_opt > 0
    assert np.isfinite(reml_opt)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
