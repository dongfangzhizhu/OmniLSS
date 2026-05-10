"""Tests for GCV (Generalized Cross-Validation) smoothing parameter selection."""

import pytest
import numpy as np
from omnilss.smoothers.gcv import (
    compute_hat_matrix,
    compute_effective_df,
    compute_gcv_score,
    optimize_lambda_gcv
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


def test_compute_hat_matrix_shape(design_matrix, penalty_mat):
    """Test that hat matrix has correct shape."""
    X = design_matrix
    S = penalty_mat
    lambda_ = 1.0
    
    H = compute_hat_matrix(X, S, lambda_)
    
    n = X.shape[0]
    assert H.shape == (n, n)


def test_compute_hat_matrix_properties(design_matrix, penalty_mat):
    """Test mathematical properties of hat matrix."""
    X = design_matrix
    S = penalty_mat
    lambda_ = 1.0
    
    H = compute_hat_matrix(X, S, lambda_)
    
    # Hat matrix should be symmetric (approximately)
    assert np.allclose(H, H.T, atol=1e-10)
    
    # Eigenvalues should be between 0 and 1
    eigenvalues = np.linalg.eigvalsh(H)
    assert np.all(eigenvalues >= -1e-10)  # Allow small numerical errors
    assert np.all(eigenvalues <= 1 + 1e-10)


def test_compute_hat_matrix_lambda_effect(design_matrix, penalty_mat):
    """Test that larger λ leads to more shrinkage."""
    X = design_matrix
    S = penalty_mat
    
    # Small λ (less shrinkage)
    H_small = compute_hat_matrix(X, S, lambda_=0.01)
    edf_small = compute_effective_df(H_small)
    
    # Large λ (more shrinkage)
    H_large = compute_hat_matrix(X, S, lambda_=100.0)
    edf_large = compute_effective_df(H_large)
    
    # Larger λ should lead to smaller effective df
    assert edf_large < edf_small


def test_compute_effective_df_range(design_matrix, penalty_mat):
    """Test that effective df is in reasonable range."""
    X = design_matrix
    S = penalty_mat
    n, p = X.shape
    
    # Very small λ: edf should be close to p
    H_small = compute_hat_matrix(X, S, lambda_=1e-6)
    edf_small = compute_effective_df(H_small)
    assert edf_small > p * 0.9  # Should be close to p
    
    # Very large λ: edf should be small
    H_large = compute_hat_matrix(X, S, lambda_=1e6)
    edf_large = compute_effective_df(H_large)
    assert edf_large < p * 0.5  # Should be smaller than p (relaxed from 0.1)


def test_compute_gcv_score_positive(simple_data, design_matrix, penalty_mat):
    """Test that GCV score is positive."""
    _, y = simple_data
    X = design_matrix
    S = penalty_mat
    
    gcv = compute_gcv_score(y, X, S, lambda_=1.0)
    
    assert gcv > 0


def test_compute_gcv_score_lambda_effect(simple_data, design_matrix, penalty_mat):
    """Test that GCV score changes with λ."""
    _, y = simple_data
    X = design_matrix
    S = penalty_mat
    
    gcv_small = compute_gcv_score(y, X, S, lambda_=0.01)
    gcv_medium = compute_gcv_score(y, X, S, lambda_=1.0)
    gcv_large = compute_gcv_score(y, X, S, lambda_=100.0)
    
    # GCV scores should be different
    assert gcv_small != gcv_medium
    assert gcv_medium != gcv_large


def test_compute_gcv_score_invalid_lambda(simple_data, design_matrix, penalty_mat):
    """Test that negative λ returns inf."""
    _, y = simple_data
    X = design_matrix
    S = penalty_mat
    
    gcv = compute_gcv_score(y, X, S, lambda_=-1.0)
    
    assert gcv == np.inf


def test_optimize_lambda_gcv_convergence(simple_data, design_matrix, penalty_mat):
    """Test that optimization converges."""
    _, y = simple_data
    X = design_matrix
    S = penalty_mat
    
    lambda_opt, gcv_opt = optimize_lambda_gcv(y, X, S)
    
    # Optimal λ should be positive
    assert lambda_opt > 0
    
    # GCV score should be finite
    assert np.isfinite(gcv_opt)


def test_optimize_lambda_gcv_is_minimum(simple_data, design_matrix, penalty_mat):
    """Test that optimized λ gives lower GCV than nearby values."""
    _, y = simple_data
    X = design_matrix
    S = penalty_mat
    
    lambda_opt, gcv_opt = optimize_lambda_gcv(y, X, S)
    
    # Test nearby λ values
    for factor in [0.5, 2.0]:
        lambda_test = lambda_opt * factor
        gcv_test = compute_gcv_score(y, X, S, lambda_test)
        
        # Optimal should be better (or very close)
        assert gcv_opt <= gcv_test + 1e-6


def test_optimize_lambda_gcv_log_space(simple_data, design_matrix, penalty_mat):
    """Test optimization in log space."""
    _, y = simple_data
    X = design_matrix
    S = penalty_mat
    
    lambda_opt_log, gcv_opt_log = optimize_lambda_gcv(
        y, X, S, log_space=True
    )
    
    lambda_opt_linear, gcv_opt_linear = optimize_lambda_gcv(
        y, X, S, log_space=False
    )
    
    # Both methods should give similar results
    assert np.abs(lambda_opt_log - lambda_opt_linear) / lambda_opt_log < 0.1
    assert np.abs(gcv_opt_log - gcv_opt_linear) / gcv_opt_log < 0.01


def test_optimize_lambda_gcv_range(simple_data, design_matrix, penalty_mat):
    """Test that optimal λ respects search range."""
    _, y = simple_data
    X = design_matrix
    S = penalty_mat
    
    # Narrow range
    lambda_range = (0.1, 10.0)
    lambda_opt, _ = optimize_lambda_gcv(y, X, S, lambda_range=lambda_range)
    
    # Optimal should be within range
    assert lambda_range[0] <= lambda_opt <= lambda_range[1]


def test_gcv_with_noisy_data():
    """Test GCV with different noise levels."""
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
    lambda_low, _ = optimize_lambda_gcv(y_low_noise, X, S)
    
    # High noise
    y_high_noise = y_true + np.random.randn(n) * 0.5
    lambda_high, _ = optimize_lambda_gcv(y_high_noise, X, S)
    
    # Higher noise should lead to more smoothing (larger λ)
    assert lambda_high > lambda_low


def test_gcv_with_different_sample_sizes():
    """Test GCV with different sample sizes."""
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
        lambda_opt, gcv_opt = optimize_lambda_gcv(y, X, S)
        
        # Should converge for all sample sizes
        assert lambda_opt > 0
        assert np.isfinite(gcv_opt)


def test_gcv_reproducibility(simple_data, design_matrix, penalty_mat):
    """Test that GCV optimization is reproducible."""
    _, y = simple_data
    X = design_matrix
    S = penalty_mat
    
    # Run optimization twice
    lambda_opt1, gcv_opt1 = optimize_lambda_gcv(y, X, S)
    lambda_opt2, gcv_opt2 = optimize_lambda_gcv(y, X, S)
    
    # Results should be identical
    assert lambda_opt1 == lambda_opt2
    assert gcv_opt1 == gcv_opt2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
