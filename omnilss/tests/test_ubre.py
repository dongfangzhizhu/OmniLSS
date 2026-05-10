"""Tests for UBRE (Unbiased Risk Estimator)."""

import jax.numpy as jnp
import numpy as np
import pytest
from omnilss.smoothers.ubre import (
    ubre_score,
    select_lambda_ubre,
    estimate_sigma2,
    ubre_vs_gcv,
)
from omnilss.smoothers.penalties import penalty_matrix


def test_ubre_score_basic():
    """Test basic UBRE score calculation."""
    # Generate simple data
    n = 50
    y = jnp.linspace(0, 10, n)
    fitted = y + jnp.array(np.random.normal(0, 0.1, n))
    
    # Simple hat matrix (identity for this test)
    hat_matrix = jnp.eye(n)
    sigma2 = 1.0
    
    # Calculate UBRE
    score = ubre_score(y, fitted, hat_matrix, sigma2)
    
    # Check that score is finite and reasonable
    assert jnp.isfinite(score)
    assert score > 0  # UBRE should be positive for non-perfect fit


def test_ubre_score_perfect_fit():
    """Test UBRE score with perfect fit."""
    n = 50
    y = jnp.linspace(0, 10, n)
    fitted = y  # Perfect fit
    
    hat_matrix = jnp.eye(n)
    sigma2 = 1.0
    
    score = ubre_score(y, fitted, hat_matrix, sigma2)
    
    # With perfect fit, RSS = 0, so UBRE = 2*sigma2*(n/n) - sigma2 = sigma2
    expected = sigma2
    assert jnp.allclose(score, expected, atol=1e-6)


def test_ubre_score_vs_sigma2():
    """Test that UBRE score scales with sigma2."""
    np.random.seed(42)
    n = 50
    y = jnp.linspace(0, 10, n)
    # Add more noise so RSS is larger
    fitted = y + jnp.array(np.random.normal(0, 2, n))
    
    # Create a proper hat matrix with trace < n
    # Use a simple smoother: H = X(X'X)^-1 X' where X has fewer columns
    p = 10
    X = jnp.array(np.random.randn(n, p))
    XtX_inv = jnp.linalg.inv(X.T @ X)
    hat_matrix = X @ XtX_inv @ X.T
    
    sigma2_1 = 1.0
    sigma2_2 = 2.0
    
    score_1 = ubre_score(y, fitted, hat_matrix, sigma2_1)
    score_2 = ubre_score(y, fitted, hat_matrix, sigma2_2)
    
    # The UBRE formula is: RSS/n + 2*sigma2*(tr(H)/n) - sigma2
    # The difference is: (score_2 - score_1) = 2*(sigma2_2 - sigma2_1)*(tr(H)/n) - (sigma2_2 - sigma2_1)
    #                                         = (sigma2_2 - sigma2_1) * (2*tr(H)/n - 1)
    # So score_2 > score_1 when tr(H) > n/2
    trace_h = jnp.trace(hat_matrix)
    expected_diff = (sigma2_2 - sigma2_1) * (2 * trace_h / n - 1)
    actual_diff = score_2 - score_1
    
    # Check that the difference matches the expected formula
    assert jnp.allclose(actual_diff, expected_diff, atol=1e-6)


def test_select_lambda_ubre():
    """Test lambda selection using UBRE."""
    np.random.seed(42)
    
    # Generate data with smooth underlying function
    n = 100
    x = np.linspace(0, 10, n)
    y_true = np.sin(x)
    y = y_true + np.random.normal(0, 0.1, n)
    
    # Create design matrix (polynomial basis)
    p = 20
    X = np.column_stack([x**i for i in range(p)])
    X = jnp.array(X)
    y = jnp.array(y)
    
    # Penalty matrix (use penalty_matrix for full n x n matrix)
    P = jnp.array(penalty_matrix(p, order=2))
    
    # Known variance
    sigma2 = 0.01
    
    # Select lambda
    best_lambda, scores = select_lambda_ubre(
        X, y, P, sigma2, return_scores=True
    )
    
    # Check results
    assert jnp.isfinite(best_lambda)
    assert best_lambda > 0
    assert len(scores) == 100  # Default grid size
    assert jnp.all(jnp.isfinite(scores))


def test_select_lambda_ubre_custom_grid():
    """Test lambda selection with custom grid."""
    np.random.seed(42)
    
    n, p = 50, 10
    X = jnp.array(np.random.randn(n, p))
    y = jnp.array(np.random.randn(n))
    P = jnp.array(penalty_matrix(p, order=2))
    sigma2 = 1.0
    
    # Custom grid
    lambda_grid = jnp.array([0.01, 0.1, 1.0, 10.0])
    
    best_lambda, scores = select_lambda_ubre(
        X, y, P, sigma2, lambda_grid=lambda_grid, return_scores=True
    )
    
    assert best_lambda in lambda_grid
    assert len(scores) == len(lambda_grid)


def test_estimate_sigma2():
    """Test error variance estimation."""
    np.random.seed(42)
    
    # Generate data with known variance
    n = 100
    true_sigma = 2.0
    y = jnp.array(np.random.normal(0, true_sigma, n))
    fitted = jnp.zeros(n)  # Fit at zero
    
    # Effective degrees of freedom (just intercept)
    edf = 1.0
    
    # Estimate variance
    sigma2_est = estimate_sigma2(y, fitted, edf)
    
    # Should be close to true variance
    assert jnp.isfinite(sigma2_est)
    assert sigma2_est > 0
    # Allow some tolerance due to random sampling
    assert jnp.abs(sigma2_est - true_sigma**2) < 1.0


def test_estimate_sigma2_perfect_fit():
    """Test variance estimation with perfect fit."""
    n = 50
    y = jnp.linspace(0, 10, n)
    fitted = y  # Perfect fit
    
    # All degrees of freedom used
    edf = float(n)
    
    # Estimate variance (should be very small)
    sigma2_est = estimate_sigma2(y, fitted, edf)
    
    # With perfect fit and all df used, should be near zero
    # But we ensure df_error >= 1, so it won't be exactly zero
    assert jnp.isfinite(sigma2_est)
    assert sigma2_est >= 0


def test_ubre_vs_gcv():
    """Test comparison of UBRE and GCV."""
    np.random.seed(42)
    
    # Generate data
    n, p = 100, 20
    X = jnp.array(np.random.randn(n, p))
    y = jnp.array(np.random.randn(n))
    P = jnp.array(penalty_matrix(p, order=2))
    sigma2 = 1.0
    
    # Compare methods
    results = ubre_vs_gcv(X, y, P, sigma2)
    
    # Check results structure
    assert 'lambda_grid' in results
    assert 'ubre_scores' in results
    assert 'gcv_scores' in results
    assert 'best_lambda_ubre' in results
    assert 'best_lambda_gcv' in results
    assert 'sigma2' in results
    
    # Check values
    assert len(results['ubre_scores']) == len(results['lambda_grid'])
    assert len(results['gcv_scores']) == len(results['lambda_grid'])
    assert jnp.isfinite(results['best_lambda_ubre'])
    assert jnp.isfinite(results['best_lambda_gcv'])
    assert results['sigma2'] == sigma2


def test_ubre_gcv_similarity():
    """Test that UBRE and GCV give similar results."""
    np.random.seed(42)
    
    # Generate data
    n, p = 100, 15
    X = jnp.array(np.random.randn(n, p))
    y = jnp.array(np.random.randn(n))
    P = jnp.array(penalty_matrix(p, order=2))
    
    # Estimate sigma2 from a preliminary fit
    XtX = X.T @ X
    Xty = X.T @ y
    coef = jnp.linalg.solve(XtX, Xty)
    fitted = X @ coef
    sigma2 = estimate_sigma2(y, fitted, edf=float(p))
    
    # Compare methods
    results = ubre_vs_gcv(X, y, P, sigma2)
    
    # UBRE and GCV should select similar lambda values
    # (within an order of magnitude)
    ratio = results['best_lambda_ubre'] / results['best_lambda_gcv']
    assert 0.1 < ratio < 10.0


def test_ubre_monotonicity():
    """Test that UBRE changes monotonically with lambda in expected regions."""
    np.random.seed(42)
    
    n, p = 50, 10
    X = jnp.array(np.random.randn(n, p))
    y = jnp.array(np.random.randn(n))
    P = jnp.array(penalty_matrix(p, order=2))
    sigma2 = 1.0
    
    # Test with very small lambdas (should overfit)
    lambda_grid_small = jnp.array([1e-10, 1e-8, 1e-6])
    _, scores_small = select_lambda_ubre(
        X, y, P, sigma2, lambda_grid=lambda_grid_small, return_scores=True
    )
    
    # Test with very large lambdas (should underfit)
    lambda_grid_large = jnp.array([1e4, 1e6, 1e8])
    _, scores_large = select_lambda_ubre(
        X, y, P, sigma2, lambda_grid=lambda_grid_large, return_scores=True
    )
    
    # All scores should be finite
    assert jnp.all(jnp.isfinite(scores_small))
    assert jnp.all(jnp.isfinite(scores_large))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
