"""Tests for regularization methods."""

import numpy as np
import pytest

from omnilss.regularization import (
    RegularizationResult,
    l1_penalty,
    l2_penalty,
    elastic_net_penalty,
    soft_threshold,
    fit_ridge,
    fit_lasso_coordinate_descent,
    fit_elastic_net,
    cross_validate_lambda,
    fit_regularized,
)


class TestPenaltyFunctions:
    """Test penalty functions."""
    
    def test_l1_penalty(self):
        """Test L1 penalty calculation."""
        import jax.numpy as jnp
        
        coef = jnp.array([1.0, -2.0, 3.0])
        lambda_ = 0.5
        
        penalty = l1_penalty(coef, lambda_)
        expected = 0.5 * (1.0 + 2.0 + 3.0)
        
        assert np.isclose(penalty, expected)
    
    def test_l2_penalty(self):
        """Test L2 penalty calculation."""
        import jax.numpy as jnp
        
        coef = jnp.array([1.0, -2.0, 3.0])
        lambda_ = 0.5
        
        penalty = l2_penalty(coef, lambda_)
        expected = 0.5 * (1.0 + 4.0 + 9.0)
        
        assert np.isclose(penalty, expected)
    
    def test_elastic_net_penalty(self):
        """Test Elastic Net penalty calculation."""
        import jax.numpy as jnp
        
        coef = jnp.array([1.0, -2.0, 3.0])
        lambda_ = 1.0
        alpha = 0.5
        
        penalty = elastic_net_penalty(coef, lambda_, alpha)
        l1 = 1.0 * (1.0 + 2.0 + 3.0)
        l2 = 1.0 * (1.0 + 4.0 + 9.0)
        expected = 0.5 * l1 + 0.5 * l2
        
        assert np.isclose(penalty, expected)
    
    def test_elastic_net_pure_lasso(self):
        """Test Elastic Net with alpha=1 equals L1."""
        import jax.numpy as jnp
        
        coef = jnp.array([1.0, -2.0, 3.0])
        lambda_ = 0.5
        
        en_penalty = elastic_net_penalty(coef, lambda_, alpha=1.0)
        l1_pen = l1_penalty(coef, lambda_)
        
        assert np.isclose(en_penalty, l1_pen)
    
    def test_elastic_net_pure_ridge(self):
        """Test Elastic Net with alpha=0 equals L2."""
        import jax.numpy as jnp
        
        coef = jnp.array([1.0, -2.0, 3.0])
        lambda_ = 0.5
        
        en_penalty = elastic_net_penalty(coef, lambda_, alpha=0.0)
        l2_pen = l2_penalty(coef, lambda_)
        
        assert np.isclose(en_penalty, l2_pen)


class TestSoftThreshold:
    """Test soft thresholding operator."""
    
    def test_soft_threshold_basic(self):
        """Test basic soft thresholding."""
        import jax.numpy as jnp
        
        x = jnp.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        threshold = 1.0
        
        result = soft_threshold(x, threshold)
        expected = jnp.array([-1.0, 0.0, 0.0, 0.0, 1.0])
        
        np.testing.assert_allclose(result, expected, atol=1e-10)
    
    def test_soft_threshold_zero(self):
        """Test soft thresholding with zero threshold."""
        import jax.numpy as jnp
        
        x = jnp.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        threshold = 0.0
        
        result = soft_threshold(x, threshold)
        
        np.testing.assert_allclose(result, x, atol=1e-10)
    
    def test_soft_threshold_large(self):
        """Test soft thresholding with large threshold."""
        import jax.numpy as jnp
        
        x = jnp.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        threshold = 10.0
        
        result = soft_threshold(x, threshold)
        expected = jnp.zeros_like(x)
        
        np.testing.assert_allclose(result, expected, atol=1e-10)


class TestRidge:
    """Test Ridge regression."""
    
    def test_ridge_basic(self):
        """Test basic Ridge regression."""
        np.random.seed(42)
        n, p = 100, 10
        X = np.random.randn(n, p)
        beta_true = np.random.randn(p)
        y = X @ beta_true + np.random.randn(n) * 0.1
        
        result = fit_ridge(X, y, lambda_=1.0)
        
        assert isinstance(result, RegularizationResult)
        assert result.coefficients.shape == (p,)
        assert result.fitted_values.shape == (n,)
        assert result.lambda_ == 1.0
        assert result.alpha == 0.0
        assert result.method == "closed_form"
    
    def test_ridge_no_intercept(self):
        """Test Ridge without intercept."""
        np.random.seed(42)
        n, p = 100, 10
        X = np.random.randn(n, p)
        y = np.random.randn(n)
        
        result = fit_ridge(X, y, lambda_=1.0, fit_intercept=False)
        
        assert result.intercept == 0.0
    
    def test_ridge_zero_lambda(self):
        """Test Ridge with lambda=0 (OLS)."""
        np.random.seed(42)
        n, p = 50, 5
        X = np.random.randn(n, p)
        beta_true = np.random.randn(p)
        y = X @ beta_true + np.random.randn(n) * 0.01
        
        result = fit_ridge(X, y, lambda_=0.0)
        
        # Should be close to OLS solution
        beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
        np.testing.assert_allclose(result.coefficients, beta_ols, atol=0.1)
    
    def test_ridge_prediction(self):
        """Test Ridge prediction."""
        np.random.seed(42)
        n, p = 100, 10
        X = np.random.randn(n, p)
        y = np.random.randn(n)
        
        result = fit_ridge(X, y, lambda_=1.0)
        
        # Predict at training points
        y_pred = result.predict(X)
        np.testing.assert_allclose(y_pred, result.fitted_values, atol=1e-10)
        
        # Predict at new points
        X_new = np.random.randn(20, p)
        y_new = result.predict(X_new)
        assert y_new.shape == (20,)


class TestLasso:
    """Test Lasso regression."""
    
    def test_lasso_basic(self):
        """Test basic Lasso regression."""
        np.random.seed(42)
        n, p = 100, 10
        X = np.random.randn(n, p)
        beta_true = np.zeros(p)
        beta_true[:3] = [1.0, -2.0, 3.0]  # Sparse
        y = X @ beta_true + np.random.randn(n) * 0.1
        
        result = fit_lasso_coordinate_descent(X, y, lambda_=0.1)
        
        assert isinstance(result, RegularizationResult)
        assert result.coefficients.shape == (p,)
        assert result.alpha == 1.0
        assert result.method == "coordinate_descent"
        assert result.n_nonzero <= p
    
    def test_lasso_sparsity(self):
        """Test that Lasso produces sparse solutions."""
        np.random.seed(42)
        n, p = 100, 20
        X = np.random.randn(n, p)
        beta_true = np.zeros(p)
        beta_true[:3] = [1.0, -2.0, 3.0]
        y = X @ beta_true + np.random.randn(n) * 0.1
        
        # Large lambda should give sparse solution
        result = fit_lasso_coordinate_descent(X, y, lambda_=1.0)
        
        assert result.n_nonzero < p
        assert result.n_nonzero >= 1
    
    def test_lasso_convergence(self):
        """Test Lasso convergence."""
        np.random.seed(42)
        n, p = 50, 10
        X = np.random.randn(n, p)
        y = np.random.randn(n)
        
        result = fit_lasso_coordinate_descent(X, y, lambda_=0.5, max_iter=1000)
        
        assert result.converged
        assert result.n_iter < 1000
    
    def test_lasso_zero_lambda(self):
        """Test Lasso with lambda=0 (OLS)."""
        np.random.seed(42)
        n, p = 50, 5
        X = np.random.randn(n, p)
        beta_true = np.random.randn(p)
        y = X @ beta_true + np.random.randn(n) * 0.01
        
        result = fit_lasso_coordinate_descent(X, y, lambda_=0.0)
        
        # Should be close to OLS
        beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
        np.testing.assert_allclose(result.coefficients, beta_ols, atol=0.1)
    
    def test_lasso_large_lambda(self):
        """Test Lasso with very large lambda (all zeros)."""
        np.random.seed(42)
        n, p = 50, 10
        X = np.random.randn(n, p)
        y = np.random.randn(n)
        
        result = fit_lasso_coordinate_descent(X, y, lambda_=1000.0)
        
        # Should give all-zero solution
        assert result.n_nonzero == 0
        np.testing.assert_allclose(result.coefficients, 0.0, atol=1e-6)


class TestElasticNet:
    """Test Elastic Net regression."""
    
    def test_elastic_net_basic(self):
        """Test basic Elastic Net regression."""
        np.random.seed(42)
        n, p = 100, 10
        X = np.random.randn(n, p)
        y = np.random.randn(n)
        
        result = fit_elastic_net(X, y, lambda_=0.5, alpha=0.5)
        
        assert isinstance(result, RegularizationResult)
        assert result.alpha == 0.5
        assert result.method == "coordinate_descent"
    
    def test_elastic_net_pure_lasso(self):
        """Test Elastic Net with alpha=1 (pure Lasso)."""
        np.random.seed(42)
        n, p = 100, 10
        X = np.random.randn(n, p)
        y = np.random.randn(n)
        lambda_ = 0.5
        
        result_en = fit_elastic_net(X, y, lambda_=lambda_, alpha=1.0)
        result_lasso = fit_lasso_coordinate_descent(X, y, lambda_=lambda_)
        
        np.testing.assert_allclose(
            result_en.coefficients, result_lasso.coefficients, atol=1e-4
        )
    
    def test_elastic_net_pure_ridge(self):
        """Test Elastic Net with alpha=0 (pure Ridge)."""
        np.random.seed(42)
        n, p = 100, 10
        X = np.random.randn(n, p)
        y = np.random.randn(n)
        lambda_ = 0.5
        
        result_en = fit_elastic_net(X, y, lambda_=lambda_, alpha=0.0)
        result_ridge = fit_ridge(X, y, lambda_=lambda_)
        
        np.testing.assert_allclose(
            result_en.coefficients, result_ridge.coefficients, atol=1e-4
        )
    
    def test_elastic_net_sparsity(self):
        """Test that Elastic Net can produce sparse solutions."""
        np.random.seed(42)
        n, p = 100, 20
        X = np.random.randn(n, p)
        beta_true = np.zeros(p)
        beta_true[:3] = [1.0, -2.0, 3.0]
        y = X @ beta_true + np.random.randn(n) * 0.1
        
        result = fit_elastic_net(X, y, lambda_=1.0, alpha=0.7)
        
        assert result.n_nonzero < p


class TestCrossValidation:
    """Test cross-validation for lambda selection."""
    
    def test_cv_lasso(self):
        """Test CV for Lasso."""
        np.random.seed(42)
        n, p = 100, 10
        X = np.random.randn(n, p)
        beta_true = np.zeros(p)
        beta_true[:3] = [1.0, -2.0, 3.0]
        y = X @ beta_true + np.random.randn(n) * 0.1
        
        best_lambda, cv_scores = cross_validate_lambda(
            X, y, method="lasso", n_folds=5
        )
        
        assert best_lambda > 0
        assert len(cv_scores) > 0
        assert np.all(cv_scores >= 0)
    
    def test_cv_ridge(self):
        """Test CV for Ridge."""
        np.random.seed(42)
        n, p = 100, 10
        X = np.random.randn(n, p)
        y = np.random.randn(n)
        
        best_lambda, cv_scores = cross_validate_lambda(
            X, y, method="ridge", n_folds=5
        )
        
        assert best_lambda > 0
        assert len(cv_scores) > 0
    
    def test_cv_custom_grid(self):
        """Test CV with custom lambda grid."""
        np.random.seed(42)
        n, p = 100, 10
        X = np.random.randn(n, p)
        y = np.random.randn(n)
        
        lambda_grid = np.logspace(-3, 1, 20)
        best_lambda, cv_scores = cross_validate_lambda(
            X, y, lambda_grid=lambda_grid, method="lasso"
        )
        
        assert best_lambda in lambda_grid
        assert len(cv_scores) == len(lambda_grid)


class TestUnifiedInterface:
    """Test unified regularization interface."""
    
    def test_fit_regularized_lasso(self):
        """Test unified interface with Lasso."""
        np.random.seed(42)
        n, p = 100, 10
        X = np.random.randn(n, p)
        y = np.random.randn(n)
        
        result = fit_regularized(X, y, lambda_=0.5, method="lasso", cv=False)
        
        assert result.alpha == 1.0
        assert result.method == "coordinate_descent"
    
    def test_fit_regularized_ridge(self):
        """Test unified interface with Ridge."""
        np.random.seed(42)
        n, p = 100, 10
        X = np.random.randn(n, p)
        y = np.random.randn(n)
        
        result = fit_regularized(X, y, lambda_=0.5, method="ridge", cv=False)
        
        assert result.alpha == 0.0
        assert result.method == "closed_form"
    
    def test_fit_regularized_elastic_net(self):
        """Test unified interface with Elastic Net."""
        np.random.seed(42)
        n, p = 100, 10
        X = np.random.randn(n, p)
        y = np.random.randn(n)
        
        result = fit_regularized(
            X, y, lambda_=0.5, alpha=0.5, method="elastic_net", cv=False
        )
        
        assert result.alpha == 0.5
    
    def test_fit_regularized_auto_method(self):
        """Test unified interface with auto method selection."""
        np.random.seed(42)
        n, p = 100, 10
        X = np.random.randn(n, p)
        y = np.random.randn(n)
        
        # alpha=1.0 should select Lasso
        result = fit_regularized(X, y, lambda_=0.5, alpha=1.0, method="auto", cv=False)
        assert result.method == "coordinate_descent"
        
        # alpha=0.0 should select Ridge
        result = fit_regularized(X, y, lambda_=0.5, alpha=0.0, method="auto", cv=False)
        assert result.method == "closed_form"
    
    def test_fit_regularized_auto_lambda(self):
        """Test unified interface with auto lambda selection."""
        np.random.seed(42)
        n, p = 100, 10
        X = np.random.randn(n, p)
        beta_true = np.zeros(p)
        beta_true[:3] = [1.0, -2.0, 3.0]
        y = X @ beta_true + np.random.randn(n) * 0.1
        
        result = fit_regularized(X, y, lambda_=None, method="lasso", cv=True, n_folds=3)
        
        assert result.lambda_ > 0
        # Should recover sparse solution
        assert result.n_nonzero <= p


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_feature(self):
        """Test with single feature."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 1)
        y = np.random.randn(n)
        
        result = fit_lasso_coordinate_descent(X, y, lambda_=0.1)
        assert result.coefficients.shape == (1,)
    
    def test_perfect_fit(self):
        """Test with perfect fit (no noise)."""
        np.random.seed(42)
        n, p = 50, 5
        X = np.random.randn(n, p)
        beta_true = np.random.randn(p)
        y = X @ beta_true  # No noise
        
        result = fit_ridge(X, y, lambda_=0.01)
        
        # Should fit very well
        np.testing.assert_allclose(result.fitted_values, y, atol=0.1)
    
    def test_collinear_features(self):
        """Test with collinear features."""
        np.random.seed(42)
        n = 100
        X1 = np.random.randn(n, 1)
        X2 = X1 + np.random.randn(n, 1) * 0.01  # Nearly collinear
        X = np.column_stack([X1, X2])
        y = np.random.randn(n)
        
        # Ridge should handle this well
        result = fit_ridge(X, y, lambda_=1.0)
        assert np.all(np.isfinite(result.coefficients))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
