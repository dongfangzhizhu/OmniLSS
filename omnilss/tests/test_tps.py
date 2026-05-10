"""Tests for Thin Plate Spline (TPS) smoother."""

import numpy as np
import pytest

from omnilss.smoothers.tps import (
    TPSResult,
    fit_tps,
    tps,
    _tps_radial_basis,
    _euclidean_distance,
    _polynomial_basis,
    _tps_basis_matrix,
    _select_knots,
)


class TestRadialBasis:
    """Test radial basis functions."""
    
    def test_1d_radial_basis(self):
        """Test 1D radial basis (cubic)."""
        r = np.array([0.0, 1.0, 2.0, 3.0])
        phi = _tps_radial_basis(r, n_dims=1, m=2)
        
        # 1D: φ(r) = r³
        expected = r ** 3
        np.testing.assert_allclose(phi, expected, atol=1e-10)
    
    def test_2d_radial_basis(self):
        """Test 2D radial basis (r² log r)."""
        r = np.array([0.1, 1.0, 2.0, 3.0])
        phi = _tps_radial_basis(r, n_dims=2, m=2)
        
        # 2D: φ(r) = r² log(r)
        expected = r ** 2 * np.log(r)
        np.testing.assert_allclose(phi, expected, rtol=1e-6)
    
    def test_3d_radial_basis(self):
        """Test 3D radial basis (r)."""
        r = np.array([0.0, 1.0, 2.0, 3.0])
        phi = _tps_radial_basis(r, n_dims=3, m=2)
        
        # 3D: φ(r) = r
        expected = r
        np.testing.assert_allclose(phi, expected, atol=1e-10)
    
    def test_radial_basis_positive(self):
        """Test that radial basis handles r=0 gracefully."""
        r = np.array([0.0, 1e-12, 1.0])
        phi = _tps_radial_basis(r, n_dims=2, m=2)
        
        # Should not have NaN or Inf
        assert np.all(np.isfinite(phi))


class TestEuclideanDistance:
    """Test Euclidean distance computation."""
    
    def test_distance_1d(self):
        """Test 1D distance."""
        X1 = np.array([[0.0], [1.0], [2.0]])
        X2 = np.array([[0.5], [1.5]])
        
        dist = _euclidean_distance(X1, X2)
        
        expected = np.array([
            [0.5, 1.5],
            [0.5, 0.5],
            [1.5, 0.5],
        ])
        
        np.testing.assert_allclose(dist, expected, atol=1e-10)
    
    def test_distance_2d(self):
        """Test 2D distance."""
        X1 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        X2 = np.array([[0.0, 0.0], [1.0, 1.0]])
        
        dist = _euclidean_distance(X1, X2)
        
        expected = np.array([
            [0.0, np.sqrt(2)],
            [1.0, 1.0],
            [1.0, 1.0],
        ])
        
        np.testing.assert_allclose(dist, expected, atol=1e-10)
    
    def test_distance_symmetric(self):
        """Test that distance is symmetric."""
        X = np.random.randn(10, 3)
        
        dist1 = _euclidean_distance(X, X)
        dist2 = dist1.T
        
        np.testing.assert_allclose(dist1, dist2, atol=1e-10)
    
    def test_distance_diagonal_zero(self):
        """Test that diagonal is zero."""
        X = np.random.randn(10, 3)
        
        dist = _euclidean_distance(X, X)
        
        # Use looser tolerance for numerical precision
        np.testing.assert_allclose(np.diag(dist), 0.0, atol=1e-7)


class TestPolynomialBasis:
    """Test polynomial basis construction."""
    
    def test_polynomial_1d(self):
        """Test 1D polynomial basis."""
        X = np.array([[1.0], [2.0], [3.0]])
        poly = _polynomial_basis(X, m=2)
        
        # Should be [1, x]
        expected = np.array([
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
        ])
        
        np.testing.assert_allclose(poly, expected, atol=1e-10)
    
    def test_polynomial_2d(self):
        """Test 2D polynomial basis."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        poly = _polynomial_basis(X, m=2)
        
        # Should be [1, x, y]
        expected = np.array([
            [1.0, 1.0, 2.0],
            [1.0, 3.0, 4.0],
        ])
        
        np.testing.assert_allclose(poly, expected, atol=1e-10)
    
    def test_polynomial_3d(self):
        """Test 3D polynomial basis."""
        X = np.array([[1.0, 2.0, 3.0]])
        poly = _polynomial_basis(X, m=2)
        
        # Should be [1, x, y, z]
        expected = np.array([[1.0, 1.0, 2.0, 3.0]])
        
        np.testing.assert_allclose(poly, expected, atol=1e-10)


class TestKnotSelection:
    """Test knot selection methods."""
    
    def test_select_knots_all(self):
        """Test selecting all points as knots."""
        X = np.random.randn(20, 2)
        knots = _select_knots(X, k=None, method="all")
        
        assert knots.shape == X.shape
        np.testing.assert_allclose(knots, X)
    
    def test_select_knots_uniform(self):
        """Test uniform knot selection."""
        X = np.random.randn(100, 2)
        k = 20
        knots = _select_knots(X, k=k, method="uniform")
        
        assert knots.shape == (k, 2)
        # Check that knots are from X
        for knot in knots:
            # At least one point in X should be close to this knot
            distances = np.linalg.norm(X - knot, axis=1)
            assert np.min(distances) < 1e-10
    
    def test_select_knots_kmeans(self):
        """Test k-means knot selection."""
        X = np.random.randn(100, 2)
        k = 20
        knots = _select_knots(X, k=k, method="kmeans")
        
        assert knots.shape == (k, 2)
        # Knots should be within the range of X
        assert np.all(knots >= X.min(axis=0) - 1.0)
        assert np.all(knots <= X.max(axis=0) + 1.0)


class TestTPSFitting:
    """Test TPS fitting."""
    
    def test_fit_tps_1d(self):
        """Test TPS fitting in 1D (should work like cubic spline)."""
        np.random.seed(42)
        n = 50
        X = np.linspace(0, 1, n).reshape(-1, 1)
        y = np.sin(2 * np.pi * X.ravel()) + np.random.normal(0, 0.1, n)
        
        result = fit_tps(X, y, k=10, method="GCV")
        
        assert isinstance(result, TPSResult)
        assert result.fitted_values.shape == (n,)
        assert result.lambda_ > 0
        assert 0 < result.edf < n
        assert result.n_dims == 1
    
    def test_fit_tps_2d(self):
        """Test TPS fitting in 2D."""
        np.random.seed(42)
        n = 100
        x1 = np.random.uniform(0, 1, n)
        x2 = np.random.uniform(0, 1, n)
        X = np.column_stack([x1, x2])
        
        # True function: sin(2πx1) * cos(2πx2)
        y_true = np.sin(2 * np.pi * x1) * np.cos(2 * np.pi * x2)
        y = y_true + np.random.normal(0, 0.1, n)
        
        result = fit_tps(X, y, k=20, method="GCV")
        
        assert isinstance(result, TPSResult)
        assert result.fitted_values.shape == (n,)
        assert result.lambda_ > 0
        assert 0 < result.edf < n
        assert result.n_dims == 2
        
        # Check fit quality (R² > 0.5)
        ss_res = np.sum((y - result.fitted_values) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot
        assert r_squared > 0.5
    
    def test_fit_tps_3d(self):
        """Test TPS fitting in 3D."""
        np.random.seed(42)
        n = 100
        X = np.random.uniform(0, 1, (n, 3))
        
        # Simple 3D function
        y = X[:, 0] + X[:, 1] ** 2 + X[:, 2] ** 3 + np.random.normal(0, 0.1, n)
        
        result = fit_tps(X, y, k=20, method="GCV")
        
        assert isinstance(result, TPSResult)
        assert result.fitted_values.shape == (n,)
        assert result.n_dims == 3
    
    def test_fit_tps_with_weights(self):
        """Test TPS fitting with weights."""
        np.random.seed(42)
        n = 50
        X = np.random.uniform(0, 1, (n, 2))
        y = X[:, 0] + X[:, 1] + np.random.normal(0, 0.1, n)
        
        # Give more weight to some observations
        w = np.ones(n)
        w[:10] = 10.0
        
        result = fit_tps(X, y, w=w, k=15, method="GCV")
        
        assert isinstance(result, TPSResult)
        assert result.fitted_values.shape == (n,)
    
    def test_fit_tps_fixed_lambda(self):
        """Test TPS fitting with fixed lambda."""
        np.random.seed(42)
        n = 50
        X = np.random.uniform(0, 1, (n, 2))
        y = X[:, 0] + X[:, 1] + np.random.normal(0, 0.1, n)
        
        lambda_fixed = 0.01
        result = fit_tps(X, y, lambda_=lambda_fixed, k=15, method="fixed")
        
        assert result.lambda_ == lambda_fixed
        assert result.selection_method == "fixed"
    
    def test_fit_tps_perfect_fit(self):
        """Test TPS with no noise (should interpolate)."""
        np.random.seed(42)
        n = 20
        X = np.random.uniform(0, 1, (n, 2))
        y = X[:, 0] + X[:, 1]  # Linear, no noise
        
        # With very small lambda, should fit perfectly
        result = fit_tps(X, y, lambda_=1e-10, k=n, method="fixed", knot_method="all")
        
        # Should fit very well
        np.testing.assert_allclose(result.fitted_values, y, atol=1e-3)


class TestTPSPrediction:
    """Test TPS prediction."""
    
    def test_predict_1d(self):
        """Test prediction in 1D."""
        np.random.seed(42)
        n = 50
        X_train = np.linspace(0, 1, n).reshape(-1, 1)
        y_train = np.sin(2 * np.pi * X_train.ravel())
        
        result = fit_tps(X_train, y_train, k=10, lambda_=0.01, method="fixed")
        
        # Predict at new points
        X_test = np.linspace(0, 1, 20).reshape(-1, 1)
        y_pred = result.predict(X_test)
        
        assert y_pred.shape == (20,)
        assert np.all(np.isfinite(y_pred))
    
    def test_predict_2d(self):
        """Test prediction in 2D."""
        np.random.seed(42)
        n = 100
        X_train = np.random.uniform(0, 1, (n, 2))
        y_train = X_train[:, 0] + X_train[:, 1]
        
        result = fit_tps(X_train, y_train, k=20, lambda_=0.01, method="fixed")
        
        # Predict at new points
        X_test = np.random.uniform(0, 1, (30, 2))
        y_pred = result.predict(X_test)
        
        assert y_pred.shape == (30,)
        assert np.all(np.isfinite(y_pred))
        
        # Should be close to true function
        y_true = X_test[:, 0] + X_test[:, 1]
        np.testing.assert_allclose(y_pred, y_true, atol=0.2)
    
    def test_predict_at_training_points(self):
        """Test that prediction at training points matches fitted values."""
        np.random.seed(42)
        n = 50
        X = np.random.uniform(0, 1, (n, 2))
        y = X[:, 0] + X[:, 1] + np.random.normal(0, 0.1, n)
        
        result = fit_tps(X, y, k=20, method="GCV")
        
        # Predict at training points
        y_pred = result.predict(X)
        
        # Should match fitted values
        np.testing.assert_allclose(y_pred, result.fitted_values, atol=1e-10)


class TestTPSFunction:
    """Test tps() function for formula interface."""
    
    def test_tps_specification(self):
        """Test tps() creates correct specification."""
        spec = tps("x1", "x2", k=20, method="GCV")
        
        assert spec["type"] == "tps"
        assert spec["variables"] == ("x1", "x2")
        assert spec["k"] == 20
        assert spec["method"] == "GCV"
    
    def test_tps_requires_multiple_variables(self):
        """Test that tps() requires at least 2 variables."""
        with pytest.raises(ValueError, match="at least 2 variables"):
            tps("x1")
    
    def test_tps_three_variables(self):
        """Test tps() with 3 variables."""
        spec = tps("x1", "x2", "x3", k=30)
        
        assert len(spec["variables"]) == 3


class TestTPSEdgeCases:
    """Test edge cases and error handling."""
    
    def test_mismatched_dimensions(self):
        """Test error when X and y have different lengths."""
        X = np.random.randn(50, 2)
        y = np.random.randn(40)
        
        with pytest.raises(ValueError, match="same length"):
            fit_tps(X, y)
    
    def test_mismatched_weights(self):
        """Test error when weights have wrong length."""
        X = np.random.randn(50, 2)
        y = np.random.randn(50)
        w = np.ones(40)
        
        with pytest.raises(ValueError, match="length"):
            fit_tps(X, y, w=w)
    
    def test_predict_wrong_dimensions(self):
        """Test error when predicting with wrong dimensions."""
        X = np.random.randn(50, 2)
        y = np.random.randn(50)
        
        result = fit_tps(X, y, k=10, lambda_=0.01, method="fixed")
        
        X_wrong = np.random.randn(20, 3)  # Wrong number of dimensions
        
        with pytest.raises(ValueError, match="must have 2 columns"):
            result.predict(X_wrong)
    
    def test_fixed_method_without_lambda(self):
        """Test error when using fixed method without lambda."""
        X = np.random.randn(50, 2)
        y = np.random.randn(50)
        
        with pytest.raises(ValueError, match="Must provide lambda"):
            fit_tps(X, y, method="fixed")
    
    def test_unknown_method(self):
        """Test error with unknown method."""
        X = np.random.randn(50, 2)
        y = np.random.randn(50)
        
        with pytest.raises(ValueError, match="Unknown method"):
            fit_tps(X, y, method="unknown")
    
    def test_reml_not_implemented(self):
        """Test that REML raises NotImplementedError."""
        X = np.random.randn(50, 2)
        y = np.random.randn(50)
        
        with pytest.raises(NotImplementedError, match="REML"):
            fit_tps(X, y, method="REML")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
