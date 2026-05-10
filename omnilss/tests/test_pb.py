"""Tests for P-splines (pb) smoother."""

import unittest

import jax.numpy as jnp
import numpy as np

from omnilss.smoothers.pb import fit_pspline


class TestPSplines(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n = 100
        self.x = np.linspace(0, 1, self.n)
        # True function: sin wave
        self.y_true = np.sin(2 * np.pi * self.x)
        # Add noise
        self.y = self.y_true + np.random.normal(0, 0.1, self.n)
    
    def test_fit_with_fixed_lambda(self):
        """Test fitting with fixed lambda."""
        result = fit_pspline(self.x, self.y, lambda_=1.0)
        
        # Check result attributes
        self.assertEqual(result.fitted_values.shape, (self.n,))
        self.assertGreater(result.edf, 0)
        self.assertLess(result.edf, result.design_matrix.shape[1])
        self.assertEqual(result.lambda_, 1.0)
    
    def test_fit_with_fixed_df(self):
        """Test fitting with fixed degrees of freedom."""
        target_df = 5.0
        result = fit_pspline(self.x, self.y, df=target_df)
        
        # Check that we achieved target df
        np.testing.assert_allclose(result.edf, target_df, rtol=1e-2)
    
    def test_fit_with_ml(self):
        """Test fitting with ML method."""
        result = fit_pspline(self.x, self.y, method="ML")
        
        # Check result
        self.assertEqual(result.fitted_values.shape, (self.n,))
        self.assertGreater(result.lambda_, 0)
        self.assertGreater(result.edf, 0)
    
    def test_fit_with_gaic(self):
        """Test fitting with GAIC method."""
        result = fit_pspline(self.x, self.y, method="GAIC", k=2.0)
        
        # Check result
        self.assertEqual(result.fitted_values.shape, (self.n,))
        self.assertGreater(result.lambda_, 0)
        self.assertGreater(result.edf, 0)
    
    def test_fit_with_gcv(self):
        """Test fitting with GCV method."""
        result = fit_pspline(self.x, self.y, method="GCV")
        
        # Check result
        self.assertEqual(result.fitted_values.shape, (self.n,))
        self.assertGreater(result.lambda_, 0)
        self.assertGreater(result.edf, 0)
    
    def test_fit_with_weights(self):
        """Test fitting with observation weights."""
        weights = np.ones(self.n)
        weights[:50] = 2.0  # Higher weight for first half
        
        result = fit_pspline(self.x, self.y, weights=weights, df=5.0)
        
        # Check result
        self.assertEqual(result.fitted_values.shape, (self.n,))
    
    def test_different_degrees(self):
        """Test fitting with different spline degrees."""
        for degree in [1, 2, 3]:
            result = fit_pspline(self.x, self.y, degree=degree, df=5.0)
            
            # Check result
            self.assertEqual(result.fitted_values.shape, (self.n,))
            np.testing.assert_allclose(result.edf, 5.0, rtol=1e-2)
    
    def test_different_penalty_orders(self):
        """Test fitting with different penalty orders."""
        for order in [0, 1, 2, 3]:
            result = fit_pspline(self.x, self.y, order=order, df=5.0)
            
            # Check result
            self.assertEqual(result.fitted_values.shape, (self.n,))
    
    def test_prediction(self):
        """Test prediction at new points."""
        result = fit_pspline(self.x, self.y, df=5.0)
        
        # Predict at new points
        x_new = np.linspace(0, 1, 50)
        y_pred = result.predict(x_new)
        
        # Check shape
        self.assertEqual(y_pred.shape, (50,))
    
    def test_smoothing_effect(self):
        """Test that smoothing reduces variance."""
        # Fit with different smoothing levels
        result_smooth = fit_pspline(self.x, self.y, df=3.0)
        result_rough = fit_pspline(self.x, self.y, df=10.0)
        
        # Smoother fit should have lower edf
        self.assertLess(result_smooth.edf, result_rough.edf)
        
        # Smoother fit should have higher lambda
        self.assertGreater(result_smooth.lambda_, result_rough.lambda_)
    
    def test_fit_quality(self):
        """Test that fit captures the true function reasonably well."""
        result = fit_pspline(self.x, self.y, df=8.0)
        
        # Compute correlation with true function
        correlation = np.corrcoef(result.fitted_values, self.y_true)[0, 1]
        
        # Should have high correlation
        self.assertGreater(correlation, 0.9)
    
    def test_invalid_df(self):
        """Test that invalid df raises error."""
        # df too large
        with self.assertRaises(ValueError):
            fit_pspline(self.x, self.y, df=100.0)
        
        # df negative
        with self.assertRaises(ValueError):
            fit_pspline(self.x, self.y, df=-1.0)
    
    def test_invalid_method(self):
        """Test that invalid method raises error."""
        with self.assertRaises(ValueError):
            fit_pspline(self.x, self.y, method="INVALID")
    
    def test_mismatched_lengths(self):
        """Test that mismatched x and y lengths raise error."""
        with self.assertRaises(ValueError):
            fit_pspline(self.x, self.y[:50])
    
    def test_convergence_ml(self):
        """Test that ML method converges."""
        result = fit_pspline(self.x, self.y, method="ML", max_iter=50)
        
        # Should converge to reasonable lambda
        self.assertGreater(result.lambda_, 1e-7)
        self.assertLess(result.lambda_, 1e7)


if __name__ == "__main__":
    unittest.main()
