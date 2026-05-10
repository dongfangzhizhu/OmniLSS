"""Tests for gamlss_ml with smooth terms integration.

This module tests the integration of smooth terms into the main
gamlss_ml() function.
"""

import unittest

import numpy as np

from omnilss.distributions import NO
from omnilss.fitting import _build_design_matrix_with_smooths, _weighted_least_squares


class TestGAMLSSMLSmoothIntegration(unittest.TestCase):
    """Test gamlss_ml with smooth terms integration."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n = 100
        self.x = np.linspace(0, 1, self.n)
        
        # True nonlinear function
        self.mu_true = 2 + 3 * np.sin(2 * np.pi * self.x)
        
        # Generate response
        self.y = np.random.normal(self.mu_true, 0.5)
        
        self.data = {
            'y': self.y,
            'x': self.x,
        }
    
    def test_build_design_with_smooth(self):
        """Test building design matrix with smooth term."""
        response, X, labels, smooth_info = _build_design_matrix_with_smooths(
            "y ~ pb(x, df=5)", self.data
        )
        
        # Check structure
        self.assertEqual(response, 'y')
        self.assertGreater(X.shape[1], 1)
        self.assertIsNotNone(smooth_info)
        self.assertEqual(len(smooth_info.smooth_fits), 1)
    
    def test_build_design_without_smooth(self):
        """Test building design matrix without smooth term."""
        response, X, labels, smooth_info = _build_design_matrix_with_smooths(
            "y ~ x", self.data
        )
        
        # Check structure
        self.assertEqual(response, 'y')
        self.assertEqual(X.shape, (self.n, 2))  # intercept + x
        self.assertIsNone(smooth_info)
    
    def test_weighted_least_squares_with_smooth(self):
        """Test weighted least squares with smooth term."""
        response, X, labels, smooth_info = _build_design_matrix_with_smooths(
            "y ~ pb(x, df=5)", self.data
        )
        
        w = np.ones(self.n)
        beta = _weighted_least_squares(X, self.y, w, smooth_info=smooth_info)
        
        # Check results
        self.assertEqual(beta.shape, (X.shape[1],))
        self.assertTrue(np.all(np.isfinite(beta)))
        
        # Check fit quality
        fitted = X @ beta
        ss_res = np.sum((self.y - fitted)**2)
        ss_tot = np.sum((self.y - np.mean(self.y))**2)
        r_squared = 1 - ss_res / ss_tot
        
        self.assertGreater(r_squared, 0.7)
    
    def test_weighted_least_squares_without_smooth(self):
        """Test weighted least squares without smooth term."""
        response, X, labels, smooth_info = _build_design_matrix_with_smooths(
            "y ~ x", self.data
        )
        
        w = np.ones(self.n)
        beta = _weighted_least_squares(X, self.y, w, smooth_info=smooth_info)
        
        # Check results
        self.assertEqual(beta.shape, (X.shape[1],))
        self.assertTrue(np.all(np.isfinite(beta)))
    
    def test_comparison_smooth_vs_linear(self):
        """Test that smooth term provides better fit than linear."""
        # Smooth model
        response_s, X_s, labels_s, smooth_info_s = _build_design_matrix_with_smooths(
            "y ~ pb(x, df=8)", self.data
        )
        w = np.ones(self.n)
        beta_s = _weighted_least_squares(X_s, self.y, w, smooth_info=smooth_info_s)
        fitted_s = X_s @ beta_s
        r_squared_s = 1 - np.sum((self.y - fitted_s)**2) / np.sum((self.y - np.mean(self.y))**2)
        
        # Linear model
        response_l, X_l, labels_l, smooth_info_l = _build_design_matrix_with_smooths(
            "y ~ x", self.data
        )
        beta_l = _weighted_least_squares(X_l, self.y, w, smooth_info=smooth_info_l)
        fitted_l = X_l @ beta_l
        r_squared_l = 1 - np.sum((self.y - fitted_l)**2) / np.sum((self.y - np.mean(self.y))**2)
        
        # Smooth should fit better for nonlinear data
        self.assertGreater(r_squared_s, r_squared_l)
        self.assertGreater(r_squared_s, 0.9)  # Should be very good fit


if __name__ == "__main__":
    unittest.main()
