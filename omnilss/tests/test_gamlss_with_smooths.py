"""Tests for GAMLSS models with smooth terms.

This module tests the integration of smooth terms (P-splines, etc.)
with GAMLSS fitting.
"""

import unittest

import numpy as np

from omnilss.distributions import NO
from omnilss.smooth_fitting import build_smooth_design, fit_penalized_wls


class TestGAMLSSWithSmoothTerms(unittest.TestCase):
    """Test GAMLSS models with smooth terms."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n = 100
        self.x = np.linspace(0, 1, self.n)
        
        # True function: nonlinear mean
        self.mu_true = 2 + 3 * np.sin(2 * np.pi * self.x)
        
        # Generate response
        self.y = np.random.normal(self.mu_true, 0.5)
        
        self.data = {
            'y': self.y,
            'x': self.x,
        }
    
    def test_smooth_term_in_mu(self):
        """Test using smooth term for mu parameter."""
        # Build design matrix with smooth term
        info = build_smooth_design("y ~ pb(x, df=8)", self.data)
        
        # Check structure
        self.assertGreater(info.X.shape[1], 1)
        self.assertEqual(len(info.smooth_fits), 1)
        
        # Fit model
        z = self.y.copy()
        w = np.ones(self.n)
        beta = fit_penalized_wls(info.X, z, w, info.smooth_fits)
        
        # Compute fitted values
        fitted = info.X @ beta
        
        # Check fit quality
        ss_res = np.sum((self.y - fitted)**2)
        ss_tot = np.sum((self.y - np.mean(self.y))**2)
        r_squared = 1 - ss_res / ss_tot
        
        # Should have good fit
        self.assertGreater(r_squared, 0.7)
    
    def test_smooth_term_convergence(self):
        """Test that smooth term fitting converges."""
        info = build_smooth_design("y ~ pb(x, df=5)", self.data)
        
        # Simple iterative fitting
        z = self.y.copy()
        w = np.ones(self.n)
        
        # Multiple iterations
        for _ in range(3):
            beta = fit_penalized_wls(info.X, z, w, info.smooth_fits)
            fitted = info.X @ beta
            z = self.y.copy()  # Reset working response
        
        # Check that fitted values are reasonable
        self.assertTrue(np.all(np.isfinite(fitted)))
        
        # Check that fitted values are in reasonable range
        self.assertGreater(np.min(fitted), np.min(self.y) - 2)
        self.assertLess(np.max(fitted), np.max(self.y) + 2)
    
    def test_multiple_smooth_terms(self):
        """Test using multiple smooth terms."""
        # Create second variable
        x2 = np.random.uniform(0, 1, self.n)
        y2 = self.y + 0.5 * np.sin(4 * np.pi * x2)
        
        data = {
            'y': y2,
            'x1': self.x,
            'x2': x2,
        }
        
        # Build design with two smooth terms
        info = build_smooth_design("y ~ pb(x1, df=5) + pb(x2, df=3)", data)
        
        # Check structure
        self.assertEqual(len(info.smooth_fits), 2)
        self.assertGreater(info.X.shape[1], 1)
        
        # Fit model
        z = y2.copy()
        w = np.ones(self.n)
        beta = fit_penalized_wls(info.X, z, w, info.smooth_fits)
        
        # Check that we get reasonable fit
        fitted = info.X @ beta
        self.assertTrue(np.all(np.isfinite(fitted)))
    
    def test_mixed_linear_and_smooth(self):
        """Test mixing linear and smooth terms."""
        # Create linear predictor
        x_linear = np.random.normal(0, 1, self.n)
        y_mixed = self.y + 1.5 * x_linear
        
        data = {
            'y': y_mixed,
            'x_smooth': self.x,
            'x_linear': x_linear,
        }
        
        # Build design with mixed terms
        info = build_smooth_design("y ~ x_linear + pb(x_smooth, df=5)", data)
        
        # Check structure
        self.assertEqual(len(info.smooth_fits), 1)
        self.assertEqual(info.linear_columns, 2)  # intercept + x_linear
        
        # Fit model
        z = y_mixed.copy()
        w = np.ones(self.n)
        beta = fit_penalized_wls(info.X, z, w, info.smooth_fits)
        
        # Check fit
        fitted = info.X @ beta
        ss_res = np.sum((y_mixed - fitted)**2)
        ss_tot = np.sum((y_mixed - np.mean(y_mixed))**2)
        r_squared = 1 - ss_res / ss_tot
        
        # Should have good fit
        self.assertGreater(r_squared, 0.7)
    
    def test_smooth_with_weights(self):
        """Test smooth terms with observation weights."""
        # Create weights (higher weight for first half)
        weights = np.ones(self.n)
        weights[:50] = 2.0
        
        # Build design
        info = build_smooth_design("y ~ pb(x, df=5)", self.data, weights=weights)
        
        # Fit with weights
        z = self.y.copy()
        beta = fit_penalized_wls(info.X, z, weights, info.smooth_fits)
        
        # Check that we get reasonable fit
        fitted = info.X @ beta
        self.assertTrue(np.all(np.isfinite(fitted)))


if __name__ == "__main__":
    unittest.main()
