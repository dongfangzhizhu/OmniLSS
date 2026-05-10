"""Tests for gamlss_ml with smooth terms.

Tests that gamlss_ml() correctly handles smooth terms in formulas.
"""

import unittest
import numpy as np
import jax.numpy as jnp
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omnilss.distributions import NO, GA
from omnilss.fitting import gamlss_ml


class TestGAMLSSMLWithSmoothTerms(unittest.TestCase):
    """Test gamlss_ml with smooth terms."""
    
    def test_gamlss_ml_with_smooth_in_mu(self):
        """Test gamlss_ml with smooth term in mu."""
        # Generate data with nonlinear relationship
        np.random.seed(42)
        n = 100
        x = np.linspace(0, 1, n)
        y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.1, n)
        
        data = {"y": y, "x": x}
        
        # Fit model with smooth term
        model = gamlss_ml(
            formula="y ~ pb(x, df=5)",
            family=NO(),
            data=data
        )
        
        # Check that model was fitted
        self.assertIsNotNone(model)
        self.assertIn("mu", model.coefficients)
        self.assertIn("sigma", model.coefficients)
        
        # Check that smooth information is stored
        self.assertIn("smooth_fits", model.additional_slots)
        self.assertIn("smooth_edf", model.additional_slots)
        
        # Check that mu has smooth fits
        smooth_fits_mu = model.additional_slots["smooth_fits"]["mu"]
        self.assertIsInstance(smooth_fits_mu, list)
        self.assertGreater(len(smooth_fits_mu), 0)
        
        # Check that edf is reasonable
        edf_mu = model.additional_slots["smooth_edf"]["mu"]
        self.assertGreater(edf_mu, 0)
        self.assertLess(edf_mu, 10)  # Should be less than number of basis functions
        
        # Check that fitted values are reasonable
        fitted_mu = np.array(model.fitted_values["mu"])
        self.assertEqual(len(fitted_mu), n)
        
        # Check that df_fit includes smooth edf
        # df_fit should be: intercept + smooth_edf + sigma_intercept
        self.assertGreater(model.df_fit, 2)  # More than just two intercepts
    
    def test_gamlss_ml_with_smooth_in_sigma(self):
        """Test gamlss_ml with smooth term in sigma."""
        # Generate data with varying variance
        np.random.seed(123)
        n = 100
        x = np.linspace(0, 1, n)
        sigma_true = 0.1 + 0.3 * x  # Increasing variance
        y = np.random.normal(0, sigma_true)
        
        data = {"y": y, "x": x}
        
        # Fit model with smooth in sigma
        model = gamlss_ml(
            formula="y ~ 1",
            sigma_formula="~ pb(x, df=4)",
            family=NO(),
            data=data
        )
        
        # Check that model was fitted
        self.assertIsNotNone(model)
        
        # Check that sigma has smooth fits
        smooth_fits_sigma = model.additional_slots["smooth_fits"]["sigma"]
        self.assertIsInstance(smooth_fits_sigma, list)
        self.assertGreater(len(smooth_fits_sigma), 0)
        
        # Check that edf is reasonable
        edf_sigma = model.additional_slots["smooth_edf"]["sigma"]
        self.assertGreater(edf_sigma, 0)
        self.assertLess(edf_sigma, 8)
    
    def test_gamlss_ml_mixed_linear_and_smooth(self):
        """Test gamlss_ml with mixed linear and smooth terms."""
        # Generate data
        np.random.seed(456)
        n = 100
        x1 = np.random.normal(0, 1, n)
        x2 = np.linspace(0, 1, n)
        y = 2.0 + 1.5 * x1 + np.sin(2 * np.pi * x2) + np.random.normal(0, 0.2, n)
        
        data = {"y": y, "x1": x1, "x2": x2}
        
        # Fit model with linear and smooth terms
        model = gamlss_ml(
            formula="y ~ x1 + pb(x2, df=5)",
            family=NO(),
            data=data
        )
        
        # Check that model was fitted
        self.assertIsNotNone(model)
        
        # Check coefficients
        mu_coef = np.array(model.coefficients["mu"])
        # Should have: intercept + x1 + basis functions for pb(x2)
        self.assertGreater(len(mu_coef), 2)
        
        # Check smooth information
        smooth_fits_mu = model.additional_slots["smooth_fits"]["mu"]
        self.assertEqual(len(smooth_fits_mu), 1)  # One smooth term
        
        # Check that fitted values are reasonable
        fitted_mu = np.array(model.fitted_values["mu"])
        self.assertEqual(len(fitted_mu), n)
    
    def test_gamlss_ml_multiple_smooths(self):
        """Test gamlss_ml with multiple smooth terms."""
        # Generate data
        np.random.seed(789)
        n = 100
        x1 = np.linspace(0, 1, n)
        x2 = np.linspace(0, 1, n)
        y = np.sin(2 * np.pi * x1) + 0.5 * np.cos(2 * np.pi * x2) + np.random.normal(0, 0.1, n)
        
        data = {"y": y, "x1": x1, "x2": x2}
        
        # Fit model with multiple smooth terms
        model = gamlss_ml(
            formula="y ~ pb(x1, df=5) + pb(x2, df=4)",
            family=NO(),
            data=data
        )
        
        # Check that model was fitted
        self.assertIsNotNone(model)
        
        # Check smooth information
        smooth_fits_mu = model.additional_slots["smooth_fits"]["mu"]
        self.assertEqual(len(smooth_fits_mu), 2)  # Two smooth terms
        
        # Check that edf is reasonable
        edf_mu = model.additional_slots["smooth_edf"]["mu"]
        self.assertGreater(edf_mu, 0)
        self.assertLess(edf_mu, 15)  # Less than total basis functions
    
    def test_gamlss_ml_no_smooth_backward_compat(self):
        """Test that gamlss_ml without smooth terms still works (backward compatibility)."""
        # Generate simple linear data
        np.random.seed(42)
        n = 50
        x = np.random.normal(0, 1, n)
        y = 2.0 + 3.0 * x + np.random.normal(0, 0.5, n)
        
        data = {"y": y, "x": x}
        
        # Fit model without smooth terms
        model = gamlss_ml(
            formula="y ~ x",
            family=NO(),
            data=data
        )
        
        # Check that model was fitted
        self.assertIsNotNone(model)
        
        # Check that smooth information is empty
        smooth_fits_mu = model.additional_slots["smooth_fits"]["mu"]
        self.assertEqual(len(smooth_fits_mu), 0)
        
        # Check that edf is 0 for smooth terms
        edf_mu = model.additional_slots["smooth_edf"]["mu"]
        self.assertEqual(edf_mu, 0.0)
        
        # Check that df_fit is correct (intercept + x + sigma_intercept)
        self.assertEqual(model.df_fit, 3.0)
    
    def test_gamlss_ml_smooth_with_weights(self):
        """Test gamlss_ml with smooth terms and observation weights."""
        # Generate data
        np.random.seed(111)
        n = 80
        x = np.linspace(0, 1, n)
        y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.15, n)
        weights = np.random.uniform(0.5, 1.5, n)
        
        data = {"y": y, "x": x}
        
        # Fit model with smooth term and weights
        model = gamlss_ml(
            formula="y ~ pb(x, df=6)",
            family=NO(),
            data=data,
            weights=weights
        )
        
        # Check that model was fitted
        self.assertIsNotNone(model)
        
        # Check that weights were used
        model_weights = np.array(model.weights)
        np.testing.assert_array_equal(model_weights, weights)
        
        # Check smooth information
        smooth_fits_mu = model.additional_slots["smooth_fits"]["mu"]
        self.assertGreater(len(smooth_fits_mu), 0)
    
    def test_gamlss_ml_smooth_gamma_family(self):
        """Test gamlss_ml with smooth terms for Gamma family."""
        # Generate gamma data
        np.random.seed(222)
        n = 100
        x = np.linspace(0, 1, n)
        mu = np.exp(0.5 + 2 * x)
        sigma = 0.3
        shape = 1 / (sigma ** 2)
        scale = mu * (sigma ** 2)
        y = np.random.gamma(shape, scale)
        
        data = {"y": y, "x": x}
        
        # Fit model with smooth term
        model = gamlss_ml(
            formula="y ~ pb(x, df=5)",
            family=GA(),
            data=data
        )
        
        # Check that model was fitted
        self.assertIsNotNone(model)
        
        # Check smooth information
        smooth_fits_mu = model.additional_slots["smooth_fits"]["mu"]
        self.assertGreater(len(smooth_fits_mu), 0)
        
        # Check that fitted values are positive (required for Gamma)
        fitted_mu = np.array(model.fitted_values["mu"])
        self.assertTrue(np.all(fitted_mu > 0))


if __name__ == "__main__":
    unittest.main()
