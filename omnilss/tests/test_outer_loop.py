"""Tests for outer loop automatic smoothing parameter selection.

This module tests the integration of GCV/REML smoothing parameter selection
into the GAMLSS fitting process using an outer loop strategy.

NOTE: These tests are for the PROTOTYPE implementation. The outer loop
structure is in place but not fully active yet.
"""

import numpy as np
import pytest

from omnilss.smoothers.smooth_parameter_selection import (
    optimize_smoothing_parameters,
    gamlss_with_automatic_smoothing,
    auto_smooth,
)


@pytest.fixture
def simple_smooth_data():
    """Generate simple data with smooth relationship."""
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 10, n)
    y = np.sin(x) + 0.2 * np.random.randn(n)
    return {"x": x, "y": y}


@pytest.fixture
def heteroscedastic_data():
    """Generate data with heteroscedastic variance."""
    np.random.seed(42)
    n = 120
    x = np.linspace(0, 10, n)
    mu = np.sin(x)
    sigma = 0.1 + 0.3 * (x / 10)  # Increasing variance
    y = mu + sigma * np.random.randn(n)
    return {"x": x, "y": y}


class TestGAMLSSWithAutomaticSmoothing:
    """Tests for gamlss_with_automatic_smoothing function."""
    
    def test_no_smooth_terms(self, simple_smooth_data):
        """Test that function handles models without smooth terms."""
        from omnilss.fitting import gamlss_ml
        
        # Fit model without smooth terms
        model = gamlss_with_automatic_smoothing(
            fit_function=gamlss_ml,
            formula="y ~ x",
            family="NO",
            data=simple_smooth_data,
            lambda_method="REML",
            verbose=False
        )
        
        # Should return valid model
        assert model is not None
        assert hasattr(model, "g_dev")
        assert hasattr(model, "fitted_values")
        assert np.isfinite(model.g_dev)
    
    def test_with_smooth_terms(self, simple_smooth_data):
        """Test outer loop with smooth terms."""
        from omnilss.fitting import gamlss_ml
        
        # Fit model with smooth term
        model = gamlss_with_automatic_smoothing(
            fit_function=gamlss_ml,
            formula="y ~ pb(x, df=10)",
            family="NO",
            data=simple_smooth_data,
            lambda_method="REML",
            max_outer_iter=3,
            verbose=False
        )
        
        # Should return valid model
        assert model is not None
        assert hasattr(model, "g_dev")
        assert np.isfinite(model.g_dev)
        
        # Should have smooth fits
        smooth_fits = model.additional_slots.get("smooth_fits", {})
        assert "mu" in smooth_fits
        assert len(smooth_fits["mu"]) > 0
    
    def test_multiple_parameters(self, heteroscedastic_data):
        """Test outer loop with smooths in multiple parameters."""
        from omnilss.fitting import gamlss_ml
        
        # Fit model with smooths in mu and sigma
        model = gamlss_with_automatic_smoothing(
            fit_function=gamlss_ml,
            formula="y ~ pb(x, df=10)",
            sigma_formula="~ pb(x, df=5)",
            family="NO",
            data=heteroscedastic_data,
            lambda_method="REML",
            max_outer_iter=3,
            verbose=False
        )
        
        # Should return valid model
        assert model is not None
        assert np.isfinite(model.g_dev)
        
        # Should have smooth fits for both parameters
        smooth_fits = model.additional_slots.get("smooth_fits", {})
        assert "mu" in smooth_fits
        assert "sigma" in smooth_fits
    
    def test_gcv_method(self, simple_smooth_data):
        """Test with GCV method."""
        from omnilss.fitting import gamlss_ml
        
        # Fit with GCV
        model = gamlss_with_automatic_smoothing(
            fit_function=gamlss_ml,
            formula="y ~ pb(x, df=10)",
            family="NO",
            data=simple_smooth_data,
            lambda_method="GCV",
            verbose=False
        )
        
        # Should return valid model
        assert model is not None
        assert np.isfinite(model.g_dev)


class TestAutoSmooth:
    """Tests for auto_smooth convenience function."""
    
    def test_basic_usage(self, simple_smooth_data):
        """Test basic usage of auto_smooth."""
        model = auto_smooth(
            formula="y ~ pb(x, df=10)",
            family="NO",
            data=simple_smooth_data,
            method="REML",
            verbose=False
        )
        
        # Should return valid model
        assert model is not None
        assert hasattr(model, "g_dev")
        assert np.isfinite(model.g_dev)
    
    def test_gcv_method(self, simple_smooth_data):
        """Test auto_smooth with GCV method."""
        model = auto_smooth(
            formula="y ~ pb(x, df=10)",
            family="NO",
            data=simple_smooth_data,
            method="GCV",
            verbose=False
        )
        
        # Should return valid model
        assert model is not None
        assert np.isfinite(model.g_dev)
    
    def test_with_sigma_formula(self, heteroscedastic_data):
        """Test auto_smooth with sigma formula."""
        model = auto_smooth(
            formula="y ~ pb(x, df=10)",
            sigma_formula="~ pb(x, df=5)",
            family="NO",
            data=heteroscedastic_data,
            method="REML",
            verbose=False
        )
        
        # Should return valid model
        assert model is not None
        assert np.isfinite(model.g_dev)
        
        # Should have smooth fits for both parameters
        smooth_fits = model.additional_slots.get("smooth_fits", {})
        assert "mu" in smooth_fits
        assert "sigma" in smooth_fits


class TestIntegration:
    """Integration tests for outer loop."""
    
    def test_comparison_with_fixed_lambda(self, simple_smooth_data):
        """Compare automatic λ selection with fixed λ."""
        from omnilss.fitting import gamlss_ml
        
        # Fit with automatic λ (prototype)
        model_auto = auto_smooth(
            formula="y ~ pb(x, df=10)",
            family="NO",
            data=simple_smooth_data,
            method="REML",
            verbose=False
        )
        
        # Fit with fixed λ (default)
        model_fixed = gamlss_ml(
            formula="y ~ pb(x, df=10)",
            family="NO",
            data=simple_smooth_data
        )
        
        # Both should be valid
        assert np.isfinite(model_auto.g_dev)
        assert np.isfinite(model_fixed.g_dev)
        
        # In prototype, they should be identical (same λ used)
        assert abs(model_auto.g_dev - model_fixed.g_dev) < 1e-6
    
    def test_model_structure(self, simple_smooth_data):
        """Test that model structure is preserved."""
        model = auto_smooth(
            formula="y ~ pb(x, df=10)",
            family="NO",
            data=simple_smooth_data,
            method="REML",
            verbose=False
        )
        
        # Check model has expected attributes
        assert hasattr(model, "g_dev")
        assert hasattr(model, "fitted_values")
        assert hasattr(model, "coefficients")
        assert hasattr(model, "design_matrices")
        assert hasattr(model, "additional_slots")
        
        # Check smooth information is stored
        smooth_fits = model.additional_slots.get("smooth_fits", {})
        assert isinstance(smooth_fits, dict)
        assert "mu" in smooth_fits
        
        # Check smooth fit info
        if len(smooth_fits["mu"]) > 0:
            smooth_fit = smooth_fits["mu"][0]
            assert hasattr(smooth_fit, "variable")
            assert hasattr(smooth_fit, "lambda_")
            assert hasattr(smooth_fit, "edf")
            assert hasattr(smooth_fit, "penalty")

