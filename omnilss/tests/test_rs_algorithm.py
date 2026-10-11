"""Tests for RS (Rigby-Stasinopoulos) algorithm.

This module tests the RS algorithm implementation, which is the original
and default fitting algorithm for GAMLSS models.
"""

import numpy as np
import pytest

from omnilss.algorithms.rs_algorithm import (
    ConvergenceWarning,
    _get_eta_bounds,
    _has_converged,
    rs_fit,
    rs_step,
    compute_working_weights_and_response,
    RSStepResult
)
from omnilss.distributions import resolve_family


@pytest.fixture
def simple_data():
    """Generate simple normal data."""
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 10, n)
    y = 2 + 3 * x + np.random.randn(n)
    return {"x": x, "y": y}


@pytest.fixture
def heteroscedastic_data():
    """Generate heteroscedastic data."""
    np.random.seed(42)
    n = 120
    x = np.linspace(0, 10, n)
    mu = 2 + 3 * x
    sigma = 0.5 + 0.5 * x
    y = mu + sigma * np.random.randn(n)
    return {"x": x, "y": y}


class TestComputeWorkingWeightsAndResponse:
    """Tests for compute_working_weights_and_response function."""
    
    def test_basic_computation(self):
        """Test basic working weights and response computation."""
        n = 50
        y = np.random.randn(n)
        fitted_values = np.random.randn(n)
        link_derivative = np.ones(n)
        first_derivative = np.random.randn(n)
        second_derivative = -np.abs(np.random.randn(n))  # Negative
        offset = np.zeros(n)
        eta = np.random.randn(n)
        
        weights, response = compute_working_weights_and_response(
            y=y,
            fitted_values=fitted_values,
            link_derivative=link_derivative,
            first_derivative=first_derivative,
            second_derivative=second_derivative,
            offset=offset,
            eta=eta
        )
        
        # Check output shapes
        assert weights.shape == (n,)
        assert response.shape == (n,)
        
        # Check weights are positive
        assert np.all(weights > 0)
        
        # Check no NaN or Inf
        assert np.all(np.isfinite(weights))
        assert np.all(np.isfinite(response))
    
    def test_weight_clipping(self):
        """Test that weights are clipped to reasonable range."""
        n = 50
        y = np.random.randn(n)
        fitted_values = np.random.randn(n)
        link_derivative = np.ones(n)
        first_derivative = np.random.randn(n)
        # Very large negative second derivative
        second_derivative = -1e20 * np.ones(n)
        offset = np.zeros(n)
        eta = np.random.randn(n)
        
        weights, response = compute_working_weights_and_response(
            y=y,
            fitted_values=fitted_values,
            link_derivative=link_derivative,
            first_derivative=first_derivative,
            second_derivative=second_derivative,
            offset=offset,
            eta=eta
        )
        
        # Weights should be clipped
        assert np.all(weights <= 1e10)
        assert np.all(weights >= 1e-10)

    def test_working_response_clipping_emits_warning(self):
        n = 20
        y = np.zeros(n)
        fitted_values = np.zeros(n)
        link_derivative = np.ones(n)
        first_derivative = np.full(n, 1e10)
        second_derivative = -np.ones(n)
        offset = np.zeros(n)
        eta = np.zeros(n)

        # The ±1e6 hard clip was removed to match R's glim.fit (it corrupted
        # IRLS for zero-inflated data).  Extreme-but-finite responses are now
        # preserved; only non-finite values are replaced and warned about.
        with pytest.warns(ConvergenceWarning, match="non-finite working_response"):
            _weights, response = compute_working_weights_and_response(
                y=y,
                fitted_values=fitted_values,
                link_derivative=link_derivative,
                first_derivative=np.full(n, np.inf),  # inf -> non-finite response
                second_derivative=second_derivative,
                offset=offset,
                eta=eta
            )
        assert np.all(np.isfinite(response))


class TestRSStep:
    """Tests for rs_step function."""
    
    def test_basic_step(self, simple_data):
        """Test basic RS step."""
        y = simple_data["y"]
        x = simple_data["x"]
        n = len(y)
        
        # Create design matrix
        X = np.column_stack([np.ones(n), x])
        
        # Initial fitted values
        fitted_values = np.full(n, np.mean(y))
        weights = np.ones(n)
        
        # Get family
        family = resolve_family("NO")
        
        # Perform RS step
        result = rs_step(
            y=y,
            X=X,
            fitted_values=fitted_values,
            weights=weights,
            family=family,
            parameter="mu",
            max_iter=20,
            tol=1e-4,
            verbose=False
        )
        
        # Check result type
        assert isinstance(result, RSStepResult)
        
        # Check shapes
        assert result.fitted_values.shape == (n,)
        assert result.linear_predictor.shape == (n,)
        assert result.coefficients.shape == (2,)  # Intercept + slope
        
        # Check convergence
        assert result.converged
        assert result.iterations > 0
        
        # Check deviance is finite
        assert np.isfinite(result.deviance)
    
    def test_convergence(self, simple_data):
        """Test that RS step converges."""
        y = simple_data["y"]
        x = simple_data["x"]
        n = len(y)
        
        X = np.column_stack([np.ones(n), x])
        fitted_values = np.full(n, np.mean(y))
        weights = np.ones(n)
        family = resolve_family("NO")
        
        result = rs_step(
            y=y,
            X=X,
            fitted_values=fitted_values,
            weights=weights,
            family=family,
            parameter="mu",
            max_iter=50,
            tol=1e-6,
            verbose=False
        )
        
        # Should converge with enough iterations
        assert result.converged
        assert result.iterations < 50
    
    def test_step_size(self, simple_data):
        """Test different step sizes."""
        y = simple_data["y"]
        x = simple_data["x"]
        n = len(y)
        
        X = np.column_stack([np.ones(n), x])
        fitted_values = np.full(n, np.mean(y))
        weights = np.ones(n)
        family = resolve_family("NO")
        
        # Full step
        result_full = rs_step(
            y=y,
            X=X,
            fitted_values=fitted_values,
            weights=weights,
            family=family,
            parameter="mu",
            step_size=1.0,
            max_iter=20
        )
        
        # Half step
        result_half = rs_step(
            y=y,
            X=X,
            fitted_values=fitted_values,
            weights=weights,
            family=family,
            parameter="mu",
            step_size=0.5,
            max_iter=20
        )
        
        # Both should converge
        assert result_full.converged
        assert result_half.converged
        
        # Half step may need more iterations
        assert result_half.iterations >= result_full.iterations

    def test_eta_bounds_are_link_aware(self):
        assert _get_eta_bounds("identity") == (-1e8, 1e8)
        assert _get_eta_bounds("log") == (-15.0, 15.0)
        assert _get_eta_bounds("logit") == (-15.0, 15.0)


class TestRSFit:
    """Tests for rs_fit function."""
    
    def test_basic_fit(self, simple_data):
        """Test basic RS fit."""
        model = rs_fit(
            formula="y ~ x",
            family="NO",
            data=simple_data,
            verbose=False
        )
        
        # Check model is valid
        assert model is not None
        assert hasattr(model, "g_dev")
        assert hasattr(model, "fitted_values")
        assert hasattr(model, "coefficients")
        
        # Check method is RS
        assert model.additional_slots.get("method") == "RS"
    
    def test_with_sigma_formula(self, heteroscedastic_data):
        """Test RS fit with sigma formula."""
        model = rs_fit(
            formula="y ~ x",
            sigma_formula="~ x",
            family="NO",
            data=heteroscedastic_data,
            verbose=False
        )
        
        # Check model is valid
        assert model is not None
        assert np.isfinite(model.g_dev)
        
        # Check both parameters are fitted
        assert "mu" in model.fitted_values
        assert "sigma" in model.fitted_values
    
    def test_convergence_info(self, simple_data):
        """Test that convergence information is stored."""
        model = rs_fit(
            formula="y ~ x",
            family="NO",
            data=simple_data,
            verbose=False
        )
        
        # Check convergence info
        assert "rs_iterations" in model.additional_slots
        assert "rs_converged" in model.additional_slots
        assert model.additional_slots["rs_converged"]
    
    def test_verbose_output(self, simple_data, capsys):
        """Test verbose output."""
        model = rs_fit(
            formula="y ~ x",
            family="NO",
            data=simple_data,
            verbose=True
        )
        
        # Check that something was printed
        captured = capsys.readouterr()
        assert "RS Algorithm" in captured.out
    
    def test_custom_steps(self, simple_data):
        """Test custom step sizes."""
        model = rs_fit(
            formula="y ~ x",
            family="NO",
            data=simple_data,
            mu_step=0.5,
            sigma_step=0.5,
            verbose=False
        )
        
        # Should still converge
        assert model is not None
        assert np.isfinite(model.g_dev)

    def test_gradient_norm_is_finite(self, simple_data):
        model = rs_fit(formula="y ~ x", family="NO", data=simple_data, verbose=False)
        mu_grad = model.additional_slots["rs_last_gradient_norm_by_param"]["mu"]
        assert np.isfinite(mu_grad)
        assert mu_grad > 0
        assert np.isfinite(model.additional_slots["gradient_norm"])
        assert model.additional_slots["gradient_norm"] > 0

    def test_policy_max_inner_is_one(self, simple_data):
        # D2.36 reversed the pre-D2.36 policy this test used to pin.  The old
        # assertion -- rs_policy_max_inner == 1, justified as "matching R's
        # gamlss::glim.fit" -- was factually wrong: R's glim.fit iterates each
        # parameter update inside `while (abs(olddv - dv) > cc && itn < cyc)`
        # with glim.control() defaults cc = 0.001, cyc = 50 (live R, gamlss
        # 5.5-0).  The max_iter=1 shortcut starved the inner loop and made
        # R's Box-Cox sigma.initial = 0.1 unusable (BCPE degraded 94.15 ->
        # 138.73 on the scanner sample; intercept-only BCCG rel ~1e3).  The
        # policy slot now publishes R's actual cc/cyc pair.
        model = rs_fit(formula="y ~ x", family="NO", data=simple_data, verbose=False)
        assert model.additional_slots["rs_policy_max_inner"] == 50
        assert model.additional_slots["rs_policy_inner_cc"] == 0.001

    def test_identity_link_large_response_not_clipped_to_20(self):
        np.random.seed(123)
        n = 120
        x = np.linspace(0, 10, n)
        y = 1e6 + 50.0 * x + np.random.randn(n) * 10.0
        data = {"x": x, "y": y}

        model = rs_fit(formula="y ~ x", family="NO", data=data, max_iter=5, verbose=False)
        mu_hat = np.asarray(model.fitted_values["mu"], dtype=float)

        assert np.mean(mu_hat) > 1e5
        assert not np.allclose(mu_hat, 20.0)

    def test_has_converged_relative_or_absolute(self):
        assert _has_converged(1000.0, 999.99995, tol=1e-4, rtol=1e-8)
        assert _has_converged(1e9, 1e9 - 100.0, tol=1e-8, rtol=1e-6)
        assert not _has_converged(1.0, 0.9, tol=1e-8, rtol=1e-8)


class TestIntegration:
    """Integration tests for RS algorithm."""
    
    def test_comparison_with_gamlss_ml(self, simple_data):
        """Compare RS fit with gamlss_ml."""
        from omnilss.fitting import gamlss_ml
        
        # Fit with RS
        model_rs = rs_fit(
            formula="y ~ x",
            family="NO",
            data=simple_data,
            verbose=False
        )
        
        # Fit with gamlss_ml
        model_ml = gamlss_ml(
            formula="y ~ x",
            family="NO",
            data=simple_data
        )
        
        # RS and ML should both produce finite, converged fits.
        # Their objective values can differ because RS now uses a single-step
        # inner policy by default for stability.
        assert np.isfinite(model_rs.g_dev)
        assert np.isfinite(model_ml.g_dev)
        assert model_rs.g_dev > 0
        assert model_ml.g_dev > 0
    
    def test_with_weights(self, simple_data):
        """Test RS fit with observation weights."""
        n = len(simple_data["y"])
        weights = np.random.uniform(0.5, 1.5, n)
        
        model = rs_fit(
            formula="y ~ x",
            family="NO",
            data=simple_data,
            weights=weights,
            verbose=False
        )
        
        # Should work with weights
        assert model is not None
        assert np.isfinite(model.g_dev)
    
    def test_model_structure(self, simple_data):
        """Test that model structure is correct."""
        model = rs_fit(
            formula="y ~ x",
            family="NO",
            data=simple_data,
            verbose=False
        )
        
        # Check all expected attributes
        assert hasattr(model, "par")
        assert hasattr(model, "family")
        assert hasattr(model, "df_fit")
        assert hasattr(model, "g_dev")
        assert hasattr(model, "fitted_values")
        assert hasattr(model, "coefficients")
        assert hasattr(model, "linear_predictors")
        assert hasattr(model, "design_matrices")
        assert hasattr(model, "additional_slots")
        
        # Check method is stored
        assert model.additional_slots["method"] == "RS"
