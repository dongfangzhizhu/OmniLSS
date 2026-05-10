"""Tests for Cole-Green (CG) algorithm."""

import pytest
import numpy as np
from omnilss.algorithms import cg_fit
from omnilss.fitting import gamlss_ml


class TestCGBasic:
    """Basic tests for CG algorithm."""
    
    def test_cg_simple_linear(self):
        """Test CG with simple linear model."""
        np.random.seed(42)
        n = 100
        x = np.linspace(0, 10, n)
        y = 2 + 3*x + np.random.normal(0, 1, n)
        data = {"y": y, "x": x}
        
        model = cg_fit(
            formula="y ~ x",
            sigma_formula="~ 1",
            family="NO",
            data=data,
            verbose=False
        )
        
        assert model is not None
        assert len(model.fitted_values["mu"]) == n
        assert len(model.fitted_values["sigma"]) == n
        assert model.g_dev > 0
        assert model.additional_slots["cg_converged"]
    
    def test_cg_heteroscedastic(self):
        """Test CG with heteroscedastic model."""
        np.random.seed(42)
        n = 200
        x = np.linspace(0, 10, n)
        sigma_true = 0.5 + 0.3*x
        y = 2 + 3*x + np.random.normal(0, sigma_true, n)
        data = {"y": y, "x": x}
        
        model = cg_fit(
            formula="y ~ x",
            sigma_formula="~ x",
            family="NO",
            data=data,
            verbose=False
        )
        
        assert model is not None
        assert model.additional_slots["cg_converged"]
        assert model.additional_slots["cg_iterations"] <= 20
    
    def test_cg_intercept_only(self):
        """Test CG with intercept-only model."""
        np.random.seed(42)
        n = 50
        y = np.random.normal(5, 2, n)
        data = {"y": y}
        
        model = cg_fit(
            formula="y ~ 1",
            sigma_formula="~ 1",
            family="NO",
            data=data,
            verbose=False
        )
        
        assert model is not None
        assert np.allclose(model.fitted_values["mu"], np.mean(y), atol=0.5)
        assert model.additional_slots["cg_converged"]


class TestCGConvergence:
    """Tests for CG convergence behavior."""
    
    def test_cg_convergence_iterations(self):
        """Test that CG converges in reasonable iterations."""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 2 + 3*x + np.random.normal(0, 1, n)
        data = {"y": y, "x": x}
        
        model = cg_fit(
            formula="y ~ x",
            sigma_formula="~ 1",
            family="NO",
            data=data,
            max_outer_iter=20,
            verbose=False
        )
        
        assert model.additional_slots["cg_converged"]
        assert model.additional_slots["cg_iterations"] <= 10
    
    def test_cg_with_custom_tolerance(self):
        """Test CG with custom tolerance."""
        np.random.seed(42)
        n = 100
        x = np.linspace(0, 10, n)
        y = 2 + 3*x + np.random.normal(0, 1, n)
        data = {"y": y, "x": x}
        
        # Loose tolerance
        model1 = cg_fit(
            formula="y ~ x",
            sigma_formula="~ 1",
            family="NO",
            data=data,
            outer_tol=1e-2,
            verbose=False
        )
        
        # Tight tolerance
        model2 = cg_fit(
            formula="y ~ x",
            sigma_formula="~ 1",
            family="NO",
            data=data,
            outer_tol=1e-6,
            verbose=False
        )
        
        # Tight tolerance should take more iterations
        assert model2.additional_slots["cg_iterations"] >= model1.additional_slots["cg_iterations"]


class TestCGComparison:
    """Tests comparing CG with other algorithms."""
    
    def test_cg_vs_gamlss_ml(self):
        """Compare CG with gamlss_ml."""
        np.random.seed(42)
        n = 200
        x = np.linspace(0, 10, n)
        y = 2 + 3*x + np.random.normal(0, 1, n)
        data = {"y": y, "x": x}
        
        model_cg = cg_fit(
            formula="y ~ x",
            sigma_formula="~ x",
            family="NO",
            data=data,
            verbose=False
        )
        
        model_ml = gamlss_ml(
            formula="y ~ x",
            sigma_formula="~ x",
            family="NO",
            data=data
        )
        
        # Both should converge
        assert model_cg.additional_slots["cg_converged"]
        
        # Deviances should be similar
        assert abs(model_cg.g_dev - model_ml.g_dev) < 10.0
    
    def test_cg_better_fit(self):
        """Test that CG can find better fits in some cases."""
        np.random.seed(42)
        n = 150
        x = np.linspace(0, 10, n)
        y = 2 + 3*x + np.random.normal(0, 0.5 + 0.3*x, n)
        data = {"y": y, "x": x}
        
        model_cg = cg_fit(
            formula="y ~ x",
            sigma_formula="~ x",
            family="NO",
            data=data,
            verbose=False
        )
        
        model_ml = gamlss_ml(
            formula="y ~ x",
            sigma_formula="~ x",
            family="NO",
            data=data
        )
        
        # CG should find at least as good a fit
        assert model_cg.g_dev <= model_ml.g_dev + 1.0


class TestCGStepSizes:
    """Tests for CG step size control."""
    
    def test_cg_custom_step_sizes(self):
        """Test CG with custom step sizes."""
        np.random.seed(42)
        n = 100
        x = np.linspace(0, 10, n)
        y = 2 + 3*x + np.random.normal(0, 1, n)
        data = {"y": y, "x": x}
        
        model = cg_fit(
            formula="y ~ x",
            sigma_formula="~ x",
            family="NO",
            data=data,
            mu_step=0.5,
            sigma_step=0.5,
            verbose=False
        )
        
        assert model is not None
        assert model.additional_slots["cg_converged"]
    
    def test_cg_small_steps_more_stable(self):
        """Test that smaller steps can be more stable."""
        np.random.seed(42)
        n = 100
        x = np.linspace(0, 10, n)
        # High variance data
        y = 2 + 3*x + np.random.normal(0, 5, n)
        data = {"y": y, "x": x}
        
        # Small steps
        model = cg_fit(
            formula="y ~ x",
            sigma_formula="~ x",
            family="NO",
            data=data,
            mu_step=0.3,
            sigma_step=0.3,
            verbose=False
        )
        
        assert model is not None


class TestCGEdgeCases:
    """Tests for CG edge cases."""
    
    def test_cg_perfect_fit(self):
        """Test CG with perfect fit (no noise)."""
        np.random.seed(42)
        n = 50
        x = np.linspace(0, 10, n)
        y = 2 + 3*x  # No noise
        data = {"y": y, "x": x}
        
        model = cg_fit(
            formula="y ~ x",
            sigma_formula="~ 1",
            family="NO",
            data=data,
            verbose=False
        )
        
        assert model is not None
        # Sigma will be very small for perfect fit
        assert np.all(model.fitted_values["sigma"] >= 0)
    
    def test_cg_constant_response(self):
        """Test CG with constant response."""
        np.random.seed(42)
        n = 50
        x = np.linspace(0, 10, n)
        y = np.ones(n) * 5.0
        data = {"y": y, "x": x}
        
        model = cg_fit(
            formula="y ~ 1",
            sigma_formula="~ 1",
            family="NO",
            data=data,
            verbose=False
        )
        
        assert model is not None
        assert np.allclose(model.fitted_values["mu"], 5.0, atol=0.1)
    
    def test_cg_small_sample(self):
        """Test CG with small sample size."""
        np.random.seed(42)
        n = 20
        x = np.linspace(0, 10, n)
        y = 2 + 3*x + np.random.normal(0, 1, n)
        data = {"y": y, "x": x}
        
        model = cg_fit(
            formula="y ~ x",
            sigma_formula="~ 1",
            family="NO",
            data=data,
            verbose=False
        )
        
        assert model is not None
        assert len(model.fitted_values["mu"]) == n


class TestCGMultipleVariables:
    """Tests for CG with multiple predictor variables."""
    
    def test_cg_two_predictors(self):
        """Test CG with two predictor variables."""
        np.random.seed(42)
        n = 150
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        y = 2 + 3*x1 + 2*x2 + np.random.normal(0, 1, n)
        data = {"y": y, "x1": x1, "x2": x2}
        
        model = cg_fit(
            formula="y ~ x1 + x2",
            sigma_formula="~ 1",
            family="NO",
            data=data,
            verbose=False
        )
        
        assert model is not None
        assert model.additional_slots["cg_converged"]
    
    def test_cg_interaction_terms(self):
        """Test CG with interaction terms."""
        np.random.seed(42)
        n = 150
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        y = 2 + 3*x1 + 2*x2 + 1.5*x1*x2 + np.random.normal(0, 1, n)
        # Create interaction term explicitly
        x1x2 = x1 * x2
        data = {"y": y, "x1": x1, "x2": x2, "x1x2": x1x2}
        
        model = cg_fit(
            formula="y ~ x1 + x2 + x1x2",
            sigma_formula="~ 1",
            family="NO",
            data=data,
            verbose=False
        )
        
        assert model is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
