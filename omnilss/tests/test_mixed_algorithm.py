"""Tests for Mixed algorithm."""

import pytest
import numpy as np
from omnilss.algorithms import mixed_fit, compare_algorithms


class TestMixedBasic:
    """Basic tests for Mixed algorithm."""
    
    def test_mixed_auto_selection(self):
        """Test Mixed with automatic algorithm selection."""
        np.random.seed(42)
        n = 100
        x = np.linspace(0, 10, n)
        y = 2 + 3*x + np.random.normal(0, 1, n)
        data = {"y": y, "x": x}
        
        model = mixed_fit(
            formula="y ~ x",
            sigma_formula="~ 1",
            family="NO",
            data=data,
            algorithm="auto",
            verbose=False
        )
        
        assert model is not None
        assert len(model.fitted_values["mu"]) == n
        assert model.additional_slots["mixed_algorithm_used"] == "rs"
        assert model.additional_slots["mixed_auto_selected"] is True
    
    def test_mixed_explicit_rs(self):
        """Test Mixed with explicit RS selection."""
        np.random.seed(42)
        n = 100
        x = np.linspace(0, 10, n)
        y = 2 + 3*x + np.random.normal(0, 1, n)
        data = {"y": y, "x": x}
        
        model = mixed_fit(
            formula="y ~ x",
            sigma_formula="~ 1",
            family="NO",
            data=data,
            algorithm="rs",
            verbose=False
        )
        
        assert model is not None
        assert model.additional_slots["mixed_algorithm_used"] == "rs"
        assert model.additional_slots["mixed_auto_selected"] is False
        assert "rs_converged" in model.additional_slots
    
    def test_mixed_explicit_cg(self):
        """Test Mixed with explicit CG selection."""
        np.random.seed(42)
        n = 100
        x = np.linspace(0, 10, n)
        y = 2 + 3*x + np.random.normal(0, 1, n)
        data = {"y": y, "x": x}
        
        model = mixed_fit(
            formula="y ~ x",
            sigma_formula="~ 1",
            family="NO",
            data=data,
            algorithm="cg",
            verbose=False
        )
        
        assert model is not None
        assert model.additional_slots["mixed_algorithm_used"] == "cg"
        assert model.additional_slots["mixed_auto_selected"] is False
        assert "cg_converged" in model.additional_slots


class TestMixedComparison:
    """Tests for algorithm comparison functionality."""
    
    def test_compare_algorithms_basic(self):
        """Test basic algorithm comparison."""
        np.random.seed(42)
        n = 100
        x = np.linspace(0, 10, n)
        y = 2 + 3*x + np.random.normal(0, 1, n)
        data = {"y": y, "x": x}
        
        models = compare_algorithms(
            formula="y ~ x",
            sigma_formula="~ 1",
            family="NO",
            data=data,
            verbose=False
        )
        
        assert "rs" in models
        assert "cg" in models
        assert models["rs"] is not None
        assert models["cg"] is not None
    
    def test_compare_algorithms_heteroscedastic(self):
        """Test algorithm comparison with heteroscedastic model."""
        np.random.seed(42)
        n = 200
        x = np.linspace(0, 10, n)
        sigma_true = 0.5 + 0.3*x
        y = 2 + 3*x + np.random.normal(0, sigma_true, n)
        data = {"y": y, "x": x}
        
        models = compare_algorithms(
            formula="y ~ x",
            sigma_formula="~ x",
            family="NO",
            data=data,
            verbose=False
        )
        
        # Both should converge
        assert models["rs"].additional_slots["rs_converged"]
        assert models["cg"].additional_slots["cg_converged"]
        
        # Deviances should be similar
        assert abs(models["rs"].g_dev - models["cg"].g_dev) < 100.0


class TestMixedEdgeCases:
    """Tests for Mixed algorithm edge cases."""
    
    def test_mixed_invalid_algorithm(self):
        """Test Mixed with invalid algorithm selection."""
        np.random.seed(42)
        n = 50
        x = np.linspace(0, 10, n)
        y = 2 + 3*x + np.random.normal(0, 1, n)
        data = {"y": y, "x": x}
        
        with pytest.raises(ValueError, match="Unknown algorithm"):
            mixed_fit(
                formula="y ~ x",
                sigma_formula="~ 1",
                family="NO",
                data=data,
                algorithm="invalid"
            )
    
    def test_mixed_no_data(self):
        """Test Mixed without data."""
        with pytest.raises(ValueError, match="data must be provided"):
            mixed_fit(
                formula="y ~ x",
                sigma_formula="~ 1",
                family="NO",
                data=None
            )
    
    def test_mixed_small_sample(self):
        """Test Mixed with small sample."""
        np.random.seed(42)
        n = 20
        x = np.linspace(0, 10, n)
        y = 2 + 3*x + np.random.normal(0, 1, n)
        data = {"y": y, "x": x}
        
        model = mixed_fit(
            formula="y ~ x",
            sigma_formula="~ 1",
            family="NO",
            data=data,
            algorithm="auto",
            verbose=False
        )
        
        assert model is not None
        assert len(model.fitted_values["mu"]) == n


class TestMixedMultipleVariables:
    """Tests for Mixed with multiple predictor variables."""
    
    def test_mixed_two_predictors(self):
        """Test Mixed with two predictor variables."""
        np.random.seed(42)
        n = 150
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        y = 2 + 3*x1 + 2*x2 + np.random.normal(0, 1, n)
        data = {"y": y, "x1": x1, "x2": x2}
        
        model = mixed_fit(
            formula="y ~ x1 + x2",
            sigma_formula="~ 1",
            family="NO",
            data=data,
            algorithm="auto",
            verbose=False
        )
        
        assert model is not None
        assert model.additional_slots["mixed_algorithm_used"] == "rs"
    
    def test_compare_two_predictors(self):
        """Test comparison with two predictor variables."""
        np.random.seed(42)
        n = 150
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        y = 2 + 3*x1 + 2*x2 + np.random.normal(0, 1, n)
        data = {"y": y, "x1": x1, "x2": x2}
        
        models = compare_algorithms(
            formula="y ~ x1 + x2",
            sigma_formula="~ 1",
            family="NO",
            data=data,
            verbose=False
        )
        
        assert models["rs"] is not None
        assert models["cg"] is not None


class TestMixedVerbose:
    """Tests for Mixed algorithm verbose output."""
    
    def test_mixed_verbose_auto(self):
        """Test Mixed verbose output with auto selection."""
        np.random.seed(42)
        n = 50
        x = np.linspace(0, 10, n)
        y = 2 + 3*x + np.random.normal(0, 1, n)
        data = {"y": y, "x": x}
        
        # Should not raise any errors
        model = mixed_fit(
            formula="y ~ x",
            sigma_formula="~ 1",
            family="NO",
            data=data,
            algorithm="auto",
            verbose=True
        )
        
        assert model is not None
    
    def test_compare_verbose(self):
        """Test comparison verbose output."""
        np.random.seed(42)
        n = 50
        x = np.linspace(0, 10, n)
        y = 2 + 3*x + np.random.normal(0, 1, n)
        data = {"y": y, "x": x}
        
        # Should not raise any errors
        models = compare_algorithms(
            formula="y ~ x",
            sigma_formula="~ 1",
            family="NO",
            data=data,
            verbose=True
        )
        
        assert models is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
