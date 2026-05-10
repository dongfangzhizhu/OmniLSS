"""Tests for model diagnostics module."""

import pytest
import numpy as np
import pandas as pd
import jax.numpy as jnp
import jax.random as jrandom

import sys
sys.path.insert(0, '../src')

from omnilss.fitting import gamlss
from omnilss import diagnostics


@pytest.fixture
def simple_normal_model():
    """Create a simple normal model for testing."""
    # Generate data
    np.random.seed(42)
    n = 200
    x = np.random.uniform(0, 10, n)
    y = 2.0 + 0.5 * x + np.random.normal(0, 1, n)
    
    data = pd.DataFrame({'x': x, 'y': y})
    
    # Fit model
    model = gamlss(
        formula="y ~ x",
        family="NO",
        data=data
    )
    
    return model


@pytest.fixture
def gamma_model():
    """Create a gamma model for testing."""
    np.random.seed(123)
    n = 200
    x = np.random.uniform(0, 5, n)
    
    # Generate gamma data
    mu = np.exp(0.5 + 0.3 * x)
    sigma = 0.5
    shape = 1.0 / (sigma ** 2)
    scale = mu / shape
    y = np.random.gamma(shape, scale, n)
    
    data = pd.DataFrame({'x': x, 'y': y})
    
    # Fit model
    model = gamlss(
        formula="y ~ x",
        family="GA",
        data=data
    )
    
    return model


class TestQuantileResiduals:
    """Test quantile residuals computation."""
    
    def test_basic_computation(self, simple_normal_model):
        """Test basic quantile residuals computation."""
        result = diagnostics.quantile_residuals(simple_normal_model)
        
        assert isinstance(result, diagnostics.QuantileResidualsResult)
        assert len(result.residuals) > 0
        assert result.n == len(result.residuals)
        
        # Check that residuals are approximately standard normal
        assert abs(result.mean) < 0.2  # Mean should be close to 0
        assert abs(result.variance - 1.0) < 0.3  # Variance should be close to 1
        assert abs(result.skewness) < 0.5  # Skewness should be close to 0
        assert abs(result.kurtosis) < 1.0  # Excess kurtosis should be close to 0
    
    def test_gamma_model(self, gamma_model):
        """Test quantile residuals for gamma model."""
        result = diagnostics.quantile_residuals(gamma_model)
        
        assert isinstance(result, diagnostics.QuantileResidualsResult)
        assert len(result.residuals) > 0
        
        # Residuals should still be approximately standard normal
        assert abs(result.mean) < 0.3
        assert abs(result.variance - 1.0) < 0.5


class TestQQPlot:
    """Test Q-Q plot data computation."""
    
    def test_basic_computation(self, simple_normal_model):
        """Test basic Q-Q plot computation."""
        result = diagnostics.qq_plot_data(simple_normal_model)
        
        assert isinstance(result, diagnostics.QQPlotResult)
        assert len(result.theoretical_quantiles) == len(result.sample_quantiles)
        assert len(result.theoretical_quantiles) > 0
        
        # Correlation should be high for well-fitted model
        assert result.correlation > 0.95
        
        # Check that quantiles are sorted
        assert np.all(np.diff(result.sample_quantiles) >= 0)
        assert np.all(np.diff(result.theoretical_quantiles) >= 0)
    
    def test_correlation_range(self, simple_normal_model):
        """Test that correlation is in valid range."""
        result = diagnostics.qq_plot_data(simple_normal_model)
        
        assert -1.0 <= result.correlation <= 1.0


class TestWormPlot:
    """Test worm plot data computation."""
    
    def test_basic_computation(self, simple_normal_model):
        """Test basic worm plot computation."""
        result = diagnostics.worm_plot_data(simple_normal_model)
        
        assert isinstance(result, diagnostics.WormPlotResult)
        assert len(result.theoretical_quantiles) == len(result.deviations)
        assert len(result.lower_band) == len(result.upper_band)
        assert result.n > 0
        
        # Check that bands are symmetric around 0
        assert np.allclose(result.lower_band, -result.upper_band, atol=1e-10)
    
    def test_confidence_bands(self, simple_normal_model):
        """Test that most points are within confidence bands."""
        result = diagnostics.worm_plot_data(simple_normal_model, confidence_level=0.95)
        
        # Count points outside bands
        outside = np.sum((result.deviations < result.lower_band) |
                        (result.deviations > result.upper_band))
        
        # Should be approximately 5% outside for 95% CI
        pct_outside = outside / result.n
        assert pct_outside < 0.15  # Allow some variation
    
    def test_different_confidence_levels(self, simple_normal_model):
        """Test different confidence levels."""
        result_90 = diagnostics.worm_plot_data(simple_normal_model, confidence_level=0.90)
        result_95 = diagnostics.worm_plot_data(simple_normal_model, confidence_level=0.95)
        result_99 = diagnostics.worm_plot_data(simple_normal_model, confidence_level=0.99)
        
        # Higher confidence level should have wider bands
        assert np.all(np.abs(result_90.upper_band) <= np.abs(result_95.upper_band))
        assert np.all(np.abs(result_95.upper_band) <= np.abs(result_99.upper_band))


class TestResidualPlot:
    """Test residual plot data computation."""
    
    def test_basic_computation(self, simple_normal_model):
        """Test basic residual plot computation."""
        result = diagnostics.residual_plot_data(simple_normal_model)
        
        assert isinstance(result, diagnostics.ResidualPlotResult)
        assert len(result.fitted_values) == len(result.residuals)
        assert len(result.index) == len(result.residuals)
        assert len(result.residuals) > 0
    
    def test_with_xvar(self, simple_normal_model):
        """Test residual plot with custom x variable."""
        n = len(simple_normal_model.y)
        xvar = np.arange(n, dtype=np.float64)
        
        result = diagnostics.residual_plot_data(simple_normal_model, xvar=xvar)
        
        assert len(result.index) == len(result.residuals)


class TestCalibrationCheck:
    """Test calibration check."""
    
    def test_basic_computation(self, simple_normal_model):
        """Test basic calibration check."""
        result = diagnostics.calibration_check(simple_normal_model, n_bins=10)
        
        assert isinstance(result, diagnostics.CalibrationResult)
        assert result.n_bins == 10
        assert len(result.predicted_probs) == 10
        assert len(result.observed_probs) == 10
        
        # Check that probabilities are in [0, 1]
        assert np.all(result.predicted_probs >= 0)
        assert np.all(result.predicted_probs <= 1)
        assert np.all(result.observed_probs >= 0)
        assert np.all(result.observed_probs <= 1)
    
    def test_different_bin_counts(self, simple_normal_model):
        """Test different numbers of bins."""
        for n_bins in [5, 10, 20]:
            result = diagnostics.calibration_check(simple_normal_model, n_bins=n_bins)
            assert result.n_bins == n_bins
            assert len(result.predicted_probs) == n_bins


class TestCentileCheck:
    """Test centile check."""
    
    def test_basic_computation(self, simple_normal_model):
        """Test basic centile check."""
        centiles = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
        result = diagnostics.centile_check(simple_normal_model, centiles=centiles)
        
        assert isinstance(result, diagnostics.CentileCheckResult)
        assert len(result.centiles) == len(centiles)
        assert np.allclose(result.centiles, centiles)
        
        # Check coverage
        if len(result.coverage) > 0:
            assert len(result.coverage) == len(centiles)
            assert len(result.expected_coverage) == len(centiles)
            
            # Coverage should be close to expected
            for i, c in enumerate(centiles):
                # Allow 10% deviation
                assert abs(result.coverage[i] - c) < 0.15
    
    def test_default_centiles(self, simple_normal_model):
        """Test with default centiles."""
        result = diagnostics.centile_check(simple_normal_model)
        
        # Default should be [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        assert len(result.centiles) == 7


class TestComprehensiveDiagnostics:
    """Test comprehensive diagnostics."""
    
    def test_basic_computation(self, simple_normal_model):
        """Test comprehensive diagnostics."""
        result = diagnostics.comprehensive_diagnostics(simple_normal_model)
        
        assert isinstance(result, diagnostics.ComprehensiveDiagnostics)
        assert isinstance(result.quantile_residuals, diagnostics.QuantileResidualsResult)
        assert isinstance(result.qq_plot, diagnostics.QQPlotResult)
        assert isinstance(result.worm_plot, diagnostics.WormPlotResult)
        assert isinstance(result.residual_plot, diagnostics.ResidualPlotResult)
        assert isinstance(result.calibration, diagnostics.CalibrationResult)
        assert isinstance(result.centile_check, diagnostics.CentileCheckResult)
    
    def test_gamma_model(self, gamma_model):
        """Test comprehensive diagnostics for gamma model."""
        result = diagnostics.comprehensive_diagnostics(gamma_model)
        
        assert isinstance(result, diagnostics.ComprehensiveDiagnostics)
        # All components should be present
        assert result.quantile_residuals.n > 0
        assert len(result.qq_plot.theoretical_quantiles) > 0


class TestPrintSummary:
    """Test diagnostic summary printing."""
    
    def test_print_summary(self, simple_normal_model, capsys):
        """Test that summary prints without errors."""
        diagnostics.print_diagnostic_summary(simple_normal_model)
        
        captured = capsys.readouterr()
        assert "GAMLSS MODEL DIAGNOSTIC SUMMARY" in captured.out
        assert "Quantile Residuals:" in captured.out
        assert "Q-Q Plot:" in captured.out
        assert "Worm Plot:" in captured.out


class TestPlotting:
    """Test plotting functions."""
    
    def test_plot_diagnostics_no_display(self, simple_normal_model, tmp_path):
        """Test that plotting works and saves to file."""
        save_path = tmp_path / "diagnostics.png"
        
        # This should not raise an error
        try:
            diagnostics.plot_diagnostics(simple_normal_model, save_path=str(save_path))
            assert save_path.exists()
        except ImportError:
            # matplotlib not installed, skip test
            pytest.skip("matplotlib not installed")


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_residuals(self):
        """Test handling of models with no valid residuals."""
        # This is hard to test without creating a pathological model
        # Skip for now
        pass
    
    def test_small_sample(self):
        """Test with very small sample size."""
        np.random.seed(42)
        n = 10  # Very small
        x = np.random.uniform(0, 10, n)
        y = 2.0 + 0.5 * x + np.random.normal(0, 1, n)
        
        data = pd.DataFrame({'x': x, 'y': y})
        
        model = gamlss(
            formula="y ~ x",
            family="NO",
            data=data
        )
        
        # Should still work
        result = diagnostics.quantile_residuals(model)
        assert result.n == n


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
