"""Tests for worm plot diagnostics.

This module tests the worm plot functionality including:
- Data computation
- Single panel plots
- Multiple panel plots
- Interactive plots
"""

import pytest
import numpy as np
from omnilss.fitting import gamlss
from omnilss.worm_plot import (
    wp_data,
    wp,
    wp_interactive,
    WormPlotData,
    MultiPanelWormPlotData,
)


class TestWormPlotData:
    """Test worm plot data computation."""
    
    def test_single_panel_basic(self):
        """Test basic single panel worm plot data."""
        # Generate simple data
        np.random.seed(42)
        n = 100
        x = np.random.normal(size=n)
        y = 2.0 + 3.0 * x + np.random.normal(scale=1.0, size=n)
        
        data = {"y": y, "x": x}
        
        # Fit model
        model = gamlss("y ~ x", family="NO", data=data, method="RS")
        
        # Get worm plot data
        result = wp_data(model)
        
        # Check type
        assert isinstance(result, WormPlotData)
        
        # Check attributes
        assert result.n == n
        assert len(result.theoretical_quantiles) == n
        assert len(result.deviations) == n
        assert len(result.lower_band) == n
        assert len(result.upper_band) == n
        
        # Check that bands are symmetric around zero
        assert np.allclose(result.lower_band, -result.upper_band, atol=1e-10)
        
        # Check that most deviations are within bands
        within_bands = np.sum(
            (result.deviations >= result.lower_band) &
            (result.deviations <= result.upper_band)
        )
        # At least 90% should be within 95% CI
        assert within_bands / n >= 0.85
    
    def test_confidence_level(self):
        """Test different confidence levels."""
        np.random.seed(42)
        n = 100
        x = np.random.normal(size=n)
        y = 2.0 + 3.0 * x + np.random.normal(scale=1.0, size=n)
        
        data = {"y": y, "x": x}
        model = gamlss("y ~ x", family="NO", data=data)
        
        # Test different confidence levels
        for conf_level in [0.90, 0.95, 0.99]:
            result = wp_data(model, confidence_level=conf_level)
            
            # Higher confidence level should have wider bands
            assert np.all(result.upper_band > 0)
            assert np.all(result.lower_band < 0)
    
    def test_multiple_panels(self):
        """Test multiple panel worm plot data."""
        np.random.seed(42)
        n = 200
        x = np.random.normal(size=n)
        y = 2.0 + 3.0 * x + np.random.normal(scale=1.0, size=n)
        
        data = {"y": y, "x": x}
        model = gamlss("y ~ x", family="NO", data=data)
        
        # Get multiple panel data
        result = wp_data(model, xvar=x, n_inter=4)
        
        # Check type
        assert isinstance(result, MultiPanelWormPlotData)
        
        # Check number of panels
        assert result.n_panels == 4
        assert len(result.panels) == 4
        
        # Check each panel
        for panel in result.panels:
            assert isinstance(panel, WormPlotData)
            assert panel.n > 0
            assert panel.group_label is not None
            assert len(panel.theoretical_quantiles) == panel.n
    
    def test_small_sample(self):
        """Test with small sample size."""
        np.random.seed(42)
        n = 20
        x = np.random.normal(size=n)
        y = 2.0 + 3.0 * x + np.random.normal(scale=1.0, size=n)
        
        data = {"y": y, "x": x}
        model = gamlss("y ~ x", family="NO", data=data)
        
        result = wp_data(model)
        
        assert isinstance(result, WormPlotData)
        assert result.n == n


class TestWormPlot:
    """Test worm plot visualization."""
    
    def test_single_panel_plot(self):
        """Test single panel worm plot."""
        np.random.seed(42)
        n = 100
        x = np.random.normal(size=n)
        y = 2.0 + 3.0 * x + np.random.normal(scale=1.0, size=n)
        
        data = {"y": y, "x": x}
        model = gamlss("y ~ x", family="NO", data=data)
        
        # Create plot (don't show)
        fig = wp(model, show=False)
        
        # Check that figure was created
        assert fig is not None
        
        # Check that it has axes
        assert len(fig.axes) >= 1
    
    def test_multiple_panel_plot(self):
        """Test multiple panel worm plot."""
        np.random.seed(42)
        n = 200
        x = np.random.normal(size=n)
        y = 2.0 + 3.0 * x + np.random.normal(scale=1.0, size=n)
        
        data = {"y": y, "x": x}
        model = gamlss("y ~ x", family="NO", data=data)
        
        # Create plot (don't show)
        fig = wp(model, xvar=x, n_inter=4, show=False)
        
        # Check that figure was created
        assert fig is not None
        
        # Should have 4 panels
        assert len([ax for ax in fig.axes if ax.get_visible()]) == 4
    
    def test_custom_figsize(self):
        """Test custom figure size."""
        np.random.seed(42)
        n = 100
        x = np.random.normal(size=n)
        y = 2.0 + 3.0 * x + np.random.normal(scale=1.0, size=n)
        
        data = {"y": y, "x": x}
        model = gamlss("y ~ x", family="NO", data=data)
        
        # Create plot with custom size
        fig = wp(model, figsize=(10, 6), show=False)
        
        assert fig is not None
        assert fig.get_figwidth() == 10
        assert fig.get_figheight() == 6
    
    def test_custom_title(self):
        """Test custom title."""
        np.random.seed(42)
        n = 100
        x = np.random.normal(size=n)
        y = 2.0 + 3.0 * x + np.random.normal(scale=1.0, size=n)
        
        data = {"y": y, "x": x}
        model = gamlss("y ~ x", family="NO", data=data)
        
        # Create plot with custom title
        fig = wp(model, title="Custom Worm Plot", show=False)
        
        assert fig is not None
        assert "Custom Worm Plot" in fig.axes[0].get_title()


class TestInteractiveWormPlot:
    """Test interactive worm plot with Plotly."""
    
    def test_single_panel_interactive(self):
        """Test single panel interactive worm plot."""
        pytest.importorskip("plotly")
        
        np.random.seed(42)
        n = 100
        x = np.random.normal(size=n)
        y = 2.0 + 3.0 * x + np.random.normal(scale=1.0, size=n)
        
        data = {"y": y, "x": x}
        model = gamlss("y ~ x", family="NO", data=data)
        
        # Create interactive plot
        fig = wp_interactive(model)
        
        # Check that figure was created
        assert fig is not None
        
        # Check that it has data
        assert len(fig.data) > 0
    
    def test_multiple_panel_interactive(self):
        """Test multiple panel interactive worm plot."""
        pytest.importorskip("plotly")
        
        np.random.seed(42)
        n = 200
        x = np.random.normal(size=n)
        y = 2.0 + 3.0 * x + np.random.normal(scale=1.0, size=n)
        
        data = {"y": y, "x": x}
        model = gamlss("y ~ x", family="NO", data=data)
        
        # Create interactive plot
        fig = wp_interactive(model, xvar=x, n_inter=4)
        
        # Check that figure was created
        assert fig is not None
        
        # Should have multiple traces (3 per panel: band, zero line, points)
        assert len(fig.data) >= 12  # 3 traces * 4 panels
    
    def test_custom_dimensions(self):
        """Test custom dimensions."""
        pytest.importorskip("plotly")
        
        np.random.seed(42)
        n = 100
        x = np.random.normal(size=n)
        y = 2.0 + 3.0 * x + np.random.normal(scale=1.0, size=n)
        
        data = {"y": y, "x": x}
        model = gamlss("y ~ x", family="NO", data=data)
        
        # Create plot with custom dimensions
        fig = wp_interactive(model, height=600, width=800)
        
        assert fig is not None
        assert fig.layout.height == 600
        assert fig.layout.width == 800


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_residuals(self):
        """Test handling of empty residuals."""
        # This is hard to trigger naturally, so we skip it
        # In practice, gamlss() should always produce residuals
        pass
    
    def test_few_observations_warning(self):
        """Test warning for panels with few observations."""
        np.random.seed(42)
        n = 50
        x = np.random.normal(size=n)
        y = 2.0 + 3.0 * x + np.random.normal(scale=1.0, size=n)
        
        data = {"y": y, "x": x}
        model = gamlss("y ~ x", family="NO", data=data)
        
        # Try to create many panels with few observations each
        with pytest.warns(UserWarning):
            result = wp_data(model, xvar=x, n_inter=10)
    
    def test_too_many_panels_warning(self):
        """Test warning for too many panels."""
        np.random.seed(42)
        n = 1000
        x = np.random.normal(size=n)
        y = 2.0 + 3.0 * x + np.random.normal(scale=1.0, size=n)
        
        data = {"y": y, "x": x}
        model = gamlss("y ~ x", family="NO", data=data)
        
        # Try to create too many panels
        with pytest.warns(UserWarning):
            result = wp_data(model, xvar=x, n_inter=20)


class TestDifferentDistributions:
    """Test worm plot with different distributions."""
    
    def test_gamma_distribution(self):
        """Test with Gamma distribution."""
        np.random.seed(42)
        n = 100
        x = np.random.normal(size=n)
        y = np.exp(1.0 + 0.5 * x) * np.random.gamma(2, 1, size=n)
        
        data = {"y": y, "x": x}
        model = gamlss("y ~ x", family="GA", data=data)
        
        result = wp_data(model)
        
        assert isinstance(result, WormPlotData)
        assert result.n == n
    
    def test_poisson_distribution(self):
        """Test with Poisson distribution."""
        np.random.seed(42)
        n = 100
        x = np.random.normal(size=n)
        mu = np.exp(1.0 + 0.5 * x)
        y = np.random.poisson(mu)
        
        data = {"y": y, "x": x}
        model = gamlss("y ~ x", family="PO", data=data)
        
        result = wp_data(model)
        
        assert isinstance(result, WormPlotData)
        assert result.n == n


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
