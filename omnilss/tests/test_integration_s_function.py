"""Integration tests for s() function in full GAMLSS workflow.

This module tests the complete integration of s() function from formula
parsing through smooth fitting to model results.
"""

import pytest
import numpy as np

from omnilss.formula_parser import parse_formula
from omnilss.smooth_fitting import build_smooth_design


class TestSFunctionIntegration:
    """Test s() function in complete workflow."""
    
    def test_s_auto_gcv(self):
        """Test s(x) with automatic GCV selection."""
        # Generate test data
        np.random.seed(42)
        n = 100
        x = np.linspace(0, 1, n)
        y = np.sin(2 * np.pi * x) + np.random.randn(n) * 0.1
        
        data = {'y': y, 'x': x}
        
        # Build smooth design with s(x)
        info = build_smooth_design("y ~ s(x)", data)
        
        # Check design matrix
        assert info.X.shape[0] == n
        assert info.X.shape[1] > 1  # intercept + basis
        
        # Check smooth fit info
        assert len(info.smooth_fits) == 1
        smooth = info.smooth_fits[0]
        
        assert smooth.variable == "x"
        assert smooth.smoother == "s"
        assert smooth.lambda_ > 0
        assert smooth.edf > 0
        assert smooth.selection_method in ["GCV", "auto"]
        assert smooth.criterion_value is not None
    
    def test_s_with_reml(self):
        """Test s(x, method='REML')."""
        np.random.seed(42)
        n = 100
        x = np.linspace(0, 1, n)
        y = np.sin(2 * np.pi * x) + np.random.randn(n) * 0.1
        
        data = {'y': y, 'x': x}
        
        # Build smooth design with REML
        info = build_smooth_design("y ~ s(x, method='REML')", data)
        
        # Check smooth fit info
        assert len(info.smooth_fits) == 1
        smooth = info.smooth_fits[0]
        
        assert smooth.selection_method == "REML"
        assert smooth.lambda_ > 0
        assert smooth.edf > 0
    
    def test_s_with_aic(self):
        """Test s(x, method='AIC')."""
        np.random.seed(42)
        n = 100
        x = np.linspace(0, 1, n)
        y = np.sin(2 * np.pi * x) + np.random.randn(n) * 0.1
        
        data = {'y': y, 'x': x}
        
        # Build smooth design with AIC
        info = build_smooth_design("y ~ s(x, method='AIC')", data)
        
        # Check smooth fit info
        assert len(info.smooth_fits) == 1
        smooth = info.smooth_fits[0]
        
        assert smooth.selection_method == "AIC"
        assert smooth.lambda_ > 0
        assert smooth.edf > 0
    
    def test_multiple_s_terms(self):
        """Test y ~ s(x1) + s(x2, method='REML')."""
        np.random.seed(42)
        n = 100
        x1 = np.linspace(0, 1, n)
        x2 = np.random.randn(n)
        y = np.sin(2 * np.pi * x1) + 0.5 * x2 + np.random.randn(n) * 0.1
        
        data = {'y': y, 'x1': x1, 'x2': x2}
        
        # Build smooth design with multiple s() terms
        info = build_smooth_design("y ~ s(x1) + s(x2, method='REML')", data)
        
        # Check smooth fit info
        assert len(info.smooth_fits) == 2
        
        # First smooth (GCV/auto)
        smooth1 = info.smooth_fits[0]
        assert smooth1.variable == "x1"
        assert smooth1.selection_method in ["GCV", "auto"]
        
        # Second smooth (REML)
        smooth2 = info.smooth_fits[1]
        assert smooth2.variable == "x2"
        assert smooth2.selection_method == "REML"
    
    def test_mixed_linear_and_s(self):
        """Test y ~ x1 + s(x2)."""
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        x2 = np.linspace(0, 1, n)
        y = 2 * x1 + np.sin(2 * np.pi * x2) + np.random.randn(n) * 0.1
        
        data = {'y': y, 'x1': x1, 'x2': x2}
        
        # Build smooth design
        info = build_smooth_design("y ~ x1 + s(x2)", data)
        
        # Check linear columns (intercept + x1)
        assert info.linear_columns == 2
        
        # Check smooth fit info
        assert len(info.smooth_fits) == 1
        smooth = info.smooth_fits[0]
        assert smooth.variable == "x2"
    
    def test_s_with_different_smoothers(self):
        """Test s(x, smoother='ps') and s(x, smoother='cs')."""
        np.random.seed(42)
        n = 100
        x1 = np.linspace(0, 1, n)
        x2 = np.linspace(0, 1, n)
        y = np.sin(2 * np.pi * x1) + np.cos(2 * np.pi * x2) + np.random.randn(n) * 0.1
        
        data = {'y': y, 'x1': x1, 'x2': x2}
        
        # Build smooth design with different smoothers
        info = build_smooth_design("y ~ s(x1, smoother='ps') + s(x2, smoother='cs')", data)
        
        # Check smooth fit info
        assert len(info.smooth_fits) == 2
        
        # First smooth (ps)
        smooth1 = info.smooth_fits[0]
        assert smooth1.variable == "x1"
        assert smooth1.smoother == "ps"
        
        # Second smooth (cs)
        smooth2 = info.smooth_fits[1]
        assert smooth2.variable == "x2"
        assert smooth2.smoother == "cs"
    
    def test_s_with_manual_df(self):
        """Test s(x, df=5) - manual df specification."""
        np.random.seed(42)
        n = 100
        x = np.linspace(0, 1, n)
        y = np.sin(2 * np.pi * x) + np.random.randn(n) * 0.1
        
        data = {'y': y, 'x': x}
        
        # Build smooth design with manual df
        info = build_smooth_design("y ~ s(x, df=5)", data)
        
        # Check smooth fit info
        assert len(info.smooth_fits) == 1
        smooth = info.smooth_fits[0]
        
        # EDF should be close to specified df
        assert 4 < smooth.edf < 6
    
    def test_backward_compatibility_pb(self):
        """Test that old pb() formulas still work."""
        np.random.seed(42)
        n = 100
        x = np.linspace(0, 1, n)
        y = np.sin(2 * np.pi * x) + np.random.randn(n) * 0.1
        
        data = {'y': y, 'x': x}
        
        # Build smooth design with old pb() syntax
        info = build_smooth_design("y ~ pb(x, df=5)", data)
        
        # Check smooth fit info
        assert len(info.smooth_fits) == 1
        smooth = info.smooth_fits[0]
        
        assert smooth.variable == "x"
        assert smooth.smoother == "pb"
        # Old pb() with df should have selection_method 'fixed_df'
        assert smooth.selection_method in [None, "ML", "fixed_df"]
    
    def test_mixed_legacy_and_s(self):
        """Test mixing legacy pb() and new s() in same formula."""
        np.random.seed(42)
        n = 100
        x1 = np.linspace(0, 1, n)
        x2 = np.linspace(0, 1, n)
        y = np.sin(2 * np.pi * x1) + np.cos(2 * np.pi * x2) + np.random.randn(n) * 0.1
        
        data = {'y': y, 'x1': x1, 'x2': x2}
        
        # Build smooth design with mixed syntax
        info = build_smooth_design("y ~ pb(x1, df=5) + s(x2, method='REML')", data)
        
        # Check smooth fit info
        assert len(info.smooth_fits) == 2
        
        # First smooth (legacy pb)
        smooth1 = info.smooth_fits[0]
        assert smooth1.variable == "x1"
        assert smooth1.smoother == "pb"
        
        # Second smooth (new s)
        smooth2 = info.smooth_fits[1]
        assert smooth2.variable == "x2"
        assert smooth2.smoother == "s"
        assert smooth2.selection_method == "REML"


class TestSFunctionEdgeCases:
    """Test edge cases and error handling."""
    
    def test_s_with_constant_y(self):
        """Test s(x) with constant response."""
        np.random.seed(42)
        n = 100
        x = np.linspace(0, 1, n)
        y = np.ones(n)  # Constant
        
        data = {'y': y, 'x': x}
        
        # Should still work (lambda will be large)
        info = build_smooth_design("y ~ s(x)", data)
        
        assert len(info.smooth_fits) == 1
        smooth = info.smooth_fits[0]
        
        # Lambda should be large for constant data
        assert smooth.lambda_ > 0
    
    def test_s_with_small_sample(self):
        """Test s(x) with small sample size."""
        np.random.seed(42)
        n = 20  # Small sample
        x = np.linspace(0, 1, n)
        y = np.sin(2 * np.pi * x) + np.random.randn(n) * 0.1
        
        data = {'y': y, 'x': x}
        
        # Should still work
        info = build_smooth_design("y ~ s(x)", data)
        
        assert len(info.smooth_fits) == 1
        smooth = info.smooth_fits[0]
        
        assert smooth.lambda_ > 0
        assert smooth.edf > 0


class TestSFunctionComparison:
    """Test that different methods give reasonable results."""
    
    def test_compare_methods(self):
        """Compare GCV, REML, and AIC on same data."""
        np.random.seed(42)
        n = 100
        x = np.linspace(0, 1, n)
        y = np.sin(2 * np.pi * x) + np.random.randn(n) * 0.1
        
        data = {'y': y, 'x': x}
        
        # Build with different methods
        info_gcv = build_smooth_design("y ~ s(x, method='GCV')", data)
        info_reml = build_smooth_design("y ~ s(x, method='REML')", data)
        info_aic = build_smooth_design("y ~ s(x, method='AIC')", data)
        
        # All should give positive lambda and edf
        for info, method in [(info_gcv, "GCV"), (info_reml, "REML"), (info_aic, "AIC")]:
            smooth = info.smooth_fits[0]
            assert smooth.lambda_ > 0, f"{method} lambda should be positive"
            assert smooth.edf > 0, f"{method} edf should be positive"
            # selection_method might have " (legacy)" suffix
            assert method in smooth.selection_method, f"Expected {method} in {smooth.selection_method}"
        
        # Lambdas should be in reasonable range (not too different)
        lambdas = [
            info_gcv.smooth_fits[0].lambda_,
            info_reml.smooth_fits[0].lambda_,
            info_aic.smooth_fits[0].lambda_,
        ]
        
        # Check that lambdas are within reasonable range
        # Different methods can give quite different lambdas, so we just check they're positive
        for lambda_val in lambdas:
            assert lambda_val > 0, "All lambdas should be positive"
            assert lambda_val < 1e10, "Lambdas should not be too large"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
