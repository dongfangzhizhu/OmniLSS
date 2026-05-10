"""Tests for formula parser with s() function support.

This module tests the formula parser's ability to recognize and parse
s() terms with automatic lambda selection.
"""

import pytest
import numpy as np

from omnilss.formula_parser import (
    parse_formula,
    build_design_matrix,
    ParsedFormula,
    SmoothTerm,
    LinearTerm,
)


class TestSFunctionParsing:
    """Test parsing of s() function in formulas."""
    
    def test_simple_s_term(self):
        """Test parsing a simple s(x) term."""
        parsed = parse_formula("y ~ s(x)")
        
        assert parsed.response == "y"
        assert len(parsed.smooth_terms) == 1
        assert len(parsed.linear_terms) == 0
        assert parsed.has_intercept is True
        
        smooth = parsed.smooth_terms[0]
        assert smooth.variable == "x"
        assert smooth.smoother == "s"
        assert hasattr(smooth, 'method')
        assert smooth.method == "auto"
    
    def test_s_with_method(self):
        """Test parsing s(x, method='REML')."""
        parsed = parse_formula("y ~ s(x, method='REML')")
        
        assert len(parsed.smooth_terms) == 1
        smooth = parsed.smooth_terms[0]
        assert smooth.variable == "x"
        assert hasattr(smooth, 'method')
        assert smooth.method == "REML"
    
    def test_s_with_method_double_quotes(self):
        """Test parsing s(x, method=\"GCV\")."""
        parsed = parse_formula('y ~ s(x, method="GCV")')
        
        assert len(parsed.smooth_terms) == 1
        smooth = parsed.smooth_terms[0]
        assert hasattr(smooth, 'method')
        assert smooth.method == "GCV"
    
    def test_s_with_smoother(self):
        """Test parsing s(x, smoother='ps')."""
        parsed = parse_formula("y ~ s(x, smoother='ps')")
        
        assert len(parsed.smooth_terms) == 1
        smooth = parsed.smooth_terms[0]
        assert smooth.variable == "x"
        assert smooth.smoother == "ps"
    
    def test_s_with_method_and_smoother(self):
        """Test parsing s(x, method='REML', smoother='cs')."""
        parsed = parse_formula("y ~ s(x, method='REML', smoother='cs')")
        
        assert len(parsed.smooth_terms) == 1
        smooth = parsed.smooth_terms[0]
        assert smooth.variable == "x"
        assert smooth.smoother == "cs"
        assert hasattr(smooth, 'method')
        assert smooth.method == "REML"
    
    def test_s_with_df(self):
        """Test parsing s(x, df=5)."""
        parsed = parse_formula("y ~ s(x, df=5)")
        
        assert len(parsed.smooth_terms) == 1
        smooth = parsed.smooth_terms[0]
        assert smooth.variable == "x"
        assert smooth.df == 5.0
    
    def test_s_with_lambda(self):
        """Test parsing s(x, lambda=0.01)."""
        parsed = parse_formula("y ~ s(x, lambda=0.01)")
        
        assert len(parsed.smooth_terms) == 1
        smooth = parsed.smooth_terms[0]
        assert smooth.variable == "x"
        assert smooth.lambda_ == 0.01
    
    def test_s_with_additional_kwargs(self):
        """Test parsing s(x, method='GCV', degree=3, n_knots=20)."""
        parsed = parse_formula("y ~ s(x, method='GCV', degree=3, n_knots=20)")
        
        assert len(parsed.smooth_terms) == 1
        smooth = parsed.smooth_terms[0]
        assert smooth.variable == "x"
        assert hasattr(smooth, 'method')
        assert smooth.method == "GCV"
        assert smooth.kwargs['degree'] == 3.0
        assert smooth.kwargs['n_knots'] == 20.0


class TestMultipleSTerms:
    """Test formulas with multiple s() terms."""
    
    def test_two_s_terms(self):
        """Test parsing y ~ s(x1) + s(x2)."""
        parsed = parse_formula("y ~ s(x1) + s(x2)")
        
        assert len(parsed.smooth_terms) == 2
        assert parsed.smooth_terms[0].variable == "x1"
        assert parsed.smooth_terms[1].variable == "x2"
    
    def test_s_terms_with_different_methods(self):
        """Test parsing y ~ s(x1, method='GCV') + s(x2, method='REML')."""
        parsed = parse_formula("y ~ s(x1, method='GCV') + s(x2, method='REML')")
        
        assert len(parsed.smooth_terms) == 2
        assert parsed.smooth_terms[0].method == "GCV"
        assert parsed.smooth_terms[1].method == "REML"
    
    def test_s_terms_with_different_smoothers(self):
        """Test parsing y ~ s(x1, smoother='pb') + s(x2, smoother='cs')."""
        parsed = parse_formula("y ~ s(x1, smoother='pb') + s(x2, smoother='cs')")
        
        assert len(parsed.smooth_terms) == 2
        assert parsed.smooth_terms[0].smoother == "pb"
        assert parsed.smooth_terms[1].smoother == "cs"


class TestMixedTerms:
    """Test formulas with mixed linear and smooth terms."""
    
    def test_linear_and_s(self):
        """Test parsing y ~ x1 + s(x2)."""
        parsed = parse_formula("y ~ x1 + s(x2)")
        
        assert len(parsed.linear_terms) == 1
        assert len(parsed.smooth_terms) == 1
        assert parsed.linear_terms[0].variable == "x1"
        assert parsed.smooth_terms[0].variable == "x2"
    
    def test_s_and_linear(self):
        """Test parsing y ~ s(x1) + x2."""
        parsed = parse_formula("y ~ s(x1) + x2")
        
        assert len(parsed.smooth_terms) == 1
        assert len(parsed.linear_terms) == 1
        assert parsed.smooth_terms[0].variable == "x1"
        assert parsed.linear_terms[0].variable == "x2"
    
    def test_multiple_linear_and_s(self):
        """Test parsing y ~ x1 + x2 + s(x3) + s(x4)."""
        parsed = parse_formula("y ~ x1 + x2 + s(x3) + s(x4)")
        
        assert len(parsed.linear_terms) == 2
        assert len(parsed.smooth_terms) == 2


class TestLegacySmoothTerms:
    """Test that legacy pb(), ps(), cs() still work."""
    
    def test_pb_term(self):
        """Test parsing y ~ pb(x, df=5)."""
        parsed = parse_formula("y ~ pb(x, df=5)")
        
        assert len(parsed.smooth_terms) == 1
        smooth = parsed.smooth_terms[0]
        assert smooth.variable == "x"
        assert smooth.smoother == "pb"
        assert smooth.df == 5.0
    
    def test_ps_term(self):
        """Test parsing y ~ ps(x, lambda=0.01)."""
        parsed = parse_formula("y ~ ps(x, lambda=0.01)")
        
        assert len(parsed.smooth_terms) == 1
        smooth = parsed.smooth_terms[0]
        assert smooth.variable == "x"
        assert smooth.smoother == "ps"
        assert smooth.lambda_ == 0.01
    
    def test_cs_term(self):
        """Test parsing y ~ cs(x, df=10)."""
        parsed = parse_formula("y ~ cs(x, df=10)")
        
        assert len(parsed.smooth_terms) == 1
        smooth = parsed.smooth_terms[0]
        assert smooth.variable == "x"
        assert smooth.smoother == "cs"
        assert smooth.df == 10.0
    
    def test_mixed_legacy_and_s(self):
        """Test parsing y ~ pb(x1, df=5) + s(x2, method='REML')."""
        parsed = parse_formula("y ~ pb(x1, df=5) + s(x2, method='REML')")
        
        assert len(parsed.smooth_terms) == 2
        
        # pb term should not have method attribute
        pb_term = parsed.smooth_terms[0]
        assert pb_term.variable == "x1"
        assert pb_term.smoother == "pb"
        assert pb_term.df == 5.0
        assert not hasattr(pb_term, 'method')
        
        # s term should have method attribute
        s_term = parsed.smooth_terms[1]
        assert s_term.variable == "x2"
        assert s_term.smoother == "s"
        assert hasattr(s_term, 'method')
        assert s_term.method == "REML"


class TestInterceptHandling:
    """Test intercept handling with s() terms."""
    
    def test_s_with_intercept(self):
        """Test parsing y ~ s(x) (should have intercept)."""
        parsed = parse_formula("y ~ s(x)")
        assert parsed.has_intercept is True
    
    def test_s_without_intercept(self):
        """Test parsing y ~ s(x) - 1."""
        parsed = parse_formula("y ~ s(x) - 1")
        assert parsed.has_intercept is False
    
    def test_s_without_intercept_alt(self):
        """Test parsing y ~ s(x) -1 (no space)."""
        parsed = parse_formula("y ~ s(x) -1")
        assert parsed.has_intercept is False


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_s_without_variable(self):
        """Test that s() without variable raises error."""
        with pytest.raises(ValueError, match="must have a variable"):
            parse_formula("y ~ s()")
    
    def test_invalid_method_value(self):
        """Test parsing with invalid method value (should still parse)."""
        # Parser doesn't validate method values, that's done by SmoothTerm
        parsed = parse_formula("y ~ s(x, method='INVALID')")
        assert len(parsed.smooth_terms) == 1
        assert parsed.smooth_terms[0].method == "INVALID"
    
    def test_s_with_spaces(self):
        """Test parsing s( x , method = 'GCV' ) with spaces."""
        parsed = parse_formula("y ~ s( x , method = 'GCV' )")
        
        assert len(parsed.smooth_terms) == 1
        smooth = parsed.smooth_terms[0]
        assert smooth.variable == "x"
        assert smooth.method == "GCV"


class TestDesignMatrixWithS:
    """Test design matrix construction with s() terms."""
    
    def test_build_design_with_s(self):
        """Test building design matrix with s(x)."""
        data = {
            'y': np.random.randn(50),
            'x': np.linspace(0, 1, 50),
        }
        
        parsed = parse_formula("y ~ s(x)")
        X, smooth_info = build_design_matrix(parsed, data, fit_smooths=True)
        
        # Should have intercept + basis functions
        assert X.shape[0] == 50
        assert X.shape[1] > 1  # intercept + basis
        
        # Should have smooth info
        assert len(smooth_info) > 0
    
    def test_build_design_with_s_and_linear(self):
        """Test building design matrix with x1 + s(x2)."""
        data = {
            'y': np.random.randn(50),
            'x1': np.random.randn(50),
            'x2': np.linspace(0, 1, 50),
        }
        
        parsed = parse_formula("y ~ x1 + s(x2)")
        X, smooth_info = build_design_matrix(parsed, data, fit_smooths=True)
        
        # Should have intercept + x1 + basis functions
        assert X.shape[0] == 50
        assert X.shape[1] > 2  # intercept + x1 + basis


class TestBackwardCompatibility:
    """Test that existing code still works."""
    
    def test_old_pb_formula(self):
        """Test that old pb() formulas still work."""
        parsed = parse_formula("y ~ pb(x, df=5)")
        
        assert len(parsed.smooth_terms) == 1
        assert parsed.smooth_terms[0].smoother == "pb"
        assert parsed.smooth_terms[0].df == 5.0
    
    def test_old_mixed_formula(self):
        """Test that old mixed formulas still work."""
        parsed = parse_formula("y ~ x1 + pb(x2, df=5) + x3")
        
        assert len(parsed.linear_terms) == 2
        assert len(parsed.smooth_terms) == 1
    
    def test_old_no_intercept(self):
        """Test that old no-intercept formulas still work."""
        parsed = parse_formula("y ~ x1 + pb(x2, df=5) - 1")
        
        assert parsed.has_intercept is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
