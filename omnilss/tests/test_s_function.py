"""Tests for the s() function and SmoothTerm class."""

import numpy as np

from omnilss.smoothers import s, SmoothTerm


def test_s_basic():
    """Test basic s() function creation."""
    term = s("x")
    
    assert isinstance(term, SmoothTerm)
    assert term.variable == "x"
    assert term.method == "auto"
    assert term.smoother == "pb"
    assert term.df is None
    assert term.lambda_ is None
    assert term.kwargs == {}


def test_s_with_method():
    """Test s() with different methods."""
    # GCV
    term = s("x", method="GCV")
    assert term.method == "GCV"
    
    # REML
    term = s("x", method="REML")
    assert term.method == "REML"
    
    # AIC
    term = s("x", method="AIC")
    assert term.method == "AIC"
    
    # ML
    term = s("x", method="ML")
    assert term.method == "ML"


def test_s_with_smoother():
    """Test s() with different smoothers."""
    # pb (default)
    term = s("x", smoother="pb")
    assert term.smoother == "pb"
    
    # ps
    term = s("x", smoother="ps")
    assert term.smoother == "ps"
    
    # cs
    term = s("x", smoother="cs")
    assert term.smoother == "cs"


def test_s_with_df():
    """Test s() with fixed df."""
    term = s("x", df=5)
    
    assert term.df == 5
    assert term.method == "manual"
    assert term.lambda_ is None


def test_s_with_lambda():
    """Test s() with fixed lambda."""
    term = s("x", lambda_=0.01)
    
    assert term.lambda_ == 0.01
    assert term.method == "manual"
    assert term.df is None


def test_s_with_kwargs():
    """Test s() with additional arguments."""
    term = s("x", n_knots=20, degree=3, order=2)
    
    assert term.kwargs == {"n_knots": 20, "degree": 3, "order": 2}


def test_s_combined():
    """Test s() with multiple options."""
    term = s("x", method="REML", smoother="ps", ps_intervals=30)
    
    assert term.variable == "x"
    assert term.method == "REML"
    assert term.smoother == "ps"
    assert term.kwargs == {"ps_intervals": 30}


def test_smooth_term_validation():
    """Test SmoothTerm validation."""
    # Valid term
    term = SmoothTerm("x", method="auto", smoother="pb")
    assert term.variable == "x"
    
    # Invalid variable
    try:
        SmoothTerm("", method="auto", smoother="pb")
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "variable must be a non-empty string" in str(e)
    
    # Invalid method
    try:
        SmoothTerm("x", method="invalid", smoother="pb")
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "method must be one of" in str(e)
    
    # Invalid smoother
    try:
        SmoothTerm("x", method="auto", smoother="invalid")
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "smoother must be one of" in str(e)
    
    # Both df and lambda_
    try:
        SmoothTerm("x", method="auto", smoother="pb", df=5, lambda_=0.01)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "Cannot specify both df and lambda_" in str(e)
    
    # Negative df
    try:
        SmoothTerm("x", method="auto", smoother="pb", df=-1)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "df must be positive" in str(e)
    
    # Negative lambda_
    try:
        SmoothTerm("x", method="auto", smoother="pb", lambda_=-0.01)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "lambda_ must be non-negative" in str(e)


def test_smooth_term_repr():
    """Test SmoothTerm string representation."""
    # Basic
    term = s("x")
    repr_str = repr(term)
    assert "s('x')" in repr_str
    
    # With method
    term = s("x", method="REML")
    repr_str = repr(term)
    assert "method='REML'" in repr_str
    
    # With smoother
    term = s("x", smoother="cs")
    repr_str = repr(term)
    assert "smoother='cs'" in repr_str
    
    # With df
    term = s("x", df=5)
    repr_str = repr(term)
    assert "df=5" in repr_str
    
    # With kwargs
    term = s("x", n_knots=20)
    repr_str = repr(term)
    assert "n_knots=20" in repr_str


def test_smooth_term_to_dict():
    """Test SmoothTerm to_dict method."""
    term = s("x", method="REML", smoother="ps", df=5, n_knots=20)
    
    d = term.to_dict()
    
    assert d["variable"] == "x"
    assert d["method"] == "manual"  # df specified, so method is manual
    assert d["smoother"] == "ps"
    assert d["df"] == 5
    assert d["lambda_"] is None
    assert d["kwargs"] == {"n_knots": 20}


def test_s_examples():
    """Test examples from docstring."""
    # Automatic lambda selection
    term = s("x")
    assert term.variable == "x"
    assert term.method == "auto"
    
    # Use REML
    term = s("x", method="REML")
    assert term.method == "REML"
    
    # Manual df
    term = s("x", df=5)
    assert term.df == 5
    assert term.method == "manual"
    
    # Cubic splines
    term = s("x", smoother="cs")
    assert term.smoother == "cs"
    
    # With knots
    term = s("x", n_knots=20)
    assert term.kwargs["n_knots"] == 20
    
    # Multiple options
    term = s("x", method="REML", smoother="ps", ps_intervals=30)
    assert term.method == "REML"
    assert term.smoother == "ps"
    assert term.kwargs["ps_intervals"] == 30


def test_s_integration_ready():
    """Test that s() creates terms ready for integration."""
    # Create various terms
    terms = [
        s("x1"),
        s("x2", method="REML"),
        s("x3", smoother="ps"),
        s("x4", df=5),
        s("x5", lambda_=0.01),
        s("x6", n_knots=30, degree=3),
    ]
    
    # All should be SmoothTerm instances
    for term in terms:
        assert isinstance(term, SmoothTerm)
        assert isinstance(term.variable, str)
        assert isinstance(term.method, str)
        assert isinstance(term.smoother, str)
        assert isinstance(term.kwargs, dict)


if __name__ == "__main__":
    # Run tests
    test_s_basic()
    test_s_with_method()
    test_s_with_smoother()
    test_s_with_df()
    test_s_with_lambda()
    test_s_with_kwargs()
    test_s_combined()
    test_smooth_term_validation()
    test_smooth_term_repr()
    test_smooth_term_to_dict()
    test_s_examples()
    test_s_integration_ready()
    print("✅ All tests passed!")
