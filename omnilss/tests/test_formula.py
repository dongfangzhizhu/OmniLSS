"""Tests for formula processing with smooth terms."""

import unittest

import numpy as np
import pytest

from omnilss.fitting import _build_design_matrix
from omnilss._fitting_utils import FormulaEvaluationError
from omnilss.formula_parser import (
    LinearTerm,
    ParsedFormula,
    SmoothTerm,
    build_design_matrix,
    parse_formula,
)


class TestFormulaParser(unittest.TestCase):
    def test_parse_simple_linear(self):
        """Test parsing simple linear formula."""
        parsed = parse_formula("y ~ x1 + x2")

        self.assertEqual(parsed.response, "y")
        self.assertEqual(len(parsed.linear_terms), 2)
        self.assertEqual(len(parsed.smooth_terms), 0)
        self.assertTrue(parsed.has_intercept)
        self.assertEqual(parsed.linear_terms[0].variable, "x1")
        self.assertEqual(parsed.linear_terms[1].variable, "x2")

    def test_parse_no_intercept(self):
        """Test parsing formula without intercept."""
        parsed = parse_formula("y ~ x1 + x2 - 1")

        self.assertFalse(parsed.has_intercept)
        self.assertEqual(len(parsed.linear_terms), 2)

    def test_parse_smooth_only(self):
        """Test parsing formula with smooth term only."""
        parsed = parse_formula("y ~ pb(x)")

        self.assertEqual(parsed.response, "y")
        self.assertEqual(len(parsed.linear_terms), 0)
        self.assertEqual(len(parsed.smooth_terms), 1)

        smooth = parsed.smooth_terms[0]
        self.assertEqual(smooth.smoother, "pb")
        self.assertEqual(smooth.variable, "x")
        self.assertIsNone(smooth.df)
        self.assertIsNone(smooth.lambda_)

    def test_parse_smooth_with_df(self):
        """Test parsing smooth term with df argument."""
        parsed = parse_formula("y ~ pb(x, df=5)")

        smooth = parsed.smooth_terms[0]
        self.assertEqual(smooth.df, 5.0)

    def test_parse_smooth_with_lambda(self):
        """Test parsing smooth term with lambda argument."""
        parsed = parse_formula("y ~ pb(x, lambda=1.5)")

        smooth = parsed.smooth_terms[0]
        self.assertEqual(smooth.lambda_, 1.5)

    def test_parse_smooth_with_kwargs(self):
        """Test parsing smooth term with additional arguments."""
        parsed = parse_formula("y ~ pb(x, df=5, degree=3, order=2)")

        smooth = parsed.smooth_terms[0]
        self.assertEqual(smooth.df, 5.0)
        self.assertEqual(smooth.kwargs["degree"], 3.0)
        self.assertEqual(smooth.kwargs["order"], 2.0)

    def test_parse_mixed_terms(self):
        """Test parsing formula with both linear and smooth terms."""
        parsed = parse_formula("y ~ x1 + pb(x2, df=5) + x3")

        self.assertEqual(len(parsed.linear_terms), 2)
        self.assertEqual(len(parsed.smooth_terms), 1)
        self.assertEqual(parsed.linear_terms[0].variable, "x1")
        self.assertEqual(parsed.linear_terms[1].variable, "x3")
        self.assertEqual(parsed.smooth_terms[0].variable, "x2")

    def test_parse_multiple_smooths(self):
        """Test parsing formula with multiple smooth terms."""
        parsed = parse_formula("y ~ pb(x1, df=5) + pb(x2, lambda=1.0)")

        self.assertEqual(len(parsed.smooth_terms), 2)
        self.assertEqual(parsed.smooth_terms[0].variable, "x1")
        self.assertEqual(parsed.smooth_terms[0].df, 5.0)
        self.assertEqual(parsed.smooth_terms[1].variable, "x2")
        self.assertEqual(parsed.smooth_terms[1].lambda_, 1.0)

    def test_parse_different_smoothers(self):
        """Test parsing different types of smoothers."""
        parsed = parse_formula("y ~ pb(x1) + ps(x2) + cs(x3)")

        self.assertEqual(len(parsed.smooth_terms), 3)
        self.assertEqual(parsed.smooth_terms[0].smoother, "pb")
        self.assertEqual(parsed.smooth_terms[1].smoother, "ps")
        self.assertEqual(parsed.smooth_terms[2].smoother, "cs")

    def test_parse_invalid_formula(self):
        """Test that invalid formulas raise errors."""
        # No tilde
        with self.assertRaises(ValueError):
            parse_formula("y x")

        # No response
        with self.assertRaises(ValueError):
            parse_formula("~ x")

        # Empty smooth term
        with self.assertRaises(ValueError):
            parse_formula("y ~ pb()")


class TestDesignMatrix(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n = 100
        self.data = {
            "y": np.random.normal(0, 1, self.n),
            "x1": np.random.normal(0, 1, self.n),
            "x2": np.random.normal(0, 1, self.n),
            "x3": np.linspace(0, 1, self.n),
        }

    def test_build_linear_only(self):
        """Test building design matrix with linear terms only."""
        parsed = parse_formula("y ~ x1 + x2")
        X, smooth_info = build_design_matrix(parsed, self.data)

        # Should have intercept + 2 linear terms
        self.assertEqual(X.shape, (self.n, 3))

        # Check intercept
        np.testing.assert_array_equal(X[:, 0], 1.0)

        # Check linear terms
        np.testing.assert_array_equal(X[:, 1], self.data["x1"])
        np.testing.assert_array_equal(X[:, 2], self.data["x2"])

        # No smooth info
        self.assertEqual(len(smooth_info), 0)

    def test_build_no_intercept(self):
        """Test building design matrix without intercept."""
        parsed = parse_formula("y ~ x1 + x2 - 1")
        X, smooth_info = build_design_matrix(parsed, self.data)

        # Should have only 2 linear terms
        self.assertEqual(X.shape, (self.n, 2))

    def test_build_smooth_only(self):
        """Test building design matrix with smooth term only."""
        parsed = parse_formula("y ~ pb(x3)")
        X, smooth_info = build_design_matrix(parsed, self.data)

        # Should have intercept + basis functions
        self.assertGreater(X.shape[1], 1)

        # Check intercept
        np.testing.assert_array_equal(X[:, 0], 1.0)

        # Should have smooth info
        self.assertEqual(len(smooth_info), 1)
        self.assertIn("smooth_0", smooth_info)

    def test_build_mixed_terms(self):
        """Test building design matrix with mixed terms."""
        parsed = parse_formula("y ~ x1 + pb(x3)")
        X, smooth_info = build_design_matrix(parsed, self.data)

        # Should have intercept + linear + basis functions
        self.assertGreater(X.shape[1], 2)

        # Check intercept
        np.testing.assert_array_equal(X[:, 0], 1.0)

        # Check linear term
        np.testing.assert_array_equal(X[:, 1], self.data["x1"])

        # Should have smooth info
        self.assertEqual(len(smooth_info), 1)

    def test_build_multiple_smooths(self):
        """Test building design matrix with multiple smooth terms."""
        parsed = parse_formula("y ~ pb(x1) + pb(x2)")
        X, smooth_info = build_design_matrix(parsed, self.data)

        # Should have intercept + 2 sets of basis functions
        self.assertGreater(X.shape[1], 1)

        # Should have 2 smooth infos
        self.assertEqual(len(smooth_info), 2)
        self.assertIn("smooth_0", smooth_info)
        self.assertIn("smooth_1", smooth_info)

    def test_missing_variable(self):
        """Test that missing variables raise error."""
        parsed = parse_formula("y ~ x_missing")

        with self.assertRaises(ValueError):
            build_design_matrix(parsed, self.data)

    def test_wrong_length_variable(self):
        """Test that wrong length variables raise error."""
        data_bad = self.data.copy()
        data_bad["x1"] = np.array([1, 2, 3])  # Wrong length

        parsed = parse_formula("y ~ x1")

        with self.assertRaises(ValueError):
            build_design_matrix(parsed, data_bad)

    def test_smooth_info_structure(self):
        """Test structure of smooth info."""
        parsed = parse_formula("y ~ pb(x3, df=5)")
        X, smooth_info = build_design_matrix(parsed, self.data)

        info = smooth_info["smooth_0"]

        # Check required fields
        self.assertIn("basis", info)
        self.assertIn("knots", info)
        self.assertIn("degree", info)
        self.assertIn("smoother", info)
        self.assertIn("variable", info)

        # Check values
        self.assertEqual(info["smoother"], "pb")
        self.assertEqual(info["variable"], "x3")
        self.assertEqual(info["df"], 5.0)


if __name__ == "__main__":
    unittest.main()


def test_parse_interaction_and_star_expansion():
    parsed = parse_formula("y ~ x1 * x2")
    names = [t.variable for t in parsed.linear_terms]
    assert names == ["x1", "x2", "x1:x2"]


def test_build_design_matrix_interaction_term():
    data = {
        "y": np.array([1.0, 2.0, 3.0]),
        "x1": np.array([1.0, 2.0, 3.0]),
        "x2": np.array([4.0, 5.0, 6.0]),
    }
    parsed = parse_formula("y ~ x1:x2")
    X, _ = build_design_matrix(parsed, data)
    assert X.shape == (3, 2)
    np.testing.assert_array_equal(X[:, 1], data["x1"] * data["x2"])


def test_build_design_matrix_factor_term():
    data = {
        "y": np.array([1.0, 2.0, 3.0, 4.0]),
        "grp": np.array(["a", "b", "a", "c"]),
    }
    parsed = parse_formula("y ~ factor(grp)")
    X, _ = build_design_matrix(parsed, data)
    assert X.shape == (4, 3)
    # treatment coding (baseline=a): columns are [b, c]
    np.testing.assert_array_equal(X[:, 1], np.array([0.0, 1.0, 0.0, 0.0]))
    np.testing.assert_array_equal(X[:, 2], np.array([0.0, 0.0, 0.0, 1.0]))


def test_formula_expression_uses_safe_ast_evaluator():
    data = {"y": np.array([1.0, 2.0, 3.0]), "x": np.array([1.0, 4.0, 9.0])}
    _, design, labels = _build_design_matrix("y ~ log(x) + sqrt(x) + I(x**2)", data)

    assert labels == ["log(x)", "sqrt(x)", "I(x**2)"]
    np.testing.assert_allclose(design[:, 1], np.log(data["x"]))
    np.testing.assert_allclose(design[:, 2], np.sqrt(data["x"]))
    np.testing.assert_allclose(design[:, 3], data["x"] ** 2)


def test_formula_expression_rejects_attribute_access():
    data = {"y": np.array([1.0, 2.0]), "x": np.array([1.0, 2.0])}

    with pytest.raises(ValueError, match="Unable to evaluate term"):
        _build_design_matrix("y ~ x.__class__", data)


def test_formula_expression_rejects_np_attribute_calls():
    data = {"y": np.array([1.0, 2.0]), "x": np.array([1.0, 2.0])}

    with pytest.raises(ValueError, match="unsupported expression node 'Attribute'"):
        _build_design_matrix("y ~ np.sqrt(x)", data)


def test_formula_expression_rejects_subscripts_and_lambdas():
    data = {"y": np.array([1.0, 2.0]), "x": np.array([1.0, 2.0])}

    with pytest.raises(ValueError, match="unsupported expression node 'Subscript'"):
        _build_design_matrix("y ~ x[0]", data)
    with pytest.raises(ValueError, match="unsupported expression node 'Lambda'"):
        _build_design_matrix("y ~ (lambda z: z)(x)", data)


def test_formula_expression_rejects_comprehensions():
    data = {"y": np.array([1.0, 2.0]), "x": np.array([1.0, 2.0])}

    with pytest.raises(ValueError, match="unsupported expression node 'ListComp'"):
        _build_design_matrix("y ~ [v for v in x]", data)


def test_formula_expression_rejects_non_allowlisted_calls():
    data = {"y": np.array([1.0, 2.0]), "x": np.array([1.0, 2.0])}

    with pytest.raises(ValueError, match="function '__import__' is not allowed"):
        _build_design_matrix("y ~ __import__('os')", data)


def test_formula_evaluation_error_exposes_term_and_reason():
    data = {"y": np.array([1.0, 2.0]), "x": np.array([1.0, 2.0])}

    with pytest.raises(FormulaEvaluationError) as excinfo:
        _build_design_matrix("y ~ x[0]", data)

    assert excinfo.value.term == "x[0]"
    assert excinfo.value.reason == "unsupported expression node 'Subscript'"
