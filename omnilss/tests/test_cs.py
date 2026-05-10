"""Tests for cs() cubic spline smoother.

R source: gamlss/R/cubicSplines-10-08-12.R

运行:
    JAX_PLATFORMS=cuda python -m unittest tests.test_cs -v   # GPU
    JAX_PLATFORMS=cpu  python -m unittest tests.test_cs -v   # CPU
"""

import unittest
import numpy as np

from omnilss.smoothers.cs import fit_cubic_spline, CSResult


class TestCSKnots(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.n = 100
        self.x = np.linspace(0, 1, self.n)
        self.y = np.sin(2 * np.pi * self.x) + np.random.normal(0, 0.2, self.n)

    def test_basic_fit(self):
        r = fit_cubic_spline(self.x, self.y)
        self.assertIsInstance(r, CSResult)
        self.assertEqual(len(r.fitted_values), self.n)
        self.assertTrue(np.all(np.isfinite(r.fitted_values)))

    def test_fit_with_df(self):
        r = fit_cubic_spline(self.x, self.y, df=5)
        self.assertGreater(r.r_squared, 0.5)
        self.assertTrue(np.all(np.isfinite(r.fitted_values)))

    def test_fit_with_weights(self):
        w = np.random.uniform(0.5, 1.5, self.n)
        r = fit_cubic_spline(self.x, self.y, weights=w, df=5)
        self.assertEqual(len(r.fitted_values), self.n)
        self.assertGreater(r.r_squared, 0.0)

    def test_gcv_default(self):
        """No df/spar → GCV selection."""
        r = fit_cubic_spline(self.x, self.y)
        self.assertGreater(r.r_squared, 0.5)
        self.assertGreater(r.edf, 1.0)

    def test_prediction(self):
        r = fit_cubic_spline(self.x, self.y, df=5)
        x_new = np.array([0.1, 0.5, 0.9])
        pred = r.predict(x_new)
        self.assertEqual(len(pred), 3)
        self.assertTrue(np.all(np.isfinite(pred)))

    def test_smoothing_effect(self):
        """Higher df → rougher fit."""
        r_smooth = fit_cubic_spline(self.x, self.y, df=3)
        r_rough = fit_cubic_spline(self.x, self.y, df=8)
        var_smooth = np.var(np.diff(r_smooth.fitted_values))
        var_rough = np.var(np.diff(r_rough.fitted_values))
        self.assertLess(var_smooth, var_rough)

    def test_r_squared_positive(self):
        r = fit_cubic_spline(self.x, self.y, df=5)
        self.assertGreater(r.r_squared, 0.0)
        self.assertLessEqual(r.r_squared, 1.0)

    def test_residuals(self):
        r = fit_cubic_spline(self.x, self.y, df=5)
        self.assertIsNotNone(r.residuals)
        np.testing.assert_allclose(
            r.residuals, self.y - r.fitted_values, rtol=1e-10
        )

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            fit_cubic_spline(self.x, self.y[:-1], df=5)

    def test_unsorted_x(self):
        """cs() should handle unsorted x."""
        idx = np.random.permutation(self.n)
        r = fit_cubic_spline(self.x[idx], self.y[idx], df=5)
        self.assertEqual(len(r.fitted_values), self.n)
        self.assertTrue(np.all(np.isfinite(r.fitted_values)))


if __name__ == "__main__":
    unittest.main()
