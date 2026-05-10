"""Tests for P-spline smooth (ps) smoother."""

import unittest

import numpy as np

from omnilss.smoothers.ps import fit_pspline_smooth, _create_ps_knots


class TestPSKnots(unittest.TestCase):
    """Test knot creation for ps()."""
    
    def test_knot_creation_basic(self):
        """Test basic knot creation."""
        x = np.linspace(0, 1, 100)
        knots = _create_ps_knots(x, ps_intervals=20, degree=3)
        
        # Should have ps_intervals + 2*degree + 1 knots
        expected_n_knots = 20 + 2*3 + 1
        self.assertEqual(len(knots), expected_n_knots)
        
    def test_knot_range_extension(self):
        """Test that knots extend beyond data range."""
        x = np.linspace(0, 1, 100)
        knots = _create_ps_knots(x, ps_intervals=20, degree=3)
        
        # Knots should extend beyond [0, 1]
        self.assertLess(knots[0], 0)
        self.assertGreater(knots[-1], 1)
        
    def test_knot_spacing(self):
        """Test that knots are evenly spaced."""
        x = np.linspace(0, 1, 100)
        knots = _create_ps_knots(x, ps_intervals=20, degree=3)
        
        # Check spacing
        diffs = np.diff(knots)
        self.assertTrue(np.allclose(diffs, diffs[0], rtol=1e-10))


class TestPSFitting(unittest.TestCase):
    """Test P-spline smooth fitting."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n = 100
        self.x = np.linspace(0, 1, self.n)
        self.y_true = 2 + 3 * np.sin(2 * np.pi * self.x)
        self.y = self.y_true + np.random.normal(0, 0.5, self.n)
        
    def test_fit_with_fixed_df(self):
        """Test fitting with fixed df."""
        result = fit_pspline_smooth(self.x, self.y, df=5)
        
        # Check result attributes
        self.assertIsNotNone(result.coefficients)
        self.assertIsNotNone(result.fitted_values)
        self.assertGreater(result.lambda_, 0)
        self.assertGreater(result.edf, 0)
        self.assertLess(result.edf, 20)  # Should be less than n_basis
        
        # Check fit quality
        self.assertEqual(len(result.fitted_values), self.n)
        self.assertGreater(result.r_squared, 0.5)
        
    def test_fit_with_fixed_lambda(self):
        """Test fitting with fixed lambda."""
        result = fit_pspline_smooth(self.x, self.y, lambda_=1.0)
        
        # Check that lambda is used
        self.assertAlmostEqual(result.lambda_, 1.0)
        
        # Check fit
        self.assertEqual(len(result.fitted_values), self.n)
        
    def test_fit_ml_method(self):
        """Test fitting with ML method."""
        result = fit_pspline_smooth(self.x, self.y, method="ML")
        
        # Check that lambda was estimated
        self.assertGreater(result.lambda_, 0)
        self.assertGreater(result.edf, 0)
        
    def test_fit_gaic_method(self):
        """Test fitting with GAIC method."""
        result = fit_pspline_smooth(self.x, self.y, method="GAIC")
        
        # Check that lambda was estimated
        self.assertGreater(result.lambda_, 0)
        self.assertGreater(result.edf, 0)
        
    def test_fit_gcv_method(self):
        """Test fitting with GCV method."""
        result = fit_pspline_smooth(self.x, self.y, method="GCV")
        
        # Check that lambda was estimated
        self.assertGreater(result.lambda_, 0)
        self.assertGreater(result.edf, 0)
        
    def test_fit_with_weights(self):
        """Test fitting with observation weights."""
        weights = np.random.uniform(0.5, 1.5, self.n)
        result = fit_pspline_smooth(self.x, self.y, weights=weights, df=5)
        
        # Check fit
        self.assertEqual(len(result.fitted_values), self.n)
        self.assertGreater(result.r_squared, 0)
        
    def test_different_ps_intervals(self):
        """Test different ps_intervals values."""
        for ps_intervals in [10, 20, 30]:
            result = fit_pspline_smooth(
                self.x, self.y, df=5, ps_intervals=ps_intervals
            )
            self.assertEqual(result.ps_intervals, ps_intervals)
            self.assertGreater(result.r_squared, 0.5)
            
    def test_different_degrees(self):
        """Test different B-spline degrees."""
        for degree in [1, 2, 3]:
            result = fit_pspline_smooth(
                self.x, self.y, df=5, degree=degree
            )
            self.assertEqual(result.degree, degree)
            self.assertGreater(result.r_squared, 0.5)
            
    def test_different_orders(self):
        """Test different penalty orders."""
        for order in [1, 2, 3]:
            result = fit_pspline_smooth(
                self.x, self.y, df=5, order=order
            )
            self.assertEqual(result.order, order)
            self.assertGreater(result.r_squared, 0.5)
            
    def test_prediction(self):
        """Test prediction at new x values."""
        result = fit_pspline_smooth(self.x, self.y, df=5)
        
        # Predict at new points
        x_new = np.linspace(0, 1, 50)
        y_pred = result.predict(x_new)
        
        # Check shape
        self.assertEqual(len(y_pred), 50)
        
        # Predictions should be reasonable
        self.assertTrue(np.all(np.isfinite(y_pred)))
        
    def test_smoothing_effect(self):
        """Test that smoothing reduces variance."""
        # Fit with different df values
        result_smooth = fit_pspline_smooth(self.x, self.y, df=3)
        result_rough = fit_pspline_smooth(self.x, self.y, df=10)
        
        # Smoother fit should have lower variance
        var_smooth = np.var(np.diff(result_smooth.fitted_values))
        var_rough = np.var(np.diff(result_rough.fitted_values))
        
        self.assertLess(var_smooth, var_rough)
        
    def test_error_handling(self):
        """Test error handling."""
        # Neither df nor lambda specified
        with self.assertRaises(ValueError):
            fit_pspline_smooth(self.x, self.y)
        
        # Mismatched lengths
        with self.assertRaises(ValueError):
            fit_pspline_smooth(self.x, self.y[:-1], df=5)
            
    def test_negative_lambda_warning(self):
        """Test warning for negative lambda."""
        with self.assertWarns(UserWarning):
            result = fit_pspline_smooth(self.x, self.y, lambda_=-1.0)
        
        # Should use lambda=0
        self.assertEqual(result.lambda_, 0.0)


class TestPSComparison(unittest.TestCase):
    """Test ps() comparison with pb()."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n = 100
        self.x = np.linspace(0, 1, self.n)
        self.y = 2 + 3 * np.sin(2 * np.pi * self.x) + np.random.normal(0, 0.5, self.n)
        
    def test_ps_vs_pb_similar_results(self):
        """Test that ps() and pb() give similar results."""
        from omnilss.smoothers.pb import fit_pspline
        
        # Fit with both methods
        result_ps = fit_pspline_smooth(self.x, self.y, df=5)
        result_pb = fit_pspline(self.x, self.y, df=5)
        
        # Results should be similar (correlation > 0.95)
        corr = np.corrcoef(result_ps.fitted_values, result_pb.fitted_values)[0, 1]
        self.assertGreater(corr, 0.95)
        
        # R-squared should be similar
        self.assertAlmostEqual(result_ps.r_squared, result_pb.r_squared, delta=0.1)


if __name__ == "__main__":
    unittest.main()
