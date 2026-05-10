"""Tests for automatic lambda selection in pb() smoother."""

import numpy as np

from omnilss.smoothers.pb import fit_pspline


def test_pb_auto_gcv():
    """Test pb() with automatic GCV selection."""
    np.random.seed(42)
    x = np.linspace(0, 1, 100)
    y = np.sin(2 * np.pi * x) + np.random.randn(100) * 0.1
    
    result = fit_pspline(x, y, method="auto")
    
    assert result.lambda_ > 0
    assert 2 < result.edf < 20
    assert result.selection_method == "GCV"
    assert result.criterion_value is not None
    assert result.fitted_values.shape == y.shape


def test_pb_reml():
    """Test pb() with REML selection."""
    np.random.seed(42)
    x = np.linspace(0, 1, 100)
    y = np.sin(2 * np.pi * x) + np.random.randn(100) * 0.1
    
    result = fit_pspline(x, y, method="REML")
    
    assert result.lambda_ > 0
    assert 2 < result.edf < 20
    assert result.selection_method == "REML"
    assert result.criterion_value is not None


def test_pb_aic():
    """Test pb() with AIC selection."""
    np.random.seed(42)
    x = np.linspace(0, 1, 100)
    y = np.sin(2 * np.pi * x) + np.random.randn(100) * 0.1
    
    result = fit_pspline(x, y, method="AIC")
    
    assert result.lambda_ > 0
    assert 2 < result.edf < 20
    assert result.selection_method == "AIC"
    assert result.criterion_value is not None


def test_pb_fixed_lambda():
    """Test pb() with fixed lambda (backward compatibility)."""
    np.random.seed(42)
    x = np.linspace(0, 1, 100)
    y = np.sin(2 * np.pi * x) + np.random.randn(100) * 0.1
    
    result = fit_pspline(x, y, lambda_=0.01)
    
    assert result.lambda_ == 0.01
    assert result.selection_method == "fixed_lambda"


def test_pb_fixed_df():
    """Test pb() with fixed df (backward compatibility)."""
    np.random.seed(42)
    x = np.linspace(0, 1, 100)
    y = np.sin(2 * np.pi * x) + np.random.randn(100) * 0.1
    
    result = fit_pspline(x, y, df=5)
    
    assert abs(result.edf - 5.0) < 0.1  # Should be close to 5
    assert result.selection_method == "fixed_df"


def test_pb_legacy_gcv():
    """Test pb() with legacy GCV method (backward compatibility)."""
    np.random.seed(42)
    x = np.linspace(0, 1, 100)
    y = np.sin(2 * np.pi * x) + np.random.randn(100) * 0.1
    
    result = fit_pspline(x, y, method="GCV")
    
    assert result.lambda_ > 0
    assert 2 < result.edf < 20
    # Legacy GCV uses old implementation
    assert result.selection_method == "GCV (legacy)"


def test_pb_with_weights():
    """Test pb() with weights and automatic selection."""
    np.random.seed(42)
    x = np.linspace(0, 1, 100)
    y = np.sin(2 * np.pi * x) + np.random.randn(100) * 0.1
    weights = np.random.uniform(0.5, 1.5, 100)
    
    result = fit_pspline(x, y, weights=weights, method="auto")
    
    assert result.lambda_ > 0
    assert 2 < result.edf < 20
    assert result.selection_method == "GCV"


def test_pb_compare_methods():
    """Compare different automatic selection methods."""
    np.random.seed(42)
    x = np.linspace(0, 1, 100)
    y = np.sin(2 * np.pi * x) + np.random.randn(100) * 0.1
    
    result_gcv = fit_pspline(x, y, method="auto")
    result_reml = fit_pspline(x, y, method="REML")
    result_aic = fit_pspline(x, y, method="AIC")
    
    # All should produce reasonable results
    for result in [result_gcv, result_reml, result_aic]:
        assert result.lambda_ > 0
        assert 2 < result.edf < 20
        assert result.rss > 0
    
    # Results should be similar but not identical
    # (different methods may select different lambdas)
    assert result_gcv.lambda_ > 0
    assert result_reml.lambda_ > 0
    assert result_aic.lambda_ > 0


def test_pb_prediction():
    """Test prediction with automatically selected lambda."""
    np.random.seed(42)
    x = np.linspace(0, 1, 100)
    y = np.sin(2 * np.pi * x) + np.random.randn(100) * 0.1
    
    result = fit_pspline(x, y, method="auto")
    
    # Predict at new points
    x_new = np.linspace(0, 1, 50)
    y_pred = result.predict(x_new)
    
    assert y_pred.shape == (50,)
    assert np.all(np.isfinite(y_pred))


if __name__ == "__main__":
    # Run tests
    test_pb_auto_gcv()
    test_pb_reml()
    test_pb_aic()
    test_pb_fixed_lambda()
    test_pb_fixed_df()
    test_pb_legacy_gcv()
    test_pb_with_weights()
    test_pb_compare_methods()
    test_pb_prediction()
    print("✅ All tests passed!")
