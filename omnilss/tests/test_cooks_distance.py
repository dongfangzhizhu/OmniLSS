"""Tests for Cook's distance diagnostic."""

import numpy as np
import pytest
from omnilss import gamlss, NO
from omnilss.diagnostics import cooks_distance


def test_cooks_distance_basic():
    """Test basic Cook's distance calculation."""
    # Generate data with one outlier
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 10, n)
    y = 5 + 0.5 * x + np.random.normal(0, 1, n)
    y[50] = 50  # Add outlier
    
    data = {'x': x, 'y': y}
    model = gamlss(formula="y ~ x", family=NO(), data=data)
    
    # Calculate Cook's distance
    result = cooks_distance(model)
    
    # Check result structure
    assert len(result.cooks_distance) == n
    assert result.threshold > 0
    assert len(result.influential) == n
    assert result.n_influential >= 0
    assert len(result.index) == n
    
    # Check that outlier has high Cook's distance
    assert result.cooks_distance[50] > result.threshold
    assert result.influential[50]
    
    # Check that most observations are not influential
    assert result.n_influential < n / 10  # Less than 10% influential


def test_cooks_distance_no_outliers():
    """Test Cook's distance with no outliers."""
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 10, n)
    y = 5 + 0.5 * x + np.random.normal(0, 0.5, n)  # Small noise
    
    data = {'x': x, 'y': y}
    model = gamlss(formula="y ~ x", family=NO(), data=data)
    
    result = cooks_distance(model)
    
    # With no outliers, few observations should be influential
    assert result.n_influential < n / 20  # Less than 5% influential


def test_cooks_distance_custom_threshold():
    """Test Cook's distance with custom threshold."""
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 10, n)
    y = 5 + 0.5 * x + np.random.normal(0, 1, n)
    
    data = {'x': x, 'y': y}
    model = gamlss(formula="y ~ x", family=NO(), data=data)
    
    # Use custom threshold
    custom_threshold = 0.1
    result = cooks_distance(model, threshold=custom_threshold)
    
    assert result.threshold == custom_threshold
    
    # More observations should be influential with higher threshold
    result_default = cooks_distance(model)
    if custom_threshold > result_default.threshold:
        assert result.n_influential >= result_default.n_influential


def test_cooks_distance_values():
    """Test that Cook's distance values are reasonable."""
    np.random.seed(42)
    n = 50
    x = np.linspace(0, 10, n)
    y = 5 + 0.5 * x + np.random.normal(0, 1, n)
    
    data = {'x': x, 'y': y}
    model = gamlss(formula="y ~ x", family=NO(), data=data)
    
    result = cooks_distance(model)
    
    # Cook's distance should be non-negative
    assert np.all(result.cooks_distance >= 0)
    
    # Cook's distance should be finite
    assert np.all(np.isfinite(result.cooks_distance))
    
    # Most Cook's distances should be small
    assert np.median(result.cooks_distance) < 0.1


def test_cooks_distance_multiple_outliers():
    """Test Cook's distance with multiple outliers."""
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 10, n)
    y = 5 + 0.5 * x + np.random.normal(0, 1, n)
    
    # Add multiple outliers
    outlier_indices = [10, 30, 50, 70, 90]
    for idx in outlier_indices:
        y[idx] = 50
    
    data = {'x': x, 'y': y}
    model = gamlss(formula="y ~ x", family=NO(), data=data)
    
    result = cooks_distance(model)
    
    # All outliers should be influential
    for idx in outlier_indices:
        assert result.influential[idx], f"Outlier at index {idx} not detected"
        assert result.cooks_distance[idx] > result.threshold


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
