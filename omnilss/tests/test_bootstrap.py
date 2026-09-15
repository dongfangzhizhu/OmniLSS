"""Tests for bootstrap methods."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omnilss.bootstrap import (
    generate_bootstrap_indices,
    resample_data,
    compute_confidence_intervals,
    nonparametric_bootstrap,
    parametric_bootstrap,
    residual_bootstrap,
    BootstrapResult,
)


def test_generate_bootstrap_indices():
    """Test bootstrap index generation."""
    key = jax.random.PRNGKey(42)
    n_obs = 100
    n_boots = 50
    
    indices = generate_bootstrap_indices(key, n_obs, n_boots)
    
    # Check shape
    assert indices.shape == (n_boots, n_obs)
    
    # Check all indices are valid
    assert jnp.all(indices >= 0)
    assert jnp.all(indices < n_obs)
    
    # Check that indices are integers
    assert indices.dtype in [jnp.int32, jnp.int64]


def test_resample_data():
    """Test data resampling."""
    data = {
        'y': jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        'x': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        'constant': 42  # Non-array value should be preserved
    }
    
    indices = jnp.array([0, 2, 2, 4, 1])
    
    resampled = resample_data(data, indices)
    
    # Check resampled values
    assert jnp.allclose(resampled['y'], jnp.array([1.0, 3.0, 3.0, 5.0, 2.0]))
    assert jnp.allclose(resampled['x'], jnp.array([10.0, 30.0, 30.0, 50.0, 20.0]))
    assert resampled['constant'] == 42


def test_compute_confidence_intervals():
    """Test confidence interval computation."""
    np.random.seed(42)
    
    # Generate bootstrap estimates (normal distribution)
    boot_estimates = jnp.array(np.random.normal(10, 2, size=(1000, 3)))
    
    lower_ci, upper_ci = compute_confidence_intervals(
        boot_estimates, alpha=0.05, method='percentile'
    )
    
    # Check shape
    assert lower_ci.shape == (3,)
    assert upper_ci.shape == (3,)
    
    # Check that lower < upper
    assert jnp.all(lower_ci < upper_ci)
    
    # Check approximate values (should be around 10 ± 1.96*2)
    assert jnp.all(jnp.abs(lower_ci - 6.0) < 1.0)
    assert jnp.all(jnp.abs(upper_ci - 14.0) < 1.0)


def test_compute_confidence_intervals_with_nan():
    """Test CI computation with NaN values."""
    boot_estimates = jnp.array([
        [1.0, 2.0, 3.0],
        [jnp.nan, jnp.nan, jnp.nan],  # Failed fit
        [1.5, 2.5, 3.5],
        [0.8, 1.8, 2.8],
    ])
    
    lower_ci, upper_ci = compute_confidence_intervals(
        boot_estimates, alpha=0.05, method='percentile'
    )
    
    # Should compute CI from valid samples only
    assert jnp.all(jnp.isfinite(lower_ci))
    assert jnp.all(jnp.isfinite(upper_ci))
    assert jnp.all(lower_ci < upper_ci)


def test_compute_confidence_intervals_bca_with_jackknife():
    """BCa intervals support the acceleration calculated from jackknife fits."""
    boot = jnp.array([[0.8], [0.9], [1.0], [1.1], [1.4], [1.8]])
    jack = jnp.array([[0.7], [0.9], [1.0], [1.1], [1.2]])
    lower, upper = compute_confidence_intervals(
        boot, alpha=0.2, method="bca", original_estimate=jnp.array([1.0]),
        jackknife_estimates=jack,
    )
    assert lower.shape == (1,)
    assert upper.shape == (1,)
    assert jnp.all(jnp.isfinite(jnp.array([lower, upper])))
    assert lower[0] <= upper[0]


def test_compute_confidence_intervals_bca_without_jackknife():
    """BCa remains usable with the documented zero-acceleration fallback."""
    boot = jnp.arange(20.0).reshape(10, 2)
    lower, upper = compute_confidence_intervals(
        boot, method="bca", original_estimate=jnp.array([9.0, 10.0])
    )
    assert lower.shape == (2,)
    assert upper.shape == (2,)
    assert jnp.all(lower <= upper)


def test_bca_matches_scipy_reference():
    from scipy.stats import bootstrap

    data = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 9.0, 18.0])
    reference = bootstrap(
        (data,), np.mean, confidence_level=0.9, n_resamples=999,
        random_state=42, method="BCa",
    )
    jack = np.array([np.mean(np.delete(data, i)) for i in range(data.size)])
    actual = compute_confidence_intervals(
        reference.bootstrap_distribution[:, None], alpha=0.1, method="bca",
        original_estimate=np.array([data.mean()]), jackknife_estimates=jack[:, None],
    )
    np.testing.assert_allclose(
        np.asarray(actual)[:, 0], reference.confidence_interval, rtol=1e-6,
    )


def test_bca_ties_use_midrank_bias_correction():
    boot = np.array([[0.0], [1.0], [1.0], [1.0], [2.0]])
    actual = compute_confidence_intervals(
        boot, alpha=0.2, method="bca", original_estimate=np.array([1.0]),
    )
    np.testing.assert_allclose(np.asarray(actual)[:, 0], np.quantile(boot, [0.1, 0.9]))


@pytest.mark.parametrize("method", ["percentile", "bca"])
def test_confidence_intervals_nonfinite_and_empty(method):
    kwargs = dict(method=method, original_estimate=np.array([1.0, 2.0]))
    good = np.array([[0.0, 1.0], [2.0, 3.0]])
    contaminated = np.vstack([good, [np.inf, 0.0], [0.0, np.nan], [-np.inf, 0.0]])
    np.testing.assert_allclose(
        compute_confidence_intervals(contaminated, **kwargs),
        compute_confidence_intervals(good, **kwargs),
    )
    for empty in (np.empty((0, 2)), np.full((3, 2), np.nan)):
        assert np.isnan(compute_confidence_intervals(empty, **kwargs)).all()


@pytest.mark.parametrize("jack", [np.empty((0, 1)), [[1.0]], [[1.0], [np.nan]], [[1.0], [np.inf]], [[1.0, 2.0], [2.0, 3.0]]])
def test_bca_rejects_invalid_jackknife(jack):
    with pytest.raises(ValueError, match="jackknife_estimates"):
        compute_confidence_intervals(
            np.ones((5, 1)), method="bca", original_estimate=np.array([1.0]),
            jackknife_estimates=jack,
        )


@pytest.mark.parametrize("original", [None, [np.nan], [np.inf], [1.0, 2.0]])
def test_bca_requires_valid_original_statistic(original):
    with pytest.raises(ValueError, match="original_estimate"):
        compute_confidence_intervals(np.ones((5, 1)), method="bca", original_estimate=original)


def test_bca_constant_statistics():
    actual = compute_confidence_intervals(
        np.ones((5, 2)), method="bca", original_estimate=np.ones(2),
        jackknife_estimates=np.ones((4, 2)),
    )
    np.testing.assert_array_equal(actual, np.ones((2, 2)))


def test_nonparametric_bootstrap_basic():
    """Test basic non-parametric bootstrap."""
    np.random.seed(42)
    
    # Simple linear model: y = 2*x + 1 + noise
    n = 100
    x = np.linspace(0, 10, n)
    y = 2 * x + 1 + np.random.normal(0, 0.5, n)
    
    data = {
        'y': jnp.array(y),
        'x': jnp.array(x)
    }
    
    # Simple fit function (OLS)
    def fit_function(d):
        X = jnp.column_stack([jnp.ones(len(d['x'])), d['x']])
        y = d['y']
        # Solve X'X beta = X'y
        XtX = X.T @ X
        Xty = X.T @ y
        beta = jnp.linalg.solve(XtX, Xty)
        return beta
    
    key = jax.random.PRNGKey(42)
    
    # Run bootstrap with small number for speed
    result = nonparametric_bootstrap(
        fit_function, data, key, n_boots=100, alpha=0.05, parallel=True
    )
    
    # Check results
    assert result.n_boots == 100
    assert result.n_successful > 90  # Most should succeed
    assert result.coefficients.shape == (100, 2)
    
    # Check that estimates are reasonable
    assert jnp.abs(result.mean[0] - 1.0) < 0.5  # Intercept ~ 1
    assert jnp.abs(result.mean[1] - 2.0) < 0.5  # Slope ~ 2
    
    # Check CI
    assert result.lower_ci is not None
    assert result.upper_ci is not None
    assert jnp.all(result.lower_ci < result.upper_ci)


def test_nonparametric_bootstrap_sequential():
    """Test non-parametric bootstrap in sequential mode."""
    np.random.seed(42)
    
    n = 50
    x = np.linspace(0, 5, n)
    y = x + np.random.normal(0, 0.3, n)
    
    data = {
        'y': jnp.array(y),
        'x': jnp.array(x)
    }
    
    def fit_function(d):
        X = jnp.column_stack([jnp.ones(len(d['x'])), d['x']])
        y = d['y']
        XtX = X.T @ X
        Xty = X.T @ y
        beta = jnp.linalg.solve(XtX, Xty)
        return beta
    
    key = jax.random.PRNGKey(42)
    
    # Run in sequential mode
    result = nonparametric_bootstrap(
        fit_function, data, key, n_boots=50, parallel=False
    )
    
    assert result.n_boots == 50
    assert result.n_successful > 45
    assert result.coefficients.shape == (50, 2)


def test_parametric_bootstrap():
    """Test parametric bootstrap."""
    np.random.seed(42)
    
    # Generate data from known distribution
    n = 100
    true_mu = 5.0
    true_sigma = 2.0
    
    key = jax.random.PRNGKey(42)
    y = jax.random.normal(key, shape=(n,)) * true_sigma + true_mu
    
    data = {
        'y': y,
        'n': n
    }
    
    # Fit function (estimate mean and std)
    def fit_function(d):
        return jnp.array([jnp.mean(d['y']), jnp.std(d['y'])])
    
    # Sample function (generate from normal)
    def sample_function(params, k):
        mu, sigma = params
        return jax.random.normal(k, shape=(n,)) * sigma + mu
    
    # Get initial fit
    fitted_params = fit_function(data)
    
    # Run parametric bootstrap
    result = parametric_bootstrap(
        fit_function, sample_function, data, fitted_params,
        key, n_boots=100, alpha=0.05, parallel=True
    )
    
    # Check results
    assert result.n_boots == 100
    assert result.n_successful == 100  # Should all succeed
    assert result.coefficients.shape == (100, 2)
    assert result.method == 'parametric'
    
    # Check estimates are reasonable
    assert jnp.abs(result.mean[0] - true_mu) < 1.0
    assert jnp.abs(result.mean[1] - true_sigma) < 1.0


def test_residual_bootstrap():
    """Test residual bootstrap."""
    np.random.seed(42)
    
    # Linear model with heteroscedastic errors
    n = 100
    x = np.linspace(0, 10, n)
    y = 2 * x + 1 + np.random.normal(0, 0.5, n)
    
    data = {
        'y': jnp.array(y),
        'x': jnp.array(x)
    }
    
    # Fit function
    def fit_function(d):
        X = jnp.column_stack([jnp.ones(len(d['x'])), d['x']])
        y = d['y']
        XtX = X.T @ X
        Xty = X.T @ y
        beta = jnp.linalg.solve(XtX, Xty)
        return beta
    
    # Get initial fit
    fitted_params = fit_function(data)
    X = jnp.column_stack([jnp.ones(n), data['x']])
    fitted_values = X @ fitted_params
    residuals = data['y'] - fitted_values
    
    key = jax.random.PRNGKey(42)
    
    # Run residual bootstrap
    result = residual_bootstrap(
        fit_function, data, fitted_values, residuals,
        key, n_boots=100, alpha=0.05, parallel=True
    )
    
    # Check results
    assert result.n_boots == 100
    assert result.n_successful > 90
    assert result.coefficients.shape == (100, 2)
    assert result.method == 'residual'
    
    # Check estimates
    assert jnp.abs(result.mean[0] - 1.0) < 0.5
    assert jnp.abs(result.mean[1] - 2.0) < 0.5


def test_bootstrap_result_summary():
    """Test BootstrapResult summary method."""
    result = BootstrapResult(
        coefficients=jnp.array([[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]]),
        mean=jnp.array([1.0, 2.0]),
        std=jnp.array([0.1, 0.1]),
        lower_ci=jnp.array([0.8, 1.8]),
        upper_ci=jnp.array([1.2, 2.2]),
        n_boots=3,
        n_successful=3,
        alpha=0.05,
        method='nonparametric'
    )
    
    summary = result.summary()
    
    # Check that summary contains key information
    assert 'Bootstrap Results' in summary
    assert 'nonparametric' in summary
    assert 'Number of bootstrap samples: 3' in summary
    assert 'Successful fits: 3' in summary
    assert '95%' in summary


def test_bootstrap_with_failures():
    """Test bootstrap handling of failed fits."""
    np.random.seed(42)
    
    n = 50
    x = np.linspace(0, 5, n)
    y = x + np.random.normal(0, 0.3, n)
    
    data = {
        'y': jnp.array(y),
        'x': jnp.array(x)
    }
    
    # Fit function that sometimes fails
    call_count = [0]
    def fit_function_with_failures(d):
        call_count[0] += 1
        # Fail every 5th call
        if call_count[0] % 5 == 0:
            raise ValueError("Simulated failure")
        
        X = jnp.column_stack([jnp.ones(len(d['x'])), d['x']])
        y = d['y']
        XtX = X.T @ X
        Xty = X.T @ y
        beta = jnp.linalg.solve(XtX, Xty)
        return beta
    
    key = jax.random.PRNGKey(42)
    
    # This should handle failures gracefully
    result = nonparametric_bootstrap(
        fit_function_with_failures, data, key,
        n_boots=20, handle_failures=True, parallel=False
    )
    
    # Some fits should have failed
    assert result.n_successful < result.n_boots
    assert result.n_successful > 0
    
    # Results should still be valid
    assert jnp.all(jnp.isfinite(result.mean))
    assert jnp.all(jnp.isfinite(result.std))


def test_bootstrap_reproducibility():
    """Test that bootstrap results are reproducible with same seed."""
    np.random.seed(42)
    
    n = 50
    x = np.linspace(0, 5, n)
    y = x + np.random.normal(0, 0.3, n)
    
    data = {
        'y': jnp.array(y),
        'x': jnp.array(x)
    }
    
    def fit_function(d):
        X = jnp.column_stack([jnp.ones(len(d['x'])), d['x']])
        y = d['y']
        XtX = X.T @ X
        Xty = X.T @ y
        beta = jnp.linalg.solve(XtX, Xty)
        return beta
    
    # Run twice with same seed
    key1 = jax.random.PRNGKey(42)
    result1 = nonparametric_bootstrap(
        fit_function, data, key1, n_boots=50, parallel=True
    )
    
    key2 = jax.random.PRNGKey(42)
    result2 = nonparametric_bootstrap(
        fit_function, data, key2, n_boots=50, parallel=True
    )
    
    # Results should be identical
    assert jnp.allclose(result1.coefficients, result2.coefficients)
    assert jnp.allclose(result1.mean, result2.mean)
    assert jnp.allclose(result1.std, result2.std)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
