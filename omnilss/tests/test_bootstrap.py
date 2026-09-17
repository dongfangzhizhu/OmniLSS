"""Tests for bootstrap methods."""

import jax
import jax.numpy as jnp
import json
import numpy as np
import pytest
import warnings
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


def test_resample_data_resamples_numpy_arrays():
    data = {
        'y': np.array([1.0, 2.0, 3.0]),
        'x': np.array([10.0, 20.0, 30.0]),
        'constant': 42,
    }

    resampled = resample_data(data, jnp.array([2, 0, 2]))

    np.testing.assert_array_equal(resampled['y'], [3.0, 1.0, 3.0])
    np.testing.assert_array_equal(resampled['x'], [30.0, 10.0, 30.0])
    assert resampled['constant'] == 42


def test_nonparametric_bootstrap_resamples_numpy_covariates():
    data = {
        'y': np.array([1.0, 2.0, 4.0, 8.0]),
        'x': np.array([10.0, 20.0, 40.0, 80.0]),
    }

    def fit(sample):
        assert np.array_equal(np.asarray(sample['y']) * 10, np.asarray(sample['x']))
        return jnp.array([jnp.mean(sample['y'])])

    result = nonparametric_bootstrap(
        fit, data, jax.random.PRNGKey(7), n_boots=8, parallel=False,
        failure_warning="never",
    )

    assert result.n_successful == 8


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
    assert result.fitted_values is None
    assert result.predictions is None
    assert result.failure_indices.shape == (0,)
    assert result.failure_reasons is None
    
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
    assert result.fitted_values is None
    assert result.predictions is None
    assert result.failure_indices.shape == (0,)
    assert result.failure_reasons is None

    # Check estimates
    assert jnp.abs(result.mean[0] - 1.0) < 0.5
    assert jnp.abs(result.mean[1] - 2.0) < 0.5


def test_parametric_and_residual_bootstrap_collect_failure_diagnostics():
    data = {'y': jnp.arange(5.0)}
    fitted_params = jnp.array([1.0])
    calls = [0]

    def failing_fit(d):
        calls[0] += 1
        if calls[0] in (2, 4):
            raise ValueError("simulated failure")
        return jnp.array([jnp.mean(d['y'])])

    def sample(_params, key):
        return jax.random.normal(key, shape=(5,))

    parametric = parametric_bootstrap(
        failing_fit, sample, data, fitted_params, jax.random.PRNGKey(0),
        n_boots=4, parallel=False,
    )
    assert parametric.n_successful == 2
    assert parametric.failure_reasons == {"ValueError": 2}
    np.testing.assert_array_equal(np.asarray(parametric.failure_indices), [1, 3])

    calls[0] = -1  # residual bootstrap performs one original fit for shape inference
    residual = residual_bootstrap(
        failing_fit, data, jnp.ones(5), jnp.zeros(5), jax.random.PRNGKey(0),
        n_boots=4, parallel=False,
    )
    assert residual.n_successful == 2
    assert residual.failure_reasons == {"ValueError": 2}
    np.testing.assert_array_equal(np.asarray(residual.failure_indices), [1, 3])
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
    assert 'Failed fits: 0' in summary
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
    assert result.n_failed == result.n_boots - result.n_successful
    assert result.failure_rate > 0
    np.testing.assert_array_equal(
        np.asarray(result.failure_indices),
        np.flatnonzero(np.isnan(np.asarray(result.coefficients)).any(axis=1)),
    )
    assert result.failure_reasons == {"ValueError": result.n_failed}


def test_bootstrap_min_success_rate():
    def fit(d):
        return jnp.array([jnp.mean(d['y'])])
    data = {'y': jnp.arange(10.0)}
    with pytest.raises(ValueError):
        nonparametric_bootstrap(fit, data, jax.random.PRNGKey(0), n_boots=4, min_success_rate=1.1)
    calls = [0]
    def intermittent_fit(d):
        calls[0] += 1
        if calls[0] % 2 == 0:
            raise ValueError("fail")
        return jnp.array([jnp.mean(d['y'])])
    with pytest.raises(RuntimeError, match="success rate"):
        nonparametric_bootstrap(
            intermittent_fit, data,
            jax.random.PRNGKey(0), n_boots=4, parallel=False,
            min_success_rate=0.75,
        )


def test_nonparametric_failure_does_not_retry_failed_original_fit():
    calls = [0]
    def fit(d):
        calls[0] += 1
        if calls[0] > 1:
            raise ValueError("bootstrap failure")
        return jnp.array([jnp.mean(d['y'])])
    result = nonparametric_bootstrap(
        fit, {'y': jnp.arange(5.0)}, jax.random.PRNGKey(0),
        n_boots=3, parallel=False, min_success_rate=0.0,
    )
    assert result.n_successful == 0
    assert jnp.isnan(result.coefficients).all()


def test_bootstrap_summary_handles_zero_boots():
    assert "0.0%" in BootstrapResult(coefficients=jnp.empty((0, 1))).summary()


def test_bootstrap_result_to_dict_exports_diagnostics():
    result = BootstrapResult(
        coefficients=jnp.array([[1.0, 2.0], [jnp.nan, jnp.inf]]),
        fitted_values=jnp.array([[3.0], [4.0]]),
        predictions=jnp.array([[5.0]]),
        mean=jnp.array([1.0, jnp.nan]),
        std=jnp.array([0.1, jnp.inf]),
        lower_ci=jnp.array([0.8, 1.8]),
        upper_ci=jnp.array([1.2, 2.2]),
        n_boots=2,
        n_successful=1,
        alpha=np.float64(0.05),
        method="residual",
        failure_indices=jnp.array([1]),
    )

    exported = result.to_dict()

    assert exported["method"] == "residual"
    assert exported["n_failed"] == 1
    assert exported["failure_rate"] == 0.5
    assert exported["coefficients"] == [[1.0, 2.0], [None, None]]
    assert exported["mean"] == [1.0, None]
    assert exported["std"] == [0.1, None]
    assert exported["failure_indices"] == [1]
    assert exported["failure_reasons"] == {}
    json.loads(json.dumps(exported))


def test_bootstrap_result_to_json_accepts_json_options():
    result = BootstrapResult(
        coefficients=jnp.array([[1.0]]),
        n_boots=1,
        n_successful=1,
    )

    exported = json.loads(result.to_json(indent=None, sort_keys=True))

    assert exported["coefficients"] == [[1.0]]
    assert exported["failure_reasons"] == {}


@pytest.mark.parametrize("n_boots", [0, -1, 1.5, True])
def test_bootstrap_rejects_invalid_boot_count(n_boots):
    fit = lambda d: jnp.array([jnp.mean(d['y'])])
    with pytest.raises(ValueError, match="n_boots"):
        nonparametric_bootstrap(fit, {'y': jnp.arange(3.0)}, jax.random.PRNGKey(0), n_boots=n_boots)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, np.nan])
def test_bootstrap_rejects_invalid_alpha(alpha):
    fit = lambda d: jnp.array([jnp.mean(d['y'])])
    with pytest.raises(ValueError, match="alpha"):
        nonparametric_bootstrap(fit, {'y': jnp.arange(3.0)}, jax.random.PRNGKey(0), alpha=alpha)


def test_bootstrap_rejects_inconsistent_data_and_residual_lengths():
    fit = lambda d: jnp.array([jnp.mean(d['y'])])
    with pytest.raises(ValueError, match=r"data\['x'\]"):
        nonparametric_bootstrap(
            fit, {'y': jnp.arange(3.0), 'x': jnp.arange(2.0)},
            jax.random.PRNGKey(0), n_boots=2,
        )
    with pytest.raises(ValueError, match="fitted_values and residuals"):
        residual_bootstrap(
            fit, {'y': jnp.arange(3.0)}, jnp.ones(2), jnp.ones(3),
            jax.random.PRNGKey(0), n_boots=2,
        )


def test_bootstrap_rejects_scalar_or_wrong_shape_fit_result():
    data = {'y': jnp.arange(3.0)}
    with pytest.raises(ValueError, match="one-dimensional"):
        nonparametric_bootstrap(lambda d: jnp.array(1.0), data, jax.random.PRNGKey(0), n_boots=2)
    with pytest.raises(ValueError, match="one-dimensional"):
        nonparametric_bootstrap(lambda d: jnp.ones((2, 1)), data, jax.random.PRNGKey(0), n_boots=2)


def test_bootstrap_rejects_inconsistent_parameter_shape():
    data = {'y': jnp.arange(3.0)}
    calls = [0]

    def fit(sample):
        if sample is data:
            return jnp.ones(2)
        calls[0] += 1
        return jnp.ones(1 if calls[0] == 1 else 2)

    with pytest.raises(ValueError, match=r"shape \(2,\).*iteration"):
        nonparametric_bootstrap(
            fit, data, jax.random.PRNGKey(0), n_boots=2, parallel=False,
        )


@pytest.mark.parametrize("method", ["parametric", "residual"])
def test_bootstrap_rejects_wrong_observation_shape(method):
    data = {'y': jnp.arange(3.0)}
    fit = lambda sample: jnp.array([jnp.mean(sample['y'])])
    key = jax.random.PRNGKey(0)

    if method == "parametric":
        call = lambda: parametric_bootstrap(
            fit, lambda params, k: jnp.zeros(2), data, jnp.array([1.0]),
            key, n_boots=2, parallel=False,
        )
        message = "sample_function result"
    else:
        call = lambda: residual_bootstrap(
            fit, data, jnp.ones(2), jnp.ones(3), key, n_boots=2,
        )
        message = "fitted_values and residuals"

    with pytest.raises(ValueError, match=message):
        call()


def test_bootstrap_failure_diagnostics_for_nonfinite_fit():
    def fit(d):
        return jnp.array([jnp.inf])
    result = nonparametric_bootstrap(
        fit, {'y': jnp.arange(4.0)}, jax.random.PRNGKey(3),
        n_boots=5, parallel=False,
    )
    expected = np.arange(5)
    np.testing.assert_array_equal(np.asarray(result.failure_indices), expected)
    assert result.failure_reasons == {"nonfinite_parameters": len(expected)}


@pytest.mark.parametrize("method", ["nonparametric", "parametric", "residual"])
def test_bootstrap_nonfinite_fit_can_be_rejected(method):
    data = {'y': jnp.arange(4.0)}
    fit = lambda d: jnp.array([jnp.inf])
    key = jax.random.PRNGKey(0)

    if method == "nonparametric":
        call = lambda: nonparametric_bootstrap(
            fit, data, key, n_boots=3, parallel=False, handle_failures=False,
        )
    elif method == "parametric":
        call = lambda: parametric_bootstrap(
            fit, lambda params, k: jnp.zeros(4), data, jnp.array([1.0]),
            key, n_boots=3, parallel=False, handle_failures=False,
        )
    else:
        call = lambda: residual_bootstrap(
            fit, data, jnp.zeros(4), jnp.zeros(4), key,
            n_boots=3, parallel=False, handle_failures=False,
        )

    with pytest.raises(RuntimeError, match="non-finite fitted parameters"):
        call()


@pytest.mark.parametrize("method", ["nonparametric", "parametric", "residual"])
def test_bootstrap_all_nonfinite_fits_report_zero_success(method):
    data = {'y': jnp.arange(4.0)}
    fit = lambda d: jnp.array([jnp.nan])
    key = jax.random.PRNGKey(1)

    if method == "nonparametric":
        result = nonparametric_bootstrap(fit, data, key, n_boots=3, parallel=False)
    elif method == "parametric":
        result = parametric_bootstrap(
            fit, lambda params, k: jnp.zeros(4), data, jnp.array([1.0]),
            key, n_boots=3, parallel=False,
        )
    else:
        result = residual_bootstrap(
            fit, data, jnp.zeros(4), jnp.zeros(4), key,
            n_boots=3, parallel=False,
        )

    assert result.n_successful == 0
    assert result.failure_reasons == {"nonfinite_parameters": 3}
    assert np.isnan(np.asarray(result.mean)).all()
    assert np.isnan(np.asarray(result.lower_ci)).all()


def test_bootstrap_nonfinite_fits_obey_min_success_rate():
    with pytest.raises(RuntimeError, match="success rate"):
        nonparametric_bootstrap(
            lambda d: jnp.array([jnp.nan]), {'y': jnp.arange(4.0)},
            jax.random.PRNGKey(0), n_boots=2, parallel=False,
            min_success_rate=0.1,
        )


@pytest.mark.parametrize("method", ["nonparametric", "parametric", "residual"])
def test_bootstrap_warns_with_failure_summary(method):
    data = {'y': jnp.arange(4.0)}
    fit = lambda d: jnp.array([jnp.nan])
    key = jax.random.PRNGKey(2)

    with pytest.warns(RuntimeWarning, match=r"3 failed fits out of 3") as caught:
        if method == "nonparametric":
            nonparametric_bootstrap(fit, data, key, n_boots=3, parallel=False)
        elif method == "parametric":
            parametric_bootstrap(
                fit, lambda params, k: jnp.zeros(4), data, jnp.array([1.0]),
                key, n_boots=3, parallel=False,
            )
        else:
            residual_bootstrap(
                fit, data, jnp.zeros(4), jnp.zeros(4), key,
                n_boots=3, parallel=False,
            )

    assert "nonfinite_parameters=3" in str(caught[0].message)


def test_bootstrap_failure_warning_can_be_disabled():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = nonparametric_bootstrap(
            lambda d: jnp.array([jnp.nan]),
            {'y': jnp.arange(4.0)},
            jax.random.PRNGKey(3), n_boots=2, parallel=False,
            failure_warning="never",
        )

    assert not [warning for warning in caught if warning.category is RuntimeWarning]
    assert result.has_failures
    assert result.primary_failure_reason == "nonfinite_parameters"
    assert result.failure_summary == {
        "n_failed": 2,
        "failure_rate": 1.0,
        "primary_reason": "nonfinite_parameters",
        "reasons": {"nonfinite_parameters": 2},
    }


def test_bootstrap_threshold_warning_only_warns_at_threshold():
    fit_calls = [0]

    def fit(data):
        fit_calls[0] += 1
        return jnp.array([jnp.nan if fit_calls[0] == 2 else 1.0])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = nonparametric_bootstrap(
            fit, {'y': jnp.arange(4.0)}, jax.random.PRNGKey(4),
            n_boots=2, parallel=False, failure_warning="threshold",
            warning_failure_rate=0.5,
        )

    assert len([warning for warning in caught if warning.category is RuntimeWarning]) == 1
    assert result.n_successful == 1

    fit_calls[0] = 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        nonparametric_bootstrap(
            fit, {'y': jnp.arange(4.0)}, jax.random.PRNGKey(4),
            n_boots=2, parallel=False, failure_warning="threshold",
            warning_failure_rate=0.6,
        )
    assert not [warning for warning in caught if warning.category is RuntimeWarning]


@pytest.mark.parametrize("failure_warning", ["invalid", None, 1])
def test_bootstrap_rejects_invalid_failure_warning_policy(failure_warning):
    with pytest.raises(ValueError, match="failure_warning"):
        nonparametric_bootstrap(
            lambda d: jnp.array([1.0]), {'y': jnp.arange(3.0)},
            jax.random.PRNGKey(0), n_boots=1,
            failure_warning=failure_warning,
        )


@pytest.mark.parametrize("warning_failure_rate", [-0.1, 1.1, np.nan])
def test_bootstrap_rejects_invalid_warning_failure_rate(warning_failure_rate):
    with pytest.raises(ValueError, match="warning_failure_rate"):
        nonparametric_bootstrap(
            lambda d: jnp.array([1.0]), {'y': jnp.arange(3.0)},
            jax.random.PRNGKey(0), n_boots=1,
            warning_failure_rate=warning_failure_rate,
        )


def test_bootstrap_result_primary_failure_reason_is_deterministic():
    result = BootstrapResult(
        coefficients=jnp.full((4, 1), jnp.nan),
        n_boots=4,
        n_successful=0,
        failure_reasons={"ValueError": 2, "TypeError": 2},
    )

    assert result.primary_failure_reason == "TypeError"
    exported = result.to_dict()
    assert exported["has_failures"] is True
    assert exported["primary_failure_reason"] == "TypeError"
    assert exported["failure_summary"]["reasons"] == {
        "ValueError": 2, "TypeError": 2,
    }


@pytest.mark.parametrize("method", ["nonparametric", "parametric", "residual"])
@pytest.mark.parametrize("policy,threshold,expected", [
    ("always", 1.0, 1), ("never", 0.0, 0),
    ("threshold", 0.5, 1), ("threshold", 0.75, 0),
])
def test_warning_policy_with_mixed_failures(method, policy, threshold, expected):
    data = {'y': jnp.arange(4.0)}
    outcomes = iter([ValueError("fit failed"), jnp.nan, 1.0, 2.0])

    def fit(d):
        if d is data:
            return jnp.array([1.0])
        value = next(outcomes)
        if isinstance(value, Exception):
            raise value
        return jnp.array([value])

    kwargs = dict(n_boots=4, parallel=False, failure_warning=policy,
                  warning_failure_rate=threshold)
    key = jax.random.PRNGKey(0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if method == "nonparametric":
            result = nonparametric_bootstrap(fit, data, key, **kwargs)
        elif method == "parametric":
            result = parametric_bootstrap(
                fit, lambda p, k: data['y'], data, jnp.array([1.0]), key, **kwargs,
            )
        else:
            result = residual_bootstrap(
                fit, data, data['y'], jnp.zeros(4), key, **kwargs,
            )
    assert len(caught) == expected
    assert result.failure_reasons == {"ValueError": 1, "nonfinite_parameters": 1}
    assert result.n_successful == 2
    assert json.loads(result.to_json())["failure_summary"] == result.failure_summary
    if expected:
        assert "2 failed fits out of 4 (50.0%)" in str(caught[0].message)
        assert caught[0].filename == __file__


@pytest.mark.parametrize("method", ["nonparametric", "parametric", "residual"])
@pytest.mark.parametrize("outcome", ["success", "mixed", "exception"])
def test_parallel_retry_matches_sequential_diagnostics(method, outcome):
    data = {'y': jnp.arange(8.0)}
    key = jax.random.PRNGKey(42)

    def fit(d):
        if d is data:
            return jnp.array([3.5])
        # Ordinary Python control flow is intentionally not vmap-compatible.
        first = float(d['y'][0])
        if outcome == "exception" or (outcome == "mixed" and first < 3):
            raise ValueError("fit failed")
        if outcome == "mixed" and first < 5:
            return jnp.array([jnp.nan])
        return jnp.array([jnp.mean(d['y'])])

    def run(parallel, handle_failures=True):
        kwargs = dict(n_boots=24, parallel=parallel,
                      handle_failures=handle_failures, failure_warning="never")
        if method == "nonparametric":
            return nonparametric_bootstrap(fit, data, key, **kwargs)
        if method == "parametric":
            return parametric_bootstrap(
                fit, lambda p, k: jax.random.randint(k, (8,), 0, 8).astype(float),
                data, jnp.array([3.5]), key, **kwargs,
            )
        return residual_bootstrap(
            fit, data, jnp.zeros(8), data['y'], key, **kwargs,
        )

    expected = run(False)
    actual = run(True)
    np.testing.assert_allclose(actual.coefficients, expected.coefficients, equal_nan=True)
    np.testing.assert_array_equal(actual.failure_indices, expected.failure_indices)
    assert actual.failure_reasons == expected.failure_reasons
    assert actual.n_successful == expected.n_successful
    assert sum((actual.failure_reasons or {}).values()) == 24 - actual.n_successful
    if outcome == "mixed":
        assert actual.failure_reasons["ValueError"] > 0
        assert actual.failure_reasons["nonfinite_parameters"] > 0
        assert actual.n_successful > 0
    with pytest.raises(jax.errors.ConcretizationTypeError):
        run(True, handle_failures=False)


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
