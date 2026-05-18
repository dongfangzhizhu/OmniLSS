"""Tests for Cole-Green (CG) fitting algorithm.

This module tests the CG algorithm implementation for GAMLSS models.
"""

import pytest
import jax.numpy as jnp
import numpy as np

from omnilss.fitting_cg import fit_cg, CGResult
from omnilss.distributions import NO, GA, BE


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def simple_normal_data():
    """Simple normal data for testing."""
    np.random.seed(42)
    n = 100
    x = np.random.randn(n)
    y = 2.0 + 3.0 * x + np.random.randn(n) * 0.5
    
    return {
        "y": jnp.array(y),
        "X_mu": jnp.column_stack([jnp.ones(n), jnp.array(x)]),
        "X_sigma": jnp.ones((n, 1)),
    }


@pytest.fixture
def intercept_only_data():
    """Intercept-only model data."""
    y = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    X = jnp.ones((5, 1))
    
    return {"y": y, "X_mu": X}


# =============================================================================
# Basic Functionality Tests
# =============================================================================

def test_fit_cg_basic(intercept_only_data):
    """Test basic CG fitting with intercept-only model."""
    result = fit_cg(
        NO(),
        intercept_only_data["y"],
        intercept_only_data["X_mu"],
        max_iter=50,
        verbose=False
    )
    
    # Should converge
    assert isinstance(result, CGResult)
    assert result.converged
    
    # Should have reasonable number of iterations
    assert result.n_iter < 50
    
    # Fitted mu should be close to mean
    expected_mean = jnp.mean(intercept_only_data["y"])
    assert jnp.allclose(result.fitted_values["mu"], expected_mean, atol=0.1)


def test_fit_cg_with_covariates(simple_normal_data):
    """Test CG fitting with covariates."""
    result = fit_cg(
        NO(),
        simple_normal_data["y"],
        simple_normal_data["X_mu"],
        simple_normal_data["X_sigma"],
        max_iter=100,
        verbose=False
    )
    
    # Should converge
    assert result.converged
    
    # Should have parameters
    assert "beta_mu" in result.params
    assert "beta_sigma" in result.params
    
    # Beta_mu should have 2 coefficients (intercept + slope)
    assert result.params["beta_mu"].shape == (2,)
    
    # Coefficients should be reasonable
    # True model: y = 2 + 3*x + N(0, 0.5^2)
    assert 1.5 < result.params["beta_mu"][0] < 2.5  # Intercept
    assert 2.5 < result.params["beta_mu"][1] < 3.5  # Slope


def test_fit_cg_result_structure(intercept_only_data):
    """Test CGResult structure."""
    result = fit_cg(
        NO(),
        intercept_only_data["y"],
        intercept_only_data["X_mu"],
        max_iter=50
    )
    
    # Check all required attributes
    assert hasattr(result, "params")
    assert hasattr(result, "fitted_values")
    assert hasattr(result, "converged")
    assert hasattr(result, "n_iter")
    assert hasattr(result, "final_deviance")
    assert hasattr(result, "deviance_history")
    
    # Check types
    assert isinstance(result.params, dict)
    assert isinstance(result.fitted_values, dict)
    assert isinstance(result.converged, (bool, jnp.ndarray))
    assert isinstance(result.n_iter, int)
    assert isinstance(result.final_deviance, (float, jnp.ndarray))
    assert isinstance(result.deviance_history, list)


# =============================================================================
# Convergence Tests
# =============================================================================

def test_cg_convergence_simple():
    """Test CG convergence on simple problem."""
    # Very simple data
    y = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
    X = jnp.ones((5, 1))
    
    result = fit_cg(NO(), y, X, max_iter=50, tol=1e-6)
    
    # Should converge quickly
    assert result.converged
    assert result.n_iter < 20
    
    # Fitted values should be constant
    assert jnp.allclose(result.fitted_values["mu"], 1.0, atol=0.01)


def test_cg_deviance_decreasing(simple_normal_data):
    """Test that deviance decreases monotonically."""
    result = fit_cg(
        NO(),
        simple_normal_data["y"],
        simple_normal_data["X_mu"],
        simple_normal_data["X_sigma"],
        max_iter=50
    )
    
    # Deviance should generally decrease
    deviances = result.deviance_history
    
    # Check that final deviance is less than initial
    assert deviances[-1] < deviances[0]
    
    # Most steps should decrease deviance
    decreases = sum(1 for i in range(1, len(deviances)) if deviances[i] < deviances[i-1])
    assert decreases > len(deviances) * 0.7  # At least 70% of steps decrease


def test_cg_max_iter():
    """Test max_iter parameter."""
    y = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    X = jnp.ones((5, 1))
    
    # Set very low max_iter with very tight tolerance
    result = fit_cg(NO(), y, X, max_iter=3, tol=1e-10)
    
    # Should either converge quickly (simple problem) or not converge
    # This is a very simple problem, so it might converge in 1-2 iterations
    assert result.n_iter <= 3


# =============================================================================
# Parameter Tests
# =============================================================================

def test_cg_start_params():
    """Test starting from custom parameters."""
    y = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    X = jnp.ones((5, 1))
    
    # Custom starting parameters
    start_params = {
        "beta_mu": jnp.array([2.0])  # Close to true mean (3.0)
    }
    
    result = fit_cg(
        NO(), y, X,
        start_params=start_params,
        max_iter=50
    )
    
    assert result.converged
    # Should converge faster with good starting values
    assert result.n_iter < 20


def test_cg_step_size():
    """Test step size parameter."""
    y = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    X = jnp.ones((5, 1))
    
    # Small step size
    result_small = fit_cg(NO(), y, X, step_size=0.5, max_iter=100)
    
    # Full step size
    result_full = fit_cg(NO(), y, X, step_size=1.0, max_iter=100)
    
    # Both should converge
    assert result_small.converged
    assert result_full.converged
    
    # Small step size may take more iterations
    # (but not always, depends on problem)
    assert result_small.n_iter > 0
    assert result_full.n_iter > 0


# =============================================================================
# Different Distributions Tests
# =============================================================================

def test_cg_gamma_distribution():
    """Test CG with Gamma distribution."""
    np.random.seed(42)
    n = 50
    
    # Generate Gamma data
    shape = 2.0
    scale = 1.0
    y = np.random.gamma(shape, scale, n)
    X = jnp.ones((n, 1))
    
    result = fit_cg(
        GA(),
        jnp.array(y),
        X,
        jnp.ones((n, 1)),  # X_sigma
        max_iter=100
    )
    
    # Should converge
    assert result.converged
    
    # Fitted mu should be close to mean
    assert jnp.mean(result.fitted_values["mu"]) > 0


def test_cg_beta_distribution():
    """Test CG with Beta distribution."""
    np.random.seed(42)
    n = 50
    
    # Generate Beta data
    alpha = 2.0
    beta = 5.0
    y = np.random.beta(alpha, beta, n)
    X = jnp.ones((n, 1))
    
    result = fit_cg(
        BE(),
        jnp.array(y),
        X,
        jnp.ones((n, 1)),  # X_sigma
        max_iter=100
    )
    
    # Should converge
    assert result.converged
    
    # Fitted mu should be in (0, 1)
    assert jnp.all(result.fitted_values["mu"] > 0)
    assert jnp.all(result.fitted_values["mu"] < 1)


# =============================================================================
# Weights Tests
# =============================================================================

def test_cg_with_weights():
    """Test CG with observation weights."""
    y = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    X = jnp.ones((5, 1))
    weights = jnp.array([1.0, 1.0, 1.0, 1.0, 10.0])  # Last obs has high weight
    
    result = fit_cg(NO(), y, X, weights=weights, max_iter=50)
    
    assert result.converged
    
    # Fitted mean should be closer to 5.0 (high weight observation)
    fitted_mean = result.fitted_values["mu"][0]
    assert fitted_mean > 3.0  # Should be > unweighted mean


# =============================================================================
# Fisher Matrix Tests
# =============================================================================

def test_cg_return_fisher():
    """Test returning Fisher information matrix."""
    y = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    X = jnp.ones((5, 1))
    
    result = fit_cg(NO(), y, X, return_fisher=True, max_iter=50)
    
    assert result.converged
    assert result.fisher_matrix is not None
    
    # Fisher should be square matrix
    assert result.fisher_matrix.ndim == 2
    assert result.fisher_matrix.shape[0] == result.fisher_matrix.shape[1]
    
    # Fisher should be positive definite
    eigenvalues = jnp.linalg.eigvalsh(result.fisher_matrix)
    assert jnp.all(eigenvalues > 0)


# =============================================================================
# Edge Cases Tests
# =============================================================================

def test_cg_constant_response():
    """Test CG with constant response."""
    y = jnp.ones(10)
    X = jnp.ones((10, 1))
    
    result = fit_cg(NO(), y, X, max_iter=50)
    
    # Should converge
    assert result.converged
    
    # Fitted values should be constant
    assert jnp.allclose(result.fitted_values["mu"], 1.0, atol=0.01)


def test_cg_small_sample():
    """Test CG with very small sample."""
    y = jnp.array([1.0, 2.0, 3.0])
    X = jnp.ones((3, 1))
    
    result = fit_cg(NO(), y, X, max_iter=50)
    
    # Should still work
    assert isinstance(result, CGResult)


def test_cg_large_sample():
    """Test CG with larger sample."""
    np.random.seed(42)
    n = 1000
    y = np.random.randn(n) + 5.0
    X = jnp.ones((n, 1))
    
    result = fit_cg(NO(), jnp.array(y), X, max_iter=100)
    
    assert result.converged
    
    # Should estimate mean accurately
    assert jnp.abs(result.fitted_values["mu"][0] - 5.0) < 0.1


# =============================================================================
# Verbose Output Tests
# =============================================================================

def test_cg_verbose_output(capsys, intercept_only_data):
    """Test verbose output."""
    fit_cg(
        NO(),
        intercept_only_data["y"],
        intercept_only_data["X_mu"],
        max_iter=50,
        verbose=True
    )
    
    # Capture output
    captured = capsys.readouterr()
    
    # Should contain iteration info
    assert "Cole-Green" in captured.out
    assert "Iter" in captured.out
    assert "deviance" in captured.out


# =============================================================================
# Comparison Tests (if RS is available)
# =============================================================================

def test_cg_vs_simple_estimate():
    """Test CG gives reasonable estimates."""
    # Simple data where we know the answer
    y = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    X = jnp.ones((5, 1))
    
    result = fit_cg(NO(), y, X, max_iter=50)
    
    # Mean should be 3.0
    assert jnp.abs(result.fitted_values["mu"][0] - 3.0) < 0.01


def test_full_observed_information_includes_cross_derivative_blocks(simple_normal_data):
    """CG observed information should retain cross-parameter Hessian blocks."""
    from omnilss.fitting_cg import (
        _compute_full_observed_information_and_score,
        _compute_log_likelihood,
    )

    family = NO()
    # Use an intentionally coupled off-optimum state so the Normal
    # mu/sigma cross block is measurably non-zero.
    params = {
        "beta_mu": jnp.array([jnp.mean(simple_normal_data["y"]), 0.0]),
        "beta_sigma": jnp.array([jnp.log(jnp.std(simple_normal_data["y"]))]),
    }
    design_matrices = {
        "X_mu": simple_normal_data["X_mu"],
        "X_sigma": simple_normal_data["X_sigma"],
        "X_nu": None,
        "X_tau": None,
    }
    data = {
        "y": simple_normal_data["y"],
        "weights": jnp.ones_like(simple_normal_data["y"]),
        **design_matrices,
    }

    def log_likelihood(params_dict, data_dict):
        return _compute_log_likelihood(params_dict, data_dict, family, design_matrices)

    fisher, score = _compute_full_observed_information_and_score(log_likelihood, params, data)

    assert fisher.shape == (3, 3)
    assert score.shape == (3,)
    assert jnp.allclose(fisher, fisher.T)
    # Flattening order is beta_mu (2 coefficients), then beta_sigma (1 coefficient).
    # The slope/sigma block is non-zero for this heteroscedastic Normal likelihood.
    assert jnp.abs(fisher[1, 2]) > 1e-3



def test_gamlss_algorithm_alias_exercises_cg_backend():
    """The legacy algorithm= alias should select the complete CG backend."""
    from omnilss.fitting import gamlss, gamlss_control

    x = np.linspace(0.0, 1.0, 40)
    y = 1.0 + 2.0 * x + np.linspace(-0.05, 0.05, 40)
    model = gamlss(
        "y ~ x",
        sigma_formula="~ 1",
        family=NO(),
        data={"y": y, "x": x},
        algorithm="CG",
        control=gamlss_control(n_cyc=20, c_crit=1e-5),
    )

    assert model.additional_slots["method"] == "CG"
    assert "cg_converged" in model.additional_slots
    assert "cg_iterations" in model.additional_slots


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_fit_cg_handles_near_singular_design_with_regularization():
    """CG should return finite results even when design columns are collinear."""
    rng = np.random.default_rng(20260518)
    n = 60
    x = rng.normal(size=n)
    # Perfectly collinear second slope column
    X_mu = jnp.column_stack([jnp.ones(n), jnp.array(x), jnp.array(2.0 * x)])
    X_sigma = jnp.ones((n, 1))
    y = jnp.array(1.5 + 0.8 * x + rng.normal(scale=0.3, size=n))

    result = fit_cg(
        NO(),
        y,
        X_mu,
        X_sigma,
        max_iter=30,
        tol=1e-5,
        regularization=1e-5,
        return_fisher=True,
    )

    assert np.isfinite(result.final_deviance)
    assert result.n_iter >= 1
    assert result.fisher_matrix is not None
    assert np.all(np.isfinite(np.asarray(result.fisher_matrix)))
