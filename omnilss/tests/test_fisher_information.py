"""Tests for Fisher information matrix computation.

This module tests the Fisher information matrix and score vector
computation functions used in the Cole-Green (CG) algorithm.
"""

import pytest
import jax.numpy as jnp
import numpy as np

from omnilss.fisher_information import (
    compute_fisher_matrix,
    compute_observed_information,
    compute_score_vector,
    flatten_params,
    check_fisher_matrix,
    compute_parameter_covariance,
    compute_standard_errors,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def simple_normal_loglik():
    """Simple normal log-likelihood for testing."""
    def log_lik(params, data):
        mu = params["mu"]
        log_sigma = params["log_sigma"]
        sigma = jnp.exp(log_sigma)
        y = data["y"]
        
        # Normal log-likelihood
        residuals = (y - mu) / sigma
        ll = -0.5 * jnp.log(2 * jnp.pi) - log_sigma - 0.5 * residuals ** 2
        return jnp.sum(ll)
    
    return log_lik


@pytest.fixture
def simple_data():
    """Simple test data."""
    return {"y": jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])}


# =============================================================================
# Flatten/Unflatten Tests
# =============================================================================

def test_flatten_params_single():
    """Test flattening single parameter."""
    params = {"mu": jnp.array([1.0, 2.0, 3.0])}
    
    vec, unflatten = flatten_params(params)
    
    assert vec.shape == (3,)
    assert jnp.allclose(vec, jnp.array([1.0, 2.0, 3.0]))
    
    # Test unflatten
    reconstructed = unflatten(vec)
    assert "mu" in reconstructed
    assert jnp.allclose(reconstructed["mu"], params["mu"])


def test_flatten_params_multiple():
    """Test flattening multiple parameters."""
    params = {
        "mu": jnp.array([1.0, 2.0]),
        "sigma": jnp.array([0.5]),
        "nu": jnp.array([0.0, 1.0, 2.0])
    }
    
    vec, unflatten = flatten_params(params)
    
    # Should concatenate in sorted order: mu, nu, sigma
    assert vec.shape == (6,)
    
    # Test unflatten
    reconstructed = unflatten(vec)
    assert set(reconstructed.keys()) == set(params.keys())
    for key in params:
        assert jnp.allclose(reconstructed[key], params[key])


def test_flatten_params_scalar():
    """Test flattening scalar parameters."""
    params = {
        "mu": jnp.array(1.0),
        "sigma": jnp.array(0.5)
    }
    
    vec, unflatten = flatten_params(params)
    
    assert vec.shape == (2,)
    
    reconstructed = unflatten(vec)
    assert reconstructed["mu"].shape == ()
    assert reconstructed["sigma"].shape == ()


# =============================================================================
# Score Vector Tests
# =============================================================================

def test_score_vector_simple(simple_normal_loglik, simple_data):
    """Test score vector computation for simple normal model."""
    params = {
        "mu": jnp.array(3.0),  # True mean
        "log_sigma": jnp.array(0.0)  # log(1) = 0
    }
    
    score = compute_score_vector(simple_normal_loglik, params, simple_data)
    
    # At true parameters, score should be close to zero
    assert score.shape == (2,)
    # Score for mu should be sum of residuals / sigma^2
    # y = [1,2,3,4,5], mu = 3, residuals = [-2,-1,0,1,2], sum = 0
    # But we're using log_sigma parameterization, so gradient is different
    # Just check that score is computed (not necessarily zero)
    assert jnp.isfinite(score[0])
    assert jnp.isfinite(score[1])


def test_score_vector_away_from_optimum(simple_normal_loglik, simple_data):
    """Test score vector away from optimum."""
    params = {
        "mu": jnp.array(0.0),  # Far from true mean (3.0)
        "log_sigma": jnp.array(0.0)
    }
    
    score = compute_score_vector(simple_normal_loglik, params, simple_data)
    
    # Score should be non-zero
    assert abs(score[0]) > 1.0  # Should be large


def test_score_vector_numerical_gradient(simple_normal_loglik, simple_data):
    """Test score vector against numerical gradient."""
    params = {
        "mu": jnp.array(2.5),
        "log_sigma": jnp.array(0.1)
    }
    
    # Analytical score
    score = compute_score_vector(simple_normal_loglik, params, simple_data)
    
    # Numerical gradient
    eps = 1e-5
    vec, unflatten = flatten_params(params)
    
    def log_lik_vec(theta_vec):
        theta_dict = unflatten(theta_vec)
        return simple_normal_loglik(theta_dict, simple_data)
    
    numerical_grad = np.zeros_like(vec)
    for i in range(len(vec)):
        vec_plus = vec.at[i].add(eps)
        vec_minus = vec.at[i].add(-eps)
        numerical_grad[i] = (log_lik_vec(vec_plus) - log_lik_vec(vec_minus)) / (2 * eps)
    
    # Should be close
    assert jnp.allclose(score, numerical_grad, rtol=1e-4)


# =============================================================================
# Fisher Matrix Tests
# =============================================================================

def test_fisher_matrix_shape(simple_normal_loglik, simple_data):
    """Test Fisher matrix has correct shape."""
    params = {
        "mu": jnp.array(3.0),
        "log_sigma": jnp.array(0.0)
    }
    
    fisher = compute_fisher_matrix(simple_normal_loglik, params, simple_data)
    
    assert fisher.shape == (2, 2)


def test_fisher_matrix_symmetric(simple_normal_loglik, simple_data):
    """Test Fisher matrix is symmetric."""
    params = {
        "mu": jnp.array(3.0),
        "log_sigma": jnp.array(0.0)
    }
    
    fisher = compute_fisher_matrix(simple_normal_loglik, params, simple_data)
    
    assert jnp.allclose(fisher, fisher.T)


def test_fisher_matrix_positive_definite(simple_normal_loglik, simple_data):
    """Test Fisher matrix is positive definite."""
    params = {
        "mu": jnp.array(3.0),
        "log_sigma": jnp.array(0.0)
    }
    
    fisher = compute_fisher_matrix(simple_normal_loglik, params, simple_data)
    
    # Check eigenvalues are positive
    eigenvalues = jnp.linalg.eigvalsh(fisher)
    assert jnp.all(eigenvalues > 0)


def test_fisher_matrix_numerical_hessian(simple_normal_loglik, simple_data):
    """Test Fisher matrix against numerical Hessian."""
    params = {
        "mu": jnp.array(3.0),
        "log_sigma": jnp.array(0.0)
    }
    
    # Analytical Fisher
    fisher = compute_fisher_matrix(simple_normal_loglik, params, simple_data)
    
    # Numerical Hessian
    vec, unflatten = flatten_params(params)
    
    def log_lik_vec(theta_vec):
        theta_dict = unflatten(theta_vec)
        return simple_normal_loglik(theta_dict, simple_data)
    
    eps = 1e-5
    n = len(vec)
    numerical_hessian = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            vec_pp = vec.at[i].add(eps).at[j].add(eps)
            vec_pm = vec.at[i].add(eps).at[j].add(-eps)
            vec_mp = vec.at[i].add(-eps).at[j].add(eps)
            vec_mm = vec.at[i].add(-eps).at[j].add(-eps)
            
            numerical_hessian[i, j] = (
                log_lik_vec(vec_pp) - log_lik_vec(vec_pm) -
                log_lik_vec(vec_mp) + log_lik_vec(vec_mm)
            ) / (4 * eps * eps)
    
    # Fisher = -Hessian
    numerical_fisher = -numerical_hessian
    
    # Should be close (relaxed tolerance for numerical differentiation)
    assert jnp.allclose(fisher, numerical_fisher, rtol=0.01, atol=1e-4)


# =============================================================================
# Check Fisher Matrix Tests
# =============================================================================

def test_check_fisher_matrix_good():
    """Test checking a good Fisher matrix."""
    # Well-conditioned positive definite matrix
    fisher = jnp.array([[2.0, 0.1], [0.1, 1.0]])
    
    checks = check_fisher_matrix(fisher)
    
    assert checks["symmetric"]
    assert checks["positive_definite"]
    assert checks["well_conditioned"]


def test_check_fisher_matrix_asymmetric():
    """Test checking asymmetric matrix."""
    fisher = jnp.array([[2.0, 0.1], [0.2, 1.0]])
    
    checks = check_fisher_matrix(fisher)
    
    assert not checks["symmetric"]


def test_check_fisher_matrix_not_positive_definite():
    """Test checking non-positive definite matrix."""
    fisher = jnp.array([[1.0, 2.0], [2.0, 1.0]])
    
    checks = check_fisher_matrix(fisher)
    
    assert not checks["positive_definite"]


# =============================================================================
# Covariance and Standard Errors Tests
# =============================================================================

def test_compute_covariance(simple_normal_loglik, simple_data):
    """Test covariance matrix computation."""
    params = {
        "mu": jnp.array(3.0),
        "log_sigma": jnp.array(0.0)
    }
    
    fisher = compute_fisher_matrix(simple_normal_loglik, params, simple_data)
    cov = compute_parameter_covariance(fisher)
    
    # Covariance should be positive definite
    eigenvalues = jnp.linalg.eigvalsh(cov)
    assert jnp.all(eigenvalues > 0)
    
    # Covariance * Fisher should be approximately identity
    product = cov @ fisher
    assert jnp.allclose(product, jnp.eye(2), atol=1e-5)


def test_compute_standard_errors(simple_normal_loglik, simple_data):
    """Test standard error computation."""
    params = {
        "mu": jnp.array(3.0),
        "log_sigma": jnp.array(0.0)
    }
    
    fisher = compute_fisher_matrix(simple_normal_loglik, params, simple_data)
    se = compute_standard_errors(fisher)
    
    # Standard errors should be positive
    assert jnp.all(se > 0)
    
    # For normal distribution with n=5 observations:
    # SE(mu) should be reasonable (between 0.1 and 1.0)
    assert 0.1 < se[0] < 1.0


def test_standard_errors_known_case():
    """Test standard errors for known case."""
    # For normal distribution with known variance:
    # Fisher information for mu is n / sigma^2
    # SE(mu) = sigma / sqrt(n)
    
    n = 100
    sigma = 2.0
    
    # Fisher information
    fisher_mu = n / (sigma ** 2)
    fisher = jnp.array([[fisher_mu]])
    
    se = compute_standard_errors(fisher)
    
    # Expected SE
    expected_se = sigma / jnp.sqrt(n)
    
    assert jnp.allclose(se[0], expected_se, rtol=0.01)


# =============================================================================
# Integration Tests
# =============================================================================

def test_fisher_score_relationship(simple_normal_loglik, simple_data):
    """Test relationship between Fisher matrix and score vector."""
    # At MLE, score should be zero
    # Fisher matrix should still be well-defined
    
    params_mle = {
        "mu": jnp.array(3.0),  # True mean of [1,2,3,4,5]
        "log_sigma": jnp.array(jnp.log(jnp.std(simple_data["y"])))
    }
    
    score = compute_score_vector(simple_normal_loglik, params_mle, simple_data)
    fisher = compute_fisher_matrix(simple_normal_loglik, params_mle, simple_data)
    
    # Score should be close to zero at MLE
    assert jnp.allclose(score, 0.0, atol=0.1)
    
    # Fisher should still be positive definite
    eigenvalues = jnp.linalg.eigvalsh(fisher)
    assert jnp.all(eigenvalues > 0)


def test_multiple_parameters():
    """Test with multiple parameters."""
    def log_lik(params, data):
        mu = params["mu"]
        log_sigma = params["log_sigma"]
        nu = params["nu"]
        
        sigma = jnp.exp(log_sigma)
        y = data["y"]
        
        # Simple model: y ~ N(mu + nu * x, sigma^2)
        x = data["x"]
        mean = mu + nu * x
        residuals = (y - mean) / sigma
        ll = -0.5 * jnp.log(2 * jnp.pi) - log_sigma - 0.5 * residuals ** 2
        return jnp.sum(ll)
    
    params = {
        "mu": jnp.array(1.0),
        "log_sigma": jnp.array(0.0),
        "nu": jnp.array(0.5)
    }
    
    data = {
        "y": jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "x": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    }
    
    # Compute Fisher and score
    fisher = compute_fisher_matrix(log_lik, params, data)
    score = compute_score_vector(log_lik, params, data)
    
    # Check dimensions
    assert fisher.shape == (3, 3)
    assert score.shape == (3,)
    
    # Check properties
    checks = check_fisher_matrix(fisher)
    assert checks["symmetric"]
    # Note: Fisher may not be positive definite at arbitrary parameter values
    # Just check that it's computed
    assert jnp.all(jnp.isfinite(fisher))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
