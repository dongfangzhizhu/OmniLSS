"""Tests for probit link function."""

import jax.numpy as jnp
import numpy as np
import pytest
from omnilss.links import probit_link, probit_inverse, probit_derivative


def test_probit_inverse_relationship():
    """Test that probit_link and probit_inverse are inverses."""
    mu = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
    eta = probit_link(mu)
    mu_recovered = probit_inverse(eta)
    assert jnp.allclose(mu, mu_recovered, atol=1e-6)


def test_probit_derivative_finite_diff():
    """Test probit_derivative using finite differences."""
    eta_test = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    deriv_analytical = probit_derivative(eta_test)
    
    # Finite difference approximation
    eps = 1e-6
    deriv_numerical = (probit_inverse(eta_test + eps) - probit_inverse(eta_test - eps)) / (2 * eps)
    
    assert jnp.allclose(deriv_analytical, deriv_numerical, atol=1e-5)


def test_probit_boundary_values():
    """Test probit link at boundary values."""
    eps = jnp.finfo(jnp.float64).eps
    
    # Near 0
    mu_low = jnp.array([eps, 0.01, 0.05])
    eta_low = probit_link(mu_low)
    assert jnp.all(jnp.isfinite(eta_low))
    assert jnp.all(eta_low < 0)  # Should be negative
    
    # Near 1
    mu_high = jnp.array([0.95, 0.99, 1.0 - eps])
    eta_high = probit_link(mu_high)
    assert jnp.all(jnp.isfinite(eta_high))
    assert jnp.all(eta_high > 0)  # Should be positive
    
    # At 0.5
    mu_mid = jnp.array([0.5])
    eta_mid = probit_link(mu_mid)
    assert jnp.allclose(eta_mid, 0.0, atol=1e-10)


def test_probit_derivative_positive():
    """Test that probit derivative is always positive."""
    eta = jnp.linspace(-5, 5, 100)
    deriv = probit_derivative(eta)
    assert jnp.all(deriv > 0)


def test_probit_vs_logit():
    """Compare probit and logit links."""
    from omnilss.links import logit_link, logit_inverse
    
    mu = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
    
    # Both should map (0,1) to real line
    eta_probit = probit_link(mu)
    eta_logit = logit_link(mu)
    
    # Both should be finite
    assert jnp.all(jnp.isfinite(eta_probit))
    assert jnp.all(jnp.isfinite(eta_logit))
    
    # At 0.5, both should be 0
    assert jnp.allclose(probit_link(jnp.array([0.5])), 0.0, atol=1e-10)
    assert jnp.allclose(logit_link(jnp.array([0.5])), 0.0, atol=1e-10)
    
    # Probit is generally less extreme than logit
    # (except near the boundaries)
    mu_mid = jnp.array([0.3, 0.4, 0.6, 0.7])
    assert jnp.all(jnp.abs(probit_link(mu_mid)) < jnp.abs(logit_link(mu_mid)))


def test_probit_symmetry():
    """Test symmetry: probit(1-p) = -probit(p)."""
    mu = jnp.array([0.1, 0.2, 0.3, 0.4])
    eta_low = probit_link(mu)
    eta_high = probit_link(1.0 - mu)
    assert jnp.allclose(eta_low, -eta_high, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
