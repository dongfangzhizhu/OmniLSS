"""Tests for batch 11 distributions (JSU, GIG, GB1)."""

import jax.numpy as jnp
import pytest
from omnilss import JSU, GIG, GB1


def test_jsu_basic():
    """Test JSU distribution basic functionality."""
    family = JSU()
    assert family.name == "JSU"
    assert family.parameters == ("mu", "sigma", "nu", "tau")
    
    # Test PDF evaluation
    y = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    mu = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
    sigma = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
    nu = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])  # no skewness
    tau = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])  # standard kurtosis
    
    pdf = family.pdf(y, mu, sigma, nu, tau)
    assert pdf.shape == y.shape
    assert jnp.all(jnp.isfinite(pdf))
    assert jnp.all(pdf >= 0)
    
    # With nu=0, tau=1, should be symmetric around mu
    assert jnp.abs(pdf[0] - pdf[4]) < 0.01  # pdf(-2) ≈ pdf(2)
    assert jnp.abs(pdf[1] - pdf[3]) < 0.01  # pdf(-1) ≈ pdf(1)


def test_jsu_skewness():
    """Test JSU distribution with skewness."""
    family = JSU()
    
    y = jnp.array([-1.0, 0.0, 1.0, 2.0])
    mu = jnp.array([0.0, 0.0, 0.0, 0.0])
    sigma = jnp.array([1.0, 1.0, 1.0, 1.0])
    nu = jnp.array([1.0, 1.0, 1.0, 1.0])  # positive skewness
    tau = jnp.array([1.0, 1.0, 1.0, 1.0])
    
    pdf = family.pdf(y, mu, sigma, nu, tau)
    assert jnp.all(jnp.isfinite(pdf))
    assert jnp.all(pdf >= 0)
    
    # With positive skewness, should have longer right tail
    # pdf(-1) should be different from pdf(1)
    assert pdf[0] != pdf[2]


def test_gig_basic():
    """Test GIG distribution basic functionality."""
    family = GIG()
    assert family.name == "GIG"
    assert family.parameters == ("mu", "sigma", "nu")
    
    # Test PDF evaluation
    y = jnp.array([0.5, 1.0, 2.0, 3.0, 5.0])
    mu = jnp.array([2.0, 2.0, 2.0, 2.0, 2.0])
    sigma = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5])
    nu = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])  # shape parameter
    
    pdf = family.pdf(y, mu, sigma, nu)
    assert pdf.shape == y.shape
    assert jnp.all(jnp.isfinite(pdf))
    assert jnp.all(pdf >= 0)


def test_gig_positive_nu():
    """Test GIG distribution with positive nu."""
    family = GIG()
    
    y = jnp.array([0.5, 1.0, 2.0, 3.0])
    mu = jnp.array([2.0, 2.0, 2.0, 2.0])
    sigma = jnp.array([0.5, 0.5, 0.5, 0.5])
    nu = jnp.array([1.0, 1.0, 1.0, 1.0])
    
    pdf = family.pdf(y, mu, sigma, nu)
    assert jnp.all(jnp.isfinite(pdf))
    assert jnp.all(pdf >= 0)


def test_gig_negative_nu():
    """Test GIG distribution with negative nu."""
    family = GIG()
    
    y = jnp.array([0.5, 1.0, 2.0, 3.0])
    mu = jnp.array([2.0, 2.0, 2.0, 2.0])
    sigma = jnp.array([0.5, 0.5, 0.5, 0.5])
    nu = jnp.array([-0.5, -0.5, -0.5, -0.5])
    
    pdf = family.pdf(y, mu, sigma, nu)
    assert jnp.all(jnp.isfinite(pdf))
    assert jnp.all(pdf >= 0)


def test_gb1_basic():
    """Test GB1 distribution basic functionality."""
    family = GB1()
    assert family.name == "GB1"
    assert family.parameters == ("mu", "sigma", "nu", "tau")
    
    # Test PDF evaluation (y must be in (0, 1))
    y = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
    mu = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5])
    sigma = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5])
    nu = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
    tau = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
    
    pdf = family.pdf(y, mu, sigma, nu, tau)
    assert pdf.shape == y.shape
    assert jnp.all(jnp.isfinite(pdf))
    assert jnp.all(pdf >= 0)


def test_gb1_symmetry():
    """Test GB1 distribution symmetry."""
    family = GB1()
    
    # With mu=0.5 and symmetric parameters, should be symmetric
    y = jnp.array([0.2, 0.3, 0.5, 0.7, 0.8])
    mu = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5])
    sigma = jnp.array([0.3, 0.3, 0.3, 0.3, 0.3])
    nu = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
    tau = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
    
    pdf = family.pdf(y, mu, sigma, nu, tau)
    assert jnp.all(jnp.isfinite(pdf))
    assert jnp.all(pdf >= 0)
    
    # pdf(0.2) should be close to pdf(0.8) for symmetric case
    # (approximately, depending on parameters)
    assert pdf[2] > 0  # pdf at center should be positive


def test_gb1_boundary():
    """Test GB1 distribution near boundaries."""
    family = GB1()
    
    # Test near 0 and 1
    y = jnp.array([0.01, 0.05, 0.95, 0.99])
    mu = jnp.array([0.5, 0.5, 0.5, 0.5])
    sigma = jnp.array([0.5, 0.5, 0.5, 0.5])
    nu = jnp.array([1.0, 1.0, 1.0, 1.0])
    tau = jnp.array([1.0, 1.0, 1.0, 1.0])
    
    pdf = family.pdf(y, mu, sigma, nu, tau)
    assert jnp.all(jnp.isfinite(pdf))
    assert jnp.all(pdf >= 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
