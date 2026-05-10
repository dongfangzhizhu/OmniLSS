"""Tests for batch 10 zero-variant distributions."""

import jax.numpy as jnp
import pytest
from omnilss import (
    ZASICHEL, ZISICHEL, ZAPIG, ZIPIG, ZABI, ZIBI, ZAZIPF,
    ZANBI, ZABNB, ZIBNB
)


def test_zasichel_basic():
    """Test ZASICHEL distribution basic functionality."""
    family = ZASICHEL()
    assert family.name == "ZASICHEL"
    assert family.parameters == ("mu", "sigma", "nu", "nu")  # base params + zero param
    
    # Test PDF evaluation
    y = jnp.array([0.0, 1.0, 2.0, 5.0])
    mu = jnp.array([2.0, 2.0, 2.0, 2.0])
    sigma = jnp.array([0.5, 0.5, 0.5, 0.5])
    nu = jnp.array([0.0, 0.0, 0.0, 0.0])
    nu_zero = jnp.array([0.3, 0.3, 0.3, 0.3])
    
    pdf = family.pdf(y, mu, sigma, nu, nu_zero)
    assert pdf.shape == y.shape
    assert jnp.all(jnp.isfinite(pdf))
    assert jnp.all(pdf >= 0)


def test_zisichel_basic():
    """Test ZISICHEL distribution basic functionality."""
    family = ZISICHEL()
    assert family.name == "ZISICHEL"
    assert family.parameters == ("mu", "sigma", "nu", "nu")
    
    y = jnp.array([0.0, 1.0, 2.0, 5.0])
    mu = jnp.array([2.0, 2.0, 2.0, 2.0])
    sigma = jnp.array([0.5, 0.5, 0.5, 0.5])
    nu = jnp.array([0.0, 0.0, 0.0, 0.0])
    nu_zero = jnp.array([0.3, 0.3, 0.3, 0.3])
    
    pdf = family.pdf(y, mu, sigma, nu, nu_zero)
    assert pdf.shape == y.shape
    assert jnp.all(jnp.isfinite(pdf))
    assert jnp.all(pdf >= 0)


def test_zapig_basic():
    """Test ZAPIG distribution basic functionality."""
    family = ZAPIG()
    assert family.name == "ZAPIG"
    assert family.parameters == ("mu", "sigma", "nu")
    
    y = jnp.array([0.0, 1.0, 2.0, 5.0])
    mu = jnp.array([2.0, 2.0, 2.0, 2.0])
    sigma = jnp.array([0.5, 0.5, 0.5, 0.5])
    nu_zero = jnp.array([0.3, 0.3, 0.3, 0.3])
    
    pdf = family.pdf(y, mu, sigma, nu_zero)
    assert pdf.shape == y.shape
    assert jnp.all(jnp.isfinite(pdf))
    assert jnp.all(pdf >= 0)


def test_zipig_basic():
    """Test ZIPIG distribution basic functionality."""
    family = ZIPIG()
    assert family.name == "ZIPIG"
    assert family.parameters == ("mu", "sigma", "nu")
    
    y = jnp.array([0.0, 1.0, 2.0, 5.0])
    mu = jnp.array([2.0, 2.0, 2.0, 2.0])
    sigma = jnp.array([0.5, 0.5, 0.5, 0.5])
    nu_zero = jnp.array([0.3, 0.3, 0.3, 0.3])
    
    pdf = family.pdf(y, mu, sigma, nu_zero)
    assert pdf.shape == y.shape
    assert jnp.all(jnp.isfinite(pdf))
    assert jnp.all(pdf >= 0)


def test_zabi_basic():
    """Test ZABI distribution basic functionality."""
    family = ZABI()
    assert family.name == "ZABI"
    assert family.parameters == ("mu", "nu")
    
    y = jnp.array([0.0, 1.0, 2.0, 5.0])
    mu = jnp.array([0.5, 0.5, 0.5, 0.5])
    nu_zero = jnp.array([0.3, 0.3, 0.3, 0.3])
    
    pdf = family.pdf(y, mu, nu_zero)
    assert pdf.shape == y.shape
    assert jnp.all(jnp.isfinite(pdf))
    assert jnp.all(pdf >= 0)


def test_zibi_basic():
    """Test ZIBI distribution basic functionality."""
    family = ZIBI()
    assert family.name == "ZIBI"
    assert family.parameters == ("mu", "nu")
    
    y = jnp.array([0.0, 1.0, 2.0, 5.0])
    mu = jnp.array([0.5, 0.5, 0.5, 0.5])
    nu_zero = jnp.array([0.3, 0.3, 0.3, 0.3])
    
    pdf = family.pdf(y, mu, nu_zero)
    assert pdf.shape == y.shape
    assert jnp.all(jnp.isfinite(pdf))
    assert jnp.all(pdf >= 0)


def test_zazipf_basic():
    """Test ZAZIPF distribution basic functionality."""
    family = ZAZIPF()
    assert family.name == "ZAZIPF"
    assert family.parameters == ("mu", "nu")
    
    y = jnp.array([0.0, 1.0, 2.0, 5.0])
    mu = jnp.array([2.0, 2.0, 2.0, 2.0])
    nu_zero = jnp.array([0.3, 0.3, 0.3, 0.3])
    
    pdf = family.pdf(y, mu, nu_zero)
    assert pdf.shape == y.shape
    assert jnp.all(jnp.isfinite(pdf))
    assert jnp.all(pdf >= 0)


def test_zanbi_basic():
    """Test ZANBI distribution basic functionality."""
    family = ZANBI()
    assert family.name == "ZANBI"
    assert family.parameters == ("mu", "sigma", "nu")
    
    y = jnp.array([0.0, 1.0, 2.0, 5.0])
    mu = jnp.array([2.0, 2.0, 2.0, 2.0])
    sigma = jnp.array([0.5, 0.5, 0.5, 0.5])
    nu_zero = jnp.array([0.3, 0.3, 0.3, 0.3])
    
    pdf = family.pdf(y, mu, sigma, nu_zero)
    assert pdf.shape == y.shape
    assert jnp.all(jnp.isfinite(pdf))
    assert jnp.all(pdf >= 0)


def test_zabnb_basic():
    """Test ZABNB distribution basic functionality."""
    family = ZABNB()
    assert family.name == "ZABNB"
    assert family.parameters == ("mu", "sigma", "nu", "nu")
    
    y = jnp.array([0.0, 1.0, 2.0, 5.0])
    mu = jnp.array([2.0, 2.0, 2.0, 2.0])
    sigma = jnp.array([0.5, 0.5, 0.5, 0.5])
    nu = jnp.array([1.0, 1.0, 1.0, 1.0])
    nu_zero = jnp.array([0.3, 0.3, 0.3, 0.3])
    
    pdf = family.pdf(y, mu, sigma, nu, nu_zero)
    assert pdf.shape == y.shape
    assert jnp.all(jnp.isfinite(pdf))
    assert jnp.all(pdf >= 0)


def test_zibnb_basic():
    """Test ZIBNB distribution basic functionality."""
    family = ZIBNB()
    assert family.name == "ZIBNB"
    assert family.parameters == ("mu", "sigma", "nu", "nu")
    
    y = jnp.array([0.0, 1.0, 2.0, 5.0])
    mu = jnp.array([2.0, 2.0, 2.0, 2.0])
    sigma = jnp.array([0.5, 0.5, 0.5, 0.5])
    nu = jnp.array([1.0, 1.0, 1.0, 1.0])
    nu_zero = jnp.array([0.3, 0.3, 0.3, 0.3])
    
    pdf = family.pdf(y, mu, sigma, nu, nu_zero)
    assert pdf.shape == y.shape
    assert jnp.all(jnp.isfinite(pdf))
    assert jnp.all(pdf >= 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
