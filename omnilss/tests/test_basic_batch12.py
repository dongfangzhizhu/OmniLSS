"""Tests for batch 12 distributions."""

import jax.numpy as jnp
import pytest
from omnilss import BEo, GEOMo, WEI2, WEI3, PARETO2o, NOF, GAF, RGE


def test_beo_basic():
    """Test BEo distribution."""
    family = BEo()
    assert family.name == "BEo"
    y = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
    mu = jnp.array([2.0, 2.0, 2.0, 2.0, 2.0])
    sigma = jnp.array([3.0, 3.0, 3.0, 3.0, 3.0])
    pdf = family.pdf(y, mu, sigma)
    assert jnp.all(jnp.isfinite(pdf))
    assert jnp.all(pdf >= 0)


def test_geomo_basic():
    """Test GEOMo distribution."""
    family = GEOMo()
    assert family.name == "GEOMo"
    y = jnp.array([0.0, 1.0, 2.0, 5.0, 10.0])
    mu = jnp.array([0.3, 0.3, 0.3, 0.3, 0.3])
    pdf = family.pdf(y, mu)
    assert jnp.all(jnp.isfinite(pdf))
    assert jnp.all(pdf >= 0)


def test_wei2_basic():
    """Test WEI2 distribution."""
    family = WEI2()
    assert family.name == "WEI2"
    y = jnp.array([0.5, 1.0, 2.0, 3.0, 5.0])
    mu = jnp.array([2.0, 2.0, 2.0, 2.0, 2.0])
    sigma = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5])
    pdf = family.pdf(y, mu, sigma)
    assert jnp.all(jnp.isfinite(pdf))
    assert jnp.all(pdf >= 0)


def test_wei3_basic():
    """Test WEI3 distribution."""
    family = WEI3()
    assert family.name == "WEI3"
    y = jnp.array([0.5, 1.0, 2.0, 3.0, 5.0])
    mu = jnp.array([2.0, 2.0, 2.0, 2.0, 2.0])
    sigma = jnp.array([1.5, 1.5, 1.5, 1.5, 1.5])
    pdf = family.pdf(y, mu, sigma)
    assert jnp.all(jnp.isfinite(pdf))
    assert jnp.all(pdf >= 0)


def test_pareto2o_basic():
    """Test PARETO2o distribution."""
    family = PARETO2o()
    assert family.name == "PARETO2o"
    y = jnp.array([0.5, 1.0, 2.0, 3.0, 5.0])
    mu = jnp.array([2.0, 2.0, 2.0, 2.0, 2.0])
    sigma = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
    pdf = family.pdf(y, mu, sigma)
    assert jnp.all(jnp.isfinite(pdf))
    assert jnp.all(pdf >= 0)


def test_nof_basic():
    """Test NOF distribution."""
    family = NOF()
    assert family.name == "NOF"
    y = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    mu = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
    sigma = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
    pdf = family.pdf(y, mu, sigma)
    assert jnp.all(jnp.isfinite(pdf))
    assert jnp.all(pdf >= 0)


def test_gaf_basic():
    """Test GAF distribution."""
    family = GAF()
    assert family.name == "GAF"
    y = jnp.array([0.5, 1.0, 2.0, 3.0, 5.0])
    mu = jnp.array([2.0, 2.0, 2.0, 2.0, 2.0])
    sigma = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5])
    pdf = family.pdf(y, mu, sigma)
    assert jnp.all(jnp.isfinite(pdf))
    assert jnp.all(pdf >= 0)


def test_rge_basic():
    """Test RGE distribution."""
    family = RGE()
    assert family.name == "RGE"
    y = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    mu = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
    sigma = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
    nu = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
    pdf = family.pdf(y, mu, sigma, nu)
    assert jnp.all(jnp.isfinite(pdf))
    assert jnp.all(pdf >= 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
