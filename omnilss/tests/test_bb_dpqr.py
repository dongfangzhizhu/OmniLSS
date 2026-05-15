"""Tests for BB (Beta-Binomial) distribution d/p/q/r functions."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jrandom

from omnilss.dpqr_functions import dBB, pBB, qBB, rBB


def test_bb_pmf_is_finite_positive_and_normalized() -> None:
    mu = 0.5
    sigma = 0.5
    bd = 10
    x = jnp.arange(0, bd + 1)

    pmf = dBB(x, mu, sigma, bd)

    assert jnp.all(jnp.isfinite(pmf))
    assert jnp.all(pmf >= 0)
    assert jnp.isclose(jnp.sum(pmf), 1.0, atol=1e-6)


def test_bb_cdf_is_bounded_monotone_and_reaches_one() -> None:
    mu = 0.5
    sigma = 0.5
    bd = 10
    q = jnp.array([0, 2, 5, 8, 10])

    probs = pBB(q, mu, sigma, bd)

    assert jnp.all((probs >= 0) & (probs <= 1))
    assert jnp.all(jnp.diff(probs) >= 0)
    assert jnp.isclose(probs[-1], 1.0, atol=1e-6)


def test_bb_quantile_is_monotone_and_consistent_with_cdf_grid() -> None:
    mu = 0.5
    sigma = 0.5
    bd = 10
    p = jnp.array([0.1, 0.25, 0.5, 0.75, 0.9])
    x = jnp.array([0, 2, 4, 6, 8, 10])

    quantiles = qBB(p, mu, sigma, bd)
    x_back = qBB(pBB(x, mu, sigma, bd), mu, sigma, bd)

    assert jnp.all(jnp.diff(quantiles) >= 0)
    assert jnp.allclose(x_back, x)


def test_bb_random_generation_respects_support() -> None:
    mu = 0.5
    sigma = 0.5
    bd = 10
    samples = rBB(jrandom.PRNGKey(42), 1000, mu, sigma, bd)

    assert samples.shape == (1000,)
    assert jnp.all((samples >= 0) & (samples <= bd))
