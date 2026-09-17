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


def test_bb_cdf_and_quantile_broadcast_each_parameter() -> None:
    q = jnp.array([2.0, 5.0, 8.0])
    mu = jnp.array([0.2, 0.5, 0.8])
    sigma = jnp.array([0.2, 0.5, 0.8])
    bd = jnp.array([5.0, 10.0, 15.0])
    cdf = pBB(q, mu, sigma, bd)
    quantiles = qBB(cdf, mu, sigma, bd)
    assert cdf.shape == q.shape
    assert jnp.all(quantiles == q)


def test_bb_quantile_probability_boundaries_and_log_upper_tail() -> None:
    p = jnp.array([0.0, 0.2, 0.8, 1.0])
    expected = qBB(p, 0.4, 0.3, 10)
    assert expected[0] == 0
    assert expected[-1] == 10
    upper_log = pBB(3, 0.4, 0.3, 10, lower_tail=False, log_p=True)
    assert jnp.isclose(jnp.exp(upper_log), pBB(3, 0.4, 0.3, 10, lower_tail=False))


def test_bb_sampling_does_not_reuse_one_binomial_random_draw() -> None:
    samples = rBB(jrandom.PRNGKey(123), 2000, 0.5, 0.5, 10)
    # A repeated key in the old implementation yielded an extreme duplicate
    # rate; independent vectorized draws should produce varied counts.
    assert jnp.unique(samples).size > 5
