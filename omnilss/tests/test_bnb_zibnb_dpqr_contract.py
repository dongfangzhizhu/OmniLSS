"""Regression contracts for BNB and ZIBNB DPQR and random generation."""

from __future__ import annotations

import jax
import numpy as np

from omnilss.distribution_registry import resolve


def test_bnb_dpqr_boundaries_and_probability_options() -> None:
    family = resolve("BNB")
    params = (2.5, 0.4, 1.5)
    x = np.arange(8.0)
    cdf = np.asarray(family.p(x, *params))
    assert np.all(np.diff(cdf) >= 0.0)
    np.testing.assert_allclose(family.p(x, *params, lower_tail=False), 1.0 - cdf)
    np.testing.assert_allclose(
        family.p(x, *params, log_p=True), np.log(cdf), rtol=1e-6, atol=1e-7
    )
    np.testing.assert_array_equal(family.q(np.array([0.0, 1.0]), *params), np.array([0.0, np.inf]))


def test_bnb_parameter_broadcasting_and_sampling_shape() -> None:
    family = resolve("BNB")
    mu = np.array([1.0, 2.0])
    values = np.asarray(family.p(np.arange(4.0)[:, None], mu, 0.5, 1.2))
    assert values.shape == (4, 2)
    samples = np.asarray(family.r(jax.random.PRNGKey(0), 7, mu, 0.5, 1.2))
    assert samples.shape == (7, 2)
    assert np.all(samples >= 0)


def test_bnb_cdf_beyond_static_support() -> None:
    family = resolve("BNB")
    params = (2500.0, 1.5, 0.8)
    support = np.arange(12001.0)
    expected = np.cumsum(np.asarray(family.d(support, *params)))
    actual = jax.jit(lambda q: family.p(q, *params))(np.array([10001.0, 12000.0]))
    np.testing.assert_allclose(actual, expected[[10001, 12000]], rtol=1e-10)
    assert actual[1] > actual[0]


def test_bnb_quantile_search_and_unresolved_tail() -> None:
    family = resolve("BNB")
    params = (2500.0, 1.5, 0.8)
    probability = np.array([1e-12, 0.5, 0.75])
    quantiles = np.asarray(jax.jit(lambda p: family.q(p, *params))(probability))
    assert quantiles[0] == 0.0
    assert quantiles[2] > 1000.0
    assert np.all(np.asarray(family.p(quantiles, *params)) >= probability)
    assert np.all(np.asarray(family.p(quantiles - 1, *params)) < probability)
    assert np.isnan(family.q(1.0 - 1e-12, *params))


def test_bnb_cdf_special_values() -> None:
    family = resolve("BNB")
    actual = np.asarray(family.p(np.array([-np.inf, np.inf, np.nan])))
    np.testing.assert_allclose(actual, [0.0, 1.0, np.nan])
    assert np.asarray(family.p(np.array([]))).shape == (0,)


def test_bnb_dynamic_quantiles_match_r_pmf_reference() -> None:
    # R gamlss.dist: first count with cumsum(dBNB(0:300000, ...)) >= p.
    # Using a PMF reference avoids the very slow repeated CDF calls in R qBNB.
    family = resolve("BNB")
    probability = np.array([0.95, 0.99, 0.999, 0.9])
    mu = np.array([2500.0, 2500.0, 2500.0, 20000.0])
    sigma = np.array([1.5, 1.5, 1.5, 0.4])
    nu = np.array([0.8, 0.8, 0.8, 1.5])
    expected = np.array([8211.0, 23956.0, 99732.0, 49362.0])
    np.testing.assert_array_equal(family.q(probability, mu, sigma, nu), expected)
    np.testing.assert_array_equal(jax.jit(family.q)(probability, mu, sigma, nu), expected)
    np.testing.assert_array_equal(
        family.q(np.log1p(-probability), mu, sigma, nu, lower_tail=False, log_p=True),
        expected,
    )


def test_bnb_dynamic_search_empty_invalid_and_block_edges() -> None:
    family = resolve("BNB")
    assert jax.jit(family.q)(np.array([])).shape == (0,)
    np.testing.assert_allclose(
        jax.jit(family.q)(np.array([0, 1, -0.1, 1.1, np.nan])),
        [0, np.inf, np.nan, np.nan, np.nan],
    )
    # Targets strictly between successive CDF values avoid rounding ambiguity.
    counts = np.array([0.0, 255.0, 256.0, 511.0, 512.0])
    params = (500.0, 0.5, 1.0)
    probability = (family.p(counts, *params) + family.p(counts - 1, *params)) / 2
    np.testing.assert_array_equal(jax.jit(family.q)(probability, *params), counts)


def test_zibnb_dynamic_quantile_inherits_base_search() -> None:
    family = resolve("ZIBNB")
    probability = 0.3 + 0.7 * np.array([0.99, 0.999])
    actual = jax.jit(family.q)(probability, 2500.0, 1.5, 0.8, 0.3)
    np.testing.assert_array_equal(actual, [23956.0, 99732.0])


def test_zibnb_zero_mass_boundaries_and_sampling_shape() -> None:
    family = resolve("ZIBNB")
    params = (2.5, 0.4, 1.5)
    for tau in (0.0, 1.0):
        cdf0 = float(np.asarray(family.p(0.0, *params[:2], params[2], tau)))
        base_zero = float(np.asarray(resolve("BNB").p(0.0, *params)))
        expected = tau + (1.0 - tau) * base_zero
        np.testing.assert_allclose(cdf0, expected, atol=1e-7)
        sample = np.asarray(family.r(jax.random.PRNGKey(1), 16, *params[:2], params[2], tau))
        assert sample.shape == (16,)
        if tau == 1.0:
            assert np.all(sample == 0)
