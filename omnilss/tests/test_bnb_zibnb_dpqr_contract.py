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
