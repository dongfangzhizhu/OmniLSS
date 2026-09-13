"""Regression tests for the exact NBII discrete DPQR implementation."""

import numpy as np

from omnilss.distribution_registry import resolve


def test_nbii_cdf_matches_sum_of_its_pmf():
    family = resolve("NBII")
    x = np.arange(0.0, 12.0)
    mu, sigma = 3.0, 0.5

    pmf = np.asarray(family.d(x, mu, sigma))
    cdf = np.asarray(family.p(x, mu, sigma))

    np.testing.assert_allclose(cdf, np.cumsum(pmf), rtol=1e-12, atol=1e-12)
    assert np.all(np.diff(cdf) >= 0.0)


def test_nbii_quantile_is_integer_and_satisfies_discrete_definition():
    family = resolve("NBII")
    probabilities = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    mu, sigma = 3.0, 0.5

    quantiles = np.asarray(family.q(probabilities, mu, sigma))
    assert np.all(np.isfinite(quantiles))
    assert np.all(quantiles >= 0.0)
    assert np.all(quantiles == np.floor(quantiles))

    cdf_at_q = np.asarray(family.p(quantiles, mu, sigma))
    cdf_before_q = np.asarray(family.p(quantiles - 1.0, mu, sigma))
    assert np.all(cdf_at_q >= probabilities - 1e-12)
    assert np.all(cdf_before_q < probabilities + 1e-12)