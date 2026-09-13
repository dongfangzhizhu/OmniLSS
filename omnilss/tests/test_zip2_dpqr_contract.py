"""DPQR contract tests for the R-compatible ZIP2 family."""

from __future__ import annotations

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from omnilss.distribution_registry import resolve
from omnilss.family_capabilities import ImplementationStatus, implementation_status_for_family


def test_zip2_public_contract_and_links() -> None:
    family = resolve("ZIP2")
    assert family.parameters == ("mu", "sigma")
    assert family.links == {"mu": "log", "sigma": "logit"}
    assert implementation_status_for_family("ZIP2") is ImplementationStatus.EXPERIMENTAL
    assert all(getattr(family, name) is not None for name in ("d", "p", "q", "r"))


def test_zip2_matches_r_density_and_cdf_golden() -> None:
    family = resolve("ZIP2")
    x = np.array([-1.0, 0.0, 1.0, 2.0, 5.0, 10.0, 2.5])
    expected_mass = np.array([
        0.0,
        0.319680961824280,
        0.0702891493724301,
        0.125516338165054,
        0.0952960536360042,
        0.0018310616745674,
        0.142749643448611,
    ])
    expected_cdf = np.array([
        0.0,
        0.319680961824280,
        0.389970111196711,
        0.515486449361764,
        0.893621190189429,
        0.999163772612210,
        0.515486449361764,
    ])
    np.testing.assert_allclose(family.d(x, 2.5, 0.3), expected_mass, rtol=2e-12, atol=1e-15)
    np.testing.assert_allclose(family.p(x, 2.5, 0.3), expected_cdf, rtol=2e-12, atol=1e-15)
    np.testing.assert_allclose(
        family.p(np.array([0.0, 1.0, 5.0]), 2.5, 0.3, lower_tail=False),
        np.array([0.680319038175720, 0.610029888803289, 0.106378809810571]),
        rtol=2e-12,
    )
    np.testing.assert_allclose(
        family.p(np.array([0.0, 1.0, 5.0]), 2.5, 0.3, log_p=True),
        np.array([-1.14043177481689, -0.941685180752436, -0.112473318225593]),
        rtol=2e-12,
    )


def test_zip2_quantiles_match_r_and_obey_probability_contract() -> None:
    family = resolve("ZIP2")
    probabilities = np.array([0.01, 0.1, 0.3, 0.31, 0.5, 0.9, 0.99])
    expected = np.array([0.0, 0.0, 0.0, 0.0, 2.0, 6.0, 8.0])
    np.testing.assert_array_equal(family.q(probabilities, 2.5, 0.3), expected)
    np.testing.assert_array_equal(family.q(1.0 - probabilities, 2.5, 0.3, lower_tail=False), expected)
    np.testing.assert_array_equal(family.q(np.log(probabilities), 2.5, 0.3, log_p=True), expected)
    assert float(family.q(0.0, 2.5, 0.3)) == 0.0
    assert np.isposinf(float(family.q(1.0, 2.5, 0.3)))
    assert np.isnan(float(family.q(-0.1, 2.5, 0.3)))
    assert np.isnan(float(family.q(1.1, 2.5, 0.3)))


def test_zip2_broadcasting_invalid_parameters_and_sampling() -> None:
    family = resolve("ZIP2")
    x = np.array([0.0, 1.0, 2.0])
    mu = np.array([1.0, 2.0, 4.0])
    sigma = np.array([0.1, 0.2, 0.4])
    assert np.asarray(family.d(x, mu, sigma)).shape == (3,)
    assert np.asarray(family.p(x, mu, sigma)).shape == (3,)
    assert np.asarray(family.q(np.array([0.2, 0.5, 0.9]), mu, sigma)).shape == (3,)
    for bad_mu, bad_sigma in ((0.0, 0.3), (-1.0, 0.3), (2.5, 0.0), (2.5, 1.0)):
        assert np.isnan(float(family.d(1.0, bad_mu, bad_sigma)))
        assert np.isnan(float(family.p(1.0, bad_mu, bad_sigma)))
        assert np.isnan(float(family.q(0.5, bad_mu, bad_sigma)))

    sample = np.asarray(family.r(jax.random.PRNGKey(42), 20_000, 2.5, 0.3))
    assert sample.shape == (20_000,)
    assert np.all(np.isfinite(sample)) and np.all(sample == np.floor(sample))
    assert abs(float(sample.mean()) - 2.5) < 0.08
    expected_zero_probability = 0.3 + 0.7 * np.exp(-2.5 / 0.7)
    assert abs(float(np.mean(sample == 0.0)) - expected_zero_probability) < 0.015