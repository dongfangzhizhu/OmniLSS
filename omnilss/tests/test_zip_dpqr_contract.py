"""DPQR contract tests for the R-compatible ZIP family."""

import jax
import math
import numpy as np

jax.config.update("jax_enable_x64", True)

from omnilss.distribution_registry import resolve
from omnilss.family_capabilities import ImplementationStatus, implementation_status_for_family


def test_zip_public_contract() -> None:
    family = resolve("ZIP")
    assert family.parameters == ("mu", "sigma")
    assert family.links == {"mu": "log", "sigma": "logit"}
    assert implementation_status_for_family("ZIP") is ImplementationStatus.EXPERIMENTAL
    assert all(getattr(family, name) is not None for name in ("d", "p", "q", "r"))


def test_zip_matches_r_mixture() -> None:
    family = resolve("ZIP")
    x = np.array([-1.0, 0.0, 1.0, 2.0, 5.0, 10.0, 2.5])
    poisson_mass = np.exp(-2.5) * np.array([1.0, 2.5, 2.5**2 / 2, 2.5**5 / 120, 2.5**10 / 3628800])
    fractional_mass = np.exp(-2.5) * 2.5**2.5 / np.exp(math.lgamma(3.5))
    mass = np.array([0.0, 0.3 + 0.7 * poisson_mass[0], *list(0.7 * poisson_mass[1:]), 0.7 * fractional_mass])
    cdf = np.array([
        0.0,
        0.3 + 0.7 * poisson_mass[0],
        0.3 + 0.7 * (poisson_mass[0] + poisson_mass[1]),
        0.3 + 0.7 * (poisson_mass[0] + poisson_mass[1] + poisson_mass[2]),
    ])
    np.testing.assert_allclose(family.d(x, 2.5, 0.3), mass, rtol=2e-12, atol=1e-15)
    np.testing.assert_allclose(family.p(np.array([-1.0, 0.0, 1.0, 2.0]), 2.5, 0.3), cdf[:4], rtol=2e-12)


def test_zip_quantiles_tails_broadcasting_and_sampling() -> None:
    family = resolve("ZIP")
    probabilities = np.array([0.01, 0.3, 0.31, 0.5, 0.9, 0.99])
    expected = np.array([0.0, 0.0, 0.0, 1.0, 4.0, 6.0])
    np.testing.assert_array_equal(family.q(probabilities, 2.5, 0.3), expected)
    np.testing.assert_array_equal(family.q(np.log(probabilities), 2.5, 0.3, log_p=True), expected)
    assert np.isposinf(float(family.q(1.0, 2.5, 0.3)))
    assert np.isnan(float(family.q(-0.1, 2.5, 0.3)))
    assert np.asarray(family.d([0, 1, 2], [1, 2, 4], [0.1, 0.2, 0.4])).shape == (3,)
    sample = np.asarray(family.r(jax.random.PRNGKey(42), 20_000, 2.5, 0.3))
    assert sample.shape == (20_000,)
    assert np.all(np.isfinite(sample)) and np.all(sample == np.floor(sample))
    assert abs(float(sample.mean()) - 1.75) < 0.08
