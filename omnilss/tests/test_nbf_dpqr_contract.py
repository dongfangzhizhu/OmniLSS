"""DPQR contract tests for the R-compatible NBF family."""

from __future__ import annotations

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from omnilss.distribution_registry import resolve
from omnilss.family_capabilities import ImplementationStatus, implementation_status_for_family


def test_nbf_is_experimental_and_has_complete_dpqr() -> None:
    assert implementation_status_for_family("NBF") is ImplementationStatus.EXPERIMENTAL
    family = resolve("NBF")
    assert all(getattr(family, name) is not None for name in ("d", "p", "q", "r"))


def test_nbf_pmf_cdf_and_quantile_contract() -> None:
    family = resolve("NBF")
    params = (2.5, 0.4, 1.5)
    support = np.arange(0.0, 80.0)
    mass = np.asarray(family.d(support, *params))
    cdf = np.asarray(family.p(support, *params))

    assert np.all(np.isfinite(mass))
    assert np.all(mass >= 0.0)
    assert np.isclose(mass.sum(), 1.0, atol=1e-12)
    assert np.all(np.diff(cdf) >= 0.0)
    probabilities = np.array([0.1, 0.5, 0.9])
    quantiles = np.asarray(family.q(probabilities, *params))
    assert np.all(np.asarray(family.p(quantiles, *params)) >= probabilities)
    np.testing.assert_array_equal(family.q(np.log(probabilities), *params, log_p=True), quantiles)
    np.testing.assert_array_equal(family.q(1.0 - probabilities, *params, lower_tail=False), quantiles)

    np.testing.assert_allclose(
        family.d(np.array([0.0, 1.0, 2.0, 5.0, 10.0]), *params),
        np.array([0.14410221379933413, 0.22068321459852416, 0.21173022567265276,
                  0.06754723123692699, 0.002946056466219639]),
        rtol=2e-13,
    )
    np.testing.assert_allclose(
        family.p(np.array([0.0, 1.0, 2.0, 5.0, 10.0]), *params),
        np.array([0.14410221379933419, 0.36478542839785821, 0.57651565407051086,
                  0.91644717798991771, 0.99723231568209025]),
        rtol=2e-13,
    )
    np.testing.assert_array_equal(quantiles, np.array([0.0, 2.0, 5.0]))


def test_nbf_boundaries_tails_broadcast_and_sampling() -> None:
    family = resolve("NBF")
    params = (2.5, 0.4, 1.5)
    assert float(family.p(-1.0, *params)) == 0.0
    assert float(family.p(np.inf, *params)) == 1.0
    assert float(family.q(0.0, *params)) == 0.0
    assert np.isposinf(float(family.q(1.0, *params)))
    assert np.isclose(float(family.p(4.0, *params, lower_tail=False)), 1.0 - float(family.p(4.0, *params)))
    assert np.isfinite(float(family.p(4.0, *params, lower_tail=False, log_p=True)))
    broadcast = np.asarray(family.p(np.array([0.0, 1.0]), np.array([1.0, 2.0]), 0.4, 1.5))
    assert broadcast.shape == (2,)
    sample = np.asarray(family.r(jax.random.PRNGKey(7), 32, *params))
    assert sample.shape == (32,)
    assert np.all(np.isfinite(sample))
    assert np.all(sample >= 0.0)
    assert np.all(sample == np.floor(sample))

    # R switches to a Poisson kernel when sigma * mu**(nu - 2) < 1e-4.
    np.testing.assert_allclose(
        family.d(np.arange(5.0), 2.5, 1e-6, 2.0),
        np.array([0.0820849986238988, 0.205212496559747, 0.256515620699684,
                  0.213763017249736, 0.133601885781085]),
        rtol=2e-13,
    )