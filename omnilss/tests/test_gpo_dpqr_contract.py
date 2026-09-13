"""DPQR contract tests for the R-compatible GPO family."""

from __future__ import annotations

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from omnilss.distribution_registry import resolve
from omnilss.family_capabilities import ImplementationStatus, implementation_status_for_family


def test_gpo_is_experimental_and_has_complete_dpqr() -> None:
    assert implementation_status_for_family("GPO") is ImplementationStatus.EXPERIMENTAL
    family = resolve("GPO")
    assert all(getattr(family, name) is not None for name in ("d", "p", "q", "r"))


def test_gpo_matches_r_golden_and_roundtrips() -> None:
    family = resolve("GPO")
    params = (2.5, 0.4)
    x = np.array([0.0, 1.0, 2.0, 5.0, 10.0])
    np.testing.assert_allclose(
        family.d(x, *params),
        np.array([0.28650479686019009, 0.21721742931305640, 0.14821765954012170,
                  0.04844506363771409, 0.009676685454776927]),
        rtol=2e-12,
    )
    np.testing.assert_allclose(
        family.p(x, *params),
        np.array([0.28650479686019009, 0.50372222617324647, 0.65193988571336825,
                  0.87042993136807378, 0.97007454710082031]),
        rtol=2e-12,
    )
    probabilities = np.array([0.1, 0.5, 0.9])
    quantiles = np.asarray(family.q(probabilities, *params))
    np.testing.assert_array_equal(quantiles, np.array([0.0, 1.0, 6.0]))
    np.testing.assert_array_equal(family.q(np.log(probabilities), *params, log_p=True), quantiles)
    np.testing.assert_array_equal(family.q(1.0 - probabilities, *params, lower_tail=False), quantiles)


def test_gpo_boundaries_broadcast_poisson_branch_and_sampling() -> None:
    family = resolve("GPO")
    params = (2.5, 0.4)
    assert float(family.p(-1.0, *params)) == 0.0
    assert float(family.p(np.inf, *params)) == 1.0
    assert float(family.q(0.0, *params)) == 0.0
    assert np.isposinf(float(family.q(1.0, *params)))
    assert np.isclose(float(family.p(4.0, *params, lower_tail=False)), 1.0 - float(family.p(4.0, *params)))
    assert np.isfinite(float(family.p(4.0, *params, lower_tail=False, log_p=True)))
    assert np.asarray(family.p(np.array([0.0, 1.0]), np.array([1.0, 2.0]), 0.4)).shape == (2,)
    np.testing.assert_allclose(
        family.d(np.arange(5.0), 2.5, 1e-6),
        np.array([0.0820849986238988, 0.205212496559747, 0.256515620699684,
                  0.213763017249736, 0.133601885781085]),
        rtol=2e-12,
    )
    sample = np.asarray(family.r(jax.random.PRNGKey(7), 32, *params))
    assert sample.shape == (32,)
    assert np.all(np.isfinite(sample)) and np.all(sample == np.floor(sample))