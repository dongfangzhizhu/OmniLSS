"""DPQR contract tests for the R-compatible SI (Sichel) family."""

from __future__ import annotations

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from omnilss.distribution_registry import resolve
from omnilss.family_capabilities import ImplementationStatus, implementation_status_for_family


def test_si_is_not_exposed_as_placeholder() -> None:
    assert implementation_status_for_family("SI") is ImplementationStatus.EXPERIMENTAL


def test_si_dpqr_roundtrip_and_boundaries() -> None:
    family = resolve("SI")
    params = (0.5, 0.2, -0.5)
    support = np.arange(0.0, 8.0)
    mass = np.asarray(family.d(support, *params))
    cdf = np.asarray(family.p(support, *params))

    assert np.all(np.isfinite(mass))
    assert np.all(mass >= 0.0)
    assert np.all(np.diff(cdf) >= 0.0)
    assert float(family.p(-1.0, *params)) == 0.0
    assert float(family.p(np.inf, *params)) == 1.0

    probabilities = np.array([0.1, 0.5, 0.9])
    quantiles = np.asarray(family.q(probabilities, *params))
    np.testing.assert_array_equal(np.asarray(family.p(quantiles, *params)) >= probabilities, True)
    np.testing.assert_array_equal(
        family.q(np.log(probabilities), *params, log_p=True), quantiles
    )
    np.testing.assert_array_equal(
        family.q(1.0 - probabilities, *params, lower_tail=False), quantiles
    )

    # Fixed golden values generated with R 4.6.0 / gamlss.dist 6.1.1.
    golden_x = np.array([0.0, 1.0, 2.0, 5.0, 10.0])
    np.testing.assert_allclose(
        family.d(golden_x, *params),
        np.array([0.62050254361206159, 0.28321936677138265, 0.07643648857506404,
                  0.0005205280808552842, 5.393795581766162e-08]),
        rtol=2e-12,
    )
    np.testing.assert_allclose(
        family.p(golden_x, *params),
        np.array([0.62050254361206159, 0.90372191038344429, 0.98015839895850831,
                  0.99989754937112940, 0.99999999002775541]),
        rtol=2e-12,
    )
    np.testing.assert_array_equal(family.q(probabilities, *params), np.array([0.0, 0.0, 1.0]))


def test_si_broadcasts_parameters_and_samples() -> None:
    family = resolve("SI")
    q = np.asarray(family.p(np.array([0.0, 1.0]), np.array([0.5, 1.0]), 0.2, -0.5))
    assert q.shape == (2,)
    sample = np.asarray(family.r(jax.random.PRNGKey(7), 16, 0.5, 0.2, -0.5))
    assert sample.shape == (16,)
    assert np.all(np.isfinite(sample))
    assert np.all(sample >= 0.0)