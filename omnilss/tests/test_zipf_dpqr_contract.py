"""DPQR contract tests for the R-compatible ZIPF family."""

import jax
import numpy as np

from omnilss.distribution_registry import resolve


def test_zipf_public_contract() -> None:
    family = resolve("ZIPF")
    assert family.name == "ZIPF"
    assert family.parameters == ("mu",)
    assert family.links == {"mu": "log"}
    assert family.type == "Discrete"
    assert all(getattr(family, name) is not None for name in ("d", "p", "q", "r"))


def test_zipf_r_support_and_r_parameterization() -> None:
    family = resolve("ZIPF")
    x = np.array([0.0, 0.5, 1.0, 2.0, 2.5, 10.0])
    expected = np.array([0.0, 0.0, 1.0 / (np.pi**2 / 6.0), 1.0 / (4.0 * np.pi**2 / 6.0),
                         2.5 ** -2.0 / (np.pi**2 / 6.0), 10.0 ** -2.0 / (np.pi**2 / 6.0)])
    np.testing.assert_allclose(family.d(x, 1.0), expected, rtol=2e-6, atol=1e-12)
    np.testing.assert_allclose(
        family.p(np.array([0.5, 1.0, 2.0, 2.5]), 1.0),
        np.array([0.0, 1.0, 1.25, 1.25]) / (np.pi**2 / 6.0),
        rtol=2e-6,
    )


def test_zipf_quantiles_tails_broadcasting_and_sampling() -> None:
    family = resolve("ZIPF")
    probabilities = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    np.testing.assert_array_equal(family.q(probabilities, 1.0), [1.0, 1.0, 1.0, 6.0, np.inf])
    np.testing.assert_array_equal(family.q(np.log(np.array([0.1, 0.5])), 1.0, log_p=True), [1.0, 1.0])
    assert np.isnan(float(family.q(-0.1, 1.0)))
    assert np.asarray(family.d([1, 2, 3], [0.5, 1.0, 2.0])).shape == (3,)
    sample = np.asarray(family.r(jax.random.PRNGKey(42), 2_000, 2.0))
    assert sample.shape == (2_000,)
    assert np.all(np.isfinite(sample)) and np.all(sample >= 1.0)