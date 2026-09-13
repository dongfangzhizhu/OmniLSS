"""DPQR boundary and probability-contract tests for SEP2, SEP3, and SEP4."""

from __future__ import annotations

import numpy as np

from omnilss.distribution_registry import resolve


def test_sep234_quantile_boundaries_and_invalid_probabilities() -> None:
    configurations = {
        "SEP2": (0.0, 1.0, 0.0, 2.0),
        "SEP3": (0.0, 1.0, 2.0, 2.0),
        "SEP4": (0.0, 1.0, 2.0, 2.0),
    }

    for name, params in configurations.items():
        family = resolve(name)
        values = np.asarray(family.q(np.array([0.0, 1.0, -0.1, 1.1]), *params))
        assert np.isneginf(values[0])
        assert np.isposinf(values[1])
        assert np.isnan(values[2:]).all()


def test_sep234_quantile_probability_options_preserve_contract() -> None:
    configurations = {
        "SEP2": (0.2, 1.3, -1.1, 1.7),
        "SEP3": (0.2, 1.3, 1.1, 1.7),
        "SEP4": (0.2, 1.3, 1.1, 1.7),
    }
    probabilities = np.array([0.1, 0.5, 0.9])

    for name, params in configurations.items():
        family = resolve(name)
        direct = np.asarray(family.q(probabilities, *params), dtype=np.float64)
        upper = np.asarray(
            family.q(probabilities, *params, lower_tail=False), dtype=np.float64
        )
        logged = np.asarray(
            family.q(np.log(probabilities), *params, log_p=True), dtype=np.float64
        )
        np.testing.assert_allclose(logged, direct, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(
            upper,
            np.asarray(family.q(1.0 - probabilities, *params), dtype=np.float64),
            rtol=1e-10,
            atol=1e-10,
        )
