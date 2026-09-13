"""Numerical DPQR contract tests for SEP1."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from omnilss.distribution_registry import resolve


def test_sep1_cdf_and_quantile_are_monotone_and_inverse() -> None:
    family = resolve("SEP1")
    params = (0.4, 1.3, -1.1, 1.7)
    p_grid = np.linspace(0.02, 0.98, 9)
    q_values = np.asarray(family.q(p_grid, *params), dtype=np.float64)
    assert np.all(np.diff(q_values) > 0.0)
    np.testing.assert_allclose(
        np.asarray(family.p(q_values, *params), dtype=np.float64),
        p_grid, rtol=2e-6, atol=2e-7,
    )


def test_sep1_tail_and_log_probability_options_are_consistent() -> None:
    family = resolve("SEP1")
    params = (0.0, 1.0, 0.8, 2.5)
    x = jnp.asarray([-8.0, -1.0, 0.0, 1.0, 8.0])
    lower = np.asarray(family.p(x, *params), dtype=np.float64)
    upper = np.asarray(family.p(x, *params, lower_tail=False), dtype=np.float64)
    np.testing.assert_allclose(lower + upper, 1.0, rtol=3e-11, atol=3e-11)
    np.testing.assert_allclose(
        np.exp(np.asarray(family.p(x, *params, lower_tail=False, log_p=True))),
        upper, rtol=1e-12, atol=1e-15,
    )


def test_sep1_quantile_boundary_and_invalid_probability_contract() -> None:
    family = resolve("SEP1")
    params = (0.0, 1.0, 0.0, 2.0)
    boundary = np.asarray(family.q(np.array([0.0, 1.0]), *params))
    assert np.isneginf(boundary[0])
    assert np.isposinf(boundary[1])
    invalid = np.asarray(family.q(np.array([-0.1, 1.1]), *params))
    assert np.isnan(invalid).all()