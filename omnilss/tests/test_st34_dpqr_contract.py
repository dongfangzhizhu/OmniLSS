"""DPQR contracts for the R-compatible ST3 and ST4 families."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from omnilss.distribution_registry import resolve


@pytest.mark.parametrize("name", ["ST3", "ST4"])
def test_quantile_boundaries_and_invalid_probabilities(name: str) -> None:
    family = resolve(name)
    p = jnp.array([0.0, 1e-8, 0.25, 0.75, 1.0, -0.1, 1.1, np.nan, np.inf])
    q = np.asarray(family.q(p, 0.2, 1.3, 0.7, 5.0))

    assert np.isneginf(q[0])
    assert np.isposinf(q[4])
    assert np.all(np.isnan(q[5:]))
    assert np.all(np.diff(q[1:5]) > 0)


@pytest.mark.parametrize("name", ["ST3", "ST4"])
def test_quantile_tail_and_log_probability_semantics(name: str) -> None:
    family = resolve(name)
    x = np.array([-1.5, 0.0, 2.0])
    p = np.asarray(family.p(x, 0.2, 1.3, 0.7, 5.0))

    np.testing.assert_allclose(
        family.q(1.0 - p, 0.2, 1.3, 0.7, 5.0, lower_tail=False), x, rtol=2e-5, atol=2e-5
    )
    np.testing.assert_allclose(
        family.q(np.log(p), 0.2, 1.3, 0.7, 5.0, log_p=True), x, rtol=2e-5, atol=2e-5
    )
    np.testing.assert_allclose(
        family.p(x, 0.2, 1.3, 0.7, 5.0, log_p=True), np.log(p), rtol=2e-12, atol=2e-12
    )


def test_st3_uses_distinct_two_piece_parameterization() -> None:
    st3 = resolve("ST3")
    x = jnp.array([-2.0, -0.5, 0.5, 2.0])
    # ST3 must use tau for the t kernel on both sides and reciprocal nu scales;
    # changing tau therefore changes both tails, unlike ST3C.
    a = np.asarray(st3.d(x, 0.0, 1.0, 0.5, 3.0))
    b = np.asarray(st3.d(x, 0.0, 1.0, 0.5, 30.0))
    assert not np.allclose(a, b)
