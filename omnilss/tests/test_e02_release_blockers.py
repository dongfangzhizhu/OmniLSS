"""Release-blocker regressions from the v0.1 execution plan (E02).

These pin the numerical-safety contracts that must hold before any release:

* link transforms stay finite at extreme linear predictors;
* the Normal family's derivatives stay finite at sigma = 0;
* a non-finite Hessian is reported, never silently replaced by a finite one.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from omnilss.algorithms.stabilized_hessian import stabilize_hessian
from omnilss.core.links import LogLink, LogitLink, SoftplusLink
from omnilss.distributions import NO


def test_links_are_finite_at_extreme_eta():
    eta = jnp.array([-1000.0, -100.0, 0.0, 100.0, 1000.0])
    for link in (LogLink(), LogitLink(), SoftplusLink()):
        assert np.isfinite(np.asarray(link.inverse(eta))).all()
        assert np.isfinite(np.asarray(link.derivative(eta))).all()


def test_softplus_transform_is_finite_for_large_values():
    out = np.asarray(SoftplusLink().transform(jnp.array([1e-9, 1.0, 800.0])))
    assert np.isfinite(out).all()
    assert out[-1] == pytest.approx(800.0, rel=1e-12)


def test_no_derivatives_are_finite_at_zero_sigma():
    family = NO()
    y = jnp.array([1.0, -2.0])
    mu = jnp.zeros(2)
    sigma = jnp.zeros(2)

    assert np.isfinite(np.asarray(family.g_dev_inc(y, mu, sigma))).all()
    for score in family.score_functions.values():
        assert np.isfinite(np.asarray(score(y, mu, sigma))).all()
    for hessian in family.hessian_functions.values():
        assert np.isfinite(np.asarray(hessian(y, mu, sigma))).all()


def test_non_finite_hessian_is_reported_not_masked():
    H = np.array([[np.nan, 0.0], [0.0, 1.0]], dtype=np.float64)
    result = stabilize_hessian(H)

    assert result.is_finite is False
    assert result.was_regularized is False
    assert result.lambda_value == 0.0
    assert np.isinf(result.condition_number)
    # The NaN must survive: replacing it with 0.0 would invent a Hessian that
    # the optimizer never saw.
    assert np.isnan(result.matrix[0, 0])


def test_finite_hessian_keeps_previous_behaviour():
    H = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    result = stabilize_hessian(H, base_lambda=1e-6)

    assert result.is_finite is True
    assert result.was_regularized is True
    assert np.isfinite(result.condition_number)
    _ = np.linalg.cholesky(result.matrix)
