"""Numeric score / hessian for truncated families.

R builds a truncated family with dldm = NULL: gamlssML() fits those families by
direct numerical maximisation (nlminb), never through an analytic score.  The
IRLS engine here expects a score, so trun() synthesises one with gamlss's own
numeric.deriv convention -- a forward difference of the truncated log-density
with delta 0.001 -- plus the -dld^2 observed-information second derivative that
gamlss uses for numerically differentiated families.

Known gap (NOT asserted here): the fit path of gamlss_ml() does not read these
callables (it differentiates through the JAX family builder, which is keyed on
the family name and therefore falls back to the base family).  The fitted values
therefore do not change; see the work log for the evidence table.
"""
from __future__ import annotations

import numpy as np
import pytest

from omnilss.gamlss_tr import trun

_Y = np.array([2.5, 3.2, 4.1, 5.0, 6.3, 7.7, 9.1])


def _family():
    return trun(par=2.0, family="WEI3", type="left")


def test_every_parameter_gets_a_score_and_a_hessian() -> None:
    family = _family()
    assert set(family.score_functions) == set(family.parameters)
    assert set(family.hessian_functions) == set(family.parameters)
    for name in family.parameters:
        assert callable(family.score_functions[name])
        assert callable(family.hessian_functions[name])


def test_mu_score_is_the_derivative_of_the_truncated_log_density() -> None:
    family = _family()
    mu = np.full(_Y.shape, 5.0)
    sigma = np.full(_Y.shape, 2.5)
    score = np.asarray(family.score_functions["mu"](y=_Y, mu=mu, sigma=sigma))
    step = 1e-6
    centre = np.asarray(family.d(_Y, log=True, mu=mu, sigma=sigma))
    upper = np.asarray(family.d(_Y, log=True, mu=mu + step, sigma=sigma))
    lower = np.asarray(family.d(_Y, log=True, mu=mu - step, sigma=sigma))
    central = (upper - lower) / (2.0 * step)
    # The family score uses a 1e-3 forward difference, so it agrees with the
    # central difference to O(step-of-the-score).
    np.testing.assert_allclose(score, central, rtol=1e-2, atol=1e-6)


def test_sigma_score_is_the_derivative_of_the_truncated_log_density() -> None:
    family = _family()
    mu = np.full(_Y.shape, 5.0)
    sigma = np.full(_Y.shape, 2.5)
    score = np.asarray(family.score_functions["sigma"](y=_Y, mu=mu, sigma=sigma))
    step = 1e-6
    centre = np.asarray(family.d(_Y, log=True, mu=mu, sigma=sigma))
    upper = np.asarray(family.d(_Y, log=True, mu=mu, sigma=sigma + step))
    lower = np.asarray(family.d(_Y, log=True, mu=mu, sigma=sigma - step))
    central = (upper - lower) / (2.0 * step)
    np.testing.assert_allclose(score, central, rtol=1e-2, atol=1e-6)


def test_the_score_differs_from_the_untruncated_one() -> None:
    """The whole point: the base analytic score is the wrong equation set."""

    from omnilss.distribution_registry import resolve

    family = _family()
    base = resolve("WEI3")
    mu = np.full(_Y.shape, 5.0)
    sigma = np.full(_Y.shape, 2.5)
    truncated = np.asarray(family.score_functions["mu"](y=_Y, mu=mu, sigma=sigma))
    untruncated = np.asarray(base.score_functions["mu"](y=_Y, mu=mu, sigma=sigma))
    assert not np.allclose(truncated, untruncated, rtol=1e-6, atol=1e-8)


def test_hessian_is_minus_the_squared_score() -> None:
    family = _family()
    mu = np.full(_Y.shape, 5.0)
    sigma = np.full(_Y.shape, 2.5)
    for name, value in (("mu", mu), ("sigma", sigma)):
        kwargs = {"y": _Y, "mu": mu, "sigma": sigma}
        score = np.asarray(family.score_functions[name](**kwargs))
        hessian = np.asarray(family.hessian_functions[name](**kwargs))
        np.testing.assert_allclose(hessian, -(score**2), rtol=1e-14)


def test_right_truncation_and_scalar_parameters_work() -> None:
    # Right truncation at 8: 9.1 lies outside the support, so drop it (the
    # density there is 0 and its log is -inf, which is the correct answer).
    inside = _Y[_Y < 8.0]
    family = trun(par=8.0, family="WEI3", type="right")
    mu = np.full(inside.shape, 5.0)
    sigma = np.full(inside.shape, 2.5)
    score = np.asarray(family.score_functions["mu"](y=inside, mu=mu, sigma=sigma))
    assert score.shape == inside.shape
    assert np.all(np.isfinite(score))
    # Scalar parameters (the intercept-only fit passes length-1 values).
    scalar = np.asarray(family.score_functions["mu"](y=inside, mu=5.0, sigma=2.5))
    assert scalar.shape == inside.shape