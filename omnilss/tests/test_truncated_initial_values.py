"""Truncated families start from their base family (start-value mapping).

R builds a brand-new family object in gen.trun() and leaves mu.initial /
sigma.initial empty, so gamlssML falls back to an internal default.  Keying the
start-value tables on the truncated name ("WEI3tr") matched no branch at all and
fell into the generic target = y fallback, which a log link turns into exp(y).
These tests pin the mapping: the truncated family must start exactly where its
base family starts.
"""
from __future__ import annotations

import numpy as np
import pytest

from omnilss._fitting_init import (
    _initial_mu_target,
    _initial_parameter_value,
    _initial_sigma,
    _start_family,
)
from omnilss.fitting import gamlss_ml
from omnilss.gamlss_tr import trun

_Y = np.array([2.5, 3.2, 4.1, 5.0, 6.3, 7.7, 9.1])


def _truncated():
    return trun(par=2.0, family="WEI3", type="left")


def test_start_family_keeps_the_truncation_but_renames_it() -> None:
    truncated = _truncated()
    view = _start_family(truncated)
    assert view.name == "WEI3"
    # Everything else still comes from the truncated family object.
    assert view.truncation_type == "left"
    assert view.truncation_parameters == (2.0,)
    assert view.base_name == "WEI3"
    assert view.support() == (2.0, None)


def test_start_family_leaves_normal_families_untouched() -> None:
    from omnilss.distribution_registry import resolve

    base = resolve("WEI3")
    assert _start_family(base) is base
    assert _start_family(_truncated()).name == "WEI3"


def test_truncated_mu_start_is_the_base_family_start() -> None:
    from omnilss.distribution_registry import resolve

    base = resolve("WEI3")
    w = np.ones(_Y.shape)
    truncated_start = _initial_mu_target(_truncated(), _Y, w)
    base_start = _initial_mu_target(base, _Y, w)
    np.testing.assert_allclose(truncated_start, base_start, rtol=0, atol=0)
    # WEI3 mu.initial is log(y + 0.01): a log-scale start, not exp(y).
    np.testing.assert_allclose(truncated_start, np.log(_Y + 0.01), rtol=1e-12)
    assert float(np.max(truncated_start)) < 3.0


def test_truncated_sigma_and_nu_starts_come_from_the_base_family() -> None:
    from omnilss.distribution_registry import resolve

    base = resolve("WEI3")
    w = np.ones(_Y.shape)
    mu = _initial_mu_target(base, _Y, w)
    assert _initial_sigma(_truncated(), _Y, mu, w) == pytest.approx(
        _initial_sigma(base, _Y, mu, w), rel=0, abs=0
    )
    assert _initial_parameter_value(_truncated(), "sigma", _Y, mu, w) == pytest.approx(
        _initial_parameter_value(base, "sigma", _Y, mu, w), rel=0, abs=0
    )


def test_the_truncated_fit_stays_on_the_right_scale() -> None:
    """Before the mapping the fit ran off to mu ~ 225 (exp(y) start)."""

    model = gamlss_ml("y ~ 1", family=_truncated(), data={"y": _Y})
    mu = float(np.ravel(model.fitted_values["mu"])[0])
    sigma = float(np.ravel(model.fitted_values["sigma"])[0])
    assert 3.0 < mu < 8.0
    assert 0.5 < sigma < 6.0
    assert np.isfinite(model.g_dev)


@pytest.mark.slow
def test_deviance_is_near_the_r_optimum() -> None:
    """R (gen.trun + gamlssML) reaches dev 29.7910 at mu 5.4143 / sigma 2.2325.

    The remaining gap is a *different* defect (truncated families inherit the
    base analytic score while R uses numerical derivatives of the truncated
    density), so this only asserts that the start fix removed the blow-up.
    """

    from omnilss.gamlss_tr import trun_d

    model = gamlss_ml("y ~ 1", family=_truncated(), data={"y": _Y})
    mu = float(np.ravel(model.fitted_values["mu"])[0])
    sigma = float(np.ravel(model.fitted_values["sigma"])[0])
    density = trun_d(2.0, "WEI3", "left")
    deviance = -2.0 * float(np.sum(density(_Y, mu=mu, sigma=sigma, log=True)))
    assert deviance < 32.0  # the pre-fix fit sat at 33.2 with mu 224

def test_truncated_quantile_residuals_use_the_truncated_cdf() -> None:
    """Regression: the CDF closure takes **params, so positional calls failed.

    trun_p.q has signature fun(q, lower_tail, log_p, **params); passing mu and
    sigma positionally bound them to lower_tail/log_p and the parameter lookup
    raised "missing family parameter mu", which made every truncated fit report
    NaN residuals.
    """

    from scipy.stats import norm

    from omnilss._fitting_residuals import _build_rqres_callable
    from omnilss.gamlss_tr import trun_p

    family = _truncated()
    kernel = _build_rqres_callable(family)
    mu = np.full(_Y.shape, 5.4142857152120971)
    sigma = np.full(_Y.shape, 2.2325056695531957)
    residuals = np.asarray(kernel(y=_Y, mu=mu, sigma=sigma), dtype=float)
    assert np.all(np.isfinite(residuals))
    probabilities = np.asarray(trun_p(2.0, "WEI3", "left")(_Y, mu=mu[0], sigma=sigma[0]))
    np.testing.assert_allclose(residuals, norm.ppf(probabilities), rtol=1e-10, atol=1e-10)