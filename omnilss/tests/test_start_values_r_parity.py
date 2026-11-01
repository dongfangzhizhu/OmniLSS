# SPDX-License-Identifier: GPL-3.0-or-later
"""R's `mu.start / sigma.start / nu.start / tau.start` (D2.131).

R source (`gamlss-5.R`):

.. code-block:: r

    if(!is.null(mu.start))
          { mu <- if(length(mu.start)>1) mu.start  else rep(mu.start,N)}

The values are on the **parameter** scale — the same scale as
`family$mu.initial`; the linear predictor is `mu.linkfun(mu.start)`
(`gamlss-5.R:431`).  A scalar is recycled to length n.

Golden reference: `omnilss/tests/fixtures/r_golden/family_start_values_2026-11-01.json`,
written by `tmp/d2131_parity.R` against live R 4.6.0 / gamlss 5.5.0 on
2026-11-01.  The regime is deliberately `n.cyc=1, i.control=glim.control(cyc=1)`:
with the default inner budget every case converges to the same optimum and the
start values become invisible; with a one-step inner loop the first cycle is a
pure function of where it started, which is exactly what this feature promises.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from omnilss import GA, NO, NET, gamlss
from omnilss.controls import GLIMControl

GOLDEN = (
    Path(__file__).parent
    / "fixtures"
    / "r_golden"
    / "family_start_values_2026-11-01.json"
)


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def _values(value) -> np.ndarray:
    return np.atleast_1d(np.asarray(value, dtype=np.float64))


def _fit(golden, family, **start):
    data = {
        "y": np.asarray(golden["data"]["y"], dtype=np.float64),
        "x": np.asarray(golden["data"]["x"], dtype=np.float64),
    }
    return gamlss(
        "y ~ x",
        family=family,
        data=data,
        method="RS",
        max_iter=1,
        i_control=GLIMControl(cyc=1),
        **start,
    )


def _assert_matches_r(model, reference, *, check_sigma=True):
    np.testing.assert_allclose(float(model.g_dev), float(reference["deviance"]), rtol=1e-6)
    np.testing.assert_allclose(
        _values(model.coefficients["mu"]), _values(reference["mu"]), rtol=1e-6
    )
    if check_sigma:
        np.testing.assert_allclose(
            _values(model.coefficients["sigma"]), _values(reference["sigma"]), rtol=1e-6
        )


def test_mu_and_sigma_start_match_live_r(golden):
    """Both starts given: the first cycle must reproduce R's numbers exactly."""
    model = _fit(golden, NO(), mu_start=20.0, sigma_start=0.3)
    _assert_matches_r(model, golden["no_both_start"])


def test_sigma_start_alone_matches_live_r(golden):
    """sigma.start is observable on its own: it moves R's first cycle, so must ours."""
    model = _fit(golden, NO(), sigma_start=0.3)
    _assert_matches_r(model, golden["no_sigma_start"])


def test_start_changes_the_first_cycle(golden):
    """The feature is real: the same data starts elsewhere when told to.

    Under the one-step inner budget R itself lands on a different deviance
    (33.786 with family defaults vs 24.007 from sigma.start=0.3), so a silent
    no-op implementation would show up here.
    """
    default = _fit(golden, NO())
    started = _fit(golden, NO(), sigma_start=0.3)
    assert float(default.g_dev) != pytest.approx(float(started.g_dev), rel=1e-3)
    assert float(started.g_dev) == pytest.approx(
        float(golden["no_both_start"]["deviance"]), rel=1e-6
    )


def test_mu_start_vector_matches_live_r(golden):
    """A length-n vector start (R: used as-is) behaves like the scalar form."""
    n = len(golden["data"]["y"])
    model = _fit(golden, NO(), mu_start=np.full(n, 20.0), sigma_start=0.3)
    _assert_matches_r(model, golden["no_both_start"])


def test_mu_start_on_a_log_link_family_matches_live_r(golden):
    """GA has a log link, so a distant mu.start survives one inner step.

    Only the mu coefficients are compared: with sigma left at the family
    default, our sigma start (residual RMSE against the *generic* mu fallback,
    i.e. zero here) is not R's `rep(1)` — a pre-existing default-start
    deviation that D2.131 does not touch (see the plan's D2.135).  The mu
    side is exactly R's, which is what the mu.start plumbing has to deliver.
    """
    model = _fit(golden, GA(), mu_start=5.0, sigma_start=0.3)
    _assert_matches_r(model, golden["ga_both_start"])

    mu_only = _fit(golden, GA(), mu_start=5.0)
    np.testing.assert_allclose(
        _values(mu_only.coefficients["mu"]),
        _values(golden["ga_mu_start"]["mu"]),
        rtol=1e-6,
    )


def test_converged_fit_is_independent_of_start_values(golden):
    """Start values move the trajectory, not the optimum."""
    data = {
        "y": np.asarray(golden["data"]["y"], dtype=np.float64),
        "x": np.asarray(golden["data"]["x"], dtype=np.float64),
    }
    default = gamlss("y ~ x", family=NO(), data=data, method="RS", max_iter=50)
    started = gamlss(
        "y ~ x",
        family=NO(),
        data=data,
        method="RS",
        max_iter=50,
        mu_start=20.0,
        sigma_start=0.3,
    )
    np.testing.assert_allclose(float(started.g_dev), float(default.g_dev), rtol=1e-8)
    np.testing.assert_allclose(
        _values(started.coefficients["mu"]),
        _values(default.coefficients["mu"]),
        rtol=1e-6,
    )


@pytest.mark.parametrize("method", ["RS_JAX", "HYBRID"])
def test_unsupported_methods_reject_start_values(method):
    """No silent ignore: routes with no R-equivalent start state fail closed."""
    data = {"y": [1.0, 2.0, 3.0, 4.0], "x": [0.0, 1.0, 2.0, 3.0]}
    with pytest.raises(ValueError, match="not supported by method"):
        gamlss("y ~ x", family=NO(), data=data, method=method, mu_start=2.0)


def test_unknown_parameter_name_is_rejected():
    data = {"y": [1.0, 2.0, 3.0, 4.0], "x": [0.0, 1.0, 2.0, 3.0]}
    with pytest.raises(ValueError, match="unknown parameter"):
        gamlss("y ~ x", family=NO(), data=data, sigma_start=1.0, nu_start=1.0)


def test_wrong_length_is_rejected():
    data = {"y": [1.0, 2.0, 3.0, 4.0], "x": [0.0, 1.0, 2.0, 3.0]}
    with pytest.raises(ValueError, match="scalar or length 4"):
        gamlss("y ~ x", family=NO(), data=data, mu_start=np.array([1.0, 2.0, 3.0]))


def test_non_finite_start_is_rejected():
    data = {"y": [1.0, 2.0, 3.0, 4.0], "x": [0.0, 1.0, 2.0, 3.0]}
    with pytest.raises(ValueError, match="non-finite"):
        gamlss("y ~ x", family=NO(), data=data, sigma_start=np.nan)


def test_fixed_parameter_rejects_a_start_value():
    """NET fixes nu/tau (R: parameters = list(nu=FALSE, tau=FALSE))."""
    data = {
        "y": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    }
    with pytest.raises(ValueError, match="fixes parameter 'nu'"):
        gamlss("y ~ x", family=NET(), data=data, nu_start=1.5)
