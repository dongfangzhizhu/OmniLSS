"""RS fitting parity for WEI2 and PE2 against live R gamlss.

Background: WEI2/RGE/PE2 RS fits diverged under the generic initial values
(the batch-one/batch-two "known limitation", deferred to D2).  Root cause
(2026-10-03): the ``FamilyHeuristicInit`` warm start hijacked the mu
initialisation with ``mu = mean(y)`` — structurally wrong for WEI2, whose mu
is a *rate* (``scale = mu^(-1/sigma)``, so ``E[Y] = Gamma(1+1/sigma)/mu``),
not a location.  With that start the first mu IWLS step collapsed mu to
~[3e-7, 0.19] and the following sigma step exploded (g_dev ~6.2e6 after one
outer iteration).  The family-aware ``_initial_mu_beta`` never ran because
``rs_fit`` prefers a (n,)-shaped warm-start mu vector.

Fix:
- ``FamilyHeuristicInit`` now omits ``mu`` for WEI2/RGE/PE2 (see
  ``warm_start_policy.FamilyHeuristicInit._NON_MEAN_MU_FAMILIES``), letting
  ``rs_fit`` fall through to ``_initial_mu_beta`` — the same "family-aware
  initialisation is the single source of truth" policy the outer loop already
  applies to non-mu parameters (Phase 2.1 note in ``rs_algorithm.rs_fit``).
- ``_fitting_init`` gained R-faithful branches for WEI2/RGE/PE2, ported
  verbatim from the ``gamlss.dist`` family objects (``mu.initial`` /
  ``sigma.initial`` / ``nu.initial`` expressions).

Live-R cases cover WEI2 (2-parameter) and PE2 (3-parameter, nu).  Coefficients
use rtol 1e-2 instead of the core-suite 1e-3: near the optimum the WEI2/PE2
log-likelihood surface is flat in (sigma, nu), so Python and R stop at slightly
different points of the same optimum (deviances agree to rtol ~1e-5; observed
coefficient deviations <= 7e-3 relative).

RGE is intentionally absent from the live-R cases: R itself cannot fit RGE
with ``gamlss()`` — ``dRGE`` hard-stops ("x must greater than mu-sigma/nu")
during the un-truncated IWLS iterations.  This happens for data sampled from
RGE itself, for intercept-only models, and with explicit safe starts
(mu.start/sigma.start/nu.start).  RGE fitting parity is therefore blocked by
an R-side defect (same status philosophy as GIG/LOGSHASH); RGE keeps its
batch-one d/p/q/r parity and stays experimental for fitting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnilss import gamlss
from omnilss._fitting_init import (
    _initial_mu_beta,
    _initial_parameter_value,
    _initial_sigma,
)
from tests._r_bridge_helper import R_AVAILABLE, R_BRIDGE_CLS as RBridge

pytestmark = pytest.mark.skipif(not R_AVAILABLE, reason="R bridge not available")


def _wei2_data() -> dict[str, np.ndarray]:
    """Weibull AFT-consistent data: y = mu_true * Weibull(shape=1.5)."""
    rng = np.random.default_rng(2026)
    n = 120
    x = rng.uniform(0.0, 1.0, n)
    mu_true = np.exp(0.5 + 0.8 * x)
    y = mu_true * rng.weibull(1.5, n)
    return {"y": y, "x": x}


def _pe2_data() -> dict[str, np.ndarray]:
    """Near-normal data for the Power Exponential type 2 (nu absorbs kurtosis)."""
    rng = np.random.default_rng(2027)
    n = 120
    x = rng.uniform(0.0, 1.0, n)
    mu_true = np.exp(0.5 + 0.8 * x)
    y = mu_true + rng.normal(0.0, 0.4, n)
    return {"y": y, "x": x}


LIVE_R_CASES = {
    "WEI2": _wei2_data,
    "PE2": _pe2_data,
}


@pytest.fixture(scope="module")
def r_bridge():
    return RBridge()


@pytest.mark.parametrize("family", sorted(LIVE_R_CASES))
def test_rs_fit_matches_r_gamlss(r_bridge, family):
    """RS fit matches R gamlss (deviance rtol 1e-3, coefficients rtol 1e-2)."""
    data = LIVE_R_CASES[family]()
    py = gamlss("y ~ x", family=family, data=data, method="RS", max_iter=100)
    r = r_bridge.call_r_gamlss(
        data=data, formula="y ~ x", family=family, sigma_formula="~1", method="RS"
    )
    assert r["success"], r.get("error", "R failed")
    assert r["converged"] is True

    # Deviance: both optimisers must land on the same optimum.
    np.testing.assert_allclose(
        float(py.deviance), float(r["deviance"]), rtol=1e-3, atol=1e-3,
        err_msg=f"{family}: deviance {py.deviance} vs R {r['deviance']}",
    )

    # Link-scale coefficients: rtol 1e-2 (flat likelihood, see module docstring).
    for param in r["coefficients"]:
        r_coef = np.ravel(np.asarray(r["coefficients"][param], dtype=float))
        py_coef = np.ravel(np.asarray(py.coefficients[param], dtype=float))
        assert r_coef.shape == py_coef.shape, (
            f"{family}: {param} shape {py_coef.shape} vs R {r_coef.shape}"
        )
        np.testing.assert_allclose(
            py_coef, r_coef, rtol=1e-2, atol=1e-3,
            err_msg=f"{family} {param} coefficients vs R",
        )


# ---------------------------------------------------------------------------
# Pure-Python unit tests for the R-ported initial values (no R required).
# R references are the gamlss.dist family-object expressions:
#   WEI2: s.Y.s <- 1.283/sqrt(var(log(y)));
#         mu <- exp(-s.Y.s * (log(y) + 0.5772/s.Y.s))   (mu link: log)
#         sigma <- rep(s.Y.s, length(y))
#   RGE:  mu <- y + 0.45*sd(y)   (identity), sigma <- 0.78*sd(y),
#         nu <- rep(0.1, length(y))
#   PE2:  mu <- (y + mean(y))/2  (identity),
#         sigma <- (abs(y - mean(y)) + sd(y))/2, nu <- rep(1.8, length(y))
# ---------------------------------------------------------------------------


def _simple_y() -> np.ndarray:
    rng = np.random.default_rng(99)
    return rng.weibull(1.5, 40) * 3.0


def test_wei2_mu_init_matches_r_expression():
    y = _simple_y()
    X = np.ones((len(y), 1))
    w = np.ones(len(y))
    log_y = np.log(y)
    s_y_s = 1.283 / np.sqrt(np.var(log_y))
    expected_target = -s_y_s * log_y - 0.5772  # log(mu) (log link)
    beta = _initial_mu_beta(
        __import__("omnilss").WEI2(), X, y, w, fixed_parameter_values={}
    )
    np.testing.assert_allclose(beta, [expected_target.mean()], rtol=1e-12)


def test_wei2_sigma_init_matches_r_expression():
    y = _simple_y()
    from omnilss import WEI2

    s_y_s = 1.283 / np.sqrt(np.var(np.log(y)))
    sigma = _initial_sigma(
        WEI2(), y, np.full(len(y), y.mean()), np.ones(len(y)),
        fixed_parameter_values={},
    )
    np.testing.assert_allclose(sigma, s_y_s, rtol=1e-12)


def test_rge_pe2_sigma_and_nu_inits_match_r():
    from omnilss import PE2, RGE

    y = np.array([1.2, 2.3, 3.1, 0.7, 2.9, 4.4, 1.9, 2.6])
    w = np.ones(len(y))
    mu = np.full(len(y), y.mean())

    assert _initial_sigma(RGE(), y, mu, w, fixed_parameter_values={}) == pytest.approx(
        0.78 * np.std(y)
    )
    assert _initial_parameter_value(RGE(), "nu", y, mu, w) == pytest.approx(0.1)

    expected_sigma = (np.mean(np.abs(y - y.mean())) + np.std(y)) / 2.0
    assert _initial_sigma(PE2(), y, mu, w, fixed_parameter_values={}) == pytest.approx(
        expected_sigma
    )
    assert _initial_parameter_value(PE2(), "nu", y, mu, w) == pytest.approx(1.8)


def test_family_heuristic_omits_mu_for_non_mean_parameterizations():
    """The warm start must not hijack mu for rate-parameterized families."""
    from omnilss.algorithms.warm_start_policy import FamilyHeuristicInit

    y = _simple_y()
    policy = FamilyHeuristicInit()
    for family_name in ("WEI2", "RGE", "PE2"):
        result = policy.initialize(
            y=y, parameter_names=("mu", "sigma"), family_name=family_name
        )
        assert "mu" not in result.initial_fitted, family_name
        assert "sigma" in result.initial_fitted, family_name
    # Mean-parameterized families keep the generic start.
    result = policy.initialize(y=y, parameter_names=("mu", "sigma"), family_name="NO")
    np.testing.assert_allclose(result.initial_fitted["mu"], np.mean(y))
