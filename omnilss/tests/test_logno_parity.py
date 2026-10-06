# SPDX-License-Identifier: GPL-3.0-or-later
"""LOGNO (log-normal) parameterization and RS fit parity with R.

Background
----------
R ``gamlss.dist`` defines two log-normal families whose ``mu`` means different
things:

- ``LOGNO``:  ``dLOGNO = dlnorm(x, meanlog = mu, sdlog = sigma)``
  → ``mu`` is the **mean of log(Y)**.
- ``LOGNO2``: ``dLOGNO2`` uses ``mean = log(mu)``
  → ``mu`` is the **median of Y**.

The Python ``dLOGNO`` / ``pLOGNO`` / ``qLOGNO`` / ``rLOGNO`` all applied the
``log(mu)`` convention, i.e. they were numerically equal to R's ``dLOGNO2``.
Live R values that pin this down::

    dLOGNO (c(2.1509, 2.3591), mu=0.8, sigma=0.4) -> 0.46200903 0.41830590
    dLOGNO2(c(2.1509, 2.3591), mu=0.8, sigma=0.4) -> 0.02181086 0.01093765

Separately, the RS algorithm needs the **expected** Hessian.  R declares
``d2ldd2 = function(sigma) -2/sigma^2`` -- note the signature takes only
sigma, which is only possible for an expectation over y.  Python used the
observed Hessian ``1/sigma^2 - 3(log y - mu)^2/sigma^4``, which turns
positive away from the optimum, so the RS working weights go bad and sigma
collapses to ~1e-18 -- while the relative-deviance test still reported
``converged=True`` (observed g_dev 8.9e12 against R's 173.05).
"""

import numpy as np
import pytest

from omnilss import LOGNO, gamlss
from omnilss.dpqr_functions import dLOGNO, pLOGNO, qLOGNO

from tests._r_bridge_helper import R_AVAILABLE, R_BRIDGE_CLS as RBridge

_r_live = pytest.mark.skipif(not R_AVAILABLE, reason="R bridge not available")


# ---------------------------------------------------------------------------
# d/p/q against values read out of live R (gamlss.dist, R-4.6.0)
# ---------------------------------------------------------------------------

_R_Y = np.array([2.1509, 2.3591, 1.9581])
_R_MU = np.array([0.8, 0.5, 1.2])
_R_SIGMA = 0.4

_R_D = np.array([0.4620090268, 0.2830681315, 0.2131186581])
_R_P = np.array([0.46601776512, 0.81479362192, 0.09340691533])
_R_Q = np.array([1.332929796, 1.648721271, 5.543469821])


def test_dLOGNO_matches_r():
    got = np.asarray(dLOGNO(_R_Y, _R_MU, _R_SIGMA))
    np.testing.assert_allclose(got, _R_D, rtol=1e-8)


def test_pLOGNO_matches_r():
    got = np.asarray(pLOGNO(_R_Y, _R_MU, _R_SIGMA))
    np.testing.assert_allclose(got, _R_P, rtol=1e-8)


def test_qLOGNO_matches_r():
    got = np.asarray(qLOGNO(np.array([0.1, 0.5, 0.9]), _R_MU, _R_SIGMA))
    np.testing.assert_allclose(got, _R_Q, rtol=1e-8)


def test_mu_is_log_scale_not_median():
    """mu must be meanlog: the density peaks at y = exp(mu), not at y = mu."""
    # With mu = 0.8 (log scale) the mode sits near exp(0.8) ~= 2.23.
    grid = np.linspace(0.5, 6.0, 200)
    dens = np.asarray(dLOGNO(grid, 0.8, 0.4))
    assert grid[int(np.argmax(dens))] == pytest.approx(np.exp(0.8 - 0.4**2), abs=0.2)
    # If mu were a median, the peak would sit at mu = 0.8 -- assert it does not.
    assert grid[int(np.argmax(dens))] > 1.5


# ---------------------------------------------------------------------------
# RS fit parity
# ---------------------------------------------------------------------------

def _logno_case(seed: int = 20261006):
    rng = np.random.default_rng(seed)
    n = 80
    x = rng.normal(0.0, 1.0, n)
    y = np.abs(rng.normal(2.0, 0.7, n)) + 0.15
    return {"y": y, "x": x}


def test_logno_rs_fit_matches_r_deviance():
    """RS fit must land on R's optimum, not diverge to sigma ~ 1e-18."""
    data = _logno_case()
    py = gamlss("y ~ x", family=LOGNO(), data=data, method="RS", max_iter=50)
    assert py.convergence_status() is True
    # Pre-fix this was 8.9e12 while still reporting converged=True.
    assert float(py.deviance) == pytest.approx(173.05494254762488, rel=1e-6)


@_r_live
def test_logno_rs_fit_matches_live_r():
    """Live-R cross-check: deviance and mu coefficients must agree."""
    data = _logno_case()
    py = gamlss("y ~ x", family=LOGNO(), data=data, method="RS", max_iter=50)
    r = RBridge().call_r_gamlss(
        data=data, formula="y ~ x", family="LOGNO", sigma_formula="~1", method="RS"
    )
    assert r["success"], r.get("error", "R failed")
    np.testing.assert_allclose(
        float(py.deviance), float(r["deviance"]), rtol=1e-6,
        err_msg=f"deviance {py.deviance} vs R {r['deviance']}",
    )
    r_mu = np.ravel(np.asarray(r["coefficients"]["mu"], dtype=float))
    py_mu = np.ravel(np.asarray(py.coefficients["mu"], dtype=float))
    np.testing.assert_allclose(py_mu, r_mu, rtol=1e-4)


def test_logno_hessian_is_expected_not_observed():
    """RS needs the expected Hessian: -2/sigma^2, independent of y.

    The observed Hessian depends on y and turns positive when the residual
    (log y - mu)^2 < sigma^2/3 -- that is what made the RS weights go bad.
    """
    fam = LOGNO()
    sigma = np.array([0.3, 0.5, 1.0])
    got = np.asarray(fam.hessian_functions["sigma"](_R_Y[:3], _R_MU[:3], sigma))
    np.testing.assert_allclose(got, -2.0 / np.square(sigma), rtol=1e-12)
    # Must not vary with y.
    other_y = np.array([10.0, 20.0, 30.0])
    got2 = np.asarray(fam.hessian_functions["sigma"](other_y, _R_MU[:3], sigma))
    np.testing.assert_allclose(got, got2, rtol=1e-12)
    assert np.all(got < 0.0), "expected Hessian must stay negative-definite"


def test_logno_score_matches_r_expression():
    """R: dldd = (1/sigma^3) * ((log y - mu)^2 - sigma^2)."""
    fam = LOGNO()
    y = np.array([2.1509, 2.3591, 1.9581])
    mu = np.array([0.8, 0.5, 1.2])
    sigma = np.array([0.4, 0.4, 0.4])
    expected = (np.square(np.log(y) - mu) - np.square(sigma)) / np.power(sigma, 3)
    got = np.asarray(fam.score_functions["sigma"](y, mu, sigma))
    np.testing.assert_allclose(got, expected, rtol=1e-10)
