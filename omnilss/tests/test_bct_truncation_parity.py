"""BCT (Box-Cox t) parity: the fitting log-density must carry R's truncation term.

The defect
----------
``BCT`` is a *truncated* family.  R's ``dBCT`` ends with::

    loglik <- loglik + fTz - log(pt(1/(sigma * abs(nu)), df = tau))

and so does our own ``dpqr_functions.dBCT``.  The log-density used by the
fitting path (``g_dev_inc`` and the numeric scores inside
``distributions.BCT``) omitted that term.

Measured before the fix, at ``sigma=1.5, nu=0.7, tau=4``:

    max | log(dBCT) - (-g_dev_inc/2) |  =  0.2199204   (constant in y)
    exp(0.2199204)                      =  1.2459

A constant offset looks harmless, and for the deviance it mostly is -- but the
term depends on ``sigma`` and ``nu``, so **every score and Hessian was wrong
too**.  The parity scanner caught it as a derivative mismatch on ``sigma``,
``nu`` and ``tau`` (``mu`` matched, because a parameter-independent constant
drops out of a derivative).

Fix
---
``_truncation_log_term`` adds ``-log(pt(1/(sigma*|nu|), df=tau))`` using the
regularized incomplete beta identity ``pt = 1 - 0.5*I_{df/(df+t^2)}(df/2, 1/2)``
-- JAX ships no Student-t CDF, and this is the same construction ``BCTo``'s
``_bcto_log_pdf`` already uses.

Resolved in D2.13
-----------------
``BCT``'s fitting path used **numeric** scores and Hessians where R uses closed
forms, and its links were ``mu: log`` / ``tau: logshiftto2`` against R's
``identity`` / ``log``.  Both are fixed now; ``test_bct_analytic_derivatives.py``
covers the replacement and the live-R comparison.  The measured effect on a
60-point dataset, RS with n_cyc=500:

    baseline (numeric scores, old links)   deviance 200.7269   (R: 199.5647)
    analytic scores, old links             deviance 209.5343   (worse!)
    analytic scores + R's links            deviance 199.5666   (rel 9.4e-06)

Note the middle row: porting the derivatives *alone* made the fit worse,
because ``tau.link = logshiftto2`` implies a ``tau > 2`` floor that RS ran into
and stalled against.  The order matters -- derivatives first, links second.
"""

import numpy as np
import pytest

from omnilss.distributions import resolve_family

# Live R (gamlss.dist 4.6.0):
#   dBCT(c(0.5,1,2,3,3.5,4), mu=c(0.8,1.2,1.6,2,2.4,2.8), sigma=1.5, nu=0.7, tau=4)
_R_D = np.array(
    [
        0.4289584289313,
        0.2719535138369,
        0.1791609025403,
        0.1298350700469,
        0.1101077527666,
        0.0955427746185,
    ]
)
_R_LOGD = np.array(
    [
        -0.846395267008,
        -1.302124132303,
        -1.719470980034,
        -2.041490325945,
        -2.206295822050,
        -2.348181230009,
    ]
)

_Y = np.array([0.5, 1.0, 2.0, 3.0, 3.5, 4.0])
_MU = np.array([0.8, 1.2, 1.6, 2.0, 2.4, 2.8])
_SIGMA = np.full(6, 1.5)
_NU = np.full(6, 0.7)
_TAU = np.full(6, 4.0)

_PARAM_POS = {"mu": 1, "sigma": 2, "nu": 3, "tau": 4}


@pytest.fixture(scope="module")
def bct():
    return resolve_family("BCT")


def test_density_matches_live_r(bct):
    got = np.asarray(bct.d(_Y, _MU, _SIGMA, _NU, _TAU), dtype=np.float64)
    assert got == pytest.approx(_R_D, rel=1e-10)


def test_log_density_matches_live_r(bct):
    """``g_dev_inc`` is -2 log L, so -g/2 must reproduce R's log density."""
    got = -np.asarray(bct.g_dev_inc(_Y, _MU, _SIGMA, _NU, _TAU), dtype=np.float64) / 2.0
    assert got == pytest.approx(_R_LOGD, rel=1e-10, abs=1e-10)


def test_density_and_deviance_agree(bct):
    """The defect in one assertion: d and g_dev_inc were two different functions.

    They must now describe the same density, not merely differ by a constant.
    """
    dens = np.asarray(bct.d(_Y, _MU, _SIGMA, _NU, _TAU), dtype=np.float64)
    dev = np.asarray(bct.g_dev_inc(_Y, _MU, _SIGMA, _NU, _TAU), dtype=np.float64)
    assert np.max(np.abs(np.log(dens) + dev / 2.0)) < 1e-12


@pytest.mark.parametrize("param", ["mu", "sigma", "nu", "tau"])
def test_scores_match_central_differences_of_the_density(bct, param):
    """The regression the scanner reported: sigma/nu/tau scores were wrong.

    ``mu`` passed even before the fix, which is exactly why the defect survived
    -- a parameter-independent constant cancels in its derivative.

    Tolerance note for ``tau``: R's ``BCT()$dldt`` differentiates ``log pt``
    with a hard-coded 0.01 forward step, so **R's own dldt is ~1e-5 away from
    the exact derivative of its own density**.  Pinning our score to the finite
    difference therefore measures R's step size, not our fidelity.  Measured at
    mu=0.8, sigma=1.5, nu=0.7, tau=4.0 (first y):

        R's dldt                    0.0094931303529283695
        our analytic score          0.009493130352900836   (|diff to R| 2.8e-14)
        our finite difference       0.0094747620749124906   (|diff to R| 1.8e-05)

    i.e. the analytic score is ~9 orders of magnitude closer to R than the
    finite difference is.  ``tau`` gets a looser bound for that reason (5e-5
    against a measured 1.84e-5); the "is it R or is it the density?" question is
    settled in ``test_bct_analytic_derivatives.py`` by comparing against live R
    directly.
    """
    pos = _PARAM_POS[param]
    h = 1e-5

    def log_dens(delta: float) -> np.ndarray:
        args = [_Y, _MU.copy(), _SIGMA.copy(), _NU.copy(), _TAU.copy()]
        args[pos] = args[pos] + delta
        return np.log(np.asarray(bct.d(*args), dtype=np.float64))

    fd = (log_dens(h) - log_dens(-h)) / (2.0 * h)
    analytic = np.asarray(bct.score_functions[param](_Y, _MU, _SIGMA, _NU, _TAU), dtype=np.float64)
    rel = np.abs(analytic - fd) / np.maximum(np.abs(fd), 1.0)
    tol = 5e-5 if param == "tau" else 1e-6
    assert np.max(rel) < tol, f"{param}: max rel diff {np.max(rel):.3e} (tol {tol:.0e})"


def test_truncation_term_depends_on_sigma_and_nu(bct):
    """Why a constant offset was not the whole story.

    The missing term is parameter-dependent, so it shifts the scores too.  This
    pins that property so nobody reintroduces it as a constant.
    """
    def trunc(sigma: float, nu: float) -> float:
        args = [_Y, _MU, np.full(6, sigma), np.full(6, nu), _TAU]
        return float(np.mean(-np.asarray(bct.g_dev_inc(*args), dtype=np.float64) / 2.0))

    base = trunc(1.5, 0.7)
    assert abs(trunc(1.5, 1.4) - base) > 1e-3
    assert abs(trunc(2.5, 0.7) - base) > 1e-3


def test_untruncated_variant_has_no_truncation():
    """``BCTo`` is R's untruncated Box-Cox t; it must stay untruncated.

    Guards against "fixing" BCTo along with BCT -- the two differ *only* by
    this term, so it is easy to over-apply the fix.
    """
    bcto = resolve_family("BCTo")
    dens = np.asarray(bcto.d(_Y, _MU, _SIGMA, _NU, _TAU), dtype=np.float64)
    dev = np.asarray(bcto.g_dev_inc(_Y, _MU, _SIGMA, _NU, _TAU), dtype=np.float64)
    assert np.max(np.abs(np.log(dens) + dev / 2.0)) < 1e-12
