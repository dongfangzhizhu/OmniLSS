"""Regression tests for WEI / WEI3 dpqr parameterizations.

Validates that:
- WEI d/p/q/r match R's WEI.R (``dweibull(x, scale=mu, shape=sigma)``), i.e.
  mu is the *scale* parameter — previously the code used a median
  parameterization (lambda = mu/(log 2)^sigma) that contradicted the fitted
  family density.
- WEI3 d/p/q/r match R's WEI3.R (``scale = mu/gamma(1+1/sigma)``), i.e.
  mu is the mean.
- WEI density agrees with the fitted family deviance increment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import weibull_min

# Machine-precision assertions require float64 (project precision policy).
import jax

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnilss.distributions import WEI
from omnilss.dpqr_functions import dWEI, pWEI, qWEI, dWEI3, pWEI3, qWEI3


def test_wei_matches_scipy_scale_parameterization() -> None:
    """d/p/q WEI == scipy weibull_min(scale=mu, c=sigma)."""
    ps = np.array([0.05, 0.5, 0.95])
    for mu, sigma in [(1.0, 1.5), (2.5, 0.8), (5.0, 3.0)]:
        xs = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0]) * mu
        np.testing.assert_allclose(
            np.asarray(dWEI(xs, mu, sigma)),
            weibull_min.pdf(xs, c=sigma, scale=mu),
            rtol=1e-9, atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(pWEI(xs, mu, sigma)),
            weibull_min.cdf(xs, c=sigma, scale=mu),
            rtol=1e-9, atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(qWEI(ps, mu, sigma)),
            weibull_min.ppf(ps, c=sigma, scale=mu),
            rtol=1e-9, atol=1e-12,
        )


def test_wei3_matches_scipy_mean_parameterization() -> None:
    """d/p/q WEI3 == scipy weibull_min(scale=mu/gamma(1+1/sigma), c=sigma)."""
    from math import gamma

    ps = np.array([0.05, 0.5, 0.95])
    for mu, sigma in [(1.0, 1.5), (2.5, 0.8)]:
        scale3 = mu / gamma(1.0 + 1.0 / sigma)
        xs = np.array([0.1, 0.5, 1.0, 2.0]) * mu
        np.testing.assert_allclose(
            np.asarray(dWEI3(xs, mu, sigma)),
            weibull_min.pdf(xs, c=sigma, scale=scale3),
            rtol=1e-9, atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(pWEI3(xs, mu, sigma)),
            weibull_min.cdf(xs, c=sigma, scale=scale3),
            rtol=1e-9, atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(qWEI3(ps, mu, sigma)),
            weibull_min.ppf(ps, c=sigma, scale=scale3),
            rtol=1e-9, atol=1e-12,
        )


def test_wei_density_matches_fitted_family() -> None:
    """dWEI(log) == -0.5*g_dev_inc: family fit and dpqr no longer contradict."""
    fam = WEI()
    xs = np.array([0.3, 1.0, 2.5])
    mu, sigma = 2.0, 1.2
    d_log = np.asarray(dWEI(xs, mu, sigma, log=True))
    fam_log = np.asarray(-0.5 * fam.g_dev_inc(xs, mu, sigma))
    np.testing.assert_allclose(d_log, fam_log, rtol=1e-9, atol=1e-12)


def test_wei_not_median_parameterization() -> None:
    """Regression guard: old code used lambda=mu/(log2)^sigma (median)."""
    # Old median parameterization made qWEI(0.5) == mu exactly.
    # Correct WEI (scale=mu): median is mu*(ln2)^(1/sigma) != mu for sigma != 1.
    q50 = float(qWEI(0.5, 2.0, 1.5))
    expected_median = 2.0 * np.power(np.log(2.0), 1.0 / 1.5)
    assert abs(q50 - expected_median) < 1e-12  # matches scale parameterization
    assert abs(q50 - 2.0) > 1e-6  # and is NOT the old median==mu behavior


def test_wei_pq_roundtrip() -> None:
    """pWEI(qWEI(p)) == p."""
    ps = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
    q = np.asarray(qWEI(ps, 1.8, 1.3))
    p_back = np.asarray(pWEI(q, 1.8, 1.3))
    np.testing.assert_allclose(p_back, ps, rtol=1e-9, atol=1e-12)


def test_wei_boundary() -> None:
    """CDF boundary: q<=0 gives 0."""
    assert float(pWEI(0.0, mu=1.0, sigma=1.5)) == 0.0
    assert float(pWEI(-1.0, mu=1.0, sigma=1.5)) == 0.0
