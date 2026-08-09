"""Regression tests for pIG/qIG: exact inverse-Gaussian CDF and quantiles.

Validates that pIG matches R's IG.R CDF formula
    F(q) = Phi((q/mu-1)/(sigma*sqrt(q))) + exp(2/(mu*sigma^2)) * Phi(-(q/mu+1)/(sigma*sqrt(q)))
and that qIG is an exact inversion (via scipy invgauss with the GAMLSS
parameter mapping IG(mu, sigma) == invgauss(mu*sigma^2)/sigma^2), instead of
the previous normal/delta-method approximation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr
from scipy.stats import invgauss, norm

# This test asserts machine-precision agreement with R formulas, which
# requires float64.  Enable x64 before importing JAX-dependent modules
# (aligns with the project precision policy, see run_suite.py).
import jax

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnilss.dpqr_functions import pIG, qIG


def _r_reference_cdf(q, mu, sigma):
    """Reference implementation of R IG.R pIG (no approximation)."""
    cdf1 = norm.cdf(((q / mu) - 1) / (sigma * np.sqrt(q)))
    z2 = -((q / mu) + 1) / (sigma * np.sqrt(q))
    lcdf2 = (2 / (mu * sigma ** 2)) + log_ndtr(z2)
    return cdf1 + np.exp(lcdf2)


def test_pig_matches_r_formula_over_mu_grid() -> None:
    """pIG equals the R IG.R CDF formula across mu in {0.5, 1, 2, 5}."""
    qs = np.array([0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
    for mu, sigma in [(0.5, 0.3), (1.0, 0.3), (2.0, 0.4), (5.0, 0.8)]:
        got = np.asarray(pIG(qs * mu, mu=mu, sigma=sigma))
        ref = _r_reference_cdf(qs * mu, mu, sigma)
        np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)


def test_qig_matches_scipy_exact_inversion() -> None:
    """qIG equals scipy invgauss ppf under the verified parameter mapping."""
    ps = np.array([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    for mu, sigma in [(0.5, 0.3), (1.0, 0.3), (2.5, 0.5), (5.0, 0.8)]:
        got = np.asarray(qIG(ps, mu=mu, sigma=sigma))
        ref = invgauss.ppf(ps, mu=mu * sigma * sigma) / (sigma * sigma)
        np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)


def test_pig_qig_roundtrip() -> None:
    """qIG(p) is an exact quantile: pIG(qIG(p)) == p."""
    ps = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
    for mu, sigma in [(0.5, 0.3), (1.0, 0.3), (2.5, 0.5)]:
        q = np.asarray(qIG(ps, mu=mu, sigma=sigma))
        p_back = np.asarray(pIG(q, mu=mu, sigma=sigma))
        np.testing.assert_allclose(p_back, ps, rtol=1e-9, atol=1e-12)


def test_qig_is_not_delta_method_approximation() -> None:
    """Heavy-parameter case where the old delta-method approximation failed."""
    # mu=0.5, sigma=0.3, p=0.01: old approximation gave q ~ mu*(1+sigma*z*sqrt(mu)+...)
    q = float(qIG(0.01, mu=0.5, sigma=0.3))
    ref = float(invgauss.ppf(0.01, mu=0.5 * 0.09) / 0.09)
    assert abs(q - ref) < 1e-12


def test_qig_lower_tail_and_log_p_semantics() -> None:
    """upper-tail and log_p input handling."""
    q_upper = np.asarray(qIG(0.05, mu=1.0, sigma=0.3, lower_tail=False))
    q_lower = np.asarray(qIG(0.95, mu=1.0, sigma=0.3, lower_tail=True))
    np.testing.assert_allclose(q_upper, q_lower, rtol=1e-9, atol=1e-12)

    q_log = np.asarray(qIG(np.log(0.95), mu=1.0, sigma=0.3, log_p=True))
    q_plain = np.asarray(qIG(0.95, mu=1.0, sigma=0.3, log_p=False))
    np.testing.assert_allclose(q_log, q_plain, rtol=1e-9, atol=1e-12)


def test_pig_boundary_behavior() -> None:
    """CDF boundary: q<=0 gives 0, q->inf gives 1."""
    mu, sigma = 1.0, 0.3
    assert float(pIG(0.0, mu=mu, sigma=sigma)) == 0.0
    assert float(pIG(np.inf, mu=mu, sigma=sigma)) == 1.0
    assert 0.0 <= float(pIG(0.01, mu=mu, sigma=sigma)) <= 1.0


def test_qig_monotonic_in_p() -> None:
    """Quantile function is monotone increasing in p."""
    ps = np.linspace(0.01, 0.99, 50)
    q = np.asarray(qIG(ps, mu=1.5, sigma=0.4))
    assert np.all(np.diff(q) > 0)
