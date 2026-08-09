"""Regression tests for qTF: exact Student-t quantiles.

Validates that qTF matches R's ``qt(p, df=nu)`` (i.e. ``mu + sigma*qt(p, df=nu)``)
instead of the previous normal approximation.  Reference values are generated
from scipy.stats.t.ppf, which agrees with R's qt to machine precision.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import t as scipy_t

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnilss.dpqr_functions import qTF


def _rtol_for(nu: float) -> float:
    # Tail quantiles of the t distribution are computed by scipy with high
    # relative accuracy; 1e-9 is far tighter than the 1e-5 project tolerance.
    return 1e-9


def test_qtf_matches_scipy_qt_over_grid() -> None:
    """qTF == mu + sigma*qt(p, df=nu) across df and probability grids."""
    ps = np.array([1e-6, 1e-4, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.9999, 1 - 1e-6])
    for nu in (3.0, 5.0, 10.0, 30.0, 100.0):
        mu, sigma = 1.5, 2.0
        q = qTF(ps, mu=mu, sigma=sigma, nu=nu)
        ref = mu + sigma * scipy_t.ppf(ps, df=nu)
        np.testing.assert_allclose(q, ref, rtol=_rtol_for(nu), atol=1e-12)


def test_qtf_known_r_value() -> None:
    """Known R value: qt(0.975, df=3) = 3.182446305283708."""
    q = float(qTF(0.975, 0.0, 1.0, 3.0))
    assert abs(q - 3.182446305283708) < 1e-12


def test_qtf_is_not_normal_approximation_in_tail() -> None:
    """Heavy tails: q(0.975; nu=3) must be ~3.18, not the normal 1.96."""
    q = float(qTF(0.975, 0.0, 1.0, 3.0))
    assert q > 3.0  # normal approximation would give ~1.96


def test_qtf_vectorized_and_broadcast() -> None:
    """Vector inputs and parameter broadcasting."""
    p = np.array([0.05, 0.5, 0.95])
    nu = np.array([3.0, 5.0, 10.0])
    q = qTF(p, mu=0.0, sigma=1.0, nu=nu)
    ref = scipy_t.ppf(p, df=nu)
    np.testing.assert_allclose(q, ref, rtol=1e-9, atol=1e-12)


def test_qtf_lower_tail_false() -> None:
    """upper-tail semantics: qTF(p, lower_tail=False) = qTF(1-p, lower_tail=True)."""
    q_upper = qTF(0.025, mu=0.0, sigma=1.0, nu=3.0, lower_tail=False)
    q_lower = qTF(0.975, mu=0.0, sigma=1.0, nu=3.0, lower_tail=True)
    np.testing.assert_allclose(q_upper, q_lower, rtol=1e-12, atol=1e-12)


def test_qtf_log_p() -> None:
    """log_p input semantics: qTF(log(p)) == qTF(p)."""
    p = 0.95
    q_log = qTF(np.log(p), mu=0.0, sigma=1.0, nu=4.0, log_p=True)
    q_plain = qTF(p, mu=0.0, sigma=1.0, nu=4.0, log_p=False)
    np.testing.assert_allclose(q_log, q_plain, rtol=1e-12, atol=1e-12)


def test_qtf_monotonic_in_p() -> None:
    """Quantile function is monotone increasing in p."""
    ps = np.linspace(0.01, 0.99, 50)
    q = qTF(ps, mu=-2.0, sigma=0.5, nu=6.0)
    assert np.all(np.diff(q) > 0)
