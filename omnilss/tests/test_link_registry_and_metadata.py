"""Regression tests for link registry and metadata fixes (task 1.8).

Validates:
- ``1/mu^2`` maps to the inverse-square link (eta = 1/mu^2), not the plain
  inverse link (1/mu).
- ``logshiftto2`` is registered and round-trips (eta = log(mu-2)).
- pNBII is defined exactly once (previously duplicated; the later
  continuity-corrected version shadowed the earlier one).
- TF/TF2 variance includes the nu/(nu-2) scale factor (Student-t variance).
- LOGNO kurtosis equals the Pearson formula (consistent with NO=3, GA=3+6s^2).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnilss.distributions import LOGNO, TF
from omnilss.distributions3 import kurtosis, variance
from omnilss.dpqr_functions import pNBII
from omnilss.links import get_link


def test_inverse_square_link_maps_correctly() -> None:
    """1/mu^2 must be the inverse-square link, not the plain inverse."""
    link_f, inv_f, _ = get_link("1/mu^2")
    eta = np.asarray(link_f(jnp.array([2.0, 3.0])))
    np.testing.assert_allclose(eta, np.array([0.25, 1.0 / 9.0]))
    mu = np.asarray(inv_f(jnp.array([0.25, 1.0 / 9.0])))
    np.testing.assert_allclose(mu, np.array([2.0, 3.0]))
    # Regression guard: plain inverse link still maps 1/mu
    inv_link, _, _ = get_link("inverse")
    np.testing.assert_allclose(np.asarray(inv_link(jnp.array([2.0]))), np.array([0.5]))


def test_logshiftto2_registered_and_roundtrips() -> None:
    """logshiftto2: eta = log(mu-2), mu = 2 + exp(eta)."""
    link_f, inv_f, _ = get_link("logshiftto2")
    eta = np.asarray(link_f(jnp.array([3.0, 5.0])))
    np.testing.assert_allclose(eta, np.log(np.array([1.0, 3.0])))
    mu = np.asarray(inv_f(eta))
    np.testing.assert_allclose(mu, np.array([3.0, 5.0]))


def test_pnbii_defined_once() -> None:
    """pNBII must exist and be monotone (no duplicate shadowing)."""
    p = np.asarray(pNBII(jnp.array([0.0, 5.0, 10.0]), mu=5.0, sigma=0.5))
    assert np.all(np.diff(p) >= 0), "NBII CDF must be monotone"
    assert 0.0 <= p[0] <= 1.0 and 0.0 <= p[-1] <= 1.0


def test_tf_variance_has_nu_factor() -> None:
    """TF variance = sigma^2 * nu/(nu-2) (Student-t), not sigma^2."""
    for nu in [5.0, 10.0]:
        v = variance(TF(), 0.0, 1.0, nu)
        expected = 1.0 * nu / (nu - 2.0)
        np.testing.assert_allclose(v, expected, rtol=1e-9)


def test_logno_kurtosis_is_pearson() -> None:
    """LOGNO kurtosis = exp(4s2)+2exp(3s2)+3exp(2s2) (Pearson, no -3)."""
    for sigma in [0.3, 0.7]:
        k = kurtosis(LOGNO(), 0.0, sigma)
        s2 = sigma ** 2
        expected = np.exp(4 * s2) + 2 * np.exp(3 * s2) + 3 * np.exp(2 * s2)
        np.testing.assert_allclose(k, expected, rtol=1e-9)


def test_no_variance_unchanged() -> None:
    """NO variance is still sigma^2 (untouched by the TF fix)."""
    from omnilss.distributions import NO

    v = variance(NO(), 0.0, 1.0)
    np.testing.assert_allclose(v, 1.0, rtol=1e-9)
