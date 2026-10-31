"""DELAPORT is R's DEL under another registry name (D2.108).

The dpqr gate used to report "no d/p/q counterpart in R" for DELAPORT, because
gamlss.dist exports the family as DEL (Delaport).  Measured on live R 4.6.0
+ gamlss.dist: our DELAPORT d/p/q and R's dDEL/pDEL/qDEL agree to ~1e-15 at
x = (0, 1, 2, 5, 10), mu = (1, 2, 3, 1.5, 4), sigma = (0.5, 1, 1.5, 0.8, 2),
nu = (0.3, 0.5, 0.7, 0.4, 0.6).  The values below are R literals.

These tests pin the identity in Python so the alias cannot drift away from the
implementation it claims to describe.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omnilss.distribution_registry import resolve  # noqa: E402

_X = np.array([0.0, 1.0, 2.0, 5.0, 10.0])
_MU = np.array([1.0, 2.0, 3.0, 1.5, 4.0])
_SIGMA = np.array([0.5, 1.0, 1.5, 0.8, 2.0])
_NU = np.array([0.3, 0.5, 0.7, 0.4, 0.6])

_R_D = np.array([
    0.40648462040149119,
    0.2759095808785817,
    0.22118058642346761,
    0.023439091898392463,
    0.014904621196295509,
])
_R_P = np.array([
    0.40648462040149119,
    0.45984930146430281,
    0.46247864885196127,
    0.9815451696674673,
    0.9617522038595403,
])
_R_Q = np.array([0.0, 1.0, 3.0, 2.0, 8.0])


def _family(name: str):
    return resolve(name)


def test_delaport_matches_r_del_density():
    family = _family("DELAPORT")
    got = np.asarray(
        family.d(jnp.asarray(_X), jnp.asarray(_MU), jnp.asarray(_SIGMA), jnp.asarray(_NU)),
        dtype=float,
    )
    np.testing.assert_allclose(got, _R_D, rtol=1e-13)


def test_delaport_matches_r_del_cdf():
    family = _family("DELAPORT")
    got = np.asarray(
        family.p(jnp.asarray(_X), jnp.asarray(_MU), jnp.asarray(_SIGMA), jnp.asarray(_NU)),
        dtype=float,
    )
    np.testing.assert_allclose(got, _R_P, rtol=1e-13)


def test_delaport_matches_r_del_quantile():
    family = _family("DELAPORT")
    probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    got = np.asarray(
        family.q(jnp.asarray(probs), jnp.asarray(_MU), jnp.asarray(_SIGMA), jnp.asarray(_NU)),
        dtype=float,
    )
    np.testing.assert_allclose(got, _R_Q, rtol=0, atol=1e-12)