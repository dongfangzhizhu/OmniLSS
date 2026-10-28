"""ZANBI normalises elementwise; R does not (D2.85).

R dZANBI computes fy0 from dNBI(0, mu, sigma) and then uses
log(1-nu) + fy - log(1-exp(fy0)) for every row.  R dNBI does not recycle the
scalar x = 0 against a vector mu: measured on live R, the length of
dNBI(0, mu = c(2, 8), sigma = c(0.1, 0.1), log = TRUE) is 1, not 2.  So the
whole vector shares the first observation f0.  Our implementation evaluates
each observation own f0, which is what the formula says it should do -- so
the two disagree whenever mu varies across rows.

That is why the scanner records ZANBI as an R-density defect rather than a
parity failure, and why these tests pin OUR behaviour: if someone later makes
us match R to turn the gate green, this file fails first.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omnilss.distribution_registry import resolve  # noqa: E402
from omnilss.distributions_b10_zero_variants import _nbi_log_pdf  # noqa: E402


def _ours(y, mu, sigma, nu):
    family = resolve("ZANBI")
    n = len(np.atleast_1d(y))
    return np.asarray(
        family.d(
            jnp.asarray(np.atleast_1d(y), dtype=jnp.float64),
            jnp.asarray(np.atleast_1d(mu), dtype=jnp.float64),
            jnp.full(n, sigma, dtype=jnp.float64),
            jnp.full(n, nu, dtype=jnp.float64),
            log=True,
        ),
        dtype=float,
    )


def test_the_normalisation_is_elementwise():
    y = np.array([1.0, 3.0])
    mu = jnp.asarray(np.array([2.0, 8.0]))
    sg = jnp.asarray(np.array([0.1, 0.1]))
    nu = 0.1
    base = np.asarray(_nbi_log_pdf(jnp.asarray(y), mu, sg), dtype=float)
    f0 = np.asarray(_nbi_log_pdf(jnp.zeros(2), mu, sg), dtype=float)
    elementwise = np.log1p(-nu) + base - np.log1p(-np.exp(f0))
    ours = _ours(y, [2.0, 8.0], 0.1, nu)
    np.testing.assert_allclose(ours, elementwise, rtol=0, atol=1e-12)


def test_it_does_not_use_the_first_rows_f0():
    y = np.array([1.0, 3.0])
    mu = jnp.asarray(np.array([2.0, 8.0]))
    sg = jnp.asarray(np.array([0.1, 0.1]))
    nu = 0.1
    base = np.asarray(_nbi_log_pdf(jnp.asarray(y), mu, sg), dtype=float)
    f0_first = np.asarray(_nbi_log_pdf(jnp.zeros(2), mu, sg), dtype=float)[0]
    shared = np.log1p(-nu) + base - np.log1p(-np.exp(f0_first))
    ours = _ours(y, [2.0, 8.0], 0.1, nu)
    assert not np.allclose(ours, shared, atol=1e-6)


def test_the_zero_atom_is_nu():
    ours = _ours(0.0, 3.0, 0.2, 0.25)
    np.testing.assert_allclose(ours, [np.log(0.25)], rtol=0, atol=1e-14)