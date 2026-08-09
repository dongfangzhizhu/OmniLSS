"""Regression tests for high-risk distribution families (task 1.9).

Validates:
- SIMPLEX family p/q/r are wired to real implementations (previously NaN
  placeholders) and match R's gamlss.dist within numerical-integration
  tolerance.
- SHASH uses the log link for nu (R SHASH.R: nu.link="log").
- TF2 applies the sigma * sqrt((nu-2)/nu) rescale (R TF2.R).
- ST1/ST2 are real Azzalini skew-t densities (not ST3C clones), matching R.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrandom

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnilss.distributions import TF2
from omnilss.distributions_b2 import SIMPLEX
from omnilss.distributions_b3 import SHASH
from omnilss.distributions_st_compat import ST1, ST2


def test_simplex_pqr_no_nan() -> None:
    """SIMPLEX p/q/r must be real values, not NaN placeholders."""
    fam = SIMPLEX()
    p = np.asarray(fam.p(jnp.array([0.3, 0.5, 0.8]), 0.5, 1.0))
    q = np.asarray(fam.q(jnp.array([0.1, 0.5, 0.9]), 0.5, 1.0))
    r = np.asarray(fam.r(jrandom.PRNGKey(0), 50, 0.5, 1.0))
    assert np.all(np.isfinite(p)) and np.all(np.isfinite(q)) and np.all(np.isfinite(r))
    assert np.all(np.diff(p) >= 0), "SIMPLEX CDF must be monotone"
    assert np.all((r > 0) & (r < 1)), "SIMPLEX samples must lie in (0,1)"


def test_simplex_cdf_matches_r() -> None:
    """pSIMPLEX against R gamlss.dist (numerical integration, loose tol)."""
    fam = SIMPLEX()
    qs = np.array([0.05, 0.2, 0.5, 0.8, 0.95])
    # R reference (R 4.5.0 gamlss.dist, case mu=0.5, sigma=1.0)
    r_ref = np.array([7.3469628556556225e-17, 1.3498980316302806e-03,
                      4.9999999996583283e-01, 9.9865010197158710e-01,
                      9.9999999998201106e-01])
    p = np.asarray(fam.p(jnp.asarray(qs), 0.5, 1.0))
    np.testing.assert_allclose(p, r_ref, rtol=1e-3, atol=1e-5)


def test_shash_nu_uses_log_link() -> None:
    """SHASH nu link must be 'log' (R SHASH.R nu.link='log')."""
    fam = SHASH()
    assert fam.links["nu"] == "log"


def test_tf2_sigma_rescale_matches_r() -> None:
    """TF2 density applies sigma*sqrt((nu-2)/nu)."""
    fam = TF2()
    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
    # R reference (mu=1.0, sigma=0.5, nu=5.0)
    r_ref = np.array([0.0038582573232608277, 0.0771538979015919391,
                      0.9801402585276303148, 0.0771538979015919391,
                      0.0038582573232608277])
    d = np.asarray(fam.d(jnp.asarray(xs), 1.0, 0.5, 5.0))
    np.testing.assert_allclose(d, r_ref, rtol=1e-9, atol=1e-12)


def test_st1_matches_r() -> None:
    """ST1 is the Azzalini type-1 skew-t density (R ST1.R)."""
    fam = ST1()
    xs = np.array([-1.5, -0.5, 0.0, 0.5, 1.5])
    # R reference (mu=0, sigma=1, nu=2, tau=5)
    r_ref = np.array([0.0037478784240444298, 0.1191057385422675113,
                      0.3796066898224944586, 0.5367313241032256599,
                      0.2452868108686658277])
    d = np.asarray(fam.g_dev_inc(jnp.asarray(xs), 0.0, 1.0, 2.0, 5.0)) / -2.0
    d = np.exp(d)
    np.testing.assert_allclose(d, r_ref, rtol=1e-9, atol=1e-12)


def test_st2_matches_r() -> None:
    """ST2 is the Azzalini type-2 skew-t density (R ST2.R)."""
    fam = ST2()
    xs = np.array([-1.5, -0.5, 0.0, 0.5, 1.5])
    # R reference (mu=0, sigma=1, nu=2, tau=5)
    r_ref = np.array([0.0042615384106467055, 0.1069539081762270610,
                      0.3796066898224944586, 0.5488831544692659437,
                      0.2447731508820635005])
    d = np.asarray(fam.g_dev_inc(jnp.asarray(xs), 0.0, 1.0, 2.0, 5.0)) / -2.0
    d = np.exp(d)
    np.testing.assert_allclose(d, r_ref, rtol=1e-9, atol=1e-12)


def test_st1_not_st3c_clone() -> None:
    """ST1 must differ from ST3C (regression guard against cloning)."""
    from omnilss.distributions_b15 import ST3C

    xs = np.array([-1.5, -0.5, 0.0, 0.5, 1.5])
    d_st1 = np.asarray(ST1().g_dev_inc(jnp.asarray(xs), 0.0, 1.0, 2.0, 5.0))
    d_st3c = np.asarray(ST3C().g_dev_inc(jnp.asarray(xs), 0.0, 1.0, 2.0, 5.0))
    assert not np.allclose(d_st1, d_st3c), "ST1 must not be an ST3C clone"
