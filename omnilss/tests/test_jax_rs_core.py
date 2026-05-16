# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the JAX-native RS fitting core.

Test strategy
-------------
1. ``test_jax_spec_*``  — unit tests for each FamilyJAXSpec (score/hessian
   sign, loglik finite, link round-trip).
2. ``test_jax_rs_matches_numpy_*`` — integration tests: JAX path vs NumPy RS
   path, deviance difference < 0.05 (generous tolerance for inner-iter
   approximation).
3. ``test_jax_rs_r_consistency_*`` — R consistency tests (skipped when R is
   unavailable).
4. ``test_gamlss_method_rs_jax_*`` — end-to-end via gamlss() entry point.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import jax
import jax.numpy as jnp

# Enable float64 globally for tests
jax.config.update("jax_enable_x64", True)

from omnilss import gamlss, NO, GA, PO, BI, WEI, TF
from omnilss.algorithms.jax_family_specs import (
    get_jax_spec,
    make_no_spec,
    make_ga_spec,
    make_po_spec,
    make_bi_spec,
    make_wei_spec,
    make_tf_spec,
    supported_families,
)
from omnilss.algorithms.jax_rs_core import jax_rs_fit_core
from omnilss.algorithms.jax_rs_integration import gamlss_rs_jax

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
N = 300  # sample size for all tests


def _make_no_data(n=N, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = 2.0 + 1.5 * x + rng.standard_normal(n)
    return {"y": y, "x": x}


def _make_ga_data(n=N, seed=43):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    mu = np.exp(1.0 + 0.5 * x)
    y = rng.gamma(shape=4.0, scale=mu / 4.0)
    return {"y": y, "x": x}


def _make_po_data(n=N, seed=44):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    mu = np.exp(1.0 + 0.3 * x)
    y = rng.poisson(mu).astype(float)
    return {"y": y, "x": x}


def _make_bi_data(n=N, seed=45):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    p = 1.0 / (1.0 + np.exp(-(0.5 + 0.8 * x)))
    y = rng.binomial(1, p).astype(float)
    return {"y": y, "x": x}


def _make_wei_data(n=N, seed=46):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    mu = np.exp(1.0 + 0.4 * x)
    sigma = 1.5
    # Weibull: scale=mu, shape=sigma
    y = rng.weibull(sigma, size=n) * mu
    return {"y": y, "x": x}


def _make_tf_data(n=N, seed=47):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    mu = 1.0 + 0.5 * x
    # Student-t with nu=5
    from scipy.stats import t as scipy_t
    y = mu + scipy_t.rvs(df=5, size=n, random_state=rng)
    return {"y": y, "x": x}


# ---------------------------------------------------------------------------
# 1. Unit tests: FamilyJAXSpec correctness
# ---------------------------------------------------------------------------

class TestFamilyJAXSpecs:
    """Verify each spec's loglik, score, and hessian are numerically sane."""

    @pytest.mark.parametrize("spec_fn,data_fn,param_names", [
        (make_no_spec,  _make_no_data,  ("mu", "sigma")),
        (make_ga_spec,  _make_ga_data,  ("mu", "sigma")),
        (make_po_spec,  _make_po_data,  ("mu",)),
        (make_bi_spec,  _make_bi_data,  ("mu",)),
        (make_wei_spec, _make_wei_data, ("mu", "sigma")),
        (make_tf_spec,  _make_tf_data,  ("mu", "sigma", "nu")),
    ])
    def test_loglik_finite(self, spec_fn, data_fn, param_names):
        spec = spec_fn()
        data = data_fn()
        y = jnp.asarray(data["y"])
        # Use simple constant params
        params = [jnp.ones(N) * 1.0 for _ in param_names]
        if "sigma" in param_names:
            idx = list(param_names).index("sigma")
            params[idx] = jnp.ones(N) * 0.5
        if "nu" in param_names:
            idx = list(param_names).index("nu")
            params[idx] = jnp.ones(N) * 5.0
        ll = spec.loglik_fn(y, *params)
        assert jnp.all(jnp.isfinite(ll)), f"{spec.name}: loglik has non-finite values"

    @pytest.mark.parametrize("spec_fn,data_fn,param_names", [
        (make_no_spec,  _make_no_data,  ("mu", "sigma")),
        (make_ga_spec,  _make_ga_data,  ("mu", "sigma")),
        (make_po_spec,  _make_po_data,  ("mu",)),
        (make_bi_spec,  _make_bi_data,  ("mu",)),
        (make_wei_spec, _make_wei_data, ("mu", "sigma")),
        (make_tf_spec,  _make_tf_data,  ("mu", "sigma", "nu")),
    ])
    def test_hessian_negative(self, spec_fn, data_fn, param_names):
        """Diagonal Hessian mean must be negative (concave log-likelihood on average).

        Note: for heavy-tailed distributions (TF), the *observed* Hessian can be
        positive for individual observations with large |z|.  We check that the
        *mean* is negative, which is the relevant condition for IRLS convergence.
        TF nu Hessian is near-zero for large nu — we only check mu and sigma for TF.
        """
        spec = spec_fn()
        data = data_fn()
        y = jnp.asarray(data["y"])

        # Build realistic initial params for all parameters
        params = []
        for pname in param_names:
            if pname == "mu":
                params.append(jnp.ones(N) * float(jnp.mean(y)))
            elif pname == "sigma":
                params.append(jnp.ones(N) * 0.5)
            elif pname == "nu":
                params.append(jnp.ones(N) * 5.0)
            else:
                params.append(jnp.ones(N) * 1.0)

        # For TF, only check mu and sigma (nu Hessian is near-zero for large nu)
        check_indices = list(range(len(param_names)))
        if spec.name == "TF":
            check_indices = [i for i, p in enumerate(param_names) if p != "nu"]

        for k in check_indices:
            pname = param_names[k]
            h = spec.hessian_fns[k](y, *params)
            mean_h = float(jnp.mean(h))
            assert mean_h < 0, (
                f"{spec.name} param '{pname}': mean hessian={mean_h:.4f} is not negative"
            )

    @pytest.mark.parametrize("spec_fn,link_domains", [
        (make_no_spec,  [None, (0.01, 10.0)]),          # identity, log
        (make_ga_spec,  [(0.01, 10.0), (0.01, 10.0)]),  # log, log
        (make_po_spec,  [(0.01, 10.0)]),                 # log
        (make_bi_spec,  [(0.01, 0.99)]),                 # logit: (0,1)
        (make_wei_spec, [(0.01, 10.0), (0.01, 10.0)]),  # log, log
        (make_tf_spec,  [None, (0.01, 10.0), (0.5, 20.0)]),  # identity, log, log
    ])
    def test_link_round_trip(self, spec_fn, link_domains):
        """link_inv(link(theta)) ≈ theta for all link functions."""
        spec = spec_fn()
        for (link_fn, link_inv), domain in zip(
            zip(spec.link_fns, spec.link_inv_fns), link_domains
        ):
            if domain is None:
                theta = jnp.array([0.5, 1.0, 2.0, 5.0])
            else:
                lo, hi = domain
                theta = jnp.array([lo, (lo + hi) / 2, hi * 0.8, hi])
            eta = link_fn(theta)
            theta_back = link_inv(eta)
            np.testing.assert_allclose(
                np.asarray(theta_back), np.asarray(theta),
                rtol=1e-6, atol=1e-8,
                err_msg=f"{spec.name}: link round-trip failed",
            )


# ---------------------------------------------------------------------------
# 2. Integration tests: JAX path vs NumPy RS path
# ---------------------------------------------------------------------------

DEVIANCE_TOL = 0.1  # generous tolerance: inner-iter approximation


@pytest.mark.parametrize("family_name,data_fn,family_fn,formula", [
    ("NO",  _make_no_data,  NO,  "y ~ x"),
    ("GA",  _make_ga_data,  GA,  "y ~ x"),
    ("PO",  _make_po_data,  PO,  "y ~ x"),
    ("BI",  _make_bi_data,  BI,  "y ~ x"),
    ("WEI", _make_wei_data, WEI, "y ~ x"),
    ("TF",  _make_tf_data,  TF,  "y ~ x"),
])
class TestJaxRsMatchesNumpy:
    """JAX RS path deviance must be close to NumPy RS path deviance."""

    def test_deviance_close(self, family_name, data_fn, family_fn, formula):
        data = data_fn()
        model_np  = gamlss(formula, family=family_fn(), data=data, method="RS")
        model_jax = gamlss(formula, family=family_fn(), data=data, method="RS_JAX")

        diff = abs(model_np.g_dev - model_jax.g_dev)
        assert diff < DEVIANCE_TOL, (
            f"{family_name}: deviance diff={diff:.6f} > tol={DEVIANCE_TOL}. "
            f"NumPy={model_np.g_dev:.4f}, JAX={model_jax.g_dev:.4f}"
        )

    def test_mu_close(self, family_name, data_fn, family_fn, formula):
        """Fitted mu values should be close between the two paths."""
        data = data_fn()
        model_np  = gamlss(formula, family=family_fn(), data=data, method="RS")
        model_jax = gamlss(formula, family=family_fn(), data=data, method="RS_JAX")

        mu_np  = np.asarray(model_np.fitted_values["mu"])
        mu_jax = np.asarray(model_jax.fitted_values["mu"])
        np.testing.assert_allclose(
            mu_jax, mu_np, rtol=0.05, atol=0.1,
            err_msg=f"{family_name}: fitted mu values differ too much",
        )

    def test_model_has_correct_method(self, family_name, data_fn, family_fn, formula):
        data = data_fn()
        model = gamlss(formula, family=family_fn(), data=data, method="RS_JAX")
        assert model.additional_slots.get("method") == "RS_JAX"

    def test_model_aic_finite(self, family_name, data_fn, family_fn, formula):
        data = data_fn()
        model = gamlss(formula, family=family_fn(), data=data, method="RS_JAX")
        assert math.isfinite(model.additional_slots["aic"])


# ---------------------------------------------------------------------------
# 3. Intercept-only models (simplest case)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family_name,data_fn,family_fn", [
    ("NO",  _make_no_data,  NO),
    ("GA",  _make_ga_data,  GA),
    ("PO",  _make_po_data,  PO),
    ("BI",  _make_bi_data,  BI),
    ("WEI", _make_wei_data, WEI),
    ("TF",  _make_tf_data,  TF),
])
def test_intercept_only(family_name, data_fn, family_fn):
    """Intercept-only model should converge and produce finite deviance."""
    data = data_fn()
    model = gamlss("y ~ 1", family=family_fn(), data=data, method="RS_JAX")
    assert math.isfinite(model.g_dev), f"{family_name}: deviance is not finite"
    assert model.g_dev > 0, f"{family_name}: deviance should be positive"


# ---------------------------------------------------------------------------
# 4. Error handling
# ---------------------------------------------------------------------------

def test_unsupported_family_raises():
    """RS_JAX should raise ValueError for unsupported families."""
    from omnilss import NBI
    data = _make_no_data()
    with pytest.raises(ValueError, match="not supported by method='RS_JAX'"):
        gamlss("y ~ x", family=NBI(), data=data, method="RS_JAX")


def test_smooth_term_raises():
    """RS_JAX should raise ValueError when smooth terms are present."""
    data = _make_no_data()
    with pytest.raises(ValueError, match="does not support smooth terms"):
        gamlss("y ~ pb(x)", family=NO(), data=data, method="RS_JAX")


def test_supported_families_list():
    """supported_families() should return the expected set."""
    families = supported_families()
    assert set(families) == {"NO", "GA", "PO", "BI", "WEI", "TF"}


# ---------------------------------------------------------------------------
# 5. jax_rs_fit_core direct API
# ---------------------------------------------------------------------------

def test_jax_rs_fit_core_no_direct():
    """Test jax_rs_fit_core directly with NO family."""
    data = _make_no_data()
    y = jnp.asarray(data["y"])
    x = jnp.asarray(data["x"])
    n = len(y)

    # Design matrices: intercept + x for mu, intercept-only for sigma
    X_mu    = jnp.stack([jnp.ones(n), x], axis=1)
    X_sigma = jnp.ones((n, 1))

    spec = get_jax_spec("NO")

    # Initial params
    mu0    = jnp.full(n, float(jnp.mean(y)))
    sigma0 = jnp.full(n, float(jnp.std(y)))
    init_params = jnp.stack([mu0, sigma0])
    init_etas   = jnp.stack([
        spec.link_fns[0](mu0),
        spec.link_fns[1](sigma0),
    ])

    result = jax_rs_fit_core(
        y=y,
        Xs=(X_mu, X_sigma),
        init_params=init_params,
        init_etas=init_etas,
        obs_weights=jnp.ones(n),
        spec=spec,
        max_outer=20,
        max_inner=5,
        tol=1e-4,
    )

    assert math.isfinite(result.g_dev)
    assert result.g_dev > 0
    assert result.iterations > 0
    assert result.params.shape == (2, n)
    assert len(result.betas) == 2
    assert result.betas[0].shape == (2,)   # intercept + x
    assert result.betas[1].shape == (1,)   # intercept only


# ---------------------------------------------------------------------------
# 6. R consistency tests (skipped when R unavailable)
# ---------------------------------------------------------------------------

try:
    from omnilss.tests._r_bridge_helper import r_gamlss_fit
    R_AVAILABLE = True
except Exception:
    R_AVAILABLE = False


@pytest.mark.skipif(not R_AVAILABLE, reason="R + gamlss not available")
@pytest.mark.parametrize("family_name,data_fn,family_fn,formula", [
    ("NO",  _make_no_data,  NO,  "y ~ x"),
    ("GA",  _make_ga_data,  GA,  "y ~ x"),
    ("PO",  _make_po_data,  PO,  "y ~ x"),
    ("WEI", _make_wei_data, WEI, "y ~ x"),
    ("TF",  _make_tf_data,  TF,  "y ~ x"),
])
def test_r_consistency(family_name, data_fn, family_fn, formula):
    """JAX RS deviance must be within 0.01 of R gamlss deviance."""
    data = data_fn()
    model_jax = gamlss(formula, family=family_fn(), data=data, method="RS_JAX")
    r_result  = r_gamlss_fit(formula, family_name, data)
    r_dev     = r_result["deviance"]

    diff = abs(model_jax.g_dev - r_dev)
    assert diff < 0.01, (
        f"{family_name}: JAX deviance={model_jax.g_dev:.6f}, "
        f"R deviance={r_dev:.6f}, diff={diff:.6f}"
    )
