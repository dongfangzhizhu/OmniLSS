"""Regression tests for the JAX RS engine NaN/convergence handling.

Validates that the JAX RS fitting loop never reports ``converged=True``
when the deviance is non-finite (NaN/Inf).  Previously a NaN deviance was
masked to ``-1e6`` and the NaN difference in the convergence check made the
loop exit with ``converged=True`` — a silent divergence reported as success.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnilss.algorithms.jax_family_specs import get_jax_spec
from omnilss.algorithms.jax_rs_core import jax_rs_fit_core


def _fit_ga(y, max_outer=30):
    n = len(y)
    spec = get_jax_spec("GA")
    X = jnp.ones((n, 1))
    mu0 = jnp.full(n, float(np.mean(np.abs(y))))
    sigma0 = jnp.full(n, 0.1)
    init_params = jnp.stack([mu0, sigma0])
    init_etas = jnp.stack([spec.link_fns[0](mu0), spec.link_fns[1](sigma0)])
    return jax_rs_fit_core(
        y=jnp.asarray(y, dtype=jnp.float64),
        Xs=(X, X),
        init_params=init_params,
        init_etas=init_etas,
        obs_weights=jnp.ones(n),
        spec=spec,
        max_outer=max_outer,
        max_inner=5,
        tol=1e-4,
    )


def test_normal_fit_converges() -> None:
    rng = np.random.default_rng(0)
    y = np.abs(rng.normal(2.0, 0.5, 100)) + 0.1
    res = _fit_ga(y)
    assert res.converged
    assert math.isfinite(res.g_dev)


def test_nan_y_never_converges() -> None:
    rng = np.random.default_rng(1)
    y = np.abs(rng.normal(2.0, 0.5, 100)) + 0.1
    y[7] = np.nan  # NaN in data -> loglik NaN -> deviance NaN
    res = _fit_ga(y)
    assert not math.isfinite(res.g_dev), "NaN data must produce non-finite deviance"
    assert not res.converged, "NaN deviance must NEVER be reported as converged"


def test_inf_y_never_converges() -> None:
    rng = np.random.default_rng(2)
    y = np.abs(rng.normal(2.0, 0.5, 100)) + 0.1
    y[9] = np.inf  # Inf -> loglik NaN
    res = _fit_ga(y)
    assert not math.isfinite(res.g_dev)
    assert not res.converged, "Inf deviance must NEVER be reported as converged"


def test_zero_y_handled_by_family_guard() -> None:
    """GA loglik guards y<=0 via jnp.maximum(y, eps); result must stay finite."""
    rng = np.random.default_rng(3)
    y = np.abs(rng.normal(2.0, 0.5, 100)) + 0.1
    y[5] = 0.0
    res = _fit_ga(y)
    assert math.isfinite(res.g_dev), "family-guarded fit must stay finite"
