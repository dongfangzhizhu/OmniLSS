# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for D2.287/D2a data-driven SICHEL / ZISICHEL / ZASICHEL dynamic scan bounds."""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from omnilss.distributions_b6 import (
    SICHEL,
    SICHEL_DENSITY_MAX_COUNT,
    _sichel_log_pdf,
    _sichel_log_pdf_at_zero,
    _sichel_scan_steps,
)
from omnilss.distributions_b10_zero_variants import ZASICHEL, ZISICHEL
from omnilss.dpqr_functions import dSICHEL


def test_sichel_scan_steps_uses_concrete_max_count():
    assert _sichel_scan_steps(np.array([0.0, 0.0])) == 0
    assert _sichel_scan_steps(np.array([0.0, 1.0, 7.0])) == 7
    assert _sichel_scan_steps(np.array([0.0, 12.2, np.nan, -1.0])) == 13
    assert _sichel_scan_steps(np.array([float(SICHEL_DENSITY_MAX_COUNT) + 50.0])) == SICHEL_DENSITY_MAX_COUNT


def test_sichel_log_pdf_at_zero_matches_general_kernel():
    for mu, sigma, nu in ((1.0, 0.5, -0.5), (2.5, 0.8, 0.3), (10.0, 0.2, 1.5)):
        fast = float(_sichel_log_pdf_at_zero(mu, sigma, nu))
        exact = float(_sichel_log_pdf(0.0, mu, sigma, nu, max_steps=16))
        np.testing.assert_allclose(fast, exact, rtol=1e-14, atol=1e-14)


def test_dynamic_bounded_sichel_matches_full_scan_and_preserves_grad():
    counts = jnp.array([0.0, 1.0, 2.0, 5.0, 12.0, 25.0], dtype=jnp.float64)
    fast = np.asarray(dSICHEL(counts, 2.0, 0.5, 0.3, log=True))
    ref = np.asarray(
        jax.vmap(lambda y: _sichel_log_pdf(y, 2.0, 0.5, 0.3, max_steps=500))(counts)
    )
    np.testing.assert_allclose(fast, ref, rtol=1e-14, atol=1e-14)

    # Reverse-mode AD with respect to parameters still works seamlessly
    grad_mu = float(jax.grad(lambda m: dSICHEL(5.0, m, 0.5, 0.3, log=True))(2.0))
    assert np.isfinite(grad_mu)


def test_sichel_zisichel_zasichel_dynamic_deviance_and_scores_are_fast():
    y = jnp.array([0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 0.0], dtype=jnp.float64)
    mu = jnp.full_like(y, 2.5)
    sigma = jnp.full_like(y, 0.6)
    nu = jnp.full_like(y, -0.3)
    tau = jnp.full_like(y, 0.2)

    sichel = SICHEL()
    zisichel = ZISICHEL()
    zasichel = ZASICHEL()

    # Warm-up compile
    _ = sichel.g_dev_inc(y=y, mu=mu, sigma=sigma, nu=nu)
    _ = zisichel.g_dev_inc(y=y, mu=mu, sigma=sigma, nu=nu, tau=tau)
    _ = zasichel.g_dev_inc(y=y, mu=mu, sigma=sigma, nu=nu, tau=tau)

    t0 = time.perf_counter()
    dev_s = np.asarray(sichel.g_dev_inc(y=y, mu=mu, sigma=sigma, nu=nu))
    dev_zi = np.asarray(zisichel.g_dev_inc(y=y, mu=mu, sigma=sigma, nu=nu, tau=tau))
    dev_za = np.asarray(zasichel.g_dev_inc(y=y, mu=mu, sigma=sigma, nu=nu, tau=tau))
    dldv_s = np.asarray(sichel.score_functions["nu"](y=y, mu=mu, sigma=sigma, nu=nu))
    dldv_zi = np.asarray(zisichel.score_functions["nu"](y=y, mu=mu, sigma=sigma, nu=nu, tau=tau))
    dldv_za = np.asarray(zasichel.score_functions["nu"](y=y, mu=mu, sigma=sigma, nu=nu, tau=tau))
    elapsed = time.perf_counter() - t0

    for arr in (dev_s, dev_zi, dev_za, dldv_s, dldv_zi, dldv_za):
        assert arr.shape == y.shape
        assert np.all(np.isfinite(arr))
    # Previously each call scanned 100,000 steps (>10s total); now <2s warm
    assert elapsed < 5.0, f"expected fast dynamic scan, took {elapsed:.3f}s"
