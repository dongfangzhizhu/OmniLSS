# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent numerical-contract validation for core distribution families."""

from __future__ import annotations

import numpy as np

from omnilss.distributions import NO, resolve_family


def test_no_dpqr_contracts_hold() -> None:
    family = resolve_family("NO")
    mu, sigma = 1.2, 0.7
    p_grid = np.linspace(0.05, 0.95, 11)
    q_vals = np.asarray(family.q(p_grid, mu=mu, sigma=sigma), dtype=np.float64)

    # Quantile monotonicity
    assert np.all(np.diff(q_vals) > 0.0)

    # p(q(p)) consistency
    p_back = np.asarray(family.p(q_vals, mu=mu, sigma=sigma), dtype=np.float64)
    np.testing.assert_allclose(p_back, p_grid, atol=1e-6, rtol=1e-6)


def test_no_expected_hessian_and_score_identity() -> None:
    family = NO()
    rng = np.random.default_rng(20260518)
    y = rng.normal(loc=0.8, scale=1.1, size=5000)

    mu = np.full_like(y, 0.8, dtype=np.float64)
    sigma = np.full_like(y, 1.1, dtype=np.float64)

    score_mu = np.asarray(family.score_functions["mu"](y, mu, sigma), dtype=np.float64)
    score_sigma = np.asarray(family.score_functions["sigma"](y, mu, sigma), dtype=np.float64)
    hess_mu = np.asarray(family.hessian_functions["mu"](y, mu, sigma), dtype=np.float64)
    hess_sigma = np.asarray(family.hessian_functions["sigma"](y, mu, sigma), dtype=np.float64)

    # Score expectations near zero under correctly specified model.
    assert abs(float(np.mean(score_mu))) < 0.05
    assert abs(float(np.mean(score_sigma))) < 0.05

    # Hessian signs must remain negative.
    assert np.all(hess_mu < 0.0)
    assert np.all(hess_sigma < 0.0)

    # Fisher identity approximation: E[score^2] ≈ -E[hessian]
    np.testing.assert_allclose(np.mean(score_mu**2), -np.mean(hess_mu), rtol=0.08)
    np.testing.assert_allclose(np.mean(score_sigma**2), -np.mean(hess_sigma), rtol=0.08)
