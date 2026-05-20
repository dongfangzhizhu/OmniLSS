from __future__ import annotations

import numpy as np

from omnilss.fast_wls import solve_weighted_least_squares_cholesky


def test_fast_wls_matches_lstsq_for_well_conditioned_problem():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 4))
    beta_true = np.array([1.5, -2.0, 0.5, 3.0])
    y = X @ beta_true + 0.01 * rng.normal(size=80)
    w = np.linspace(0.5, 2.0, 80)

    beta_chol = solve_weighted_least_squares_cholesky(X, y, w)

    sqrt_w = np.sqrt(w)
    Xw = X * sqrt_w[:, None]
    yw = y * sqrt_w
    beta_ref, *_ = np.linalg.lstsq(Xw, yw, rcond=None)

    assert np.allclose(beta_chol, beta_ref, atol=1e-6)


def test_fast_wls_handles_rank_deficiency_with_ridge():
    X = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float64)
    y = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    w = np.ones(3, dtype=np.float64)
    beta = solve_weighted_least_squares_cholesky(X, y, w, ridge=1e-6)
    assert np.isfinite(beta).all()
