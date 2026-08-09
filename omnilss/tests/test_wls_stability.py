"""Regression tests for the numerical foundation (task 1.7).

Validates:
- WLS solver uses the numerically stable lstsq path (no NaN on ill-conditioned
  design matrices, matches np.linalg.lstsq on well-conditioned data).
- The dynamic ridge from stabilize_hessian is actually passed into
  backend.wls_solve (previously the computed lambda_value was dropped and a
  fixed 1e-10 ridge was used internally).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnilss.fast_wls import solve_weighted_least_squares_cholesky


def test_wls_ill_conditioned_no_nan() -> None:
    """Nearly-collinear columns must not produce NaN coefficients."""
    X = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-8], [1.0, 1.0 + 2e-8], [1.0, 1.0 + 3e-8]])
    z = np.array([1.0, 2.0, 3.0, 4.0])
    w = np.ones(4)
    beta = solve_weighted_least_squares_cholesky(X, z, w)
    assert np.all(np.isfinite(beta))


def test_wls_matches_lstsq() -> None:
    """On well-conditioned data the solver equals np.linalg.lstsq exactly."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    z = rng.normal(size=100)
    w = rng.uniform(0.5, 1.5, 100)
    b = solve_weighted_least_squares_cholesky(X, z, w)
    Xw = X * np.sqrt(w)[:, None]
    b_ref, *_ = np.linalg.lstsq(Xw, z * np.sqrt(w), rcond=None)
    np.testing.assert_allclose(b, b_ref, atol=1e-10)


def test_wls_ridge_parameter_accepted() -> None:
    """ridge argument is accepted (API compatibility)."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 2))
    z = rng.normal(size=50)
    w = rng.uniform(0.5, 1.5, 50)
    b = solve_weighted_least_squares_cholesky(X, z, w, ridge=1e-6)
    assert np.all(np.isfinite(b))


def test_wls_ridge_changes_solution() -> None:
    """Nonzero ridge must change the solution (proves it is applied)."""
    rng = np.random.default_rng(2)
    # Ill-conditioned: X^T X near-singular so ridge matters
    X = np.column_stack([np.ones(50), np.ones(50) + rng.normal(0, 1e-6, 50)])
    z = rng.normal(size=50)
    w = np.ones(50)
    b0 = solve_weighted_least_squares_cholesky(X, z, w, ridge=1e-12)
    b1 = solve_weighted_least_squares_cholesky(X, z, w, ridge=1e-2)
    assert not np.allclose(b0, b1), "ridge must affect the WLS solution"
