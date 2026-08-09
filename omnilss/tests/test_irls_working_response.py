"""Regression tests for the IRLS working-response clip fix.

Previously the working response was hard-clipped to ±1e6, which corrupted
IRLS for zero-inflated data (e.g. y=0 in Poisson): the clipped values
dominated the WLS solve and the fit did NOT converge to the Poisson MLE.
R's glim.fit applies no such clip.  These tests pin the correct behaviour.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnilss.distributions import PO
from omnilss.fitting import gamlss


def _poisson_data(n=100, seed=42):
    np.random.seed(seed)  # match the reference values below (np.random.seed)
    x = np.random.normal(0, 1, n)
    y = np.random.poisson(np.exp(0.5 + 0.8 * x))
    return {"y": y, "x": x}


def test_poisson_fit_converges_to_mle() -> None:
    """PO fit must equal the Poisson MLE (glm coefficients), not a corrupted
    solution caused by the old ±1e6 working-response clip."""
    data = _poisson_data()
    model = gamlss("y ~ x", family=PO(), data=data)
    coef = np.asarray(model.coefficients["mu"])
    # MLE for this data (verified against R glm): (0.54545, 0.73095)
    np.testing.assert_allclose(coef, [0.54545, 0.73095], rtol=1e-3, atol=1e-4)


def test_poisson_fit_matches_r_deviance() -> None:
    """Deviance (-2 logL) must match R's gamlss PO fit (~330.44)."""
    data = _poisson_data()
    model = gamlss("y ~ x", family=PO(), data=data)
    np.testing.assert_allclose(model.g_dev, 330.4365, rtol=1e-4)


def test_poisson_with_zeros_still_converges() -> None:
    """Zero-inflated-ish Poisson (many y=0) must still converge to MLE."""
    rng = np.random.default_rng(7)
    n = 120
    x = rng.normal(0, 1, n)
    y = rng.poisson(np.exp(0.2 + 0.5 * x))  # low mean -> many zeros
    assert np.sum(y == 0) > 10, "test data should contain zeros"
    model = gamlss("y ~ x", family=PO(), data={"y": y, "x": x})
    # Compare against glm MLE
    X = np.column_stack([np.ones(n), x])
    # IRLS to MLE reference
    beta = np.zeros(2)
    for _ in range(50):
        mu = np.exp(X @ beta)
        W = mu
        z = (X @ beta) + (y - mu) / mu
        beta_new, *_ = np.linalg.lstsq(X * np.sqrt(W)[:, None], z * np.sqrt(W), rcond=None)
        if np.allclose(beta_new, beta, atol=1e-9):
            beta = beta_new
            break
        beta = beta_new
    coef = np.asarray(model.coefficients["mu"])
    np.testing.assert_allclose(coef, beta, rtol=1e-3, atol=1e-3)
