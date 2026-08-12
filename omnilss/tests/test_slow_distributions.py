"""Slow-distribution JIT detection + BE/ZAGA fit smoke tests (Phase 4.4).

Converts the previous print-only script into real assertions: should_use_jit
behaviour across sample sizes, and BE/ZAGA fits converge with finite deviance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnilss import gamlss
from omnilss.fitting_jit import should_use_jit


@pytest.mark.parametrize(
    "dist, n, expected_jit",
    [
        ("BE", 50, False),
        ("BE", 500, True),
        ("BE", 5000, True),
        ("ZAGA", 50, False),
        ("ZAGA", 500, True),
        ("ZAGA", 5000, True),
    ],
)
def test_jit_detection_thresholds(dist, n, expected_jit):
    """JIT optimisation is enabled above the configured sample thresholds."""
    assert should_use_jit(dist, n) is expected_jit


def test_be_fit_converges():
    """BE model fits with finite deviance and reports convergence."""
    np.random.seed(42)
    n = 200
    x1 = np.random.randn(n)
    mu = 1 / (1 + np.exp(-(0.5 + 0.3 * x1)))
    y = np.random.beta(mu * 10, (1 - mu) * 10)
    model = gamlss("y ~ x1", family="BE", data={"y": y, "x1": x1})

    assert np.isfinite(model.deviance)
    assert model.additional_slots.get("converged", False) is True


def test_zaga_fit_converges():
    """ZAGA (zero-altered) model fits with finite deviance."""
    np.random.seed(42)
    n = 200
    x1 = np.random.randn(n)
    pi = 1 / (1 + np.exp(-(-1 + 0.3 * x1)))
    mu = np.exp(1.5 + 0.5 * x1)
    sigma = 0.6

    y = np.zeros(n)
    for i in range(n):
        if np.random.rand() < pi[i]:
            y[i] = 0
        else:
            shape = 1 / sigma**2
            y[i] = np.random.gamma(shape, mu[i] / shape)

    model = gamlss("y ~ x1", family="ZAGA", data={"y": y, "x1": x1})
    assert np.isfinite(model.deviance)
