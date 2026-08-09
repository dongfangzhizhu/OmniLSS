"""Regression tests for the global float64 (x64) precision policy (task 0.7).

Validates that importing omnilss enables JAX float64 globally, so that
``jnp.asarray(..., dtype=jnp.float64)`` calls throughout the codebase are
not silently truncated to float32.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp
import numpy as np

import omnilss  # noqa: F401  (must enable x64 on import)


def test_x64_enabled_after_import() -> None:
    assert jax.config.jax_enable_x64 is True


def test_float64_array_not_truncated() -> None:
    a = jnp.asarray([0.1], dtype=jnp.float64)
    assert a.dtype == jnp.float64


def test_fitting_produces_float64() -> None:
    """A fitted model's parameter values must be float64 (not float32)."""
    from omnilss.distributions import NO
    from omnilss.fitting import gamlss

    rng = np.random.default_rng(0)
    n = 60
    x = rng.uniform(0, 1, n)
    y = 1.0 + 2.0 * x + rng.normal(0, 0.4, n)
    model = gamlss("y ~ x", family=NO(), data={"y": y, "x": x}, method="RS")
    mu = np.asarray(model.fitted_values["mu"])
    assert mu.dtype == np.float64, f"fitted mu must be float64, got {mu.dtype}"
