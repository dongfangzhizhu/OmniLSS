"""JAX-native building blocks for RS fitting.

This module provides standalone JAX helpers and an experimental rs_fit_jax entry
point without changing existing rs_algorithm.py behavior.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _jax_wls(X: jnp.ndarray, z: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """Solve weighted least squares using JAX lstsq."""
    sqrt_w = jnp.sqrt(jnp.clip(w, 1e-12, jnp.inf))
    Xw = X * sqrt_w[:, None]
    zw = z * sqrt_w
    beta, _, _, _ = jnp.linalg.lstsq(Xw, zw, rcond=None)
    return beta


_jax_wls = jax.jit(_jax_wls)


def _jax_penalized_wls(
    X: jnp.ndarray,
    z: jnp.ndarray,
    w: jnp.ndarray,
    penalty: jnp.ndarray,
) -> jnp.ndarray:
    """Solve penalized weighted least squares fully in JAX."""
    sqrt_w = jnp.sqrt(jnp.clip(w, 1e-12, jnp.inf))
    Xw = X * sqrt_w[:, None]
    zw = z * sqrt_w
    xtwx = Xw.T @ Xw
    xtwz = Xw.T @ zw
    ridge = 1e-10 * jnp.eye(X.shape[1], dtype=X.dtype)
    return jnp.linalg.solve(xtwx + penalty + ridge, xtwz)


_jax_penalized_wls = jax.jit(_jax_penalized_wls)
