# SPDX-License-Identifier: GPL-3.0-or-later
"""Numerical-stability helpers for OmniLSS.

This module centralizes common safe math primitives used across fitting and
family implementations to reduce ad-hoc clipping and NaN handling.
"""

from __future__ import annotations

import jax.numpy as jnp


def safe_exp(x: jnp.ndarray, *, clip_min: float = -700.0, clip_max: float = 80.0) -> jnp.ndarray:
    """Exponentiation with pre-clipping to reduce overflow/underflow risk."""
    x_arr = jnp.asarray(x)
    return jnp.exp(jnp.clip(x_arr, clip_min, clip_max))


def safe_log(x: jnp.ndarray, *, eps: float = 1e-12) -> jnp.ndarray:
    """Natural log with lower bound to avoid ``log(0)``."""
    x_arr = jnp.asarray(x)
    return jnp.log(jnp.clip(x_arr, eps, jnp.inf))


def safe_softplus(x: jnp.ndarray) -> jnp.ndarray:
    """Stable softplus implementation via ``logaddexp``."""
    x_arr = jnp.asarray(x)
    return jnp.logaddexp(x_arr, 0.0)


def safe_divide(numerator: jnp.ndarray, denominator: jnp.ndarray, *, eps: float = 1e-12) -> jnp.ndarray:
    """Elementwise division with denominator floor and finite-value sanitization."""
    num = jnp.asarray(numerator)
    den = jnp.asarray(denominator)
    den_safe = jnp.where(jnp.abs(den) < eps, jnp.sign(den) * eps + (den == 0.0) * eps, den)
    result = num / den_safe
    return jnp.nan_to_num(result, nan=0.0, posinf=1e12, neginf=-1e12)
