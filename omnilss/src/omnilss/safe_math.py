# SPDX-License-Identifier: GPL-3.0-or-later
"""Numerical-stability helpers for OmniLSS."""

from __future__ import annotations

import jax.numpy as jnp


def safe_exp(x: jnp.ndarray, *, clip_min: float = -700.0, clip_max: float = 80.0) -> jnp.ndarray:
    x_arr = jnp.asarray(x)
    return jnp.exp(jnp.clip(x_arr, clip_min, clip_max))


def safe_log(x: jnp.ndarray, *, eps: float = 1e-12) -> jnp.ndarray:
    x_arr = jnp.asarray(x)
    return jnp.log(jnp.clip(x_arr, eps, jnp.inf))


def safe_log1p(x: jnp.ndarray, *, min_x: float = -0.999999) -> jnp.ndarray:
    x_arr = jnp.asarray(x)
    return jnp.log1p(jnp.maximum(x_arr, min_x))


def safe_softplus(x: jnp.ndarray) -> jnp.ndarray:
    x_arr = jnp.asarray(x)
    return jnp.logaddexp(x_arr, 0.0)


def safe_sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    x_arr = jnp.asarray(x)
    return 1.0 / (1.0 + jnp.exp(-jnp.clip(x_arr, -80.0, 80.0)))


def safe_divide(numerator: jnp.ndarray, denominator: jnp.ndarray, *, eps: float = 1e-12) -> jnp.ndarray:
    num = jnp.asarray(numerator)
    den = jnp.asarray(denominator)
    den_safe = jnp.where(jnp.abs(den) < eps, jnp.sign(den) * eps + (den == 0.0) * eps, den)
    result = num / den_safe
    return jnp.nan_to_num(result, nan=0.0, posinf=1e12, neginf=-1e12)


def safe_sqrt(x: jnp.ndarray, *, eps: float = 0.0) -> jnp.ndarray:
    x_arr = jnp.asarray(x)
    return jnp.sqrt(jnp.maximum(x_arr, eps))
