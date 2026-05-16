# SPDX-License-Identifier: GPL-3.0-or-later
"""Numerical stability utilities for optimization routines."""

from __future__ import annotations

import jax.numpy as jnp


def sanitize_gradient(grad: jnp.ndarray, *, clip_value: float = 1e6) -> jnp.ndarray:
    """Replace non-finite gradient entries and clip magnitude."""
    g = jnp.nan_to_num(jnp.asarray(grad), nan=0.0, posinf=clip_value, neginf=-clip_value)
    return jnp.clip(g, -clip_value, clip_value)


def regularize_hessian(hessian: jnp.ndarray, *, lam: float = 1e-6) -> jnp.ndarray:
    """Apply Tikhonov regularization ``H + λI``."""
    h = jnp.asarray(hessian)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("Hessian must be a square 2D matrix.")
    n = h.shape[0]
    return h + lam * jnp.eye(n, dtype=h.dtype)


def step_halving(
    current_params: jnp.ndarray,
    candidate_params: jnp.ndarray,
    *,
    factor: float = 0.5,
) -> jnp.ndarray:
    """Return midpoint-like damped update between current and candidate params."""
    c = jnp.asarray(current_params)
    cand = jnp.asarray(candidate_params)
    if factor <= 0.0 or factor >= 1.0:
        raise ValueError("factor must be in (0, 1).")
    return c + factor * (cand - c)
