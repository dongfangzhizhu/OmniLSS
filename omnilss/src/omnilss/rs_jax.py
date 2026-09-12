"""JAX linear algebra helpers used by the RS runtime."""

from __future__ import annotations

import jax.numpy as jnp


def _jax_wls(X, z, w):
    """Solve weighted least squares using a JAX-compatible QR/SVD solve."""
    sqrt_w = jnp.sqrt(w)
    return jnp.linalg.lstsq(X * sqrt_w[:, None], z * sqrt_w, rcond=None)[0]


def _jax_penalized_wls(X, z, w, penalty):
    """Solve penalized weighted least squares in normal-equation form."""
    sqrt_w = jnp.sqrt(w)
    Xw = X * sqrt_w[:, None]
    zw = z * sqrt_w
    return jnp.linalg.solve(Xw.T @ Xw + penalty, Xw.T @ zw)


__all__ = ["_jax_wls", "_jax_penalized_wls"]