"""Phase 1 stable parameterization transforms."""

from __future__ import annotations

import jax.numpy as jnp

from .safe_math import safe_sigmoid, safe_softplus


def positive_transform(x):
    return safe_softplus(jnp.asarray(x))


def bounded_transform(x, lo=0.0, hi=1.0):
    s = safe_sigmoid(jnp.asarray(x))
    return lo + (hi - lo) * s


def ordered_transform(raw):
    r = jnp.asarray(raw)
    if r.ndim != 1:
        raise ValueError("ordered_transform expects 1D input")
    base = r[0]
    inc = safe_softplus(r[1:])
    return jnp.concatenate([jnp.array([base]), base + jnp.cumsum(inc)])
