"""Phase 1 autodiff validation helpers."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from .algorithms.jax_family_specs import get_jax_spec


def finite_difference_vs_autodiff_no_mu(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    eps: float = 1e-5,
) -> dict[str, float]:
    spec = get_jax_spec("NO")
    yj = jnp.asarray(y)
    muj = jnp.asarray(mu)
    sj = jnp.asarray(sigma)

    lp = spec.loglik_fn(yj, muj + eps, sj)
    lm = spec.loglik_fn(yj, muj - eps, sj)
    fd = (lp - lm) / (2.0 * eps)
    ad = spec.score_fns[0](yj, muj, sj)

    err = np.asarray(ad - fd)
    return {
        "max_abs_error": float(np.max(np.abs(err))),
        "mean_abs_error": float(np.mean(np.abs(err))),
    }
