"""R-aligned diagnostic top-deviance helper.

R reference: `gamlss/R/dtop.R::dtop`.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .model import GAMLSSModel
from .operations import is_gamlss


@dataclass(frozen=True)
class DTopResult:
    """Top-contributing observations by deviance increment."""

    indices: np.ndarray
    deviance_increment: np.ndarray
    weighted_deviance_increment: np.ndarray
    y: np.ndarray


def dtop(object: GAMLSSModel, n: int = 10, descending: bool = True) -> DTopResult:
    """Return top observations ranked by per-observation deviance increment."""

    if not is_gamlss(object):
        raise TypeError("this is not a gamlss object")
    if n <= 0:
        raise ValueError("n must be positive")

    family = object.family
    y = jnp.asarray(object.y, dtype=jnp.float64)
    params = [
        jnp.asarray(object.fitted_values[p], dtype=jnp.float64) for p in object.par
    ]

    if hasattr(family, "g_dev_inc") and callable(family.g_dev_inc):
        dev_inc = jnp.asarray(family.g_dev_inc(y, *params), dtype=jnp.float64)
    elif hasattr(family, "d") and callable(family.d):
        loglik = jnp.asarray(family.d(y, *params, log=True), dtype=jnp.float64)
        dev_inc = -2.0 * loglik
    else:
        raise ValueError(
            f"family {getattr(family, 'name', '<unknown>')} has no g_dev_inc() or d()"
        )

    dev_inc_np = np.asarray(dev_inc, dtype=np.float64)
    if object.weights is None:
        weights = np.ones_like(dev_inc_np, dtype=np.float64)
    else:
        weights = np.asarray(object.weights, dtype=np.float64)
        if weights.shape != dev_inc_np.shape:
            raise ValueError("weights must have same shape as observations")
    weighted_dev_inc = weights * dev_inc_np

    order = np.argsort(weighted_dev_inc)
    if descending:
        order = order[::-1]
    top = np.asarray(order[: min(n, order.shape[0])], dtype=np.int64)

    return DTopResult(
        indices=top,
        deviance_increment=np.asarray(dev_inc_np[top], dtype=np.float64),
        weighted_deviance_increment=np.asarray(weighted_dev_inc[top], dtype=np.float64),
        y=np.asarray(y[top], dtype=np.float64),
    )


__all__ = ["dtop", "DTopResult"]
