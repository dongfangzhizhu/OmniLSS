"""R-aligned generic likelihood helper.

R reference: `gamlss/R/gen-Likelihood.R::gen.likelihood`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from .model import GAMLSSModel
from .operations import is_gamlss


@dataclass(frozen=True)
class GenLikelihoodResult:
    family: str
    log_likelihood: np.ndarray
    deviance_increment: np.ndarray
    total_log_likelihood: float
    total_deviance: float
    weighted_total_log_likelihood: float
    weighted_total_deviance: float


def gen_likelihood(object: GAMLSSModel) -> GenLikelihoodResult:
    """Compute per-observation log-likelihood style diagnostics from a fitted model."""

    if not is_gamlss(object):
        raise TypeError("this is not a gamlss object")

    family = object.family
    y = jnp.asarray(object.y, dtype=jnp.float64)
    params = [
        jnp.asarray(object.fitted_values[p], dtype=jnp.float64) for p in object.par
    ]

    if hasattr(family, "d") and callable(family.d):
        loglik = jnp.asarray(family.d(y, *params, log=True), dtype=jnp.float64)
        dev_inc = -2.0 * loglik
    elif hasattr(family, "g_dev_inc") and callable(family.g_dev_inc):
        dev_inc = jnp.asarray(family.g_dev_inc(y, *params), dtype=jnp.float64)
        loglik = -0.5 * dev_inc
    else:
        raise ValueError(
            f"family {getattr(family, 'name', '<unknown>')} has no d() or g_dev_inc()"
        )

    loglik_np = np.asarray(loglik, dtype=np.float64)
    dev_inc_np = np.asarray(dev_inc, dtype=np.float64)
    if loglik_np.shape != dev_inc_np.shape:
        raise ValueError("log-likelihood and deviance increment shape mismatch")

    if object.weights is None:
        weights = np.ones_like(loglik_np, dtype=np.float64)
    else:
        weights = np.asarray(object.weights, dtype=np.float64)
        if weights.shape != loglik_np.shape:
            raise ValueError("weights must have same shape as observations")

    return GenLikelihoodResult(
        family=str(getattr(family, "name", "")),
        log_likelihood=loglik_np,
        deviance_increment=dev_inc_np,
        total_log_likelihood=float(np.sum(loglik_np)),
        total_deviance=float(np.sum(dev_inc_np)),
        weighted_total_log_likelihood=float(np.sum(weights * loglik_np)),
        weighted_total_deviance=float(np.sum(weights * dev_inc_np)),
    )


__all__ = ["gen_likelihood", "GenLikelihoodResult"]
