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


def gen_likelihood(object: GAMLSSModel) -> GenLikelihoodResult:
    """Compute per-observation log-likelihood style diagnostics from a fitted model."""

    if not is_gamlss(object):
        raise TypeError("this is not a gamlss object")

    family = object.family
    y = jnp.asarray(object.y, dtype=jnp.float64)
    params = [jnp.asarray(object.fitted_values[p], dtype=jnp.float64) for p in object.par]

    if hasattr(family, "d") and callable(family.d):
        loglik = jnp.asarray(family.d(y, *params, log=True), dtype=jnp.float64)
        dev_inc = -2.0 * loglik
    elif hasattr(family, "g_dev_inc") and callable(family.g_dev_inc):
        dev_inc = jnp.asarray(family.g_dev_inc(y, *params), dtype=jnp.float64)
        loglik = -0.5 * dev_inc
    else:
        raise ValueError(f"family {getattr(family, 'name', '<unknown>')} has no d() or g_dev_inc()")

    return GenLikelihoodResult(
        family=str(getattr(family, "name", "")),
        log_likelihood=np.asarray(loglik, dtype=np.float64),
        deviance_increment=np.asarray(dev_inc, dtype=np.float64),
        total_log_likelihood=float(jnp.sum(loglik)),
        total_deviance=float(jnp.sum(dev_inc)),
    )


__all__ = ["gen_likelihood", "GenLikelihoodResult"]
