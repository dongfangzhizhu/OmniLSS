"""Compatibility constructor for `GP` alias.

R reference: `gamlss.dist/R/GeneralisedPareto.R::GP`.
Survival: ``(1 + y / mu)**(-sigma)`` for positive mu and sigma.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import jax.random as random

from .ad import build_ad_family
from .links import log_derivative, log_inverse, log_link
from .families import FamilyDefinition


@dataclass(frozen=True)
class GPFamily(FamilyDefinition):
    """Generalised Pareto family using GAMLSS' ``mu, sigma`` parameterisation."""


def _log_pdf(y, mu, sigma):
    y, mu, sigma = (jnp.asarray(v, dtype=jnp.float64) for v in (y, mu, sigma))
    value = jnp.log(sigma) - jnp.log(mu) - (1.0 + sigma) * jnp.log1p(y / mu)
    return jnp.where((y > 0) & (mu > 0) & (sigma > 0), value, -jnp.inf)


def dGP(x, mu=1.0, sigma=1.0, log=False):
    value = _log_pdf(x, mu, sigma)
    return value if log else jnp.exp(value)


def pGP(q, mu=1.0, sigma=1.0, lower_tail=True, log_p=False):
    q, mu, sigma = (jnp.asarray(v, dtype=jnp.float64) for v in (q, mu, sigma))
    valid = (mu > 0) & (sigma > 0)
    log_survival = -sigma * jnp.log1p(jnp.maximum(q, 0) / mu)
    log_result = jnp.where(q <= 0, -jnp.inf if lower_tail else 0.0,
                           jnp.log(-jnp.expm1(log_survival)) if lower_tail else log_survival)
    result = log_result if log_p else (-jnp.expm1(log_survival) if lower_tail else jnp.exp(log_survival))
    return jnp.where(valid, result, jnp.nan)


def qGP(p, mu=1.0, sigma=1.0, lower_tail=True, log_p=False):
    p, mu, sigma = (jnp.asarray(v, dtype=jnp.float64) for v in (p, mu, sigma))
    valid = (mu > 0) & (sigma > 0)
    probability = jnp.exp(p) if log_p else p
    probability = jnp.where(lower_tail, probability, 1.0 - probability)
    log_one_minus_p = jnp.log1p(-probability)
    result = mu * jnp.expm1(-log_one_minus_p / sigma)
    result = jnp.where(probability <= 0, 0.0, jnp.where(probability >= 1, jnp.inf, result))
    return jnp.where(valid & (probability >= 0) & (probability <= 1), result, jnp.nan)


def rGP(key, n, mu=1.0, sigma=1.0):
    return qGP(random.uniform(key, (n,)), mu, sigma)


def GP() -> FamilyDefinition:
    return build_ad_family(GPFamily, "GP", ("mu", "sigma"), _log_pdf,
                           links={"mu": "log", "sigma": "log"},
                           link_functions={"mu": log_link, "sigma": log_link},
                           link_inverses={"mu": log_inverse, "sigma": log_inverse},
                           link_derivatives={"mu": log_derivative, "sigma": log_derivative},
                           d=dGP, p=pGP, q=qGP, r=rGP)
