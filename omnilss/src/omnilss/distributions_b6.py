"""GAMLSS distributions: Batch 6 — Discrete Special Series.

These are specialized discrete distributions often used for count data with specific properties.

Distributions:
- PIG     (Poisson Inverse Gaussian)
- SICHEL  (Sichel distribution)
- SI      (Skellam distribution)
- DPO     (Double Poisson)
- DEL     (Delaporte distribution)
- YULE    (Yule distribution)
- WARING  (Waring distribution)

R source: gamlss.dist package
"""

from __future__ import annotations

# Enable float64 precision for numerical accuracy

from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, betaln, logsumexp
from jax.scipy.special import i0e, i1e  # Modified Bessel functions

from .ad import build_ad_family
from .families import FamilyDefinition
from .links import (
    identity_derivative,
    identity_inverse,
    identity_link,
    log_derivative,
    log_inverse,
    log_link,
    logit_derivative,
    logit_inverse,
    logit_link,
)
from .bessel import log_bessel_iv

# ------------------------------------------------------------------
# 1. PIG — Poisson Inverse Gaussian
#    R: dPIG(x, mu, sigma)
#    A Poisson distribution where the rate parameter follows an Inverse Gaussian distribution
# ------------------------------------------------------------------

@dataclass(frozen=True)
class PoissonInverseGaussianFamily(FamilyDefinition):
    """Poisson Inverse Gaussian distribution (`PIG`)."""


from .pig import logpmf as _pig_log_pdf


def PIG() -> PoissonInverseGaussianFamily:
    from .dpqr_functions import dPIG, pPIG, qPIG, rPIG
    return build_ad_family(
        family_class=PoissonInverseGaussianFamily,
        name="PIG",
        parameters=("mu", "sigma"),
        log_pdf_func=_pig_log_pdf,
        type_="Discrete",
        links={"mu": "log", "sigma": "log"},
        link_functions={"mu": log_link, "sigma": log_link},
        link_inverses={"mu": log_inverse, "sigma": log_inverse},
        link_derivatives={"mu": log_derivative, "sigma": log_derivative},
        d=dPIG,
        p=pPIG,
        q=qPIG,
        r=rPIG,
    )


# ------------------------------------------------------------------
# 2. SICHEL — Sichel distribution
#    R: dSICHEL(x, mu, sigma, nu)
#    A generalization of PIG with an additional shape parameter
# ------------------------------------------------------------------

@dataclass(frozen=True)
class SichelFamily(FamilyDefinition):
    """Sichel distribution (`SICHEL`)."""


# Static upper bound for the JAX scan.  It is deliberately a module constant:
# changing it requires recompilation, but keeps the kernel JIT-compatible.
SICHEL_DENSITY_MAX_COUNT = 100000
# Static support budget used by probability recursion.  Keep this separate
# from qSICHEL's candidate search cap.
SICHEL_PROBABILITY_MAX_COUNT = SICHEL_DENSITY_MAX_COUNT


def _sichel_log_pmf_sequence(mu, sigma, nu, max_count):
    """Return log PMFs for counts 0..max_count using one static scan."""
    from .bessel import log_bessel_kv
    eps = jnp.finfo(jnp.float64).eps
    inv_sigma = 1.0 / sigma
    log_k_nu = log_bessel_kv(nu, inv_sigma)
    c = jnp.exp(log_bessel_kv(nu + 1.0, inv_sigma) - log_k_nu)
    alpha = jnp.sqrt(1.0 + 2.0 * sigma * mu / c) / sigma
    log_k_alpha = log_bessel_kv(nu, alpha)
    lbes = log_bessel_kv(nu + 1.0, alpha) - log_k_alpha
    tofy0 = (mu / c) * jnp.power(1.0 + 2.0 * sigma * mu / c, -0.5) * jnp.exp(lbes)
    base = -nu * jnp.log(sigma * alpha) + log_k_alpha - log_k_nu
    log0 = base

    def step(carry, j):
        prev, log_mass = carry
        prev = jnp.clip(prev, eps, 1e100)
        jf = j.astype(jnp.float64)
        current = (c * sigma * (2.0 * (jf + nu) / mu) + 1.0 / prev)
        current = current * jnp.square(mu / (sigma * alpha * c))
        current = jnp.clip(current, eps, 1e100)
        next_log = log_mass + jnp.log(prev) - jnp.log(jf)
        return (current, next_log), next_log

    (_, _), tail = jax.lax.scan(step, (tofy0, log0), jnp.arange(1, max_count + 1))
    return jnp.concatenate((jnp.reshape(log0, (1,)), tail))


def _sichel_log_pdf(y, mu, sigma, nu):
    """Sichel distribution log-pdf.
    
    The Sichel distribution is a three-parameter generalization of PIG.
    
    R formula (from gamlss.dist):
    logfy = -lgamma(y+1) - nu*log(sigma*alpha) + sumlty + log(K_nu(alpha)) - log(K_nu(1/sigma))
    
    where:
    - c = K_{nu+1}(1/sigma) / K_nu(1/sigma)
    - alpha = sqrt(1 + 2*sigma*mu/c) / sigma
    - sumlty is computed via recursive formula (see tofySICHEL2.c)
    
    The recursive formula computes:
    tofY[0] = (mu/c) * (1 + 2*sigma*mu/c)^(-0.5) * exp(lbes)
    tofY[j] = (c*sigma*(2*(j+nu)/mu) + (1/tofY[j-1])) * (mu/(sigma*alpha*c))^2
    sumlty = sum(log(tofY[j-1])) for j=1 to y
    """
    from .bessel import log_bessel_kv
    
    eps = jnp.finfo(jnp.float64).eps
    # This function is the scalar kernel consumed by build_ad_family's vmap.
    # Do not accumulate with an array-shaped carry here: vmap supplies one
    # observation and one parameter tuple per invocation.
    y = jnp.asarray(y, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    valid = ((mu > 0) & (sigma > 0) & jnp.isfinite(mu)
             & jnp.isfinite(sigma) & jnp.isfinite(nu))
    support = (y >= 0) & jnp.isfinite(y) & (y == jnp.floor(y))
    undefined = jnp.isnan(y) | (support & (y > SICHEL_DENSITY_MAX_COUNT))
    mu = jnp.where(valid, mu, 1.)
    sigma = jnp.where(valid, sigma, 1.)
    nu = jnp.where(valid, nu, -.5)
    y = jnp.where(support & ~undefined, y, 0.)
    
    # Compute c = K_{nu+1}(1/sigma) / K_nu(1/sigma)
    inv_sigma = 1.0 / sigma
    log_k_nu = log_bessel_kv(nu, inv_sigma)
    log_k_nu_plus_1 = log_bessel_kv(nu + 1.0, inv_sigma)
    c = jnp.exp(log_k_nu_plus_1 - log_k_nu)
    
    # Compute alpha
    alpha = jnp.sqrt(1.0 + 2.0 * sigma * mu / c) / sigma
    
    # Compute lbes = log(K_{nu+1}(alpha) / K_nu(alpha))
    log_k_alpha_nu = log_bessel_kv(nu, alpha)
    log_k_alpha_nu_plus_1 = log_bessel_kv(nu + 1.0, alpha)
    lbes = log_k_alpha_nu_plus_1 - log_k_alpha_nu
    
    # Compute sumlty via recursive formula using lax.scan
    # tofY[0] = (mu/c) * (1 + 2*sigma*mu/c)^(-0.5) * exp(lbes)
    tofy_0 = (mu / c) * jnp.power(1.0 + 2.0 * sigma * mu / c, -0.5) * jnp.exp(lbes)
    
    # For y=0, sumlty = 0
    # For y>0, we need to compute the recursive sum
    # Since inputs are already scalar (vmapped by ad.py), compute directly
    y_int = jnp.clip(jnp.round(y).astype(jnp.int32), 0, SICHEL_DENSITY_MAX_COUNT)
    
    def scan_fun_masked(carry, j):
        tofy_prev, sum_log = carry
        # Clip tofy_prev to safe range to prevent extreme values
        tofy_prev_safe = jnp.clip(tofy_prev, eps, 1e100)
        
        # tofY[j] = (c*sigma*(2*(j+nu)/mu) + (1/tofY[j-1])) * (mu/(sigma*alpha*c))^2
        j_float = j.astype(jnp.float64)
        tofy_j = (c * sigma * (2.0 * (j_float + nu) / mu) + (1.0 / tofy_prev_safe)) * \
                 jnp.power(mu / (sigma * alpha * c), 2.0)
        
        # Clip tofy_j to safe range
        tofy_j = jnp.clip(tofy_j, eps, 1e100)
        
        # Only add to sum if j <= y_int
        log_term = jnp.where(j <= y_int, jnp.log(tofy_prev_safe), 0.0)
        sum_log = sum_log + log_term
        
        return (tofy_j, sum_log), None
    
    # Initial state: (tofY[0], sumlty=0)
    init_state = (tofy_0, jnp.asarray(0.0, dtype=jnp.float64))
    
    # Run scan with a static bound; terms after y are masked out.
    j_seq = jnp.arange(1, SICHEL_DENSITY_MAX_COUNT + 1)
    (_, sumlty), _ = jax.lax.scan(scan_fun_masked, init_state, j_seq)
    
    # For y=0, sumlty should be 0
    sumlty = jnp.where(y < 0.5, 0.0, sumlty)
    
    # Compute log-pdf
    # logfy = -lgamma(y+1) - nu*log(sigma*alpha) + sumlty + log(K_nu(alpha)) - log(K_nu(1/sigma))
    log_pdf = (
        -gammaln(y + 1.0) 
        - nu * jnp.log(sigma * alpha) 
        + sumlty 
        + log_k_alpha_nu 
        - log_k_nu
    )
    
    log_pdf = jnp.where(support, log_pdf, -jnp.inf)
    return jnp.where(valid & ~undefined, log_pdf, jnp.nan)


def SICHEL() -> SichelFamily:
    from .dpqr_functions import pSICHEL, qSICHEL, rSICHEL
    return build_ad_family(
        family_class=SichelFamily,
        name="SICHEL",
        parameters=("mu", "sigma", "nu"),
        log_pdf_func=_sichel_log_pdf,
        type_="Discrete",
        links={"mu": "log", "sigma": "log", "nu": "identity"},
        link_functions={"mu": log_link, "sigma": log_link, "nu": identity_link},
        link_inverses={"mu": log_inverse, "sigma": log_inverse, "nu": identity_inverse},
        link_derivatives={"mu": log_derivative, "sigma": log_derivative, "nu": identity_derivative},
        p=pSICHEL,
        q=qSICHEL,
        r=rSICHEL,
    )


# ------------------------------------------------------------------
# 3. SI — Skellam distribution
#    R: dSI(x, mu, sigma)
#    Difference of two Poisson random variables
# ------------------------------------------------------------------

@dataclass(frozen=True)
class SkellamFamily(FamilyDefinition):
    """R `SI` family, not the Skellam distribution.

    The historical implementation used a Skellam approximation. This has been
    replaced with the actual R `SI` parameterization.
    """


def _si_log_pdf(y, mu, sigma, nu=-0.5):
    """R `SI` log-pdf using the same recursive form as gamlss.dist."""
    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    y_int = jnp.clip(jnp.round(jnp.maximum(y, 0.0)).astype(jnp.int32), 0, 512)

    from .bessel import log_bessel_kv

    alpha = jnp.sqrt(1.0 + 2.0 * sigma * mu) / sigma
    lbes = log_bessel_kv(nu + 1.0, alpha) - log_bessel_kv(nu, alpha)
    tofy1 = mu * jnp.power(1.0 + 2.0 * sigma * mu, -0.5) * jnp.exp(lbes)

    def scan_fun(carry, j):
        prev, sum_log = carry
        prev = jnp.clip(prev, eps, 1e100)
        jf = j.astype(jnp.float64)
        current = (sigma * (2.0 * (jf + nu) / mu) + 1.0 / prev) * jnp.square(mu / (sigma * alpha))
        current = jnp.clip(current, eps, 1e100)
        sum_log = sum_log + jnp.where(j <= y_int, jnp.log(prev), 0.0)
        return (current, sum_log), None

    (_, sumlty), _ = jax.lax.scan(scan_fun, (tofy1, 0.0), jnp.arange(1, 513))
    log_pdf = (
        -gammaln(y_int + 1.0)
        - nu * jnp.log(sigma * alpha)
        + jnp.where(y_int == 0, 0.0, sumlty)
        + log_bessel_kv(nu, alpha)
        - log_bessel_kv(nu, 1.0 / sigma)
    )
    return log_pdf


SI_PROBABILITY_MAX_COUNT = 10000


def _si_log_pmf_grid(mu, sigma, nu):
    """Return SI log-PMFs from the recurrence used by ``gamlss.dist``.

    The masses are deliberately not renormalized: R computes the same raw
    recurrence, and normalizing a finite prefix changes every CDF value.
    """
    from .bessel import log_bessel_kv

    eps = jnp.finfo(jnp.float64).eps
    valid = (mu > 0.0) & (sigma > 0.0)
    safe_mu = jnp.where(valid, mu, 1.0)
    safe_sigma = jnp.where(valid, sigma, 1.0)
    alpha = jnp.sqrt(1.0 + 2.0 * safe_sigma * safe_mu) / safe_sigma
    lbes = log_bessel_kv(nu + 1.0, alpha) - log_bessel_kv(nu, alpha)
    initial_ratio = safe_mu * jnp.power(1.0 + 2.0 * safe_sigma * safe_mu, -0.5) * jnp.exp(lbes)
    log_zero = (
        -nu * jnp.log(safe_sigma * alpha)
        + log_bessel_kv(nu, alpha)
        - log_bessel_kv(nu, 1.0 / safe_sigma)
    )

    def step(state, count):
        previous_ratio, previous_log_mass = state
        count_float = count.astype(jnp.float64)
        log_mass = previous_log_mass + jnp.log(jnp.maximum(previous_ratio, eps)) - jnp.log(count_float)
        ratio = (
            safe_sigma * (2.0 * (count_float + nu) / safe_mu)
            + 1.0 / jnp.maximum(previous_ratio, eps)
        ) * jnp.square(safe_mu / (safe_sigma * alpha))
        return (ratio, log_mass), log_mass

    support = jnp.arange(SI_PROBABILITY_MAX_COUNT + 1, dtype=jnp.float64)
    _, remaining = jax.lax.scan(step, (initial_ratio, log_zero), support[1:])
    log_mass = jnp.concatenate((log_zero[None], remaining))
    return support, jnp.where(valid, log_mass, jnp.nan)


def _si_p(q, mu=0.5, sigma=0.02, nu=-0.5, lower_tail=True, log_p=False):
    """CDF for SI using the discrete first-support convention of R."""
    q, mu, sigma, nu = jnp.broadcast_arrays(*[
        jnp.asarray(value, dtype=jnp.float64) for value in (q, mu, sigma, nu)
    ])
    flat_q = q.reshape(-1)
    flat_mu, flat_sigma, flat_nu = (value.reshape(-1) for value in (mu, sigma, nu))
    support = jnp.arange(SI_PROBABILITY_MAX_COUNT + 1, dtype=jnp.float64)

    def one(qi, mui, sigmai, nui):
        _, log_mass = _si_log_pmf_grid(mui, sigmai, nui)
        cdf = jnp.sum(jnp.where(support <= jnp.floor(qi), jnp.exp(log_mass), 0.0))
        cdf = jnp.where(qi < 0.0, 0.0, cdf)
        cdf = jnp.where((qi > SI_PROBABILITY_MAX_COUNT) & jnp.isfinite(qi), jnp.nan, cdf)
        cdf = jnp.where(jnp.isposinf(qi), 1.0, cdf)
        return cdf

    cdf = jax.vmap(one)(flat_q, flat_mu, flat_sigma, flat_nu).reshape(q.shape)
    cdf = jnp.where(lower_tail, cdf, 1.0 - cdf)
    return jnp.log(cdf) if log_p else cdf


def _si_q(p, mu=0.5, sigma=0.02, nu=-0.5, lower_tail=True, log_p=False,
          max_value=SI_PROBABILITY_MAX_COUNT):
    """Quantile for SI: smallest non-negative integer with CDF >= p."""
    p, mu, sigma, nu = jnp.broadcast_arrays(*[
        jnp.asarray(value, dtype=jnp.float64) for value in (p, mu, sigma, nu)
    ])
    prob = jnp.exp(p) if log_p else p
    prob = jnp.where(lower_tail, prob, 1.0 - prob)
    support = jnp.arange(max_value + 1, dtype=jnp.float64)

    def one(pi, mui, sigmai, nui):
        _, log_mass = _si_log_pmf_grid(mui, sigmai, nui)
        cdf = jnp.cumsum(jnp.exp(log_mass[: max_value + 1]))
        reached = cdf >= pi
        result = jnp.argmax(reached).astype(jnp.float64)
        result = jnp.where(jnp.any(reached), result, jnp.nan)
        result = jnp.where(pi == 0.0, 0.0, result)
        result = jnp.where(pi >= 1.0, jnp.inf, result)
        result = jnp.where((pi < 0.0) | (pi > 1.0) | jnp.isnan(pi), jnp.nan, result)
        return result

    return jax.vmap(one)(prob.reshape(-1), mu.reshape(-1), sigma.reshape(-1), nu.reshape(-1)).reshape(p.shape)


def _si_r(key, n, mu=0.5, sigma=0.02, nu=-0.5):
    """Generate SI variates by inverse transform sampling."""
    import jax.random as jr
    return _si_q(jr.uniform(key, shape=(n,), dtype=jnp.float64), mu, sigma, nu)


def SI() -> SkellamFamily:
    """R-compatible Sichel (simplified) family with complete DPQR wiring."""
    return build_ad_family(
        family_class=SkellamFamily,
        name="SI",
        parameters=("mu", "sigma", "nu"),
        log_pdf_func=_si_log_pdf,
        type_="Discrete",
        links={"mu": "log", "sigma": "log", "nu": "identity"},
        link_functions={"mu": log_link, "sigma": log_link, "nu": identity_link},
        link_inverses={"mu": log_inverse, "sigma": log_inverse, "nu": identity_inverse},
        link_derivatives={"mu": log_derivative, "sigma": log_derivative, "nu": identity_derivative},
        p=_si_p,
        q=_si_q,
        r=_si_r,
    )


# ------------------------------------------------------------------
# 4. DPO — Double Poisson
#    R: dDPO(x, mu, sigma)
#    A generalization of Poisson allowing for over- or under-dispersion
# ------------------------------------------------------------------

@dataclass(frozen=True)
class DoublePoissonFamily(FamilyDefinition):
    """Double Poisson distribution (`DPO`)."""


def _dpo_log_pdf(y, mu, sigma):
    """Double Poisson log-pdf with an explicit numerical normalizing constant."""
    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    y = jnp.clip(jnp.round(jnp.maximum(y, 0.0)), 0.0, 512.0)

    def raw_log_mass(k):
        k = jnp.asarray(k, dtype=jnp.float64)
        log_k = jnp.where(k > 0.0, jnp.log(k), 0.0)
        return (
            -0.5 * jnp.log(sigma)
            - (mu / sigma)
            - gammaln(k + 1.0)
            + k * log_k
            - k
            + (k * jnp.log(mu)) / sigma
            + k / sigma
            - (k * log_k) / sigma
        )

    support = jnp.arange(0.0, 513.0)
    log_norm = -logsumexp(jax.vmap(raw_log_mass)(support))
    return raw_log_mass(y) + log_norm


def _dpo_log_masses(mu, sigma, max_value=512):
    """Return normalized DPO log masses on the numerical support grid."""
    support = jnp.arange(max_value + 1, dtype=jnp.float64)
    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.maximum(jnp.asarray(mu, dtype=jnp.float64), eps)
    sigma = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float64), eps)
    k = support.reshape((-1,) + (1,) * mu.ndim)
    log_k = jnp.where(k > 0.0, jnp.log(k), 0.0)
    raw = (
        -0.5 * jnp.log(sigma)
        - mu / sigma
        - gammaln(k + 1.0)
        + k * log_k - k
        + k * jnp.log(mu) / sigma + k / sigma
        - k * log_k / sigma
    )
    return support, raw - logsumexp(raw, axis=0, keepdims=True)


def _dpo_p(q, mu=1.0, sigma=1.0, lower_tail=True, log_p=False):
    q, mu, sigma = jnp.broadcast_arrays(jnp.asarray(q, dtype=jnp.float64), jnp.asarray(mu, dtype=jnp.float64), jnp.asarray(sigma, dtype=jnp.float64))
    support, log_mass = _dpo_log_masses(mu.reshape(-1), sigma.reshape(-1))
    cdf = jnp.sum(jnp.where(support[:, None] <= jnp.floor(q).reshape(1, -1), jnp.exp(log_mass), 0.0), axis=0)
    cdf = jnp.where(q.reshape(-1) < 0, 0.0, cdf)
    cdf = jnp.where(jnp.isposinf(q.reshape(-1)), 1.0, cdf)
    cdf = jnp.where(lower_tail, cdf, 1.0 - cdf)
    cdf = jnp.where(log_p, jnp.log(cdf), cdf)
    return cdf.reshape(q.shape)


def _dpo_q(p, mu=1.0, sigma=1.0, lower_tail=True, log_p=False, max_value=512):
    p, mu, sigma = jnp.broadcast_arrays(jnp.asarray(p, dtype=jnp.float64), jnp.asarray(mu, dtype=jnp.float64), jnp.asarray(sigma, dtype=jnp.float64))
    prob = jnp.exp(p) if log_p else p
    prob = jnp.where(lower_tail, prob, 1.0 - prob)
    support, log_mass = _dpo_log_masses(mu.reshape(-1), sigma.reshape(-1), max_value)
    cdf = jnp.cumsum(jnp.exp(log_mass), axis=0)
    result = support[jnp.argmax(cdf >= prob.reshape(1, -1), axis=0)]
    result = jnp.where(prob.reshape(-1) >= 1.0, jnp.inf, result)
    result = jnp.where(prob.reshape(-1) == 0.0, 0.0, result)
    result = jnp.where((prob.reshape(-1) < 0.0) | (prob.reshape(-1) > 1.0), jnp.nan, result)
    return result.reshape(p.shape)


def _dpo_r(key, n, mu=1.0, sigma=1.0):
    import jax.random as _jr
    return _dpo_q(_jr.uniform(key, shape=(n,), dtype=jnp.float64), mu, sigma)


def DPO() -> DoublePoissonFamily:
    return build_ad_family(
        family_class=DoublePoissonFamily,
        name="DPO",
        parameters=("mu", "sigma"),
        log_pdf_func=_dpo_log_pdf,
        type_="Discrete",
        p=_dpo_p,
        q=_dpo_q,
        r=_dpo_r,
        links={"mu": "log", "sigma": "log"},
        link_functions={"mu": log_link, "sigma": log_link},
        link_inverses={"mu": log_inverse, "sigma": log_inverse},
        link_derivatives={"mu": log_derivative, "sigma": log_derivative},
    )


# ------------------------------------------------------------------
# 5. DEL — Delaporte distribution
#    R: dDEL(x, mu, sigma, nu)
#    A shifted compound Poisson-Gamma distribution
# ------------------------------------------------------------------

@dataclass(frozen=True)
class DelaportFamily(FamilyDefinition):
    """Delaporte distribution (`DEL`)."""


def _del_log_masses(mu, sigma, nu, max_value=10000):
    """True DEL masses on 0..max_value; never renormalize truncated tails.

    Uses the gamlss.dist tofydel2 ratio recurrence. Parameters require
    mu > 0, sigma > 0 and 0 < nu < 1. As in R, sigma <= 1e-4
    uses the Poisson approximation.
    """
    mu, sigma, nu = jnp.broadcast_arrays(*[
        jnp.asarray(v, dtype=jnp.float64) for v in (mu, sigma, nu)
    ])
    valid = (mu > 0) & (sigma > 0) & (nu > 0) & (nu < 1)
    valid &= jnp.isfinite(mu) & jnp.isfinite(sigma) & jnp.isfinite(nu)
    mu = jnp.where(valid, mu, 1.0)
    sigma = jnp.where(valid, sigma, 1.0)
    nu = jnp.where(valid, nu, 0.5)
    scale = mu * sigma * (1.0 - nu)
    ratio = scale / (1.0 + scale)
    lam = mu * nu
    initial = lam + mu * (1.0 - nu) / (1.0 + scale)
    log_zero = -lam - jnp.log1p(scale) / sigma

    def step(state, j):
        previous, log_mass = state
        log_mass = log_mass + jnp.log(previous) - jnp.log(j)
        current = ratio * (j + lam - lam * j / previous) + mu / (1.0 + scale)
        return (current, log_mass), log_mass

    support = jnp.arange(max_value + 1, dtype=jnp.float64)
    _, rest = jax.lax.scan(step, (initial, log_zero), support[1:])
    masses = jnp.concatenate((log_zero[None, ...], rest), axis=0)
    x = support.reshape((-1,) + (1,) * mu.ndim)
    poisson = x * jnp.log(mu) - mu - gammaln(x + 1.0)
    masses = jnp.where(sigma <= 1e-4, poisson, masses)
    return support, jnp.where(valid, masses, jnp.nan)


def _del_log_pdf(y, mu, sigma, nu):
    """DEL log PMF, differentiable in parameters, evaluated up to 10000.

    Finite integer observations above the numerical limit return NaN,
    rather than silently clipping to a different observation.
    """
    y, mu, sigma, nu = jnp.broadcast_arrays(*[
        jnp.asarray(v, dtype=jnp.float64) for v in (y, mu, sigma, nu)
    ])
    _, masses = _del_log_masses(mu, sigma, nu)
    index = jnp.clip(jnp.nan_to_num(y), 0, 10000).astype(jnp.int32)
    result = jnp.take_along_axis(masses, index[None, ...], axis=0)[0]
    result = jnp.where(y > 10000, jnp.nan, result)
    result = jnp.where((y < 0) | (y != jnp.floor(y)) | jnp.isinf(y), -jnp.inf, result)
    return jnp.where(jnp.isnan(y), jnp.nan, result)


def _del_p(q, mu=1.0, sigma=1.0, nu=0.5, lower_tail=True, log_p=False):
    """DEL CDF; finite queries above 10000 return NaN (numerical limit)."""
    q, mu, sigma, nu = jnp.broadcast_arrays(*[
        jnp.asarray(v, dtype=jnp.float64) for v in (q, mu, sigma, nu)
    ])
    _, masses = _del_log_masses(mu, sigma, nu)
    cdf = jnp.cumsum(jnp.exp(masses), axis=0)
    index = jnp.clip(jnp.nan_to_num(jnp.floor(q)), 0, 10000).astype(jnp.int32)
    result = jnp.take_along_axis(cdf, index[None, ...], axis=0)[0]
    result = jnp.clip(result, 0.0, 1.0)
    result = jnp.where(q < 0, 0.0, result)
    result = jnp.where(q >= 10001, jnp.nan, result)
    result = jnp.where(jnp.isposinf(q), 1.0, result)
    result = jnp.where(jnp.isnan(q), jnp.nan, result)
    result = jnp.where(lower_tail, result, 1.0 - result)
    return jnp.log(result) if log_p else result


def _del_q(p, mu=1.0, sigma=1.0, nu=0.5, lower_tail=True, log_p=False, max_value=10000):
    """First integer with CDF >= p; unresolved finite quantiles return NaN."""
    p, mu, sigma, nu = jnp.broadcast_arrays(*[
        jnp.asarray(v, dtype=jnp.float64) for v in (p, mu, sigma, nu)
    ])
    prob = jnp.exp(p) if log_p else p
    prob = jnp.where(lower_tail, prob, 1.0 - prob)
    support, masses = _del_log_masses(mu, sigma, nu, max_value)
    reached = jnp.cumsum(jnp.exp(masses), axis=0) >= prob
    result = support[jnp.argmax(reached, axis=0)]
    result = jnp.where(jnp.any(reached, axis=0), result, jnp.nan)
    result = jnp.where(prob == 0.0, 0.0, result)
    result = jnp.where(prob == 1.0, jnp.inf, result)
    return jnp.where((prob < 0) | (prob > 1) | jnp.isnan(prob), jnp.nan, result)


def _del_r(key, n, mu=1.0, sigma=1.0, nu=0.5):
    """Sample the Poisson-Gamma mixture without a finite support cutoff."""
    import jax.random as jr
    mu, sigma, nu = jnp.broadcast_arrays(*[
        jnp.broadcast_to(jnp.asarray(v, dtype=jnp.float64), (n,))
        for v in (mu, sigma, nu)
    ])
    gamma_key, poisson_key = jr.split(key)
    mixing = jr.gamma(gamma_key, 1.0 / sigma) * mu * sigma * (1.0 - nu)
    rate = jnp.where(sigma <= 1e-4, mu, mu * nu + mixing)
    sample = jr.poisson(poisson_key, rate).astype(jnp.float64)
    valid = (mu > 0) & (sigma > 0) & (nu > 0) & (nu < 1)
    valid &= jnp.isfinite(mu) & jnp.isfinite(sigma) & jnp.isfinite(nu)
    return jnp.where(valid, sample, jnp.nan)


def DEL() -> DelaportFamily:
    return build_ad_family(
        family_class=DelaportFamily,
        name="DEL",
        parameters=("mu", "sigma", "nu"),
        log_pdf_func=_del_log_pdf,
        type_="Discrete",
        p=_del_p,
        q=_del_q,
        r=_del_r,
        links={"mu": "log", "sigma": "log", "nu": "logit"},
        link_functions={"mu": log_link, "sigma": log_link, "nu": logit_link},
        link_inverses={"mu": log_inverse, "sigma": log_inverse, "nu": logit_inverse},
        link_derivatives={"mu": log_derivative, "sigma": log_derivative, "nu": logit_derivative},
    )


# ------------------------------------------------------------------
# 6. YULE — Yule distribution
#    R: dYULE(x, mu)
#    A discrete distribution arising from a pure birth process
# ------------------------------------------------------------------

@dataclass(frozen=True)
class YuleFamily(FamilyDefinition):
    """Yule distribution (`YULE`)."""


def _yule_log_pdf(y, mu):
    """Yule distribution log-pdf.
    
    The Yule distribution is a discrete distribution with support on {0, 1, 2, ...}.
    
    R formula:
        lambda = (mu + 1) / mu
        log p(y) = lbeta(lambda + 1, y + 1) - lbeta(lambda, 1)
    """
    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.maximum(mu, eps)
    
    # Transform mu to lambda parameter
    lambda_param = (mu + 1.0) / mu
    
    # log p(y) = lbeta(lambda + 1, y + 1) - lbeta(lambda, 1)
    log_pdf = betaln(lambda_param + 1.0, y + 1.0) - betaln(lambda_param, 1.0)
    
    return log_pdf


def _yule_p(q, mu=2.0, lower_tail=True, log_p=False):
    """Yule CDF, equivalent to R's cumulative ``dYULE`` definition.

    The survival probability has the closed form
    ``lambda * beta(lambda, floor(q) + 2)``.  Using it avoids repeatedly
    summing the heavy-tailed PMF and remains stable for large quantiles.
    """
    import numpy as _np
    from scipy.special import betaln as _bl

    q_arr, mu_arr = _np.broadcast_arrays(
        _np.asarray(q, dtype=_np.float64), _np.asarray(mu, dtype=_np.float64)
    )
    if _np.any(mu_arr <= 0.0):
        raise ValueError("mu must be > 0")

    q_floor = _np.floor(q_arr)
    lam = (mu_arr + 1.0) / mu_arr
    survival = _np.exp(_np.log(lam) + _bl(lam, q_floor + 2.0))
    with _np.errstate(divide="ignore", invalid="ignore"):
        cdf = _np.where(q_arr < 0.0, 0.0, -_np.expm1(_np.log(survival)))
    cdf = _np.where(_np.isposinf(q_arr), 1.0, _np.clip(cdf, 0.0, 1.0))
    if not lower_tail:
        cdf = 1.0 - cdf
    if log_p:
        with _np.errstate(divide="ignore"):
            cdf = _np.log(cdf)
    return jnp.asarray(cdf, dtype=jnp.float64)


def _yule_q(p, mu=2.0, lower_tail=True, log_p=False, max_value=10000):
    """Yule quantile using binary search over the closed-form CDF."""
    import numpy as _np
    from scipy.special import betaln as _bl

    p_arr = _np.asarray(p, dtype=_np.float64)
    if log_p:
        p_arr = _np.exp(p_arr)
    if not lower_tail:
        p_arr = 1.0 - p_arr
    p_arr, mu_arr = _np.broadcast_arrays(p_arr, _np.asarray(mu, dtype=_np.float64))
    if _np.any(mu_arr <= 0.0):
        raise ValueError("mu must be > 0")

    result = _np.empty(p_arr.shape, dtype=_np.float64)
    for index in _np.ndindex(p_arr.shape):
        pi = float(p_arr[index])
        mui = float(mu_arr[index])
        if pi < 0.0 or pi > 1.0 or _np.isnan(pi):
            result[index] = _np.nan
            continue
        if pi == 0.0:
            result[index] = 0.0
            continue
        if pi == 1.0:
            result[index] = _np.inf
            continue

        lam = (mui + 1.0) / mui
        low, high = 0, int(max_value)
        while low < high:
            mid = (low + high) // 2
            survival = _np.exp(_np.log(lam) + _bl(lam, mid + 2.0))
            cdf = -_np.expm1(_np.log(survival))
            if cdf >= pi:
                high = mid
            else:
                low = mid + 1
        result[index] = float(low)
    return jnp.asarray(result, dtype=jnp.float64)


def _yule_r(key, n, mu=2.0):
    import jax.numpy as _jnp
    import jax.random as _jr

    u = _jr.uniform(key, shape=(n,), dtype=_jnp.float64)
    return _yule_q(u, mu)


def YULE() -> YuleFamily:
    return build_ad_family(
        family_class=YuleFamily,
        name="YULE",
        parameters=("mu",),
        log_pdf_func=_yule_log_pdf,
        type_="Discrete",
        p=_yule_p,
        q=_yule_q,
        r=_yule_r,
        links={"mu": "log"},
        link_functions={"mu": log_link},
        link_inverses={"mu": log_inverse},
        link_derivatives={"mu": log_derivative},
    )


# ------------------------------------------------------------------
# 7. WARING — Waring distribution
#    R: dWARING(x, mu, sigma)
#    A generalization of the Yule distribution
# ------------------------------------------------------------------

@dataclass(frozen=True)
class WaringFamily(FamilyDefinition):
    """Waring distribution (`WARING`)."""


def _waring_log_pdf(y, mu, sigma):
    """Waring distribution log-pdf.
    
    The Waring distribution is a generalization of the Yule distribution.
    
    R formula:
        log p(y) = lbeta(y + mu/sigma, 1/sigma + 2) - lbeta(mu/sigma, 1/sigma + 1)
    """
    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.maximum(mu, eps)
    # `lbeta()` becomes numerically unstable for sigma values extremely close to
    # zero because the transformed shape parameters grow like 1 / sigma.
    sigma = jnp.maximum(sigma, 1e-8)
    
    # Transform parameters
    a = mu / sigma
    b = 1.0 / sigma
    
    # log p(y) = lbeta(y + a, b + 2) - lbeta(a, b + 1)
    log_pdf = betaln(y + a, b + 2.0) - betaln(a, b + 1.0)
    
    return log_pdf


def _waring_p(q, mu=2.0, sigma=2.0, lower_tail=True, log_p=False):
    """CDF of WARING, following R's finite discrete summation.

    R's implementation evaluates ``sum(dWARING(0:q))``.  The fixed scan keeps
    this operation JAX-compatible while masking terms above each requested q.
    ``10000`` is the same upper bound used by ``qWARING`` in the R source.
    """
    q, mu, sigma = jnp.broadcast_arrays(
        jnp.asarray(q, dtype=jnp.float64),
        jnp.asarray(mu, dtype=jnp.float64),
        jnp.asarray(sigma, dtype=jnp.float64),
    )
    q_int = jnp.floor(q).astype(jnp.int32)
    support = jnp.arange(10001, dtype=jnp.float64)
    log_mass = _waring_log_pdf(support[:, None], mu.reshape(1, -1), sigma.reshape(1, -1))
    mass = jnp.exp(log_mass)
    cdf = jnp.sum(jnp.where(support[:, None] <= q_int.reshape(1, -1), mass, 0.0), axis=0)
    cdf = jnp.where(q < 0, 0.0, cdf)
    cdf = jnp.where(jnp.isposinf(q), 1.0, cdf)
    cdf = jnp.where(lower_tail, cdf, 1.0 - cdf)
    cdf = jnp.where(log_p, jnp.log(cdf), cdf)
    return cdf.reshape(q.shape)


def _waring_q(p, mu=2.0, sigma=2.0, lower_tail=True, log_p=False, max_value=10000):
    """Discrete inverse CDF using the same first-crossing rule as R."""
    p, mu, sigma = jnp.broadcast_arrays(
        jnp.asarray(p, dtype=jnp.float64),
        jnp.asarray(mu, dtype=jnp.float64),
        jnp.asarray(sigma, dtype=jnp.float64),
    )
    prob = jnp.exp(p) if log_p else p
    prob = jnp.where(lower_tail, prob, 1.0 - prob)
    support = jnp.arange(max_value + 1, dtype=jnp.float64)
    masses = jnp.exp(_waring_log_pdf(support[:, None], mu.reshape(1, -1), sigma.reshape(1, -1)))
    cdf = jnp.cumsum(masses, axis=0)
    first = jnp.argmax(cdf >= prob.reshape(1, -1), axis=0).astype(jnp.float64)
    result = jnp.where(prob >= 1.0, jnp.inf, first)
    result = jnp.where(prob == 0.0, 0.0, result)
    result = jnp.where((prob < 0.0) | (prob > 1.0), jnp.nan, result)
    return result.reshape(p.shape)


def _waring_r(key, n, mu=2.0, sigma=2.0):
    """Generate WARING variates by inverse transform sampling."""
    import jax.random as _jr
    return _waring_q(_jr.uniform(key, shape=(n,), dtype=jnp.float64), mu, sigma)


def WARING() -> WaringFamily:
    """Waring distribution.
    
    Note: WARING uses observed Hessian with safeguards for numerical stability.
    The observed Hessian can be positive for some observations, so we ensure
    it's always negative by taking the minimum with a small negative value.
    """
    family = build_ad_family(
        family_class=WaringFamily,
        name="WARING",
        parameters=("mu", "sigma"),
        log_pdf_func=_waring_log_pdf,
        type_="Discrete",
        p=_waring_p,
        q=_waring_q,
        r=_waring_r,
        links={"mu": "log", "sigma": "log"},
        link_functions={"mu": log_link, "sigma": log_link},
        link_inverses={"mu": log_inverse, "sigma": log_inverse},
        link_derivatives={"mu": log_derivative, "sigma": log_derivative},
    )
    
    # Store original Hessian functions
    original_hessian_mu = family.hessian_functions["mu"]
    original_hessian_sigma = family.hessian_functions["sigma"]
    
    def _safe_hessian_mu(*args, **kwargs):
        """Safe Hessian for mu: ensure it's always negative."""
        h = original_hessian_mu(*args, **kwargs)
        # Ensure Hessian is negative (for concave log-likelihood)
        # If positive, clip to small negative value
        return jnp.where(h > 0.0, -1e-7, h)
    
    def _safe_hessian_sigma(*args, **kwargs):
        """Safe Hessian for sigma: ensure it's always negative."""
        h = original_hessian_sigma(*args, **kwargs)
        # Ensure Hessian is negative (for concave log-likelihood)
        # If positive, clip to small negative value
        return jnp.where(h > 0.0, -1e-7, h)
    
    # Replace Hessian functions
    family.hessian_functions["mu"] = _safe_hessian_mu
    family.hessian_functions["sigma"] = _safe_hessian_sigma
    
    return family
