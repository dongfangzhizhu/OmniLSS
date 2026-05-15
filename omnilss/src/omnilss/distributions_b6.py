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


def _pig_log_pdf(y, mu, sigma):
    """Poisson Inverse Gaussian log-pdf aligned to R's recursive implementation."""
    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    y_int = jnp.clip(jnp.round(jnp.maximum(y, 0.0)).astype(jnp.int32), 0, 512)
    base = (1.0 - jnp.sqrt(1.0 + 2.0 * sigma * mu)) / sigma
    tofy1 = mu * jnp.power(1.0 + 2.0 * sigma * mu, -0.5)

    def scan_fun(carry, j):
        prev, sum_log = carry
        prev = jnp.clip(prev, eps, 1e100)
        jf = j.astype(jnp.float64)
        current = (sigma * (2.0 * jf - 1.0) / mu + 1.0 / prev) * jnp.square(tofy1)
        current = jnp.clip(current, eps, 1e100)
        sum_log = sum_log + jnp.where(j <= y_int, jnp.log(prev), 0.0)
        return (current, sum_log), None

    (_, sumlty), _ = jax.lax.scan(scan_fun, (tofy1, 0.0), jnp.arange(1, 513))
    log_pdf = -gammaln(y_int + 1.0) + base + jnp.where(y_int == 0, 0.0, sumlty)
    return log_pdf


def PIG() -> PoissonInverseGaussianFamily:
    from .dpqr_functions import pPIG, qPIG, rPIG
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
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    y = jnp.maximum(y, 0.0)
    
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
    y_int = jnp.clip(jnp.round(y).astype(jnp.int32), 0, 100)
    
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
    init_state = (tofy_0, 0.0)
    
    # Run scan from j=1 to 100 with masking
    j_seq = jnp.arange(1, 101)
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
    
    return log_pdf


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


def SI() -> SkellamFamily:
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


def DPO() -> DoublePoissonFamily:
    return build_ad_family(
        family_class=DoublePoissonFamily,
        name="DPO",
        parameters=("mu", "sigma"),
        log_pdf_func=_dpo_log_pdf,
        type_="Discrete",
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


def _del_log_pdf(y, mu, sigma, nu):
    """Delaporte distribution log-pdf.
    
    The Delaporte distribution is a shifted compound Poisson-Gamma distribution.
    
    R formula (from gamlss.dist):
    logpy0 = -mu*nu - (1/sigma)*log(1 + mu*sigma*(1-nu))
    logfy = logpy0 - lgamma(y+1) + sumlty
    
    where sumlty is computed via recursive formula (see tofydel2.c):
    tofY[0] = mu*nu + mu*(1-nu)/(1 + mu*sigma*(1-nu))
    tofY[j] = (j + mu*nu + 1/(sigma*(1-nu)) - (mu*nu*j)/tofY[j-1]) / (1 + 1/(mu*sigma*(1-nu)))
    sumlty = sum(log(tofY[j-1])) for j=1 to y
    """
    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.clip(nu, eps, 1.0 - eps)  # nu must be in (0, 1)
    y = jnp.maximum(y, 0.0)
    
    # Compute logpy0 = -mu*nu - (1/sigma)*log(1 + mu*sigma*(1-nu))
    logpy0 = -mu * nu - (1.0 / sigma) * jnp.log(1.0 + mu * sigma * (1.0 - nu))
    
    # Compute sumlty via recursive formula using lax.scan
    # tofY[0] = mu*nu + mu*(1-nu)/(1 + mu*sigma*(1-nu))
    tofy_0 = mu * nu + mu * (1.0 - nu) / (1.0 + mu * sigma * (1.0 - nu))
    
    # For y=0, sumlty = 0
    # For y>0, compute recursively using lax.scan
    y_int = jnp.clip(jnp.round(y).astype(jnp.int32), 0, 100)
    
    def scan_fun(carry, j):
        tofy_prev, sum_log = carry
        j_float = j.astype(jnp.float64)
        
        # Clip tofy_prev to safe range to prevent extreme values
        tofy_prev_safe = jnp.clip(tofy_prev, eps, 1e100)
        
        # dum = 1 + 1/(mu*sigma*(1-nu))
        dum = 1.0 + 1.0 / (mu * sigma * (1.0 - nu))
        
        # tofY[j] = (j + mu*nu + 1/(sigma*(1-nu)) - (mu*nu*j)/tofY[j-1]) / dum
        tofy_j = (j_float + mu * nu + 1.0 / (sigma * (1.0 - nu)) - 
                  (mu * nu * j_float) / tofy_prev_safe) / dum
        
        # Clip tofy_j to safe range
        tofy_j = jnp.clip(tofy_j, eps, 1e100)
        
        # Add log(tofY[j-1]) to sum using safe value
        sum_log = sum_log + jnp.log(tofy_prev_safe)
        return (tofy_j, sum_log), None
    
    # Initial state: (tofY[0], sumlty=0)
    init_state = (tofy_0, 0.0)
    
    # Run scan from j=1 to y_int
    # Create sequence of j values from 1 to 100
    j_seq = jnp.arange(1, 101)
    (_, sumlty), _ = jax.lax.scan(scan_fun, init_state, j_seq)
    
    # For y=0, sumlty should be 0
    # For y>0, we need to extract the sum at position y_int
    # Since scan runs for all 100 iterations, we need to recompute with proper masking
    # Alternative: use conditional to only accumulate up to y_int
    
    def scan_fun_masked(carry, j):
        tofy_prev, sum_log = carry
        j_float = j.astype(jnp.float64)
        
        # Clip tofy_prev to safe range to prevent extreme values
        tofy_prev_safe = jnp.clip(tofy_prev, eps, 1e100)
        
        # dum = 1 + 1/(mu*sigma*(1-nu))
        dum = 1.0 + 1.0 / (mu * sigma * (1.0 - nu))
        
        # tofY[j] = (j + mu*nu + 1/(sigma*(1-nu)) - (mu*nu*j)/tofY[j-1]) / dum
        tofy_j = (j_float + mu * nu + 1.0 / (sigma * (1.0 - nu)) - 
                  (mu * nu * j_float) / tofy_prev_safe) / dum
        
        # Clip tofy_j to safe range
        tofy_j = jnp.clip(tofy_j, eps, 1e100)
        
        # Only add to sum if j <= y_int
        log_term = jnp.where(j <= y_int, jnp.log(tofy_prev_safe), 0.0)
        sum_log = sum_log + log_term
        
        return (tofy_j, sum_log), None
    
    # Run scan with masking
    (_, sumlty), _ = jax.lax.scan(scan_fun_masked, init_state, j_seq)
    
    # For y=0, sumlty should be 0
    sumlty = jnp.where(y < 0.5, 0.0, sumlty)
    
    # Compute log-pdf
    # logfy = logpy0 - lgamma(y+1) + sumlty
    log_pdf = logpy0 - gammaln(y + 1.0) + sumlty
    
    # For very small sigma, use Poisson approximation
    log_pdf = jnp.where(
        sigma <= 0.0001,
        y * jnp.log(mu) - mu - gammaln(y + 1.0),  # Poisson log-pdf
        log_pdf
    )
    
    return log_pdf


def DEL() -> DelaportFamily:
    return build_ad_family(
        family_class=DelaportFamily,
        name="DEL",
        parameters=("mu", "sigma", "nu"),
        log_pdf_func=_del_log_pdf,
        type_="Discrete",
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


def YULE() -> YuleFamily:
    return build_ad_family(
        family_class=YuleFamily,
        name="YULE",
        parameters=("mu",),
        log_pdf_func=_yule_log_pdf,
        type_="Discrete",
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
