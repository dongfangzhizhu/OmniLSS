"""GAMLSS distributions: Batch 8 — Additional continuous distributions.

These are commonly used continuous distributions with various shapes and properties.

Distributions:
- GG       (Generalized Gamma)
- GB2      (Generalized Beta Type 2)
- PARETO   (Pareto distribution)
- NET      (Normal-Exponential-t)
- LNO      (Log Normal with location parameter)

R source: gamlss.dist package
"""

from __future__ import annotations

# Enable float64 precision for numerical accuracy

from dataclasses import dataclass
import math

import jax.numpy as jnp
from jax.scipy.special import gammaln, betaln
from jax.scipy.stats import norm

from .ad import build_ad_family
from .families import FamilyDefinition
from .links import (
    identity_derivative,
    identity_inverse,
    identity_link,
    log_derivative,
    log_inverse,
    log_link,
)

# ------------------------------------------------------------------
# 1. GG — Generalized Gamma
#    R: dGG(x, mu, sigma, nu)
#    A flexible three-parameter distribution that includes Gamma, Weibull, and others as special cases
# ------------------------------------------------------------------

@dataclass(frozen=True)
class GeneralizedGammaFamily(FamilyDefinition):
    """Generalized Gamma distribution (`GG`)."""


def _gg_log_pdf(y, mu, sigma, nu):
    """Generalized Gamma log-pdf.
    
    The Generalized Gamma distribution is a flexible three-parameter family
    that includes Gamma, Weibull, and Exponential as special cases.
    
    Parameters:
    - y: observed value (> 0)
    - mu: location parameter (> 0)
    - sigma: scale parameter (> 0)
    - nu: shape parameter (can be positive or negative)
    
    Parameterization (from R gamlss.dist):
    z = (y/mu)^nu
    For |nu| > 1e-6:
        log p(y) = dGA(z, mu=1, sigma=sigma*|nu|, log=TRUE) + log(|nu|*z/y)
    For |nu| <= 1e-6 (nu ~ 0, log-normal limit):
        log p(y) = -log(y) - 0.5*log(2π) - log(sigma) - 0.5*(log(y)-log(mu))^2/sigma^2
    """
    eps = jnp.finfo(jnp.float64).eps
    y = jnp.maximum(y, eps)
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    
    # Compute z = (y/mu)^nu
    z = jnp.power(y / mu, nu)
    
    # For |nu| > 1e-6, use Gamma distribution on z
    # theta = 1/(sigma^2 * nu^2)
    theta = 1.0 / (sigma * sigma * nu * nu)
    
    # Gamma log-pdf: log p(z) = (theta-1)*log(z) - z*theta - log(Gamma(theta)) - theta*log(theta)
    # But we need dGA(z, mu=1, sigma=sigma*|nu|)
    # For GA with mu=1: shape = 1/sigma^2, scale = sigma^2
    # So for sigma' = sigma*|nu|: shape = 1/(sigma*|nu|)^2 = theta
    sigma_prime = sigma * jnp.abs(nu)
    shape = 1.0 / (sigma_prime * sigma_prime)  # This equals theta
    
    # Gamma(shape, scale=1/shape) log-pdf
    log_pdf_z = (shape - 1.0) * jnp.log(z) - z * shape - gammaln(shape) + shape * jnp.log(shape)
    
    # Jacobian: |nu| * z / y
    log_jacobian = jnp.log(jnp.abs(nu)) + jnp.log(z) - jnp.log(y)
    
    # Total log-pdf
    log_pdf = log_pdf_z + log_jacobian
    
    # For nu ~ 0, use log-normal limit
    log_pdf_lognormal = (
        -jnp.log(y) - 0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma) -
        0.5 * jnp.square((jnp.log(y) - jnp.log(mu)) / sigma)
    )
    
    # Use log-normal when |nu| < 1e-6
    log_pdf = jnp.where(jnp.abs(nu) > 1e-6, log_pdf, log_pdf_lognormal)
    
    return log_pdf


def GG() -> GeneralizedGammaFamily:
    from .dpqr_functions import pGG, qGG, rGG
    return build_ad_family(
        family_class=GeneralizedGammaFamily,
        name="GG",
        parameters=("mu", "sigma", "nu"),
        log_pdf_func=_gg_log_pdf,
        type_="Continuous",
        links={"mu": "log", "sigma": "log", "nu": "log"},
        link_functions={"mu": log_link, "sigma": log_link, "nu": log_link},
        link_inverses={"mu": log_inverse, "sigma": log_inverse, "nu": log_inverse},
        link_derivatives={"mu": log_derivative, "sigma": log_derivative, "nu": log_derivative},
        p=pGG,
        q=qGG,
        r=rGG,
    )


# ------------------------------------------------------------------
# 2. GB2 — Generalized Beta Type 2
#    R: dGB2(x, mu, sigma, nu, tau)
#    A four-parameter distribution that includes many distributions as special cases
# ------------------------------------------------------------------

@dataclass(frozen=True)
class GeneralizedBeta2Family(FamilyDefinition):
    """Generalized Beta Type 2 distribution (`GB2`)."""


def _gb2_log_pdf(y, mu, sigma, nu, tau):
    """Generalized Beta Type 2 log-pdf.
    
    The GB2 distribution is a very flexible four-parameter family.
    
    R formula (from gamlss.dist):
    f(y|μ,σ,ν,τ) = |σ| y^(σν-1) / {μ^(σν) B(ν,τ) [1+(y/μ)^σ]^(ν+τ)}
    
    Parameters:
    - y: observed value (> 0)
    - mu: location parameter (> 0)
    - sigma: scale parameter (> 0)
    - nu: shape parameter 1 (> 0)
    - tau: shape parameter 2 (> 0)
    
    Numerical stabilization:
    - Clip sigma to [0.1, 10.0] to prevent overflow in power operations
    - Clip tau to [eps, 100.0] to prevent parameter explosion
    - Use log-space computation: sigma * log(y/mu) instead of log((y/mu)^sigma)
    """
    eps = jnp.finfo(jnp.float64).eps
    y = jnp.maximum(y, eps)
    mu = jnp.maximum(mu, eps)
    # Clip sigma to reasonable range to prevent overflow
    sigma = jnp.clip(sigma, 0.1, 10.0)
    nu = jnp.maximum(nu, eps)
    # Clip tau to prevent explosion (tau < 100)
    tau = jnp.clip(tau, eps, 100.0)
    
    # Use log-space computation to avoid overflow in power operations
    # Instead of: (y/mu)^sigma, compute: exp(sigma * log(y/mu))
    # This is more stable for large sigma values
    log_ratio = jnp.log(y / mu)
    
    # log f(y) = log(|σ|) + (σν-1)*log(y) - σν*log(μ) - log(B(ν,τ)) - (ν+τ)*log(1+(y/μ)^σ)
    # Rewrite using log-space:
    # log(1 + (y/μ)^σ) = log(1 + exp(σ * log(y/μ)))
    # Use log1p for numerical stability: log1p(x) = log(1 + x)
    log_pdf = (
        jnp.log(jnp.abs(sigma)) +
        (sigma * nu - 1.0) * jnp.log(y) -
        sigma * nu * jnp.log(mu) -
        betaln(nu, tau) -
        (nu + tau) * jnp.log1p(jnp.exp(sigma * log_ratio))
    )
    
    return log_pdf


def GB2() -> GeneralizedBeta2Family:
    """Generalized Beta Type 2 distribution.
    
    All parameters use log link to ensure positivity.
    """
    return build_ad_family(
        family_class=GeneralizedBeta2Family,
        name="GB2",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_gb2_log_pdf,
        type_="Continuous",
        links={"mu": "log", "sigma": "log", "nu": "log", "tau": "log"},
        link_functions={"mu": log_link, "sigma": log_link, "nu": log_link, "tau": log_link},
        link_inverses={"mu": log_inverse, "sigma": log_inverse, "nu": log_inverse, "tau": log_inverse},
        link_derivatives={"mu": log_derivative, "sigma": log_derivative, "nu": log_derivative, "tau": log_derivative},
    )


# ------------------------------------------------------------------
# 3. PARETO — Pareto distribution
#    R: dPARETO(x, mu)
#    A simple one-parameter power-law distribution
# ------------------------------------------------------------------

@dataclass(frozen=True)
class ParetoFamily(FamilyDefinition):
    """Pareto distribution (`PARETO`)."""


def _pareto_log_pdf(y, mu):
    """Pareto distribution log-pdf.
    
    The Pareto distribution is a power-law distribution often used for
    modeling wealth, city sizes, and other phenomena.
    
    Parameters:
    - y: observed value (>= 1)
    - mu: shape parameter (> 0)
    
    log p(y) = log(mu) - (mu + 1)*log(y)
    """
    eps = jnp.finfo(jnp.float64).eps
    y = jnp.asarray(y, dtype=jnp.float64)
    mu = jnp.maximum(mu, eps)
    
    log_pdf = jnp.log(mu) - (mu + 1.0) * jnp.log(y)
    return jnp.where(y <= 1.0, -jnp.inf, log_pdf)


def PARETO() -> ParetoFamily:
    return build_ad_family(
        family_class=ParetoFamily,
        name="PARETO",
        parameters=("mu",),
        log_pdf_func=_pareto_log_pdf,
        type_="Continuous",
        links={"mu": "log"},
        link_functions={"mu": log_link},
        link_inverses={"mu": log_inverse},
        link_derivatives={"mu": log_derivative},
    )


# ------------------------------------------------------------------
# 4. NET — Normal-Exponential-t distribution
#    R: dNET(x, mu, sigma, nu, tau)
#    A flexible distribution with piecewise density (Normal-Exponential-t)
#    Note: In R gamlss, nu and tau are typically fixed (not estimated)
# ------------------------------------------------------------------

@dataclass(frozen=True)
class NormalExponentialTFamily(FamilyDefinition):
    """Normal-Exponential-t distribution (`NET`)."""


def _net_log_pdf(y, mu, sigma, nu, tau):
    """Normal-Exponential-t log-pdf.
    
    The NET distribution uses a piecewise density function:
    - Normal in the center (|z| <= k1)
    - Exponential in the middle (k1 < |z| <= k2)
    - Power tail beyond (|z| > k2)
    
    Parameters:
    - y: observed value
    - mu: location parameter
    - sigma: scale parameter (> 0)
    - nu: threshold parameter k1 (> 0)
    - tau: threshold parameter k2 (> 0), must be >= nu
    
    From R gamlss.dist source:
    k1 = nu, k2 = tau
    c1 = (1 - 2*pnorm(-k1)) * sqrt(2*pi)
    c2 = (2/k1) * exp(-(k1^2)/2)
    c3 = 2*exp(-k1*k2 + (k1^2)/2) / ((k1*k2 - 1)*k1)
    ct = 1/(c1 + c2 + c3)  # Normalizing constant
    
    tc = (y - mu)/sigma
    d1 = (|tc| <= k1) * (-(tc^2)/2)
    d2 = (k1 < |tc| <= k2) * (-k1*|tc| + (k1^2)/2)
    d3 = (|tc| > k2) * (-k1*k2*log(|tc|/k2) - k1*k2 + (k1^2)/2)
    
    log p(y) = log(ct) - log(sigma) + d1 + d2 + d3
    """
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.maximum(nu, eps)
    tau = jnp.maximum(tau, nu)  # Ensure tau >= nu
    
    k1 = nu
    k2 = tau
    
    # Normalizing constant components
    from jax.scipy.stats import norm as jax_norm
    c1 = (1.0 - 2.0 * jax_norm.cdf(-k1)) * jnp.sqrt(2.0 * jnp.pi)
    c2 = (2.0 / k1) * jnp.exp(-(k1 * k1) / 2.0)
    c3 = 2.0 * jnp.exp(-k1 * k2 + (k1 * k1) / 2.0) / ((k1 * k2 - 1.0) * k1)
    ct = 1.0 / (c1 + c2 + c3)
    
    # Standardized residual
    tc = (y - mu) / sigma
    abs_tc = jnp.abs(tc)
    
    # Piecewise log-density (without normalizing constant and sigma term)
    # Region 1: |tc| <= k1 (Normal)
    d1 = jnp.where(abs_tc <= k1, -(tc * tc) / 2.0, 0.0)
    
    # Region 2: k1 < |tc| <= k2 (Exponential)
    d2 = jnp.where((abs_tc > k1) & (abs_tc <= k2), -k1 * abs_tc + (k1 * k1) / 2.0, 0.0)
    
    # Region 3: |tc| > k2 (Power tail)
    # Handle log(0) case when tc = 0
    log_term = jnp.where(
        abs_tc > k2,
        -k1 * k2 * jnp.log(abs_tc / k2) - k1 * k2 + (k1 * k1) / 2.0,
        0.0
    )
    d3 = log_term
    
    # Total log-pdf
    log_pdf = jnp.log(ct) - jnp.log(sigma) + d1 + d2 + d3
    
    return log_pdf


def NET() -> NormalExponentialTFamily:
    return build_ad_family(
        family_class=NormalExponentialTFamily,
        name="NET",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_net_log_pdf,
        type_="Continuous",
        links={"mu": "identity", "sigma": "log", "nu": "log", "tau": "log"},
        link_functions={"mu": identity_link, "sigma": log_link, "nu": log_link, "tau": log_link},
        link_inverses={"mu": identity_inverse, "sigma": log_inverse, "nu": log_inverse, "tau": log_inverse},
        link_derivatives={"mu": identity_derivative, "sigma": log_derivative, "nu": log_derivative, "tau": log_derivative},
    )


# ------------------------------------------------------------------
# 5. LNO — Log Normal with optional Box-Cox power nu
#    R: dLNO(x, mu, sigma, nu)
#    In fitting, nu is fixed at 0 by default; density helpers can still evaluate
#    arbitrary nu values for R consistency checks.
# ------------------------------------------------------------------

@dataclass(frozen=True)
class LogNormalLocationFamily(FamilyDefinition):
    """Log Normal distribution (`LNO`) - alias for LOGNO."""


def _lno_log_pdf(y, mu, sigma, nu=0.0):
    """Log Normal with optional Box-Cox power parameter."""
    eps = jnp.finfo(jnp.float64).eps
    y = jnp.maximum(y, eps)
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.asarray(nu, dtype=jnp.float64)

    z = jnp.where(jnp.abs(nu) > 1e-12, (jnp.power(y, nu) - 1.0) / nu, jnp.log(y))
    log_pdf = -0.5 * jnp.log(2.0 * math.pi) - jnp.log(sigma) - 0.5 * jnp.square((z - mu) / sigma) + (nu - 1.0) * jnp.log(y)
    return log_pdf


def LNO() -> LogNormalLocationFamily:
    return build_ad_family(
        family_class=LogNormalLocationFamily,
        name="LNO",
        parameters=("mu", "sigma"),
        log_pdf_func=_lno_log_pdf,
        type_="Continuous",
        links={"mu": "identity", "sigma": "log"},
        link_functions={"mu": identity_link, "sigma": log_link},
        link_inverses={"mu": identity_inverse, "sigma": log_inverse},
        link_derivatives={"mu": identity_derivative, "sigma": log_derivative},
    )
