"""Additional distributions - Batch 16: PARETO1, PARETO1o, SEP, DPO1, LOGSHASHo, BCCGo."""

from __future__ import annotations

# Enable float64 precision for numerical accuracy
import jax
jax.config.update("jax_enable_x64", True)



from dataclasses import dataclass
from typing import Tuple

import jax.numpy as jnp
from jax import Array
from jax.scipy.special import ndtr

from .ad import build_ad_family
from .families import FamilyDefinition

eps = jnp.finfo(jnp.float64).eps


# =============================================================================
# PARETO1 - Pareto Type 1 (1-parameter)
# =============================================================================


@dataclass(frozen=True)
class Pareto1Family(FamilyDefinition):
    """Pareto Type 1 distribution (`PARETO1`).
    
    One-parameter Pareto distribution with support y > 0.
    
    Parameters:
        mu: shape parameter (mu > 0)
    
    PDF: f(y|mu) = mu * (1+y)^(-mu-1)
    """


def _pareto1_log_pdf(y: Array, mu: Array) -> Array:
    """Log-PDF for PARETO1 distribution.
    
    log f(y|mu) = log(mu) - (mu+1)*log(1+y)
    """
    mu = jnp.maximum(mu, eps)
    y = jnp.maximum(y, 0.0)
    
    log_pdf = jnp.log(mu) - (mu + 1.0) * jnp.log(1.0 + y)
    
    # Zero density for y <= 0
    log_pdf = jnp.where(y > 0.0, log_pdf, -jnp.inf)
    
    return log_pdf


def PARETO1() -> Pareto1Family:
    """Pareto Type 1 distribution (1-parameter).
    
    Returns:
        Pareto1Family with automatic differentiation support
    """
    return build_ad_family(
        family_class=Pareto1Family,
        log_pdf_func=_pareto1_log_pdf,
        name="PARETO1",
        parameters=("mu",),
    )


# =============================================================================
# PARETO1o - Pareto Type 1 with fixed mu (2-parameter)
# =============================================================================


@dataclass(frozen=True)
class Pareto1oFamily(FamilyDefinition):
    """Pareto Type 1 with fixed mu (`PARETO1o`).
    
    Two-parameter Pareto distribution where mu is fixed (location parameter)
    and sigma is the shape parameter.
    
    Parameters:
        mu: location parameter (fixed, mu > 0)
        sigma: shape parameter (sigma > 0)
    
    PDF: f(y|mu,sigma) = sigma * mu^sigma * y^(-sigma-1)  for y > mu
    """
    
    fixed_parameters: Tuple[str, ...] = ("mu",)


def _pareto1o_log_pdf(y: Array, mu: Array, sigma: Array) -> Array:
    """Log-PDF for PARETO1o distribution.
    
    log f(y|mu,sigma) = log(sigma) + sigma*log(mu) - (sigma+1)*log(y)
    """
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    y = jnp.maximum(y, eps)
    
    log_pdf = jnp.log(sigma) + sigma * jnp.log(mu) - (sigma + 1.0) * jnp.log(y)
    
    # Zero density for y <= mu
    log_pdf = jnp.where(y > mu, log_pdf, -jnp.inf)
    
    return log_pdf


def PARETO1o() -> Pareto1oFamily:
    """Pareto Type 1 with fixed mu.
    
    Returns:
        Pareto1oFamily with automatic differentiation support
    """
    return build_ad_family(
        family_class=Pareto1oFamily,
        log_pdf_func=_pareto1o_log_pdf,
        name="PARETO1o",
        parameters=("mu", "sigma"),
    )


# =============================================================================
# SEP - Skew Exponential Power (base)
# =============================================================================


@dataclass(frozen=True)
class SEPFamily(FamilyDefinition):
    """Skew Exponential Power distribution (`SEP`).
    
    Base SEP distribution. SEP1-SEP4 are variants with different
    parameterizations.
    
    Parameters:
        mu: location parameter
        sigma: scale parameter (sigma > 0)
        nu: skewness parameter (nu > 0)
        tau: kurtosis parameter (tau > 0)
    
    This is a flexible distribution that can model skewness and kurtosis.
    """


def _sep_log_pdf(y: Array, mu: Array, sigma: Array, nu: Array, tau: Array) -> Array:
    """Log-PDF for SEP distribution.
    
    This is a simplified implementation. The full SEP distribution
    involves complex normalizing constants and power exponential terms.
    
    For now, we use a simplified form similar to SEP1.
    """
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.maximum(nu, eps)
    tau = jnp.maximum(tau, eps)
    
    z = (y - mu) / sigma
    
    # Simplified SEP: similar to power exponential with skewness
    # Full implementation would require numerical integration for normalizing constant
    
    # Use asymmetric power exponential form
    z_adj = jnp.where(z >= 0, z / nu, z * nu)
    
    log_pdf = -jnp.log(sigma) - 0.5 * jnp.log(2.0 * jnp.pi) - jnp.power(jnp.abs(z_adj), tau)
    
    return log_pdf


def SEP() -> SEPFamily:
    """Skew Exponential Power distribution (base).
    
    Note: This is a simplified implementation. SEP1-SEP4 variants
    are already implemented in distributions_b15.py with more
    specific parameterizations.
    
    Returns:
        SEPFamily with automatic differentiation support
    """
    return build_ad_family(
        family_class=SEPFamily,
        log_pdf_func=_sep_log_pdf,
        name="SEP",
        parameters=("mu", "sigma", "nu", "tau"),
    )



# =============================================================================
# DPO1 - Double Poisson Type 1 (alternative parameterization)
# =============================================================================


@dataclass(frozen=True)
class DPO1Family(FamilyDefinition):
    """Double Poisson Type 1 distribution (`DPO1`).
    
    Alternative parameterization of Double Poisson with approximate derivatives.
    
    Parameters:
        mu: mean parameter (mu > 0)
        sigma: dispersion parameter (sigma > 0)
    
    This is similar to DPO but uses approximate derivatives for the
    normalizing constant.
    """


def _dpo1_log_pdf(y: Array, mu: Array, sigma: Array) -> Array:
    """Log-PDF for DPO1 distribution.
    
    Uses approximate normalizing constant similar to DPO.
    """
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    y = jnp.maximum(y, 0.0)
    
    # Double Poisson log-likelihood
    # log f(y|mu,sigma) = (y*log(mu) - mu - log(y!))/sigma - 0.5*log(2*pi*sigma*y) + log(C)
    
    # Approximate normalizing constant
    log_c = jnp.where(
        y > 0.5,
        0.5 * jnp.log(2.0 * jnp.pi * sigma * y),
        0.0
    )
    
    # Poisson part
    log_poisson = y * jnp.log(mu) - mu - jnp.where(y > 0, jnp.lgamma(y + 1.0), 0.0)
    
    log_pdf = log_poisson / sigma - 0.5 * jnp.log(2.0 * jnp.pi * sigma * y) + log_c
    
    # Zero density for y < 0
    log_pdf = jnp.where(y >= 0.0, log_pdf, -jnp.inf)
    
    return log_pdf


def DPO1() -> DPO1Family:
    """Double Poisson Type 1 distribution.
    
    Alternative parameterization with approximate derivatives.
    
    Returns:
        DPO1Family with automatic differentiation support
    """
    return build_ad_family(
        family_class=DPO1Family,
        log_pdf_func=_dpo1_log_pdf,
        name="DPO1",
        parameters=("mu", "sigma"),
    )


# =============================================================================
# LOGSHASHo - Log Sinh-Arcsinh Original
# =============================================================================


@dataclass(frozen=True)
class LOGSHASHoFamily(FamilyDefinition):
    """Log Sinh-Arcsinh Original distribution (`LOGSHASHo`).
    
    Original parameterization of log SHASH distribution.
    
    Parameters:
        mu: location parameter
        sigma: scale parameter (sigma > 0)
        nu: skewness parameter (nu > 0)
        tau: kurtosis parameter (tau > 0)
    
    For x > 0:
    z = (x - mu) / sigma
    r = sinh(tau * asinh(z) - nu)
    c = cosh(tau * asinh(z) - nu)
    """


def _logshasho_log_pdf(y: Array, mu: Array, sigma: Array, nu: Array, tau: Array) -> Array:
    """Log-PDF for LOGSHASHo distribution.
    
    log f(y|mu,sigma,nu,tau) = -log(sigma) + log(tau) - 0.5*log(2*pi)
                                - 0.5*log(1+z^2) + log(c) - 0.5*r^2
    where:
        z = (y - mu) / sigma
        r = sinh(tau * asinh(z) - nu)
        c = cosh(tau * asinh(z) - nu)
    """
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.maximum(nu, eps)
    tau = jnp.maximum(tau, eps)
    
    z = (y - mu) / sigma
    
    # Compute sinh-arcsinh transformation
    asinh_z = jnp.arcsinh(z)
    arg = tau * asinh_z - nu
    
    r = jnp.sinh(arg)
    c = jnp.cosh(arg)
    
    log_pdf = (
        -jnp.log(sigma)
        + jnp.log(tau)
        - 0.5 * jnp.log(2.0 * jnp.pi)
        - 0.5 * jnp.log(1.0 + z * z)
        + jnp.log(c)
        - 0.5 * r * r
    )
    
    # Zero density for y <= 0
    log_pdf = jnp.where(y > 0.0, log_pdf, -jnp.inf)
    
    return log_pdf


def LOGSHASHo() -> LOGSHASHoFamily:
    """Log Sinh-Arcsinh Original distribution.
    
    Returns:
        LOGSHASHoFamily with automatic differentiation support
    """
    return build_ad_family(
        family_class=LOGSHASHoFamily,
        log_pdf_func=_logshasho_log_pdf,
        name="LOGSHASHo",
        parameters=("mu", "sigma", "nu", "tau"),
    )


# =============================================================================
# BCCGo - Box-Cox Cole-Green Original
# =============================================================================


@dataclass(frozen=True)
class BCCGoFamily(FamilyDefinition):
    """Box-Cox Cole-Green Original distribution (`BCCGo`).
    
    Original parameterization of BCCG distribution.
    
    Parameters:
        mu: location parameter (mu > 0)
        sigma: scale parameter (sigma > 0)
        nu: shape parameter
    
    For x > 0:
    z = ((x/mu)^nu - 1) / (nu * sigma)  if nu != 0
    z = log(x/mu) / sigma               if nu = 0
    
    PDF includes truncation adjustment: pnorm(1/(sigma*|nu|))
    """


def _bccgo_log_pdf(y: Array, mu: Array, sigma: Array, nu: Array) -> Array:
    """Log-PDF for BCCGo distribution.
    
    log f(y|mu,sigma,nu) = nu*log(y/mu) - log(sigma) - z^2/2 - log(y)
                           - 0.5*log(2*pi) - log(Phi(1/(sigma*|nu|)))
    where:
        z = ((y/mu)^nu - 1) / (nu * sigma)  if nu != 0
        z = log(y/mu) / sigma               if nu = 0
    """
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    y = jnp.maximum(y, eps)
    
    # Compute z transformation
    ratio = y / mu
    
    z = jnp.where(
        jnp.abs(nu) > eps,
        (jnp.power(ratio, nu) - 1.0) / (nu * sigma),
        jnp.log(ratio) / sigma
    )
    
    # Truncation adjustment
    trunc_arg = 1.0 / (sigma * jnp.maximum(jnp.abs(nu), eps))
    log_trunc = jnp.log(ndtr(trunc_arg))  # log(Phi(trunc_arg))
    
    log_pdf = (
        nu * jnp.log(ratio)
        - jnp.log(sigma)
        - 0.5 * z * z
        - jnp.log(y)
        - 0.5 * jnp.log(2.0 * jnp.pi)
        - log_trunc
    )
    
    # Zero density for y <= 0
    log_pdf = jnp.where(y > 0.0, log_pdf, -jnp.inf)
    
    return log_pdf


def BCCGo() -> BCCGoFamily:
    """Box-Cox Cole-Green Original distribution.
    
    Returns:
        BCCGoFamily with automatic differentiation support
    """
    return build_ad_family(
        family_class=BCCGoFamily,
        log_pdf_func=_bccgo_log_pdf,
        name="BCCGo",
        parameters=("mu", "sigma", "nu"),
    )
