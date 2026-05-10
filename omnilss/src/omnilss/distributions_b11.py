"""Batch 11: JSU, GIG, GB1 distributions."""

from __future__ import annotations

# Enable float64 precision for numerical accuracy
import jax
jax.config.update("jax_enable_x64", True)

from dataclasses import dataclass
import jax.numpy as jnp
from jax.scipy.special import gammaln, betaln

from .ad import build_ad_family
from .families import FamilyDefinition
from .links import (
    log_link, log_inverse, log_derivative,
    identity_link, identity_inverse, identity_derivative
)


# ============================================================================
# JSU - Johnson's SU Distribution
# ============================================================================

@dataclass(frozen=True)
class JSUFamily(FamilyDefinition):
    """Johnson's SU distribution family."""
    pass


def _jsu_log_pdf(y, mu, sigma, nu, tau):
    """Johnson's SU log-PDF.
    
    Parameters:
    - mu: location parameter
    - sigma: scale parameter (> 0)
    - nu: skewness parameter
    - tau: kurtosis parameter (> 0)
    
    Transformation: z = (y - mu) / sigma
    r = sinh^{-1}(z)
    w = -nu + tau * r
    
    PDF: f(y) = (tau / (sigma * sqrt(2π))) * (1 / sqrt(1 + z^2)) * exp(-w^2 / 2)
    """
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    tau = jnp.maximum(tau, eps)
    
    # Standardize
    z = (y - mu) / sigma
    
    # Sinh-arcsinh transformation
    r = jnp.arcsinh(z)
    w = -nu + tau * r
    
    # Log-PDF
    log_pdf = (
        jnp.log(tau) 
        - jnp.log(sigma) 
        - 0.5 * jnp.log(2.0 * jnp.pi)
        - 0.5 * jnp.log1p(z * z)
        - 0.5 * w * w
    )
    
    return log_pdf


def JSU():
    """Johnson's SU distribution.
    
    A flexible distribution using sinh-arcsinh transformation.
    Can model skewness and kurtosis independently.
    
    Parameters:
    - mu: location
    - sigma: scale (> 0)
    - nu: skewness
    - tau: kurtosis (> 0)
    
    Link functions:
    - mu: identity
    - sigma: log
    - nu: identity
    - tau: log
    """
    return build_ad_family(
        family_class=JSUFamily,
        name="JSU",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_jsu_log_pdf,
        type_="Continuous",
        links={"mu": "identity", "sigma": "log", "nu": "identity", "tau": "log"},
        link_functions={
            "mu": identity_link,
            "sigma": log_link,
            "nu": identity_link,
            "tau": log_link,
        },
        link_inverses={
            "mu": identity_inverse,
            "sigma": log_inverse,
            "nu": identity_inverse,
            "tau": log_inverse,
        },
        link_derivatives={
            "mu": identity_derivative,
            "sigma": log_derivative,
            "nu": identity_derivative,
            "tau": log_derivative,
        },
    )


# ============================================================================
# GIG - Generalized Inverse Gaussian Distribution
# ============================================================================

@dataclass(frozen=True)
class GIGFamily(FamilyDefinition):
    """Generalized Inverse Gaussian distribution family."""
    pass


def _gig_log_pdf(y, mu, sigma, nu):
    """Generalized Inverse Gaussian log-PDF.
    
    Parameters:
    - mu: location parameter (> 0)
    - sigma: scale parameter (> 0)
    - nu: shape parameter (can be negative)
    
    The GIG distribution is defined as:
    f(y; λ, χ, ψ) ∝ y^(λ-1) * exp(-(χ/y + ψ*y)/2)
    
    Parameterization:
    - λ = nu
    - χ = 1/sigma^2
    - ψ = 1/(mu^2 * sigma^2)
    
    This ensures E[Y] = mu.
    """
    from .bessel import log_bessel_kv
    
    eps = jnp.finfo(jnp.float64).eps
    y = jnp.maximum(y, eps)
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    
    # Parameterization
    lambda_ = nu
    chi = 1.0 / (sigma * sigma)
    psi = 1.0 / (mu * mu * sigma * sigma)
    
    # Normalizing constant: K_λ(sqrt(χψ))
    sqrt_chi_psi = jnp.sqrt(chi * psi)
    log_bessel_k = log_bessel_kv(lambda_, sqrt_chi_psi)
    
    # Log-PDF
    log_pdf = (
        (lambda_ - 1.0) * jnp.log(y)
        - 0.5 * (chi / y + psi * y)
        - log_bessel_k
        - 0.5 * lambda_ * jnp.log(chi / psi)
    )
    
    return log_pdf


def GIG():
    """Generalized Inverse Gaussian distribution.
    
    A three-parameter distribution that includes inverse Gaussian
    and gamma as special cases.
    
    Parameters:
    - mu: location (> 0)
    - sigma: scale (> 0)
    - nu: shape (can be negative)
    
    Link functions:
    - mu: log
    - sigma: log
    - nu: identity
    """
    return build_ad_family(
        family_class=GIGFamily,
        name="GIG",
        parameters=("mu", "sigma", "nu"),
        log_pdf_func=_gig_log_pdf,
        type_="Continuous",
        links={"mu": "log", "sigma": "log", "nu": "identity"},
        link_functions={
            "mu": log_link,
            "sigma": log_link,
            "nu": identity_link,
        },
        link_inverses={
            "mu": log_inverse,
            "sigma": log_inverse,
            "nu": identity_inverse,
        },
        link_derivatives={
            "mu": log_derivative,
            "sigma": log_derivative,
            "nu": identity_derivative,
        },
    )


# ============================================================================
# GB1 - Generalized Beta Type 1 Distribution
# ============================================================================

@dataclass(frozen=True)
class GB1Family(FamilyDefinition):
    """Generalized Beta Type 1 distribution family."""
    pass


def _gb1_log_pdf(y, mu, sigma, nu, tau):
    """Generalized Beta Type 1 log-PDF.
    
    Parameters:
    - mu: location parameter (0 < mu < 1)
    - sigma: scale parameter (> 0)
    - nu: shape parameter 1 (> 0)
    - tau: shape parameter 2 (> 0)
    
    The GB1 is a four-parameter extension of the beta distribution.
    
    PDF: f(y) ∝ y^(a-1) * (1-y)^(b-1) * [1 + ((y/(1-y))^c - 1) / d]^(-a-b)
    
    Simplified parameterization:
    - a = 1/sigma^2
    - b = (1-mu)/(mu * sigma^2)
    - c = nu
    - d = tau
    """
    eps = jnp.finfo(jnp.float64).eps
    y = jnp.clip(y, eps, 1.0 - eps)
    mu = jnp.clip(mu, eps, 1.0 - eps)
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.maximum(nu, eps)
    tau = jnp.maximum(tau, eps)
    
    # Parameters
    a = 1.0 / (sigma * sigma)
    b = (1.0 - mu) / (mu * sigma * sigma)
    c = nu
    d = tau
    
    # Log-PDF components
    log_y_term = (a - 1.0) * jnp.log(y)
    log_1my_term = (b - 1.0) * jnp.log1p(-y)
    
    # Ratio term: y / (1-y)
    ratio = y / (1.0 - y)
    ratio_c = jnp.power(ratio, c)
    
    # Main term: [1 + (ratio^c - 1) / d]^(-a-b)
    inner = 1.0 + (ratio_c - 1.0) / d
    log_main_term = -(a + b) * jnp.log(inner)
    
    # Normalizing constant (Beta function)
    log_beta_ab = betaln(a, b)
    
    log_pdf = log_y_term + log_1my_term + log_main_term - log_beta_ab
    
    return log_pdf


def GB1():
    """Generalized Beta Type 1 distribution.
    
    A four-parameter extension of the beta distribution for
    modeling bounded data with flexible shapes.
    
    Parameters:
    - mu: location (0 < mu < 1)
    - sigma: scale (> 0)
    - nu: shape 1 (> 0)
    - tau: shape 2 (> 0)
    
    Link functions:
    - mu: logit
    - sigma: log
    - nu: log
    - tau: log
    """
    from .links import logit_link, logit_inverse, logit_derivative
    
    return build_ad_family(
        family_class=GB1Family,
        name="GB1",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_gb1_log_pdf,
        type_="Continuous",
        links={"mu": "logit", "sigma": "log", "nu": "log", "tau": "log"},
        link_functions={
            "mu": logit_link,
            "sigma": log_link,
            "nu": log_link,
            "tau": log_link,
        },
        link_inverses={
            "mu": logit_inverse,
            "sigma": log_inverse,
            "nu": log_inverse,
            "tau": log_inverse,
        },
        link_derivatives={
            "mu": logit_derivative,
            "sigma": log_derivative,
            "nu": log_derivative,
            "tau": log_derivative,
        },
    )
