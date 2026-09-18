"""Batch 11: GIG and GB1 distributions."""

from __future__ import annotations

# Enable float64 precision for numerical accuracy
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
    """Shared differentiable density in R's bounded GB1 parameterization."""
    from .gb1 import log_density
    return log_density(y, mu, sigma, nu, tau)


def GB1():
    """Generalized Beta Type 1 distribution.
    
    A four-parameter extension of the beta distribution for
    modeling bounded data with flexible shapes.
    
    Parameters:
    - mu: location (0 < mu < 1)
    - sigma: scale (0 < sigma < 1)
    - nu: shape 1 (> 0)
    - tau: shape 2 (> 0)
    
    Link functions:
    - mu: logit
    - sigma: logit
    - nu: log
    - tau: log
    """
    from .links import logit_link, logit_inverse, logit_derivative
    from .gb1 import dGB1, pGB1, qGB1, rGB1
    
    return build_ad_family(
        family_class=GB1Family,
        name="GB1",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_gb1_log_pdf,
        type_="Continuous",
        d=dGB1, p=pGB1, q=qGB1, r=rGB1,
        links={"mu": "logit", "sigma": "logit", "nu": "log", "tau": "log"},
        link_functions={
            "mu": logit_link,
            "sigma": logit_link,
            "nu": log_link,
            "tau": log_link,
        },
        link_inverses={
            "mu": logit_inverse,
            "sigma": logit_inverse,
            "nu": log_inverse,
            "tau": log_inverse,
        },
        link_derivatives={
            "mu": logit_derivative,
            "sigma": logit_derivative,
            "nu": log_derivative,
            "tau": log_derivative,
        },
    )
