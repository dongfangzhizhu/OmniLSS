"""Batch 13: More type variants and family extensions."""

from __future__ import annotations

# Enable float64 precision for numerical accuracy
from dataclasses import dataclass
import jax.numpy as jnp
from jax.scipy.special import gammaln

from .ad import build_ad_family
from .families import FamilyDefinition
from .links import (
    log_link, log_inverse, log_derivative,
    identity_link, identity_inverse, identity_derivative
)


# ============================================================================
# IGA - Inverse Gaussian Alternative
# ============================================================================

@dataclass(frozen=True)
class IGAFamily(FamilyDefinition):
    """Inverse Gaussian Alternative parameterization family."""
    pass


def _iga_log_pdf(y, mu, sigma):
    """Inverse Gaussian Alternative log-PDF."""
    eps = jnp.finfo(jnp.float64).eps
    y = jnp.maximum(y, eps)
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    
    # Alternative: lambda = 1/sigma^2
    lambda_ = 1.0 / (sigma * sigma)
    
    return (
        0.5 * jnp.log(lambda_) - 0.5 * jnp.log(2.0 * jnp.pi * y * y * y)
        - lambda_ * jnp.square(y - mu) / (2.0 * mu * mu * y)
    )


def IGA():
    """Inverse Gaussian Alternative distribution."""
    return build_ad_family(
        family_class=IGAFamily,
        name="IGA",
        parameters=("mu", "sigma"),
        log_pdf_func=_iga_log_pdf,
        type_="Continuous",
        links={"mu": "log", "sigma": "log"},
        link_functions={"mu": log_link, "sigma": log_link},
        link_inverses={"mu": log_inverse, "sigma": log_inverse},
        link_derivatives={"mu": log_derivative, "sigma": log_derivative},
    )


# ============================================================================
# PIG2 - Poisson Inverse Gaussian Type 2
# ============================================================================

@dataclass(frozen=True)
class PIG2Family(FamilyDefinition):
    """Poisson Inverse Gaussian Type 2 family."""
    pass


def _pig2_log_pdf(y, mu, sigma):
    """PIG Type 2 log-PDF with alternative parameterization."""
    from .bessel import log_bessel_kv
    
    eps = jnp.finfo(jnp.float64).eps
    y = jnp.maximum(y, 0.0)
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    
    # Type 2: different relationship between parameters
    lambda_ = 1.0 / sigma
    theta = mu * sigma
    
    sqrt_term = jnp.sqrt(lambda_ * (lambda_ + 2.0 * theta * y))
    log_bessel = log_bessel_kv(y - 0.5, sqrt_term)
    
    return (
        0.5 * jnp.log(lambda_) - 0.5 * jnp.log(2.0 * jnp.pi)
        + y * jnp.log(theta) - gammaln(y + 1.0)
        - lambda_ - theta * y
        + log_bessel
        + (y - 0.5) * (jnp.log(lambda_) - jnp.log(lambda_ + 2.0 * theta * y)) / 2.0
    )


def PIG2():
    """Poisson Inverse Gaussian Type 2 distribution."""
    return build_ad_family(
        family_class=PIG2Family,
        name="PIG2",
        parameters=("mu", "sigma"),
        log_pdf_func=_pig2_log_pdf,
        type_="Discrete",
        links={"mu": "log", "sigma": "log"},
        link_functions={"mu": log_link, "sigma": log_link},
        link_inverses={"mu": log_inverse, "sigma": log_inverse},
        link_derivatives={"mu": log_derivative, "sigma": log_derivative},
    )


# ============================================================================
# PE2 - Power Exponential Type 2
# ============================================================================

@dataclass(frozen=True)
class PE2Family(FamilyDefinition):
    """Power Exponential Type 2 family."""
    pass


def _pe2_log_pdf(y, mu, sigma, nu):
    """Power Exponential Type 2 log-PDF."""
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.clip(nu, eps, 10.0)  # Avoid extreme values
    
    z = jnp.abs(y - mu) / sigma
    
    # Type 2: alternative normalization
    log_norm = (
        jnp.log(nu) - jnp.log(2.0 * sigma)
        - gammaln(1.0 / nu)
    )
    
    return log_norm - jnp.power(z, nu)


def PE2():
    """Power Exponential Type 2 distribution."""
    return build_ad_family(
        family_class=PE2Family,
        name="PE2",
        parameters=("mu", "sigma", "nu"),
        log_pdf_func=_pe2_log_pdf,
        type_="Continuous",
        links={"mu": "identity", "sigma": "log", "nu": "log"},
        link_functions={
            "mu": identity_link,
            "sigma": log_link,
            "nu": log_link,
        },
        link_inverses={
            "mu": identity_inverse,
            "sigma": log_inverse,
            "nu": log_inverse,
        },
        link_derivatives={
            "mu": identity_derivative,
            "sigma": log_derivative,
            "nu": log_derivative,
        },
    )


# ============================================================================
# LOGITNO - Logit Normal
# ============================================================================

@dataclass(frozen=True)
class LOGITNOFamily(FamilyDefinition):
    """Logit Normal distribution family."""
    pass


def _logitno_log_pdf(y, mu, sigma):
    """Logit Normal log-PDF.
    
    If Y ~ LogitNormal(mu, sigma), then logit(Y) ~ Normal(mu, sigma).
    """
    eps = jnp.finfo(jnp.float64).eps
    y = jnp.clip(y, eps, 1.0 - eps)
    sigma = jnp.maximum(sigma, eps)
    
    # logit(y) = log(y / (1-y))
    logit_y = jnp.log(y) - jnp.log1p(-y)
    
    # Jacobian: d(logit(y))/dy = 1 / (y * (1-y))
    log_jacobian = -jnp.log(y) - jnp.log1p(-y)
    
    # Normal log-PDF for logit(y)
    z = (logit_y - mu) / sigma
    log_normal = -0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma) - 0.5 * z * z
    
    return log_normal + log_jacobian


def LOGITNO():
    """Logit Normal distribution."""
    return build_ad_family(
        family_class=LOGITNOFamily,
        name="LOGITNO",
        parameters=("mu", "sigma"),
        log_pdf_func=_logitno_log_pdf,
        type_="Continuous",
        links={"mu": "identity", "sigma": "log"},
        link_functions={"mu": identity_link, "sigma": log_link},
        link_inverses={"mu": identity_inverse, "sigma": log_inverse},
        link_derivatives={"mu": identity_derivative, "sigma": log_derivative},
    )


# ============================================================================
# GeneralisedPoisson - Generalized Poisson
# ============================================================================

@dataclass(frozen=True)
class GeneralisedPoissonFamily(FamilyDefinition):
    """Generalized Poisson distribution family."""
    pass


def _gp_log_pdf(y, mu, sigma):
    """Generalized Poisson log-PDF.
    
    GP(mu, sigma) with E[Y] = mu.
    """
    eps = jnp.finfo(jnp.float64).eps
    y = jnp.maximum(y, 0.0)
    mu = jnp.maximum(mu, eps)
    sigma = jnp.clip(sigma, -0.99, 0.99)  # Constraint for GP
    
    # Parameterization: theta = mu, alpha = sigma
    theta = mu
    alpha = sigma
    
    lambda_ = theta / (1.0 + alpha)
    
    return (
        jnp.log(lambda_) + (y - 1.0) * jnp.log(lambda_ + alpha * y)
        - lambda_ - alpha * y
        - gammaln(y + 1.0)
    )


def GeneralisedPoisson():
    """Generalized Poisson distribution."""
    return build_ad_family(
        family_class=GeneralisedPoissonFamily,
        name="GeneralisedPoisson",
        parameters=("mu", "sigma"),
        log_pdf_func=_gp_log_pdf,
        type_="Discrete",
        links={"mu": "log", "sigma": "identity"},
        link_functions={"mu": log_link, "sigma": identity_link},
        link_inverses={"mu": log_inverse, "sigma": identity_inverse},
        link_derivatives={"mu": log_derivative, "sigma": identity_derivative},
    )


# ============================================================================
# DELAPORT - Delaport (alternative spelling)
# ============================================================================

@dataclass(frozen=True)
class DELAPORTFamily(FamilyDefinition):
    """Delaport distribution family."""
    pass


def _delaport_log_pdf(y, mu, sigma, nu):
    """Delaport log-PDF (Poisson + Shifted Gamma mixture)."""
    eps = jnp.finfo(jnp.float64).eps
    y = jnp.maximum(y, 0.0)
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.maximum(nu, eps)
    
    # Delaport = Poisson(lambda) + Gamma(alpha, beta)
    # Parameterization: mu = lambda + alpha*beta, sigma, nu
    lambda_ = mu / (1.0 + sigma * nu)
    alpha = 1.0 / (sigma * sigma)
    beta = sigma * sigma * nu
    
    # For integer y, sum over k from 0 to y
    # Approximation: use dominant term
    k = jnp.floor(y / 2.0)
    
    # Poisson part
    log_pois = k * jnp.log(lambda_) - lambda_ - gammaln(k + 1.0)
    
    # Gamma part for (y - k)
    y_minus_k = y - k
    log_gamma = (
        (alpha - 1.0) * jnp.log(jnp.maximum(y_minus_k, eps))
        - y_minus_k / beta
        - alpha * jnp.log(beta)
        - gammaln(alpha)
    )
    
    return log_pois + log_gamma


def DELAPORT():
    """Delaport distribution."""
    return build_ad_family(
        family_class=DELAPORTFamily,
        name="DELAPORT",
        parameters=("mu", "sigma", "nu"),
        log_pdf_func=_delaport_log_pdf,
        type_="Discrete",
        links={"mu": "log", "sigma": "log", "nu": "log"},
        link_functions={
            "mu": log_link,
            "sigma": log_link,
            "nu": log_link,
        },
        link_inverses={
            "mu": log_inverse,
            "sigma": log_inverse,
            "nu": log_inverse,
        },
        link_derivatives={
            "mu": log_derivative,
            "sigma": log_derivative,
            "nu": log_derivative,
        },
    )
