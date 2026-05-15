from __future__ import annotations

# Enable float64 precision for numerical accuracy

"""Batch 14: Skew distribution variants and SHASH variants."""
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
# SHASHo - SHASH Original Parameterization
# ============================================================================

@dataclass(frozen=True)
class SHASHoFamily(FamilyDefinition):
    """SHASH Original parameterization family."""
    pass


def _shasho_log_pdf(y, mu, sigma, nu, tau):
    """SHASH Original log-PDF."""
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    tau = jnp.maximum(tau, eps)
    
    z = (y - mu) / sigma
    r = jnp.sinh((jnp.arcsinh(z) - nu) / tau)
    
    return (
        -jnp.log(sigma) - jnp.log(tau)
        - 0.5 * jnp.log(2.0 * jnp.pi)
        - 0.5 * jnp.log1p(z * z)
        - 0.5 * r * r
    )


def SHASHo():
    """SHASH with original parameterization."""
    return build_ad_family(
        family_class=SHASHoFamily,
        name="SHASHo",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_shasho_log_pdf,
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
# SHASHo2 - SHASH Original 2
# ============================================================================

@dataclass(frozen=True)
class SHASHo2Family(FamilyDefinition):
    """SHASH Original 2 parameterization family."""
    pass


def _shasho2_log_pdf(y, mu, sigma, nu, tau):
    """SHASH Original 2 log-PDF with different parameterization."""
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    tau = jnp.maximum(tau, eps)
    
    # Alternative transformation
    z = (y - mu) / sigma
    asinh_z = jnp.arcsinh(z)
    r = jnp.sinh(tau * asinh_z + nu)
    
    return (
        -jnp.log(sigma) + jnp.log(tau)
        - 0.5 * jnp.log(2.0 * jnp.pi)
        - 0.5 * jnp.log1p(z * z)
        - 0.5 * r * r
    )


def SHASHo2():
    """SHASH with original 2 parameterization."""
    return build_ad_family(
        family_class=SHASHo2Family,
        name="SHASHo2",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_shasho2_log_pdf,
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
# JSUo - Johnson SU Original
# ============================================================================

@dataclass(frozen=True)
class JSUoFamily(FamilyDefinition):
    """Johnson SU Original parameterization family."""
    pass


def _jsuo_log_pdf(y, mu, sigma, nu, tau):
    """JSU Original log-PDF with direct gamma, delta, xi, lambda."""
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    tau = jnp.maximum(tau, eps)
    
    # Original: gamma=nu, delta=tau, xi=mu, lambda=sigma
    z = (y - mu) / sigma
    r = nu + tau * jnp.arcsinh(z)
    
    return (
        jnp.log(tau) - jnp.log(sigma)
        - 0.5 * jnp.log(2.0 * jnp.pi)
        - 0.5 * jnp.log1p(z * z)
        - 0.5 * r * r
    )


def JSUo():
    """Johnson SU with original parameterization."""
    return build_ad_family(
        family_class=JSUoFamily,
        name="JSUo",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_jsuo_log_pdf,
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
# ST5 - Skew t Type 5
# ============================================================================

@dataclass(frozen=True)
class ST5Family(FamilyDefinition):
    """Skew t Type 5 family."""
    pass


def _st5_log_pdf(y, mu, sigma, nu, tau):
    """Skew t Type 5 log-PDF."""
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    tau = jnp.maximum(tau, 2.0 + eps)  # df > 2
    
    z = (y - mu) / sigma
    
    # Skew factor
    skew_factor = jnp.exp(nu * z)
    
    # t distribution part
    log_t = (
        gammaln((tau + 1.0) / 2.0) - gammaln(tau / 2.0)
        - 0.5 * jnp.log(tau * jnp.pi)
        - jnp.log(sigma)
        - ((tau + 1.0) / 2.0) * jnp.log1p(z * z / tau)
    )
    
    return log_t + jnp.log(skew_factor)


def ST5():
    """Skew t Type 5 distribution."""
    return build_ad_family(
        family_class=ST5Family,
        name="ST5",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_st5_log_pdf,
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
# BCPEo - Box-Cox Power Exponential Original
# ============================================================================

@dataclass(frozen=True)
class BCPEoFamily(FamilyDefinition):
    """Box-Cox Power Exponential Original family."""
    pass


def _bcpeo_log_pdf(y, mu, sigma, nu, tau):
    """BCPE Original log-PDF."""
    eps = jnp.finfo(jnp.float64).eps
    y = jnp.maximum(y, eps)
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    tau = jnp.clip(tau, 0.1, 10.0)
    
    # Box-Cox transformation
    z = jnp.where(
        jnp.abs(nu) > eps,
        (jnp.power(y, nu) - 1.0) / nu,
        jnp.log(y)
    )
    
    z_std = (z - mu) / sigma
    
    # Power exponential
    log_pe = (
        jnp.log(tau) - jnp.log(2.0 * sigma) - gammaln(1.0 / tau)
        - jnp.power(jnp.abs(z_std), tau)
    )
    
    # Jacobian
    log_jacobian = (nu - 1.0) * jnp.log(y)
    
    return log_pe + log_jacobian


def BCPEo():
    """Box-Cox Power Exponential with original parameterization."""
    return build_ad_family(
        family_class=BCPEoFamily,
        name="BCPEo",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_bcpeo_log_pdf,
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
# BCTo - Box-Cox t Original
# ============================================================================

@dataclass(frozen=True)
class BCToFamily(FamilyDefinition):
    """Box-Cox t Original family."""
    pass


def _bcto_log_pdf(y, mu, sigma, nu, tau):
    """BCT Original log-PDF."""
    eps = jnp.finfo(jnp.float64).eps
    y = jnp.maximum(y, eps)
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    tau = jnp.maximum(tau, 2.0 + eps)
    
    # Box-Cox transformation
    z = jnp.where(
        jnp.abs(nu) > eps,
        (jnp.power(y, nu) - 1.0) / nu,
        jnp.log(y)
    )
    
    z_std = (z - mu) / sigma
    
    # t distribution
    log_t = (
        gammaln((tau + 1.0) / 2.0) - gammaln(tau / 2.0)
        - 0.5 * jnp.log(tau * jnp.pi) - jnp.log(sigma)
        - ((tau + 1.0) / 2.0) * jnp.log1p(z_std * z_std / tau)
    )
    
    # Jacobian
    log_jacobian = (nu - 1.0) * jnp.log(y)
    
    return log_t + log_jacobian


def BCTo():
    """Box-Cox t with original parameterization."""
    return build_ad_family(
        family_class=BCToFamily,
        name="BCTo",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_bcto_log_pdf,
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
