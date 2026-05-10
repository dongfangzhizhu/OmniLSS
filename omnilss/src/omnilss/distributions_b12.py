from __future__ import annotations

# Enable float64 precision for numerical accuracy
import jax
jax.config.update("jax_enable_x64", True)


"""Batch 12: Alternative parameterizations and type variants."""
from dataclasses import dataclass
import jax.numpy as jnp
from jax.scipy.special import gammaln, betaln

from .ad import build_ad_family
from .families import FamilyDefinition
from .links import (
    log_link, log_inverse, log_derivative,
    logit_link, logit_inverse, logit_derivative,
    identity_link, identity_inverse, identity_derivative
)


# ============================================================================
# BEo - Beta (Original Parameterization)
# ============================================================================

@dataclass(frozen=True)
class BEoFamily(FamilyDefinition):
    """Beta distribution (original parameterization) family."""
    pass


def _beo_log_pdf(y, mu, sigma):
    """Beta (original) log-PDF.
    
    Original parameterization: shape parameters alpha and beta directly.
    - mu = alpha
    - sigma = beta
    """
    eps = jnp.finfo(jnp.float64).eps
    y = jnp.clip(y, eps, 1.0 - eps)
    alpha = jnp.maximum(mu, eps)
    beta = jnp.maximum(sigma, eps)
    
    return (alpha - 1.0) * jnp.log(y) + (beta - 1.0) * jnp.log1p(-y) - betaln(alpha, beta)


def BEo():
    """Beta distribution with original parameterization."""
    return build_ad_family(
        family_class=BEoFamily,
        name="BEo",
        parameters=("mu", "sigma"),
        log_pdf_func=_beo_log_pdf,
        type_="Continuous",
        links={"mu": "log", "sigma": "log"},
        link_functions={"mu": log_link, "sigma": log_link},
        link_inverses={"mu": log_inverse, "sigma": log_inverse},
        link_derivatives={"mu": log_derivative, "sigma": log_derivative},
    )


# ============================================================================
# GEOMo - Geometric (Original Parameterization)
# ============================================================================

@dataclass(frozen=True)
class GEOMoFamily(FamilyDefinition):
    """Geometric distribution (original parameterization) family."""
    pass


def _geomo_log_pdf(y, mu):
    """Geometric (original) log-PDF.
    
    Original: P(Y=k) = (1-p)^k * p for k=0,1,2,...
    Here mu = p (success probability).
    """
    eps = jnp.finfo(jnp.float64).eps
    p = jnp.clip(mu, eps, 1.0 - eps)
    
    return y * jnp.log1p(-p) + jnp.log(p)


def GEOMo():
    """Geometric distribution with original parameterization."""
    return build_ad_family(
        family_class=GEOMoFamily,
        name="GEOMo",
        parameters=("mu",),
        log_pdf_func=_geomo_log_pdf,
        type_="Discrete",
        links={"mu": "logit"},
        link_functions={"mu": logit_link},
        link_inverses={"mu": logit_inverse},
        link_derivatives={"mu": logit_derivative},
    )


# ============================================================================
# WEI2 - Weibull Type 2
# ============================================================================

@dataclass(frozen=True)
class WEI2Family(FamilyDefinition):
    """Weibull Type 2 distribution family."""
    pass


def _wei2_log_pdf(y, mu, sigma):
    """Weibull Type 2 log-PDF.
    
    Alternative parameterization with different scale/shape relationship.
    """
    eps = jnp.finfo(jnp.float64).eps
    y = jnp.maximum(y, eps)
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    
    # Type 2: shape = 1/sigma, scale = mu * sigma
    shape = 1.0 / sigma
    scale = mu * sigma
    
    z = y / scale
    
    return (
        jnp.log(shape) - jnp.log(scale)
        + (shape - 1.0) * (jnp.log(y) - jnp.log(scale))
        - jnp.power(z, shape)
    )


def WEI2():
    """Weibull Type 2 distribution."""
    return build_ad_family(
        family_class=WEI2Family,
        name="WEI2",
        parameters=("mu", "sigma"),
        log_pdf_func=_wei2_log_pdf,
        type_="Continuous",
        links={"mu": "log", "sigma": "log"},
        link_functions={"mu": log_link, "sigma": log_link},
        link_inverses={"mu": log_inverse, "sigma": log_inverse},
        link_derivatives={"mu": log_derivative, "sigma": log_derivative},
    )


# ============================================================================
# WEI3 - Weibull Type 3
# ============================================================================

@dataclass(frozen=True)
class WEI3Family(FamilyDefinition):
    """Weibull Type 3 distribution family."""
    pass


def _wei3_log_pdf(y, mu, sigma):
    """Weibull Type 3 log-PDF.
    
    Another alternative parameterization.
    """
    eps = jnp.finfo(jnp.float64).eps
    y = jnp.maximum(y, eps)
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    
    # Type 3: shape = sigma, scale = mu / gamma(1 + 1/sigma)
    shape = sigma
    log_gamma_term = gammaln(1.0 + 1.0 / sigma)
    scale = mu / jnp.exp(log_gamma_term)
    
    z = y / scale
    
    return (
        jnp.log(shape) - jnp.log(scale)
        + (shape - 1.0) * (jnp.log(y) - jnp.log(scale))
        - jnp.power(z, shape)
    )


def WEI3():
    """Weibull Type 3 distribution."""
    return build_ad_family(
        family_class=WEI3Family,
        name="WEI3",
        parameters=("mu", "sigma"),
        log_pdf_func=_wei3_log_pdf,
        type_="Continuous",
        links={"mu": "log", "sigma": "log"},
        link_functions={"mu": log_link, "sigma": log_link},
        link_inverses={"mu": log_inverse, "sigma": log_inverse},
        link_derivatives={"mu": log_derivative, "sigma": log_derivative},
    )


# ============================================================================
# PARETO2o - Pareto Type 2 (Original)
# ============================================================================

@dataclass(frozen=True)
class PARETO2oFamily(FamilyDefinition):
    """Pareto Type 2 (original parameterization) family."""
    pass


def _pareto2o_log_pdf(y, mu, sigma):
    """Pareto Type 2 (original) log-PDF.
    
    Original parameterization with direct shape/scale.
    """
    eps = jnp.finfo(jnp.float64).eps
    y = jnp.maximum(y, eps)
    alpha = jnp.maximum(mu, eps)  # shape
    scale = jnp.maximum(sigma, eps)  # scale
    
    return (
        jnp.log(alpha) + alpha * jnp.log(scale)
        - (alpha + 1.0) * jnp.log(y + scale)
    )


def PARETO2o():
    """Pareto Type 2 with original parameterization."""
    return build_ad_family(
        family_class=PARETO2oFamily,
        name="PARETO2o",
        parameters=("mu", "sigma"),
        log_pdf_func=_pareto2o_log_pdf,
        type_="Continuous",
        links={"mu": "log", "sigma": "log"},
        link_functions={"mu": log_link, "sigma": log_link},
        link_inverses={"mu": log_inverse, "sigma": log_inverse},
        link_derivatives={"mu": log_derivative, "sigma": log_derivative},
    )


# ============================================================================
# NOF - Normal Family (alternative)
# ============================================================================

@dataclass(frozen=True)
class NOFFamily(FamilyDefinition):
    """Normal Family distribution."""
    pass


def _nof_log_pdf(y, mu, sigma):
    """Normal Family log-PDF.
    
    Standard normal with alternative naming.
    """
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    
    z = (y - mu) / sigma
    
    return -0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma) - 0.5 * z * z


def NOF():
    """Normal Family distribution."""
    return build_ad_family(
        family_class=NOFFamily,
        name="NOF",
        parameters=("mu", "sigma"),
        log_pdf_func=_nof_log_pdf,
        type_="Continuous",
        links={"mu": "identity", "sigma": "log"},
        link_functions={"mu": identity_link, "sigma": log_link},
        link_inverses={"mu": identity_inverse, "sigma": log_inverse},
        link_derivatives={"mu": identity_derivative, "sigma": log_derivative},
    )


# ============================================================================
# GAF - Gamma Family (alternative)
# ============================================================================

@dataclass(frozen=True)
class GAFFamily(FamilyDefinition):
    """Gamma Family distribution."""
    pass


def _gaf_log_pdf(y, mu, sigma):
    """Gamma Family log-PDF.
    
    Standard gamma with alternative naming.
    """
    eps = jnp.finfo(jnp.float64).eps
    y = jnp.maximum(y, eps)
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    
    shape = 1.0 / (sigma * sigma)
    scale = mu * sigma * sigma
    
    return (
        (shape - 1.0) * jnp.log(y) - y / scale
        - shape * jnp.log(scale) - gammaln(shape)
    )


def GAF():
    """Gamma Family distribution."""
    return build_ad_family(
        family_class=GAFFamily,
        name="GAF",
        parameters=("mu", "sigma"),
        log_pdf_func=_gaf_log_pdf,
        type_="Continuous",
        links={"mu": "log", "sigma": "log"},
        link_functions={"mu": log_link, "sigma": log_link},
        link_inverses={"mu": log_inverse, "sigma": log_inverse},
        link_derivatives={"mu": log_derivative, "sigma": log_derivative},
    )


# ============================================================================
# RGE - Reverse Gumbel Extended
# ============================================================================

@dataclass(frozen=True)
class RGEFamily(FamilyDefinition):
    """Reverse Gumbel Extended distribution family."""
    pass


def _rge_log_pdf(y, mu, sigma, nu):
    """Reverse Gumbel Extended log-PDF.
    
    Extension of reverse Gumbel with additional shape parameter.
    """
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.maximum(nu, eps)
    
    z = (y - mu) / sigma
    
    # Extended form with nu parameter
    exp_z = jnp.exp(z)
    
    return (
        -jnp.log(sigma) - z - nu * exp_z
        + (nu - 1.0) * jnp.log(nu)
        - gammaln(nu)
    )


def RGE():
    """Reverse Gumbel Extended distribution."""
    return build_ad_family(
        family_class=RGEFamily,
        name="RGE",
        parameters=("mu", "sigma", "nu"),
        log_pdf_func=_rge_log_pdf,
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
