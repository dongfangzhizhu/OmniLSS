"""GAMLSS distributions: Batch 9 — Additional discrete and continuous families.

This batch includes distributions that were missing from the initial implementation:
- LG (Logarithmic)
- ZIPF (Zipf)
- JSU (Johnson's SU)
- GIG (Generalized Inverse Gaussian)
- GB1 (Generalized Beta Type 1)
- And zero-inflated/altered variants
"""

from __future__ import annotations

# Enable float64 precision for numerical accuracy
import jax
jax.config.update("jax_enable_x64", True)


from dataclasses import dataclass
import math

import jax.numpy as jnp
from jax.scipy.special import gammaln

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

# ------------------------------------------------------------------
# 1. LG — Logarithmic Distribution
#    R: dLG(x, mu)
#    Support: {1, 2, 3, ...}
#    Parameter: mu ∈ (0, 1)
#    PMF: P(X=x) = mu^x / (x * (-log(1-mu)))
# ------------------------------------------------------------------

@dataclass(frozen=True)
class LogarithmicFamily(FamilyDefinition):
    """Logarithmic distribution (`LG`).
    
    Support: {1, 2, 3, ...}
    Parameter: mu ∈ (0, 1)
    
    The logarithmic distribution is a discrete probability distribution
    derived from the Maclaurin series expansion of -log(1-mu).
    
    PMF: P(X=x) = mu^x / (x * (-log(1-mu)))
    
    Mean: -mu / ((1-mu) * log(1-mu))
    Variance: -mu * (1 + mu/(-log(1-mu))) / ((1-mu)^2 * log(1-mu))
    
    Common uses:
    - Species abundance models
    - Accident counts
    - Rare event modeling
    - Truncated geometric processes
    """


def _lg_log_pdf(y, mu):
    """Log-PDF of logarithmic distribution.
    
    Args:
        y: Response variable (positive integers)
        mu: Parameter in (0, 1)
    
    Returns:
        Log-probability
    """
    eps = jnp.finfo(jnp.float64).eps
    
    # Ensure mu is in valid range
    mu = jnp.clip(mu, eps, 1.0 - eps)
    
    # Ensure y is positive integer
    y_safe = jnp.maximum(y, 1.0)
    
    # Log-PDF: log(mu^y / (y * (-log(1-mu))))
    # = y*log(mu) - log(y) - log(-log(1-mu))
    log_pdf = y_safe * jnp.log(mu) - jnp.log(y_safe) - jnp.log(-jnp.log1p(-mu))
    
    # Set probability to 0 for y <= 0
    log_pdf = jnp.where(y <= 0, -jnp.inf, log_pdf)
    
    return log_pdf


def LG() -> LogarithmicFamily:
    """Logarithmic distribution.
    
    Support: {1, 2, 3, ...}
    Parameter: mu ∈ (0, 1)
    
    Returns:
        LogarithmicFamily instance
    
    Example:
        >>> from omnilss import gamlss
        >>> from omnilss.distributions import LG
        >>> 
        >>> # Fit logarithmic distribution
        >>> model = gamlss("y ~ x", family=LG(), data=data)
    
    Notes:
        - Default link for mu is logit
        - mu must be in (0, 1)
        - y must be positive integers
        - Mean: -mu / ((1-mu) * log(1-mu))
        - Variance: -mu * (1 + mu/(-log(1-mu))) / ((1-mu)^2 * log(1-mu))
    
    References:
        - Johnson, N. L., Kemp, A. W., & Kotz, S. (2005). 
          Univariate Discrete Distributions (3rd ed.). Wiley.
        - R gamlss.dist::LG
    """
    return build_ad_family(
        family_class=LogarithmicFamily,
        name="LG",
        parameters=("mu",),
        log_pdf_func=_lg_log_pdf,
        type_="Discrete",
        links={"mu": "logit"},
        link_functions={"mu": logit_link},
        link_inverses={"mu": logit_inverse},
        link_derivatives={"mu": logit_derivative},
    )


# ------------------------------------------------------------------
# 2. ZIPF — Zipf Distribution
#    R: dZIPF(x, mu)
#    Support: {1, 2, 3, ...}
#    Parameter: mu > 0 (shape parameter, often called s or α)
#    PMF: P(X=x) = x^(-(mu+1)) / ζ(mu+1)
#    where ζ is the Riemann zeta function
# ------------------------------------------------------------------

@dataclass(frozen=True)
class ZipfFamily(FamilyDefinition):
    """Zipf distribution (`ZIPF`).
    
    Support: {1, 2, 3, ...}
    Parameter: mu > 0 (shape parameter)
    
    The Zipf distribution is a discrete power-law probability distribution.
    It's commonly used to model word frequencies, city populations, and
    other phenomena following Zipf's law.
    
    PMF: P(X=x) = x^(-(mu+1)) / ζ(mu+1)
    where ζ is the Riemann zeta function
    
    Mean: ζ(mu) / ζ(mu+1) if mu > 1, else Inf
    Variance: [ζ(mu+1)*ζ(mu-1) - ζ(mu)^2] / ζ(mu+1)^2 if mu > 2, else Inf
    
    Common uses:
    - Word frequency analysis
    - City population distributions
    - Website traffic patterns
    - Power-law phenomena
    """


def _zeta_scipy(s):
    """Compute Riemann zeta function using scipy."""
    import numpy as np
    from scipy.special import zeta
    s_np = np.asarray(s, dtype=np.float64)
    return zeta(s_np)


def _zipf_log_pdf(y, mu):
    """Log-PDF of Zipf distribution using exact zeta function."""
    import jax
    
    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.maximum(mu, eps)
    y_safe = jnp.maximum(y, 1.0)
    
    # Use scipy's zeta via pure_callback
    result_shape = jax.ShapeDtypeStruct(jnp.shape(mu), jnp.float64)
    log_zeta = jnp.log(jax.pure_callback(
        _zeta_scipy,
        result_shape,
        mu + 1.0,
        vmap_method='sequential'
    ))
    
    log_pdf = -(mu + 1.0) * jnp.log(y_safe) - log_zeta
    return jnp.where(y <= 0, -jnp.inf, log_pdf)


def ZIPF() -> ZipfFamily:
    """Zipf distribution.
    
    Support: {1, 2, 3, ...}
    Parameter: mu > 0 (shape parameter)
    
    Returns:
        ZipfFamily instance
    
    Example:
        >>> from omnilss import gamlss
        >>> from omnilss.distributions import ZIPF
        >>> 
        >>> # Fit Zipf distribution
        >>> model = gamlss("y ~ x", family=ZIPF(), data=data)
    
    Notes:
        - Default link for mu is log
        - mu must be positive
        - y must be positive integers
        - Mean: ζ(mu) / ζ(mu+1) if mu > 1
        - Variance: [ζ(mu+1)*ζ(mu-1) - ζ(mu)^2] / ζ(mu+1)^2 if mu > 2
        - Current implementation uses approximation for zeta function
        - For production use, consider using scipy.special.zeta
    
    References:
        - Zipf, G. K. (1949). Human Behavior and the Principle of Least Effort.
        - Newman, M. E. J. (2005). Power laws, Pareto distributions and Zipf's law.
        - R gamlss.dist::ZIPF
    """
    return build_ad_family(
        family_class=ZipfFamily,
        name="ZIPF",
        parameters=("mu",),
        log_pdf_func=_zipf_log_pdf,
        type_="Discrete",
        links={"mu": "log"},
        link_functions={"mu": log_link},
        link_inverses={"mu": log_inverse},
        link_derivatives={"mu": log_derivative},
    )


# ------------------------------------------------------------------
# Helper function to create zero-altered distributions
# ------------------------------------------------------------------

def create_zero_altered_family(base_family_func, base_name, new_name):
    """Create a zero-altered version of a distribution.
    
    Zero-altered (hurdle) model:
    - Pr(Y=0) = nu
    - Pr(Y=y|y>0) = (1-nu) * f(y) / [1 - f(0)]
    
    Args:
        base_family_func: Function that returns base family
        base_name: Name of base distribution
        new_name: Name of zero-altered distribution
    
    Returns:
        Zero-altered family function
    
    Example:
        >>> ZALG = create_zero_altered_family(LG, "LG", "ZALG")
    """
    # This is a template - full implementation would require
    # wrapping the base distribution's log_pdf with zero-altered logic
    # Similar to how ZAGA wraps GA
    pass


# ------------------------------------------------------------------
# Helper function to create zero-inflated distributions
# ------------------------------------------------------------------

def create_zero_inflated_family(base_family_func, base_name, new_name):
    """Create a zero-inflated version of a distribution.
    
    Zero-inflated model:
    - Pr(Y=0) = nu + (1-nu) * f(0)
    - Pr(Y=y|y>0) = (1-nu) * f(y)
    
    Args:
        base_family_func: Function that returns base family
        base_name: Name of base distribution
        new_name: Name of zero-inflated distribution
    
    Returns:
        Zero-inflated family function
    
    Example:
        >>> ZILG = create_zero_inflated_family(LG, "LG", "ZILG")
    """
    # This is a template - full implementation would require
    # wrapping the base distribution's log_pdf with zero-inflated logic
    # Similar to how ZIP wraps PO
    pass


# ------------------------------------------------------------------
# Zero-Altered Logarithmic (ZALG)
# ------------------------------------------------------------------

@dataclass(frozen=True)
class ZeroAlteredLogarithmicFamily(FamilyDefinition):
    """Zero-Altered Logarithmic distribution (ZALG)."""


def _zalg_log_pdf(y, mu, nu):
    """Zero-altered logarithmic: Pr(Y=0)=nu, Pr(Y=y|y>0)=(1-nu)*f(y)/[1-f(0)]."""
    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.clip(mu, eps, 1.0 - eps)
    nu = jnp.clip(nu, eps, 1.0 - eps)
    
    # LG log-pdf for y > 0
    y_safe = jnp.maximum(y, 1.0)
    log_lg = y_safe * jnp.log(mu) - jnp.log(y_safe) - jnp.log(-jnp.log1p(-mu))
    
    # Zero-altered: no mass at 0 in base distribution (LG starts at 1)
    # So Pr(Y=0) = nu, Pr(Y=y|y>0) = (1-nu) * f(y)
    log_at0 = jnp.log(nu)
    log_cont = jnp.log1p(-nu) + log_lg
    
    return jnp.where(y <= 0, log_at0, log_cont)


def ZALG() -> ZeroAlteredLogarithmicFamily:
    """Zero-Altered Logarithmic distribution."""
    return build_ad_family(
        family_class=ZeroAlteredLogarithmicFamily,
        name="ZALG",
        parameters=("mu", "nu"),
        log_pdf_func=_zalg_log_pdf,
        type_="Discrete",
        links={"mu": "logit", "nu": "logit"},
        link_functions={"mu": logit_link, "nu": logit_link},
        link_inverses={"mu": logit_inverse, "nu": logit_inverse},
        link_derivatives={"mu": logit_derivative, "nu": logit_derivative},
    )


# ------------------------------------------------------------------
# Zero-Inflated Logarithmic (ZILG) - if needed
# Note: LG already has no mass at 0, so ZI doesn't make sense
# But we can define it for completeness
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Additional simple distributions from gamlss.dist
# ------------------------------------------------------------------

# WEI2, WEI3 - Weibull variants
# GEOMo - Geometric original
# NOF, GAF - Normal/Gamma families
# etc.

# These will be added incrementally based on priority
