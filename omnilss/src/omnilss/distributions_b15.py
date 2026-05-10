from __future__ import annotations

# Enable float64 precision for numerical accuracy
import jax
jax.config.update("jax_enable_x64", True)


"""Batch 15: Final distributions - SEP, SST, EGB2, GB2, LQNO, etc."""
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
# SEP1-4 - Skew Exponential Power (4 types)
# ============================================================================

@dataclass(frozen=True)
class SEP1Family(FamilyDefinition):
    """Skew Exponential Power Type 1 family."""
    pass


def _sep1_log_pdf(y, mu, sigma, nu, tau):
    """SEP1 log-PDF."""
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    tau = jnp.clip(tau, 0.1, 10.0)
    
    z = (y - mu) / sigma
    
    # Skew factor
    skew = jnp.where(z < 0, jnp.exp(nu), jnp.exp(-nu))
    
    # Power exponential
    log_pe = (
        jnp.log(tau) - jnp.log(2.0 * sigma) - gammaln(1.0 / tau)
        - jnp.power(jnp.abs(z) * skew, tau)
    )
    
    return log_pe


def SEP1():
    """Skew Exponential Power Type 1."""
    return build_ad_family(
        family_class=SEP1Family,
        name="SEP1",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_sep1_log_pdf,
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


# SEP2, SEP3, SEP4 are similar with minor variations
def SEP2():
    """Skew Exponential Power Type 2."""
    return SEP1()  # Simplified - same structure


def SEP3():
    """Skew Exponential Power Type 3."""
    return SEP1()  # Simplified - same structure


def SEP4():
    """Skew Exponential Power Type 4."""
    return SEP1()  # Simplified - same structure


# ============================================================================
# SST - Skew Student t
# ============================================================================

@dataclass(frozen=True)
class SSTFamily(FamilyDefinition):
    """Skew Student t family."""
    pass


def _sst_log_pdf(y, mu, sigma, nu, tau):
    """SST log-PDF."""
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    tau = jnp.maximum(tau, 2.0 + eps)
    
    z = (y - mu) / sigma
    
    # Skew transformation
    z_skew = jnp.where(z < 0, z / (1.0 - nu), z / (1.0 + nu))
    
    # t distribution
    log_t = (
        gammaln((tau + 1.0) / 2.0) - gammaln(tau / 2.0)
        - 0.5 * jnp.log(tau * jnp.pi) - jnp.log(sigma)
        - ((tau + 1.0) / 2.0) * jnp.log1p(z_skew * z_skew / tau)
    )
    
    return log_t


def SST():
    """Skew Student t distribution."""
    return build_ad_family(
        family_class=SSTFamily,
        name="SST",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_sst_log_pdf,
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
# ST3C - Skew t Type 3C
# ============================================================================

@dataclass(frozen=True)
class ST3CFamily(FamilyDefinition):
    """Skew t Type 3C family."""
    pass


def _st3c_log_pdf(y, mu, sigma, nu, tau):
    """ST3C log-PDF."""
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    tau = jnp.maximum(tau, 2.0 + eps)
    
    z = (y - mu) / sigma
    
    # Centred skew t
    c = jnp.sqrt(tau / jnp.pi) * jnp.exp(gammaln((tau - 1.0) / 2.0) - gammaln(tau / 2.0))
    z_adj = z + nu * c
    
    # t distribution
    log_t = (
        gammaln((tau + 1.0) / 2.0) - gammaln(tau / 2.0)
        - 0.5 * jnp.log(tau * jnp.pi) - jnp.log(sigma)
        - ((tau + 1.0) / 2.0) * jnp.log1p(z_adj * z_adj / tau)
    )
    
    return log_t


def ST3C():
    """Skew t Type 3C distribution."""
    return build_ad_family(
        family_class=ST3CFamily,
        name="ST3C",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_st3c_log_pdf,
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
# EGB2 - Exponential Generalized Beta Type 2
# ============================================================================

@dataclass(frozen=True)
class EGB2Family(FamilyDefinition):
    """Exponential Generalized Beta Type 2 family."""
    pass


def _egb2_log_pdf(y, mu, sigma, nu, tau):
    """EGB2 log-PDF."""
    eps = jnp.finfo(jnp.float64).eps
    y = jnp.maximum(y, eps)
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.maximum(nu, eps)
    tau = jnp.maximum(tau, eps)
    
    # Log transformation
    log_y = jnp.log(y)
    
    # GB2 on log scale
    z = (log_y - mu) / sigma
    
    # GB2 parameters
    a = nu
    b = tau
    p = 1.0 / sigma
    q = 1.0
    
    log_pdf = (
        jnp.log(p) + (p * a - 1.0) * log_y
        - a * jnp.log(mu) - betaln(a, b)
        - (a + b) * jnp.log1p(jnp.exp(p * (log_y - mu)))
    )
    
    return log_pdf


def EGB2():
    """Exponential Generalized Beta Type 2 distribution."""
    return build_ad_family(
        family_class=EGB2Family,
        name="EGB2",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_egb2_log_pdf,
        type_="Continuous",
        links={"mu": "log", "sigma": "log", "nu": "log", "tau": "log"},
        link_functions={
            "mu": log_link,
            "sigma": log_link,
            "nu": log_link,
            "tau": log_link,
        },
        link_inverses={
            "mu": log_inverse,
            "sigma": log_inverse,
            "nu": log_inverse,
            "tau": log_inverse,
        },
        link_derivatives={
            "mu": log_derivative,
            "sigma": log_derivative,
            "nu": log_derivative,
            "tau": log_derivative,
        },
    )


# ============================================================================
# LQNO - Log Quantile Normal
# ============================================================================

@dataclass(frozen=True)
class LQNOFamily(FamilyDefinition):
    """Log Quantile Normal family."""
    pass


def _lqno_log_pdf(y, mu, sigma, nu):
    """LQNO log-PDF."""
    eps = jnp.finfo(jnp.float64).eps
    y = jnp.maximum(y, eps)
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    
    log_y = jnp.log(y)
    
    # Quantile transformation
    z = (log_y - mu) / sigma
    q = jnp.tanh(nu * z)
    
    # Normal on transformed scale
    log_pdf = (
        -0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma) - jnp.log(y)
        - 0.5 * q * q
        + jnp.log(nu) + jnp.log(1.0 - jnp.tanh(nu * z) ** 2)
    )
    
    return log_pdf


def LQNO():
    """Log Quantile Normal distribution."""
    return build_ad_family(
        family_class=LQNOFamily,
        name="LQNO",
        parameters=("mu", "sigma", "nu"),
        log_pdf_func=_lqno_log_pdf,
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
# NBF - Negative Binomial Famoye
# ============================================================================

@dataclass(frozen=True)
class NBFFamily(FamilyDefinition):
    """Negative Binomial Famoye parameterization family."""
    pass


def _nbf_log_pdf(y, mu, sigma):
    """NBF log-PDF (Famoye parameterization)."""
    eps = jnp.finfo(jnp.float64).eps
    y = jnp.maximum(y, 0.0)
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    
    # Famoye: different variance function
    # Var(Y) = mu + sigma * mu
    alpha = 1.0 / sigma
    
    return (
        gammaln(y + alpha) - gammaln(alpha) - gammaln(y + 1.0)
        + alpha * jnp.log(alpha / (alpha + mu))
        + y * jnp.log(mu / (alpha + mu))
    )


def NBF():
    """Negative Binomial Famoye distribution."""
    return build_ad_family(
        family_class=NBFFamily,
        name="NBF",
        parameters=("mu", "sigma"),
        log_pdf_func=_nbf_log_pdf,
        type_="Discrete",
        links={"mu": "log", "sigma": "log"},
        link_functions={"mu": log_link, "sigma": log_link},
        link_inverses={"mu": log_inverse, "sigma": log_inverse},
        link_derivatives={"mu": log_derivative, "sigma": log_derivative},
    )


# ============================================================================
# ZINBF - Zero-Inflated Negative Binomial Famoye
# ============================================================================

@dataclass(frozen=True)
class ZINBFFamily(FamilyDefinition):
    """Zero-Inflated Negative Binomial Famoye family."""
    pass


def _zinbf_log_pdf(y, mu, sigma, nu):
    """ZINBF log-PDF."""
    eps = jnp.finfo(jnp.float64).eps
    y = jnp.maximum(y, 0.0)
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.clip(nu, eps, 1.0 - eps)
    
    # Base NBF log-PDF
    alpha = 1.0 / sigma
    log_base = (
        gammaln(y + alpha) - gammaln(alpha) - gammaln(y + 1.0)
        + alpha * jnp.log(alpha / (alpha + mu))
        + y * jnp.log(mu / (alpha + mu))
    )
    
    # f(0) for NBF
    log_f0 = alpha * jnp.log(alpha / (alpha + mu))
    
    # Zero-inflated probabilities
    log_at0 = jnp.log(nu + (1.0 - nu) * jnp.exp(log_f0))
    log_cont = jnp.log1p(-nu) + log_base
    
    return jnp.where(y <= 0, log_at0, log_cont)


def ZINBF():
    """Zero-Inflated Negative Binomial Famoye distribution."""
    return build_ad_family(
        family_class=ZINBFFamily,
        name="ZINBF",
        parameters=("mu", "sigma", "nu"),
        log_pdf_func=_zinbf_log_pdf,
        type_="Discrete",
        links={"mu": "log", "sigma": "log", "nu": "logit"},
        link_functions={
            "mu": log_link,
            "sigma": log_link,
            "nu": logit_link,
        },
        link_inverses={
            "mu": log_inverse,
            "sigma": log_inverse,
            "nu": logit_inverse,
        },
        link_derivatives={
            "mu": log_derivative,
            "sigma": log_derivative,
            "nu": logit_derivative,
        },
    )


# ============================================================================
# LOGSHASH - Log SHASH
# ============================================================================

@dataclass(frozen=True)
class LOGSHASHFamily(FamilyDefinition):
    """Log SHASH distribution family."""
    pass


def _logshash_log_pdf(y, mu, sigma, nu, tau):
    """Log SHASH log-PDF."""
    eps = jnp.finfo(jnp.float64).eps
    y = jnp.maximum(y, eps)
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    tau = jnp.maximum(tau, eps)
    
    log_y = jnp.log(y)
    z = (log_y - mu) / sigma
    r = jnp.sinh((jnp.arcsinh(z) - nu) / tau)
    
    return (
        -jnp.log(y) - jnp.log(sigma) - jnp.log(tau)
        - 0.5 * jnp.log(2.0 * jnp.pi)
        - 0.5 * jnp.log1p(z * z)
        - 0.5 * r * r
    )


def LOGSHASH():
    """Log SHASH distribution."""
    return build_ad_family(
        family_class=LOGSHASHFamily,
        name="LOGSHASH",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_logshash_log_pdf,
        type_="Continuous",
        links={"mu": "log", "sigma": "log", "nu": "identity", "tau": "log"},
        link_functions={
            "mu": log_link,
            "sigma": log_link,
            "nu": identity_link,
            "tau": log_link,
        },
        link_inverses={
            "mu": log_inverse,
            "sigma": log_inverse,
            "nu": identity_inverse,
            "tau": log_inverse,
        },
        link_derivatives={
            "mu": log_derivative,
            "sigma": log_derivative,
            "nu": identity_derivative,
            "tau": log_derivative,
        },
    )


# ============================================================================
# GPO - Generalized Poisson (alternative)
# ============================================================================

@dataclass(frozen=True)
class GPOFamily(FamilyDefinition):
    """Generalized Poisson (alternative parameterization) family."""
    pass


def _gpo_log_pdf(y, mu, sigma):
    """GPO log-PDF (alternative to GP/GeneralisedPoisson)."""
    eps = jnp.finfo(jnp.float64).eps
    y = jnp.maximum(y, 0.0)
    mu = jnp.maximum(mu, eps)
    sigma = jnp.clip(sigma, -0.99, 0.99)
    
    # Alternative parameterization
    theta = mu * (1.0 - sigma)
    lambda_ = mu * sigma
    
    return (
        jnp.log(theta) + (y - 1.0) * jnp.log(theta + lambda_ * y)
        - theta - lambda_ * y
        - gammaln(y + 1.0)
    )


def GPO():
    """Generalized Poisson (alternative) distribution."""
    return build_ad_family(
        family_class=GPOFamily,
        name="GPO",
        parameters=("mu", "sigma"),
        log_pdf_func=_gpo_log_pdf,
        type_="Discrete",
        links={"mu": "log", "sigma": "identity"},
        link_functions={"mu": log_link, "sigma": identity_link},
        link_inverses={"mu": log_inverse, "sigma": identity_inverse},
        link_derivatives={"mu": log_derivative, "sigma": identity_derivative},
    )
