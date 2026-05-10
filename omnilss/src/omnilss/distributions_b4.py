"""GAMLSS distributions: Batch 4 — Beta-inflated families.

These are mixture models for proportional data on (0,1) that may contain
exact 0s and/or 1s (boundary mass points).

Distributions:
- BEINF   (Beta inflated at 0 and 1, 4-parameter)
- BEINF0  (Beta inflated at 0, 3-parameter)
- BEINF1  (Beta inflated at 1, 3-parameter)
- BEZI    (Beta zero-inflated, 3-parameter)
- BEOI    (Beta one-inflated, 3-parameter)
"""

from __future__ import annotations

# Enable float64 precision for numerical accuracy
import jax
jax.config.update("jax_enable_x64", True)


from dataclasses import dataclass

import jax.numpy as jnp
from jax.scipy.special import betaln

from .ad import build_ad_family
from .families import FamilyDefinition
from .links import (
    log_derivative,
    log_inverse,
    log_link,
    logit_derivative,
    logit_inverse,
    logit_link,
)
from .dpqr_functions import dBEINF, pBEINF, qBEINF, rBEINF


def _beta_log_pdf(y, a, b):
    """Core log pdf of Beta(a, b). No boundary checking — handled by caller."""
    eps = jnp.finfo(jnp.float64).eps
    y = jnp.clip(y, eps, 1.0 - eps)
    return (a - 1.0) * jnp.log(y) + (b - 1.0) * jnp.log1p(-y) - betaln(a, b)


def _beinf_ab(mu, sigma):
    """Convert (mu, sigma) parameterization used by BEINF-family to Beta (a, b)."""
    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.clip(mu, eps, 1.0 - eps)
    sigma = jnp.clip(sigma, eps, 1.0 - eps)
    a = mu * (1.0 - sigma**2) / sigma**2
    b = a * (1.0 - mu) / mu
    return jnp.maximum(a, eps), jnp.maximum(b, eps)


# ------------------------------------------------------------------
# 1. BEINF — Beta inflated at 0 AND 1 (4-param)
#    R: logfy[0<x<1] = dbeta(x,a,b,log=TRUE) - log(1+nu+tau)
#       logfy[x=0]   = log(nu)                - log(1+nu+tau)
#       logfy[x=1]   = log(tau)               - log(1+nu+tau)
#    Note: AD is used but the log-pdf must be continuous for AD to work.
#    We implement via soft masks using jnp.where (no grad through discontinuity).
# ------------------------------------------------------------------

@dataclass(frozen=True)
class BetaInflatedFamily(FamilyDefinition):
    """Beta-inflated at 0 and 1 (`BEINF`)."""


def _beinf_log_pdf(y, mu, sigma, nu, tau):
    eps = jnp.finfo(jnp.float64).eps
    nu = jnp.maximum(nu, eps)
    tau = jnp.maximum(tau, eps)
    a, b = _beinf_ab(mu, sigma)
    log_denom = jnp.log(1.0 + nu + tau)
    log_cont = _beta_log_pdf(y, a, b) - log_denom
    log_at0 = jnp.log(nu) - log_denom
    log_at1 = jnp.log(tau) - log_denom
    # Use soft routing via where; AD differentiates through the chosen branch
    return jnp.where(y <= 0.0, log_at0, jnp.where(y >= 1.0, log_at1, log_cont))


def BEINF() -> BetaInflatedFamily:
    return build_ad_family(
        family_class=BetaInflatedFamily,
        name="BEINF",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_beinf_log_pdf,
        type_="Mixed",
        links={"mu": "logit", "sigma": "logit", "nu": "log", "tau": "log"},
        link_functions={"mu": logit_link, "sigma": logit_link, "nu": log_link, "tau": log_link},
        link_inverses={"mu": logit_inverse, "sigma": logit_inverse, "nu": log_inverse, "tau": log_inverse},
        link_derivatives={"mu": logit_derivative, "sigma": logit_derivative, "nu": log_derivative, "tau": log_derivative},
        d=dBEINF,
        p=pBEINF,
        q=qBEINF,
        r=rBEINF,
    )


# ------------------------------------------------------------------
# 2. BEINF0 — Beta inflated at 0 only (3-param)
#    R: logfy[x>0] = dbeta(x,a,b,log=TRUE) - log(1+nu)
#       logfy[x=0] = log(nu) - log(1+nu)
# ------------------------------------------------------------------

@dataclass(frozen=True)
class BetaInflated0Family(FamilyDefinition):
    """Beta inflated at 0 (`BEINF0`)."""


def _beinf0_log_pdf(y, mu, sigma, nu):
    eps = jnp.finfo(jnp.float64).eps
    nu = jnp.maximum(nu, eps)
    a, b = _beinf_ab(mu, sigma)
    log_denom = jnp.log1p(nu)
    log_cont = _beta_log_pdf(y, a, b) - log_denom
    log_at0 = jnp.log(nu) - log_denom
    return jnp.where(y <= 0.0, log_at0, log_cont)


def BEINF0() -> BetaInflated0Family:
    return build_ad_family(
        family_class=BetaInflated0Family,
        name="BEINF0",
        parameters=("mu", "sigma", "nu"),
        log_pdf_func=_beinf0_log_pdf,
        type_="Mixed",
        links={"mu": "logit", "sigma": "logit", "nu": "log"},
        link_functions={"mu": logit_link, "sigma": logit_link, "nu": log_link},
        link_inverses={"mu": logit_inverse, "sigma": logit_inverse, "nu": log_inverse},
        link_derivatives={"mu": logit_derivative, "sigma": logit_derivative, "nu": log_derivative},
    )


# ------------------------------------------------------------------
# 3. BEINF1 — Beta inflated at 1 only (3-param)
#    R: logfy[x<1] = dbeta(x,a,b,log=TRUE) - log(1+nu)
#       logfy[x=1] = log(nu) - log(1+nu)
# ------------------------------------------------------------------

@dataclass(frozen=True)
class BetaInflated1Family(FamilyDefinition):
    """Beta inflated at 1 (`BEINF1`)."""


def _beinf1_log_pdf(y, mu, sigma, nu):
    eps = jnp.finfo(jnp.float64).eps
    nu = jnp.maximum(nu, eps)
    a, b = _beinf_ab(mu, sigma)
    log_denom = jnp.log1p(nu)
    log_cont = _beta_log_pdf(y, a, b) - log_denom
    log_at1 = jnp.log(nu) - log_denom
    return jnp.where(y >= 1.0, log_at1, log_cont)


def BEINF1() -> BetaInflated1Family:
    return build_ad_family(
        family_class=BetaInflated1Family,
        name="BEINF1",
        parameters=("mu", "sigma", "nu"),
        log_pdf_func=_beinf1_log_pdf,
        type_="Mixed",
        links={"mu": "logit", "sigma": "logit", "nu": "log"},
        link_functions={"mu": logit_link, "sigma": logit_link, "nu": log_link},
        link_inverses={"mu": logit_inverse, "sigma": logit_inverse, "nu": log_inverse},
        link_derivatives={"mu": logit_derivative, "sigma": logit_derivative, "nu": log_derivative},
    )


# ------------------------------------------------------------------
# 4. BEZI — Beta zero-inflated (RE-parameterized sigma = precision)
#    R uses different sigma parameterization:
#    a = mu * sigma, b = (1-mu) * sigma
#    logfy[x=0] = log(nu), logfy[0<x<1] = log(1-nu) + dbeta(x,a,b)
# ------------------------------------------------------------------

@dataclass(frozen=True)
class BetaZeroInflatedFamily(FamilyDefinition):
    """Beta zero-inflated distribution (`BEZI`)."""


def _bezi_log_pdf(y, mu, sigma, nu):
    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.clip(mu, eps, 1.0 - eps)
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.clip(nu, eps, 1.0 - eps)
    a = jnp.maximum(mu * sigma, eps)
    b = jnp.maximum((1.0 - mu) * sigma, eps)
    log_cont = jnp.log1p(-nu) + _beta_log_pdf(y, a, b)
    log_at0 = jnp.log(nu)
    return jnp.where(y <= 0.0, log_at0, log_cont)


def BEZI() -> BetaZeroInflatedFamily:
    return build_ad_family(
        family_class=BetaZeroInflatedFamily,
        name="BEZI",
        parameters=("mu", "sigma", "nu"),
        log_pdf_func=_bezi_log_pdf,
        type_="Mixed",
        links={"mu": "logit", "sigma": "log", "nu": "logit"},
        link_functions={"mu": logit_link, "sigma": log_link, "nu": logit_link},
        link_inverses={"mu": logit_inverse, "sigma": log_inverse, "nu": logit_inverse},
        link_derivatives={"mu": logit_derivative, "sigma": log_derivative, "nu": logit_derivative},
    )


# ------------------------------------------------------------------
# 5. BEOI — Beta one-inflated (sigma = precision parameter)
#    R: a = mu*sigma, b = (1-mu)*sigma
#       logfy[x=1] = log(nu), logfy[0<x<1] = log(1-nu) + dbeta(x,a,b)
# ------------------------------------------------------------------

@dataclass(frozen=True)
class BetaOneInflatedFamily(FamilyDefinition):
    """Beta one-inflated distribution (`BEOI`)."""


def _beoi_log_pdf(y, mu, sigma, nu):
    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.clip(mu, eps, 1.0 - eps)
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.clip(nu, eps, 1.0 - eps)
    a = jnp.maximum(mu * sigma, eps)
    b = jnp.maximum((1.0 - mu) * sigma, eps)
    log_cont = jnp.log1p(-nu) + _beta_log_pdf(y, a, b)
    log_at1 = jnp.log(nu)
    return jnp.where(y >= 1.0, log_at1, log_cont)


def BEOI() -> BetaOneInflatedFamily:
    return build_ad_family(
        family_class=BetaOneInflatedFamily,
        name="BEOI",
        parameters=("mu", "sigma", "nu"),
        log_pdf_func=_beoi_log_pdf,
        type_="Mixed",
        links={"mu": "logit", "sigma": "log", "nu": "logit"},
        link_functions={"mu": logit_link, "sigma": log_link, "nu": logit_link},
        link_inverses={"mu": logit_inverse, "sigma": log_inverse, "nu": logit_inverse},
        link_derivatives={"mu": logit_derivative, "sigma": log_derivative, "nu": logit_derivative},
    )
