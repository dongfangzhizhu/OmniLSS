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
    identity_link, identity_inverse, identity_derivative,
    logit_link, logit_inverse, logit_derivative,
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
    """PIG2 log-PMF using the parameterization defined by R's PIG2."""
    from .dpqr_functions import dPIG2
    return dPIG2(y, mu, sigma, log=True)


def PIG2():
    """Poisson Inverse Gaussian Type 2 distribution."""
    from .dpqr_functions import dPIG2, pPIG2, qPIG2, rPIG2
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
        d=dPIG2,
        p=pPIG2,
        q=qPIG2,
        r=rPIG2,
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
    """Logit Normal log-PDF (R LOGIT_NO.R dLOGITNO).

    If Y ~ LogitNormal(mu, sigma) with mu in (0,1), then
    logit(Y) ~ Normal(logit(mu), sigma).  R's dLOGITNO takes mu as the
    probability-scale location and transforms internally.
    """
    eps = jnp.finfo(jnp.float64).eps
    y = jnp.clip(y, eps, 1.0 - eps)
    mu = jnp.clip(mu, eps, 1.0 - eps)
    sigma = jnp.maximum(sigma, eps)
    
    # logit(y) = log(y / (1-y));  logit(mu) = log(mu / (1-mu))
    logit_y = jnp.log(y) - jnp.log1p(-y)
    logit_mu = jnp.log(mu) - jnp.log1p(-mu)
    
    # Jacobian: d(logit(y))/dy = 1 / (y * (1-y))
    log_jacobian = -jnp.log(y) - jnp.log1p(-y)
    
    # Normal log-PDF for logit(y)
    z = (logit_y - logit_mu) / sigma
    log_normal = -0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma) - 0.5 * z * z
    
    return log_normal + log_jacobian


def _logitno_p(q, mu=0.5, sigma=1.0, lower_tail=True, log_p=False):
    """R LOGIT_NO.R pLOGITNO: pnorm(logit(q), logit(mu), sigma)."""
    import numpy as _np
    import jax.numpy as jnp
    from scipy.stats import norm as _sn

    q_arr = _np.atleast_1d(_np.asarray(q, dtype=_np.float64))
    mu_arr = _np.atleast_1d(_np.asarray(mu, dtype=_np.float64))
    sigma_arr = _np.atleast_1d(_np.asarray(sigma, dtype=_np.float64))
    n = max(len(q_arr), len(mu_arr), len(sigma_arr))
    out = []
    for i in range(n):
        qi = float(q_arr[i % len(q_arr)])
        mui = float(mu_arr[i % len(mu_arr)])
        si = float(sigma_arr[i % len(sigma_arr)])
        if qi <= 0.0:
            c = 0.0
        elif qi >= 1.0:
            c = 1.0
        else:
            qz = _np.log(qi / (1.0 - qi))
            c = _sn.cdf(qz, loc=_np.log(mui / (1.0 - mui)), scale=si)
        out.append(float(min(max(c, 0.0), 1.0)))
    cdf = _np.array(out)
    if not lower_tail:
        cdf = 1.0 - cdf
    if log_p:
        cdf = _np.log(cdf)
    return jnp.asarray(cdf, dtype=jnp.float64)


def _logitno_q(p, mu=0.5, sigma=1.0, lower_tail=True, log_p=False):
    """R LOGIT_NO.R qLOGITNO: 1/(1+exp(-qnorm(p, logit(mu), sigma)))."""
    import numpy as _np
    import jax.numpy as jnp
    from scipy.stats import norm as _sn

    p_arr = _np.atleast_1d(_np.asarray(p, dtype=_np.float64))
    if log_p:
        p_arr = _np.exp(p_arr)
    if not lower_tail:
        p_arr = 1.0 - p_arr
    p_arr = _np.clip(p_arr, 1e-15, 1.0 - 1e-15)
    mu_arr = _np.atleast_1d(_np.asarray(mu, dtype=_np.float64))
    sigma_arr = _np.atleast_1d(_np.asarray(sigma, dtype=_np.float64))
    n = max(len(p_arr), len(mu_arr), len(sigma_arr))
    out = []
    for i in range(n):
        pi = float(p_arr[i % len(p_arr)])
        mui = float(mu_arr[i % len(mu_arr)])
        si = float(sigma_arr[i % len(sigma_arr)])
        qz = _sn.ppf(pi, loc=_np.log(mui / (1.0 - mui)), scale=si)
        out.append(float(1.0 / (1.0 + _np.exp(-qz))))
    res = _np.array(out)
    return jnp.asarray(res, dtype=jnp.float64)


def _logitno_r(key, n, mu=0.5, sigma=1.0):
    import jax.numpy as _jnp
    import jax.random as _jr

    u = _jr.uniform(key, shape=(n,), dtype=_jnp.float64)
    return _logitno_q(u, mu, sigma)


def LOGITNO():
    """Logit Normal distribution."""
    return build_ad_family(
        family_class=LOGITNOFamily,
        name="LOGITNO",
        parameters=("mu", "sigma"),
        log_pdf_func=_logitno_log_pdf,
        type_="Continuous",
        p=_logitno_p,
        q=_logitno_q,
        r=_logitno_r,
        links={"mu": "logit", "sigma": "log"},
        link_functions={"mu": logit_link, "sigma": log_link},
        link_inverses={"mu": logit_inverse, "sigma": log_inverse},
        link_derivatives={"mu": logit_derivative, "sigma": log_derivative},
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
    """Compatibility spelling of the GAMLSS DEL log PMF."""
    from .distributions_b6 import _del_log_pdf
    return _del_log_pdf(y, mu, sigma, nu)


def DELAPORT():
    """DEL compatibility spelling: mu > 0, sigma > 0, 0 < nu < 1.

    Replaces the former unnormalized approximation. Uses DEL's numerical
    limits and Poisson approximation; the old unrestricted nu is unsupported.
    """
    from .distributions_b6 import _del_p, _del_q, _del_r
    return build_ad_family(
        family_class=DELAPORTFamily,
        name="DELAPORT",
        parameters=("mu", "sigma", "nu"),
        log_pdf_func=_delaport_log_pdf,
        type_="Discrete",
        p=_del_p,
        q=_del_q,
        r=_del_r,
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
