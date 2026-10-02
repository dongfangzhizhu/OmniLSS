from __future__ import annotations

# Enable float64 precision for numerical accuracy

"""Batch 12: Alternative parameterizations and type variants."""
from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
from jax.scipy.special import gammaln, betaln, betainc

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


def _beo_d(x, mu=0.5, sigma=0.2, log=False):
    """BEo density. Mirrors R `dBEo` (Beta(shape1=mu, shape2=sigma))."""
    eps = jnp.finfo(jnp.float64).eps
    alpha = jnp.maximum(mu, eps)
    beta = jnp.maximum(sigma, eps)
    # log f(y) = (alpha-1)ln y + (beta-1)ln(1-y) - B(alpha, beta)
    log_fy = (alpha - 1.0) * jnp.log(jnp.clip(x, eps, 1.0 - eps)) \
        + (beta - 1.0) * jnp.log1p(-jnp.clip(x, eps, 1.0 - eps)) - betaln(alpha, beta)
    log_fy = jnp.where((x <= 0) | (x >= 1), -jnp.inf, log_fy)
    return log_fy if log else jnp.exp(log_fy)


def _beo_p(q, mu=0.5, sigma=0.2, lower_tail=True, log_p=False):
    """BEo CDF. Mirrors R `pBEo` via regularized incomplete beta function."""
    eps = jnp.finfo(jnp.float64).eps
    alpha = jnp.maximum(mu, eps)
    beta = jnp.maximum(sigma, eps)
    cdf = betainc(alpha, beta, jnp.clip(q, 0.0, 1.0))
    cdf = jnp.where(q < 0, 0.0, cdf)
    cdf = jnp.where(q >= 1, 1.0, cdf)
    if not lower_tail:
        cdf = 1.0 - cdf
    if log_p:
        cdf = jnp.log(cdf)
    return cdf


def _beo_q(p, mu=0.5, sigma=0.2, lower_tail=True, log_p=False):
    """BEo quantile. Inverts the Beta CDF (scipy-based)."""
    p_arr = np.asarray(p, dtype=np.float64)
    mu_arr = np.asarray(mu, dtype=np.float64)
    sigma_arr = np.asarray(sigma, dtype=np.float64)
    if log_p:
        p_arr = np.exp(p_arr)
    if not lower_tail:
        p_arr = 1.0 - p_arr
    from scipy.stats import beta as _scipy_beta
    q = np.asarray(_scipy_beta.ppf(np.clip(p_arr, 1e-16, 1.0 - 1e-16),
                                   mu_arr, sigma_arr), dtype=np.float64)
    q = np.where(p_arr <= 0, 0.0, q)
    q = np.where(p_arr >= 1, 1.0, q)
    return jnp.asarray(q, dtype=jnp.float64)


def _beo_r(key, n, mu=0.5, sigma=0.2):
    """BEo random generation via inverse-CDF sampling (matches R `rBEo`)."""
    mu_arr = np.asarray(mu, dtype=np.float64)
    sigma_arr = np.asarray(sigma, dtype=np.float64)
    u = np.random.default_rng(int(key[0])).uniform(size=n)
    q = np.asarray(_scipy_beta.ppf(u, mu_arr, sigma_arr), dtype=np.float64)
    return jnp.asarray(q, dtype=jnp.float64)


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
        d=_beo_d,
        p=_beo_p,
        q=_beo_q,
        r=_beo_r,
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


def _geomo_d(x, mu=0.5, log=False):
    """GEOMo PMF. Mirrors R `dGEOMo` (Geometric(prob=mu))."""
    eps = jnp.finfo(jnp.float64).eps
    p = jnp.clip(mu, eps, 1.0 - eps)
    log_fy = jnp.log(p) + x * jnp.log1p(-p)
    log_fy = jnp.where(x < 0, -jnp.inf, log_fy)
    return log_fy if log else jnp.exp(log_fy)


def _geomo_p(q, mu=0.5, lower_tail=True, log_p=False):
    """GEOMo CDF. Mirrors R `pGEOMo` (Geometric(prob=mu))."""
    eps = jnp.finfo(jnp.float64).eps
    p = jnp.clip(mu, eps, 1.0 - eps)
    # P(Y <= q) = 1 - (1-p)^(floor(q)+1)
    cdf = 1.0 - jnp.power(1.0 - p, jnp.floor(q) + 1.0)
    cdf = jnp.where(q < 0, 0.0, cdf)
    if not lower_tail:
        cdf = 1.0 - cdf
    if log_p:
        cdf = jnp.log(cdf)
    return cdf


def _geomo_q(p, mu=0.5, lower_tail=True, log_p=False):
    """GEOMo quantile. Mirrors R `qGEOMo` (Geometric(prob=mu), 0-based failures)."""
    p_arr = np.asarray(p, dtype=np.float64)
    mu_arr = np.asarray(mu, dtype=np.float64)
    if log_p:
        p_arr = np.exp(p_arr)
    if not lower_tail:
        p_arr = 1.0 - p_arr
    prob = np.clip(mu_arr, 1e-16, 1.0 - 1e-16)
    # Inverse of P(X<=q) = 1 - (1-prob)^(q+1): ratio = ln(1-p)/ln(1-prob).
    ratio = np.log1p(-np.clip(p_arr, 0.0, 1.0 - 1e-16)) / np.log1p(-prob)
    q = np.floor(ratio)
    # Exact CDF boundary correction to match R's `qgeom` (smallest q with CDF>=p).
    cdf = 1.0 - np.power(1.0 - prob, np.maximum(q, 0.0) + 1.0)
    q = np.where(cdf + 1e-12 < p_arr, q + 1.0, q)
    cdf_prev = 1.0 - np.power(1.0 - prob, np.maximum(q - 1.0, 0.0) + 1.0)
    q = np.where((cdf_prev >= p_arr - 1e-12) & (q > 0.0), q - 1.0, q)
    q = np.where(p_arr <= 0, 0.0, q)
    q = np.where(p_arr >= 1, np.inf, q)
    return jnp.asarray(q, dtype=jnp.float64)


def _geomo_r(key, n, mu=0.5):
    """GEOMo random generation. Mirrors R `rGEOMo` via inverse-CDF."""
    mu_arr = np.asarray(mu, dtype=np.float64)
    u = np.random.default_rng(int(key[0])).uniform(size=n)
    q = np.asarray(_geomo_q(u, mu_arr), dtype=np.float64)
    return jnp.asarray(q, dtype=jnp.float64)


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
        d=_geomo_d,
        p=_geomo_p,
        q=_geomo_q,
        r=_geomo_r,
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

    R ``dWEI2`` is ``dweibull(x, scale = mu^(-1/sigma), shape = sigma)``; the
    earlier Python version used ``shape = 1/sigma, scale = mu * sigma``, which is
    a different distribution and silently disagreed with R.
    """
    from .dpqr_batch_e import dWEI2 as _dWEI2

    return _dWEI2(y, mu, sigma, log=True)


def WEI2():
    """Weibull Type 2 distribution."""
    from .dpqr_batch_e import dWEI2, pWEI2, qWEI2, rWEI2

    return build_ad_family(
        family_class=WEI2Family,
        name="WEI2",
        parameters=("mu", "sigma"),
        log_pdf_func=_wei2_log_pdf,
        type_="Continuous",
        d=dWEI2,
        p=pWEI2,
        q=qWEI2,
        r=rWEI2,
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
    from omnilss.dpqr_functions import dWEI3 as _dWEI3
    from omnilss.dpqr_functions import pWEI3 as _pWEI3
    from omnilss.dpqr_functions import qWEI3 as _qWEI3
    from omnilss.dpqr_functions import rWEI3 as _rWEI3

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
        d=_dWEI3,
        p=_pWEI3,
        q=_qWEI3,
        r=_rWEI3,
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

    Original parameterization (R `PARETO2o.R`):
        mu   = scale (mu > 0)
        sigma = shape (sigma > 0)
    log f(y|mu,sigma) = log(sigma) + sigma*log(mu) - (sigma+1)*log(y + mu)
    """
    eps = jnp.finfo(jnp.float64).eps
    y = jnp.maximum(y, eps)
    scale = jnp.maximum(mu, eps)
    shape = jnp.maximum(sigma, eps)

    log_pdf = jnp.log(shape) + shape * jnp.log(scale) \
        - (shape + 1.0) * jnp.log(y + scale)
    log_pdf = jnp.where(y <= 0, -jnp.inf, log_pdf)

    return log_pdf


def _pareto2o_d(x, mu=1.0, sigma=0.5, log=False):
    """PARETO2o density. Mirrors R `dPARETO2o` (shape=sigma, scale=mu)."""
    eps = jnp.finfo(jnp.float64).eps
    scale = jnp.maximum(mu, eps)
    shape = jnp.maximum(sigma, eps)
    log_fy = jnp.log(shape) + shape * jnp.log(scale) \
        - (shape + 1.0) * jnp.log(jnp.maximum(x, eps) + scale)
    log_fy = jnp.where(x <= 0, -jnp.inf, log_fy)
    return log_fy if log else jnp.exp(log_fy)


def _pareto2o_p(q, mu=1.0, sigma=0.5, lower_tail=True, log_p=False):
    """PARETO2o CDF. Mirrors R `pPARETO2o`: F(q) = 1 - (mu/(mu+q))^sigma."""
    eps = jnp.finfo(jnp.float64).eps
    scale = jnp.maximum(mu, eps)
    shape = jnp.maximum(sigma, eps)
    cdf = 1.0 - jnp.power(scale / (jnp.maximum(q, 0.0) + scale), shape)
    cdf = jnp.where(q <= 0, 0.0, cdf)
    if not lower_tail:
        cdf = 1.0 - cdf
    if log_p:
        cdf = jnp.log(cdf)
    return cdf


def _pareto2o_q(p, mu=1.0, sigma=0.5, lower_tail=True, log_p=False):
    """PARETO2o quantile. Mirrors R `qPARETO2o`: Q(p) = mu*((1-p)^(-1/sigma) - 1)."""
    eps = jnp.finfo(jnp.float64).eps
    p_val = jnp.asarray(p, dtype=jnp.float64)
    if log_p:
        p_val = jnp.exp(p_val)
    if not lower_tail:
        p_val = 1.0 - p_val
    scale = jnp.maximum(mu, eps)
    shape = jnp.maximum(sigma, eps)
    q = scale * (jnp.power(1.0 - jnp.clip(p_val, 0.0, 1.0 - eps), -1.0 / shape) - 1.0)
    q = jnp.where(p_val <= 0, 0.0, q)
    q = jnp.where(p_val >= 1, jnp.inf, q)
    return q


def _pareto2o_r(key, n, mu=1.0, sigma=0.5):
    """PARETO2o random generation. Mirrors R `rPARETO2o` via inverse-CDF."""
    mu_arr = np.asarray(mu, dtype=np.float64)
    sigma_arr = np.asarray(sigma, dtype=np.float64)
    u = np.random.default_rng(int(key[0])).uniform(size=n)
    eps = np.finfo(np.float64).eps
    scale = np.maximum(mu_arr, eps)
    shape = np.maximum(sigma_arr, eps)
    q = scale * (np.power(1.0 - np.clip(u, 0.0, 1.0 - eps), -1.0 / shape) - 1.0)
    return jnp.asarray(q, dtype=jnp.float64)


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
        d=_pareto2o_d,
        p=_pareto2o_p,
        q=_pareto2o_q,
        r=_pareto2o_r,
    )


# ============================================================================
# NOF - Normal Family (alternative)
# ============================================================================

@dataclass(frozen=True)
class NOFFamily(FamilyDefinition):
    """Normal Family distribution."""
    pass


def _nof_log_pdf(y, mu, sigma, nu):
    """Normal Family log-PDF.

    Mirrors R ``dNOF``: ``sigma1 = sigma*|mu|^(nu/2)`` and ``y ~ N(mu, sigma1)``.
    With R's default ``nu = 0`` this collapses to the previous two-parameter
    ``N(mu, sigma)`` form, so the extension is backwards compatible at the
    default.
    """
    from .dpqr_batch_e import dNOF

    return dNOF(y, mu, sigma, nu, log=True)


def NOF():
    """Normal Family distribution (R ``NOF``: variance function ``sigma^2*|mu|^nu``)."""
    from .dpqr_batch_e import dNOF, pNOF, qNOF, rNOF

    return build_ad_family(
        family_class=NOFFamily,
        name="NOF",
        parameters=("mu", "sigma", "nu"),
        log_pdf_func=_nof_log_pdf,
        type_="Continuous",
        links={"mu": "identity", "sigma": "log", "nu": "identity"},
        link_functions={"mu": identity_link, "sigma": log_link, "nu": identity_link},
        link_inverses={"mu": identity_inverse, "sigma": log_inverse, "nu": identity_inverse},
        link_derivatives={
            "mu": identity_derivative,
            "sigma": log_derivative,
            "nu": identity_derivative,
        },
        d=dNOF,
        p=pNOF,
        q=qNOF,
        r=rNOF,
    )


# ============================================================================
# GAF - Gamma Family (alternative)
# ============================================================================

@dataclass(frozen=True)
class GAFFamily(FamilyDefinition):
    """Gamma Family distribution."""
    pass


def _gaf_log_pdf(y, mu, sigma, nu):
    """Gamma Family log-PDF.

    Mirrors R ``dGAF``: ``sigma1 = sigma*mu^(nu/2-1)`` and
    ``y ~ Gamma(shape = 1/sigma1^2, scale = mu*sigma1^2)``.  With R's default
    ``nu = 2`` this reduces to the previous two-parameter form.
    """
    from .dpqr_batch_e import dGAF

    return dGAF(y, mu, sigma, nu, log=True)


def GAF():
    """Gamma Family distribution (R ``GAF``: variance function ``sigma^2*mu^nu``)."""
    from .dpqr_batch_e import dGAF, pGAF, qGAF, rGAF

    return build_ad_family(
        family_class=GAFFamily,
        name="GAF",
        parameters=("mu", "sigma", "nu"),
        log_pdf_func=_gaf_log_pdf,
        type_="Continuous",
        links={"mu": "log", "sigma": "log", "nu": "identity"},
        link_functions={"mu": log_link, "sigma": log_link, "nu": identity_link},
        link_inverses={"mu": log_inverse, "sigma": log_inverse, "nu": identity_inverse},
        link_derivatives={
            "mu": log_derivative,
            "sigma": log_derivative,
            "nu": identity_derivative,
        },
        d=dGAF,
        p=pGAF,
        q=qGAF,
        r=rGAF,
    )


# ============================================================================
# RGE - Reverse Gumbel Extended
# ============================================================================

@dataclass(frozen=True)
class RGEFamily(FamilyDefinition):
    """Reverse Gumbel Extended distribution family."""
    pass


def _rge_log_pdf(y, mu, sigma, nu):
    """Reverse Generalized Extreme log-PDF.

    Mirrors R ``dRGE``: with ``z = nu*(y-mu)/sigma`` the log density is
    ``-log(sigma) + (1/nu - 1)*log(1+z) - (1+z)^(1/nu)`` on ``1+z > 0``, with the
    reverse-Gumbel limit used when ``|nu|`` is below R's 1e-3 switch.  The
    earlier Python version was an unrelated "reverse Gumbel extended" formula
    that disagreed with R everywhere.
    """
    from .dpqr_batch_e import dRGE as _dRGE

    return _dRGE(y, mu, sigma, nu, log=True)


def RGE():
    """Reverse Generalized Extreme distribution."""
    from .dpqr_batch_e import dRGE, pRGE, qRGE, rRGE

    return build_ad_family(
        family_class=RGEFamily,
        name="RGE",
        parameters=("mu", "sigma", "nu"),
        log_pdf_func=_rge_log_pdf,
        type_="Continuous",
        d=dRGE,
        p=pRGE,
        q=qRGE,
        r=rRGE,
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
