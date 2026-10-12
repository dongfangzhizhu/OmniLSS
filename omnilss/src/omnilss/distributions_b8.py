"""GAMLSS distributions: Batch 8 — Additional continuous distributions.

These are commonly used continuous distributions with various shapes and properties.

Distributions:
- GG       (Generalized Gamma)
- GB2      (Generalized Beta Type 2)
- PARETO   (Pareto distribution)
- NET      (Normal-Exponential-t)
- LNO      (Log Normal with location parameter)

R source: gamlss.dist package
"""

from __future__ import annotations

# Enable float64 precision for numerical accuracy

from dataclasses import dataclass
import math

import jax.numpy as jnp
from jax.scipy.special import gammaln, betaln
from jax.scipy.stats import norm

from .ad import build_ad_family
from .families import FamilyDefinition
from .links import (
    identity_derivative,
    identity_inverse,
    identity_link,
    log_derivative,
    log_inverse,
    log_link,
)

# ------------------------------------------------------------------
# 1. GG — Generalized Gamma
#    R: dGG(x, mu, sigma, nu)
#    A flexible three-parameter distribution that includes Gamma, Weibull, and others as special cases
# ------------------------------------------------------------------

@dataclass(frozen=True)
class GeneralizedGammaFamily(FamilyDefinition):
    """Generalized Gamma distribution (`GG`)."""


def _gg_log_pdf(y, mu, sigma, nu):
    """Generalized Gamma log-pdf.
    
    The Generalized Gamma distribution is a flexible three-parameter family
    that includes Gamma, Weibull, and Exponential as special cases.
    
    Parameters:
    - y: observed value (> 0)
    - mu: location parameter (> 0)
    - sigma: scale parameter (> 0)
    - nu: shape parameter (can be positive or negative)
    
    Parameterization (from R gamlss.dist):
    z = (y/mu)^nu
    For |nu| > 1e-6:
        log p(y) = dGA(z, mu=1, sigma=sigma*|nu|, log=TRUE) + log(|nu|*z/y)
    For |nu| <= 1e-6 (nu ~ 0, log-normal limit):
        log p(y) = -log(y) - 0.5*log(2π) - log(sigma) - 0.5*(log(y)-log(mu))^2/sigma^2
    """
    eps = jnp.finfo(jnp.float64).eps
    y = jnp.maximum(y, eps)
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    
    # Compute z = (y/mu)^nu
    z = jnp.power(y / mu, nu)
    
    # For |nu| > 1e-6, use Gamma distribution on z
    # theta = 1/(sigma^2 * nu^2)
    theta = 1.0 / (sigma * sigma * nu * nu)
    
    # Gamma log-pdf: log p(z) = (theta-1)*log(z) - z*theta - log(Gamma(theta)) - theta*log(theta)
    # But we need dGA(z, mu=1, sigma=sigma*|nu|)
    # For GA with mu=1: shape = 1/sigma^2, scale = sigma^2
    # So for sigma' = sigma*|nu|: shape = 1/(sigma*|nu|)^2 = theta
    sigma_prime = sigma * jnp.abs(nu)
    shape = 1.0 / (sigma_prime * sigma_prime)  # This equals theta
    
    # Gamma(shape, scale=1/shape) log-pdf
    log_pdf_z = (shape - 1.0) * jnp.log(z) - z * shape - gammaln(shape) + shape * jnp.log(shape)
    
    # Jacobian: |nu| * z / y
    log_jacobian = jnp.log(jnp.abs(nu)) + jnp.log(z) - jnp.log(y)
    
    # Total log-pdf
    log_pdf = log_pdf_z + log_jacobian
    
    # For nu ~ 0, use log-normal limit
    log_pdf_lognormal = (
        -jnp.log(y) - 0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma) -
        0.5 * jnp.square((jnp.log(y) - jnp.log(mu)) / sigma)
    )
    
    # Use log-normal when |nu| < 1e-6
    log_pdf = jnp.where(jnp.abs(nu) > 1e-6, log_pdf, log_pdf_lognormal)
    
    return log_pdf


def GG() -> GeneralizedGammaFamily:
    from .dpqr_functions import pGG, qGG, rGG
    return build_ad_family(
        family_class=GeneralizedGammaFamily,
        name="GG",
        parameters=("mu", "sigma", "nu"),
        log_pdf_func=_gg_log_pdf,
        type_="Continuous",
        links={"mu": "log", "sigma": "log", "nu": "log"},
        link_functions={"mu": log_link, "sigma": log_link, "nu": log_link},
        link_inverses={"mu": log_inverse, "sigma": log_inverse, "nu": log_inverse},
        link_derivatives={"mu": log_derivative, "sigma": log_derivative, "nu": log_derivative},
        p=pGG,
        q=qGG,
        r=rGG,
    )


# ------------------------------------------------------------------
# 2. GB2 — Generalized Beta Type 2
#    R: dGB2(x, mu, sigma, nu, tau)
#    A four-parameter distribution that includes many distributions as special cases
# ------------------------------------------------------------------

@dataclass(frozen=True)
class GeneralizedBeta2Family(FamilyDefinition):
    """Generalized Beta Type 2 distribution (`GB2`)."""


def _gb2_log_pdf(y, mu, sigma, nu, tau):
    """Use the public JAX log-density for both fitting and prediction."""
    from .dpqr_functions import dGB2

    return dGB2(y, mu, sigma, nu, tau, log=True)


def GB2() -> GeneralizedBeta2Family:
    """Generalized Beta Type 2 distribution.
    
    All parameters use log link to ensure positivity.
    """
    from .dpqr_functions import dGB2, pGB2, qGB2, rGB2
    return build_ad_family(
        family_class=GeneralizedBeta2Family,
        name="GB2",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_gb2_log_pdf,
        type_="Continuous",
        d=dGB2,
        p=pGB2,
        q=qGB2,
        r=rGB2,
        links={"mu": "log", "sigma": "log", "nu": "log", "tau": "log"},
        link_functions={"mu": log_link, "sigma": log_link, "nu": log_link, "tau": log_link},
        link_inverses={"mu": log_inverse, "sigma": log_inverse, "nu": log_inverse, "tau": log_inverse},
        link_derivatives={"mu": log_derivative, "sigma": log_derivative, "nu": log_derivative, "tau": log_derivative},
    )


# ------------------------------------------------------------------
# 3. PARETO — Pareto distribution
#    R: dPARETO(x, mu)
#    A simple one-parameter power-law distribution
# ------------------------------------------------------------------

@dataclass(frozen=True)
class ParetoFamily(FamilyDefinition):
    """Pareto distribution (`PARETO`)."""


def _pareto_log_pdf(y, mu):
    """Pareto distribution log-pdf.
    
    The Pareto distribution is a power-law distribution often used for
    modeling wealth, city sizes, and other phenomena.
    
    Parameters:
    - y: observed value (>= 1)
    - mu: shape parameter (> 0)
    
    log p(y) = log(mu) - (mu + 1)*log(y)
    """
    eps = jnp.finfo(jnp.float64).eps
    y = jnp.asarray(y, dtype=jnp.float64)
    mu = jnp.maximum(mu, eps)
    
    log_pdf = jnp.log(mu) - (mu + 1.0) * jnp.log(y)
    return jnp.where(y <= 1.0, -jnp.inf, log_pdf)


def PARETO() -> ParetoFamily:
    from .dpqr_batch_e import dPARETO, pPARETO, qPARETO, rPARETO

    return build_ad_family(
        family_class=ParetoFamily,
        name="PARETO",
        parameters=("mu",),
        log_pdf_func=_pareto_log_pdf,
        type_="Continuous",
        d=dPARETO,
        p=pPARETO,
        q=qPARETO,
        r=rPARETO,
        links={"mu": "log"},
        link_functions={"mu": log_link},
        link_inverses={"mu": log_inverse},
        link_derivatives={"mu": log_derivative},
    )


# ------------------------------------------------------------------
# 4. NET — Normal-Exponential-t distribution
#    R: dNET(x, mu, sigma, nu, tau)
#    A flexible distribution with piecewise density (Normal-Exponential-t)
#    Note: In R gamlss, nu and tau are typically fixed (not estimated)
# ------------------------------------------------------------------

@dataclass(frozen=True)
class NormalExponentialTFamily(FamilyDefinition):
    """Normal-Exponential-t distribution (`NET`)."""


def _net_log_pdf(y, mu, sigma, nu, tau):
    """Normal-Exponential-t log-pdf.
    
    The NET distribution uses a piecewise density function:
    - Normal in the center (|z| <= k1)
    - Exponential in the middle (k1 < |z| <= k2)
    - Power tail beyond (|z| > k2)
    
    Parameters:
    - y: observed value
    - mu: location parameter
    - sigma: scale parameter (> 0)
    - nu: threshold parameter k1 (> 0)
    - tau: threshold parameter k2 (> 0), must be >= nu
    
    From R gamlss.dist source:
    k1 = nu, k2 = tau
    c1 = (1 - 2*pnorm(-k1)) * sqrt(2*pi)
    c2 = (2/k1) * exp(-(k1^2)/2)
    c3 = 2*exp(-k1*k2 + (k1^2)/2) / ((k1*k2 - 1)*k1)
    ct = 1/(c1 + c2 + c3)  # Normalizing constant
    
    tc = (y - mu)/sigma
    d1 = (|tc| <= k1) * (-(tc^2)/2)
    d2 = (k1 < |tc| <= k2) * (-k1*|tc| + (k1^2)/2)
    d3 = (|tc| > k2) * (-k1*k2*log(|tc|/k2) - k1*k2 + (k1^2)/2)
    
    log p(y) = log(ct) - log(sigma) + d1 + d2 + d3
    """
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.maximum(nu, eps)
    tau = jnp.maximum(tau, nu)  # Ensure tau >= nu
    
    k1 = nu
    k2 = tau
    
    # Normalizing constant components
    from jax.scipy.stats import norm as jax_norm
    c1 = (1.0 - 2.0 * jax_norm.cdf(-k1)) * jnp.sqrt(2.0 * jnp.pi)
    c2 = (2.0 / k1) * jnp.exp(-(k1 * k1) / 2.0)
    c3 = 2.0 * jnp.exp(-k1 * k2 + (k1 * k1) / 2.0) / ((k1 * k2 - 1.0) * k1)
    ct = 1.0 / (c1 + c2 + c3)
    
    # Standardized residual
    tc = (y - mu) / sigma
    abs_tc = jnp.abs(tc)
    
    # Piecewise log-density (without normalizing constant and sigma term)
    # Region 1: |tc| <= k1 (Normal)
    d1 = jnp.where(abs_tc <= k1, -(tc * tc) / 2.0, 0.0)
    
    # Region 2: k1 < |tc| <= k2 (Exponential)
    d2 = jnp.where((abs_tc > k1) & (abs_tc <= k2), -k1 * abs_tc + (k1 * k1) / 2.0, 0.0)
    
    # Region 3: |tc| > k2 (Power tail)
    # Handle log(0) case when tc = 0
    log_term = jnp.where(
        abs_tc > k2,
        -k1 * k2 * jnp.log(abs_tc / k2) - k1 * k2 + (k1 * k1) / 2.0,
        0.0
    )
    d3 = log_term
    
    # Total log-pdf
    log_pdf = jnp.log(ct) - jnp.log(sigma) + d1 + d2 + d3
    
    return log_pdf


def _net_p(q, mu=0.0, sigma=1.0, nu=1.5, tau=2.0, lower_tail=True, log_p=False):
    """R NET.R pNET: piecewise analytic CDF of the Normal-Exponential-t."""
    import numpy as _np
    import jax.numpy as jnp
    from scipy.stats import norm as _sn

    q_arr = _np.atleast_1d(_np.asarray(q, dtype=_np.float64))
    mu_arr = _np.atleast_1d(_np.asarray(mu, dtype=_np.float64))
    sigma_arr = _np.atleast_1d(_np.asarray(sigma, dtype=_np.float64))
    nu_arr = _np.atleast_1d(_np.asarray(nu, dtype=_np.float64))
    tau_arr = _np.atleast_1d(_np.asarray(tau, dtype=_np.float64))
    n = max(len(q_arr), len(mu_arr), len(sigma_arr), len(nu_arr), len(tau_arr))
    out = []
    for i in range(n):
        qi = float(q_arr[i % len(q_arr)])
        mui = float(mu_arr[i % len(mu_arr)])
        si = float(sigma_arr[i % len(sigma_arr)])
        k1 = float(nu_arr[i % len(nu_arr)])
        k2 = float(tau_arr[i % len(tau_arr)])
        if si <= 0 or k1 <= 0 or k2 <= 0 or k2 < k1:
            out.append(_np.nan)
            continue
        c1 = (1.0 - 2.0 * _sn.cdf(-k1)) * _np.sqrt(2.0 * _np.pi)
        c2 = (2.0 / k1) * _np.exp(-(k1 * k1) / 2.0)
        c3 = 2.0 * _np.exp(-k1 * k2 + (k1 * k1) / 2.0) / ((k1 * k2 - 1.0) * k1)
        ct = 1.0 / (c1 + c2 + c3)
        tc = (qi - mui) / si
        # Fk2: cdf up to -k2
        Fk2 = (ct * (k2 ** (k1 * k2)) / (k1 * k2 - 1.0)) * _np.exp(-k1 * k2 + (k1 * k1) / 2.0) * (
            _np.abs(tc) ** (-k1 * k2 + 1.0) if tc != 0.0 else 1.0
        )
        cdf1 = (ct / (k1 * (k1 * k2 - 1.0))) * _np.exp(-k1 * k2 + (k1 * k1) / 2.0)
        Fk1 = cdf1 + (ct / k1) * _np.exp(-k1 * _np.abs(tc) + (k1 * k1) / 2.0)
        cdf2 = cdf1 + (ct / k1) * _np.exp(-(k1 * k1) / 2.0)
        F0 = cdf2 + ct * _np.sqrt(2.0 * _np.pi) * (_sn.cdf(-_np.abs(tc)) - _sn.cdf(-k1))
        if tc <= -k2:
            c = Fk2
        elif tc <= -k1:
            c = Fk1
        elif tc <= 0:
            c = F0
        elif tc <= k1:
            c = 1.0 - F0
        elif tc <= k2:
            c = 1.0 - Fk1
        else:
            c = 1.0 - Fk2
        out.append(float(min(max(c, 0.0), 1.0)))
    cdf = _np.array(out)
    if not lower_tail:
        cdf = 1.0 - cdf
    if log_p:
        cdf = _np.log(cdf)
    return jnp.asarray(cdf, dtype=jnp.float64)


def _net_q(p, mu=0.0, sigma=1.0, nu=1.5, tau=2.0, lower_tail=True, log_p=False):
    """R NET.R qNET: piecewise analytic quantile of the Normal-Exponential-t."""
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
    nu_arr = _np.atleast_1d(_np.asarray(nu, dtype=_np.float64))
    tau_arr = _np.atleast_1d(_np.asarray(tau, dtype=_np.float64))
    n = max(len(p_arr), len(mu_arr), len(sigma_arr), len(nu_arr), len(tau_arr))
    out = []
    for i in range(n):
        pi = float(p_arr[i % len(p_arr)])
        mui = float(mu_arr[i % len(mu_arr)])
        si = float(sigma_arr[i % len(sigma_arr)])
        nu1 = float(nu_arr[i % len(nu_arr)])
        tau1 = float(tau_arr[i % len(tau_arr)])
        if si <= 0 or nu1 <= 0 or tau1 <= 0 or tau1 < nu1:
            out.append(_np.nan)
            continue
        p2 = pi if pi <= 0.5 else 1.0 - pi
        c1 = (1.0 - 2.0 * _sn.cdf(-nu1)) * _np.sqrt(2.0 * _np.pi)
        c2 = (2.0 / nu1) * _np.exp(-(nu1 * nu1) / 2.0)
        c3 = 2.0 * _np.exp(-nu1 * tau1 + (nu1 * nu1) / 2.0) / ((nu1 * tau1 - 1.0) * nu1)
        ct = 1.0 / (c1 + c2 + c3)
        b = (ct / (nu1 * (nu1 * tau1 - 1.0))) * _np.exp(-nu1 * tau1 + (nu1 * nu1) / 2.0)
        p1 = p2
        z1 = -(b * nu1 * (tau1 ** (nu1 * tau1)) / p1) ** (1.0 / (nu1 * tau1 - 1.0))
        p1 = p1 if p1 >= (nu1 * tau1 * b) else 0.5
        z2 = -(nu1 / 2.0) + (1.0 / nu1) * _np.log((nu1 / ct) * (p1 - b))
        z3 = _sn.ppf(_sn.cdf(-nu1) + (1.0 / _np.sqrt(2.0 * _np.pi)) * ((p1 - b) / ct - (1.0 / nu1) * _np.exp(-(nu1 * nu1) / 2.0)))
        if p2 <= (nu1 * tau1 * b):
            q1 = z1
        elif p1 <= (b + (ct / nu1) * _np.exp(-(nu1 * nu1) / 2.0)):
            q1 = z2
        else:
            q1 = z3
        qq = q1 if pi <= 0.5 else -q1
        out.append(float(mui + si * qq))
    res = _np.array(out)
    return jnp.asarray(res, dtype=jnp.float64)


def _net_r(key, n, mu=0.0, sigma=1.0, nu=1.5, tau=2.0):
    import jax.numpy as _jnp
    import jax.random as _jr

    u = _jr.uniform(key, shape=(n,), dtype=_jnp.float64)
    return _net_q(u, mu, sigma, nu, tau)


def NET() -> NormalExponentialTFamily:
    # D2.39: live R's NET.R marks `parameters = list(mu=TRUE, sigma=TRUE,
    # nu=FALSE, tau=FALSE)` -- nu and tau are *non-estimable* shape constants
    # that R fixes at nu.initial = 1.5 / tau.initial = 2.  R's gamlss returns
    # nu.coefficients = NULL for NET even when an explicit nu.formula is
    # passed (verified against live R: fitted nu/tau stay 1.5/2).  The old
    # links here were also the D2.34 "same name, different meaning" trap:
    # R's nu.link / tau.link are *identity*, our log links started the IRLS
    # at eta=log(1.5)=0.405 / log(2)=0.693 and the unbounded nu/tau updates
    # walked the fit 4.8e-2 rel away from R's deviance with nu violating the
    # tau >= nu domain (live R's dNET stops on it).
    return build_ad_family(
        family_class=NormalExponentialTFamily,
        name="NET",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_net_log_pdf,
        type_="Continuous",
        p=_net_p,
        q=_net_q,
        r=_net_r,
        fixed_parameters=("nu", "tau"),
        fixed_parameter_defaults={"nu": 1.5, "tau": 2.0},
        links={"mu": "identity", "sigma": "log", "nu": "identity", "tau": "identity"},
        link_functions={"mu": identity_link, "sigma": log_link, "nu": identity_link, "tau": identity_link},
        link_inverses={"mu": identity_inverse, "sigma": log_inverse, "nu": identity_inverse, "tau": identity_inverse},
        link_derivatives={"mu": identity_derivative, "sigma": log_derivative, "nu": identity_derivative, "tau": identity_derivative},
    )


# ------------------------------------------------------------------
# 5. LNO — Log Normal with optional Box-Cox power nu
#    R: dLNO(x, mu, sigma, nu)
#    In fitting, nu is fixed at 0 by default; density helpers can still evaluate
#    arbitrary nu values for R consistency checks.
# ------------------------------------------------------------------

@dataclass(frozen=True)
class LogNormalLocationFamily(FamilyDefinition):
    """Log Normal distribution (`LNO`) - alias for LOGNO."""


def _lno_log_pdf(y, mu, sigma, nu=0.0):
    """Log Normal with optional Box-Cox power parameter."""
    eps = jnp.finfo(jnp.float64).eps
    y = jnp.maximum(y, eps)
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.asarray(nu, dtype=jnp.float64)

    z = jnp.where(jnp.abs(nu) > 1e-12, (jnp.power(y, nu) - 1.0) / nu, jnp.log(y))
    log_pdf = -0.5 * jnp.log(2.0 * math.pi) - jnp.log(sigma) - 0.5 * jnp.square((z - mu) / sigma) + (nu - 1.0) * jnp.log(y)
    return log_pdf


def _lno_p(q, mu=1.0, sigma=0.1, nu=0.0, lower_tail=True, log_p=False):
    """R LNO.R pLNO: Box-Cox transform then pnorm."""
    import numpy as _np
    import jax.numpy as jnp
    from scipy.stats import norm as _sn

    q_arr = _np.atleast_1d(_np.asarray(q, dtype=_np.float64))
    mu_arr = _np.atleast_1d(_np.asarray(mu, dtype=_np.float64))
    sigma_arr = _np.atleast_1d(_np.asarray(sigma, dtype=_np.float64))
    nu_arr = _np.atleast_1d(_np.asarray(nu, dtype=_np.float64))
    n = max(len(q_arr), len(mu_arr), len(sigma_arr), len(nu_arr))
    out = []
    for i in range(n):
        qi = float(q_arr[i % len(q_arr)])
        mui = float(mu_arr[i % len(mu_arr)])
        si = float(sigma_arr[i % len(sigma_arr)])
        nui = float(nu_arr[i % len(nu_arr)])
        if qi <= 0.0:
            c = 0.0
        else:
            if abs(nui) > 1e-12:
                z = (qi ** nui - 1.0) / nui
            else:
                z = _np.log(qi)
            c = _sn.cdf(z, loc=mui, scale=si)
        out.append(float(min(max(c, 0.0), 1.0)))
    cdf = _np.array(out)
    if not lower_tail:
        cdf = 1.0 - cdf
    if log_p:
        cdf = _np.log(cdf)
    return jnp.asarray(cdf, dtype=jnp.float64)


def _lno_q(p, mu=1.0, sigma=0.1, nu=0.0, lower_tail=True, log_p=False):
    """R LNO.R qLNO: inverse Box-Cox of mu + sigma*qnorm(p)."""
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
    nu_arr = _np.atleast_1d(_np.asarray(nu, dtype=_np.float64))
    n = max(len(p_arr), len(mu_arr), len(sigma_arr), len(nu_arr))
    out = []
    for i in range(n):
        pi = float(p_arr[i % len(p_arr)])
        mui = float(mu_arr[i % len(mu_arr)])
        si = float(sigma_arr[i % len(sigma_arr)])
        nui = float(nu_arr[i % len(nu_arr)])
        z = mui + si * _sn.ppf(pi)
        if abs(nui) > 1e-12:
            x = (nui * z + 1.0) ** (1.0 / nui)
        else:
            x = _np.exp(z)
        out.append(float(x))
    res = _np.array(out)
    return jnp.asarray(res, dtype=jnp.float64)


def _lno_r(key, n, mu=1.0, sigma=0.1, nu=0.0):
    import jax.numpy as _jnp
    import jax.random as _jr

    u = _jr.uniform(key, shape=(n,), dtype=_jnp.float64)
    return _lno_q(u, mu, sigma, nu)


def LNO() -> LogNormalLocationFamily:
    return build_ad_family(
        family_class=LogNormalLocationFamily,
        name="LNO",
        parameters=("mu", "sigma"),
        log_pdf_func=_lno_log_pdf,
        type_="Continuous",
        p=_lno_p,
        q=_lno_q,
        r=_lno_r,
        links={"mu": "identity", "sigma": "log"},
        link_functions={"mu": identity_link, "sigma": log_link},
        link_inverses={"mu": identity_inverse, "sigma": log_inverse},
        link_derivatives={"mu": identity_derivative, "sigma": log_derivative},
    )
