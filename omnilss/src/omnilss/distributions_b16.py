"""Additional distributions - Batch 16: PARETO1, PARETO1o, SEP, DPO1, LOGSHASHo, BCCGo."""

from __future__ import annotations

# Enable float64 precision for numerical accuracy


from dataclasses import dataclass
from typing import Tuple

import jax.numpy as jnp
from jax import Array
from jax.scipy.special import ndtr

from .ad import build_ad_family
from .families import FamilyDefinition

from .links import (
    identity_link, identity_inverse, identity_derivative,
    log_link, log_inverse, log_derivative,
)
eps = jnp.finfo(jnp.float64).eps


# =============================================================================
# PARETO1 - Pareto Type 1 (1-parameter)
# =============================================================================


@dataclass(frozen=True)
class Pareto1Family(FamilyDefinition):
    """Pareto Type 1 distribution (`PARETO1`).
    
    One-parameter Pareto distribution with support y > 0.
    
    Parameters:
        mu: shape parameter (mu > 0)
    
    PDF: f(y|mu) = mu * (1+y)^(-mu-1)
    """


def _pareto1_log_pdf(y: Array, mu: Array) -> Array:
    """Log-PDF for PARETO1 distribution.
    
    log f(y|mu) = log(mu) - (mu+1)*log(1+y)
    """
    mu = jnp.maximum(mu, eps)
    y = jnp.maximum(y, 0.0)
    
    log_pdf = jnp.log(mu) - (mu + 1.0) * jnp.log(1.0 + y)
    
    # Zero density for y <= 0
    log_pdf = jnp.where(y > 0.0, log_pdf, -jnp.inf)
    
    return log_pdf


def PARETO1() -> Pareto1Family:
    """Pareto Type 1 distribution (1-parameter).
    
    Returns:
        Pareto1Family with automatic differentiation support
    """
    from .dpqr_batch_e import dPARETO1, pPARETO1, qPARETO1, rPARETO1

    return build_ad_family(
        family_class=Pareto1Family,
        log_pdf_func=_pareto1_log_pdf,
        name="PARETO1",
        parameters=("mu",),
        d=dPARETO1,
        p=pPARETO1,
        q=qPARETO1,
        r=rPARETO1,
    )


# =============================================================================
# PARETO1o - Pareto Type 1 with fixed mu (2-parameter)
# =============================================================================


@dataclass(frozen=True)
class Pareto1oFamily(FamilyDefinition):
    """Pareto Type 1 with fixed mu (`PARETO1o`).
    
    Two-parameter Pareto distribution where mu is fixed (location parameter)
    and sigma is the shape parameter.
    
    Parameters:
        mu: location parameter (fixed, mu > 0)
        sigma: shape parameter (sigma > 0)
    
    PDF: f(y|mu,sigma) = sigma * mu^sigma * y^(-sigma-1)  for y > mu
    """
    
    fixed_parameters: Tuple[str, ...] = ("mu",)


def _pareto1o_log_pdf(y: Array, mu: Array, sigma: Array) -> Array:
    """Log-PDF for PARETO1o distribution.
    
    log f(y|mu,sigma) = log(sigma) + sigma*log(mu) - (sigma+1)*log(y)
    """
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    y = jnp.maximum(y, eps)
    
    log_pdf = jnp.log(sigma) + sigma * jnp.log(mu) - (sigma + 1.0) * jnp.log(y)
    
    # Zero density for y <= mu
    log_pdf = jnp.where(y > mu, log_pdf, -jnp.inf)
    
    return log_pdf


def PARETO1o() -> Pareto1oFamily:
    """Pareto Type 1 with fixed mu.
    
    Returns:
        Pareto1oFamily with automatic differentiation support
    """
    return build_ad_family(
        family_class=Pareto1oFamily,
        log_pdf_func=_pareto1o_log_pdf,
        name="PARETO1o",
        parameters=("mu", "sigma"),
    )


# =============================================================================
# SEP - Skew Exponential Power (base)
# =============================================================================


@dataclass(frozen=True)
class SEPFamily(FamilyDefinition):
    """Skew Exponential Power distribution (`SEP`).
    
    Base SEP distribution. SEP1-SEP4 are variants with different
    parameterizations.
    
    Parameters:
        mu: location parameter
        sigma: scale parameter (sigma > 0)
        nu: skewness parameter (nu > 0)
        tau: kurtosis parameter (tau > 0)
    
    This is a flexible distribution that can model skewness and kurtosis.
    """


def _sep_log_pdf(y, mu, sigma, nu, tau):
    """SEP log-PDF matching R SEP.R dSEP.

    z = (y-mu)/sigma
    w = sign(z)*|z|^(tau/2)*nu*sqrt(2/tau)
    loglik = log(pnorm(w)) - |z|^tau/tau - log(sigma) - lgamma(1/tau) - (1/tau-1)*log(tau)
    """
    import jax.numpy as jnp
    from jax.scipy.special import gammaln as _gl

    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    tau = jnp.maximum(tau, eps)

    z = (y - mu) / sigma
    z_abs = jnp.abs(z)
    w = jnp.sign(z) * jnp.power(z_abs, tau / 2.0) * nu * jnp.sqrt(2.0 / tau)

    # log(pnorm(w)) - use jax.scipy.stats.norm.logcdf
    from jax.scipy.stats import norm as _jn
    lcdf = _jn.logcdf(w)

    loglik = lcdf - jnp.power(z_abs, tau) / tau - jnp.log(sigma) - _gl(1.0 / tau) - (1.0 / tau - 1.0) * jnp.log(tau)
    return loglik


def _sep_std_log_density(z, ni, ti):
    """SEP 标准化（mu=0, sigma=1）log 密度，纯 scipy（R SEP.R dSEP）。

    z = x;  w = sign(z)*|z|^(tau/2)*nu*sqrt(2/tau)
    loglik = log(pnorm(w)) - |z|^tau/tau - lgamma(1/tau) - (1/tau-1)*log(tau)
    """
    import numpy as _np
    from scipy.stats import norm as _sn
    from scipy.special import gammaln as _gl

    z_arr = _np.asarray(z, dtype=_np.float64)
    z_abs = _np.abs(z_arr)
    w = _np.sign(z_arr) * _np.power(z_abs, ti / 2.0) * ni * _np.sqrt(2.0 / ti)
    return _sn.logcdf(w) - _np.power(z_abs, ti) / ti - _gl(1.0 / ti) - (1.0 / ti - 1.0) * _np.log(ti)


def _sep_p(q, mu=0.0, sigma=1.0, nu=0.0, tau=2.0, lower_tail=True, log_p=False):
    """SEP CDF via numerical integration (R SEP.R pSEP)."""
    import numpy as np
    import jax.numpy as jnp
    from scipy.integrate import quad

    def _density(z, _ni, _ti):
        return np.exp(_sep_std_log_density(z, _ni, _ti))

    q_arr = np.atleast_1d(np.asarray(q, dtype=np.float64))
    mu_arr = np.atleast_1d(np.asarray(mu, dtype=np.float64))
    sigma_arr = np.atleast_1d(np.asarray(sigma, dtype=np.float64))
    nu_arr = np.atleast_1d(np.asarray(nu, dtype=np.float64))
    tau_arr = np.atleast_1d(np.asarray(tau, dtype=np.float64))
    n = max(len(q_arr), len(mu_arr), len(sigma_arr), len(nu_arr), len(tau_arr))

    out = []
    for i in range(n):
        qi = float(q_arr[i % len(q_arr)])
        mui = float(mu_arr[i % len(mu_arr)])
        si = float(sigma_arr[i % len(sigma_arr)])
        ni = float(nu_arr[i % len(nu_arr)])
        ti = float(tau_arr[i % len(tau_arr)])
        if si <= 0 or ti <= 0:
            out.append(np.nan)
            continue
        # Standardized: mu=0, sigma=1 for the integrand, shift the limit
        z_lim = (qi - mui) / si
        if z_lim <= -200.0:
            val = 0.0
        elif z_lim >= 200.0:
            val = 1.0
        else:
            try:
                val, _ = quad(lambda z: _density(z, ni, ti), -np.inf, z_lim, limit=300)
            except Exception:
                val = np.nan
            out.append(float(min(max(val, 0.0), 1.0)))
            continue
        out.append(float(val))

    cdf = np.array(out)
    if not lower_tail:
        cdf = 1.0 - cdf
    if log_p:
        cdf = np.log(cdf)
    return jnp.asarray(cdf, dtype=jnp.float64)


def _sep_q(p, mu=0.0, sigma=1.0, nu=0.0, tau=2.0, lower_tail=True, log_p=False):
    """SEP quantile via root-finding (R SEP.R qSEP)."""
    import numpy as np
    import jax.numpy as jnp
    from scipy.optimize import brentq

    p_arr = np.atleast_1d(np.asarray(p, dtype=np.float64))
    if log_p:
        p_arr = np.exp(p_arr)
    if not lower_tail:
        p_arr = 1.0 - p_arr
    p_arr = np.clip(p_arr, 1e-10, 1.0 - 1e-10)
    mu_arr = np.atleast_1d(np.asarray(mu, dtype=np.float64))
    sigma_arr = np.atleast_1d(np.asarray(sigma, dtype=np.float64))
    nu_arr = np.atleast_1d(np.asarray(nu, dtype=np.float64))
    tau_arr = np.atleast_1d(np.asarray(tau, dtype=np.float64))
    n = max(len(p_arr), len(mu_arr), len(sigma_arr), len(nu_arr), len(tau_arr))

    out = []
    for i in range(n):
        pi = float(p_arr[i % len(p_arr)])
        mui = float(mu_arr[i % len(mu_arr)])
        si = float(sigma_arr[i % len(sigma_arr)])
        ni = float(nu_arr[i % len(nu_arr)])
        ti = float(tau_arr[i % len(tau_arr)])
        if si <= 0 or ti <= 0:
            out.append(np.nan)
            continue

        def obj(x, _pi=pi, _mui=mui, _si=si, _ni=ni, _ti=ti):
            return float(np.asarray(_sep_p(np.array([x]), _mui, _si, _ni, _ti)).reshape(-1)[0]) - _pi

        try:
            h_mu = obj(mui)
        except Exception:
            out.append(np.nan)
            continue
        found = False
        if h_mu < 0.0:
            lo, hi = mui, mui + si
            span = si
            for _ in range(80):
                try:
                    h_hi = obj(hi)
                except Exception:
                    h_hi = np.nan
                if h_hi >= 0.0 or (not np.isfinite(h_hi)):
                    found = np.isfinite(h_hi)
                    break
                span *= 2.0
                hi = mui + span
        else:
            lo, hi = mui - si, mui
            span = si
            for _ in range(80):
                try:
                    h_lo = obj(lo)
                except Exception:
                    h_lo = np.nan
                if h_lo <= 0.0 or (not np.isfinite(h_lo)):
                    found = np.isfinite(h_lo)
                    break
                span *= 2.0
                lo = mui - span
        if found:
            try:
                out.append(brentq(obj, lo, hi, xtol=1e-10))
                continue
            except Exception:
                pass
        out.append(np.nan)

    res = np.array(out)
    return jnp.asarray(res, dtype=jnp.float64)


def _sep_r(key, n, mu=0.0, sigma=1.0, nu=0.0, tau=2.0):
    """SEP random generation via inverse-CDF sampling."""
    import jax.random as jr

    u = jr.uniform(key, shape=(n,), dtype=jnp.float64)
    return _sep_q(u, mu, sigma, nu, tau)


def SEP() -> SEPFamily:
    """Skew Exponential Power distribution (R SEP)."""
    return build_ad_family(
        family_class=SEPFamily,
        log_pdf_func=_sep_log_pdf,
        name="SEP",
        parameters=("mu", "sigma", "nu", "tau"),
        type_="Continuous",
        p=_sep_p,
        q=_sep_q,
        r=_sep_r,
        links={"mu": "identity", "sigma": "log", "nu": "identity", "tau": "log"},
        link_functions={"mu": identity_link, "sigma": log_link, "nu": identity_link, "tau": log_link},
        link_inverses={"mu": identity_inverse, "sigma": log_inverse, "nu": identity_inverse, "tau": log_inverse},
        link_derivatives={"mu": identity_derivative, "sigma": log_derivative, "nu": identity_derivative, "tau": log_derivative},
    )



# =============================================================================
# DPO1 - Double Poisson Type 1 (alternative parameterization)
# =============================================================================


@dataclass(frozen=True)
class DPO1Family(FamilyDefinition):
    """Double Poisson Type 1 distribution (`DPO1`).
    
    Alternative parameterization of Double Poisson with approximate derivatives.
    
    Parameters:
        mu: mean parameter (mu > 0)
        sigma: dispersion parameter (sigma > 0)
    
    This is similar to DPO but uses approximate derivatives for the
    normalizing constant.
    """


def _dpo1_log_pdf(y: Array, mu: Array, sigma: Array) -> Array:
    """Log-PDF for DPO1 distribution.
    
    Uses approximate normalizing constant similar to DPO.
    """
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    y = jnp.maximum(y, 0.0)
    
    # Double Poisson log-likelihood
    # log f(y|mu,sigma) = (y*log(mu) - mu - log(y!))/sigma - 0.5*log(2*pi*sigma*y) + log(C)
    
    # Approximate normalizing constant
    log_c = jnp.where(
        y > 0.5,
        0.5 * jnp.log(2.0 * jnp.pi * sigma * y),
        0.0
    )
    
    # Poisson part
    log_poisson = y * jnp.log(mu) - mu - jnp.where(y > 0, jnp.lgamma(y + 1.0), 0.0)
    
    log_pdf = log_poisson / sigma - 0.5 * jnp.log(2.0 * jnp.pi * sigma * y) + log_c
    
    # Zero density for y < 0
    log_pdf = jnp.where(y >= 0.0, log_pdf, -jnp.inf)
    
    return log_pdf


def DPO1() -> DPO1Family:
    """Double Poisson Type 1 distribution.
    
    Alternative parameterization with approximate derivatives.
    
    Returns:
        DPO1Family with automatic differentiation support
    """
    return build_ad_family(
        family_class=DPO1Family,
        log_pdf_func=_dpo1_log_pdf,
        name="DPO1",
        parameters=("mu", "sigma"),
    )


# =============================================================================
# LOGSHASHo - Log Sinh-Arcsinh Original
# =============================================================================


@dataclass(frozen=True)
class LOGSHASHoFamily(FamilyDefinition):
    """Log Sinh-Arcsinh Original distribution (`LOGSHASHo`).
    
    Original parameterization of log SHASH distribution.
    
    Parameters:
        mu: location parameter
        sigma: scale parameter (sigma > 0)
        nu: skewness parameter (nu > 0)
        tau: kurtosis parameter (tau > 0)
    
    For x > 0:
    z = (x - mu) / sigma
    r = sinh(tau * asinh(z) - nu)
    c = cosh(tau * asinh(z) - nu)
    """


def _logshasho_log_pdf(y: Array, mu: Array, sigma: Array, nu: Array, tau: Array) -> Array:
    """Log-PDF for LOGSHASHo distribution.
    
    log f(y|mu,sigma,nu,tau) = -log(sigma) + log(tau) - 0.5*log(2*pi)
                                - 0.5*log(1+z^2) + log(c) - 0.5*r^2
    where:
        z = (y - mu) / sigma
        r = sinh(tau * asinh(z) - nu)
        c = cosh(tau * asinh(z) - nu)
    """
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.maximum(nu, eps)
    tau = jnp.maximum(tau, eps)
    
    z = (y - mu) / sigma
    
    # Compute sinh-arcsinh transformation
    asinh_z = jnp.arcsinh(z)
    arg = tau * asinh_z - nu
    
    r = jnp.sinh(arg)
    c = jnp.cosh(arg)
    
    log_pdf = (
        -jnp.log(sigma)
        + jnp.log(tau)
        - 0.5 * jnp.log(2.0 * jnp.pi)
        - 0.5 * jnp.log(1.0 + z * z)
        + jnp.log(c)
        - 0.5 * r * r
    )
    
    # Zero density for y <= 0
    log_pdf = jnp.where(y > 0.0, log_pdf, -jnp.inf)
    
    return log_pdf


def LOGSHASHo() -> LOGSHASHoFamily:
    """Log Sinh-Arcsinh Original distribution.
    
    Returns:
        LOGSHASHoFamily with automatic differentiation support
    """
    return build_ad_family(
        family_class=LOGSHASHoFamily,
        log_pdf_func=_logshasho_log_pdf,
        name="LOGSHASHo",
        parameters=("mu", "sigma", "nu", "tau"),
    )


# =============================================================================
# BCCGo - Box-Cox Cole-Green Original
# =============================================================================


@dataclass(frozen=True)
class BCCGoFamily(FamilyDefinition):
    """Box-Cox Cole-Green Original distribution (`BCCGo`).
    
    Original parameterization of BCCG distribution.
    
    Parameters:
        mu: location parameter (mu > 0)
        sigma: scale parameter (sigma > 0)
        nu: shape parameter
    
    For x > 0:
    z = ((x/mu)^nu - 1) / (nu * sigma)  if nu != 0
    z = log(x/mu) / sigma               if nu = 0
    
    PDF includes truncation adjustment: pnorm(1/(sigma*|nu|))
    """


def _bccgo_log_pdf(y: Array, mu: Array, sigma: Array, nu: Array) -> Array:
    """Log-PDF for BCCGo distribution.
    
    log f(y|mu,sigma,nu) = nu*log(y/mu) - log(sigma) - z^2/2 - log(y)
                           - 0.5*log(2*pi) - log(Phi(1/(sigma*|nu|)))
    where:
        z = ((y/mu)^nu - 1) / (nu * sigma)  if nu != 0
        z = log(y/mu) / sigma               if nu = 0
    """
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    y = jnp.maximum(y, eps)
    
    # Compute z transformation
    ratio = y / mu
    
    z = jnp.where(
        jnp.abs(nu) > eps,
        (jnp.power(ratio, nu) - 1.0) / (nu * sigma),
        jnp.log(ratio) / sigma
    )
    
    # Truncation adjustment
    trunc_arg = 1.0 / (sigma * jnp.maximum(jnp.abs(nu), eps))
    log_trunc = jnp.log(ndtr(trunc_arg))  # log(Phi(trunc_arg))
    
    log_pdf = (
        nu * jnp.log(ratio)
        - jnp.log(sigma)
        - 0.5 * z * z
        - jnp.log(y)
        - 0.5 * jnp.log(2.0 * jnp.pi)
        - log_trunc
    )
    
    # Zero density for y <= 0
    log_pdf = jnp.where(y > 0.0, log_pdf, -jnp.inf)
    
    return log_pdf


def BCCGo() -> BCCGoFamily:
    """Box-Cox Cole-Green Original distribution.
    
    Returns:
        BCCGoFamily with automatic differentiation support
    """
    from .dpqr_functions import pBCCG, qBCCG, rBCCG

    return build_ad_family(
        family_class=BCCGoFamily,
        log_pdf_func=_bccgo_log_pdf,
        name="BCCGo",
        parameters=("mu", "sigma", "nu"),
        type_="Continuous",
        p=pBCCG,
        q=qBCCG,
        r=rBCCG,
        links={"mu": "log", "sigma": "log", "nu": "identity"},
        link_functions={"mu": log_link, "sigma": log_link, "nu": identity_link},
        link_inverses={"mu": log_inverse, "sigma": log_inverse, "nu": identity_inverse},
        link_derivatives={"mu": log_derivative, "sigma": log_derivative, "nu": identity_derivative},
    )
