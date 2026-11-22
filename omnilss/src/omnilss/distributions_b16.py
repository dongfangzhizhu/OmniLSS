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
    identity_link,
    identity_inverse,
    identity_derivative,
    log_link,
    log_inverse,
    log_derivative,
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

    R's ``PARETO1`` default is ``mu.link = "log"`` (mu > 0).  Without this the
    family carries ``link_functions=None``; once it stopped being a placeholder
    the capability gate let it through and ``rs_fit`` died with an opaque
    ``TypeError: 'NoneType' object is not subscriptable``.

    Returns:
        Pareto1Family with automatic differentiation support
    """
    from .dpqr_batch_e import dPARETO1, pPARETO1, qPARETO1, rPARETO1

    return build_ad_family(
        family_class=Pareto1Family,
        log_pdf_func=_pareto1_log_pdf,
        name="PARETO1",
        parameters=("mu",),
        links={"mu": "log"},
        link_functions={"mu": log_link},
        link_inverses={"mu": log_inverse},
        link_derivatives={"mu": log_derivative},
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

    Outside the support (y <= mu) R's ``dPARETO1o`` returns ``0`` **on the
    log scale** as well: it computes ``lfy`` first and only then applies
    ``fy <- ifelse(x <= mu, 0, fy)``, i.e. the ``ifelse`` sits *after* the
    ``log`` branch (live R 6.1.1 verified).  The consequence that matters is
    ``G.dev.incr = -2 * dPARETO1o(y, mu, sigma, log = TRUE) = 0`` there, not
    ``+inf`` -- and with ``mu`` fixed at ``min(y)`` the minimum observation
    always sits exactly on the boundary, so every PARETO1o fit evaluates the
    deviance at that point.  We therefore return ``0.0`` (not ``-inf``) to
    keep ``g_dev_inc`` on R's contract.  The *density* is unaffected: the
    family wires the explicit ``dPARETO1o`` kernel, which returns 0 outside
    the support (``exp(0) = 1`` would be wrong).
    """
    interior = jnp.log(sigma) + sigma * jnp.log(mu) - (sigma + 1.0) * jnp.log(y)
    return jnp.where(y > mu, interior, 0.0)


# R's own derivative slots for PARETO1o (PARETO1_mu_fixed.R), transcribed
# verbatim -- including the parts that are inconsistent with R's density:
#
#   dldd  <- (1/sigma) + log(mu) - log(y)      # NO support guard
#   d2ldd2 <- -1/sigma^2                        # NO support guard
#
# R's ``dPARETO1o`` is zero for ``y <= mu`` (and hence so is its
# ``G.dev.incr`` contribution), but ``dldd`` keeps evaluating the interior
# formula there.  With ``mu`` fixed at ``min(y)`` the minimum observation
# always sits exactly on the edge, so its phantom ``1/sigma`` term is part of
# *every* PARETO1o fit: R's IRLS fixed point for sigma is
# ``n / sum(log(y/mu))`` (all n observations), while the deviance R reports
# excludes the boundary observation.  Measured on live R 6.1.1 (n = 30
# Pareto(1, 0.7) sample): R lands at eta = -0.18359862254654988, which is
# ``log(n / sum(log(y/mu)))`` to 10 digits, and does not move under
# ``c.crit = 1e-10`` -- i.e. it is R's fixed point, not an early stop.  To
# keep our fit (and reported deviance) on R's, the sigma score/hessian must
# be R's unguarded formulas; the AD-of-log_pdf versions vanish at
# ``y <= mu`` and would land at the deviance minimum instead (143.15351
# against R's 143.18722 on that sample -- lower, but a different model
# answer).  Only ``sigma`` needs an override: ``mu`` is fixed, and the AD
# values already equal R's interior formulas elsewhere.


def _pareto1o_dldd(y: Array, mu: Array, sigma: Array) -> Array:
    """R's ``dldd`` slot: ``(1/sigma) + log(mu) - log(y)``, unguarded."""
    return 1.0 / sigma + jnp.log(mu) - jnp.log(y)


def _pareto1o_d2ldd2(y: Array, mu: Array, sigma: Array) -> Array:
    """R's ``d2ldd2`` slot: ``-1/sigma^2``, unguarded."""
    return jnp.full_like(jnp.asarray(y, dtype=jnp.float64), -1.0 / (sigma * sigma))


_PARETO1O_SCORES = {"sigma": _pareto1o_dldd}
_PARETO1O_HESSIANS = {"sigma": _pareto1o_d2ldd2}


def PARETO1o() -> Pareto1oFamily:
    """Pareto Type 1 with fixed mu.

    R's ``PARETO1o`` defaults are ``mu.link = "log"`` / ``sigma.link = "log"``
    (``PARETO1_mu_fixed.R``).  ``mu`` is *fixed*: the family object marks it
    ``parameters = list(mu = FALSE, sigma = TRUE)`` and pins it at
    ``mu.initial = rep(min(y), length(y))`` -- a data-dependent default, which
    is why it is resolved in ``fitting._resolve_fixed_parameter_values``
    rather than in a constant ``fixed_parameter_defaults`` mapping.  ``sigma``
    starts at R's ``rep(0.5, length(y))`` (see ``_initial_sigma``).
    """
    from .dpqr_batch_e import dPARETO1o, pPARETO1o, qPARETO1o, rPARETO1o

    return build_ad_family(
        family_class=Pareto1oFamily,
        log_pdf_func=_pareto1o_log_pdf,
        name="PARETO1o",
        parameters=("mu", "sigma"),
        fixed_parameters=("mu",),
        links={"mu": "log", "sigma": "log"},
        link_functions={"mu": log_link, "sigma": log_link},
        link_inverses={"mu": log_inverse, "sigma": log_inverse},
        link_derivatives={"mu": log_derivative, "sigma": log_derivative},
        score_overrides=_PARETO1O_SCORES,
        hessian_overrides=_PARETO1O_HESSIANS,
        d=dPARETO1o,
        p=pPARETO1o,
        q=qPARETO1o,
        r=rPARETO1o,
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

    loglik = (
        lcdf
        - jnp.power(z_abs, tau) / tau
        - jnp.log(sigma)
        - _gl(1.0 / tau)
        - (1.0 / tau - 1.0) * jnp.log(tau)
    )
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
    return (
        _sn.logcdf(w)
        - _np.power(z_abs, ti) / ti
        - _gl(1.0 / ti)
        - (1.0 / ti - 1.0) * _np.log(ti)
    )


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
            return (
                float(
                    np.asarray(_sep_p(np.array([x]), _mui, _si, _ni, _ti)).reshape(-1)[
                        0
                    ]
                )
                - _pi
            )

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


def DPO1() -> DPO1Family:
    """Double Poisson type 1, sharing DPO kernels (D2.80).

    The previous body was not a density at all.  It added
    0.5*log(2*pi*sigma*y) as an "approximate normalizing constant" and then
    subtracted exactly the same term in the same expression, leaving

        log f(y) = (y log mu - mu - log y!) / sigma,

    i.e. a scaled Poisson that sums to 1 only at sigma = 1, plus a +inf spike
    at y = 0 (the subtracted log(0)).  Measured: sum over y = +inf for
    sigma = 1, 0.5 and 2, with mean 3.000 / 0.456 / 10.171 for mu = 3 -- while
    DPO, the R-verified double Poisson, sums to 1.000 with mean 3.013 / 2.999
    on the same grid.  The family also had no CDF and was typed Continuous.

    It now shares DPO kernels (density, CDF, quantiles, sampler), the Discrete
    type and the log links; the scanner probes it as R DPO.  gamlss.dist
    exports no dDPO1, so DPO was always the only reference.
    """
    from .distributions_b6 import DPO

    base = DPO()
    return DPO1Family(
        name="DPO1",
        parameters=base.parameters,
        g_dev_inc=base.g_dev_inc,
        type=base.type,
        links=base.links,
        link_functions=base.link_functions,
        link_inverses=base.link_inverses,
        link_derivatives=base.link_derivatives,
        score_functions=base.score_functions,
        hessian_functions=base.hessian_functions,
        fixed_parameters=base.fixed_parameters,
        d=base.d,
        p=base.p,
        q=base.q,
        r=base.r,
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


def _logshasho_log_pdf(
    y: Array, mu: Array, sigma: Array, nu: Array, tau: Array
) -> Array:
    """Log-density of the log-Sinh-Arcsinh: Y = exp(X), X ~ SHASHo (D2.81).

    R's own dLOGSHASHo (vendored gamlss.dist/R/LOGSHASHo.R:127) evaluates the
    SHASH density at z = (x - mu)/sigma -- on x, not on log(x) -- and then
    masks x <= 0.  R's pLOGSHASHo in the same file does the opposite:
    cdf <- pSHASHo(log(q), ...).  The two halves of R's own family describe
    different distributions, and the density half is not a density: with
    mu = 0, sigma = 1, nu = 0, tau = 1 it integrates to 0.500 over (0, inf),
    and with mu = -1 to 0.0023.

    Three of the four functions (p, q, r) agree with the log-transform
    reading, so that is the one implemented here: the density is
    f_SHASHo(log y) / y, which is exactly d/dy pLOGSHASHo.  The density and
    the CDF can no longer disagree, and the family integrates to 1.
    """
    from .distributions_b3 import _shasho_log_pdf

    eps = jnp.finfo(jnp.float64).eps
    y_safe = jnp.maximum(y, eps)
    log_y = jnp.log(y_safe)
    log_pdf = _shasho_log_pdf(log_y, mu, sigma, nu, tau) - log_y
    return jnp.where(y > 0.0, log_pdf, -jnp.inf)


def _logshasho_p(q, mu=0.0, sigma=1.0, nu=0.5, tau=0.5, lower_tail=True, log_p=False):
    """CDF: R pLOGSHASHo -- pSHASHo applied to log(q)."""
    from .distributions_b3 import _shasho_p

    q = jnp.asarray(q, dtype=jnp.float64)
    eps = jnp.finfo(jnp.float64).eps
    cdf = _shasho_p(jnp.log(jnp.maximum(q, eps)), mu, sigma, nu, tau, lower_tail, log_p)
    if lower_tail:
        below = -jnp.inf if log_p else 0.0
    else:
        below = 0.0
    return jnp.where(q > 0.0, cdf, below)


def _logshasho_q(p, mu=0.0, sigma=1.0, nu=0.5, tau=0.5, lower_tail=True, log_p=False):
    """Quantile: exp of the SHASHo quantile (the transform is monotone)."""
    from .distributions_b3 import _shasho_q

    return jnp.exp(_shasho_q(p, mu, sigma, nu, tau, lower_tail, log_p))


def _logshasho_r(key, n, mu=0.0, sigma=1.0, nu=0.5, tau=0.5):
    """Sampler: exp of a SHASHo draw."""
    from .distributions_b3 import _shasho_r

    return jnp.exp(_shasho_r(key, n, mu, sigma, nu, tau))


def LOGSHASHo() -> LOGSHASHoFamily:
    """Log Sinh-Arcsinh Original distribution.

    R's ``LOGSHASHo`` defaults are ``mu.link="identity"``,
    ``sigma.link="log"``, ``nu.link="log"``, ``tau.link="log"``.

    Returns:
        LOGSHASHoFamily with automatic differentiation support
    """
    return build_ad_family(
        family_class=LOGSHASHoFamily,
        log_pdf_func=_logshasho_log_pdf,
        name="LOGSHASHo",
        parameters=("mu", "sigma", "nu", "tau"),
        type_="Continuous",
        p=_logshasho_p,
        q=_logshasho_q,
        r=_logshasho_r,
        links={"mu": "identity", "sigma": "log", "nu": "log", "tau": "log"},
        link_functions={
            "mu": identity_link,
            "sigma": log_link,
            "nu": log_link,
            "tau": log_link,
        },
        link_inverses={
            "mu": identity_inverse,
            "sigma": log_inverse,
            "nu": log_inverse,
            "tau": log_inverse,
        },
        link_derivatives={
            "mu": identity_derivative,
            "sigma": log_derivative,
            "nu": log_derivative,
            "tau": log_derivative,
        },
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
        jnp.log(ratio) / sigma,
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
        link_derivatives={
            "mu": log_derivative,
            "sigma": log_derivative,
            "nu": identity_derivative,
        },
    )
