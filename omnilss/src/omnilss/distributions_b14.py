from __future__ import annotations

# Enable float64 precision for numerical accuracy

"""Batch 14: Skew distribution variants and SHASH variants."""
from dataclasses import dataclass
import jax.numpy as jnp
from jax.scipy.special import gammaln

from .ad import build_ad_family
from .dpqr_functions import pJSUo, qJSUo, rJSUo, pSHASHo2, qSHASHo2, rSHASHo2
from .families import FamilyDefinition
from .links import (
    log_link, log_inverse, log_derivative,
    identity_link, identity_inverse, identity_derivative
)


# ============================================================================
# SHASHo2 - SHASH Original 2
#
# NOTE: a second (wrong) ``SHASHo`` factory used to live in this file.  It was
# deleted in D2.40: its density (``sinh((asinh(z)-nu)/tau)`` with a ``-log``
# on tau and no ``log cosh`` term) is not R's ``dSHASHo``, and since
# ``omnilss/__init__.py`` re-exported *this* module's symbol, the public
# ``omnilss.SHASHo()`` returned the wrong distribution while the registry's
# ``SHASHO`` resolved to the correct one in ``distributions_b3``.  The single
# SHASHo implementation now lives in ``distributions_b3.SHASHo`` (verified
# against live R); ``distributions_o_alias_compat`` clones from there.
# ============================================================================

@dataclass(frozen=True)
class SHASHo2Family(FamilyDefinition):
    """SHASH Original 2 parameterization family."""
    pass


def _shasho2_log_pdf(y, mu, sigma, nu, tau):
    """SHASH Original-2 log-PDF -- delegates to the one and only SHASHo2 density.

    Transcribed from ``gamlss.dist:::dSHASHo2`` (live R 6.1.1, verified):
    ``z = (y-mu)/(sigma*tau)``, ``arg = tau*asinh(z) - nu``,
    ``loglik = -log(sigma) - 0.5*log(2*pi) - 0.5*log(1+z^2) + log(cosh(arg))
    - 0.5*sinh(arg)^2`` -- note **no** ``log(tau)`` term (it is absorbed into
    the ``1/tau`` inside ``z``).

    The independent reimplementation this replaces used ``z = (y-mu)/sigma``,
    ``sinh(tau*asinh(z) + nu)`` and a spurious ``+ log(tau)``: a different
    distribution from R's.  Delegating to ``dSHASHo2`` keeps the scoring path
    and the public ``d`` on the same formula (one family, one density).
    """
    from .dpqr_functions import dSHASHo2

    return dSHASHo2(y, mu, sigma, nu, tau, log=True)


# ---------------------------------------------------------------------------
# R's closed-form scores / diagonal second derivatives for SHASHo2 (D2.40).
#
# Why these exist: the AD Hessian is the *true* second derivative of the
# log-density, and for SHASHo2 it is **positive over most of a typical
# sample** (measured 32/40 observations at R's own start values on the
# parity sample).  R feeds the IRLS ``d2ldX2 = -dldX^2`` instead -- an
# observed-information approximation that is negative by construction.  A
# positive Hessian makes the IRLS weight ``w = -(d2/deta^2)`` negative,
# which clips to the 1e-10 floor in ``fitting.rs_step`` and explodes the
# step: the first RS cycle drove mu to +-1e8 and sigma to eps, deviance NaN,
# while live R converged to 26.02 on the same data.  D2.15 established the
# target is *what R feeds the IRLS*: a better second derivative makes
# parity worse.  Transcribed verbatim from the vendored gamlss.dist 6.1-8
# (``R/gamlss.dist/R/SHASHo2.R``, family-object slots dldm/dldd/dldv/dldt
# and d2ldm2/d2ldd2/d2ldv2/d2ldt2).
# ---------------------------------------------------------------------------

def _shasho2_parts(y, mu, sigma, nu, tau):
    """R's z / x / c / r with sigma, tau guarded as in ``dSHASHo2``."""
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    tau = jnp.maximum(tau, eps)
    z = (y - mu) / (sigma * tau)
    x = tau * jnp.arcsinh(z) - nu
    return z, x, jnp.cosh(x), jnp.sinh(x), sigma, tau


def _shasho2_dldm(y, mu, sigma, nu, tau):
    # R: dldm <- (1/(tau*sigma*(1+z^2)^.5)) *
    #        (r*tau*c - (tau*sinh(x))/c + z/(1+z^2)^.5)
    z, x, c, r, sigma, tau = _shasho2_parts(y, mu, sigma, nu, tau)
    s = jnp.sqrt(1.0 + jnp.square(z))
    return (1.0 / (tau * sigma * s)) * (r * tau * c - (tau * jnp.sinh(x)) / c + z / s)


def _shasho2_dldd(y, mu, sigma, nu, tau):
    # R: dldd <- (z/(sigma*(1+z^2)^.5)) *
    #        (r*tau*c - (tau*sinh(x))/c + z/(1+z^2)^.5) - 1/sigma
    z, x, c, r, sigma, tau = _shasho2_parts(y, mu, sigma, nu, tau)
    s = jnp.sqrt(1.0 + jnp.square(z))
    return (z / (sigma * s)) * (r * tau * c - (tau * jnp.sinh(x)) / c + z / s) - 1.0 / sigma


def _shasho2_dldv(y, mu, sigma, nu, tau):
    # R: dldv <- r*cosh(x) - (1/c)*sinh(x)
    z, x, c, r, sigma, tau = _shasho2_parts(y, mu, sigma, nu, tau)
    return r * c - jnp.sinh(x) / c


def _shasho2_dldt(y, mu, sigma, nu, tau):
    # R: dldt <- (-r*cosh(x) + (1/c)*sinh(x)) *
    #        (asinh(z) + (1/(1+z^2)^.5)*(-z)) + z^2/(tau*(1+z^2))
    z, x, c, r, sigma, tau = _shasho2_parts(y, mu, sigma, nu, tau)
    s = jnp.sqrt(1.0 + jnp.square(z))
    return (
        (-r * c + jnp.sinh(x) / c) * (jnp.arcsinh(z) - z / s)
        + jnp.square(z) / (tau * (1.0 + jnp.square(z)))
    )


def _shasho2_d2ldm2(y, mu, sigma, nu, tau):
    d = _shasho2_dldm(y, mu, sigma, nu, tau)
    return -d * d


def _shasho2_d2ldd2(y, mu, sigma, nu, tau):
    d = _shasho2_dldd(y, mu, sigma, nu, tau)
    return -d * d


def _shasho2_d2ldv2(y, mu, sigma, nu, tau):
    d = _shasho2_dldv(y, mu, sigma, nu, tau)
    return -d * d


def _shasho2_d2ldt2(y, mu, sigma, nu, tau):
    d = _shasho2_dldt(y, mu, sigma, nu, tau)
    return -d * d


_SHASHO2_SCORES = {
    "mu": _shasho2_dldm,
    "sigma": _shasho2_dldd,
    "nu": _shasho2_dldv,
    "tau": _shasho2_dldt,
}
_SHASHO2_HESSIANS = {
    "mu": _shasho2_d2ldm2,
    "sigma": _shasho2_d2ldd2,
    "nu": _shasho2_d2ldv2,
    "tau": _shasho2_d2ldt2,
}


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
        score_overrides=_SHASHO2_SCORES,
        hessian_overrides=_SHASHO2_HESSIANS,
        p=pSHASHo2,
        q=qSHASHo2,
        r=rSHASHo2,
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


# ---------------------------------------------------------------------------
# R's closed-form scores / diagonal second derivatives for JSUo (D2.60).
#
# Why these exist: the AD Hessian is the *true* second derivative of the
# log-density, while live R's JSUo() feeds the IRLS the observed-information
# approximation d2ldX2 = -dldX^2 for every parameter (floored at -1e-15
# inside the family function; the floor is re-applied downstream in
# compute_working_weights_and_response).  The two differ, and the IRLS
# step is built from the Hessian -- so with the AD Hessian our trajectory
# leaves R's from the very first inner fit even when the initial values
# match.  Measured on the scanner sample: with R's initials alone the first
# RS cycle landed at mu = [3.41, -0.51] against R's [1.84, +0.36] (slope
# sign flipped), and the 20-cycle deviance stayed 0.4% above R's.  The
# scores themselves are the true derivatives and already matched R's dldm
# through the deriv gate; transcribing them here keeps the family off the
# AD path entirely (same pattern as SHASHo2 in D2.40).
#
# Transcribed from live gamlss.dist 6.1.1 (JSUo() family-object slots
# dldm/dldd/dldv/dldt and d2ldm2/d2ldd2/d2ldv2/d2ldt2).
# ---------------------------------------------------------------------------

def _jsuo_dldm(y, mu, sigma, nu, tau):
    z = (y - mu) / sigma
    r = nu + tau * jnp.arcsinh(z)
    return z / (sigma * (z * z + 1.0)) + (r * tau) / (sigma * jnp.sqrt(z * z + 1.0))


def _jsuo_dldd(y, mu, sigma, nu, tau):
    z = (y - mu) / sigma
    r = nu + tau * jnp.arcsinh(z)
    return -1.0 / (sigma * (z * z + 1.0)) + (r * tau * z) / (
        sigma * jnp.sqrt(z * z + 1.0)
    )


def _jsuo_dldv(y, mu, sigma, nu, tau):
    z = (y - mu) / sigma
    r = nu + tau * jnp.arcsinh(z)
    return -r


def _jsuo_dldt(y, mu, sigma, nu, tau):
    z = (y - mu) / sigma
    r = nu + tau * jnp.arcsinh(z)
    return (1.0 + r * nu - r * r) / tau


def _jsuo_d2ldm2(y, mu, sigma, nu, tau):
    d = _jsuo_dldm(y, mu, sigma, nu, tau)
    return -d * d


def _jsuo_d2ldd2(y, mu, sigma, nu, tau):
    d = _jsuo_dldd(y, mu, sigma, nu, tau)
    return -d * d


def _jsuo_d2ldv2(y, mu, sigma, nu, tau):
    d = _jsuo_dldv(y, mu, sigma, nu, tau)
    return -d * d


def _jsuo_d2ldt2(y, mu, sigma, nu, tau):
    d = _jsuo_dldt(y, mu, sigma, nu, tau)
    return -d * d


_JSUO_SCORES = {
    "mu": _jsuo_dldm,
    "sigma": _jsuo_dldd,
    "nu": _jsuo_dldv,
    "tau": _jsuo_dldt,
}
_JSUO_HESSIANS = {
    "mu": _jsuo_d2ldm2,
    "sigma": _jsuo_d2ldd2,
    "nu": _jsuo_d2ldv2,
    "tau": _jsuo_d2ldt2,
}


def JSUo():
    """Johnson SU with original parameterization."""
    return build_ad_family(
        family_class=JSUoFamily,
        name="JSUo",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_jsuo_log_pdf,
        type_="Continuous",
        d=None,
        p=pJSUo,
        q=qJSUo,
        r=rJSUo,
        score_overrides=_JSUO_SCORES,
        hessian_overrides=_JSUO_HESSIANS,
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
# ST5 - Skew t Type 5 (Jones & Faddy)
# ============================================================================

@dataclass(frozen=True)
class ST5Family(FamilyDefinition):
    """Skew t Type 5 family (Jones & Faddy)."""
    pass


def _st5_log_pdf(y, mu, sigma, nu, tau):
    """ST5 log-PDF matching R ST5.R dST5.

    The Jones & Faddy skew-t distribution. R formula:
      z = (x-mu)/sigma;  ve = 2/tau;  lam = 2*nu/(tau*sqrt(2*tau+nu^2))
      a = (ve+lam)/2;  b = (ve-lam)/2
      loglik = (a+.5)log(1+z/sqrt(a+b+z^2)) + (b+.5)log(1-z/sqrt(a+b+z^2))
               - (a+b-1)log(2) - .5log(a+b) - lbeta(a,b) - log(sigma)
    """
    import jax.numpy as jnp
    from jax.scipy.special import gammaln as _gl, betaln as _bl

    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    tau = jnp.maximum(tau, eps)

    z = (y - mu) / sigma
    ve = 2.0 / tau
    lam = 2.0 * nu / (tau * jnp.sqrt(2.0 * tau + nu * nu))
    a = (ve + lam) / 2.0
    b = (ve - lam) / 2.0
    ab = a + b

    sq = jnp.sqrt(ab + z * z)
    log1p = jnp.log(1.0 + z / sq)
    log1m = jnp.log(1.0 - z / sq)

    loglik = (
        (a + 0.5) * log1p
        + (b + 0.5) * log1m
        - (ab - 1.0) * jnp.log(2.0)
        - 0.5 * jnp.log(ab)
        - _bl(a, b)
        - jnp.log(sigma)
    )
    return loglik


def _st5_p(q, mu=0.0, sigma=1.0, nu=0.0, tau=1.0, lower_tail=True, log_p=False):
    """ST5 CDF matching R ST5.R pST5.

    p = pbeta(alpha, a, b) where alpha = (1+z/sqrt(a+b+z^2))/2.
    """
    import numpy as np
    import jax.numpy as jnp
    from scipy.special import betainc

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
        zi = (qi - mui) / si
        ve = 2.0 / ti
        lam = 2.0 * ni / (ti * np.sqrt(2.0 * ti + ni * ni))
        a = (ve + lam) / 2.0
        b = (ve - lam) / 2.0
        alpha = (1.0 + zi / np.sqrt(a + b + zi * zi)) / 2.0
        # pbeta = regularized incomplete beta
        out.append(float(betainc(a, b, alpha)))

    cdf = np.array(out)
    if not lower_tail:
        cdf = 1.0 - cdf
    if log_p:
        cdf = np.log(cdf)
    return jnp.asarray(cdf, dtype=jnp.float64)


def _st5_q(p, mu=0.0, sigma=1.0, nu=0.0, tau=1.0, lower_tail=True, log_p=False):
    """ST5 quantile matching R ST5.R qST5.

    Balpha = qbeta(p,a,b); zalpha = sqrt(a+b)*(2*Balpha-1)/(2*sqrt(Balpha*(1-Balpha))).
    """
    import numpy as np
    import jax.numpy as jnp
    from scipy.special import betaincinv

    p_arr = np.atleast_1d(np.asarray(p, dtype=np.float64))
    if log_p:
        p_arr = np.exp(p_arr)
    if not lower_tail:
        p_arr = 1.0 - p_arr
    p_arr = np.clip(p_arr, 1e-16, 1.0 - 1e-16)
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
        ve = 2.0 / ti
        lam = 2.0 * ni / (ti * np.sqrt(2.0 * ti + ni * ni))
        a = (ve + lam) / 2.0
        b = (ve - lam) / 2.0
        Balpha = float(betaincinv(a, b, pi))
        zalpha = np.sqrt(a + b) * (2.0 * Balpha - 1.0) / (2.0 * np.sqrt(Balpha * (1.0 - Balpha)))
        out.append(float(mui + si * zalpha))

    res = np.array(out)
    return jnp.asarray(res, dtype=jnp.float64)


def _st5_r(key, n, mu=0.0, sigma=1.0, nu=0.0, tau=1.0):
    """ST5 random generation via inverse-CDF sampling."""
    import jax.random as jr

    u = jr.uniform(key, shape=(n,), dtype=jnp.float64)
    return _st5_q(u, mu, sigma, nu, tau)


# ---------------------------------------------------------------------------
# R's IRLS Hessian form for ST5 (D2.62).
#
# R's ST5() carries d2ldX2 = -dldX^2 for every parameter, with the floor at
# -1e-15 for mu/sigma and **-1e-4** for nu/tau (a coarser floor than the
# pipeline's own -1e-15).  Our AD path instead feeds the true second
# derivative, which leaves R's RS trajectory once the fit starts moving --
# the JSUO situation (D2.60).  R's scores are the exact derivatives (verified
# against a central difference of R's own dST5: rel 1e-10 at four probe
# points), and so are our AD scores (the deriv gate passes), so the Hessian
# overrides are built from the AD score: -dld^2 with R's per-parameter floor.
# ---------------------------------------------------------------------------

def _st5_deriv_parts(y, mu, sigma, nu, tau):
    """R's u / r / su, guarded as in the density."""
    from jax.scipy.special import digamma  # noqa: F401  (used by callers)

    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    tau = jnp.maximum(tau, eps)
    d = y - mu
    u = jnp.sqrt(tau) * d * jnp.power(2.0 * sigma * sigma + tau * d * d, -0.5)
    r = jnp.sqrt(nu * nu + 2.0 * tau)
    return u, r, jnp.sqrt(1.0 - u * u)


def _st5_dldm(y, mu, sigma, nu, tau):
    u, r, su = _st5_deriv_parts(y, mu, sigma, nu, tau)
    num = (
        (1.0 - nu / r + tau / 2.0) * su * (1.0 + u)
        - (1.0 + nu / r + tau / 2.0) * su * (1.0 - u)
    )
    return num / (sigma * jnp.sqrt(2.0 * tau))


def _st5_dldd(y, mu, sigma, nu, tau):
    u, r, su = _st5_deriv_parts(y, mu, sigma, nu, tau)
    num = (
        (1.0 - nu / r + tau / 2.0) * u * (1.0 + u)
        - (1.0 + nu / r + tau / 2.0) * u * (1.0 - u)
    )
    return -1.0 / sigma + num / (sigma * tau)


def _st5_dldv(y, mu, sigma, nu, tau):
    from jax.scipy.special import digamma

    u, r, su = _st5_deriv_parts(y, mu, sigma, nu, tau)
    val = (
        -digamma((1.0 + nu / r) / tau)
        + digamma((1.0 - nu / r) / tau)
        + jnp.log(1.0 + u)
        - jnp.log(1.0 - u)
    )
    return 2.0 * val / jnp.power(r, 3.0)


def _st5_dldt(y, mu, sigma, nu, tau):
    from jax.scipy.special import digamma

    u, r, su = _st5_deriv_parts(y, mu, sigma, nu, tau)
    acc = 4.0 * digamma(2.0 / tau) - tau - 4.0 * jnp.log(2.0)
    acc = acc + (
        (1.0 - nu / r + tau / 2.0) * u * (1.0 + u)
        - (1.0 + nu / r + tau / 2.0) * u * (1.0 - u)
    )
    acc = acc + 2.0 * (1.0 + nu * (r * r + tau) / jnp.power(r, 3.0)) * (
        jnp.log(1.0 + u) - digamma((1.0 + nu / r) / tau)
    )
    acc = acc + 2.0 * (1.0 - nu * (r * r + tau) / jnp.power(r, 3.0)) * (
        jnp.log(1.0 - u) - digamma((1.0 - nu / r) / tau)
    )
    return -acc / (2.0 * tau * tau)


def _st5_neg_sq(d):
    return -d * d


def _st5_d2ldm2(y, mu, sigma, nu, tau):
    v = _st5_neg_sq(_st5_dldm(y, mu, sigma, nu, tau))
    return jnp.where(v < -1e-15, v, -1e-15)


def _st5_d2ldd2(y, mu, sigma, nu, tau):
    v = _st5_neg_sq(_st5_dldd(y, mu, sigma, nu, tau))
    return jnp.where(v < -1e-15, v, -1e-15)


def _st5_d2ldv2(y, mu, sigma, nu, tau):
    v = _st5_neg_sq(_st5_dldv(y, mu, sigma, nu, tau))
    return jnp.where(v < -1e-4, v, -1e-4)


def _st5_d2ldt2(y, mu, sigma, nu, tau):
    v = _st5_neg_sq(_st5_dldt(y, mu, sigma, nu, tau))
    return jnp.where(v < -1e-4, v, -1e-4)


_ST5_SCORES = {
    "mu": _st5_dldm,
    "sigma": _st5_dldd,
    "nu": _st5_dldv,
    "tau": _st5_dldt,
}
_ST5_HESSIANS = {
    "mu": _st5_d2ldm2,
    "sigma": _st5_d2ldd2,
    "nu": _st5_d2ldv2,
    "tau": _st5_d2ldt2,
}


def ST5():
    """Skew t Type 5 (Jones & Faddy) distribution."""
    return build_ad_family(
        family_class=ST5Family,
        name="ST5",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_st5_log_pdf,
        type_="Continuous",
        score_overrides=_ST5_SCORES,
        hessian_overrides=_ST5_HESSIANS,
        p=_st5_p,
        q=_st5_q,
        r=_st5_r,
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
    """BCPE Original log-PDF -- delegates to the one and only BCPE density.

    R's ``dBCPEo`` body is byte-identical to ``dBCPE``'s
    (``identical(deparse(body(dBCPEo)), deparse(body(dBCPE)))`` is ``TRUE`` in
    gamlss.dist); the only difference between the two families is ``mu.link``.
    This used to be an independent reimplementation, and it drifted: it called
    ``jax.scipy.special.gammainc`` with a first argument of ``1/tau`` so ``gamlss
    RS`` died with ``Differentiation rule for 'igamma_grad_a' not implemented``
    -- the family could not be fitted at all.

    D2.15 established the rule for this cluster: one family, one density.  The
    fitting path and ``dpqr_functions.dBCPE`` now read the same code, so they
    cannot disagree.
    """
    from .dpqr_functions import dBCPE

    return dBCPE(y, mu, sigma, nu, tau, log=True)


def BCPEo():
    """Box-Cox Power Exponential with original parameterization."""
    from .distributions import BCPE
    from .dpqr_functions import pBCPE, qBCPE, rBCPE

    # R's closed-form derivatives, borrowed from BCPE (identical density).
    # AD cannot be used for tau: dBCPE reaches gammainc(1/tau, s) and
    # differentiating twice raises "igamma_grad_a not implemented", which is
    # why this family could not be fitted at all before.
    _base = BCPE()
    return build_ad_family(
        score_overrides=dict(_base.score_functions),
        hessian_overrides=dict(_base.hessian_functions),
        family_class=BCPEoFamily,
        name="BCPEo",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_bcpeo_log_pdf,
        type_="Continuous",
        p=pBCPE,
        q=qBCPE,
        r=rBCPE,
        # gamlss.dist::BCTo() / ::BCPEo() use mu.link = "log".  The "o"
        # suffix does NOT mean "untruncated": in R, dBCTo's body is identical
        # to dBCT's (verified with identical(deparse(...)) == TRUE) and the
        # only difference is this link.  BCCGo already had it right; these two
        # were on "identity", which silently reported coefficients on the
        # wrong scale.
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
# BCTo - Box-Cox t Original
# ============================================================================

@dataclass(frozen=True)
class BCToFamily(FamilyDefinition):
    """Box-Cox t Original family."""
    pass


def _bcto_log_pdf(y, mu, sigma, nu, tau):
    """BCT Original log-PDF -- delegates to the one and only BCT density.

    R's ``dBCTo`` body is byte-identical to ``dBCT``'s (verified with
    ``identical(deparse(...))``); the two families differ only in ``mu.link``.

    The independent reimplementation this replaces called ``betainc`` with
    ``a = tau/2``, and JAX has no derivative for ``betainc`` in ``a``/``b``
    (``a`` *is* ``df/2``), so every RS step raised ``Betainc gradient with
    respect to a and b not supported`` and ``BCTo`` could not be fitted.  The
    dedicated ``_log_pt_t`` helper in ``dpqr_functions`` (D2.13) carries a
    ``custom_jvp`` for exactly this reason -- delegating gets it for free.
    """
    from .dpqr_functions import dBCT

    return dBCT(y, mu, sigma, nu, tau, log=True)


def BCTo():
    """Box-Cox t with original parameterization."""
    from .distributions import BCT
    from .dpqr_functions import pBCT, qBCT, rBCT

    # R's closed-form derivatives, borrowed from BCT (identical density).
    # AD cannot be used for tau: dBCT reaches betainc(tau/2, 1/2, x), which has
    # no gradient in a/b, so the Hessian trace raises even though D2.13's
    # custom_jvp makes the first derivative work.
    _base = BCT()
    return build_ad_family(
        score_overrides=dict(_base.score_functions),
        hessian_overrides=dict(_base.hessian_functions),
        family_class=BCToFamily,
        name="BCTo",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_bcto_log_pdf,
        type_="Continuous",
        p=pBCT,
        q=qBCT,
        r=rBCT,
        # gamlss.dist::BCTo() / ::BCPEo() use mu.link = "log".  The "o"
        # suffix does NOT mean "untruncated": in R, dBCTo's body is identical
        # to dBCT's (verified with identical(deparse(...)) == TRUE) and the
        # only difference is this link.  BCCGo already had it right; these two
        # were on "identity", which silently reported coefficients on the
        # wrong scale.
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
