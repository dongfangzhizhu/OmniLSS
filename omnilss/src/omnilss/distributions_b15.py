from __future__ import annotations

# Enable float64 precision for numerical accuracy

"""Batch 15: Final distributions - SEP, SST, EGB2, GB2, LQNO, etc."""
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax.scipy.special import gammainc, gammaln, betaln
from scipy.stats import nbinom as _nbinom
from scipy.stats import poisson as _poisson
from jax.scipy.stats import t as _jst, norm as _jnorm

from .ad import build_ad_family
from .dpqr_batch_f import dEGB2, dLQNO, pEGB2, pLQNO, qEGB2, qLQNO, rEGB2, rLQNO
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


@dataclass(frozen=True)
class SEP2Family(FamilyDefinition):
    """Skew Exponential Power Type 2 family."""
    pass


@dataclass(frozen=True)
class SEP3Family(FamilyDefinition):
    """Skew Exponential Power Type 3 family."""
    pass


@dataclass(frozen=True)
class SEP4Family(FamilyDefinition):
    """Skew Exponential Power Type 4 family."""
    pass


def _sep1_log_pdf(y, mu, sigma, nu, tau):
    """SEP1 log-PDF matching R SEP1.R dSEP1.

    z = (y-mu)/sigma; w = nu*z
    sz = |z|^tau/tau;  s = |w|^tau/tau
    lpdf = (1-1/tau)log(tau) - sz - lgamma(1/tau) - log(2)
    lcdf = log(0.5*(1 + pgamma(s, 1/tau, 1)*sign(w)))
    loglik = lpdf + lcdf + log(2) - log(sigma)
    """
    import jax.numpy as jnp
    from jax.scipy.special import gammaln as _gl, gammainc as _gic

    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    tau = jnp.maximum(tau, eps)

    z = (y - mu) / sigma
    w = nu * z
    z_abs = jnp.abs(z)
    sz = jnp.power(z_abs, tau) / tau
    s = jnp.power(jnp.abs(w), tau) / tau

    # R: lpdf = (1-1/tau)*log(tau) - sz - lgamma(1/tau) - log(2)
    lpdf = (1.0 - 1.0 / tau) * jnp.log(tau) - sz - _gl(1.0 / tau) - jnp.log(2.0)
    # pgamma(s, shape=1/tau, scale=1) = gammainc(1/tau, s)
    pgamma_s = _gic(1.0 / tau, s)
    lcdf = jnp.log(0.5 * (1.0 + pgamma_s * jnp.sign(w)))
    # Handle s==0 case: R uses lcdf2 = log(0.5 + w*exp((1-1/tau)*log(tau) - lgamma(1/tau) - log(2)))
    lcdf2 = jnp.log(0.5 + w * jnp.exp((1.0 - 1.0 / tau) * jnp.log(tau) - _gl(1.0 / tau) - jnp.log(2.0)))
    lcdf = jnp.where(s == 0, lcdf2, lcdf)

    loglik = lpdf + lcdf + jnp.log(2.0) - jnp.log(sigma)
    return loglik


def _sep1_std_log_density(z, ni, ti):
    """SEP1 标准化（mu=0, sigma=1）log 密度，纯 scipy（R SEP1.R dSEP1）。

    z = x; w = nu*z; sz = |z|^tau/tau; s = |w|^tau/tau
    lpdf = (1-1/tau)log(tau) - sz - lgamma(1/tau) - log(2)
    lcdf = log(0.5*(1 + pgamma(s, 1/tau, 1)*sign(w)))
    loglik = lpdf + lcdf + log(2)
    """
    import numpy as _np
    from scipy.special import gammaln as _gl, gammainc as _gic

    z_arr = _np.asarray(z, dtype=_np.float64)
    w = ni * z_arr
    sz = _np.power(_np.abs(z_arr), ti) / ti
    s = _np.power(_np.abs(w), ti) / ti
    lpdf = (1.0 - 1.0 / ti) * _np.log(ti) - sz - _gl(1.0 / ti) - _np.log(2.0)
    pgamma_s = _gic(1.0 / ti, s)
    lcdf = _np.log(0.5 * (1.0 + pgamma_s * _np.sign(w)))
    lcdf2 = _np.log(0.5 + w * _np.exp((1.0 - 1.0 / ti) * _np.log(ti) - _gl(1.0 / ti) - _np.log(2.0)))
    lcdf = _np.where(s == 0, lcdf2, lcdf)
    return lpdf + lcdf + _np.log(2.0)


def _sep1_p(q, mu=0.0, sigma=1.0, nu=0.0, tau=2.0, lower_tail=True, log_p=False):
    """SEP1 CDF via numerical integration (R SEP1.R pSEP1)."""
    import numpy as np
    import jax.numpy as jnp
    from scipy.integrate import quad

    def _density(z, _ni, _ti):
        return np.exp(_sep1_std_log_density(z, _ni, _ti))

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
        z_lim = (qi - mui) / si
        if z_lim <= -200.0:
            val = 0.0
        elif z_lim >= 200.0:
            val = 1.0
        else:
            try:
                # Integrating the requested tail directly avoids losing all
                # precision when an upper-tail probability is formed as
                # ``1 - lower_tail`` (R's lower.tail=FALSE semantics).
                if lower_tail:
                    val, _ = quad(lambda z: _density(z, ni, ti), -np.inf, z_lim, limit=300)
                else:
                    val, _ = quad(lambda z: _density(z, ni, ti), z_lim, np.inf, limit=300)
            except Exception:
                val = np.nan
            out.append(float(min(max(val, 0.0), 1.0)))
            continue
        out.append(float(1.0 - val if not lower_tail else val))

    cdf = np.array(out)
    if log_p:
        cdf = np.log(cdf)
    return jnp.asarray(cdf, dtype=jnp.float64)


def _sep1_q(p, mu=0.0, sigma=1.0, nu=0.0, tau=2.0, lower_tail=True, log_p=False):
    """SEP1 quantile via root-finding (R SEP1.R qSEP1)."""
    import numpy as np
    import jax.numpy as jnp
    from scipy.optimize import brentq

    p_arr = np.atleast_1d(np.asarray(p, dtype=np.float64))
    if log_p:
        p_arr = np.exp(p_arr)
    if not lower_tail:
        p_arr = 1.0 - p_arr
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
        if not 0.0 <= pi <= 1.0 or not np.isfinite(pi):
            out.append(np.nan)
            continue
        if pi == 0.0:
            out.append(-np.inf)
            continue
        if pi == 1.0:
            out.append(np.inf)
            continue

        def obj(x, _pi=pi, _mui=mui, _si=si, _ni=ni, _ti=ti):
            return float(np.asarray(_sep1_p(np.array([x]), _mui, _si, _ni, _ti)).reshape(-1)[0]) - _pi

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


def _sep1_r(key, n, mu=0.0, sigma=1.0, nu=0.0, tau=2.0):
    """SEP1 random generation via inverse-CDF sampling."""
    import jax.random as jr

    u = jr.uniform(key, shape=(n,), dtype=jnp.float64)
    return _sep1_q(u, mu, sigma, nu, tau)


# ---------------------------------------------------------------------------
# R's closed-form scores / diagonal second derivatives for SEP1 (D2.61).
#
# Why these exist: SEP1's density reaches gammainc(1/tau, s).  JAX can
# differentiate that once (igamma_grad_a covers d/da), so the score functions
# build and the deriv gate passed -- but the *second* derivative re-enters
# igamma_grad_a and raises "Differentiation rule for 'igamma_grad_a' not
# implemented".  The RS fit needs the Hessian, so SEP1 could not be fitted at
# all: rsdev reported "python raised NotImplementedError" on every sample.
# This is exactly the BCT/BCPE pattern documented in build_ad_family: hand in
# R's own derivatives instead of teaching JAX another special function.
#
# R's SEP1() slots, transcribed verbatim from live gamlss.dist 6.1.1:
#   dldm/dldd/dldv   closed forms built from lpdf, lcdf and pgamma
#   dldt             analytic part + a *forward difference* of the CDF part
#                    in tau (t1 = tau + 2e-5, t2 = tau - 2e-5) -- R itself
#                    cannot differentiate pgamma(s, 1/tau) in tau, so parity
#                    means using R's difference, not a better derivative.
#   d2ldX2           -dldX^2 (observed information), floored downstream.
#
# Note on z == 0: R's dldt contains |z|^tau * log|z|, which is 0 * -Inf = NaN
# at z = 0 exactly.  An exact zero would poison the whole score vector, and
# the true limit there is 0, so the log argument is floored at eps.
# ---------------------------------------------------------------------------

def _sep1_deriv_parts(y, mu, sigma, nu, tau):
    """R's z / w / s / lpdf / lcdf, guarded as in the density."""
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    tau = jnp.maximum(tau, eps)
    z = (y - mu) / sigma
    w = nu * z
    s = jnp.power(jnp.abs(w), tau) / tau
    lpdf = (1.0 - 1.0 / tau) * jnp.log(tau) - s - gammaln(1.0 / tau) - jnp.log(2.0)
    pgamma_s = gammainc(1.0 / tau, s)
    lcdf = jnp.log(0.5 * (1.0 + pgamma_s * jnp.sign(w)))
    lcdf2 = jnp.log(
        0.5
        + w * jnp.exp(
            (1.0 - 1.0 / tau) * jnp.log(tau) - gammaln(1.0 / tau) - jnp.log(2.0)
        )
    )
    lcdf = jnp.where(s == 0, lcdf2, lcdf)
    return z, w, s, lpdf, lcdf


def _sep1_dldm(y, mu, sigma, nu, tau):
    z, w, s, lpdf, lcdf = _sep1_deriv_parts(y, mu, sigma, nu, tau)
    z_abs = jnp.abs(z)
    return (
        -jnp.exp(lpdf - lcdf) * nu / sigma
        + jnp.sign(z) * jnp.power(z_abs, tau - 1.0) / sigma
    )


def _sep1_dldd(y, mu, sigma, nu, tau):
    z, w, s, lpdf, lcdf = _sep1_deriv_parts(y, mu, sigma, nu, tau)
    return (
        -jnp.exp(lpdf - lcdf) * nu * z / sigma
        + (jnp.power(jnp.abs(z), tau) - 1.0) / sigma
    )


def _sep1_dldv(y, mu, sigma, nu, tau):
    z, w, s, lpdf, lcdf = _sep1_deriv_parts(y, mu, sigma, nu, tau)
    dwdv = w / nu
    return jnp.exp(lpdf - lcdf) * dwdv


def _sep1_dldt(y, mu, sigma, nu, tau):
    from jax.scipy.special import digamma

    eps = jnp.finfo(jnp.float64).eps
    tau = jnp.maximum(tau, eps)
    z, w, s, lpdf, lcdf = _sep1_deriv_parts(y, mu, sigma, nu, tau)
    z_abs = jnp.abs(z)
    # R: dldf <- (log(tau) + tau - 1 + digamma(1/tau)
    #             - tau*|z|^tau*log|z| + |z|^tau) / tau^2
    z_log = jnp.log(jnp.where(z_abs > 0.0, z_abs, eps))
    z_tau = jnp.power(z_abs, tau)
    dldf = (
        jnp.log(tau) + tau - 1.0 + digamma(1.0 / tau)
        - tau * z_tau * z_log + z_tau
    ) / (tau * tau)
    # R's forward difference for the CDF part (delta = 2e-5 either side).
    t1 = tau + 2e-05
    t2 = tau - 2e-05
    w_abs = jnp.abs(w)
    s1 = jnp.power(w_abs, t1) / t1
    s2 = jnp.power(w_abs, t2) / t2
    dldcdf1 = jnp.sign(w) * (
        jnp.log(1.0 + gammainc(1.0 / t1, s1))
        - jnp.log(1.0 + gammainc(1.0 / t2, s2))
    ) / 4e-05
    lcdf1 = jnp.log(
        0.5 + w * jnp.exp((1.0 - 1.0 / t1) * jnp.log(t1) - gammaln(1.0 / t1) - jnp.log(2.0))
    )
    lcdf2 = jnp.log(
        0.5 + w * jnp.exp((1.0 - 1.0 / t2) * jnp.log(t2) - gammaln(1.0 / t2) - jnp.log(2.0))
    )
    dldcdf2 = (lcdf1 - lcdf2) / 4e-05
    dldcdf = jnp.where(s == 0, dldcdf2, dldcdf1)
    return dldf + dldcdf


def _sep1_d2ldm2(y, mu, sigma, nu, tau):
    d = _sep1_dldm(y, mu, sigma, nu, tau)
    return -d * d


def _sep1_d2ldd2(y, mu, sigma, nu, tau):
    d = _sep1_dldd(y, mu, sigma, nu, tau)
    return -d * d


def _sep1_d2ldv2(y, mu, sigma, nu, tau):
    d = _sep1_dldv(y, mu, sigma, nu, tau)
    return -d * d


def _sep1_d2ldt2(y, mu, sigma, nu, tau):
    d = _sep1_dldt(y, mu, sigma, nu, tau)
    return -d * d


_SEP1_SCORES = {
    "mu": _sep1_dldm,
    "sigma": _sep1_dldd,
    "nu": _sep1_dldv,
    "tau": _sep1_dldt,
}
_SEP1_HESSIANS = {
    "mu": _sep1_d2ldm2,
    "sigma": _sep1_d2ldd2,
    "nu": _sep1_d2ldv2,
    "tau": _sep1_d2ldt2,
}


def SEP1():
    """Skew Exponential Power Type 1 (R SEP1)."""
    return build_ad_family(
        family_class=SEP1Family,
        name="SEP1",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_sep1_log_pdf,
        type_="Continuous",
        score_overrides=_SEP1_SCORES,
        hessian_overrides=_SEP1_HESSIANS,
        p=_sep1_p,
        q=_sep1_q,
        r=_sep1_r,
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


def _sep2_log_pdf(y, mu, sigma, nu, tau):
    """SEP2 log-PDF (identical to SEP per R SEP2.R comment)."""
    import jax.numpy as jnp
    from jax.scipy.special import gammaln as _gl
    from jax.scipy.stats import norm as _jn

    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    tau = jnp.maximum(tau, eps)

    z = (y - mu) / sigma
    z_abs = jnp.abs(z)
    w = jnp.sign(z) * jnp.power(z_abs, tau / 2.0) * nu * jnp.sqrt(2.0 / tau)

    lcdf = _jn.logcdf(w)
    loglik = lcdf - jnp.power(z_abs, tau) / tau - jnp.log(sigma) - _gl(1.0 / tau) - (1.0 / tau - 1.0) * jnp.log(tau)
    return loglik


def _sep3_log_pdf(y, mu, sigma, nu, tau):
    """SEP3 log-PDF matching R SEP3.R dSEP3.

    loglik = ifelse(x<mu, -0.5*(nu*|z|)^tau, -0.5*(|z|/nu)^tau)
             - log(sigma) + log(nu) - log(1+nu^2) - log(2)/tau - lgamma(1+1/tau)
    """
    import jax.numpy as jnp
    from jax.scipy.special import gammaln as _gl

    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.maximum(nu, eps)
    tau = jnp.maximum(tau, eps)

    z = (y - mu) / sigma
    z_abs = jnp.abs(z)
    loglik1 = -0.5 * jnp.power(nu * z_abs, tau)
    loglik2 = -0.5 * jnp.power(z_abs / nu, tau)
    loglik = jnp.where(y < mu, loglik1, loglik2)
    loglik = loglik - jnp.log(sigma) + jnp.log(nu) - jnp.log(1.0 + nu * nu) - jnp.log(2.0) / tau - _gl(1.0 + 1.0 / tau)
    return loglik


def _sep4_log_pdf(y, mu, sigma, nu, tau):
    """SEP4 log-PDF matching R SEP4.R dSEP4.

    k1 = gamma(1+1/nu); k2 = gamma(1+1/tau)
    loglik = ifelse(x<mu, -|z|^nu, -|z|^tau) - log(k1+k2) - log(sigma)
    """
    import jax.numpy as jnp
    from jax.scipy.special import gammaln as _gl

    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.maximum(nu, eps)
    tau = jnp.maximum(tau, eps)

    z = (y - mu) / sigma
    z_abs = jnp.abs(z)
    lk1 = _gl(1.0 + 1.0 / nu)
    lk2 = _gl(1.0 + 1.0 / tau)
    loglik1 = -jnp.power(z_abs, nu)
    loglik2 = -jnp.power(z_abs, tau)
    loglik = jnp.where(y < mu, loglik1, loglik2)
    loglik = loglik - jnp.log(jnp.exp(lk1) + jnp.exp(lk2)) - jnp.log(sigma)
    return loglik


# SEP3/SEP4 have closed-form p/q via pgamma/qgamma
def _sep3_p(q, mu=0.0, sigma=1.0, nu=2.0, tau=2.0, lower_tail=True, log_p=False):
    """SEP3 CDF matching R SEP3.R pSEP3."""
    import numpy as np
    import jax.numpy as jnp
    from scipy.special import gammaincc, gammainc

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
        if si <= 0 or ni <= 0 or ti <= 0:
            out.append(np.nan)
            continue
        k = ni * ni
        # z1 = nu*(q-mu)/(sigma*2^(1/tau)); s1 = |z1|^tau
        z1 = ni * (qi - mui) / (si * np.power(2.0, 1.0 / ti))
        z2 = (qi - mui) / (si * ni * np.power(2.0, 1.0 / ti))
        s1 = np.power(np.abs(z1), ti)
        s2 = np.power(np.abs(z2), ti)
        # cdf1 = 1 - pgamma(s1, 1/tau, 1) = gammaincc(1/tau, s1)
        cdf1 = float(gammaincc(1.0 / ti, s1))
        # cdf2 = 1 + k*pgamma(s2, 1/tau, 1) = 1 + k*gammainc(1/tau, s2)
        cdf2 = 1.0 + k * float(gammainc(1.0 / ti, s2))
        cdf = cdf1 if qi < mui else cdf2
        cdf = cdf / (1.0 + k)
        # Handle tau > 10000 (limiting case)
        if ti > 10000:
            cdf = (qi - (mui - si / ni)) / (si * (1.0 / ni + ni))
        out.append(float(min(max(cdf, 0.0), 1.0)))

    cdf = np.array(out)
    if not lower_tail:
        cdf = 1.0 - cdf
    if log_p:
        cdf = np.log(cdf)
    return jnp.asarray(cdf, dtype=jnp.float64)


def _sep3_q(p, mu=0.0, sigma=1.0, nu=2.0, tau=2.0, lower_tail=True, log_p=False):
    """SEP3 quantile matching R SEP3.R qSEP3."""
    import numpy as np
    import jax.numpy as jnp
    from scipy.special import gammaincinv, gammainccinv

    p_arr = np.atleast_1d(np.asarray(p, dtype=np.float64))
    if log_p:
        p_arr = np.exp(p_arr)
    if not lower_tail:
        p_arr = 1.0 - p_arr
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
        if not np.isfinite(pi) or pi < 0.0 or pi > 1.0 or si <= 0 or ti <= 0:
            out.append(np.nan)
            continue
        if pi == 0.0:
            out.append(-np.inf)
            continue
        if pi == 1.0:
            out.append(np.inf)
            continue
        k = ni * ni
        # q1 = mu - sigma*2^(1/tau)/nu * (qgamma(1-p*(1+k), 1/tau, 1))^(1/tau)
        # qgamma(a, shape, scale) = gammaincinv(shape, a) * scale
        # Note: qgamma gives lower quantile. 1-p*(1+k) is the survival probability.
        qgamma1 = float(gammaincinv(1.0 / ti, 1.0 - pi * (1.0 + k)))
        q1 = mui - si * np.power(2.0, 1.0 / ti) / ni * np.power(qgamma1, 1.0 / ti)
        # q2 = mu + sigma*nu*2^(1/tau) * (qgamma(-1/k*(1-p*(1+k)), 1/tau, 1))^(1/tau)
        arg2 = -1.0 / k * (1.0 - pi * (1.0 + k))
        qgamma2 = float(gammaincinv(1.0 / ti, arg2))
        q2 = mui + si * ni * np.power(2.0, 1.0 / ti) * np.power(qgamma2, 1.0 / ti)
        q = q1 if pi < (1.0 / (1.0 + k)) else q2
        out.append(float(q))

    res = np.array(out)
    if np.ndim(p) == 0:
        return float(res[0])
    return jnp.asarray(res, dtype=jnp.float64)


def _sep3_r(key, n, mu=0.0, sigma=1.0, nu=2.0, tau=2.0):
    """SEP3 random generation via inverse-CDF sampling."""
    import jax.random as jr

    u = jr.uniform(key, shape=(n,), dtype=jnp.float64)
    return _sep3_q(u, mu, sigma, nu, tau)


def _sep4_p(q, mu=0.0, sigma=1.0, nu=2.0, tau=2.0, lower_tail=True, log_p=False):
    """SEP4 CDF matching R SEP4.R pSEP4."""
    import numpy as np
    import jax.numpy as jnp
    from scipy.special import gammaincc, gammainc

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
        if si <= 0 or ni <= 0 or ti <= 0:
            out.append(np.nan)
            continue
        from scipy.special import gammaln as _sp_gammaln
        lk1 = _sp_gammaln(1.0 + 1.0 / ni)
        lk2 = _sp_gammaln(1.0 + 1.0 / ti)
        k = np.exp(lk2 - lk1)
        z = (qi - mui) / si
        s1 = np.power(np.abs(z), ni)
        s2 = np.power(np.abs(z), ti)
        # cdf1 = 1 - pgamma(s1, 1/nu, 1) = gammaincc(1/nu, s1)
        cdf1 = float(gammaincc(1.0 / ni, s1))
        # cdf2 = 1 + k*pgamma(s2, 1/tau, 1) = 1 + k*gammainc(1/tau, s2)
        cdf2 = 1.0 + k * float(gammainc(1.0 / ti, s2))
        cdf = cdf1 if qi < mui else cdf2
        cdf = cdf / (1.0 + k)
        out.append(float(min(max(cdf, 0.0), 1.0)))

    cdf = np.array(out)
    if not lower_tail:
        cdf = 1.0 - cdf
    if log_p:
        cdf = np.log(cdf)
    return jnp.asarray(cdf, dtype=jnp.float64)


def _sep4_q(p, mu=0.0, sigma=1.0, nu=2.0, tau=2.0, lower_tail=True, log_p=False):
    """SEP4 quantile via root-finding (R SEP4.R qSEP4 uses uniroot)."""
    import numpy as np
    import jax.numpy as jnp
    from scipy.optimize import brentq

    p_arr = np.atleast_1d(np.asarray(p, dtype=np.float64))
    if log_p:
        p_arr = np.exp(p_arr)
    if not lower_tail:
        p_arr = 1.0 - p_arr
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
        if not np.isfinite(pi) or pi < 0.0 or pi > 1.0 or si <= 0 or ni <= 0 or ti <= 0:
            out.append(np.nan)
            continue
        if pi == 0.0:
            out.append(-np.inf)
            continue
        if pi == 1.0:
            out.append(np.inf)
            continue

        def obj(x, _pi=pi, _mui=mui, _si=si, _ni=ni, _ti=ti):
            return float(np.asarray(_sep4_p(np.array([x]), _mui, _si, _ni, _ti)).reshape(-1)[0]) - _pi

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


def _sep4_r(key, n, mu=0.0, sigma=1.0, nu=2.0, tau=2.0):
    """SEP4 random generation via inverse-CDF sampling."""
    import jax.random as jr

    u = jr.uniform(key, shape=(n,), dtype=jnp.float64)
    return _sep4_q(u, mu, sigma, nu, tau)



def _sep2_std_log_density(z, ni, ti):
    """SEP2 标准化 log 密度（与 SEP 相同，R SEP2.R 注释明确相同）。"""
    import numpy as _np
    from scipy.stats import norm as _sn
    from scipy.special import gammaln as _gl

    z_arr = _np.asarray(z, dtype=_np.float64)
    z_abs = _np.abs(z_arr)
    w = _np.sign(z_arr) * _np.power(z_abs, ti / 2.0) * ni * _np.sqrt(2.0 / ti)
    return _sn.logcdf(w) - _np.power(z_abs, ti) / ti - _gl(1.0 / ti) - (1.0 / ti - 1.0) * _np.log(ti)


def _sep2_p(q, mu=0.0, sigma=1.0, nu=0.0, tau=2.0, lower_tail=True, log_p=False):
    """SEP2 CDF (same as SEP: numerical integration)."""
    import numpy as np
    import jax.numpy as jnp
    from scipy.integrate import quad

    def _density(z, _ni, _ti):
        return np.exp(_sep2_std_log_density(z, _ni, _ti))

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


def _sep2_q(p, mu=0.0, sigma=1.0, nu=0.0, tau=2.0, lower_tail=True, log_p=False):
    """SEP2 quantile (same as SEP: root-finding)."""
    import numpy as np
    import jax.numpy as jnp
    from scipy.optimize import brentq

    p_arr = np.atleast_1d(np.asarray(p, dtype=np.float64))
    if log_p:
        p_arr = np.exp(p_arr)
    if not lower_tail:
        p_arr = 1.0 - p_arr
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
        if not np.isfinite(pi) or pi < 0.0 or pi > 1.0 or si <= 0 or ti <= 0:
            out.append(np.nan)
            continue
        if pi == 0.0:
            out.append(-np.inf)
            continue
        if pi == 1.0:
            out.append(np.inf)
            continue

        def obj(x, _pi=pi, _mui=mui, _si=si, _ni=ni, _ti=ti):
            return float(np.asarray(_sep2_p(np.array([x]), _mui, _si, _ni, _ti)).reshape(-1)[0]) - _pi

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


def _sep2_r(key, n, mu=0.0, sigma=1.0, nu=0.0, tau=2.0):
    """SEP2 random generation via inverse-CDF sampling."""
    import jax.random as jr

    u = jr.uniform(key, shape=(n,), dtype=jnp.float64)
    return _sep2_q(u, mu, sigma, nu, tau)


def SEP2():
    """Skew Exponential Power Type 2 (same as SEP per R SEP2.R)."""
    return build_ad_family(
        family_class=SEP2Family,
        name="SEP2",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_sep2_log_pdf,
        type_="Continuous",
        p=_sep2_p,
        q=_sep2_q,
        r=_sep2_r,
        links={"mu": "identity", "sigma": "log", "nu": "identity", "tau": "log"},
        link_functions={"mu": identity_link, "sigma": log_link, "nu": identity_link, "tau": log_link},
        link_inverses={"mu": identity_inverse, "sigma": log_inverse, "nu": identity_inverse, "tau": log_inverse},
        link_derivatives={"mu": identity_derivative, "sigma": log_derivative, "nu": identity_derivative, "tau": log_derivative},
    )


def _fisher_hessians(family):
    """R's Fisher-scoring second derivatives: -dld*^2, floored at -1e-15.

    SEP3/SEP4 define d2ldm2/d2ldd2/d2ldv2/d2ldt2 as minus the squared score
    (SEP3.R lines 44-51, SEP4.R lines 44-51), not as the observed curvature that
    autodiff would give.  At tau = 1 SEP3 is an asymmetric Laplace whose
    log-density is piecewise linear in mu, so the AD Hessian is exactly zero and
    the IWLS weights collapse (measured: mu jumped to -1.2e6 in one step while
    R converges).  The Fisher form keeps the fit on R's trajectory.
    """

    import jax.numpy as jnp

    hessians = {}
    for name, score in (family.score_functions or {}).items():
        def make(score_fn):
            def hessian(**kwargs):
                value = score_fn(**kwargs)
                return jnp.minimum(-value * value, -1e-15)

            return hessian

        hessians[name] = make(score)
    return hessians


def SEP3():
    """Skew Exponential Power Type 3 (R SEP3)."""
    family = build_ad_family(
        family_class=SEP3Family,
        name="SEP3",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_sep3_log_pdf,
        type_="Continuous",
        p=_sep3_p,
        q=_sep3_q,
        r=_sep3_r,
        links={"mu": "identity", "sigma": "log", "nu": "log", "tau": "log"},
        link_functions={"mu": identity_link, "sigma": log_link, "nu": log_link, "tau": log_link},
        link_inverses={"mu": identity_inverse, "sigma": log_inverse, "nu": log_inverse, "tau": log_inverse},
        link_derivatives={"mu": identity_derivative, "sigma": log_derivative, "nu": log_derivative, "tau": log_derivative},
    )
    from dataclasses import replace

    return replace(family, hessian_functions=_fisher_hessians(family))


def SEP4():
    """Skew Exponential Power Type 4 (R SEP4)."""
    return build_ad_family(
        family_class=SEP4Family,
        name="SEP4",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_sep4_log_pdf,
        type_="Continuous",
        p=_sep4_p,
        q=_sep4_q,
        r=_sep4_r,
        links={"mu": "identity", "sigma": "log", "nu": "log", "tau": "log"},
        link_functions={"mu": identity_link, "sigma": log_link, "nu": log_link, "tau": log_link},
        link_inverses={"mu": identity_inverse, "sigma": log_inverse, "nu": log_inverse, "tau": log_inverse},
        link_derivatives={"mu": identity_derivative, "sigma": log_derivative, "nu": log_derivative, "tau": log_derivative},
    )
    from dataclasses import replace

    return replace(family, hessian_functions=_fisher_hessians(family))


# ============================================================================
# SST - Skew Student t
# ============================================================================

@dataclass(frozen=True)
class SSTFamily(FamilyDefinition):
    """Skew Student t family."""
    pass


def _sst_log_pdf(y, mu, sigma, nu, tau):
    """SST log-PDF matching R SST.R dSST.

    Standardized skew-t: a location-scale reparameterization of ST3C.
      m1 = 2*sqrt(tau)*(nu^2-1) / ((tau-1)*beta(1/2, tau/2)*nu)
      m2 = tau*(nu^3+1/nu^3) / ((tau-2)*(nu+1/nu))
      s1 = sqrt(m2 - m1^2)
      mu1 = mu - sigma*m1/s1;  sigma1 = sigma/s1
      fy = dST3(y, mu1, sigma1, nu, tau)
    """
    import jax.numpy as jnp
    from jax.scipy.special import betaln, gammaln as _gl
    from jax.scipy.stats import t as _jt

    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.maximum(nu, eps)
    tau = jnp.maximum(tau, 2.0 + eps)

    # beta(1/2, tau/2) = exp(betaln(0.5, tau/2))
    beta_0_5_tau2 = jnp.exp(betaln(0.5, tau / 2.0))
    m1 = 2.0 * jnp.sqrt(tau) * (nu * nu - 1.0) / ((tau - 1.0) * beta_0_5_tau2 * nu)
    m2 = tau * (nu * nu * nu + 1.0 / (nu * nu * nu)) / ((tau - 2.0) * (nu + 1.0 / nu))
    s1 = jnp.sqrt(m2 - m1 * m1)
    s1 = jnp.maximum(s1, eps)

    mu1 = mu - sigma * m1 / s1
    sigma1 = sigma / s1

    # Call ST3C log-density
    from .distributions_b15 import _st3c_log_pdf
    return _st3c_log_pdf(y, mu1, sigma1, nu, tau)


def _sst_p(q, mu=0.0, sigma=1.0, nu=0.8, tau=7.0, lower_tail=True, log_p=False):
    """SST CDF matching R SST.R pSST (transformation of pST3C)."""
    import numpy as np
    import jax.numpy as jnp
    from .distributions_b15 import _st3c_p

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
        if si <= 0 or ni <= 0 or ti <= 2.0:
            out.append(np.nan)
            continue
        # Same transformation as in dSST
        from scipy.special import beta as _beta
        beta_0_5_t2 = _beta(0.5, ti / 2.0)
        m1 = 2.0 * np.sqrt(ti) * (ni * ni - 1.0) / ((ti - 1.0) * beta_0_5_t2 * ni)
        m2 = ti * (ni * ni * ni + 1.0 / (ni * ni * ni)) / ((ti - 2.0) * (ni + 1.0 / ni))
        s1 = np.sqrt(m2 - m1 * m1)
        mu1 = mui - si * m1 / s1
        sigma1 = si / s1
        out.append(float(np.asarray(_st3c_p(np.array([qi]), mu1, sigma1, ni, ti)).reshape(-1)[0]))

    cdf = np.array(out)
    if not lower_tail:
        cdf = 1.0 - cdf
    if log_p:
        cdf = np.log(cdf)
    return jnp.asarray(cdf, dtype=jnp.float64)


def _sst_q(p, mu=0.0, sigma=1.0, nu=0.8, tau=7.0, lower_tail=True, log_p=False):
    """SST quantile matching R SST.R qSST (transformation of qST3C)."""
    import numpy as np
    import jax.numpy as jnp
    from .distributions_b15 import _st3c_q

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
        if si <= 0 or ni <= 0 or ti <= 2.0:
            out.append(np.nan)
            continue
        from scipy.special import beta as _beta
        beta_0_5_t2 = _beta(0.5, ti / 2.0)
        m1 = 2.0 * np.sqrt(ti) * (ni * ni - 1.0) / ((ti - 1.0) * beta_0_5_t2 * ni)
        m2 = ti * (ni * ni * ni + 1.0 / (ni * ni * ni)) / ((ti - 2.0) * (ni + 1.0 / ni))
        s1 = np.sqrt(m2 - m1 * m1)
        mu1 = mui - si * m1 / s1
        sigma1 = si / s1
        q_raw = float(np.asarray(_st3c_q(np.array([pi]), mu1, sigma1, ni, ti)).reshape(-1)[0])
        out.append(float(q_raw))

    res = np.array(out)
    if np.ndim(p) == 0:
        return float(res[0])
    return jnp.asarray(res, dtype=jnp.float64)


def _sst_r(key, n, mu=0.0, sigma=1.0, nu=0.8, tau=7.0):
    """SST random generation via inverse-CDF sampling."""
    import jax.random as jr

    u = jr.uniform(key, shape=(n,), dtype=jnp.float64)
    return _sst_q(u, mu, sigma, nu, tau)


def SST():
    """Standardized Skew t distribution (R SST)."""
    return build_ad_family(
        family_class=SSTFamily,
        name="SST",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_sst_log_pdf,
        type_="Continuous",
        p=_sst_p,
        q=_sst_q,
        r=_sst_r,
        links={"mu": "identity", "sigma": "log", "nu": "log", "tau": "logshiftto2"},
        link_functions={"mu": identity_link, "sigma": log_link, "nu": log_link, "tau": log_link},
        link_inverses={"mu": identity_inverse, "sigma": log_inverse, "nu": log_inverse, "tau": log_inverse},
        link_derivatives={"mu": identity_derivative, "sigma": log_derivative, "nu": log_derivative, "tau": log_derivative},
    )


# ============================================================================
# ST3C - Skew t Type 3C
# ============================================================================

@dataclass(frozen=True)
class ST3CFamily(FamilyDefinition):
    """Skew t Type 3C family."""
    pass


def _st3c_log_pdf(y, mu, sigma, nu, tau):
    """ST3C log-PDF (R `st3_dst3`, Jones & Faddy skew-t).

    e = y - mu; z = e/sigma.
    loglik = dt(nu*z, df=tau)                       if e < 0
           = dt(e/(sigma*nu), df=tau)               if e >= 0
    (with normal density substituted when tau >= 1e6)
    + log(2*nu/(1+nu^2)) - log(sigma)
    """
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.maximum(nu, eps)
    # R's ST3C()$tau.valid is `all(tau > 0)` -- no floor at 1.  A `maximum(tau,
    # 1 + eps)` clamp here made the density *constant* in tau for tau < 1
    # (measured: tau = 0.3 / 0.5 / 0.7 all returned -1.2309075820904529,
    # while R returns -1.6432 / -1.4347 / -1.3248).  That is what the
    # scanner's `dpqr d` gate was reporting as rel 8.2e-02 on the grid
    # tau = linspace(0.5, 3).
    #
    # The floor now sits at R's own boundary only.  Note SST is different:
    # its m2 carries a (tau - 2) denominator, so R's dSST is NaN below 2 and
    # its 2 + eps clamp marks a real domain edge rather than narrowing one.
    tau = jnp.maximum(tau, eps)

    e = y - mu
    z = e / sigma

    # t density branch (tau < 1e6) vs normal branch (tau >= 1e6)
    use_t = tau < 1e6
    # e<0 branch
    l1_t = _jst.logpdf(nu * z, tau)
    l1_n = _jnorm.logpdf(nu * z)
    l1 = jnp.where(use_t, l1_t, l1_n)
    # e>=0 branch
    l2_t = _jst.logpdf(e / (sigma * nu), tau)
    l2_n = _jnorm.logpdf(e / (sigma * nu))
    l2 = jnp.where(use_t, l2_t, l2_n)

    loglik = jnp.where(e < 0, l1, l2)
    loglik = loglik + jnp.log(2.0 * nu / (1.0 + nu * nu)) - jnp.log(sigma)
    return loglik


def _st3c_p(q, mu=0.0, sigma=1.0, nu=1.0, tau=10.0, lower_tail=True, log_p=False):
    """ST3C CDF (R `st3_pst3`, Jones & Faddy skew-t)."""
    import numpy as _np
    from scipy.stats import t as _spt
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
        si = max(float(sigma_arr[i % len(sigma_arr)]), 1e-16)
        nui = max(float(nu_arr[i % len(nu_arr)]), 1e-16)
        # R allows tau < 1 (df < 1): ST3C.R's C routine has no floor, and
        # live R answers pST3C(-1, mu=0, sigma=1, nu=0.5, tau=0.5) = 0.6058544583.
        # Flooring at 1 silently turned df = 0.5 into df = 1 (measured
        # 0.699467 where R answers 0.715361 at the scanner probe point, D2.57).
        ti = max(float(tau_arr[i % len(tau_arr)]), 1e-12)
        e = qi - mui
        nu2 = nui * nui
        if e < 0:
            c = 2.0 * _spt.cdf(nui * e / si, ti)
        else:
            c = 1.0 + 2.0 * nu2 * (_spt.cdf(e / (si * nui), ti) - 0.5)
        out.append(float(c / (1.0 + nu2)))
    cdf = _np.array(out)
    if not lower_tail:
        cdf = 1.0 - cdf
    if log_p:
        cdf = _np.log(cdf)
    return jnp.asarray(cdf, dtype=jnp.float64)


def _st3c_q(p, mu=0.0, sigma=1.0, nu=1.0, tau=10.0, lower_tail=True, log_p=False):
    """ST3C quantile (R `st3_qst3`)."""
    import numpy as _np
    from scipy.stats import t as _spt
    p_arr = _np.atleast_1d(_np.asarray(p, dtype=_np.float64))
    if log_p:
        p_arr = _np.exp(p_arr)
    if not lower_tail:
        p_arr = 1.0 - p_arr
    p_arr = _np.clip(p_arr, 1e-16, 1.0 - 1e-16)
    mu_arr = _np.atleast_1d(_np.asarray(mu, dtype=_np.float64))
    sigma_arr = _np.atleast_1d(_np.asarray(sigma, dtype=_np.float64))
    nu_arr = _np.atleast_1d(_np.asarray(nu, dtype=_np.float64))
    tau_arr = _np.atleast_1d(_np.asarray(tau, dtype=_np.float64))
    n = max(len(p_arr), len(mu_arr), len(sigma_arr), len(nu_arr), len(tau_arr))
    out = []
    for i in range(n):
        pi = float(p_arr[i % len(p_arr)])
        mui = float(mu_arr[i % len(mu_arr)])
        si = max(float(sigma_arr[i % len(sigma_arr)]), 1e-16)
        nui = max(float(nu_arr[i % len(nu_arr)]), 1e-16)
        # R allows tau < 1 (df < 1): ST3C.R's C routine has no floor, and
        # live R answers pST3C(-1, mu=0, sigma=1, nu=0.5, tau=0.5) = 0.6058544583.
        # Flooring at 1 silently turned df = 0.5 into df = 1 (measured
        # 0.699467 where R answers 0.715361 at the scanner probe point, D2.57).
        ti = max(float(tau_arr[i % len(tau_arr)]), 1e-12)
        nu2 = nui * nui
        if pi < 1.0 / (1.0 + nu2):
            q = mui + (si / nui) * _spt.ppf(pi * (1.0 + nu2) / 2.0, ti)
        else:
            q = mui + (si * nui) * _spt.ppf((pi * (1.0 + nu2) - 1.0) / (2.0 * nu2) + 0.5, ti)
        out.append(float(q))
    q_arr = _np.array(out)
    return jnp.asarray(q_arr, dtype=jnp.float64)


def _st3c_r(key, n, mu=0.0, sigma=1.0, nu=1.0, tau=10.0):
    """ST3C random generation via inverse-CDF sampling."""
    u = jrandom.uniform(key, shape=(n,), dtype=jnp.float64)
    return _st3c_q(u, mu, sigma, nu, tau)


def ST3C():
    """Skew t Type 3C distribution."""
    return build_ad_family(
        family_class=ST3CFamily,
        name="ST3C",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_st3c_log_pdf,
        type_="Continuous",
        p=_st3c_p,
        q=_st3c_q,
        r=_st3c_r,
        # R's ST3C()$nu.link is "log", not "identity" -- verified against live R,
        # together with nu.valid = all(nu > 0) and nu.initial = 1.  The
        # neighbouring SEP1/SEP2 do use "identity" for nu, so this is a
        # per-family fact to look up rather than a pattern to assume.
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


# ============================================================================
# EGB2 - Exponential Generalized Beta Type 2
# ============================================================================

@dataclass(frozen=True)
class EGB2Family(FamilyDefinition):
    """Exponential Generalized Beta Type 2 family."""
    pass


def _egb2_log_pdf(y, mu, sigma, nu, tau):
    """EGB2 log-density.

    R's ``gamlss.dist::dEGB2`` uses ``z = (x - mu)/sigma`` on the **raw**
    response — an earlier revision of this module applied the same formula to
    ``log(y)``, which is a different distribution.  The kernel now lives in
    :mod:`omnilss.dpqr_batch_f` and is reused verbatim.
    """
    eps = jnp.finfo(jnp.float64).eps
    return dEGB2(
        y,
        mu,
        jnp.maximum(sigma, eps),
        jnp.maximum(nu, eps),
        jnp.maximum(tau, eps),
        log=True,
    )


def EGB2():
    """Exponential Generalized Beta Type 2 distribution.

    R ``EGB2``: ``z = (x-mu)/sigma`` with ``mu.link = "identity"``,
    ``sigma.link = nu.link = tau.link = "log"`` and ``nopar = 4``.  ``mu`` is a
    location parameter on the whole real line, hence the identity link.
    """
    return build_ad_family(
        family_class=EGB2Family,
        name="EGB2",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_egb2_log_pdf,
        type_="Continuous",
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
        d=dEGB2,
        p=pEGB2,
        q=qEGB2,
        r=rEGB2,
    )


# ============================================================================
# LQNO - Log Quantile Normal
# ============================================================================

@dataclass(frozen=True)
class LQNOFamily(FamilyDefinition):
    """Log Quantile Normal family."""
    pass


def _lqno_log_pdf(y, mu, sigma):
    """LQNO log-density.

    R's ``gamlss.dist::LQNO`` is a **two**-parameter family: a Normal with mean
    ``mu`` and a mean-dependent standard deviation ``sqrt(mu * (1 + sigma*mu))``
    (see ``R/gamlss.dist/R/LQNO.R``, identical in the installed 6.1.1).

    An earlier revision of this module shipped a three-parameter ``tanh``
    "log quantile normal" that had nothing to do with the R family; it is gone,
    and the kernel lives in :mod:`omnilss.dpqr_batch_f` next to its d/p/q/r.
    """
    eps = jnp.finfo(jnp.float64).eps
    return dLQNO(y, jnp.maximum(mu, eps), jnp.maximum(sigma, eps), log=True)


def LQNO():
    """Log Quantile Normal distribution.

    ``Y ~ Normal(mu, sqrt(mu * (1 + sigma * mu)))`` with ``mu > 0``, ``sigma > 0``
    and ``mu.link = sigma.link = "log"`` — the same signature as
    ``gamlss.dist::LQNO``.
    """
    return build_ad_family(
        family_class=LQNOFamily,
        name="LQNO",
        parameters=("mu", "sigma"),
        log_pdf_func=_lqno_log_pdf,
        type_="Continuous",
        links={"mu": "log", "sigma": "log"},
        link_functions={
            "mu": log_link,
            "sigma": log_link,
        },
        link_inverses={
            "mu": log_inverse,
            "sigma": log_inverse,
        },
        link_derivatives={
            "mu": log_derivative,
            "sigma": log_derivative,
        },
        d=dLQNO,
        p=pLQNO,
        q=qLQNO,
        r=rLQNO,
    )


# ============================================================================
# NBF - Negative Binomial Famoye
# ============================================================================

@dataclass(frozen=True)
class NBFFamily(FamilyDefinition):
    """Negative Binomial Famoye parameterization family."""
    pass


def _nbf_log_pdf(y, mu, sigma, nu):
    """R ``NBF`` log-PMF with ``sigma1 = sigma * mu**(nu - 2)``."""
    eps = jnp.finfo(jnp.float64).eps
    valid = (mu > 0.0) & (sigma > 0.0) & jnp.isfinite(nu)
    support = (y >= 0.0) & (y == jnp.floor(y))
    safe_y = jnp.where(support, y, 0.0)
    safe_mu = jnp.where(mu > 0.0, mu, 1.0)
    safe_sigma = jnp.where(sigma > 0.0, sigma, 1.0)
    sigma1 = safe_sigma * jnp.power(safe_mu, nu - 2.0)
    size = 1.0 / jnp.maximum(sigma1, eps)
    nb_log_mass = (
        gammaln(safe_y + size) - gammaln(size) - gammaln(safe_y + 1.0)
        + size * jnp.log(size / (size + safe_mu))
        + safe_y * jnp.log(safe_mu / (size + safe_mu))
    )
    poisson_log_mass = -safe_mu + safe_y * jnp.log(safe_mu) - gammaln(safe_y + 1.0)
    result = jnp.where(sigma1 < 1e-4, poisson_log_mass, nb_log_mass)
    result = jnp.where(support, result, -jnp.inf)
    return jnp.where(valid, result, jnp.nan)


def _nbf_parameters(mu, sigma, nu):
    """Return SciPy's ``(n, p)`` parameters for the R NBF parameterization."""
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    sigma1 = sigma * jnp.power(mu, nu - 2.0)
    n = 1.0 / sigma1
    p = n / (n + mu)
    return sigma1, n, p


def dNBF(x, mu=1.0, sigma=1.0, nu=2.0, log=False):
    """NBF PMF using the exact negative-binomial parameterization."""
    x, mu, sigma, nu = (jnp.asarray(v, dtype=jnp.float64) for v in (x, mu, sigma, nu))
    sigma1, n, p = _nbf_parameters(mu, sigma, nu)
    nb = _nbinom.logpmf(x, n, p) if log else _nbinom.pmf(x, n, p)
    pois = _poisson.logpmf(x, mu) if log else _poisson.pmf(x, mu)
    result = jnp.where(sigma1 < 1e-4, pois, nb)
    result = jnp.where((x >= 0) & (x == jnp.floor(x)), result, -jnp.inf if log else 0.0)
    return jnp.where((mu > 0) & (sigma > 0) & jnp.isfinite(nu), result, jnp.nan)


def pNBF(q, mu=1.0, sigma=1.0, nu=2.0, lower_tail=True, log_p=False):
    """NBF CDF/survival function with R-compatible tail and log options."""
    q, mu, sigma, nu = (jnp.asarray(v, dtype=jnp.float64) for v in (q, mu, sigma, nu))
    sigma1, n, p = _nbf_parameters(mu, sigma, nu)
    nb = _nbinom.cdf(q, n, p) if lower_tail else _nbinom.sf(q, n, p)
    pois = _poisson.cdf(q, mu) if lower_tail else _poisson.sf(q, mu)
    probability = jnp.where(sigma1 < 1e-4, pois, nb)
    result = jnp.log(probability) if log_p else probability
    valid = (mu > 0) & (sigma > 0) & jnp.isfinite(nu)
    return jnp.where(valid, result, jnp.nan)


def qNBF(probability, mu=1.0, sigma=1.0, nu=2.0, lower_tail=True, log_p=False):
    """NBF discrete quantile (smallest integer with CDF >= probability)."""
    probability, mu, sigma, nu = (jnp.asarray(v, dtype=jnp.float64) for v in (probability, mu, sigma, nu))
    sigma1, n, p = _nbf_parameters(mu, sigma, nu)
    probability = jnp.exp(probability) if log_p else probability
    probability = jnp.where(lower_tail, probability, 1.0 - probability)
    result = jnp.where(
        sigma1 < 1e-4,
        _poisson.ppf(probability, mu),
        _nbinom.ppf(probability, n, p),
    )
    # Count distributions have a finite lower support boundary.
    result = jnp.where(probability == 0.0, 0.0, result)
    result = jnp.where(probability == 1.0, jnp.inf, result)
    valid = ((mu > 0) & (sigma > 0) & jnp.isfinite(nu)
             & (probability >= 0) & (probability <= 1))
    return jnp.where(valid, result, jnp.nan)


def rNBF(key, n, mu=1.0, sigma=1.0, nu=2.0):
    """Generate NBF variates by inverse-transform sampling."""
    uniforms = jrandom.uniform(key, shape=(n,), dtype=jnp.float64)
    return qNBF(uniforms, mu, sigma, nu)


def NBF():
    """Negative Binomial Famoye distribution."""
    return build_ad_family(
        family_class=NBFFamily,
        name="NBF",
        parameters=("mu", "sigma", "nu"),
        log_pdf_func=_nbf_log_pdf,
        type_="Discrete",
        links={"mu": "log", "sigma": "log", "nu": "log"},
        link_functions={"mu": log_link, "sigma": log_link, "nu": log_link},
        link_inverses={"mu": log_inverse, "sigma": log_inverse, "nu": log_inverse},
        link_derivatives={"mu": log_derivative, "sigma": log_derivative, "nu": log_derivative},
        d=dNBF,
        p=pNBF,
        q=qNBF,
        r=rNBF,
    )


# ============================================================================
# ZINBF - Zero-Inflated Negative Binomial Famoye
# ============================================================================

@dataclass(frozen=True)
class ZINBFFamily(FamilyDefinition):
    """Zero-Inflated Negative Binomial Famoye family."""
    pass


def _zinbf_log_pdf(y, mu, sigma, nu, tau=0.1):
    """ZINBF log-PDF."""
    eps = jnp.finfo(jnp.float64).eps
    y = jnp.maximum(y, 0.0)
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    tau = jnp.clip(tau, eps, 1.0 - eps)
    
    # Base NBF log-PDF
    alpha = 1.0 / (sigma * jnp.power(mu, nu - 2.0))
    log_base = (
        gammaln(y + alpha) - gammaln(alpha) - gammaln(y + 1.0)
        + alpha * jnp.log(alpha / (alpha + mu))
        + y * jnp.log(mu / (alpha + mu))
    )
    
    # f(0) for NBF
    log_f0 = alpha * jnp.log(alpha / (alpha + mu))
    
    # Zero-inflated probabilities
    log_at0 = jnp.log(tau + (1.0 - tau) * jnp.exp(log_f0))
    log_cont = jnp.log1p(-tau) + log_base
    
    return jnp.where(y <= 0, log_at0, log_cont)


def dZINBF(x, mu=1.0, sigma=1.0, nu=2.0, tau=0.1, log=False):
    x, mu, sigma, nu, tau = jnp.broadcast_arrays(*[jnp.asarray(v, dtype=jnp.float64) for v in (x, mu, sigma, nu, tau)])
    alpha = 1.0 / sigma
    sigma1 = sigma * jnp.power(mu, nu - 2.0); alpha = 1.0 / sigma1
    base = gammaln(x + alpha) - gammaln(alpha) - gammaln(x + 1) + alpha*jnp.log(alpha/(alpha+mu)) + x*jnp.log(mu/(alpha+mu))
    base0 = alpha*jnp.log(alpha/(alpha+mu))
    value = jnp.where(x == 0, jnp.logaddexp(jnp.log(tau), jnp.log1p(-tau) + base0), jnp.log1p(-tau) + base)
    value = jnp.where((x >= 0) & (x == jnp.floor(x)), value, -jnp.inf)
    return value if log else jnp.exp(value)


def pZINBF(q, mu=1.0, sigma=1.0, nu=2.0, tau=0.1, lower_tail=True, log_p=False):
    q, mu, sigma, nu, tau = jnp.broadcast_arrays(*[jnp.asarray(v, dtype=jnp.float64) for v in (q, mu, sigma, nu, tau)])
    alpha = 1.0 / (sigma * jnp.power(mu, nu - 2.0))
    base = _nbinom.cdf(q, alpha, alpha/(alpha+mu))
    value = jnp.where(q < 0, 0.0, tau + (1.0 - tau) * base)
    value = jnp.where(q < 0, 0.0 if lower_tail else 1.0, value)
    value = jnp.where(lower_tail, value, 1.0 - value)
    return jnp.log(value) if log_p else value


def qZINBF(probability, mu=1.0, sigma=1.0, nu=2.0, tau=0.1, lower_tail=True, log_p=False):
    probability, mu, sigma, nu, tau = jnp.broadcast_arrays(*[jnp.asarray(v, dtype=jnp.float64) for v in (probability, mu, sigma, nu, tau)])
    p = jnp.exp(probability) if log_p else probability
    p = jnp.where(lower_tail, p, 1.0 - p)
    f0 = jnp.exp((1.0 / (sigma * jnp.power(mu, nu - 2.0))) * jnp.log(
        (1.0 / (sigma * jnp.power(mu, nu - 2.0))) /
        (1.0 / (sigma * jnp.power(mu, nu - 2.0)) + mu)))
    zero_cutoff = tau + (1.0 - tau) * f0
    base_p = jnp.clip((p - tau) / (1.0 - tau), 0.0, 1.0)
    alpha = 1.0 / (sigma * jnp.power(mu, nu - 2.0))
    return jnp.where(p <= zero_cutoff, 0.0, _nbinom.ppf(base_p, alpha, alpha/(alpha+mu)))


def rZINBF(key, n, mu=1.0, sigma=1.0, nu=2.0, tau=0.1):
    zero_key, base_key = jrandom.split(key)
    mu, sigma, nu, tau = [jnp.asarray(v, dtype=jnp.float64) for v in (mu, sigma, nu, tau)]
    shape = (n,) if all(v.ndim == 0 for v in (mu, sigma, nu, tau)) else jnp.broadcast_shapes(mu.shape, sigma.shape, nu.shape, tau.shape)
    return jnp.where(jrandom.bernoulli(zero_key, tau, shape=shape), 0.0, qNBF(jrandom.uniform(base_key, shape=shape), mu, sigma, nu))


def ZINBF():
    """Zero-Inflated Negative Binomial Famoye distribution."""
    return build_ad_family(
        family_class=ZINBFFamily,
        name="ZINBF",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_zinbf_log_pdf,
        type_="Discrete",
        links={"mu": "log", "sigma": "log", "nu": "log", "tau": "logit"},
        link_functions={
            "mu": log_link,
            "sigma": log_link,
            "nu": log_link,
            "tau": logit_link,
        },
        link_inverses={
            "mu": log_inverse,
            "sigma": log_inverse,
            "nu": log_inverse,
            "tau": logit_inverse,
        },
        link_derivatives={
            "mu": log_derivative,
            "sigma": log_derivative,
            "nu": log_derivative,
            "tau": logit_derivative,
        },
        d=dZINBF,
        p=pZINBF,
        q=qZINBF,
        r=rZINBF,
    )


# ============================================================================
# LOGSHASH - Log SHASH
# ============================================================================

@dataclass(frozen=True)
class LOGSHASHFamily(FamilyDefinition):
    """Log SHASH distribution family."""
    pass


def _logshash_log_pdf(y, mu, sigma, nu, tau):
    """LOGSHASH log-density.

    Delegates to dpqr_batch_f.dLOGSHASH so the AD gradients and the public d
    share one kernel.  R defines the family as the log-transform of SHASH
    (dLOGSHASH(x) = dSHASH(log(x))/x), and the family object's score slots
    are SHASH's slots at log(y) -- the Jacobian -log(y) is constant in the
    parameters, so those slots are exactly the gradients of this function.
    The previous implementation used a different (Jones-Pewsey style)
    sinh((asinh(z)-nu)/tau) parameterisation with no log(c) term --
    self-consistent enough to pass the AD deriv gate, but not R's
    distribution.
    """
    from .dpqr_batch_f import dLOGSHASH

    eps = jnp.finfo(jnp.float64).eps
    return dLOGSHASH(
        y, mu, jnp.maximum(sigma, eps), jnp.maximum(nu, eps), jnp.maximum(tau, eps), log=True
    )


def LOGSHASH():
    """Log SHASH distribution (R LOGSHASH).

    Links are R's defaults (gamlss.dist development source, 15-07-2025):
    mu.link="identity", sigma/nu/tau.link="log".  The previous wiring had
    mu on log and nu on identity -- both wrong.
    """
    from .dpqr_batch_f import dLOGSHASH, pLOGSHASH, qLOGSHASH, rLOGSHASH

    return build_ad_family(
        family_class=LOGSHASHFamily,
        name="LOGSHASH",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_logshash_log_pdf,
        type_="Continuous",
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
        d=dLOGSHASH,
        p=pLOGSHASH,
        q=qLOGSHASH,
        r=rLOGSHASH,
    )


# ============================================================================
# GPO - Generalized Poisson (alternative)
# ============================================================================

@dataclass(frozen=True)
class GPOFamily(FamilyDefinition):
    """Generalized Poisson (alternative parameterization) family."""
    pass


GPO_PROBABILITY_MAX_COUNT = 10000


def _gpo_log_pdf(y, mu, sigma):
    """R ``GPO`` log-PMF for positive ``mu`` and ``sigma``."""
    valid = (mu > 0.0) & (sigma > 0.0) & jnp.isfinite(mu) & jnp.isfinite(sigma)
    support = (y >= 0.0) & (y == jnp.floor(y))
    safe_y = jnp.where(support, y, 0.0)
    safe_mu = jnp.where(mu > 0.0, mu, 1.0)
    safe_sigma = jnp.where(sigma > 0.0, sigma, 1.0)
    denominator = 1.0 + safe_sigma * safe_mu
    log_mass = (
        safe_y * jnp.log(safe_mu / denominator)
        + (safe_y - 1.0) * jnp.log1p(safe_sigma * safe_y)
        - safe_mu * (1.0 + safe_sigma * safe_y) / denominator
        - gammaln(safe_y + 1.0)
    )
    poisson_log_mass = -safe_mu + safe_y * jnp.log(safe_mu) - gammaln(safe_y + 1.0)
    result = jnp.where(safe_sigma < 1e-4, poisson_log_mass, log_mass)
    result = jnp.where(support, result, -jnp.inf)
    return jnp.where(valid, result, jnp.nan)


def _gpo_log_mass_grid(mu, sigma, max_value=GPO_PROBABILITY_MAX_COUNT):
    support = jnp.arange(max_value + 1, dtype=jnp.float64)
    return support, _gpo_log_pdf(support, mu, sigma)


def _gpo_p(q, mu=1.0, sigma=1.0, lower_tail=True, log_p=False):
    q, mu, sigma = jnp.broadcast_arrays(*[
        jnp.asarray(value, dtype=jnp.float64) for value in (q, mu, sigma)
    ])
    flat_q, flat_mu, flat_sigma = (value.reshape(-1) for value in (q, mu, sigma))
    support = jnp.arange(GPO_PROBABILITY_MAX_COUNT + 1, dtype=jnp.float64)

    def one(qi, mui, sigmai):
        _, log_mass = _gpo_log_mass_grid(mui, sigmai)
        cdf = jnp.sum(jnp.where(support <= jnp.floor(qi), jnp.exp(log_mass), 0.0))
        cdf = jnp.where(qi < 0.0, 0.0, cdf)
        cdf = jnp.where((qi > GPO_PROBABILITY_MAX_COUNT) & jnp.isfinite(qi), jnp.nan, cdf)
        return jnp.where(jnp.isposinf(qi), 1.0, cdf)

    probability = jax.vmap(one)(flat_q, flat_mu, flat_sigma).reshape(q.shape)
    probability = jnp.where(lower_tail, probability, 1.0 - probability)
    result = jnp.log(probability) if log_p else probability
    valid = (mu > 0.0) & (sigma > 0.0) & jnp.isfinite(mu) & jnp.isfinite(sigma)
    return jnp.where(valid, result, jnp.nan)


def _gpo_q(p, mu=1.0, sigma=1.0, lower_tail=True, log_p=False,
           max_value=GPO_PROBABILITY_MAX_COUNT):
    p, mu, sigma = jnp.broadcast_arrays(*[
        jnp.asarray(value, dtype=jnp.float64) for value in (p, mu, sigma)
    ])
    probability = jnp.exp(p) if log_p else p
    probability = jnp.where(lower_tail, probability, 1.0 - probability)

    def one(pi, mui, sigmai):
        support, log_mass = _gpo_log_mass_grid(mui, sigmai, max_value)
        reached = jnp.cumsum(jnp.exp(log_mass)) >= pi
        result = jnp.where(jnp.any(reached), jnp.argmax(reached), jnp.nan).astype(jnp.float64)
        result = jnp.where(pi == 0.0, 0.0, result)
        result = jnp.where(pi >= 1.0, jnp.inf, result)
        return jnp.where((pi < 0.0) | (pi > 1.0) | jnp.isnan(pi), jnp.nan, result)

    result = jax.vmap(one)(probability.reshape(-1), mu.reshape(-1), sigma.reshape(-1)).reshape(p.shape)
    valid = (mu > 0.0) & (sigma > 0.0) & jnp.isfinite(mu) & jnp.isfinite(sigma)
    return jnp.where(valid, result, jnp.nan)


def _gpo_r(key, n, mu=1.0, sigma=1.0):
    return _gpo_q(jrandom.uniform(key, shape=(n,), dtype=jnp.float64), mu, sigma)


def GPO():
    """Generalized Poisson (alternative) distribution."""
    return build_ad_family(
        family_class=GPOFamily,
        name="GPO",
        parameters=("mu", "sigma"),
        log_pdf_func=_gpo_log_pdf,
        type_="Discrete",
        links={"mu": "log", "sigma": "log"},
        link_functions={"mu": log_link, "sigma": log_link},
        link_inverses={"mu": log_inverse, "sigma": log_inverse},
        link_derivatives={"mu": log_derivative, "sigma": log_derivative},
        p=_gpo_p,
        q=_gpo_q,
        r=_gpo_r,
    )
