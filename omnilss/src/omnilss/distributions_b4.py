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
    # R routing for dBEINF is by *exact* equality (BEINF.R): 0 and 1 carry the
    # boundary masses, the open interval carries the beta density, and
    # "fy <- ifelse(x < 0 | x > 1, 0, fy)" zeroes everything outside.  Testing
    # y <= 0 / y >= 1 instead sent every x > 1 to the tau mass at 1 (measured:
    # dBEINF(2) came out tau/(1+nu+tau) where R answers 0), which is how the
    # scanner probe grid -- 0..7 for a [0,1] family -- produced its
    # "d: max rel 4.286e-01".
    result = jnp.where(y == 0.0, log_at0, jnp.where(y == 1.0, log_at1, log_cont))
    return jnp.where((y < 0.0) | (y > 1.0), -jnp.inf, result)


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
    # R (BEINF0.R): x == 0 carries the mass, 0 < x < 1 the beta density, and
    # "fy <- ifelse(x < 0 | x >= 1, 0, fy)" -- x >= 1 is zero, unlike BEINF
    # where x = 1 is a mass point.  The old y <= 0 routing ran x >= 1 through
    # the beta density clipped to 1-eps, measured 1.7e+11 at x = 1.
    result = jnp.where(y == 0.0, log_at0, log_cont)
    return jnp.where((y < 0.0) | (y >= 1.0), -jnp.inf, result)


def _beinf0_p(q, mu=0.5, sigma=0.1, nu=0.1, lower_tail=True, log_p=False):
    """R BEINF0.R pBEINF0: cdf = (nu + pbeta(q,a,b))/(1+nu) for q>0, 0 for q<=0."""
    import numpy as _np
    import jax.numpy as jnp
    from scipy.stats import beta as _sbeta

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
        a = mui * (1.0 - si * si) / (si * si)
        b = a * (1.0 - mui) / mui
        if qi > 0.0:
            c = (nui + _sbeta.cdf(qi, a, b)) / (1.0 + nui)
        elif qi == 0.0:
            c = nui / (1.0 + nui)
        else:
            c = 0.0
        out.append(float(min(max(c, 0.0), 1.0)))
    cdf = _np.array(out)
    if not lower_tail:
        cdf = 1.0 - cdf
    if log_p:
        cdf = _np.log(cdf)
    return jnp.asarray(cdf, dtype=jnp.float64)


def _beinf0_q(p, mu=0.5, sigma=0.1, nu=0.1, lower_tail=True, log_p=False):
    """R BEINF0.R qBEINF0: q=0 if p<=nu/(1+nu), else qbeta((p-nu/(1+nu))/(1/(1+nu)),a,b)."""
    import numpy as _np
    import jax.numpy as jnp
    from scipy.stats import beta as _sbeta

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
        a = mui * (1.0 - si * si) / (si * si)
        b = a * (1.0 - mui) / mui
        thr = nui / (1.0 + nui)
        if pi <= thr:
            out.append(0.0)
        else:
            pp = (pi - thr) / (1.0 / (1.0 + nui))
            out.append(float(_sbeta.ppf(pp, a, b)))
    res = _np.array(out)
    return jnp.asarray(res, dtype=jnp.float64)


def _beinf0_r(key, n, mu=0.5, sigma=0.1, nu=0.1):
    import jax.numpy as _jnp
    import jax.random as _jr

    u = _jr.uniform(key, shape=(n,), dtype=_jnp.float64)
    return _beinf0_q(u, mu, sigma, nu)


def BEINF0() -> BetaInflated0Family:
    return build_ad_family(
        family_class=BetaInflated0Family,
        name="BEINF0",
        parameters=("mu", "sigma", "nu"),
        log_pdf_func=_beinf0_log_pdf,
        type_="Mixed",
        p=_beinf0_p,
        q=_beinf0_q,
        r=_beinf0_r,
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
    # R (BEINF1.R): x == 1 carries the mass, 0 < x < 1 the beta density, and
    # "fy <- ifelse(x <= 0 | x > 1, 0, fy)" zeroes x = 0 and everything
    # outside -- note this is a *different* boundary rule from BEINF, where
    # x = 0 is a mass point (the D2.19 lesson: same family name, own rules).
    result = jnp.where(y == 1.0, log_at1, log_cont)
    return jnp.where((y <= 0.0) | (y > 1.0), -jnp.inf, result)


def _beinf1_p(q, mu=0.5, sigma=0.1, nu=0.1, lower_tail=True, log_p=False):
    """R BEINF1.R pBEINF1: cdf = (pbeta(q,a,b))/(1+nu) for q<1, 1 for q>=1."""
    import numpy as _np
    import jax.numpy as jnp
    from scipy.stats import beta as _sbeta

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
        a = mui * (1.0 - si * si) / (si * si)
        b = a * (1.0 - mui) / mui
        if qi >= 1.0:
            c = 1.0
        elif qi > 0.0:
            c = _sbeta.cdf(qi, a, b) / (1.0 + nui)
        else:
            c = 0.0
        out.append(float(min(max(c, 0.0), 1.0)))
    cdf = _np.array(out)
    if not lower_tail:
        cdf = 1.0 - cdf
    if log_p:
        cdf = _np.log(cdf)
    return jnp.asarray(cdf, dtype=jnp.float64)


def _beinf1_q(p, mu=0.5, sigma=0.1, nu=0.1, lower_tail=True, log_p=False):
    """R BEINF1.R qBEINF1."""
    import numpy as _np
    import jax.numpy as jnp
    from scipy.stats import beta as _sbeta

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
        a = mui * (1.0 - si * si) / (si * si)
        b = a * (1.0 - mui) / mui
        thr = 1.0 / (1.0 + nui)
        if pi >= thr:
            out.append(1.0)
        else:
            out.append(float(_sbeta.ppf(pi * (1.0 + nui), a, b)))
    res = _np.array(out)
    return jnp.asarray(res, dtype=jnp.float64)


def _beinf1_r(key, n, mu=0.5, sigma=0.1, nu=0.1):
    import jax.numpy as _jnp
    import jax.random as _jr

    u = _jr.uniform(key, shape=(n,), dtype=_jnp.float64)
    return _beinf1_q(u, mu, sigma, nu)


def BEINF1() -> BetaInflated1Family:
    return build_ad_family(
        family_class=BetaInflated1Family,
        name="BEINF1",
        parameters=("mu", "sigma", "nu"),
        log_pdf_func=_beinf1_log_pdf,
        type_="Mixed",
        p=_beinf1_p,
        q=_beinf1_q,
        r=_beinf1_r,
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
    # R (BEZI.r): x == 0 carries the mass, 0 < x < 1 the continuous part, and
    # "fy <- ifelse(x < 0 | x >= 1, 0, fy)" zeroes x >= 1 and x < 0.  The old
    # y <= 0 routing sent every x > 1 through the beta density (clipped to
    # 1-eps inside _beta_log_pdf), measured at 3121.8 for x = 2.41 at
    # mu = 0.39, sigma = 1.21 -- the scanner 3.122e+03.
    result = jnp.where(y == 0.0, log_at0, log_cont)
    return jnp.where((y < 0.0) | (y >= 1.0), -jnp.inf, result)


def _bezi_p(q, mu=0.5, sigma=1.0, nu=0.1, lower_tail=True, log_p=False):
    """R BEZI.R pBEZI: cdf = nu + (1-nu)*pbeta(q,a,b) for 0<q<1, 1 for q>=1."""
    import numpy as _np
    import jax.numpy as jnp
    from scipy.stats import beta as _sbeta

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
        a = mui * si
        b = (1.0 - mui) * si
        if qi >= 1.0:
            c = 1.0
        elif qi > 0.0:
            c = nui + (1.0 - nui) * _sbeta.cdf(qi, a, b)
        elif qi == 0.0:
            c = nui
        else:
            c = 0.0
        out.append(float(min(max(c, 0.0), 1.0)))
    cdf = _np.array(out)
    if not lower_tail:
        cdf = 1.0 - cdf
    if log_p:
        cdf = _np.log(cdf)
    return jnp.asarray(cdf, dtype=jnp.float64)


def _bezi_q(p, mu=0.5, sigma=1.0, nu=0.1, lower_tail=True, log_p=False):
    """R BEZI.R qBEZI: q=0 if nu>=p, else qbeta((p-nu)/(1-nu),a,b)."""
    import numpy as _np
    import jax.numpy as jnp
    from scipy.stats import beta as _sbeta

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
        a = mui * si
        b = (1.0 - mui) * si
        if nui >= pi:
            out.append(0.0)
        else:
            out.append(float(_sbeta.ppf((pi - nui) / (1.0 - nui), a, b)))
    res = _np.array(out)
    return jnp.asarray(res, dtype=jnp.float64)


def _bezi_r(key, n, mu=0.5, sigma=1.0, nu=0.1):
    import jax.numpy as _jnp
    import jax.random as _jr

    u = _jr.uniform(key, shape=(n,), dtype=_jnp.float64)
    return _bezi_q(u, mu, sigma, nu)


def BEZI() -> BetaZeroInflatedFamily:
    return build_ad_family(
        family_class=BetaZeroInflatedFamily,
        name="BEZI",
        parameters=("mu", "sigma", "nu"),
        log_pdf_func=_bezi_log_pdf,
        type_="Mixed",
        p=_bezi_p,
        q=_bezi_q,
        r=_bezi_r,
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
    # R (BEOI.r): x == 1 carries the mass, 0 < x < 1 the continuous part, and
    # "fy <- ifelse(x <= 0 | x > 1, 0, fy)" -- x = 0 is zero mass here, unlike
    # BEINF where it is a mass point.
    result = jnp.where(y == 1.0, log_at1, log_cont)
    return jnp.where((y <= 0.0) | (y > 1.0), -jnp.inf, result)


def _beoi_p(q, mu=0.5, sigma=1.0, nu=0.1, lower_tail=True, log_p=False):
    """R BEOI.R pBEOI: cdf = (1-nu)*pbeta(q,a,b) for 0<q<1, 1 for q>=1."""
    import numpy as _np
    import jax.numpy as jnp
    from scipy.stats import beta as _sbeta

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
        a = mui * si
        b = (1.0 - mui) * si
        if qi >= 1.0:
            c = 1.0
        elif qi > 0.0:
            c = (1.0 - nui) * _sbeta.cdf(qi, a, b)
        else:
            c = 0.0
        out.append(float(min(max(c, 0.0), 1.0)))
    cdf = _np.array(out)
    if not lower_tail:
        cdf = 1.0 - cdf
    if log_p:
        cdf = _np.log(cdf)
    return jnp.asarray(cdf, dtype=jnp.float64)


def _beoi_q(p, mu=0.5, sigma=1.0, nu=0.1, lower_tail=True, log_p=False):
    """R BEOI.R qBEOI: q=1 if p>=(1-nu), else qbeta(p/(1-nu),a,b)."""
    import numpy as _np
    import jax.numpy as jnp
    from scipy.stats import beta as _sbeta

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
        a = mui * si
        b = (1.0 - mui) * si
        if pi >= 1.0 - nui:
            out.append(1.0)
        else:
            out.append(float(_sbeta.ppf(pi / (1.0 - nui), a, b)))
    res = _np.array(out)
    return jnp.asarray(res, dtype=jnp.float64)


def _beoi_r(key, n, mu=0.5, sigma=1.0, nu=0.1):
    import jax.numpy as _jnp
    import jax.random as _jr

    u = _jr.uniform(key, shape=(n,), dtype=_jnp.float64)
    return _beoi_q(u, mu, sigma, nu)


def BEOI() -> BetaOneInflatedFamily:
    return build_ad_family(
        family_class=BetaOneInflatedFamily,
        name="BEOI",
        parameters=("mu", "sigma", "nu"),
        log_pdf_func=_beoi_log_pdf,
        type_="Mixed",
        p=_beoi_p,
        q=_beoi_q,
        r=_beoi_r,
        links={"mu": "logit", "sigma": "log", "nu": "logit"},
        link_functions={"mu": logit_link, "sigma": log_link, "nu": logit_link},
        link_inverses={"mu": logit_inverse, "sigma": log_inverse, "nu": logit_inverse},
        link_derivatives={"mu": logit_derivative, "sigma": log_derivative, "nu": logit_derivative},
    )
