from __future__ import annotations

# Enable float64 precision for numerical accuracy

"""Batch 14: Skew distribution variants and SHASH variants."""
from dataclasses import dataclass
import jax.numpy as jnp
from jax.scipy.special import gammaln

from .ad import build_ad_family
from .families import FamilyDefinition
from .links import (
    log_link, log_inverse, log_derivative,
    identity_link, identity_inverse, identity_derivative
)


# ============================================================================
# SHASHo - SHASH Original Parameterization
# ============================================================================

@dataclass(frozen=True)
class SHASHoFamily(FamilyDefinition):
    """SHASH Original parameterization family."""
    pass


def _shasho_log_pdf(y, mu, sigma, nu, tau):
    """SHASH Original log-PDF."""
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    tau = jnp.maximum(tau, eps)
    
    z = (y - mu) / sigma
    r = jnp.sinh((jnp.arcsinh(z) - nu) / tau)
    
    return (
        -jnp.log(sigma) - jnp.log(tau)
        - 0.5 * jnp.log(2.0 * jnp.pi)
        - 0.5 * jnp.log1p(z * z)
        - 0.5 * r * r
    )


def SHASHo():
    """SHASH with original parameterization."""
    return build_ad_family(
        family_class=SHASHoFamily,
        name="SHASHo",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_shasho_log_pdf,
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
# SHASHo2 - SHASH Original 2
# ============================================================================

@dataclass(frozen=True)
class SHASHo2Family(FamilyDefinition):
    """SHASH Original 2 parameterization family."""
    pass


def _shasho2_log_pdf(y, mu, sigma, nu, tau):
    """SHASH Original 2 log-PDF with different parameterization."""
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    tau = jnp.maximum(tau, eps)
    
    # Alternative transformation
    z = (y - mu) / sigma
    asinh_z = jnp.arcsinh(z)
    r = jnp.sinh(tau * asinh_z + nu)
    
    return (
        -jnp.log(sigma) + jnp.log(tau)
        - 0.5 * jnp.log(2.0 * jnp.pi)
        - 0.5 * jnp.log1p(z * z)
        - 0.5 * r * r
    )


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


def JSUo():
    """Johnson SU with original parameterization."""
    return build_ad_family(
        family_class=JSUoFamily,
        name="JSUo",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_jsuo_log_pdf,
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


def ST5():
    """Skew t Type 5 (Jones & Faddy) distribution."""
    return build_ad_family(
        family_class=ST5Family,
        name="ST5",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_st5_log_pdf,
        type_="Continuous",
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
