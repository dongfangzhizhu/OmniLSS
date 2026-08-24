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
    """BCPE Original log-PDF."""
    eps = jnp.finfo(jnp.float64).eps
    y = jnp.maximum(y, eps)
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    tau = jnp.clip(tau, 0.1, 10.0)
    
    # Box-Cox transformation
    z = jnp.where(
        jnp.abs(nu) > eps,
        (jnp.power(y, nu) - 1.0) / nu,
        jnp.log(y)
    )
    
    z_std = (z - mu) / sigma
    
    # Power exponential
    log_pe = (
        jnp.log(tau) - jnp.log(2.0 * sigma) - gammaln(1.0 / tau)
        - jnp.power(jnp.abs(z_std), tau)
    )
    
    # Jacobian
    log_jacobian = (nu - 1.0) * jnp.log(y)
    
    return log_pe + log_jacobian


def BCPEo():
    """Box-Cox Power Exponential with original parameterization."""
    return build_ad_family(
        family_class=BCPEoFamily,
        name="BCPEo",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_bcpeo_log_pdf,
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
# BCTo - Box-Cox t Original
# ============================================================================

@dataclass(frozen=True)
class BCToFamily(FamilyDefinition):
    """Box-Cox t Original family."""
    pass


def _bcto_log_pdf(y, mu, sigma, nu, tau):
    """BCT Original log-PDF."""
    eps = jnp.finfo(jnp.float64).eps
    y = jnp.maximum(y, eps)
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    tau = jnp.maximum(tau, 2.0 + eps)
    
    # Box-Cox transformation
    z = jnp.where(
        jnp.abs(nu) > eps,
        (jnp.power(y, nu) - 1.0) / nu,
        jnp.log(y)
    )
    
    z_std = (z - mu) / sigma
    
    # t distribution
    log_t = (
        gammaln((tau + 1.0) / 2.0) - gammaln(tau / 2.0)
        - 0.5 * jnp.log(tau * jnp.pi) - jnp.log(sigma)
        - ((tau + 1.0) / 2.0) * jnp.log1p(z_std * z_std / tau)
    )
    
    # Jacobian
    log_jacobian = (nu - 1.0) * jnp.log(y)
    
    return log_t + log_jacobian


def BCTo():
    """Box-Cox t with original parameterization."""
    return build_ad_family(
        family_class=BCToFamily,
        name="BCTo",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_bcto_log_pdf,
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
