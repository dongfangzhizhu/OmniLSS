"""GAMLSS distributions: Batch 2 extensions using JAX AD.

This batch adds:
- NO2  (Normal parameterized by variance, sigma^2)
- LOGNO2 (Log-Normal parameterized by log-mean and sd)
- PE   (Power Exponential, 3-parameter)
- SIMPLEX (Simplex distribution on (0,1))
- exGAUS (Exponentially-modified Gaussian, 3-parameter)
"""

from __future__ import annotations

# Enable float64 precision for numerical accuracy
import jax
jax.config.update("jax_enable_x64", True)


from dataclasses import dataclass
import math

import jax.numpy as jnp
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
    logit_derivative,
    logit_inverse,
    logit_link,
)
from .dpqr_functions import dPE, pPE, qPE, rPE

# ------------------------------------------------------------------
# 1. NO2 — Normal parameterized by variance σ²
#    R: dNO2(x, mu, sigma) uses sd = sqrt(sigma)
# ------------------------------------------------------------------

@dataclass(frozen=True)
class Normal2Family(FamilyDefinition):
    """Normal distribution parameterized by variance (`NO2`)."""


def _no2_log_pdf(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    # sigma here is the variance (σ²), so sd = sqrt(sigma)
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    sd = jnp.sqrt(sigma)
    return -0.5 * jnp.log(2.0 * math.pi) - jnp.log(sd) - 0.5 * jnp.square((y - mu) / sd)


def NO2() -> Normal2Family:
    return build_ad_family(
        family_class=Normal2Family,
        name="NO2",
        parameters=("mu", "sigma"),
        log_pdf_func=_no2_log_pdf,
        type_="Continuous",
        links={"mu": "identity", "sigma": "log"},
        link_functions={"mu": identity_link, "sigma": log_link},
        link_inverses={"mu": identity_inverse, "sigma": log_inverse},
        link_derivatives={"mu": identity_derivative, "sigma": log_derivative},
    )


# ------------------------------------------------------------------
# 2. LOGNO2 — Log-Normal parameterized by log-mean (mu) and sd (sigma)
#    R: loglik <- dnorm(log(x), mean=log(mu), sd=sigma) - log(x)
# ------------------------------------------------------------------

@dataclass(frozen=True)
class LogNormal2Family(FamilyDefinition):
    """Log-Normal distribution (mu = mean of log Y) (`LOGNO2`)."""


def _logno2_log_pdf(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    eps = jnp.finfo(jnp.float64).eps
    y = jnp.maximum(y, eps)
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    log_y = jnp.log(y)
    log_mu = jnp.log(mu)
    return norm.logpdf(log_y, loc=log_mu, scale=sigma) - log_y


def LOGNO2() -> LogNormal2Family:
    """Log-Normal distribution (mu = mean of log Y) with expected Hessian.
    
    Uses JAX auto-differentiation for score functions (first derivatives)
    but uses expected information (Fisher information) for hessians to ensure
    numerical stability, following the R GAMLSS implementation.
    
    Parameterization:
    - log(Y) ~ N(log(mu), sigma)
    - E[log(Y)] = log(mu)
    - Var[log(Y)] = sigma^2
    
    R source reference:
    - file: `gamlss.dist/R/LOGNO2.R`
    - function: `LOGNO2()`
    """
    # Build family with AD for scores
    family = build_ad_family(
        family_class=LogNormal2Family,
        name="LOGNO2",
        parameters=("mu", "sigma"),
        log_pdf_func=_logno2_log_pdf,
        type_="Continuous",
        links={"mu": "log", "sigma": "log"},
        link_functions={"mu": log_link, "sigma": log_link},
        link_inverses={"mu": log_inverse, "sigma": log_inverse},
        link_derivatives={"mu": log_derivative, "sigma": log_derivative},
    )
    
    # Override hessians with expected information (Fisher information)
    # For LOGNO2: log(Y) ~ N(log(mu), sigma)
    # E[∂²log L/∂mu²] = -1/(mu² σ²)
    # E[∂²log L/∂σ²] = -2/σ²
    
    def d2ldm2_expected(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """Expected second derivative wrt mu (Fisher information)."""
        mu = jnp.asarray(mu, dtype=jnp.float64)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        eps = jnp.finfo(jnp.float64).eps
        mu = jnp.maximum(mu, eps)
        sigma = jnp.maximum(sigma, eps)
        return -1.0 / (jnp.square(mu) * jnp.square(sigma))
    
    def d2lds2_expected(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """Expected second derivative wrt sigma (Fisher information)."""
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        eps = jnp.finfo(jnp.float64).eps
        sigma = jnp.maximum(sigma, eps)
        return -2.0 / jnp.square(sigma)
    
    # Replace hessian functions with expected versions
    hessian_functions = {
        "mu": d2ldm2_expected,
        "sigma": d2lds2_expected,
    }
    
    # Create new family with updated hessians
    return LogNormal2Family(
        name=family.name,
        parameters=family.parameters,
        g_dev_inc=family.g_dev_inc,
        type=family.type,
        links=family.links,
        link_functions=family.link_functions,
        link_inverses=family.link_inverses,
        link_derivatives=family.link_derivatives,
        score_functions=family.score_functions,  # Keep AD scores
        hessian_functions=hessian_functions,  # Use expected hessians
        d=family.d,  # Keep d from build_ad_family
        p=family.p,  # Keep p from build_ad_family
        q=family.q,  # Keep q from build_ad_family
        r=family.r,  # Keep r from build_ad_family
    )


# ------------------------------------------------------------------
# 3. PE — Power Exponential (3-parameter, symmetric)
#    R: -log(sigma) + log(nu) - log.c - 0.5*|z/c|^nu - (1+1/nu)*log(2) - lgamma(1/nu)
#    where log.c = 0.5*(-(2/nu)*log(2) + lgamma(1/nu) - lgamma(3/nu))
# ------------------------------------------------------------------

@dataclass(frozen=True)
class PowerExponentialFamily(FamilyDefinition):
    """Power Exponential (Generalized Normal) distribution (`PE`)."""


def _pe_log_pdf(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, nu: jnp.ndarray) -> jnp.ndarray:
    from jax.scipy.special import gammaln
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.maximum(nu, eps)
    log_c = 0.5 * (-(2.0 / nu) * jnp.log(2.0) + gammaln(1.0 / nu) - gammaln(3.0 / nu))
    c = jnp.exp(log_c)
    z = (y - mu) / sigma
    return (
        -jnp.log(sigma)
        + jnp.log(nu)
        - log_c
        - 0.5 * jnp.power(jnp.abs(z / c), nu)
        - (1.0 + 1.0 / nu) * jnp.log(2.0)
        - gammaln(1.0 / nu)
    )


def PE() -> PowerExponentialFamily:
    family = build_ad_family(
        family_class=PowerExponentialFamily,
        name="PE",
        parameters=("mu", "sigma", "nu"),
        log_pdf_func=_pe_log_pdf,
        type_="Continuous",
        links={"mu": "identity", "sigma": "log", "nu": "log"},
        link_functions={"mu": identity_link, "sigma": log_link, "nu": log_link},
        link_inverses={"mu": identity_inverse, "sigma": log_inverse, "nu": log_inverse},
        link_derivatives={"mu": identity_derivative, "sigma": log_derivative, "nu": log_derivative},
        d=dPE,
        p=pPE,
        q=qPE,
        r=rPE,
    )

    eps = jnp.finfo(jnp.float64).eps

    def _safe_terms(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, nu: jnp.ndarray):
        from jax.scipy.special import digamma, gammaln

        y = jnp.asarray(y, dtype=jnp.float64)
        mu = jnp.asarray(mu, dtype=jnp.float64)
        sigma = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float64), eps)
        nu = jnp.maximum(jnp.asarray(nu, dtype=jnp.float64), eps)
        log_c = 0.5 * (-(2.0 / nu) * jnp.log(2.0) + gammaln(1.0 / nu) - gammaln(3.0 / nu))
        c = jnp.exp(log_c)
        z = (y - mu) / sigma
        abs_z = jnp.maximum(jnp.abs(z), eps)
        abs_ratio = jnp.maximum(jnp.abs(z / c), eps)
        dlogc_dv = (1.0 / (2.0 * jnp.square(nu))) * (
            2.0 * jnp.log(2.0) - digamma(1.0 / nu) + 3.0 * digamma(3.0 / nu)
        )
        return sigma, nu, z, abs_z, abs_ratio, dlogc_dv

    def dldm(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, nu: jnp.ndarray) -> jnp.ndarray:
        sigma, nu, z, abs_z, abs_ratio, _ = _safe_terms(y, mu, sigma, nu)
        return ((jnp.sign(z) * nu) / (2.0 * sigma * abs_z)) * jnp.power(abs_ratio, nu)

    def d2ldm2(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, nu: jnp.ndarray) -> jnp.ndarray:
        from jax.scipy.special import gammaln

        sigma, nu, _, _, _, _ = _safe_terms(y, mu, sigma, nu)
        dldm_val = dldm(y, mu, sigma, nu)
        gamma_term = jnp.exp(gammaln(2.0 - (1.0 / nu)) + gammaln(3.0 / nu) - 2.0 * gammaln(1.0 / nu))
        hess = -(jnp.square(nu) * gamma_term) / jnp.square(sigma)
        hess = jnp.where(nu < 1.05, -jnp.square(dldm_val), hess)
        return jnp.where(hess < -1e-15, hess, -1e-15)

    def dldd(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, nu: jnp.ndarray) -> jnp.ndarray:
        sigma, nu, _, _, abs_ratio, _ = _safe_terms(y, mu, sigma, nu)
        return (((nu / 2.0) * jnp.power(abs_ratio, nu)) - 1.0) / sigma

    def d2ldd2(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, nu: jnp.ndarray) -> jnp.ndarray:
        sigma, nu, _, _, _, _ = _safe_terms(y, mu, sigma, nu)
        hess = -nu / jnp.square(sigma)
        return jnp.where(hess < -1e-15, hess, -1e-15)

    def dldv(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, nu: jnp.ndarray) -> jnp.ndarray:
        from jax.scipy.special import digamma

        _, nu, _, _, abs_ratio, dlogc_dv = _safe_terms(y, mu, sigma, nu)
        pow_term = jnp.power(abs_ratio, nu)
        score = (1.0 / nu) - 0.5 * (jnp.log(abs_ratio) * pow_term)
        score = score + jnp.log(2.0) / jnp.square(nu) + digamma(1.0 / nu) / jnp.square(nu)
        score = score + (-1.0 + (nu / 2.0) * pow_term) * dlogc_dv
        return score

    def d2ldv2(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, nu: jnp.ndarray) -> jnp.ndarray:
        from jax.scipy.special import digamma, polygamma

        _, nu, _, _, _, dlogc_dv = _safe_terms(y, mu, sigma, nu)
        p = (1.0 + nu) / nu
        part1 = p * polygamma(1, p) + 2.0 * jnp.square(digamma(p))
        part2 = digamma(p) * (jnp.log(2.0) + 3.0 - 3.0 * digamma(3.0 / nu) - nu)
        part3 = -3.0 * digamma(3.0 / nu) * (1.0 + jnp.log(2.0))
        part4 = -(nu + jnp.log(2.0)) * jnp.log(2.0)
        part5 = -nu + jnp.power(nu, 4) * jnp.square(dlogc_dv)
        hess = -(part1 + part2 + part3 + part4 + part5) / jnp.power(nu, 3)
        return jnp.where(hess < -1e-15, hess, -1e-15)

    return PowerExponentialFamily(
        name=family.name,
        parameters=family.parameters,
        g_dev_inc=family.g_dev_inc,
        type=family.type,
        links=family.links,
        link_functions=family.link_functions,
        link_inverses=family.link_inverses,
        link_derivatives=family.link_derivatives,
        score_functions={"mu": dldm, "sigma": dldd, "nu": dldv},
        hessian_functions={"mu": d2ldm2, "sigma": d2ldd2, "nu": d2ldv2},
        d=family.d,  # Keep d from build_ad_family
        p=family.p,  # Keep p from build_ad_family
        q=family.q,  # Keep q from build_ad_family
        r=family.r,  # Keep r from build_ad_family
    )


# ------------------------------------------------------------------
# 4. SIMPLEX — Simplex distribution on (0,1)
#    R: logpdf = -((x-mu)/(mu*(1-mu)))^2/(2*x*(1-x)*sigma^2)
#                - (log(2pi*sigma^2) + 3*(log(x)+log(1-x)))/2
# ------------------------------------------------------------------

@dataclass(frozen=True)
class SimplexFamily(FamilyDefinition):
    """Simplex distribution on (0,1) (`SIMPLEX`)."""


def _simplex_log_pdf(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    eps = jnp.finfo(jnp.float64).eps
    y = jnp.clip(y, eps, 1.0 - eps)
    mu = jnp.clip(mu, eps, 1.0 - eps)
    sigma = jnp.maximum(sigma, eps)
    d = (y - mu) / (mu * (1.0 - mu))
    return (
        -jnp.square(d) / (2.0 * y * (1.0 - y) * jnp.square(sigma))
        - 0.5 * (jnp.log(2.0 * math.pi * jnp.square(sigma)) + 3.0 * (jnp.log(y) + jnp.log(1.0 - y)))
    )


def SIMPLEX() -> SimplexFamily:
    return build_ad_family(
        family_class=SimplexFamily,
        name="SIMPLEX",
        parameters=("mu", "sigma"),
        log_pdf_func=_simplex_log_pdf,
        type_="Continuous",
        links={"mu": "logit", "sigma": "log"},
        link_functions={"mu": logit_link, "sigma": log_link},
        link_inverses={"mu": logit_inverse, "sigma": log_inverse},
        link_derivatives={"mu": logit_derivative, "sigma": log_derivative},
    )


# ------------------------------------------------------------------
# 5. exGAUS — Exponentially-modified Gaussian (3-parameter)
#    R: logfy = ifelse(nu > 0.05*sigma,
#                  -log(nu) - (z + sigma^2/(2*nu))/nu + log(pnorm(z/sigma)),
#                  dnorm(x, mu, sigma, log=TRUE))
#    where z = x - mu - sigma^2/nu
# ------------------------------------------------------------------

@dataclass(frozen=True)
class ExGaussianFamily(FamilyDefinition):
    """Exponentially-modified Gaussian distribution (`exGAUS`)."""


def _exgaus_log_pdf(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, nu: jnp.ndarray) -> jnp.ndarray:
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.maximum(nu, eps)
    z = y - mu - jnp.square(sigma) / nu
    # Use the exponential-modified formula (nu large enough relative to sigma)
    log_fy_emg = (
        -jnp.log(nu)
        - (z + jnp.square(sigma) / (2.0 * nu)) / nu
        + norm.logcdf(z / sigma)
    )
    # Fallback to pure normal when nu is very small
    log_fy_normal = norm.logpdf(y, loc=mu, scale=sigma)
    return jnp.where(nu > 0.05 * sigma, log_fy_emg, log_fy_normal)


def exGAUS() -> ExGaussianFamily:
    return build_ad_family(
        family_class=ExGaussianFamily,
        name="exGAUS",
        parameters=("mu", "sigma", "nu"),
        log_pdf_func=_exgaus_log_pdf,
        type_="Continuous",
        links={"mu": "identity", "sigma": "log", "nu": "log"},
        link_functions={"mu": identity_link, "sigma": log_link, "nu": log_link},
        link_inverses={"mu": identity_inverse, "sigma": log_inverse, "nu": log_inverse},
        link_derivatives={"mu": identity_derivative, "sigma": log_derivative, "nu": log_derivative},
    )
