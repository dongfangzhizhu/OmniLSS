"""GAMLSS distributions: Batch 1 (Continuous & Discrete expansions).

This batch introduces auto-differentiated families to rapidly expand
distribution support without the manual gradient derivations.

Distributions included:
- GU (Gumbel)
- RG (Reverse Gumbel)
- IGAMMA (Inverse Gamma)
- PARETO2 (Pareto Type II)
- NBII (Negative Binomial Type II)
"""

from __future__ import annotations

# Enable float64 precision for numerical accuracy
import jax
jax.config.update("jax_enable_x64", True)


from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

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
# 1. Gumbel (GU)
# ------------------------------------------------------------------

@dataclass(frozen=True)
class GumbelFamily(FamilyDefinition):
    """Gumbel distribution (`GU`)."""


def _gu_log_pdf(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    # R: log.lik <- -log(sigma) + ((x - mu)/sigma) - exp((x - mu)/sigma)
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    z = (y - mu) / sigma
    return -jnp.log(sigma) + z - jnp.exp(z)


def GU() -> GumbelFamily:
    from .dpqr_functions import dGU, pGU, qGU, rGU
    
    base_family = build_ad_family(
        family_class=GumbelFamily,
        name="GU",
        parameters=("mu", "sigma"),
        log_pdf_func=_gu_log_pdf,
        type_="Continuous",
        links={"mu": "identity", "sigma": "log"},
        link_functions={"mu": identity_link, "sigma": log_link},
        link_inverses={"mu": identity_inverse, "sigma": log_inverse},
        link_derivatives={"mu": identity_derivative, "sigma": log_derivative},
    )
    
    # Override with real p/q/r implementations
    return GumbelFamily(
        name=base_family.name,
        parameters=base_family.parameters,
        g_dev_inc=base_family.g_dev_inc,
        type=base_family.type,
        links=base_family.links,
        link_functions=base_family.link_functions,
        link_inverses=base_family.link_inverses,
        link_derivatives=base_family.link_derivatives,
        score_functions=base_family.score_functions,
        hessian_functions=base_family.hessian_functions,
        fixed_parameters=base_family.fixed_parameters,
        d=dGU,
        p=pGU,
        q=qGU,
        r=rGU,
    )


# ------------------------------------------------------------------
# 2. Reverse Gumbel (RG)
# ------------------------------------------------------------------

@dataclass(frozen=True)
class ReverseGumbelFamily(FamilyDefinition):
    """Reverse Gumbel distribution (`RG`)."""


def _rg_log_pdf(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    # R: log.lik <- (-log(sigma) - ((x - mu)/sigma) - exp(-(x - mu)/sigma))
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    z = (y - mu) / sigma
    return -jnp.log(sigma) - z - jnp.exp(-z)


def RG() -> ReverseGumbelFamily:
    from .dpqr_functions import dRG, pRG, qRG, rRG
    
    base_family = build_ad_family(
        family_class=ReverseGumbelFamily,
        name="RG",
        parameters=("mu", "sigma"),
        log_pdf_func=_rg_log_pdf,
        type_="Continuous",
        links={"mu": "identity", "sigma": "log"},
        link_functions={"mu": identity_link, "sigma": log_link},
        link_inverses={"mu": identity_inverse, "sigma": log_inverse},
        link_derivatives={"mu": identity_derivative, "sigma": log_derivative},
    )
    
    # Override with real p/q/r implementations
    return ReverseGumbelFamily(
        name=base_family.name,
        parameters=base_family.parameters,
        g_dev_inc=base_family.g_dev_inc,
        type=base_family.type,
        links=base_family.links,
        link_functions=base_family.link_functions,
        link_inverses=base_family.link_inverses,
        link_derivatives=base_family.link_derivatives,
        score_functions=base_family.score_functions,
        hessian_functions=base_family.hessian_functions,
        fixed_parameters=base_family.fixed_parameters,
        d=dRG,
        p=pRG,
        q=qRG,
        r=rRG,
    )


# ------------------------------------------------------------------
# 3. Inverse Gamma (IGAMMA)
# ------------------------------------------------------------------

@dataclass(frozen=True)
class InverseGammaFamily(FamilyDefinition):
    """Inverse Gamma distribution (`IGAMMA`)."""


def _igamma_log_pdf(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    eps = jnp.finfo(jnp.float64).eps
    y = jnp.maximum(y, eps)
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    alpha = 1.0 / jnp.square(sigma)
    return (
        alpha * jnp.log(mu)
        + alpha * jnp.log(alpha + 1.0)
        - gammaln(alpha)
        - (alpha + 1.0) * jnp.log(y)
        - (mu * (alpha + 1.0)) / y
    )


def IGAMMA() -> InverseGammaFamily:
    """Inverse Gamma family with expected Hessian (Fisher information).
    
    R parameterization (from gamlss.dist/R/IGAMMA.R):
    - alpha = 1 / sigma^2
    - PDF: alpha * log(mu) + alpha * log(alpha + 1) - lgamma(alpha) 
           - (alpha + 1) * log(y) - (mu * (alpha + 1)) / y
    - mu is NOT the mean! E[Y] = mu * (alpha + 1) / alpha = mu * (1 + sigma^2)
    
    Hessian method:
    - R uses expected Hessian (Fisher information)
    - d2ldm2 = -1 / (sigma^2 * mu^2)
    - d2ldd2 = -4 * (-(sigma^2 * (1 + 2*sigma^2)) / (1 + sigma^2)^2 + trigamma(1/sigma^2)) / sigma^6
    - Does not depend on y (Fisher information)
    
    R source reference:
    - file: `gamlss.dist/R/IGAMMA.R`
    - function: `IGAMMA()`
    """
    from .ad import build_ad_family
    
    # Build base family with AD for score functions
    base_family = build_ad_family(
        family_class=InverseGammaFamily,
        name="IGAMMA",
        parameters=("mu", "sigma"),
        log_pdf_func=_igamma_log_pdf,
        type_="Continuous",
        links={"mu": "log", "sigma": "log"},
        link_functions={"mu": log_link, "sigma": log_link},
        link_inverses={"mu": log_inverse, "sigma": log_inverse},
        link_derivatives={"mu": log_derivative, "sigma": log_derivative},
    )
    
    # Override with expected Hessian (Fisher information) as R does
    eps = jnp.finfo(jnp.float64).eps
    
    def d2ldm2_expected(y, mu, sigma, **kwargs):
        """Expected Hessian for mu.
        
        R formula: -1 / (sigma^2 * mu^2)
        Does not depend on y (Fisher information).
        """
        mu = jnp.maximum(jnp.asarray(mu, dtype=jnp.float64), eps)
        sigma = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float64), eps)
        d2ldm2 = -1.0 / (jnp.square(sigma) * jnp.square(mu))
        # Ensure negative for numerical stability
        d2ldm2 = jnp.where(d2ldm2 < -eps, d2ldm2, -eps)
        return d2ldm2
    
    def d2lds2_expected(y, mu, sigma, **kwargs):
        """Expected Hessian for sigma.
        
        R formula: -4 * (-(sigma^2 * (1 + 2*sigma^2)) / (1 + sigma^2)^2 + trigamma(1/sigma^2)) / sigma^6
        Does not depend on y or mu (Fisher information).
        """
        from jax.scipy.special import polygamma
        
        sigma = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float64), eps)
        alpha = 1.0 / jnp.square(sigma)
        
        # trigamma(x) = polygamma(1, x)
        trigamma_val = polygamma(1, alpha)
        
        # First term: -(sigma^2 * (1 + 2*sigma^2)) / (1 + sigma^2)^2
        sigma_sq = jnp.square(sigma)
        term1 = -(sigma_sq * (1.0 + 2.0 * sigma_sq)) / jnp.square(1.0 + sigma_sq)
        
        # Second term: trigamma(1/sigma^2)
        term2 = trigamma_val
        
        # Combine: -4 * (term1 + term2) / sigma^6
        d2lds2 = -4.0 * (term1 + term2) / jnp.power(sigma, 6)
        
        # Ensure negative for numerical stability
        d2lds2 = jnp.where(d2lds2 < -eps, d2lds2, -eps)
        return d2lds2
    
    # Create new family with expected Hessian
    return InverseGammaFamily(
        name="IGAMMA",
        parameters=("mu", "sigma"),
        g_dev_inc=base_family.g_dev_inc,
        type="Continuous",
        links={"mu": "log", "sigma": "log"},
        link_functions={"mu": log_link, "sigma": log_link},
        link_inverses={"mu": log_inverse, "sigma": log_inverse},
        link_derivatives={"mu": log_derivative, "sigma": log_derivative},
        score_functions={
            "mu": base_family.score_functions["mu"],
            "sigma": base_family.score_functions["sigma"],
        },
        hessian_functions={
            "mu": d2ldm2_expected,
            "sigma": d2lds2_expected,
        },
        d=base_family.d,  # Keep d from build_ad_family
        p=base_family.p,  # Keep p from build_ad_family
        q=base_family.q,  # Keep q from build_ad_family
        r=base_family.r,  # Keep r from build_ad_family
    )


# ------------------------------------------------------------------
# 4. Pareto Type II (PARETO2)
# ------------------------------------------------------------------

@dataclass(frozen=True)
class Pareto2Family(FamilyDefinition):
    """Pareto Type II (Lomax) distribution (`PARETO2`)."""


def _pareto2_log_pdf(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    eps = jnp.finfo(jnp.float64).eps
    y = jnp.maximum(y, 1e-15)  # Can be zero but we add mu so it's safe
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    # lfy <- -log(sigma) + (1/sigma) * log(mu) - ((1/sigma) + 1) * log(x + mu)
    return -jnp.log(sigma) + (1.0 / sigma) * jnp.log(mu) - ((1.0 / sigma) + 1.0) * jnp.log(y + mu)


def PARETO2() -> Pareto2Family:
    """Pareto Type II (Lomax) distribution with expected Hessian (Fisher information).
    
    Uses expected Hessian (Fisher information) following R gamlss.dist implementation.
    R uses analytical formulas that do not depend on y.
    
    Parameterization:
    - shape = 1/sigma
    - scale parameter related to mu
    - E(Y) = mu (for sigma < 1, finite mean)
    - Var(Y) = mu^2 * sigma^2 / (1 - sigma^2) (for sigma < 1)
    
    Constraints:
    - Y > 0
    - mu > 0
    - sigma > 0
    - For finite mean: sigma < 1
    - For finite variance: sigma < 1/sqrt(2) ≈ 0.707
    
    R source reference:
    - file: `gamlss.dist/R/PARETO2.R`
    - function: `PARETO2()`
    - d2ldm2: -(1/(mu^2 * (1 + 2*sigma)))
    - d2ldd2: -(1/sigma^2)
    """
    from .ad import build_ad_family
    
    # Build family with AD for scores
    family = build_ad_family(
        family_class=Pareto2Family,
        name="PARETO2",
        parameters=("mu", "sigma"),
        log_pdf_func=_pareto2_log_pdf,
        type_="Continuous",
        links={"mu": "log", "sigma": "log"},
        link_functions={"mu": log_link, "sigma": log_link},
        link_inverses={"mu": log_inverse, "sigma": log_inverse},
        link_derivatives={"mu": log_derivative, "sigma": log_derivative},
    )
    
    # Define expected Hessian functions (Fisher information) matching R
    def d2ldm2_expected(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """Expected second derivative wrt mu (Fisher information).
        
        R formula: -(1/(mu^2 * (1 + 2*sigma)))
        """
        mu = jnp.asarray(mu, dtype=jnp.float64)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        eps = jnp.finfo(jnp.float64).eps
        mu = jnp.maximum(mu, eps)
        sigma = jnp.maximum(sigma, eps)
        
        # R formula
        d2ldm2 = -(1.0 / (jnp.square(mu) * (1.0 + 2.0 * sigma)))
        
        # Ensure negative
        d2ldm2 = jnp.where(d2ldm2 < -1e-15, d2ldm2, -1e-15)
        
        return d2ldm2
    
    def d2lds2_expected(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """Expected second derivative wrt sigma (Fisher information).
        
        R formula: -(1/sigma^2)
        """
        mu = jnp.asarray(mu, dtype=jnp.float64)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        eps = jnp.finfo(jnp.float64).eps
        sigma = jnp.maximum(sigma, eps)
        
        # R formula
        d2lds2 = -(1.0 / jnp.square(sigma))
        
        # Ensure negative
        d2lds2 = jnp.where(d2lds2 < -1e-15, d2lds2, -1e-15)
        
        return d2lds2
    
    # Replace hessian functions with expected versions
    hessian_functions = {
        "mu": d2ldm2_expected,
        "sigma": d2lds2_expected,
    }
    
    # Create new family with updated hessians
    return Pareto2Family(
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
# 5. Negative Binomial Type II (NBII)
# ------------------------------------------------------------------

@dataclass(frozen=True)
class NegativeBinomial2Family(FamilyDefinition):
    """Negative Binomial Type II distribution (`NBII`)."""


def _nb2_log_pdf(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    eps = jnp.finfo(jnp.float64).eps
    y = jnp.maximum(y, 0.0)
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    
    # In NBII, size = mu / sigma (Whereas NBI has size = 1 / sigma)
    size = mu / sigma
    
    # dnbinom(x, size, mu) log PDF formulation
    return (
        gammaln(y + size)
        - gammaln(size)
        - gammaln(y + 1.0)
        + size * jnp.log(size / (size + mu))
        + y * jnp.log(mu / (size + mu))
    )


def NBII() -> NegativeBinomial2Family:
    """Negative Binomial Type II following R gamlss.dist implementation.
    
    R uses expected Hessian (Fisher information) approximation:
    - d2ldm2: Uses analytical formula, but if >= 0, uses -dldm^2
    - d2ldd2: Uses -dldd^2 (expected Hessian approximation)
    - Both are clamped to be < -1e-15
    
    Parameterization:
    - size = mu / sigma
    - E(Y) = mu
    - Var(Y) = mu + mu²/size = mu(1 + sigma)
    
    R source reference:
    - file: `gamlss.dist/R/NBII.R`
    - function: `NBII()`
    """
    
    def _log_density(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
        return _nb2_log_pdf(y, mu, sigma)
    
    def g_dev_inc(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
        return -2.0 * _log_density(y, mu, sigma)
    
    # Score functions (first derivatives)
    def dldm(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
        y = jnp.asarray(y, dtype=jnp.float64)
        mu = jnp.asarray(mu, dtype=jnp.float64)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        eps = jnp.finfo(jnp.float64).eps
        mu = jnp.maximum(mu, eps)
        sigma = jnp.maximum(sigma, eps)
        size = mu / sigma
        # R: (1/sigma) * (digamma(y + (mu/sigma)) - digamma((mu/sigma)) - log(1 + sigma))
        from jax.scipy.special import digamma
        return (1.0 / sigma) * (digamma(y + size) - digamma(size) - jnp.log(1.0 + sigma))
    
    def dldd(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
        y = jnp.asarray(y, dtype=jnp.float64)
        mu = jnp.asarray(mu, dtype=jnp.float64)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        eps = jnp.finfo(jnp.float64).eps
        mu = jnp.maximum(mu, eps)
        sigma = jnp.maximum(sigma, eps)
        size = mu / sigma
        # R: -(mu/(sigma^2)) * (digamma(y + (mu/sigma)) - digamma((mu/sigma)) - log(1 + sigma)) + (y - mu)/(sigma * (1 + sigma))
        from jax.scipy.special import digamma
        term1 = -(mu / jnp.square(sigma)) * (digamma(y + size) - digamma(size) - jnp.log(1.0 + sigma))
        term2 = (y - mu) / (sigma * (1.0 + sigma))
        return term1 + term2
    
    # Hessian functions (second derivatives) - following R implementation
    def d2ldm2(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
        y = jnp.asarray(y, dtype=jnp.float64)
        mu = jnp.asarray(mu, dtype=jnp.float64)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        eps = jnp.finfo(jnp.float64).eps
        mu = jnp.maximum(mu, eps)
        sigma = jnp.maximum(sigma, eps)
        size = mu / sigma
        
        # R implementation:
        # dldm <- (1/sigma) * (digamma(y + (mu/sigma)) - digamma((mu/sigma)) - log(1 + sigma))
        # d2ldm2 <- ((1/sigma)^2) * (trigamma(y + (mu/sigma)) - trigamma((mu/sigma)))
        # d2ldm2 <- if (any(d2ldm2 >= 0)) -dldm^2 else d2ldm2
        # d2ldm2 <- ifelse(d2ldm2 < -1e-15, d2ldm2, -1e-15)
        
        from jax.scipy.special import polygamma
        dldm_val = dldm(y, mu, sigma)
        d2ldm2_val = jnp.square(1.0 / sigma) * (polygamma(1, y + size) - polygamma(1, size))
        
        # If any d2ldm2 >= 0, use -dldm^2 instead
        d2ldm2_val = jnp.where(d2ldm2_val >= 0, -jnp.square(dldm_val), d2ldm2_val)
        
        # Clamp to be < -1e-15
        d2ldm2_val = jnp.where(d2ldm2_val < -1e-15, d2ldm2_val, -1e-15)
        
        return d2ldm2_val
    
    def d2ldd2(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
        y = jnp.asarray(y, dtype=jnp.float64)
        mu = jnp.asarray(mu, dtype=jnp.float64)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        eps = jnp.finfo(jnp.float64).eps
        mu = jnp.maximum(mu, eps)
        sigma = jnp.maximum(sigma, eps)
        
        # R implementation:
        # dldd <- -(mu/(sigma^2)) * (digamma(y + (mu/sigma)) - digamma((mu/sigma)) - log(1 + sigma)) + (y - mu)/(sigma * (1 + sigma))
        # d2ldd2 <- -dldd^2
        # d2ldd2 <- ifelse(d2ldd2 < -1e-15, d2ldd2, -1e-15)
        
        dldd_val = dldd(y, mu, sigma)
        d2ldd2_val = -jnp.square(dldd_val)
        
        # Clamp to be < -1e-15
        d2ldd2_val = jnp.where(d2ldd2_val < -1e-15, d2ldd2_val, -1e-15)
        
        return d2ldd2_val
    
    return NegativeBinomial2Family(
        name="NBII",
        parameters=("mu", "sigma"),
        g_dev_inc=g_dev_inc,
        type="Discrete",
        links={"mu": "log", "sigma": "log"},
        link_functions={"mu": log_link, "sigma": log_link},
        link_inverses={"mu": log_inverse, "sigma": log_inverse},
        link_derivatives={"mu": log_derivative, "sigma": log_derivative},
        score_functions={"mu": dldm, "sigma": dldd},
        hessian_functions={"mu": d2ldm2, "sigma": d2ldd2},
        d=lambda x, mu, sigma, log=False: _log_density(x, mu, sigma) if log else jnp.exp(_log_density(x, mu, sigma)),
        p=lambda q, mu, sigma, lower_tail=True, log_p=False: jnp.full_like(jnp.asarray(q, dtype=jnp.float64), jnp.nan),
        q=lambda p, mu, sigma, lower_tail=True, log_p=False: jnp.full_like(jnp.asarray(p, dtype=jnp.float64), jnp.nan),
        r=lambda key, n, mu, sigma: jnp.full((n,), jnp.nan, dtype=jnp.float64),
    )
