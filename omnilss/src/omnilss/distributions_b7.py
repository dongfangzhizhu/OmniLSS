"""GAMLSS distributions: Batch 7 — Multinomial and Compound families.

These are distributions for multinomial and compound count data.

Distributions:
- MN3      (Multinomial with 3 categories)
- MN4      (Multinomial with 4 categories)
- MN5      (Multinomial with 5 categories)
- BB       (Beta Binomial)
- BNB      (Beta Negative Binomial)

R source: gamlss.dist package
"""

from __future__ import annotations

# Enable float64 precision for numerical accuracy

from dataclasses import dataclass
import math

import jax.numpy as jnp
from jax.scipy.special import gammaln, betaln

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

# ------------------------------------------------------------------
# 1. BB — Beta Binomial
#    R: dBB(x, mu, sigma, bd)
#    A binomial distribution where the probability parameter follows a Beta distribution
# ------------------------------------------------------------------

@dataclass(frozen=True)
class BetaBinomialFamily(FamilyDefinition):
    """Beta Binomial distribution (`BB`)."""


def _bb_log_pdf(y, mu, sigma, bd):
    """Beta Binomial log-pdf.
    
    The Beta Binomial distribution is a binomial distribution where
    the probability parameter p follows a Beta(alpha, beta) distribution.
    
    Parameters:
    - y: observed count (0 to bd)
    - mu: mean probability (0 to 1)
    - sigma: dispersion parameter (> 0)
    - bd: binomial denominator (number of trials, FIXED parameter from data)
    
    R parameterization (from gamlss.dist):
    alpha = mu * (1/sigma - 1)
    beta = (1 - mu) * (1/sigma - 1)
    
    Note: bd must be provided in the data and will not be estimated.
    """
    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.clip(mu, eps, 1.0 - eps)
    sigma = jnp.maximum(sigma, eps)
    # Ensure bd is positive (it should come from data)
    bd = jnp.maximum(bd, 1.0)
    
    # R parameterization from gamlss.dist/R/BB.R:
    # k = 1/sigma
    # alpha = mu * k = mu/sigma
    # beta = (1-mu) * k = (1-mu)/sigma
    # 
    # logfy = lgamma(bd+1) - lgamma(y+1) - lgamma(bd-y+1)
    #         + lgamma(k) + lgamma(y + mu*k) + lgamma(bd + (1-mu)*k - y)
    #         - lgamma(mu*k) - lgamma((1-mu)*k) - lgamma(bd + k)
    #
    # Note: When sigma < 0.0001, R uses binomial limit
    
    k = 1.0 / sigma
    
    # Numerical guards
    # When sigma is very small (k very large), we approach the binomial limit
    # R uses sigma >= 1e-10, so k <= 1e10
    k = jnp.minimum(k, 1e10)
    
    log_pdf = (
        gammaln(bd + 1.0) - gammaln(y + 1.0) - gammaln(bd - y + 1.0) +
        gammaln(k) +
        gammaln(y + mu * k) +
        gammaln(bd + (1.0 - mu) * k - y) -
        gammaln(mu * k) -
        gammaln((1.0 - mu) * k) -
        gammaln(bd + k)
    )
    
    return log_pdf


def BB() -> BetaBinomialFamily:
    """Beta Binomial distribution.
    
    Note: 'bd' (binomial denominator) is a fixed parameter that must be
    provided in the data. It will not be estimated.
    
    Usage:
        data = {"y": counts, "bd": denominators}
        model = gamlss(formula="y ~ x", family=BB(), data=data)
    """
    from .dpqr_functions import pBB, qBB, rBB
    return build_ad_family(
        family_class=BetaBinomialFamily,
        name="BB",
        parameters=("mu", "sigma", "bd"),
        log_pdf_func=_bb_log_pdf,
        type_="Discrete",
        links={"mu": "logit", "sigma": "log", "bd": "identity"},
        link_functions={"mu": logit_link, "sigma": log_link, "bd": identity_link},
        link_inverses={"mu": logit_inverse, "sigma": log_inverse, "bd": identity_inverse},
        link_derivatives={"mu": logit_derivative, "sigma": log_derivative, "bd": identity_derivative},
        fixed_parameters=("bd",),  # bd is fixed, not estimated
        p=pBB,
        q=qBB,
        r=rBB,
    )


# ------------------------------------------------------------------
# 2. BNB — Beta Negative Binomial
#    R: dBNB(x, mu, sigma, nu)
#    A negative binomial where the probability parameter follows a Beta distribution
# ------------------------------------------------------------------

@dataclass(frozen=True)
class BetaNegativeBinomialFamily(FamilyDefinition):
    """Beta Negative Binomial distribution (`BNB`)."""


def _bnb_log_pdf(y, mu, sigma, nu):
    """Beta Negative Binomial log-pdf.
    
    The Beta Negative Binomial is a compound distribution combining
    Negative Binomial and Beta distributions.
    
    R parameterization (from gamlss.dist):
    m = (1/sigma) + 1
    n = (mu*nu)/sigma
    k = 1/nu
    
    log p(y) = log(Beta(y+n, m+k)) - log(Beta(n,m)) - log(Gamma(y+1)) - log(Gamma(k)) + log(Gamma(y+k))
    
    Parameters:
    - y: observed count
    - mu: mean parameter (> 0)
    - sigma: dispersion parameter (> 0)
    - nu: shape parameter (> 0)
    """
    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.maximum(nu, eps)
    
    # R parameterization
    m = (1.0 / sigma) + 1.0
    n = (mu * nu) / sigma
    k = 1.0 / nu
    
    # log p(y) = lbeta(y+n, m+k) - lbeta(n,m) - lgamma(y+1) - lgamma(k) + lgamma(y+k)
    log_pdf = (
        betaln(y + n, m + k) - betaln(n, m) - 
        gammaln(y + 1.0) - gammaln(k) + gammaln(y + k)
    )
    
    return log_pdf


def BNB() -> BetaNegativeBinomialFamily:
    from .dpqr_functions import pBNB, qBNB, rBNB
    return build_ad_family(
        family_class=BetaNegativeBinomialFamily,
        name="BNB",
        parameters=("mu", "sigma", "nu"),
        log_pdf_func=_bnb_log_pdf,
        type_="Discrete",
        links={"mu": "log", "sigma": "log", "nu": "log"},
        link_functions={"mu": log_link, "sigma": log_link, "nu": log_link},
        link_inverses={"mu": log_inverse, "sigma": log_inverse, "nu": log_inverse},
        link_derivatives={"mu": log_derivative, "sigma": log_derivative, "nu": log_derivative},
        p=pBNB,
        q=qBNB,
        r=rBNB,
    )


# ------------------------------------------------------------------
# 3-5. MN3, MN4, MN5 — Multinomial distributions
#    R: dMN3(y, mu1, mu2), dMN4(y, mu1, mu2, mu3), dMN5(y, mu1, mu2, mu3, mu4)
#    Multinomial distributions with 3, 4, or 5 categories
#
#    Note: These are multivariate distributions, which don't fit well
#    into the univariate GAMLSS framework. We'll implement simplified
#    versions that work with the first category count.
# ------------------------------------------------------------------

@dataclass(frozen=True)
class Multinomial3Family(FamilyDefinition):
    """Multinomial distribution with 3 categories (`MN3`).
    
    This is a categorical distribution for responses with 3 levels (1, 2, 3).
    Not a true multinomial count distribution, but a categorical response model.
    """


def _mn3_log_pdf(y, mu, sigma):
    """MN3 log-pdf for categorical response with 3 levels.
    
    R parameterization:
    - P(Y=1) = mu/(1+mu+sigma)
    - P(Y=2) = sigma/(1+mu+sigma)
    - P(Y=3) = 1/(1+mu+sigma)
    
    Parameters:
    - y: categorical response (1, 2, or 3)
    - mu: parameter for category 1 (mu > 0)
    - sigma: parameter for category 2 (sigma > 0)
    
    Note: y should be integer-valued (1, 2, or 3).
    """
    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    
    # Compute log probabilities for each category
    # log(P(Y=1)) = log(mu) - log(1+mu+sigma)
    # log(P(Y=2)) = log(sigma) - log(1+mu+sigma)
    # log(P(Y=3)) = log(1) - log(1+mu+sigma) = -log(1+mu+sigma)
    
    log_normalizer = jnp.log(1.0 + mu + sigma)
    
    # Use indicator functions to select the appropriate log probability
    # y should be 1, 2, or 3
    y_int = jnp.round(y).astype(jnp.int32)
    
    log_p1 = jnp.log(mu) - log_normalizer
    log_p2 = jnp.log(sigma) - log_normalizer
    log_p3 = -log_normalizer
    
    # Select log probability based on y value
    log_pdf = jnp.where(y_int == 1, log_p1,
                        jnp.where(y_int == 2, log_p2, log_p3))
    
    return log_pdf


def MN3() -> Multinomial3Family:
    """Multinomial distribution with 3 categories.
    
    This is a categorical distribution for responses with 3 levels (1, 2, 3).
    
    **Parameterization**:
    - P(Y=1) = mu/(1+mu+sigma)
    - P(Y=2) = sigma/(1+mu+sigma)
    - P(Y=3) = 1/(1+mu+sigma)
    
    **Parameters**:
    - mu: positive parameter for category 1
    - sigma: positive parameter for category 2
    
    **Response**: Integer-valued categorical variable (1, 2, or 3)
    
    **Links**: log link for both mu and sigma (ensuring positivity)
    
    **Note**: This is NOT a multinomial count distribution, but a categorical
    response model. The response y should be a single integer (1, 2, or 3)
    indicating which category was observed.
    """
    return build_ad_family(
        family_class=Multinomial3Family,
        name="MN3",
        parameters=("mu", "sigma"),
        log_pdf_func=_mn3_log_pdf,
        type_="Discrete",
        links={"mu": "log", "sigma": "log"},
        link_functions={"mu": log_link, "sigma": log_link},
        link_inverses={"mu": log_inverse, "sigma": log_inverse},
        link_derivatives={"mu": log_derivative, "sigma": log_derivative},
    )


@dataclass(frozen=True)
class Multinomial4Family(FamilyDefinition):
    """Multinomial distribution with 4 categories (`MN4`).
    
    This is a categorical distribution for responses with 4 levels (1, 2, 3, 4).
    Not a true multinomial count distribution, but a categorical response model.
    """


def _mn4_log_pdf(y, mu, sigma, nu):
    """MN4 log-pdf for categorical response with 4 levels.
    
    R parameterization:
    - P(Y=1) = mu/(1+mu+sigma+nu)
    - P(Y=2) = sigma/(1+mu+sigma+nu)
    - P(Y=3) = nu/(1+mu+sigma+nu)
    - P(Y=4) = 1/(1+mu+sigma+nu)
    
    Parameters:
    - y: categorical response (1, 2, 3, or 4)
    - mu: parameter for category 1 (mu > 0)
    - sigma: parameter for category 2 (sigma > 0)
    - nu: parameter for category 3 (nu > 0)
    
    Note: y should be integer-valued (1, 2, 3, or 4).
    """
    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.maximum(nu, eps)
    
    # Compute log probabilities for each category
    log_normalizer = jnp.log(1.0 + mu + sigma + nu)
    
    y_int = jnp.round(y).astype(jnp.int32)
    
    log_p1 = jnp.log(mu) - log_normalizer
    log_p2 = jnp.log(sigma) - log_normalizer
    log_p3 = jnp.log(nu) - log_normalizer
    log_p4 = -log_normalizer
    
    # Select log probability based on y value
    log_pdf = jnp.where(y_int == 1, log_p1,
                        jnp.where(y_int == 2, log_p2,
                                  jnp.where(y_int == 3, log_p3, log_p4)))
    
    return log_pdf


def MN4() -> Multinomial4Family:
    """Multinomial distribution with 4 categories.
    
    This is a categorical distribution for responses with 4 levels (1, 2, 3, 4).
    
    **Parameterization**:
    - P(Y=1) = mu/(1+mu+sigma+nu)
    - P(Y=2) = sigma/(1+mu+sigma+nu)
    - P(Y=3) = nu/(1+mu+sigma+nu)
    - P(Y=4) = 1/(1+mu+sigma+nu)
    
    **Parameters**:
    - mu: positive parameter for category 1
    - sigma: positive parameter for category 2
    - nu: positive parameter for category 3
    
    **Response**: Integer-valued categorical variable (1, 2, 3, or 4)
    
    **Links**: log link for all parameters (ensuring positivity)
    
    **Note**: This is NOT a multinomial count distribution, but a categorical
    response model. The response y should be a single integer (1, 2, 3, or 4)
    indicating which category was observed.
    """
    return build_ad_family(
        family_class=Multinomial4Family,
        name="MN4",
        parameters=("mu", "sigma", "nu"),
        log_pdf_func=_mn4_log_pdf,
        type_="Discrete",
        links={"mu": "log", "sigma": "log", "nu": "log"},
        link_functions={"mu": log_link, "sigma": log_link, "nu": log_link},
        link_inverses={"mu": log_inverse, "sigma": log_inverse, "nu": log_inverse},
        link_derivatives={"mu": log_derivative, "sigma": log_derivative, "nu": log_derivative},
    )


@dataclass(frozen=True)
class Multinomial5Family(FamilyDefinition):
    """Multinomial distribution with 5 categories (`MN5`).
    
    This is a categorical distribution for responses with 5 levels (1, 2, 3, 4, 5).
    Not a true multinomial count distribution, but a categorical response model.
    """


def _mn5_log_pdf(y, mu, sigma, nu, tau):
    """MN5 log-pdf for categorical response with 5 levels.
    
    R parameterization:
    - P(Y=1) = mu/(1+mu+sigma+nu+tau)
    - P(Y=2) = sigma/(1+mu+sigma+nu+tau)
    - P(Y=3) = nu/(1+mu+sigma+nu+tau)
    - P(Y=4) = tau/(1+mu+sigma+nu+tau)
    - P(Y=5) = 1/(1+mu+sigma+nu+tau)
    
    Parameters:
    - y: categorical response (1, 2, 3, 4, or 5)
    - mu: parameter for category 1 (mu > 0)
    - sigma: parameter for category 2 (sigma > 0)
    - nu: parameter for category 3 (nu > 0)
    - tau: parameter for category 4 (tau > 0)
    
    Note: y should be integer-valued (1, 2, 3, 4, or 5).
    """
    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.maximum(nu, eps)
    tau = jnp.maximum(tau, eps)
    
    # Compute log probabilities for each category
    log_normalizer = jnp.log(1.0 + mu + sigma + nu + tau)
    
    y_int = jnp.round(y).astype(jnp.int32)
    
    log_p1 = jnp.log(mu) - log_normalizer
    log_p2 = jnp.log(sigma) - log_normalizer
    log_p3 = jnp.log(nu) - log_normalizer
    log_p4 = jnp.log(tau) - log_normalizer
    log_p5 = -log_normalizer
    
    # Select log probability based on y value
    log_pdf = jnp.where(y_int == 1, log_p1,
                        jnp.where(y_int == 2, log_p2,
                                  jnp.where(y_int == 3, log_p3,
                                            jnp.where(y_int == 4, log_p4, log_p5))))
    
    return log_pdf


def MN5() -> Multinomial5Family:
    """Multinomial distribution with 5 categories.
    
    This is a categorical distribution for responses with 5 levels (1, 2, 3, 4, 5).
    
    **Parameterization**:
    - P(Y=1) = mu/(1+mu+sigma+nu+tau)
    - P(Y=2) = sigma/(1+mu+sigma+nu+tau)
    - P(Y=3) = nu/(1+mu+sigma+nu+tau)
    - P(Y=4) = tau/(1+mu+sigma+nu+tau)
    - P(Y=5) = 1/(1+mu+sigma+nu+tau)
    
    **Parameters**:
    - mu: positive parameter for category 1
    - sigma: positive parameter for category 2
    - nu: positive parameter for category 3
    - tau: positive parameter for category 4
    
    **Response**: Integer-valued categorical variable (1, 2, 3, 4, or 5)
    
    **Links**: log link for all parameters (ensuring positivity)
    
    **Note**: This is NOT a multinomial count distribution, but a categorical
    response model. The response y should be a single integer (1, 2, 3, 4, or 5)
    indicating which category was observed.
    """
    return build_ad_family(
        family_class=Multinomial5Family,
        name="MN5",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_mn5_log_pdf,
        type_="Discrete",
        links={"mu": "log", "sigma": "log", "nu": "log", "tau": "log"},
        link_functions={"mu": log_link, "sigma": log_link, "nu": log_link, "tau": log_link},
        link_inverses={"mu": log_inverse, "sigma": log_inverse, "nu": log_inverse, "tau": log_inverse},
        link_derivatives={"mu": log_derivative, "sigma": log_derivative, "nu": log_derivative, "tau": log_derivative},
    )
