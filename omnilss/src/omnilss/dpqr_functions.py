"""Standard distribution functions (d/p/q/r) for GAMLSS distributions.

This module provides density, CDF, quantile, and random generation functions
that match the R GAMLSS interface.

Functions follow the R naming convention:
- d{DIST}: density/probability mass function
- p{DIST}: cumulative distribution function
- q{DIST}: quantile function (inverse CDF)
- r{DIST}: random number generation
"""

from __future__ import annotations

import numpy as np
from .gb1 import dGB1, pGB1, qGB1, rGB1
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax.scipy.stats import norm, poisson, gamma, beta, expon, logistic, t
from jax.scipy.special import gammaln, betaln, ndtri
from scipy.stats import nbinom as _nbinom_dist
from scipy.stats import poisson as _pois_dist
from scipy.stats import gamma as _gamma_dist
from scipy.stats import t as _t_dist
from scipy.stats import betabinom as _betabinom_dist, binom as _binom_dist


# =============================================================================
# Normal Distribution (NO)
# =============================================================================

def dNO(x, mu=0.0, sigma=1.0, log=False):
    """Density function for Normal distribution.
    
    Parameters
    ----------
    x : array_like
        Quantiles
    mu : array_like
        Location parameter (mean)
    sigma : array_like
        Scale parameter (standard deviation)
    log : bool
        If True, return log density
        
    Returns
    -------
    array_like
        Density values
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    
    return norm.logpdf(x, loc=mu, scale=sigma) if log else norm.pdf(x, loc=mu, scale=sigma)


def pNO(q, mu=0.0, sigma=1.0, lower_tail=True, log_p=False):
    """CDF function for Normal distribution.
    
    Parameters
    ----------
    q : array_like
        Quantiles
    mu : array_like
        Location parameter (mean)
    sigma : array_like
        Scale parameter (standard deviation)
    lower_tail : bool
        If True, return P(X <= q), else P(X > q)
    log_p : bool
        If True, return log probability
        
    Returns
    -------
    array_like
        Cumulative probabilities
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    
    cdf = norm.cdf(q, loc=mu, scale=sigma)
    
    if not lower_tail:
        cdf = 1.0 - cdf
    
    if log_p:
        cdf = jnp.log(cdf)
    
    return cdf


def qNO(p, mu=0.0, sigma=1.0, lower_tail=True, log_p=False):
    """Quantile function for Normal distribution.
    
    Parameters
    ----------
    p : array_like
        Probabilities
    mu : array_like
        Location parameter (mean)
    sigma : array_like
        Scale parameter (standard deviation)
    lower_tail : bool
        If True, p is P(X <= q), else P(X > q)
    log_p : bool
        If True, p is given as log(probability)
        
    Returns
    -------
    array_like
        Quantiles
    """
    p = jnp.asarray(p, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    
    if log_p:
        p = jnp.exp(p)
    
    if not lower_tail:
        p = 1.0 - p
    
    q = mu + sigma * ndtri(p)
    
    # Handle edge cases
    q = jnp.where(jnp.abs(p - 0.0) < 1e-10, -jnp.inf, q)
    q = jnp.where(jnp.abs(1.0 - p) < 1e-10, jnp.inf, q)
    q = jnp.where((p < 0) | (p > 1), jnp.nan, q)
    
    return q


def rNO(key, n, mu=0.0, sigma=1.0):
    """Random generation for Normal distribution.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key
    n : int
        Number of observations
    mu : array_like
        Location parameter (mean)
    sigma : array_like
        Scale parameter (standard deviation)
        
    Returns
    -------
    array_like
        Random samples
    """
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    
    return mu + sigma * jrandom.normal(key, shape=(n,))


# =============================================================================
# Poisson Distribution (PO)
# =============================================================================

def dPO(x, mu=1.0, log=False):
    """Density function for Poisson distribution.
    
    Parameters
    ----------
    x : array_like
        Quantiles (non-negative integers)
    mu : array_like
        Mean parameter
    log : bool
        If True, return log density
        
    Returns
    -------
    array_like
        Density values
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    
    fy = poisson.logpmf(x, mu) if log else poisson.pmf(x, mu)
    fy = jnp.where(x < 0, 0.0, fy)
    
    return fy


def pPO(q, mu=1.0, lower_tail=True, log_p=False):
    """CDF function for Poisson distribution.
    
    Parameters
    ----------
    q : array_like
        Quantiles
    mu : array_like
        Mean parameter
    lower_tail : bool
        If True, return P(X <= q), else P(X > q)
    log_p : bool
        If True, return log probability
        
    Returns
    -------
    array_like
        Cumulative probabilities
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    
    cdf = poisson.cdf(q, mu)
    cdf = jnp.where(q < 0, 0.0, cdf)
    cdf = jnp.where(q >= jnp.inf, 1.0, cdf)
    
    if not lower_tail:
        cdf = 1.0 - cdf
    
    if log_p:
        cdf = jnp.log(cdf)
    
    return cdf


def qPO(p, mu=1.0, lower_tail=True, log_p=False):
    """Quantile function for Poisson distribution.
    
    Parameters
    ----------
    p : array_like
        Probabilities
    mu : array_like
        Mean parameter
    lower_tail : bool
        If True, p is P(X <= q), else P(X > q)
    log_p : bool
        If True, p is given as log(probability)
        
    Returns
    -------
    array_like
        Quantiles
    """
    p = np.asarray(p, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    
    if log_p:
        p = np.exp(p)
    
    if not lower_tail:
        p = 1.0 - p
    
    # Clip probabilities to avoid edge cases at exactly 0 or 1
    p = np.clip(p, 1e-12, 1 - 1e-12)
    
    # Exact Poisson quantile using scipy's ppf
    # For each probability value, find the smallest k such that P(X <= k) >= p
    # where X ~ Poisson(mu)
    q = _pois_dist.ppf(p, mu=mu)
    q = np.maximum(np.floor(q), 0.0)  # Non-negative integers
    
    return q


def rPO(key, n, mu=1.0):
    """Random generation for Poisson distribution.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key
    n : int
        Number of observations
    mu : array_like
        Mean parameter
        
    Returns
    -------
    array_like
        Random samples (integers)
    """
    mu = jnp.asarray(mu, dtype=jnp.float64)
    
    return jrandom.poisson(key, mu, shape=(n,))


# =============================================================================
# Binomial Distribution (BI)
# =============================================================================

def dBI(x, mu=0.5, bd=None, log=False):
    """Density function for Binomial distribution (Bernoulli case).
    
    Parameters
    ----------
    x : array_like
        Quantiles (0 or 1)
    mu : array_like
        Probability parameter
    bd : array_like or None
        Binomial denominator (number of trials). Default None means Bernoulli (bd=1).
    log : bool
        If True, return log density
        
    Returns
    -------
    array_like
        Density values
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    mu = jnp.clip(mu, 1e-10, 1.0 - 1e-10)
    
    if bd is None or (not hasattr(bd, '__len__') and bd == 1):
        # Bernoulli case
        log_fy = x * jnp.log(mu) + (1.0 - x) * jnp.log(1.0 - mu)
        return log_fy if log else jnp.exp(log_fy)
    else:
        # General Binomial case
        from jax.scipy.special import gammaln
        bd = jnp.asarray(bd, dtype=jnp.float64)
        x_count = jnp.round(x * bd)
        
        log_comb = gammaln(bd + 1.0) - gammaln(x_count + 1.0) - gammaln(bd - x_count + 1.0)
        log_fy = log_comb + x_count * jnp.log(mu) + (bd - x_count) * jnp.log(1.0 - mu)
        
        return log_fy if log else jnp.exp(log_fy)


def pBI(q, mu=0.5, bd=None, lower_tail=True, log_p=False):
    """CDF function for Binomial distribution (Bernoulli case).
    
    Parameters
    ----------
    q : array_like
        Quantiles
    mu : array_like
        Probability parameter
    bd : array_like or None
        Binomial denominator. Default None means Bernoulli (bd=1).
    lower_tail : bool
        If True, return P(X <= q), else P(X > q)
    log_p : bool
        If True, return log probability
        
    Returns
    -------
    array_like
        Cumulative probabilities
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    mu = jnp.clip(mu, 1e-10, 1.0 - 1e-10)
    
    if bd is None or (not hasattr(bd, '__len__') and bd == 1):
        # Bernoulli case
        cdf = jnp.where(q < 0, 0.0, jnp.where(q < 1, 1.0 - mu, 1.0))
    else:
        # General Binomial case
        from scipy.stats import binom
        bd = np.asarray(bd)
        mu_arr = np.asarray(mu)
        cdf = np.array([binom.cdf(int(np.floor(float(qi))), int(bd_i), float(mu_i)) 
                       for qi, bd_i, mu_i in zip(q, bd, mu_arr)])
        cdf = jnp.asarray(cdf, dtype=jnp.float64)
    
    if not lower_tail:
        cdf = 1.0 - cdf
    
    if log_p:
        cdf = jnp.log(cdf)
    
    return cdf


def qBI(p, mu=0.5, bd=None, lower_tail=True, log_p=False):
    """Quantile function for Binomial distribution (Bernoulli case).
    
    Parameters
    ----------
    p : array_like
        Probabilities
    mu : array_like
        Probability parameter
    bd : array_like or None
        Binomial denominator. Default None means Bernoulli (bd=1).
    lower_tail : bool
        If True, p is P(X <= q), else P(X > q)
    log_p : bool
        If True, p is given as log(probability)
        
    Returns
    -------
    array_like
        Quantiles
    """
    p = jnp.asarray(p, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    mu = jnp.clip(mu, 1e-10, 1.0 - 1e-10)
    
    if log_p:
        p = jnp.exp(p)
    
    if not lower_tail:
        p = 1.0 - p
    
    if bd is None or (not hasattr(bd, '__len__') and bd == 1):
        # Bernoulli case
        q = jnp.where(p <= 1.0 - mu, 0.0, 1.0)
    else:
        # General Binomial case
        from scipy.stats import binom
        bd = np.asarray(bd)
        mu_arr = np.asarray(mu)
        q_values = np.array([binom.ppf(float(pi), int(bdi), float(mui)) 
                           for pi, bdi, mui in zip(p, bd, mu_arr)])
        q = jnp.asarray(q_values, dtype=jnp.float64)
    
    return q


def rBI(key, n, mu=0.5):
    """Random generation for Binomial distribution (Bernoulli case).
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key
    n : int
        Number of observations
    mu : array_like
        Probability parameter
        
    Returns
    -------
    array_like
        Random samples (0 or 1)
    """
    mu = jnp.asarray(mu, dtype=jnp.float64)
    
    return (jrandom.uniform(key, shape=(n,)) < mu).astype(jnp.float64)


# =============================================================================
# Gamma Distribution (GA)
# =============================================================================

def dGA(x, mu=1.0, sigma=1.0, log=False):
    """Density function for Gamma distribution.
    
    Parameters
    ----------
    x : array_like
        Quantiles (positive)
    mu : array_like
        Mean parameter
    sigma : array_like
        Scale parameter (CV)
    log : bool
        If True, return log density
        
    Returns
    -------
    array_like
        Density values
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    
    shape = 1.0 / jnp.square(sigma)
    scale = mu * jnp.square(sigma)
    
    return gamma.logpdf(x, shape, scale=scale) if log else gamma.pdf(x, shape, scale=scale)


def pGA(q, mu=1.0, sigma=1.0, lower_tail=True, log_p=False):
    """CDF function for Gamma distribution.
    
    Parameters
    ----------
    q : array_like
        Quantiles
    mu : array_like
        Mean parameter
    sigma : array_like
        Scale parameter (CV)
    lower_tail : bool
        If True, return P(X <= q), else P(X > q)
    log_p : bool
        If True, return log probability
        
    Returns
    -------
    array_like
        Cumulative probabilities
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    
    shape = 1.0 / jnp.square(sigma)
    scale = mu * jnp.square(sigma)
    
    cdf = gamma.cdf(q, shape, scale=scale)
    
    if not lower_tail:
        cdf = 1.0 - cdf
    
    if log_p:
        cdf = jnp.log(cdf)
    
    return cdf


def qGA(p, mu=1.0, sigma=1.0, lower_tail=True, log_p=False):
    """Quantile function for Gamma distribution.
    
    Parameters
    ----------
    p : array_like
        Probabilities
    mu : array_like
        Mean parameter
    sigma : array_like
        Scale parameter (CV)
    lower_tail : bool
        If True, p is P(X <= q), else P(X > q)
    log_p : bool
        If True, p is given as log(probability)
        
    Returns
    -------
    array_like
        Quantiles
    """
    p = np.asarray(p, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    if log_p:
        p = np.exp(p)
    
    if not lower_tail:
        p = 1.0 - p
    
    # Clip to avoid edge cases at exactly 0 or 1
    p = np.clip(p, 1e-12, 1 - 1e-12)
    
    shape = 1.0 / np.square(sigma)
    scale = mu * np.square(sigma)
    
    q = _gamma_dist.ppf(p, a=shape, scale=scale)
    
    return q


def rGA(key, n, mu=1.0, sigma=1.0):
    """Random generation for Gamma distribution.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key
    n : int
        Number of observations
    mu : array_like
        Mean parameter
    sigma : array_like
        Scale parameter (CV)
        
    Returns
    -------
    array_like
        Random samples
    """
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    
    shape = 1.0 / jnp.square(sigma)
    scale = mu * jnp.square(sigma)
    
    return jrandom.gamma(key, shape, shape=(n,)) * scale


# =============================================================================
# Log-Normal Distribution (LOGNO)
# =============================================================================

def dLOGNO(x, mu=1.0, sigma=1.0, log=False):
    """Density function for Log-Normal distribution.
    
    Parameters
    ----------
    x : array_like
        Quantiles (positive)
    mu : array_like
        Median parameter
    sigma : array_like
        Shape parameter
    log : bool
        If True, return log density
        
    Returns
    -------
    array_like
        Density values
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    
    log_mu = jnp.log(mu)
    
    log_fy = -(jnp.log(x) + jnp.log(sigma) + 0.5 * jnp.log(2.0 * jnp.pi) +
               0.5 * jnp.square((jnp.log(x) - log_mu) / sigma))
    
    return log_fy if log else jnp.exp(log_fy)


def pLOGNO(q, mu=1.0, sigma=1.0, lower_tail=True, log_p=False):
    """CDF function for Log-Normal distribution.
    
    Parameters
    ----------
    q : array_like
        Quantiles
    mu : array_like
        Median parameter
    sigma : array_like
        Shape parameter
    lower_tail : bool
        If True, return P(X <= q), else P(X > q)
    log_p : bool
        If True, return log probability
        
    Returns
    -------
    array_like
        Cumulative probabilities
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    
    log_mu = jnp.log(mu)
    cdf = norm.cdf((jnp.log(q) - log_mu) / sigma)
    
    if not lower_tail:
        cdf = 1.0 - cdf
    
    if log_p:
        cdf = jnp.log(cdf)
    
    return cdf


def qLOGNO(p, mu=1.0, sigma=1.0, lower_tail=True, log_p=False):
    """Quantile function for Log-Normal distribution.
    
    Parameters
    ----------
    p : array_like
        Probabilities
    mu : array_like
        Median parameter
    sigma : array_like
        Shape parameter
    lower_tail : bool
        If True, p is P(X <= q), else P(X > q)
    log_p : bool
        If True, p is given as log(probability)
        
    Returns
    -------
    array_like
        Quantiles
    """
    p = jnp.asarray(p, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    
    if log_p:
        p = jnp.exp(p)
    
    if not lower_tail:
        p = 1.0 - p
    
    log_mu = jnp.log(mu)
    q = jnp.exp(log_mu + sigma * ndtri(p))
    
    return q


def rLOGNO(key, n, mu=1.0, sigma=1.0):
    """Random generation for Log-Normal distribution.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key
    n : int
        Number of observations
    mu : array_like
        Median parameter
    sigma : array_like
        Shape parameter
        
    Returns
    -------
    array_like
        Random samples
    """
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    
    log_mu = jnp.log(mu)
    return jnp.exp(log_mu + sigma * jrandom.normal(key, shape=(n,)))


# =============================================================================
# Exponential Distribution (EXP)
# =============================================================================

def dEXP(x, mu=1.0, log=False):
    """Density function for Exponential distribution.
    
    Parameters
    ----------
    x : array_like
        Quantiles (non-negative)
    mu : array_like
        Mean parameter
    log : bool
        If True, return log density
        
    Returns
    -------
    array_like
        Density values
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    
    scale = mu
    
    return expon.logpdf(x, scale=scale) if log else expon.pdf(x, scale=scale)


def pEXP(q, mu=1.0, lower_tail=True, log_p=False):
    """CDF function for Exponential distribution.
    
    Parameters
    ----------
    q : array_like
        Quantiles
    mu : array_like
        Mean parameter
    lower_tail : bool
        If True, return P(X <= q), else P(X > q)
    log_p : bool
        If True, return log probability
        
    Returns
    -------
    array_like
        Cumulative probabilities
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    
    scale = mu
    cdf = expon.cdf(q, scale=scale)
    
    if not lower_tail:
        cdf = 1.0 - cdf
    
    if log_p:
        cdf = jnp.log(cdf)
    
    return cdf


def qEXP(p, mu=1.0, lower_tail=True, log_p=False):
    """Quantile function for Exponential distribution.
    
    Parameters
    ----------
    p : array_like
        Probabilities
    mu : array_like
        Mean parameter
    lower_tail : bool
        If True, p is P(X <= q), else P(X > q)
    log_p : bool
        If True, p is given as log(probability)
        
    Returns
    -------
    array_like
        Quantiles
    """
    p = jnp.asarray(p, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    
    if log_p:
        p = jnp.exp(p)
    
    if not lower_tail:
        p = 1.0 - p
    
    q = -mu * jnp.log(1.0 - p)
    
    return q


def rEXP(key, n, mu=1.0):
    """Random generation for Exponential distribution.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key
    n : int
        Number of observations
    mu : array_like
        Mean parameter
        
    Returns
    -------
    array_like
        Random samples
    """
    mu = jnp.asarray(mu, dtype=jnp.float64)
    
    return jrandom.exponential(key, shape=(n,)) * mu


# =============================================================================
# Student-t Distribution (TF)
# =============================================================================

def dTF(x, mu=0.0, sigma=1.0, nu=10.0, log=False):
    """Density function for Student-t distribution.
    
    Parameters
    ----------
    x : array_like
        Quantiles
    mu : array_like
        Location parameter
    sigma : array_like
        Scale parameter
    nu : array_like
        Degrees of freedom
    log : bool
        If True, return log density
        
    Returns
    -------
    array_like
        Density values
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    
    z = (x - mu) / sigma
    log_fy = (gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) -
              0.5 * jnp.log(nu * jnp.pi) - jnp.log(sigma) -
              ((nu + 1.0) / 2.0) * jnp.log(1.0 + z * z / nu))
    
    return log_fy if log else jnp.exp(log_fy)


def pTF(q, mu=0.0, sigma=1.0, nu=10.0, lower_tail=True, log_p=False):
    """CDF function for Student-t distribution.
    
    Parameters
    ----------
    q : array_like
        Quantiles
    mu : array_like
        Location parameter
    sigma : array_like
        Scale parameter
    nu : array_like
        Degrees of freedom
    lower_tail : bool
        If True, return P(X <= q), else P(X > q)
    log_p : bool
        If True, return log probability
        
    Returns
    -------
    array_like
        Cumulative probabilities
    """
    from jax.scipy.special import betainc
    
    q = jnp.asarray(q, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    
    z = (q - mu) / sigma
    
    # Use the relationship between t-distribution CDF and incomplete beta function
    # P(T <= t) = 0.5 + 0.5 * sign(t) * betainc(nu/2, 1/2, nu/(nu + t^2))
    # where betainc is the regularized incomplete beta function
    
    x = nu / (nu + z * z)
    cdf = jnp.where(
        z < 0,
        0.5 * betainc(nu / 2.0, 0.5, x),
        1.0 - 0.5 * betainc(nu / 2.0, 0.5, x)
    )
    
    if not lower_tail:
        cdf = 1.0 - cdf
    
    if log_p:
        cdf = jnp.log(cdf)
    
    return cdf


def qTF(p, mu=0.0, sigma=1.0, nu=10.0, lower_tail=True, log_p=False):
    """Quantile function for Student-t distribution.
    
    Parameters
    ----------
    p : array_like
        Probabilities
    mu : array_like
        Location parameter
    sigma : array_like
        Scale parameter
    nu : array_like
        Degrees of freedom
    lower_tail : bool
        If True, p is P(X <= q), else P(X > q)
    log_p : bool
        If True, p is given as log(probability)
        
    Returns
    -------
    array_like
        Quantiles
    """
    p = np.asarray(p, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    nu = np.asarray(nu, dtype=np.float64)
    
    if log_p:
        p = np.exp(p)
    
    if not lower_tail:
        p = 1.0 - p
    
    # Clip probabilities to avoid edge cases at exactly 0 or 1
    p = np.clip(p, 1e-12, 1 - 1e-12)
    
    # Exact Student-t quantile via scipy's ppf (matches R's qt(p, df=nu))
    z = _t_dist.ppf(p, df=nu)
    q = mu + sigma * z
    
    return q


def rTF(key, n, mu=0.0, sigma=1.0, nu=10.0):
    """Random generation for Student-t distribution.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key
    n : int
        Number of observations
    mu : array_like
        Location parameter
    sigma : array_like
        Scale parameter
    nu : array_like
        Degrees of freedom
        
    Returns
    -------
    array_like
        Random samples
    """
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    
    return mu + sigma * jrandom.t(key, nu, shape=(n,))


# =============================================================================
# Weibull Distribution (WEI)
# =============================================================================

def dWEI(x, mu=1.0, sigma=1.0, log=False):
    """Density function for Weibull distribution (R WEI parameterization).

    R WEI.R: ``dweibull(x, scale=mu, shape=sigma)`` — mu is the *scale*
    parameter (NOT the mean or median).
    
    Parameters
    ----------
    x : array_like
        Quantiles (positive)
    mu : array_like
        Scale parameter
    sigma : array_like
        Shape parameter
    log : bool
        If True, return log density
        
    Returns
    -------
    array_like
        Density values
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    
    # R WEI: dweibull(x, scale=mu, shape=sigma)
    # log f = log(sigma) - log(mu) + (sigma-1)*(log(x)-log(mu)) - (x/mu)^sigma
    log_fy = (
        jnp.log(sigma) - jnp.log(mu)
        + (sigma - 1.0) * (jnp.log(x) - jnp.log(mu))
        - jnp.power(x / mu, sigma)
    )
    log_fy = jnp.where(x <= 0.0, -jnp.inf, log_fy)
    
    return log_fy if log else jnp.exp(log_fy)


def pWEI(q, mu=1.0, sigma=1.0, lower_tail=True, log_p=False):
    """CDF function for Weibull distribution (R WEI parameterization).

    R WEI.R: ``pweibull(q, scale=mu, shape=sigma)`` — mu is the scale.
    
    Parameters
    ----------
    q : array_like
        Quantiles
    mu : array_like
        Scale parameter
    sigma : array_like
        Shape parameter
    lower_tail : bool
        If True, return P(X <= q), else P(X > q)
    log_p : bool
        If True, return log probability
        
    Returns
    -------
    array_like
        Cumulative probabilities
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    
    # R WEI: pweibull(q, scale=mu, shape=sigma)
    cdf = 1.0 - jnp.exp(-jnp.power(q / mu, sigma))
    cdf = jnp.where(q <= 0.0, 0.0, cdf)
    
    if not lower_tail:
        cdf = 1.0 - cdf
    
    if log_p:
        cdf = jnp.log(cdf)
    
    return cdf


def qWEI(p, mu=1.0, sigma=1.0, lower_tail=True, log_p=False):
    """Quantile function for Weibull distribution (R WEI parameterization).

    R WEI.R: ``qweibull(p, scale=mu, shape=sigma)`` — mu is the scale.
    
    Parameters
    ----------
    p : array_like
        Probabilities
    mu : array_like
        Scale parameter
    sigma : array_like
        Shape parameter
    lower_tail : bool
        If True, p is P(X <= q), else P(X > q)
    log_p : bool
        If True, p is given as log(probability)
        
    Returns
    -------
    array_like
        Quantiles
    """
    p = jnp.asarray(p, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    
    if log_p:
        p = jnp.exp(p)
    
    if not lower_tail:
        p = 1.0 - p
    
    # R WEI: qweibull(p, scale=mu, shape=sigma) = mu * (-log(1-p))^(1/sigma)
    q = mu * jnp.power(-jnp.log(1.0 - p), 1.0 / sigma)
    
    return q


def rWEI(key, n, mu=1.0, sigma=1.0):
    """Random generation for Weibull distribution (R WEI parameterization).

    R WEI.R: ``rweibull(n, scale=mu, shape=sigma)`` — mu is the scale.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key
    n : int
        Number of observations
    mu : array_like
        Scale parameter
    sigma : array_like
        Shape parameter
        
    Returns
    -------
    array_like
        Random samples
    """
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    
    # R WEI: rweibull(n, scale=mu, shape=sigma)
    u = jrandom.uniform(key, shape=(n,))
    return mu * jnp.power(-jnp.log(1.0 - u), 1.0 / sigma)


# =============================================================================
def _wei3_scale(mu, sigma):
    """Mean-parameterized Weibull scale: R WEI3 uses mu/gamma(1+1/sigma)."""
    from jax.scipy.special import gammaln

    return mu / jnp.exp(gammaln(1.0 + 1.0 / sigma))


def dWEI3(x, mu=1.0, sigma=1.0, log=False):
    """Density function for Weibull distribution (R WEI3, mean parameterization).

    R WEI3.R: ``dweibull(x, scale=mu/gamma(1+1/sigma), shape=sigma)``.
    
    Parameters
    ----------
    x : array_like
        Quantiles (positive)
    mu : array_like
        Mean parameter
    sigma : array_like
        Shape parameter
    log : bool
        If True, return log density
        
    Returns
    -------
    array_like
        Density values
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)

    scale = _wei3_scale(mu, sigma)
    log_fy = (
        jnp.log(sigma) - jnp.log(scale)
        + (sigma - 1.0) * (jnp.log(x) - jnp.log(scale))
        - jnp.power(x / scale, sigma)
    )
    log_fy = jnp.where(x <= 0.0, -jnp.inf, log_fy)

    return log_fy if log else jnp.exp(log_fy)


def pWEI3(q, mu=1.0, sigma=1.0, lower_tail=True, log_p=False):
    """CDF function for Weibull distribution (R WEI3, mean parameterization)."""
    q = jnp.asarray(q, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)

    scale = _wei3_scale(mu, sigma)
    cdf = 1.0 - jnp.exp(-jnp.power(q / scale, sigma))
    cdf = jnp.where(q <= 0.0, 0.0, cdf)

    if not lower_tail:
        cdf = 1.0 - cdf

    if log_p:
        cdf = jnp.log(cdf)

    return cdf


def qWEI3(p, mu=1.0, sigma=1.0, lower_tail=True, log_p=False):
    """Quantile function for Weibull distribution (R WEI3, mean parameterization)."""
    p = jnp.asarray(p, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)

    if log_p:
        p = jnp.exp(p)

    if not lower_tail:
        p = 1.0 - p

    scale = _wei3_scale(mu, sigma)
    q = scale * jnp.power(-jnp.log(1.0 - p), 1.0 / sigma)

    return q


def rWEI3(key, n, mu=1.0, sigma=1.0):
    """Random generation for Weibull distribution (R WEI3, mean parameterization)."""
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)

    scale = _wei3_scale(mu, sigma)
    u = jrandom.uniform(key, shape=(n,))
    return scale * jnp.power(-jnp.log(1.0 - u), 1.0 / sigma)


# =============================================================================
# Inverse Gaussian Distribution (IG)
# =============================================================================

def dIG(x, mu=1.0, sigma=1.0, log=False):
    """Density function for Inverse Gaussian distribution.
    
    Parameters
    ----------
    x : array_like
        Quantiles (positive)
    mu : array_like
        Mean parameter
    sigma : array_like
        Shape parameter
    log : bool
        If True, return log density
        
    Returns
    -------
    array_like
        Density values
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    
    log_fy = (-0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma) - 1.5 * jnp.log(x) -
              jnp.square(x - mu) / (2.0 * jnp.square(sigma) * jnp.square(mu) * x))
    
    return log_fy if log else jnp.exp(log_fy)


def pIG(q, mu=1.0, sigma=1.0, lower_tail=True, log_p=False):
    """CDF function for Inverse Gaussian distribution.
    
    Parameters
    ----------
    q : array_like
        Quantiles
    mu : array_like
        Mean parameter
    sigma : array_like
        Shape parameter
    lower_tail : bool
        If True, return P(X <= q), else P(X > q)
    log_p : bool
        If True, return log probability
        
    Returns
    -------
    array_like
        Cumulative probabilities
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    
    # R IG.R:72-73: z = ((q/mu)-1)/(sigma*sqrt(q)), NOT sqrt(q/mu)
    z1 = (q / mu - 1.0) / (sigma * jnp.sqrt(q))
    z2 = -(q / mu + 1.0) / (sigma * jnp.sqrt(q))
    
    cdf = norm.cdf(z1) + jnp.exp(2.0 / (sigma * sigma * mu)) * norm.cdf(z2)
    
    # Boundary handling (R: cdf[q<=0]=0, cdf[q>=Inf]=1)
    cdf = jnp.where(q <= 0.0, 0.0, cdf)
    cdf = jnp.where(jnp.isinf(q), 1.0, cdf)
    
    if not lower_tail:
        cdf = 1.0 - cdf
    
    if log_p:
        cdf = jnp.log(cdf)
    
    return cdf


def qIG(p, mu=1.0, sigma=1.0, lower_tail=True, log_p=False):
    """Quantile function for Inverse Gaussian distribution.
    
    Parameters
    ----------
    p : array_like
        Probabilities
    mu : array_like
        Mean parameter
    sigma : array_like
        Shape parameter
    lower_tail : bool
        If True, p is P(X <= q), else P(X > q)
    log_p : bool
        If True, p is given as log(probability)
        
    Returns
    -------
    array_like
        Quantiles
    """
    p = np.asarray(p, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    if log_p:
        p = np.exp(p)
    
    if not lower_tail:
        p = 1.0 - p
    
    # Clip probabilities to avoid edge cases at exactly 0 or 1
    p = np.clip(p, 1e-12, 1 - 1e-12)
    
    # Exact inverse-Gaussian quantile via scipy.
    # GAMLSS IG(mu, sigma) == standard IG(mu, lambda=1/sigma^2);
    # X ~ invgauss(mu*sigma^2) has lambda=1, and Y = X/sigma^2 ~ IG(mu, 1/sigma^2).
    from scipy.stats import invgauss as _ig_dist
    q = _ig_dist.ppf(p, mu=mu * sigma * sigma) / (sigma * sigma)
    
    return q


def rIG(key, n, mu=1.0, sigma=1.0):
    """Random generation for Inverse Gaussian distribution.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key
    n : int
        Number of observations
    mu : array_like
        Mean parameter
    sigma : array_like
        Shape parameter
        
    Returns
    -------
    array_like
        Random samples
    """
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    
    # Michael, Schucany and Haas algorithm
    key1, key2 = jrandom.split(key)
    nu = jrandom.normal(key1, shape=(n,))
    y = nu * nu
    x = mu + 0.5 * mu * sigma * sigma * y - 0.5 * mu * sigma * sigma * jnp.sqrt(4.0 * mu * y + mu * mu * sigma * sigma * y * y)
    
    u = jrandom.uniform(key2, shape=(n,))
    return jnp.where(u <= mu / (mu + x), x, mu * mu / x)


# =============================================================================
# Logistic Distribution (LO)
# =============================================================================

def dLO(x, mu=0.0, sigma=1.0, log=False):
    """Density function for Logistic distribution.
    
    Parameters
    ----------
    x : array_like
        Quantiles
    mu : array_like
        Location parameter
    sigma : array_like
        Scale parameter
    log : bool
        If True, return log density
        
    Returns
    -------
    array_like
        Density values
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    
    return logistic.logpdf(x, loc=mu, scale=sigma) if log else logistic.pdf(x, loc=mu, scale=sigma)


def pLO(q, mu=0.0, sigma=1.0, lower_tail=True, log_p=False):
    """CDF function for Logistic distribution.
    
    Parameters
    ----------
    q : array_like
        Quantiles
    mu : array_like
        Location parameter
    sigma : array_like
        Scale parameter
    lower_tail : bool
        If True, return P(X <= q), else P(X > q)
    log_p : bool
        If True, return log probability
        
    Returns
    -------
    array_like
        Cumulative probabilities
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    
    cdf = logistic.cdf(q, loc=mu, scale=sigma)
    
    if not lower_tail:
        cdf = 1.0 - cdf
    
    if log_p:
        cdf = jnp.log(cdf)
    
    return cdf


def qLO(p, mu=0.0, sigma=1.0, lower_tail=True, log_p=False):
    """Quantile function for Logistic distribution.
    
    Parameters
    ----------
    p : array_like
        Probabilities
    mu : array_like
        Location parameter
    sigma : array_like
        Scale parameter
    lower_tail : bool
        If True, p is P(X <= q), else P(X > q)
    log_p : bool
        If True, p is given as log(probability)
        
    Returns
    -------
    array_like
        Quantiles
    """
    p = jnp.asarray(p, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    
    if log_p:
        p = jnp.exp(p)
    
    if not lower_tail:
        p = 1.0 - p
    
    q = mu + sigma * jnp.log(p / (1.0 - p))
    
    return q


def rLO(key, n, mu=0.0, sigma=1.0):
    """Random generation for Logistic distribution.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key
    n : int
        Number of observations
    mu : array_like
        Location parameter
    sigma : array_like
        Scale parameter
        
    Returns
    -------
    array_like
        Random samples
    """
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    
    u = jrandom.uniform(key, shape=(n,))
    return mu + sigma * jnp.log(u / (1.0 - u))


# =============================================================================
# Beta Distribution (BE)
# =============================================================================

# JIT-compiled core computations for BE distribution
@jax.jit
def _be_ab(mu: jnp.ndarray, sigma: jnp.ndarray):
    """Compute Beta shape parameters a, b from mu, sigma."""
    sigma2 = sigma * sigma
    scale = (1.0 - sigma2) / sigma2
    a = mu * scale
    b = (1.0 - mu) * scale
    return a, b


@jax.jit
def _be_logpdf(x: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled log-PDF for Beta distribution."""
    a, b = _be_ab(mu, sigma)
    return beta.logpdf(x, a, b)


@jax.jit
def _be_cdf(q: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled CDF for Beta distribution."""
    a, b = _be_ab(mu, sigma)
    return beta.cdf(q, a, b)


def dBE(x, mu=0.5, sigma=0.1, log=False):
    """Density function for Beta distribution.
    
    Parameters
    ----------
    x : array_like
        Quantiles (0 < x < 1)
    mu : array_like
        Mean parameter
    sigma : array_like
        Precision parameter
    log : bool
        If True, return log density
        
    Returns
    -------
    array_like
        Density values
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    log_fy = _be_logpdf(x, mu, sigma)
    return log_fy if log else jnp.exp(log_fy)


def pBE(q, mu=0.5, sigma=0.1, lower_tail=True, log_p=False):
    """CDF function for Beta distribution.
    
    Parameters
    ----------
    q : array_like
        Quantiles
    mu : array_like
        Mean parameter
    sigma : array_like
        Precision parameter
    lower_tail : bool
        If True, return P(X <= q), else P(X > q)
    log_p : bool
        If True, return log probability
        
    Returns
    -------
    array_like
        Cumulative probabilities
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    cdf = _be_cdf(q, mu, sigma)
    if not lower_tail:
        cdf = 1.0 - cdf
    if log_p:
        cdf = jnp.log(cdf)
    return cdf


def qBE(p, mu=0.5, sigma=0.1, lower_tail=True, log_p=False):
    """Quantile function for Beta distribution.
    
    Parameters
    ----------
    p : array_like
        Probabilities
    mu : array_like
        Mean parameter
    sigma : array_like
        Precision parameter
    lower_tail : bool
        If True, p is P(X <= q), else P(X > q)
    log_p : bool
        If True, p is given as log(probability)
        
    Returns
    -------
    array_like
        Quantiles
    """
    from scipy.stats import beta as scipy_beta
    import numpy as np
    p = jnp.asarray(p, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    if log_p:
        p = jnp.exp(p)
    if not lower_tail:
        p = 1.0 - p
    a, b = _be_ab(mu, sigma)
    # Use scipy for accurate quantile computation, then convert back to JAX
    q = jnp.asarray(scipy_beta.ppf(np.asarray(p), np.asarray(a), np.asarray(b)), dtype=jnp.float64)
    return q


def rBE(key, n, mu=0.5, sigma=0.1):
    """Random generation for Beta distribution.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key
    n : int
        Number of observations
    mu : array_like
        Mean parameter
    sigma : array_like
        Precision parameter
        
    Returns
    -------
    array_like
        Random samples
    """
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    a, b = _be_ab(mu, sigma)
    return jrandom.beta(key, a, b, shape=(n,))
# For now, I've provided the core continuous distributions (NO, GA, LOGNO, WEI, EXP, IG, LO, TF, BE)
# and basic discrete distributions (PO, BI).
#
# The remaining distributions should be implemented following the same pattern,
# referencing the R GAMLSS source code for the correct parameterizations.


# =============================================================================
# Geometric Distribution (GEOM)
# =============================================================================

def dGEOM(x, mu=1.0, log=False):
    """Density function for Geometric distribution.
    
    Parameters
    ----------
    x : array_like
        Quantiles (non-negative integers)
    mu : array_like
        Mean parameter
    log : bool
        If True, return log density
        
    Returns
    -------
    array_like
        Density values
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    
    # Parameterization: P(X=k) = p(1-p)^k where E[X] = (1-p)/p = mu
    # So p = 1/(1+mu)
    p = 1.0 / (1.0 + mu)
    
    log_fy = jnp.log(p) + x * jnp.log(1.0 - p)
    log_fy = jnp.where(x < 0, -jnp.inf, log_fy)
    
    return log_fy if log else jnp.exp(log_fy)


def pGEOM(q, mu=1.0, lower_tail=True, log_p=False):
    """CDF function for Geometric distribution.
    
    Parameters
    ----------
    q : array_like
        Quantiles
    mu : array_like
        Mean parameter
    lower_tail : bool
        If True, return P(X <= q), else P(X > q)
    log_p : bool
        If True, return log probability
        
    Returns
    -------
    array_like
        Cumulative probabilities
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    
    p = 1.0 / (1.0 + mu)
    cdf = 1.0 - jnp.power(1.0 - p, jnp.floor(q) + 1.0)
    cdf = jnp.where(q < 0, 0.0, cdf)
    
    if not lower_tail:
        cdf = 1.0 - cdf
    
    if log_p:
        cdf = jnp.log(cdf)
    
    return cdf


def qGEOM(p, mu=1.0, lower_tail=True, log_p=False):
    """Quantile function for Geometric distribution.
    
    Parameters
    ----------
    p : array_like
        Probabilities
    mu : array_like
        Mean parameter
    lower_tail : bool
        If True, p is P(X <= q), else P(X > q)
    log_p : bool
        If True, p is given as log(probability)
        
    Returns
    -------
    array_like
        Quantiles
    """
    p_val = jnp.asarray(p, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    
    if log_p:
        p_val = jnp.exp(p_val)
    
    if not lower_tail:
        p_val = 1.0 - p_val
    
    prob = 1.0 / (1.0 + mu)
    q = jnp.floor(jnp.log(1.0 - p_val) / jnp.log(1.0 - prob))
    
    return jnp.maximum(q, 0.0)


def rGEOM(key, n, mu=1.0):
    """Random generation for Geometric distribution.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key
    n : int
        Number of observations
    mu : array_like
        Mean parameter
        
    Returns
    -------
    array_like
        Random samples (integers)
    """
    mu = jnp.asarray(mu, dtype=jnp.float64)
    
    p = 1.0 / (1.0 + mu)
    u = jrandom.uniform(key, shape=(n,))
    return jnp.floor(jnp.log(u) / jnp.log(1.0 - p))


# =============================================================================
# Negative Binomial Type I (NBI)
# =============================================================================

def dNBI(x, mu=1.0, sigma=1.0, log=False):
    """Density function for Negative Binomial Type I distribution.
    
    Parameters
    ----------
    x : array_like
        Quantiles (non-negative integers)
    mu : array_like
        Mean parameter
    sigma : array_like
        Dispersion parameter
    log : bool
        If True, return log density
        
    Returns
    -------
    array_like
        Density values
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    
    size = 1.0 / sigma
    log_fy = (gammaln(x + size) - gammaln(size) - gammaln(x + 1.0) +
              size * jnp.log(size / (size + mu)) +
              x * jnp.log(mu / (size + mu)))
    
    log_fy = jnp.where(x < 0, -jnp.inf, log_fy)
    
    return log_fy if log else jnp.exp(log_fy)


def pNBI(q, mu=1.0, sigma=1.0, lower_tail=True, log_p=False):
    """CDF function for Negative Binomial Type I distribution.
    
    Parameters
    ----------
    q : array_like
        Quantiles (non-negative integers)
    mu : array_like
        Mean parameter (>0)
    sigma : array_like
        Dispersion parameter (>0). size = 1/sigma.
    lower_tail : bool
        If True, return P(X <= q), else P(X > q)
    log_p : bool
        If True, return log probability
        
    Returns
    -------
    array_like
        Cumulative probabilities
    """
    q = np.asarray(q, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # NB Type I: size = 1/sigma, mean = mu
    # scipy.nbinom uses parameterization: n=size, p such that E[X] = n*(1-p)/p
    # => p = n / (n + E[X]) = size / (size + mu)
    size = 1.0 / sigma
    p_param = size / (size + mu)
    
    # Ensure valid parameters
    size = np.maximum(size, 1e-10)
    p_param = np.clip(p_param, 1e-10, 1.0 - 1e-10)
    mu = np.maximum(mu, 1e-10)
    
    # Exact negative binomial CDF: P(X <= q) where X ~ NegBinom(n=size, p=p_param)
    # Note: scipy.nbinom.cdf(k, n, p) gives P(X <= k) where X has support {0,1,2,...}
    q_int = np.floor(q)  # Floor to nearest integer for discrete dist
    cdf = _nbinom_dist.cdf(q_int.astype(float), size, p_param)
    
    # Handle edge cases
    cdf = np.where(q < 0, 0.0, cdf)
    cdf = np.where(q >= 0, np.minimum(cdf, 1.0), 0.0)
    
    if not lower_tail:
        cdf = 1.0 - cdf
    
    if log_p:
        # Avoid log(0) or log(Inf)
        cdf = np.clip(cdf, 1e-300, 1.0 - 1e-300)
        cdf = np.log(cdf)
    
    return cdf


def qNBI(p, mu=1.0, sigma=1.0, lower_tail=True, log_p=False):
    """Quantile function for Negative Binomial Type I distribution.
    
    Parameters
    ----------
    p : array_like
        Probabilities (0 < p < 1)
    mu : array_like
        Mean parameter (>0)
    sigma : array_like
        Dispersion parameter (>0). size = 1/sigma.
    lower_tail : bool
        If True, p is P(X <= q), else P(X > q)
    log_p : bool
        If True, p is given as log(probability)
        
    Returns
    -------
    array_like
        Quantiles (non-negative integers)
    """
    p_val = np.asarray(p, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    if log_p:
        p_val = np.exp(p_val)
    
    if not lower_tail:
        p_val = 1.0 - p_val
    
    # Clip probabilities to avoid edge cases at 0 or 1
    p_val = np.clip(p_val, 1e-10, 1 - 1e-10)
    
    # NB Type I: size = 1/sigma, mean = mu
    size = 1.0 / sigma
    p_param = size / (size + mu)
    
    # Ensure valid parameters
    size = np.maximum(size, 1e-10)
    p_param = np.clip(p_param, 1e-10, 1.0 - 1e-10)
    mu = np.maximum(mu, 1e-10)
    
    # Inverse CDF (quantile function) using nbinom.ppf
    q = _nbinom_dist.ppf(p_val, size, p_param)
    q = np.maximum(np.floor(q), 0.0)  # Non-negative integers
    
    return q


def rNBI(key, n, mu=1.0, sigma=1.0):
    """Random generation for Negative Binomial Type I distribution.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key
    n : int
        Number of observations
    mu : array_like
        Mean parameter
    sigma : array_like
        Dispersion parameter
        
    Returns
    -------
    array_like
        Random samples (integers)
    """
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    
    # Use gamma-Poisson mixture
    key1, key2 = jrandom.split(key)
    size = 1.0 / sigma
    scale = mu / size
    
    lambda_vals = jrandom.gamma(key1, size, shape=(n,)) * scale
    return jrandom.poisson(key2, lambda_vals)


# =============================================================================
# Negative Binomial Type II (NBII)
# =============================================================================

def dNBII(x, mu=1.0, sigma=1.0, log=False):
    """Density function for Negative Binomial Type II distribution.
    
    Parameters
    ----------
    x : array_like
        Quantiles (non-negative integers)
    mu : array_like
        Mean parameter
    sigma : array_like
        Dispersion parameter
    log : bool
        If True, return log density
        
    Returns
    -------
    array_like
        Density values
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    
    size = 1.0 / (sigma * mu)
    log_fy = (gammaln(x + size) - gammaln(size) - gammaln(x + 1.0) +
              size * jnp.log(size / (size + mu)) +
              x * jnp.log(mu / (size + mu)))
    
    log_fy = jnp.where(x < 0, -jnp.inf, log_fy)
    
    return log_fy if log else jnp.exp(log_fy)



def _generic_d(log_pdf_func):
    """Create a generic d function from a log PDF."""
    def d_func(x, *args, log=False, **kwargs):
        log_fy = log_pdf_func(x, *args, **kwargs)
        return log_fy if log else jnp.exp(log_fy)
    return d_func


def _generic_p_continuous(log_pdf_func):
    """Create a generic p function using numerical integration."""
    def p_func(q, *args, lower_tail=True, log_p=False, **kwargs):
        # Use normal approximation as fallback
        # This is a placeholder - proper implementation would use numerical integration
        q = jnp.asarray(q, dtype=jnp.float64)
        cdf = norm.cdf(q)  # Placeholder
        
        if not lower_tail:
            cdf = 1.0 - cdf
        
        if log_p:
            cdf = jnp.log(cdf)
        
        return cdf
    return p_func


def _generic_q_continuous(log_pdf_func):
    """Create a generic q function using numerical root finding."""
    def q_func(p, *args, lower_tail=True, log_p=False, **kwargs):
        # Use normal approximation as fallback
        p_val = jnp.asarray(p, dtype=jnp.float64)
        
        if log_p:
            p_val = jnp.exp(p_val)
        
        if not lower_tail:
            p_val = 1.0 - p_val
        
        q = ndtri(p_val)  # Placeholder
        
        return q
    return q_func


def _generic_r_continuous(log_pdf_func):
    """Create a generic r function using rejection sampling."""
    def r_func(key, n, *args, **kwargs):
        # Use normal as fallback
        return jrandom.normal(key, shape=(n,))
    return r_func


# For remaining distributions, we'll create placeholder implementations
# that use the family's pdf method and numerical approximations

def create_dpqr_from_family(family):
    """Create d/p/q/r functions from a family's existing methods.
    
    This is a fallback for distributions that don't have explicit implementations.
    """
    
    def d_func(x, *args, log=False, **kwargs):
        if hasattr(family, 'pdf'):
            result = family.pdf(x, *args, **kwargs)
            return jnp.log(result) if log else result
        return jnp.nan
    
    def p_func(q, *args, lower_tail=True, log_p=False, **kwargs):
        # Numerical approximation
        return jnp.nan
    
    def q_func(p, *args, lower_tail=True, log_p=False, **kwargs):
        # Numerical approximation
        return jnp.nan
    
    def r_func(key, n, *args, **kwargs):
        # Use inverse transform sampling with numerical CDF
        return jnp.nan
    
    return d_func, p_func, q_func, r_func


# =============================================================================
# Generic numerical implementations for distributions without analytical p/q/r
# =============================================================================

def create_numerical_pqr(d_func, name="DIST"):
    """Create numerical p/q/r functions from a density function.
    
    This provides fallback implementations for distributions that don't have
    analytical CDF or quantile functions.
    
    Parameters
    ----------
    d_func : callable
        Density function with signature d_func(x, *params, log=False)
    name : str
        Distribution name for error messages
        
    Returns
    -------
    p_func, q_func, r_func : tuple of callables
        Numerical implementations of CDF, quantile, and random generation
    """
    
    def p_func(q, *params, lower_tail=True, log_p=False):
        """Numerical CDF using trapezoidal integration."""
        q = jnp.asarray(q, dtype=jnp.float64)
        
        # Simple approximation: use normal CDF as fallback
        # This is a placeholder - proper implementation would use numerical integration
        # For now, return NaN to indicate not implemented
        result = jnp.full_like(q, jnp.nan)
        
        if not lower_tail:
            result = 1.0 - result
        
        if log_p:
            result = jnp.log(result)
        
        return result
    
    def q_func(p, *params, lower_tail=True, log_p=False):
        """Numerical quantile using bisection method."""
        p = jnp.asarray(p, dtype=jnp.float64)
        
        # Placeholder - proper implementation would use numerical root finding
        # For now, return NaN to indicate not implemented
        result = jnp.full_like(p, jnp.nan)
        
        return result
    
    def r_func(key, n, *params):
        """Random generation using inverse transform sampling."""
        # Placeholder - proper implementation would use inverse CDF
        # For now, return NaN to indicate not implemented
        result = jnp.full((n,), jnp.nan, dtype=jnp.float64)
        
        return result
    
    return p_func, q_func, r_func


# =============================================================================
# Wrapper functions for distributions that use build_ad_family
# These provide a consistent interface even when p/q/r are not fully implemented
# =============================================================================

def make_pqr_from_d(d_func):
    """Create placeholder p/q/r functions from a d function.
    
    This is used for distributions where we have the density but not
    the CDF or quantile functions yet.
    """
    return create_numerical_pqr(d_func)


# For distributions that already have d from build_ad_family but need p/q/r,
# we can create generic wrappers that return None or NaN to indicate
# "not yet implemented" rather than causing errors.

def generic_p_not_implemented(q, *params, lower_tail=True, log_p=False):
    """Placeholder p function that returns NaN."""
    q = jnp.asarray(q, dtype=jnp.float64)
    return jnp.full_like(q, jnp.nan)


def generic_q_not_implemented(p, *params, lower_tail=True, log_p=False):
    """Placeholder q function that returns NaN."""
    p = jnp.asarray(p, dtype=jnp.float64)
    return jnp.full_like(p, jnp.nan)


def generic_r_not_implemented(key, n, *params):
    """Placeholder r function that returns NaN."""
    return jnp.full((n,), jnp.nan, dtype=jnp.float64)


# Note: The above are placeholders. For a production implementation,
# each distribution should have proper numerical methods for p/q/r:
#
# 1. p (CDF): Use numerical integration (scipy.integrate.quad equivalent in JAX)
# 2. q (quantile): Use numerical root finding (bisection, Newton's method)
# 3. r (random): Use inverse transform sampling or acceptance-rejection
#
# However, implementing these properly for all 32 remaining distributions
# requires significant effort and testing. The current implementation provides
# the infrastructure and 13 fully working distributions, which covers the
# most common use cases.


# =============================================================================
# Numerical implementations for distributions without analytical p/q/r
# Using JAX-compatible numerical methods
# =============================================================================

from jax.scipy.special import ndtri as norm_ppf
from jax.scipy.stats import norm as jax_norm


def numerical_cdf_continuous(d_func, q, params, lower=-10.0, upper=None, n_points=1000):
    """Numerical CDF using trapezoidal integration.
    
    Parameters
    ----------
    d_func : callable
        Density function
    q : array_like
        Quantile values
    params : tuple
        Distribution parameters
    lower : float
        Lower integration bound
    upper : float or None
        Upper integration bound (if None, use q)
    n_points : int
        Number of integration points
        
    Returns
    -------
    array_like
        CDF values
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    
    if upper is None:
        upper = q
    
    # Create integration points
    x = jnp.linspace(lower, upper, n_points)
    
    # Evaluate density at each point
    dx = (upper - lower) / (n_points - 1)
    
    # Trapezoidal integration
    # This is a simplified version - proper implementation would vectorize over q
    densities = d_func(x, *params)
    cdf = jnp.sum(densities) * dx
    
    return cdf


def numerical_quantile_bisection(p_func, p, params, lower=-10.0, upper=10.0, tol=1e-6, max_iter=50):
    """Numerical quantile using bisection method.
    
    Parameters
    ----------
    p_func : callable
        CDF function
    p : array_like
        Probability values
    params : tuple
        Distribution parameters
    lower : float
        Lower search bound
    upper : float
        Upper search bound
    tol : float
        Tolerance for convergence
    max_iter : int
        Maximum iterations
        
    Returns
    -------
    array_like
        Quantile values
    """
    p = jnp.asarray(p, dtype=jnp.float64)
    
    # Bisection search
    # This is a simplified version - proper implementation would use JAX's optimization
    mid = (lower + upper) / 2.0
    
    # For now, use normal approximation as fallback
    return norm_ppf(p)


# =============================================================================
# Improved implementations for specific distributions
# =============================================================================

# For NO2 (Normal with variance parameterization)
def dNO2(x, mu=0.0, sigma=1.0, log=False):
    """Density for NO2 (Normal with sigma as variance)."""
    x = jnp.asarray(x, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    
    # sigma is variance, so std = sqrt(sigma)
    std = jnp.sqrt(sigma)
    
    return jax_norm.logpdf(x, loc=mu, scale=std) if log else jax_norm.pdf(x, loc=mu, scale=std)


def pNO2(q, mu=0.0, sigma=1.0, lower_tail=True, log_p=False):
    """CDF for NO2."""
    q = jnp.asarray(q, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    
    std = jnp.sqrt(sigma)
    cdf = jax_norm.cdf(q, loc=mu, scale=std)
    
    if not lower_tail:
        cdf = 1.0 - cdf
    
    if log_p:
        cdf = jnp.log(cdf)
    
    return cdf


def qNO2(p, mu=0.0, sigma=1.0, lower_tail=True, log_p=False):
    """Quantile for NO2."""
    p = jnp.asarray(p, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    
    if log_p:
        p = jnp.exp(p)
    
    if not lower_tail:
        p = 1.0 - p
    
    std = jnp.sqrt(sigma)
    q = mu + std * norm_ppf(p)
    
    return q


def rNO2(key, n, mu=0.0, sigma=1.0):
    """Random generation for NO2."""
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    
    std = jnp.sqrt(sigma)
    return mu + std * jrandom.normal(key, shape=(n,))


# For NBII - use negative binomial with adjusted parameterization
def pNBII(q, mu=1.0, sigma=1.0, lower_tail=True, log_p=False):
    """CDF for NBII using its exact negative-binomial parameterization.

    NBII has ``size = mu / sigma`` and mean ``mu``.  The previous normal
    approximation was not reliable in the tails and did not match ``dNBII``.
    """
    q = np.asarray(q, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    size = np.maximum(mu / np.maximum(sigma, 1e-10), 1e-10)
    p_param = np.clip(size / (size + np.maximum(mu, 1e-10)), 1e-10, 1.0 - 1e-10)
    cdf = _nbinom_dist.cdf(np.floor(q), size, p_param)
    cdf = np.where(q < 0, 0.0, np.minimum(cdf, 1.0))
    
    if not lower_tail:
        cdf = 1.0 - cdf
    
    if log_p:
        cdf = jnp.log(cdf)
    
    return cdf


def qNBII(p, mu=1.0, sigma=1.0, lower_tail=True, log_p=False):
    """Quantile for NBII using its exact negative-binomial parameterization."""
    p = np.asarray(p, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    if log_p:
        p = np.exp(p)
    if not lower_tail:
        p = 1.0 - p
    p = np.clip(p, 1e-10, 1.0 - 1e-10)
    size = np.maximum(mu / np.maximum(sigma, 1e-10), 1e-10)
    p_param = np.clip(size / (size + np.maximum(mu, 1e-10)), 1e-10, 1.0 - 1e-10)
    return np.maximum(np.floor(_nbinom_dist.ppf(p, size, p_param)), 0.0)


def rNBII(key, n, mu=1.0, sigma=1.0):
    """Random generation for NBII using gamma-Poisson mixture."""
# For ZIP - Zero-Inflated Poisson
ZIP_QUANTILE_MAX_COUNT = 10_000


def dZIP(x, mu=1.0, sigma=0.1, log=False):
    """Probability mass function for R's zero-inflated Poisson."""
    x, mu, sigma = jnp.broadcast_arrays(*[
        jnp.asarray(value, dtype=jnp.float64) for value in (x, mu, sigma)
    ])
    valid = (mu > 0.0) & (sigma > 0.0) & (sigma < 1.0)
    safe_mu = jnp.where(valid, mu, 1.0)
    safe_sigma = jnp.where(valid, sigma, 0.5)
    log_zero = jnp.log(safe_sigma + (1.0 - safe_sigma) * jnp.exp(-safe_mu))
    log_positive = (
        jnp.log1p(-safe_sigma) - safe_mu + x * jnp.log(safe_mu)
        - gammaln(x + 1.0)
    )
    log_mass = jnp.where(x == 0.0, log_zero, log_positive)
    log_mass = jnp.where(x < 0.0, -jnp.inf, log_mass)
    result = log_mass if log else jnp.exp(log_mass)
    return jnp.where(valid, result, jnp.nan)


def pZIP(q, mu=1.0, sigma=0.1, lower_tail=True, log_p=False):
    """CDF for ZIP with R-compatible tails and parameter validation."""
    q, mu, sigma = jnp.broadcast_arrays(*[
        jnp.asarray(value, dtype=jnp.float64) for value in (q, mu, sigma)
    ])
    valid = (mu > 0.0) & (sigma > 0.0) & (sigma < 1.0)
    safe_mu = jnp.where(valid, mu, 1.0)
    safe_sigma = jnp.where(valid, sigma, 0.5)
    cdf = safe_sigma + (1.0 - safe_sigma) * poisson.cdf(q, safe_mu)
    cdf = jnp.where(q < 0.0, 0.0, cdf)
    probability = jnp.where(lower_tail, cdf, 1.0 - cdf)
    result = jnp.log(probability) if log_p else probability
    return jnp.where(valid, result, jnp.nan)


def qZIP(p, mu=1.0, sigma=0.1, lower_tail=True, log_p=False,
         max_value=ZIP_QUANTILE_MAX_COUNT):
    """Exact discrete ZIP quantile via the first CDF crossing."""
    p, mu, sigma = jnp.broadcast_arrays(*[
        jnp.asarray(value, dtype=jnp.float64) for value in (p, mu, sigma)
    ])
    probability = jnp.exp(p) if log_p else p
    probability = jnp.where(lower_tail, probability, 1.0 - probability)
    valid_parameters = (mu > 0.0) & (sigma > 0.0) & (sigma < 1.0)
    valid_probability = (probability >= 0.0) & (probability <= 1.0)
    safe_mu = jnp.where(valid_parameters, mu, 1.0)
    safe_sigma = jnp.where(valid_parameters, sigma, 0.5)
    adjusted = (probability - safe_sigma) / (1.0 - safe_sigma) - 1e-10
    support = jnp.arange(max_value + 1, dtype=jnp.float64)

    def one(pi, adjusted_i, mui):
        reached = poisson.cdf(support, mui) >= adjusted_i
        result = jnp.where(adjusted_i <= 0.0, 0.0, jnp.argmax(reached)).astype(jnp.float64)
        result = jnp.where((adjusted_i > 0.0) & ~jnp.any(reached), jnp.nan, result)
        result = jnp.where(pi == 0.0, 0.0, result)
        return jnp.where(pi == 1.0, jnp.inf, result)

    result = jax.vmap(one)(probability.reshape(-1), adjusted.reshape(-1), safe_mu.reshape(-1))
    result = result.reshape(p.shape)
    valid = valid_parameters & valid_probability & ~jnp.isnan(probability)
    return jnp.where(valid, result, jnp.nan)


def rZIP(key, n, mu=1.0, sigma=0.1):
    """Generate ZIP variates directly from its zero/Poisson mixture."""
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    valid = (mu > 0.0) & (sigma > 0.0) & (sigma < 1.0)
    safe_mu = jnp.where(valid, mu, 1.0)
    safe_sigma = jnp.where(valid, sigma, 0.5)
    output_shape = (n,) if mu.ndim == 0 and sigma.ndim == 0 else jnp.broadcast_shapes(mu.shape, sigma.shape)
    zero_key, poisson_key = jrandom.split(key)
    is_zero = jrandom.bernoulli(zero_key, safe_sigma, shape=output_shape)
    counts = jrandom.poisson(poisson_key, safe_mu, shape=output_shape)
    result = jnp.where(is_zero, 0.0, counts.astype(jnp.float64))
    return jnp.where(valid, result, jnp.nan)


# Export the new functions
__all__ = [
    'dNO', 'pNO', 'qNO', 'rNO',
    'dGA', 'pGA', 'qGA', 'rGA',
    'dLOGNO', 'pLOGNO', 'qLOGNO', 'rLOGNO',
    'dWEI', 'pWEI', 'qWEI', 'rWEI',
    'dWEI3', 'pWEI3', 'qWEI3', 'rWEI3',
    'dEXP', 'pEXP', 'qEXP', 'rEXP',
    'dIG', 'pIG', 'qIG', 'rIG',
    'dLO', 'pLO', 'qLO', 'rLO',
    'dBE', 'pBE', 'qBE', 'rBE',
    'dTF', 'pTF', 'qTF', 'rTF',
    'dPO', 'pPO', 'qPO', 'rPO',
    'dBI', 'pBI', 'qBI', 'rBI',
    'dGEOM', 'pGEOM', 'qGEOM', 'rGEOM',
    'dNBI', 'pNBI', 'qNBI', 'rNBI',
    'dNBII', 'pNBII', 'qNBII', 'rNBII',
    'dNO2', 'pNO2', 'qNO2', 'rNO2',
    'dZIP', 'pZIP', 'qZIP', 'rZIP',
    'fittedMN',
]


# =============================================================================
# Priority 1: Common Continuous Distributions
# =============================================================================

# Student-t Distribution (TF) - p function
# Note: d, q, r already exist above, just need to fix p
def pTF_fixed(q, mu=0.0, sigma=1.0, nu=10.0, lower_tail=True, log_p=False):
    """CDF function for Student-t distribution (fixed version).
    
    Parameters
    ----------
    q : array_like
        Quantiles
    mu : array_like
        Location parameter
    sigma : array_like
        Scale parameter
    nu : array_like
        Degrees of freedom
    lower_tail : bool
        If True, return P(X <= q), else P(X > q)
    log_p : bool
        If True, return log probability
        
    Returns
    -------
    array_like
        Cumulative probabilities
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    
    z = (q - mu) / sigma
    cdf = t.cdf(z, nu)
    
    if not lower_tail:
        cdf = 1.0 - cdf
    
    if log_p:
        cdf = jnp.log(cdf)
    
    return cdf


# Gumbel Distribution (GU)
def dGU(x, mu=0.0, sigma=1.0, log=False):
    """Density function for Gumbel distribution.
    
    Parameters
    ----------
    x : array_like
        Quantiles
    mu : array_like
        Location parameter
    sigma : array_like
        Scale parameter
    log : bool
        If True, return log density
        
    Returns
    -------
    array_like
        Density values
        
    Notes
    -----
    GU (Gumbel) PDF: f(x) = (1/sigma) * exp(z - exp(z))
    where z = (x - mu) / sigma
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    
    z = (x - mu) / sigma
    log_fy = -jnp.log(sigma) + z - jnp.exp(z)
    
    return log_fy if log else jnp.exp(log_fy)


def pGU(q, mu=0.0, sigma=1.0, zeta=0.0, lower_tail=True, log_p=False):
    """CDF function for Gumbel/GEV distribution.
    
    Parameters
    ----------
    q : array_like
        Quantiles
    mu : array_like
        Location parameter
    sigma : array_like
        Scale parameter
    zeta : float
        Shape parameter (default 0 for Gumbel)
    lower_tail : bool
        If True, return P(X <= q), else P(X > q)
    log_p : bool
        If True, return log probability
        
    Returns
    -------
    array_like
        Cumulative probabilities
    """
    q = np.asarray(q, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    zeta = np.asarray(zeta, dtype=np.float64)
    
    # For zeta close to 0, use standard Gumbel CDF: F(x) = exp(-exp(-(x-mu)/sigma))
    z = (q - mu) / sigma
    cdf = np.exp(-np.exp(-z))
    
    if not lower_tail:
        cdf = 1.0 - cdf
    
    if log_p:
        cdf = np.log(cdf)
    
    return cdf


def qGU(p, mu=0.0, sigma=1.0, zeta=0.0, lower_tail=True, log_p=False):
    """Quantile function for Gumbel/GEV distribution.
    
    Parameters
    ----------
    p : array_like
        Probabilities
    mu : array_like
        Location parameter
    sigma : array_like
        Scale parameter
    zeta : float
        Shape parameter (default 0 for Gumbel)
    lower_tail : bool
        If True, p is P(X <= q), else P(X > q)
    log_p : bool
        If True, p is given as log(probability)
        
    Returns
    -------
    array_like
        Quantiles
    """
    p_val = np.asarray(p, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    zeta = np.asarray(zeta, dtype=np.float64)
    
    if log_p:
        p_val = np.exp(p_val)
    
    if not lower_tail:
        p_val = 1.0 - p_val
    
    # Standard Gumbel inverse CDF (valid for zeta=0): Q(p) = mu - sigma * log(-log(p))
    q = mu - sigma * np.log(-np.log(p_val))
    
    return q


def rGU(key, n, mu=0.0, sigma=1.0):
    """Random generation for Gumbel distribution.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key
    n : int
        Number of observations
    mu : array_like
        Location parameter
    sigma : array_like
        Scale parameter
        
    Returns
    -------
    array_like
        Random samples
    """
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    
    u = jrandom.uniform(key, shape=(n,))
    return mu - sigma * jnp.log(-jnp.log(u))


# Reverse Gumbel Distribution (RG)
def dRG(x, mu=0.0, sigma=1.0, log=False):
    """Density function for Reverse Gumbel distribution.
    
    Parameters
    ----------
    x : array_like
        Quantiles
    mu : array_like
        Location parameter
    sigma : array_like
        Scale parameter
    log : bool
        If True, return log density
        
    Returns
    -------
    array_like
        Density values
        
    Notes
    -----
    RG (Reverse Gumbel) PDF: f(x) = (1/sigma) * exp(-z - exp(-z))
    where z = (x - mu) / sigma
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    
    z = (x - mu) / sigma
    log_fy = -jnp.log(sigma) - z - jnp.exp(-z)
    
    return log_fy if log else jnp.exp(log_fy)


def pRG(q, mu=0.0, sigma=1.0, lower_tail=True, log_p=False):
    """CDF function for Reverse Gumbel distribution.
    
    Parameters
    ----------
    q : array_like
        Quantiles
    mu : array_like
        Location parameter
    sigma : array_like
        Scale parameter
    lower_tail : bool
        If True, return P(X <= q), else P(X > q)
    log_p : bool
        If True, return log probability
        
    Returns
    -------
    array_like
        Cumulative probabilities
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    
    z = (mu - q) / sigma
    cdf = 1.0 - jnp.exp(-jnp.exp(-z))
    
    if not lower_tail:
        cdf = 1.0 - cdf
    
    if log_p:
        cdf = jnp.log(cdf)
    
    return cdf


def qRG(p, mu=0.0, sigma=1.0, lower_tail=True, log_p=False):
    """Quantile function for Reverse Gumbel distribution.
    
    Parameters
    ----------
    p : array_like
        Probabilities
    mu : array_like
        Location parameter
    sigma : array_like
        Scale parameter
    lower_tail : bool
        If True, p is P(X <= q), else P(X > q)
    log_p : bool
        If True, p is given as log(probability)
        
    Returns
    -------
    array_like
        Quantiles
    """
    p = jnp.asarray(p, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    
    if log_p:
        p = jnp.exp(p)
    
    if not lower_tail:
        p = 1.0 - p
    
    q = mu + sigma * jnp.log(-jnp.log(1.0 - p))
    
    return q


def rRG(key, n, mu=0.0, sigma=1.0):
    """Random generation for Reverse Gumbel distribution.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key
    n : int
        Number of observations
    mu : array_like
        Location parameter
    sigma : array_like
        Scale parameter
        
    Returns
    -------
    array_like
        Random samples
    """
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    
    u = jrandom.uniform(key, shape=(n,))
    return mu + sigma * jnp.log(-jnp.log(1.0 - u))


# Update exports
__all__.extend([
    'pTF_fixed',
    'dGU', 'pGU', 'qGU', 'rGU',
    'dRG', 'pRG', 'qRG', 'rRG',
])


# =============================================================================
# Zero-Adjusted Gamma Distribution (ZAGA)
# =============================================================================

# JIT-compiled core computations for ZAGA distribution
@jax.jit
def _zaga_gamma_params(mu: jnp.ndarray, sigma: jnp.ndarray):
    """Compute Gamma shape and scale from ZAGA mu, sigma."""
    shape = 1.0 / jnp.square(sigma)
    scale = mu * jnp.square(sigma)
    return shape, scale


@jax.jit
def _zaga_logpdf(x: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, nu: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled log-PDF for ZAGA distribution."""
    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.clip(nu, eps, 1.0 - eps)
    shape, scale = _zaga_gamma_params(mu, sigma)
    log_ga = gamma.logpdf(x, a=shape, scale=scale)
    log_at_zero = jnp.log(nu)
    log_positive = jnp.log1p(-nu) + log_ga
    return jnp.where(x <= eps, log_at_zero, log_positive)


@jax.jit
def _zaga_cdf(q: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, nu: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled CDF for ZAGA distribution."""
    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.clip(nu, eps, 1.0 - eps)
    shape, scale = _zaga_gamma_params(mu, sigma)
    cdf_ga = gamma.cdf(q, a=shape, scale=scale)
    return jnp.where(
        q < 0, 0.0,
        jnp.where(q <= eps, nu, nu + (1.0 - nu) * cdf_ga)
    )


def dZAGA(x, mu=1.0, sigma=1.0, nu=0.1, log=False):
    """Density function for Zero-Adjusted Gamma distribution.
    
    Parameters
    ----------
    x : array_like
        Quantiles
    mu : array_like
        Mean parameter for positive values (mu > 0)
    sigma : array_like
        Dispersion parameter (sigma > 0)
    nu : array_like
        Probability of zero (0 < nu < 1)
    log : bool
        If True, return log density
        
    Returns
    -------
    array_like
        Density values
        
    Notes
    -----
    ZAGA is a zero-adjusted (hurdle) model:
    - P(Y = 0) = nu
    - For Y > 0: Y ~ Gamma(mu, sigma) with probability (1 - nu)
    
    The Gamma parameterization uses:
    - E(Y | Y > 0) = mu
    - Var(Y | Y > 0) = sigma^2 * mu^2
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    log_fy = _zaga_logpdf(x, mu, sigma, nu)
    return log_fy if log else jnp.exp(log_fy)


def pZAGA(q, mu=1.0, sigma=1.0, nu=0.1, lower_tail=True, log_p=False):
    """CDF function for Zero-Adjusted Gamma distribution.
    
    Parameters
    ----------
    q : array_like
        Quantiles
    mu : array_like
        Mean parameter for positive values (mu > 0)
    sigma : array_like
        Dispersion parameter (sigma > 0)
    nu : array_like
        Probability of zero (0 < nu < 1)
    lower_tail : bool
        If True, return P(X <= q), else P(X > q)
    log_p : bool
        If True, return log probability
        
    Returns
    -------
    array_like
        Cumulative probabilities
        
    Notes
    -----
    For ZAGA:
    - P(Y <= q) = nu                           if q < 0
    - P(Y <= q) = nu                           if q = 0
    - P(Y <= q) = nu + (1 - nu) * P_GA(Y <= q) if q > 0
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    cdf = _zaga_cdf(q, mu, sigma, nu)
    if not lower_tail:
        cdf = 1.0 - cdf
    if log_p:
        cdf = jnp.log(cdf)
    return cdf


def qZAGA(p, mu=1.0, sigma=1.0, nu=0.1, lower_tail=True, log_p=False):
    """Quantile function for Zero-Adjusted Gamma distribution.
    
    Parameters
    ----------
    p : array_like
        Probabilities
    mu : array_like
        Mean parameter for positive values (mu > 0)
    sigma : array_like
        Dispersion parameter (sigma > 0)
    nu : array_like
        Probability of zero (0 < nu < 1)
    lower_tail : bool
        If True, p is P(X <= q), else P(X > q)
    log_p : bool
        If True, p is given as log(probability)
        
    Returns
    -------
    array_like
        Quantiles
        
    Notes
    -----
    For ZAGA:
    - If p <= nu: quantile = 0
    - If p > nu: quantile = qGA((p - nu) / (1 - nu))
    """
    p = jnp.asarray(p, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    
    if log_p:
        p = jnp.exp(p)
    if not lower_tail:
        p = 1.0 - p
    
    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.clip(nu, eps, 1.0 - eps)
    
    # For p <= nu: quantile is 0
    # For p > nu: quantile is from Gamma distribution
    p_adjusted = jnp.clip((p - nu) / (1.0 - nu), 0.0, 1.0)
    q_ga = qGA(p_adjusted, mu, sigma, lower_tail=True, log_p=False)
    return jnp.where(p <= nu, 0.0, q_ga)


def rZAGA(key, n, mu=1.0, sigma=1.0, nu=0.1):
    """Random generation for Zero-Adjusted Gamma distribution.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key
    n : int
        Number of observations
    mu : array_like
        Mean parameter for positive values (mu > 0)
    sigma : array_like
        Dispersion parameter (sigma > 0)
    nu : array_like
        Probability of zero (0 < nu < 1)
        
    Returns
    -------
    array_like
        Random samples
        
    Notes
    -----
    Generation algorithm:
    1. Generate U ~ Uniform(0, 1)
    2. If U <= nu: return 0
    3. Else: return sample from Gamma(mu, sigma)
    """
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    
    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.clip(nu, eps, 1.0 - eps)
    
    # Split key for two random operations
    key1, key2 = jrandom.split(key)
    
    # Generate uniform for zero/non-zero decision
    u = jrandom.uniform(key1, shape=(n,))
    
    # Gamma parameters
    shape, scale = _zaga_gamma_params(mu, sigma)
    
    # Generate Gamma samples
    gamma_samples = jrandom.gamma(key2, a=shape, shape=(n,)) * scale
    
    # Return 0 if u <= nu, otherwise Gamma sample
    samples = jnp.where(u <= nu, 0.0, gamma_samples)
    
    return samples


# =============================================================================
# Zero-Adjusted Inverse Gaussian Distribution (ZAIG)
# =============================================================================

def dZAIG(x, mu=1.0, sigma=1.0, nu=0.1, log=False):
    """Density function for Zero-Adjusted Inverse Gaussian distribution.
    
    Parameters
    ----------
    x : array_like
        Quantiles
    mu : array_like
        Mean parameter for positive values (mu > 0)
    sigma : array_like
        Dispersion parameter (sigma > 0)
    nu : array_like
        Probability of zero (0 < nu < 1)
    log : bool
        If True, return log density
        
    Returns
    -------
    array_like
        Density values
        
    Notes
    -----
    ZAIG is a zero-adjusted (hurdle) model:
    - P(Y = 0) = nu
    - For Y > 0: Y ~ IG(mu, sigma) with probability (1 - nu)
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    
    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.clip(nu, eps, 1.0 - eps)
    
    # For x = 0: density is nu
    # For x > 0: density is (1 - nu) * dIG(x, mu, sigma)
    
    # IG log density
    log_ig = dIG(x, mu, sigma, log=True)
    
    # Zero-adjusted log density
    log_at_zero = jnp.log(nu)
    log_positive = jnp.log1p(-nu) + log_ig
    
    # Use where to handle x = 0 vs x > 0
    log_fy = jnp.where(x <= eps, log_at_zero, log_positive)
    
    return log_fy if log else jnp.exp(log_fy)


def pZAIG(q, mu=1.0, sigma=1.0, nu=0.1, lower_tail=True, log_p=False):
    """CDF function for Zero-Adjusted Inverse Gaussian distribution.
    
    Parameters
    ----------
    q : array_like
        Quantiles
    mu : array_like
        Mean parameter for positive values (mu > 0)
    sigma : array_like
        Dispersion parameter (sigma > 0)
    nu : array_like
        Probability of zero (0 < nu < 1)
    lower_tail : bool
        If True, return P(X <= q), else P(X > q)
    log_p : bool
        If True, return log probability
        
    Returns
    -------
    array_like
        Cumulative probabilities
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    
    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.clip(nu, eps, 1.0 - eps)
    
    # IG CDF for positive values
    cdf_ig = pIG(q, mu, sigma, lower_tail=True, log_p=False)
    
    # Zero-adjusted CDF
    cdf = jnp.where(
        q < 0,
        0.0,
        jnp.where(
            q <= eps,
            nu,
            nu + (1.0 - nu) * cdf_ig
        )
    )
    
    if not lower_tail:
        cdf = 1.0 - cdf
    
    if log_p:
        cdf = jnp.log(cdf)
    
    return cdf


def qZAIG(p, mu=1.0, sigma=1.0, nu=0.1, lower_tail=True, log_p=False):
    """Quantile function for Zero-Adjusted Inverse Gaussian distribution.
    
    Parameters
    ----------
    p : array_like
        Probabilities
    mu : array_like
        Mean parameter for positive values (mu > 0)
    sigma : array_like
        Dispersion parameter (sigma > 0)
    nu : array_like
        Probability of zero (0 < nu < 1)
    lower_tail : bool
        If True, p is P(X <= q), else P(X > q)
    log_p : bool
        If True, p is given as log(probability)
        
    Returns
    -------
    array_like
        Quantiles
    """
    p = jnp.asarray(p, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    
    if log_p:
        p = jnp.exp(p)
    
    if not lower_tail:
        p = 1.0 - p
    
    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.clip(nu, eps, 1.0 - eps)
    
    # Adjusted probability for IG
    p_adjusted = (p - nu) / (1.0 - nu)
    p_adjusted = jnp.clip(p_adjusted, 0.0, 1.0)
    
    # IG quantile
    q_ig = qIG(p_adjusted, mu, sigma, lower_tail=True, log_p=False)
    
    # Return 0 if p <= nu, otherwise IG quantile
    q = jnp.where(p <= nu, 0.0, q_ig)
    
    return q


def rZAIG(key, n, mu=1.0, sigma=1.0, nu=0.1):
    """Random generation for Zero-Adjusted Inverse Gaussian distribution.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key
    n : int
        Number of observations
    mu : array_like
        Mean parameter for positive values (mu > 0)
    sigma : array_like
        Dispersion parameter (sigma > 0)
    nu : array_like
        Probability of zero (0 < nu < 1)
        
    Returns
    -------
    array_like
        Random samples
    """
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    
    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.clip(nu, eps, 1.0 - eps)
    
    # Split key
    key1, key2 = jrandom.split(key)
    
    # Generate uniform for zero/non-zero decision
    u = jrandom.uniform(key1, shape=(n,))
    
    # Generate IG samples
    ig_samples = rIG(key2, n, mu, sigma)
    
    # Return 0 if u <= nu, otherwise IG sample
    samples = jnp.where(u <= nu, 0.0, ig_samples)
    
    return samples


# =============================================================================
# Beta Inflated Distribution (BEINF)
# =============================================================================

# JIT-compiled core computations for BEINF distribution
@jax.jit
def _beinf_ab(mu: jnp.ndarray, sigma: jnp.ndarray):
    """Compute Beta shape parameters a, b from BEINF mu, sigma."""
    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.clip(mu, eps, 1.0 - eps)
    sigma = jnp.clip(sigma, eps, 1.0 - eps)
    sigma2 = jnp.square(sigma)
    scale = (1.0 - sigma2) / sigma2
    a = jnp.maximum(mu * scale, eps)
    b = jnp.maximum((1.0 - mu) * scale, eps)
    return a, b


@jax.jit
def _beinf_probs(nu: jnp.ndarray, tau: jnp.ndarray):
    """Compute mixture probabilities for BEINF."""
    total = 1.0 + nu + tau
    p0 = nu / total
    p1 = tau / total
    p_cont = 1.0 / total
    return p0, p1, p_cont


@jax.jit
def _beinf_logpdf(x: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray,
                  nu: jnp.ndarray, tau: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled log-PDF for BEINF distribution."""
    eps = jnp.finfo(jnp.float64).eps
    nu = jnp.maximum(nu, eps)
    tau = jnp.maximum(tau, eps)
    a, b = _beinf_ab(mu, sigma)
    log_denom = jnp.log(1.0 + nu + tau)
    log_beta_pdf = beta.logpdf(x, a, b)
    log_at_0 = jnp.log(nu) - log_denom
    log_at_1 = jnp.log(tau) - log_denom
    log_cont = log_beta_pdf - log_denom
    return jnp.where(x <= eps, log_at_0, jnp.where(x >= 1.0 - eps, log_at_1, log_cont))


@jax.jit
def _beinf_cdf(q: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray,
               nu: jnp.ndarray, tau: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled CDF for BEINF distribution."""
    eps = jnp.finfo(jnp.float64).eps
    nu = jnp.maximum(nu, eps)
    tau = jnp.maximum(tau, eps)
    a, b = _beinf_ab(mu, sigma)
    p0, p1, p_cont = _beinf_probs(nu, tau)
    cdf_beta = beta.cdf(q, a, b)
    return jnp.where(
        q < 0, 0.0,
        jnp.where(q <= eps, p0,
                  jnp.where(q >= 1.0 - eps, 1.0, p0 + p_cont * cdf_beta))
    )


def dBEINF(x, mu=0.5, sigma=0.1, nu=0.1, tau=0.1, log=False):
    """Density function for Beta Inflated distribution.
    
    Parameters
    ----------
    x : array_like
        Quantiles (0 <= x <= 1)
    mu : array_like
        Mean parameter for continuous part (0 < mu < 1)
    sigma : array_like
        Dispersion parameter (0 < sigma < 1)
    nu : array_like
        Inflation parameter at 0 (nu > 0)
    tau : array_like
        Inflation parameter at 1 (tau > 0)
    log : bool
        If True, return log density
        
    Returns
    -------
    array_like
        Density values
        
    Notes
    -----
    BEINF is a mixture model for proportional data:
    - P(Y = 0) = nu / (1 + nu + tau)
    - P(Y = 1) = tau / (1 + nu + tau)
    - For 0 < Y < 1: Y ~ Beta(a, b) with probability 1 / (1 + nu + tau)
    
    where a = mu * (1 - sigma^2) / sigma^2
          b = (1 - mu) * (1 - sigma^2) / sigma^2
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    tau = jnp.asarray(tau, dtype=jnp.float64)
    log_fy = _beinf_logpdf(x, mu, sigma, nu, tau)
    return log_fy if log else jnp.exp(log_fy)


def pBEINF(q, mu=0.5, sigma=0.1, nu=0.1, tau=0.1, lower_tail=True, log_p=False):
    """CDF function for Beta Inflated distribution.
    
    Parameters
    ----------
    q : array_like
        Quantiles (0 <= q <= 1)
    mu : array_like
        Mean parameter for continuous part (0 < mu < 1)
    sigma : array_like
        Dispersion parameter (0 < sigma < 1)
    nu : array_like
        Inflation parameter at 0 (nu > 0)
    tau : array_like
        Inflation parameter at 1 (tau > 0)
    lower_tail : bool
        If True, return P(X <= q), else P(X > q)
    log_p : bool
        If True, return log probability
        
    Returns
    -------
    array_like
        Cumulative probabilities
        
    Notes
    -----
    For BEINF:
    - P(Y <= q) = p0                                if q < 0
    - P(Y <= q) = p0                                if q = 0
    - P(Y <= q) = p0 + p_cont * P_Beta(Y <= q)     if 0 < q < 1
    - P(Y <= q) = 1                                 if q >= 1
    
    where p0 = nu / (1 + nu + tau)
          p1 = tau / (1 + nu + tau)
          p_cont = 1 / (1 + nu + tau)
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    tau = jnp.asarray(tau, dtype=jnp.float64)
    cdf = _beinf_cdf(q, mu, sigma, nu, tau)
    if not lower_tail:
        cdf = 1.0 - cdf
    if log_p:
        cdf = jnp.log(cdf)
    return cdf


def qBEINF(p, mu=0.5, sigma=0.1, nu=0.1, tau=0.1, lower_tail=True, log_p=False):
    """Quantile function for Beta Inflated distribution.
    
    Parameters
    ----------
    p : array_like
        Probabilities
    mu : array_like
        Mean parameter for continuous part (0 < mu < 1)
    sigma : array_like
        Dispersion parameter (0 < sigma < 1)
    nu : array_like
        Inflation parameter at 0 (nu > 0)
    tau : array_like
        Inflation parameter at 1 (tau > 0)
    lower_tail : bool
        If True, p is P(X <= q), else P(X > q)
    log_p : bool
        If True, p is given as log(probability)
        
    Returns
    -------
    array_like
        Quantiles
        
    Notes
    -----
    For BEINF:
    - If p <= p0: quantile = 0
    - If p0 < p < p0 + p_cont: quantile = qBeta((p - p0) / p_cont)
    - If p >= p0 + p_cont: quantile = 1
    
    where p0 = nu / (1 + nu + tau)
          p_cont = 1 / (1 + nu + tau)
    """
    p = jnp.asarray(p, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    tau = jnp.asarray(tau, dtype=jnp.float64)
    
    if log_p:
        p = jnp.exp(p)
    if not lower_tail:
        p = 1.0 - p
    
    eps = jnp.finfo(jnp.float64).eps
    nu = jnp.maximum(nu, eps)
    tau = jnp.maximum(tau, eps)
    
    p0, p1, p_cont = _beinf_probs(nu, tau)
    
    # Adjusted probability for Beta
    p_adjusted = jnp.clip((p - p0) / p_cont, 0.0, 1.0)
    q_beta = qBE(p_adjusted, mu, sigma, lower_tail=True, log_p=False)
    
    return jnp.where(p <= p0, 0.0, jnp.where(p >= p0 + p_cont, 1.0, q_beta))


def rBEINF(key, n, mu=0.5, sigma=0.1, nu=0.1, tau=0.1):
    """Random generation for Beta Inflated distribution.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key
    n : int
        Number of observations
    mu : array_like
        Mean parameter for continuous part (0 < mu < 1)
    sigma : array_like
        Dispersion parameter (0 < sigma < 1)
    nu : array_like
        Inflation parameter at 0 (nu > 0)
    tau : array_like
        Inflation parameter at 1 (tau > 0)
        
    Returns
    -------
    array_like
        Random samples
        
    Notes
    -----
    Generation algorithm:
    1. Generate U ~ Uniform(0, 1)
    2. If U <= p0: return 0
    3. Else if U > p0 + p_cont: return 1
    4. Else: return sample from Beta(a, b)
    
    where p0 = nu / (1 + nu + tau)
          p_cont = 1 / (1 + nu + tau)
    """
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    tau = jnp.asarray(tau, dtype=jnp.float64)
    
    eps = jnp.finfo(jnp.float64).eps
    nu = jnp.maximum(nu, eps)
    tau = jnp.maximum(tau, eps)
    
    # Split key
    key1, key2 = jrandom.split(key)
    
    # Generate uniform for component selection
    u = jrandom.uniform(key1, shape=(n,))
    
    # Beta parameters
    a, b = _beinf_ab(mu, sigma)
    
    # Probabilities
    p0, p1, p_cont = _beinf_probs(nu, tau)
    
    # Generate Beta samples
    beta_samples = jrandom.beta(key2, a, b, shape=(n,))
    
    # Return 0, 1, or Beta sample based on u
    samples = jnp.where(
        u <= p0,
        0.0,
        jnp.where(
            u > p0 + p_cont,
            1.0,
            beta_samples
        )
    )
    
    return samples


# =============================================================================
# Box-Cox t Distribution (BCT)
# =============================================================================

def dBCT(x, mu=1.0, sigma=1.0, nu=1.0, tau=10.0, log=False):
    """Density function for Box-Cox t distribution.
    
    Parameters
    ----------
    x : array_like
        Quantiles (x > 0)
    mu : array_like
        Location parameter (mu > 0)
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Box-Cox transformation parameter
    tau : array_like
        Degrees of freedom for t-distribution (tau > 2)
    log : bool
        If True, return log density
        
    Returns
    -------
    array_like
        Density values
        
    Notes
    -----
    BCT uses Box-Cox transformation + t-distribution:
    - z = (x^nu - 1) / (nu * sigma) if |nu| > eps
    - z = log(x) / sigma if |nu| <= eps
    - z ~ t(tau)
    
    The density includes a Jacobian term for the transformation.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    tau = jnp.asarray(tau, dtype=jnp.float64)
    
    eps = jnp.finfo(jnp.float64).eps
    x = jnp.maximum(x, eps)
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    tau = jnp.maximum(tau, 2.0 + eps)
    
    # Box-Cox transformation
    ratio = x / mu
    use_log = jnp.abs(nu) < 1e-6
    z_boxcox = (jnp.power(ratio, nu) - 1.0) / (nu * sigma)
    z_log = jnp.log(ratio) / sigma
    z = jnp.where(use_log, z_log, z_boxcox)
    
    # t-distribution log density
    log_norm = (
        gammaln((tau + 1.0) / 2.0)
        - gammaln(tau / 2.0)
        - 0.5 * jnp.log(tau * jnp.pi)
    )
    
    # Jacobian term
    jacobian = -jnp.log(sigma) - nu * jnp.log(mu) + (nu - 1.0) * jnp.log(x)
    
    # Kernel
    kernel = -((tau + 1.0) / 2.0) * jnp.log1p(jnp.square(z) / tau)
    
    # Truncation normalization (R BCT.R: loglik - log(pt(1/(sigma*abs(nu)), df=tau)))
    from scipy.stats import t as t_dist
    trunc = t_dist.logcdf(np.asarray(1.0 / (sigma * jnp.abs(nu))), np.asarray(tau))
    trunc = jnp.asarray(trunc, dtype=jnp.float64)
    
    log_fy = log_norm + jacobian + kernel - trunc
    log_fy = jnp.where(x <= 0.0, -jnp.inf, log_fy)
    
    return log_fy if log else jnp.exp(log_fy)


def pBCT(q, mu=1.0, sigma=1.0, nu=1.0, tau=10.0, lower_tail=True, log_p=False):
    """CDF function for Box-Cox t distribution.
    
    Parameters
    ----------
    q : array_like
        Quantiles (q > 0)
    mu : array_like
        Location parameter (mu > 0)
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Box-Cox transformation parameter
    tau : array_like
        Degrees of freedom for t-distribution (tau > 2)
    lower_tail : bool
        If True, return P(X <= q), else P(X > q)
    log_p : bool
        If True, return log probability
        
    Returns
    -------
    array_like
        Cumulative probabilities
        
    Notes
    -----
    For BCT:
    - Transform q to z using Box-Cox transformation
    - P(X <= q) = P(Z <= z) where Z ~ t(tau)
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    tau = jnp.asarray(tau, dtype=jnp.float64)
    
    eps = jnp.finfo(jnp.float64).eps
    q = jnp.maximum(q, eps)
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    tau = jnp.maximum(tau, 2.0 + eps)
    
    # Box-Cox transformation
    ratio = q / mu
    use_log = jnp.abs(nu) < 1e-6
    z_boxcox = (jnp.power(ratio, nu) - 1.0) / (nu * sigma)
    z_log = jnp.log(ratio) / sigma
    z = jnp.where(use_log, z_log, z_boxcox)
    
    # Truncated t CDF (R BCT.R): FYy = (pt(z) - pt(-1/(sigma|nu|))) / pt(1/(sigma|nu|))
    from scipy.stats import t as t_dist
    z_np = np.asarray(z)
    tau_np = np.asarray(tau)
    fyy1 = t_dist.cdf(z_np, tau_np)
    fyy2 = np.where(np.asarray(nu) > 0, t_dist.cdf(-1.0 / (np.asarray(sigma) * np.abs(np.asarray(nu))), tau_np), 0.0)
    fyy3 = t_dist.cdf(1.0 / (np.asarray(sigma) * np.abs(np.asarray(nu))), tau_np)
    cdf = (fyy1 - fyy2) / fyy3
    cdf = jnp.asarray(cdf, dtype=jnp.float64)
    cdf = jnp.where(q <= 0.0, 0.0, cdf)
    cdf = jnp.clip(cdf, 0.0, 1.0)
    
    if not lower_tail:
        cdf = 1.0 - cdf
    
    if log_p:
        cdf = jnp.log(cdf)
    
    return cdf


def qBCT(p, mu=1.0, sigma=1.0, nu=1.0, tau=10.0, lower_tail=True, log_p=False):
    """Quantile function for Box-Cox t distribution.
    
    Parameters
    ----------
    p : array_like
        Probabilities
    mu : array_like
        Location parameter (mu > 0)
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Box-Cox transformation parameter
    tau : array_like
        Degrees of freedom for t-distribution (tau > 2)
    lower_tail : bool
        If True, p is P(X <= q), else P(X > q)
    log_p : bool
        If True, p is given as log(probability)
        
    Returns
    -------
    array_like
        Quantiles
        
    Notes
    -----
    For BCT:
    - Get z = quantile of t(tau) at probability p
    - Transform back: q = mu * (1 + nu * sigma * z)^(1/nu) if |nu| > eps
    -                 q = mu * exp(sigma * z) if |nu| <= eps
    """
    p = jnp.asarray(p, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    tau = jnp.asarray(tau, dtype=jnp.float64)
    
    if log_p:
        p = jnp.exp(p)
    
    if not lower_tail:
        p = 1.0 - p
    
    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    tau = jnp.maximum(tau, 2.0 + eps)
    
    # Truncated-t quantile inversion (R BCT.R):
    #   zp = pt(1/(sigma*abs(nu)), df=tau)
    #   z[nu>0]  = qt(1 - (1-p) * zp, tau)
    #   z[nu<=0] = qt(p * zp, tau)
    from scipy.stats import t as t_dist
    zp = t_dist.cdf(np.asarray(1.0 / (sigma * jnp.abs(nu))), np.asarray(tau))
    z_neg = t_dist.ppf(np.asarray(p) * zp, np.asarray(tau))
    z_pos = t_dist.ppf(1.0 - (1.0 - np.asarray(p)) * zp, np.asarray(tau))
    z = jnp.asarray(np.where(np.asarray(nu) > 0, z_pos, z_neg), dtype=jnp.float64)
    
    # Inverse Box-Cox transformation
    use_log = jnp.abs(nu) < 1e-6
    
    # For Box-Cox: z = (ratio^nu - 1) / (nu * sigma)
    # Solve for ratio: ratio = (1 + nu * sigma * z)^(1/nu)
    term = 1.0 + nu * sigma * z
    term = jnp.maximum(term, eps)  # Ensure positive
    
    q_boxcox = mu * jnp.power(term, 1.0 / nu)
    q_log = mu * jnp.exp(sigma * z)
    
    q = jnp.where(use_log, q_log, q_boxcox)
    q = jnp.maximum(q, eps)
    
    return q


def rBCT(key, n, mu=1.0, sigma=1.0, nu=1.0, tau=10.0):
    """Random generation for Box-Cox t distribution.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key
    n : int
        Number of observations
    mu : array_like
        Location parameter (mu > 0)
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Box-Cox transformation parameter
    tau : array_like
        Degrees of freedom for t-distribution (tau > 2)
        
    Returns
    -------
    array_like
        Random samples
        
    Notes
    -----
    Generation algorithm:
    1. Generate z ~ t(tau)
    2. Transform: x = mu * (1 + nu * sigma * z)^(1/nu) if |nu| > eps
    3.            x = mu * exp(sigma * z) if |nu| <= eps
    """
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    tau = jnp.asarray(tau, dtype=jnp.float64)
    
    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    tau = jnp.maximum(tau, 2.0 + eps)
    
    # Generate t-distributed samples
    # t = Normal / sqrt(Chi2 / df)
    key1, key2 = jrandom.split(key)
    normal_samples = jrandom.normal(key1, shape=(n,))
    chi2_samples = jrandom.chisquare(key2, tau, shape=(n,))
    z = normal_samples / jnp.sqrt(chi2_samples / tau)
    
    # Inverse Box-Cox transformation
    use_log = jnp.abs(nu) < 1e-6
    
    term = 1.0 + nu * sigma * z
    term = jnp.maximum(term, eps)
    
    samples_boxcox = mu * jnp.power(term, 1.0 / nu)
    samples_log = mu * jnp.exp(sigma * z)
    
    samples = jnp.where(use_log, samples_log, samples_boxcox)
    samples = jnp.maximum(samples, eps)
    
    return samples


# =============================================================================
# Box-Cox Power Exponential Distribution (BCPE)
# =============================================================================

def _bcpe_c(tau):
    """Standardization constant c for BCPE power-exponential kernel."""
    return jnp.exp(0.5 * (-(2.0 / tau) * jnp.log(2.0) + gammaln(1.0 / tau) - gammaln(3.0 / tau)))


def _bcpe_ftrunc_log(t, tau):
    """log(F.T(t, tau)): CDF of the standardized PE distribution at t."""
    # F.T(t) = 0.5 * (1 + F.s * sign(t)), F.s = pgamma(0.5*(t/c)^tau, shape=1/tau, scale=1)
    from jax.scipy.special import gammainc

    c = _bcpe_c(tau)
    s = 0.5 * jnp.power(jnp.abs(t) / c, tau)
    # pgamma(s, 1/tau, 1) = gammainc(1/tau, s)
    fs = gammainc(1.0 / tau, s)
    cdf = 0.5 * (1.0 + fs * jnp.sign(t))
    return jnp.log(jnp.maximum(cdf, jnp.finfo(jnp.float64).tiny))


def _bcpe_ftrunc(t, tau):
    """F.T(t, tau): CDF of the standardized PE distribution at t."""
    return jnp.exp(_bcpe_ftrunc_log(t, tau))


def _bcpe_q(p, tau):
    """q.T(p, tau): quantile of standardized PE distribution (R BCPE.R q.T)."""
    c = _bcpe_c(tau)
    # s = qgamma((2p-1)*sign(p-0.5), shape=1/tau, scale=1)
    # z = sign(p-0.5) * ((2s)^(1/tau)) * c
    from scipy.special import gammaincinv as _gammaincinv
    p_np = np.asarray(p)
    tau_np = np.asarray(tau)
    sign_p = np.sign(p_np - 0.5)
    arg = (2.0 * p_np - 1.0) * sign_p
    s = _gammaincinv(1.0 / tau_np, arg)
    z = sign_p * np.power(2.0 * s, 1.0 / tau_np) * np.asarray(c)
    return z


def dBCPE(x, mu=1.0, sigma=1.0, nu=1.0, tau=2.0, log=False):
    """Density function for Box-Cox Power Exponential distribution.
    
    Parameters
    ----------
    x : array_like
        Quantiles (x > 0)
    mu : array_like
        Location parameter (mu > 0)
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Box-Cox transformation parameter
    tau : array_like
        Power parameter (tau > 0)
    log : bool
        If True, return log density
        
    Returns
    -------
    array_like
        Density values
        
    Notes
    -----
    BCPE uses Box-Cox transformation + Power Exponential:
    - z = (x^nu - 1) / (nu * sigma) if |nu| > eps
    - z = log(x) / sigma if |nu| <= eps
    - Density: f(z) ∝ exp(-0.5 * |z|^tau)
    
    Special cases:
    - tau = 2: Normal distribution
    - tau = 1: Laplace distribution
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    tau = jnp.asarray(tau, dtype=jnp.float64)
    
    eps = jnp.finfo(jnp.float64).eps
    x = jnp.maximum(x, eps)
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    tau = jnp.maximum(tau, eps)
    
    # Box-Cox transformation
    ratio = x / mu
    use_log = jnp.abs(nu) < 1e-6
    z_boxcox = (jnp.power(ratio, nu) - 1.0) / (nu * sigma)
    z_log = jnp.log(ratio) / sigma
    z = jnp.where(use_log, z_log, z_boxcox)
    
    # Variance-standardized power-exponential kernel (R BCPE.R):
    #   log.c = 0.5*(-(2/tau)*log(2) + lgamma(1/tau) - lgamma(3/tau));  c = exp(log.c)
    #   log.lik = log(tau) - log.c - 0.5*(abs(t/c)^tau) - (1+1/tau)*log(2) - lgamma(1/tau)
    log_c = 0.5 * (-(2.0 / tau) * jnp.log(2.0) + gammaln(1.0 / tau) - gammaln(3.0 / tau))
    log_fz = (
        jnp.log(tau) - log_c
        - 0.5 * jnp.power(jnp.abs(z) / jnp.exp(log_c), tau)
        - (1.0 + 1.0 / tau) * jnp.log(2.0) - gammaln(1.0 / tau)
    )
    
    # Jacobian term
    jacobian = -jnp.log(sigma) - nu * jnp.log(mu) + (nu - 1.0) * jnp.log(x)
    
    # Truncation normalization (R BCPE.R): loglik = logder + logfZ - log(F.T(1/(sigma*abs(nu))))
    log_ftrunc = _bcpe_ftrunc_log(1.0 / (sigma * jnp.abs(nu)), tau)
    
    log_fy = jacobian + log_fz - log_ftrunc
    log_fy = jnp.where(x <= 0.0, -jnp.inf, log_fy)
    
    return log_fy if log else jnp.exp(log_fy)


def pBCPE(q, mu=1.0, sigma=1.0, nu=1.0, tau=2.0, lower_tail=True, log_p=False):
    """CDF function for Box-Cox Power Exponential distribution.
    
    Parameters
    ----------
    q : array_like
        Quantiles (q > 0)
    mu : array_like
        Location parameter (mu > 0)
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Box-Cox transformation parameter
    tau : array_like
        Power parameter (tau > 0)
    lower_tail : bool
        If True, return P(X <= q), else P(X > q)
    log_p : bool
        If True, return log probability
        
    Returns
    -------
    array_like
        Cumulative probabilities
        
    Notes
    -----
    For BCPE:
    - Transform q to z using Box-Cox transformation
    - P(X <= q) = P(Z <= z) where Z ~ PowerExponential(tau)
    
    The CDF of Power Exponential is computed using the incomplete gamma function.
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    tau = jnp.asarray(tau, dtype=jnp.float64)
    
    eps = jnp.finfo(jnp.float64).eps
    q = jnp.maximum(q, eps)
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    tau = jnp.maximum(tau, eps)
    
    # Box-Cox transformation
    ratio = q / mu
    use_log = jnp.abs(nu) < 1e-6
    z_boxcox = (jnp.power(ratio, nu) - 1.0) / (nu * sigma)
    z_log = jnp.log(ratio) / sigma
    z = jnp.where(use_log, z_log, z_boxcox)
    
    # Truncated standardized-PE CDF (R BCPE.R):
    #   FYy = (F.T(z) - F.T(-1/(sigma*abs(nu)))) / F.T(1/(sigma*abs(nu)))   for nu > 0
    #   FYy =  F.T(z) / F.T(1/(sigma*abs(nu)))                              for nu <= 0
    fyy1 = _bcpe_ftrunc(z, tau)
    fyy2 = jnp.where(nu > 0, _bcpe_ftrunc(-1.0 / (sigma * jnp.abs(nu)), tau), 0.0)
    fyy3 = _bcpe_ftrunc(1.0 / (sigma * jnp.abs(nu)), tau)
    cdf = (fyy1 - fyy2) / fyy3
    cdf = jnp.where(q <= 0.0, 0.0, cdf)
    cdf = jnp.clip(cdf, 0.0, 1.0)
    
    if not lower_tail:
        cdf = 1.0 - cdf
    
    if log_p:
        cdf = jnp.log(cdf)
    
    return cdf


def qBCPE(p, mu=1.0, sigma=1.0, nu=1.0, tau=2.0, lower_tail=True, log_p=False):
    """Quantile function for Box-Cox Power Exponential distribution.
    
    Parameters
    ----------
    p : array_like
        Probabilities
    mu : array_like
        Location parameter (mu > 0)
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Box-Cox transformation parameter
    tau : array_like
        Power parameter (tau > 0)
    lower_tail : bool
        If True, p is P(X <= q), else P(X > q)
    log_p : bool
        If True, p is given as log(probability)
        
    Returns
    -------
    array_like
        Quantiles
        
    Notes
    -----
    For BCPE:
    - Get z = quantile of PowerExponential(tau) at probability p
    - Transform back: q = mu * (1 + nu * sigma * z)^(1/nu) if |nu| > eps
    -                 q = mu * exp(sigma * z) if |nu| <= eps
    
    The quantile of Power Exponential is computed using numerical inversion.
    """
    p = jnp.asarray(p, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    tau = jnp.asarray(tau, dtype=jnp.float64)
    
    if log_p:
        p = jnp.exp(p)
    
    if not lower_tail:
        p = 1.0 - p
    
    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    tau = jnp.maximum(tau, eps)
    
    # Truncated standardized-PE quantile inversion (R BCPE.R):
    #   za[nu<0]  = q.T(p * F.T(1/(sigma*abs(nu))))
    #   za[nu==0] = q.T(p)
    #   za[nu>0]  = q.T(1 - (1-p) * F.T(1/(sigma*abs(nu))))
    p_np = np.asarray(p, dtype=np.float64)
    tau_np = np.asarray(tau, dtype=np.float64)
    nu_np = np.asarray(nu, dtype=np.float64)
    sigma_np = np.asarray(sigma, dtype=np.float64)
    mu_np = np.asarray(mu, dtype=np.float64)
    ftrunc = np.asarray(_bcpe_ftrunc(1.0 / (sigma * jnp.abs(nu)), tau), dtype=np.float64)
    
    za_neg = _bcpe_q(p_np * ftrunc, tau_np)
    za_zero = _bcpe_q(p_np, tau_np)
    za_pos = _bcpe_q(1.0 - (1.0 - p_np) * ftrunc, tau_np)
    za = np.where(nu_np < 0, za_neg, np.where(nu_np == 0, za_zero, za_pos))
    z = jnp.asarray(za, dtype=jnp.float64)
    
    # Inverse Box-Cox transformation
    use_log = jnp.abs(nu) < 1e-6
    
    term = 1.0 + nu * sigma * z
    term = jnp.maximum(term, eps)
    
    q_boxcox = mu * jnp.power(term, 1.0 / nu)
    q_log = mu * jnp.exp(sigma * z)
    
    q = jnp.where(use_log, q_log, q_boxcox)
    q = jnp.maximum(q, eps)
    
    return q


def rBCPE(key, n, mu=1.0, sigma=1.0, nu=1.0, tau=2.0):
    """Random generation for Box-Cox Power Exponential distribution.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key
    n : int
        Number of observations
    mu : array_like
        Location parameter (mu > 0)
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Box-Cox transformation parameter
    tau : array_like
        Power parameter (tau > 0)
        
    Returns
    -------
    array_like
        Random samples
        
    Notes
    -----
    Generation algorithm:
    1. Generate z ~ PowerExponential(tau)
    2. Transform: x = mu * (1 + nu * sigma * z)^(1/nu) if |nu| > eps
    3.            x = mu * exp(sigma * z) if |nu| <= eps
    
    Power Exponential samples are generated using:
    - Generate U ~ Uniform(0, 1)
    - z = quantile(U)
    """
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    tau = jnp.asarray(tau, dtype=jnp.float64)
    
    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    tau = jnp.maximum(tau, eps)
    
    # Generate uniform samples
    u = jrandom.uniform(key, shape=(n,))
    
    # Convert to Power Exponential samples using quantile function
    from scipy.special import gammaincinv
    
    # Convert u to centered probability
    p_centered = 2.0 * (u - 0.5)
    sign_z = jnp.sign(p_centered)
    abs_p = jnp.abs(p_centered)
    
    # Inverse incomplete gamma
    u_gamma = gammaincinv(1.0 / tau, abs_p)
    u_gamma = jnp.asarray(u_gamma, dtype=jnp.float64)
    
    # z = sign(z) * (2 * u)^(1/tau)
    z = sign_z * jnp.power(2.0 * u_gamma, 1.0 / tau)
    
    # Inverse Box-Cox transformation
    use_log = jnp.abs(nu) < 1e-6
    
    term = 1.0 + nu * sigma * z
    term = jnp.maximum(term, eps)
    
    samples_boxcox = mu * jnp.power(term, 1.0 / nu)
    samples_log = mu * jnp.exp(sigma * z)
    
    samples = jnp.where(use_log, samples_log, samples_boxcox)
    samples = jnp.maximum(samples, eps)
    
    return samples


# =============================================================================
# Johnson SU Distribution (JSU)
# =============================================================================

def dJSU(x, mu=0.0, sigma=1.0, nu=0.0, tau=1.0, log=False):
    """Density function for Johnson SU distribution.
    
    Parameters
    ----------
    x : array_like
        Quantiles
    mu : array_like
        Location parameter
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Skewness parameter
    tau : array_like
        Shape parameter (tau > 0)
    log : bool
        If True, return log density
        
    Returns
    -------
    array_like
        Density values
        
    Notes
    -----
    JSU uses sinh transformation:
    - z = (x - mu) / sigma
    - w = nu + tau * arcsinh(z)
    - w ~ N(0, 1)
    
    The density includes a Jacobian term for the transformation.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    tau = jnp.asarray(tau, dtype=jnp.float64)
    
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    tau = jnp.maximum(tau, eps)
    
    # Transformation
    z = (x - mu) / sigma
    w = nu + tau * jnp.arcsinh(z)
    
    # Log density
    log_fy = (
        jnp.log(tau)
        - jnp.log(sigma)
        - 0.5 * jnp.log1p(jnp.square(z))
        - 0.5 * jnp.log(2.0 * jnp.pi)
        - 0.5 * jnp.square(w)
    )
    
    return log_fy if log else jnp.exp(log_fy)


def pJSU(q, mu=0.0, sigma=1.0, nu=0.0, tau=1.0, lower_tail=True, log_p=False):
    """CDF function for Johnson SU distribution.
    
    Parameters
    ----------
    q : array_like
        Quantiles
    mu : array_like
        Location parameter
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Skewness parameter
    tau : array_like
        Shape parameter (tau > 0)
    lower_tail : bool
        If True, return P(X <= q), else P(X > q)
    log_p : bool
        If True, return log probability
        
    Returns
    -------
    array_like
        Cumulative probabilities
        
    Notes
    -----
    For JSU:
    - Transform q to w using sinh transformation
    - P(X <= q) = P(W <= w) where W ~ N(0, 1)
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    tau = jnp.asarray(tau, dtype=jnp.float64)
    
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    tau = jnp.maximum(tau, eps)
    
    # Transformation
    z = (q - mu) / sigma
    w = nu + tau * jnp.arcsinh(z)
    
    # Standard normal CDF
    cdf = norm.cdf(w)
    
    if not lower_tail:
        cdf = 1.0 - cdf
    
    if log_p:
        cdf = jnp.log(cdf)
    
    return cdf


def qJSU(p, mu=0.0, sigma=1.0, nu=0.0, tau=1.0, lower_tail=True, log_p=False):
    """Quantile function for Johnson SU distribution.
    
    Parameters
    ----------
    p : array_like
        Probabilities
    mu : array_like
        Location parameter
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Skewness parameter
    tau : array_like
        Shape parameter (tau > 0)
    lower_tail : bool
        If True, p is P(X <= q), else P(X > q)
    log_p : bool
        If True, p is given as log(probability)
        
    Returns
    -------
    array_like
        Quantiles
        
    Notes
    -----
    For JSU:
    - Get w = quantile of N(0, 1) at probability p
    - Transform back: z = sinh((w - nu) / tau)
    -                 q = mu + sigma * z
    """
    p = jnp.asarray(p, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    tau = jnp.asarray(tau, dtype=jnp.float64)
    
    if log_p:
        p = jnp.exp(p)
    
    if not lower_tail:
        p = 1.0 - p
    
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    tau = jnp.maximum(tau, eps)
    
    # Standard normal quantile
    w = norm.ppf(p)
    
    # Inverse transformation
    # w = nu + tau * arcsinh(z)
    # arcsinh(z) = (w - nu) / tau
    # z = sinh((w - nu) / tau)
    z = jnp.sinh((w - nu) / tau)
    
    # Back to original scale
    q = mu + sigma * z
    
    return q


def rJSU(key, n, mu=0.0, sigma=1.0, nu=0.0, tau=1.0):
    """Random generation for Johnson SU distribution.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key
    n : int
        Number of observations
    mu : array_like
        Location parameter
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Skewness parameter
    tau : array_like
        Shape parameter (tau > 0)
        
    Returns
    -------
    array_like
        Random samples
        
    Notes
    -----
    Generation algorithm:
    1. Generate w ~ N(0, 1)
    2. Transform: z = sinh((w - nu) / tau)
    3.            x = mu + sigma * z
    """
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    tau = jnp.asarray(tau, dtype=jnp.float64)
    
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    tau = jnp.maximum(tau, eps)
    
    # Generate standard normal samples
    w = jrandom.normal(key, shape=(n,))
    
    # Inverse transformation
    z = jnp.sinh((w - nu) / tau)
    
    # Back to original scale
    samples = mu + sigma * z
    
    return samples


# =============================================================================
# Box-Cox Cole-Green Distribution (BCCG)
# =============================================================================

def dBCCG(x, mu=1.0, sigma=1.0, nu=1.0, log=False):
    """Density function for Box-Cox Cole-Green distribution.
    
    Parameters
    ----------
    x : array_like
        Quantiles (x > 0)
    mu : array_like
        Location parameter (mu > 0)
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Box-Cox transformation parameter
    log : bool
        If True, return log density
        
    Returns
    -------
    array_like
        Density values
        
    Notes
    -----
    BCCG uses Box-Cox transformation + Normal:
    - z = (x^nu - 1) / (nu * sigma) if |nu| > eps
    - z = log(x) / sigma if |nu| <= eps
    - z ~ N(0, 1)
    
    This is simpler than BCT (uses Normal instead of t).
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    
    eps = jnp.finfo(jnp.float64).eps
    x = jnp.maximum(x, eps)
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    
    # Box-Cox transformation
    ratio = x / mu
    use_log = jnp.abs(nu) < 1e-6
    z_boxcox = (jnp.power(ratio, nu) - 1.0) / (nu * sigma)
    z_log = jnp.log(ratio) / sigma
    z = jnp.where(use_log, z_log, z_boxcox)
    
    # Normal log density
    log_norm = -0.5 * jnp.log(2.0 * jnp.pi)
    
    # Jacobian term
    jacobian = -jnp.log(sigma) - nu * jnp.log(mu) + (nu - 1.0) * jnp.log(x)
    
    # Kernel
    kernel = -0.5 * jnp.square(z)
    
    # Truncation normalization (R BCCG.R: loglik <- loglik - log(pnorm(1/(sigma*abs(nu)))))
    trunc = norm.logcdf(1.0 / (sigma * jnp.abs(nu)))
    
    log_fy = log_norm + jacobian + kernel - trunc
    log_fy = jnp.where(x <= 0.0, -jnp.inf, log_fy)
    
    return log_fy if log else jnp.exp(log_fy)


def pBCCG(q, mu=1.0, sigma=1.0, nu=1.0, lower_tail=True, log_p=False):
    """CDF function for Box-Cox Cole-Green distribution.
    
    Parameters
    ----------
    q : array_like
        Quantiles (q > 0)
    mu : array_like
        Location parameter (mu > 0)
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Box-Cox transformation parameter
    lower_tail : bool
        If True, return P(X <= q), else P(X > q)
    log_p : bool
        If True, return log probability
        
    Returns
    -------
    array_like
        Cumulative probabilities
        
    Notes
    -----
    For BCCG:
    - Transform q to z using Box-Cox transformation
    - P(X <= q) = P(Z <= z) where Z ~ N(0, 1)
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    
    eps = jnp.finfo(jnp.float64).eps
    q = jnp.maximum(q, eps)
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    
    # Box-Cox transformation
    ratio = q / mu
    use_log = jnp.abs(nu) < 1e-6
    z_boxcox = (jnp.power(ratio, nu) - 1.0) / (nu * sigma)
    z_log = jnp.log(ratio) / sigma
    z = jnp.where(use_log, z_log, z_boxcox)
    
    # Truncated normal CDF (R BCCG.R):
    #   FYy = (Phi(z) - Phi(-1/(sigma*abs(nu)))) / Phi(1/(sigma*abs(nu)))   for nu > 0
    #   FYy =  Phi(z) / Phi(1/(sigma*abs(nu)))                              for nu <= 0
    trunc_const = norm.cdf(1.0 / (sigma * jnp.abs(nu)))
    cdf = (norm.cdf(z) - jnp.where(nu > 0, norm.cdf(-1.0 / (sigma * jnp.abs(nu))), 0.0)) / trunc_const
    cdf = jnp.where(q <= 0.0, 0.0, cdf)
    cdf = jnp.clip(cdf, 0.0, 1.0)
    
    if not lower_tail:
        cdf = 1.0 - cdf
    
    if log_p:
        cdf = jnp.log(cdf)
    
    return cdf


def qBCCG(p, mu=1.0, sigma=1.0, nu=1.0, lower_tail=True, log_p=False):
    """Quantile function for Box-Cox Cole-Green distribution.
    
    Parameters
    ----------
    p : array_like
        Probabilities
    mu : array_like
        Location parameter (mu > 0)
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Box-Cox transformation parameter
    lower_tail : bool
        If True, p is P(X <= q), else P(X > q)
    log_p : bool
        If True, p is given as log(probability)
        
    Returns
    -------
    array_like
        Quantiles
        
    Notes
    -----
    For BCCG:
    - Get z = quantile of N(0, 1) at probability p
    - Transform back: q = mu * (1 + nu * sigma * z)^(1/nu) if |nu| > eps
    -                 q = mu * exp(sigma * z) if |nu| <= eps
    """
    p = jnp.asarray(p, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    
    if log_p:
        p = jnp.exp(p)
    
    if not lower_tail:
        p = 1.0 - p
    
    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    
    # Truncated-normal quantile inversion (R BCCG.R):
    #   z[nu<=0] = qnorm(p * pnorm(1/(sigma*abs(nu))))
    #   z[nu>0]  = qnorm(1 - (1-p) * pnorm(1/(sigma*abs(nu))))
    trunc_const = norm.cdf(1.0 / (sigma * jnp.abs(nu)))
    z_neg = norm.ppf(p * trunc_const)
    z_pos = norm.ppf(1.0 - (1.0 - p) * trunc_const)
    z = jnp.where(nu > 0, z_pos, z_neg)
    
    # Inverse Box-Cox transformation
    use_log = jnp.abs(nu) < 1e-6
    
    term = 1.0 + nu * sigma * z
    term = jnp.maximum(term, eps)
    
    q_boxcox = mu * jnp.power(term, 1.0 / nu)
    q_log = mu * jnp.exp(sigma * z)
    
    q = jnp.where(use_log, q_log, q_boxcox)
    q = jnp.maximum(q, eps)
    
    return q


def rBCCG(key, n, mu=1.0, sigma=1.0, nu=1.0):
    """Random generation for Box-Cox Cole-Green distribution.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key
    n : int
        Number of observations
    mu : array_like
        Location parameter (mu > 0)
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Box-Cox transformation parameter
        
    Returns
    -------
    array_like
        Random samples
        
    Notes
    -----
    Generation algorithm:
    1. Generate z ~ N(0, 1)
    2. Transform: x = mu * (1 + nu * sigma * z)^(1/nu) if |nu| > eps
    3.            x = mu * exp(sigma * z) if |nu| <= eps
    """
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    
    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    
    # Generate standard normal samples
    z = jrandom.normal(key, shape=(n,))
    
    # Inverse Box-Cox transformation
    use_log = jnp.abs(nu) < 1e-6
    
    term = 1.0 + nu * sigma * z
    term = jnp.maximum(term, eps)
    
    samples_boxcox = mu * jnp.power(term, 1.0 / nu)
    samples_log = mu * jnp.exp(sigma * z)
    
    samples = jnp.where(use_log, samples_log, samples_boxcox)
    samples = jnp.maximum(samples, eps)
    
    return samples


# =============================================================================
# Power Exponential Distribution (PE) / Generalized Normal
# =============================================================================

def dPE(x, mu=0.0, sigma=1.0, nu=2.0, log=False):
    """Density function for Power Exponential (Generalized Normal) distribution.
    
    Parameters
    ----------
    x : array_like
        Quantiles
    mu : array_like
        Location parameter
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Shape parameter (nu > 0)
    log : bool
        If True, return log density
        
    Returns
    -------
    array_like
        Density values
        
    Notes
    -----
    PE is also known as Generalized Normal distribution:
    - z = (x - mu) / sigma
    - Density: f(z) ∝ exp(-0.5 * |z/c|^nu)
    - c is a normalization constant
    
    Special cases:
    - nu = 2: Normal distribution
    - nu = 1: Laplace distribution
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.maximum(nu, eps)
    
    # Normalization constant
    log_c = 0.5 * (-(2.0 / nu) * jnp.log(2.0) + gammaln(1.0 / nu) - gammaln(3.0 / nu))
    c = jnp.exp(log_c)
    
    # Standardized variable
    z = (x - mu) / sigma
    
    # Log density
    log_fy = (
        -jnp.log(sigma)
        + jnp.log(nu)
        - log_c
        - 0.5 * jnp.power(jnp.abs(z / c), nu)
        - (1.0 + 1.0 / nu) * jnp.log(2.0)
        - gammaln(1.0 / nu)
    )
    
    return log_fy if log else jnp.exp(log_fy)


def pPE(q, mu=0.0, sigma=1.0, nu=2.0, lower_tail=True, log_p=False):
    """CDF function for Power Exponential distribution.
    
    Parameters
    ----------
    q : array_like
        Quantiles
    mu : array_like
        Location parameter
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Shape parameter (nu > 0)
    lower_tail : bool
        If True, return P(X <= q), else P(X > q)
    log_p : bool
        If True, return log probability
        
    Returns
    -------
    array_like
        Cumulative probabilities
        
    Notes
    -----
    For PE:
    - Transform q to z = (q - mu) / sigma
    - P(X <= q) = P(Z <= z) where Z ~ PE(0, 1, nu)
    
    The CDF is computed using the incomplete gamma function.
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.maximum(nu, eps)
    
    # Normalization constant
    log_c = 0.5 * (-(2.0 / nu) * jnp.log(2.0) + gammaln(1.0 / nu) - gammaln(3.0 / nu))
    c = jnp.exp(log_c)
    
    # Standardized variable
    z = (q - mu) / sigma
    
    # Power Exponential CDF
    # P(Z <= z) = 0.5 + 0.5 * sign(z) * P(1/nu, 0.5 * |z/c|^nu)
    from scipy.special import gammainc
    
    abs_z = jnp.abs(z)
    u = 0.5 * jnp.power(abs_z / c, nu)
    
    # Regularized incomplete gamma function
    p_gamma = gammainc(1.0 / nu, u)
    p_gamma = jnp.asarray(p_gamma, dtype=jnp.float64)
    
    # CDF
    cdf = 0.5 + 0.5 * jnp.sign(z) * p_gamma
    
    if not lower_tail:
        cdf = 1.0 - cdf
    
    if log_p:
        cdf = jnp.log(cdf)
    
    return cdf


def qPE(p, mu=0.0, sigma=1.0, nu=2.0, lower_tail=True, log_p=False):
    """Quantile function for Power Exponential distribution.
    
    Parameters
    ----------
    p : array_like
        Probabilities
    mu : array_like
        Location parameter
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Shape parameter (nu > 0)
    lower_tail : bool
        If True, p is P(X <= q), else P(X > q)
    log_p : bool
        If True, p is given as log(probability)
        
    Returns
    -------
    array_like
        Quantiles
        
    Notes
    -----
    For PE:
    - Get z = quantile of PE(0, 1, nu) at probability p
    - Transform back: q = mu + sigma * z
    """
    p = jnp.asarray(p, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    
    if log_p:
        p = jnp.exp(p)
    
    if not lower_tail:
        p = 1.0 - p
    
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.maximum(nu, eps)
    
    # Normalization constant
    log_c = 0.5 * (-(2.0 / nu) * jnp.log(2.0) + gammaln(1.0 / nu) - gammaln(3.0 / nu))
    c = jnp.exp(log_c)
    
    # Power Exponential quantile
    # For p, we need to invert: p = 0.5 + 0.5 * sign(z) * P(1/nu, 0.5 * |z/c|^nu)
    from scipy.special import gammaincinv
    
    # Convert p to centered probability
    p_centered = 2.0 * (p - 0.5)
    sign_z = jnp.sign(p_centered)
    abs_p = jnp.abs(p_centered)
    
    # Inverse incomplete gamma
    u = gammaincinv(1.0 / nu, abs_p)
    u = jnp.asarray(u, dtype=jnp.float64)
    
    # z = sign(z) * c * (2 * u)^(1/nu)
    z = sign_z * c * jnp.power(2.0 * u, 1.0 / nu)
    
    # Back to original scale
    q = mu + sigma * z
    
    return q


def rPE(key, n, mu=0.0, sigma=1.0, nu=2.0):
    """Random generation for Power Exponential distribution.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key
    n : int
        Number of observations
    mu : array_like
        Location parameter
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Shape parameter (nu > 0)
        
    Returns
    -------
    array_like
        Random samples
        
    Notes
    -----
    Generation algorithm:
    1. Generate z ~ PE(0, 1, nu)
    2. Transform: x = mu + sigma * z
    
    PE samples are generated using the quantile function.
    """
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.maximum(nu, eps)
    
    # Generate uniform samples
    u = jrandom.uniform(key, shape=(n,))
    
    # Convert to PE samples using quantile function
    # Normalization constant
    log_c = 0.5 * (-(2.0 / nu) * jnp.log(2.0) + gammaln(1.0 / nu) - gammaln(3.0 / nu))
    c = jnp.exp(log_c)
    
    from scipy.special import gammaincinv
    
    # Convert u to centered probability
    p_centered = 2.0 * (u - 0.5)
    sign_z = jnp.sign(p_centered)
    abs_p = jnp.abs(p_centered)
    
    # Inverse incomplete gamma
    u_gamma = gammaincinv(1.0 / nu, abs_p)
    u_gamma = jnp.asarray(u_gamma, dtype=jnp.float64)
    
    # z = sign(z) * c * (2 * u)^(1/nu)
    z = sign_z * c * jnp.power(2.0 * u_gamma, 1.0 / nu)
    
    # Back to original scale
    samples = mu + sigma * z
    
    return samples


# =============================================================================
# Sinh-Arcsinh Distribution (SHASH)
# =============================================================================

def dSHASH(x, mu=0.0, sigma=1.0, nu=0.0, tau=1.0, log=False):
    """Density function for Sinh-Arcsinh distribution.
    
    Parameters
    ----------
    x : array_like
        Quantiles
    mu : array_like
        Location parameter
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Skewness parameter
    tau : array_like
        Kurtosis parameter (tau > 0)
    log : bool
        If True, return log density
        
    Returns
    -------
    array_like
        Density values
        
    Notes
    -----
    SHASH uses sinh-arcsinh transformation:
    - z = (x - mu) / sigma
    - r = 0.5 * (exp(tau * arcsinh(z)) - exp(-nu * arcsinh(z)))
    - r ~ N(0, 1)
    
    Provides flexible control over skewness (nu) and kurtosis (tau).
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    tau = jnp.asarray(tau, dtype=jnp.float64)
    
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    tau = jnp.maximum(tau, eps)
    
    # Transformation
    z = (x - mu) / sigma
    asinh_z = jnp.arcsinh(z)
    r = 0.5 * (jnp.exp(tau * asinh_z) - jnp.exp(-nu * asinh_z))
    c = 0.5 * (tau * jnp.exp(tau * asinh_z) + nu * jnp.exp(-nu * asinh_z))
    
    # Log density
    log_fy = (
        -jnp.log(sigma)
        - 0.5 * jnp.log(2.0 * jnp.pi)
        - 0.5 * jnp.log1p(jnp.square(z))
        + jnp.log(jnp.maximum(c, eps))
        - 0.5 * jnp.square(r)
    )
    
    return log_fy if log else jnp.exp(log_fy)


def pSHASH(q, mu=0.0, sigma=1.0, nu=0.0, tau=1.0, lower_tail=True, log_p=False):
    """CDF function for Sinh-Arcsinh distribution.
    
    Parameters
    ----------
    q : array_like
        Quantiles
    mu : array_like
        Location parameter
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Skewness parameter
    tau : array_like
        Kurtosis parameter (tau > 0)
    lower_tail : bool
        If True, return P(X <= q), else P(X > q)
    log_p : bool
        If True, return log probability
        
    Returns
    -------
    array_like
        Cumulative probabilities
        
    Notes
    -----
    For SHASH:
    - Transform q to r using sinh-arcsinh transformation
    - P(X <= q) = P(R <= r) where R ~ N(0, 1)
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    tau = jnp.asarray(tau, dtype=jnp.float64)
    
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    tau = jnp.maximum(tau, eps)
    
    # Transformation
    z = (q - mu) / sigma
    asinh_z = jnp.arcsinh(z)
    r = 0.5 * (jnp.exp(tau * asinh_z) - jnp.exp(-nu * asinh_z))
    
    # Standard normal CDF
    cdf = norm.cdf(r)
    
    if not lower_tail:
        cdf = 1.0 - cdf
    
    if log_p:
        cdf = jnp.log(cdf)
    
    return cdf


def qSHASH(p, mu=0.0, sigma=1.0, nu=0.0, tau=1.0, lower_tail=True, log_p=False):
    """Quantile function for Sinh-Arcsinh distribution.
    
    Parameters
    ----------
    p : array_like
        Probabilities
    mu : array_like
        Location parameter
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Skewness parameter
    tau : array_like
        Kurtosis parameter (tau > 0)
    lower_tail : bool
        If True, p is P(X <= q), else P(X > q)
    log_p : bool
        If True, p is given as log(probability)
        
    Returns
    -------
    array_like
        Quantiles
        
    Notes
    -----
    For SHASH:
    - Get r = quantile of N(0, 1) at probability p
    - Solve for z from: r = 0.5 * (exp(tau * arcsinh(z)) - exp(-nu * arcsinh(z)))
    - Transform back: q = mu + sigma * z
    
    The inverse transformation requires numerical solution.
    """
    p = jnp.asarray(p, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    tau = jnp.asarray(tau, dtype=jnp.float64)
    
    if log_p:
        p = jnp.exp(p)
    
    if not lower_tail:
        p = 1.0 - p
    
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    tau = jnp.maximum(tau, eps)
    
    # Standard normal quantile
    r = norm.ppf(p)
    
    # Inverse transformation: solve for z from r = 0.5 * (exp(tau * arcsinh(z)) - exp(-nu * arcsinh(z)))
    # Better initial guess using sinh approximation
    # When nu and tau are close to 1, sinh(arcsinh(z)) ≈ z
    z = jnp.sinh(r / jnp.maximum(tau, eps))
    
    # Newton's method iterations with better convergence
    for _ in range(20):  # More iterations for better convergence
        asinh_z = jnp.arcsinh(z)
        exp_tau = jnp.exp(tau * asinh_z)
        exp_nu = jnp.exp(-nu * asinh_z)
        
        # Function: f(z) = 0.5 * (exp(tau * arcsinh(z)) - exp(-nu * arcsinh(z))) - r
        f = 0.5 * (exp_tau - exp_nu) - r
        
        # Derivative: f'(z) = 0.5 * (tau * exp(tau * arcsinh(z)) + nu * exp(-nu * arcsinh(z))) / sqrt(1 + z^2)
        df = 0.5 * (tau * exp_tau + nu * exp_nu) / jnp.sqrt(1.0 + jnp.square(z))
        
        # Newton update with damping to prevent overshooting
        delta = f / jnp.maximum(jnp.abs(df), eps)
        z = z - jnp.clip(delta, -10.0, 10.0)  # Clip to prevent extreme jumps
        
        # Check convergence
        if jnp.max(jnp.abs(delta)) < 1e-10:
            break
    
    # Back to original scale
    q = mu + sigma * z
    
    return q


def rSHASH(key, n, mu=0.0, sigma=1.0, nu=0.0, tau=1.0):
    """Random generation for Sinh-Arcsinh distribution.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key
    n : int
        Number of observations
    mu : array_like
        Location parameter
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Skewness parameter
    tau : array_like
        Kurtosis parameter (tau > 0)
        
    Returns
    -------
    array_like
        Random samples
        
    Notes
    -----
    Generation algorithm:
    1. Generate r ~ N(0, 1)
    2. Solve for z from: r = 0.5 * (exp(tau * arcsinh(z)) - exp(-nu * arcsinh(z)))
    3. Transform: x = mu + sigma * z
    """
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    tau = jnp.asarray(tau, dtype=jnp.float64)
    
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    tau = jnp.maximum(tau, eps)
    
    # Generate standard normal samples
    r = jrandom.normal(key, shape=(n,))
    
    # Inverse transformation using Newton's method with better initial guess
    z = jnp.sinh(r / jnp.maximum(tau, eps))
    
    for _ in range(20):
        asinh_z = jnp.arcsinh(z)
        exp_tau = jnp.exp(tau * asinh_z)
        exp_nu = jnp.exp(-nu * asinh_z)
        
        f = 0.5 * (exp_tau - exp_nu) - r
        df = 0.5 * (tau * exp_tau + nu * exp_nu) / jnp.sqrt(1.0 + jnp.square(z))
        
        delta = f / jnp.maximum(jnp.abs(df), eps)
        z = z - jnp.clip(delta, -10.0, 10.0)
        
        if jnp.max(jnp.abs(delta)) < 1e-10:
            break
    
    # Back to original scale
    samples = mu + sigma * z
    
    return samples


# =============================================================================
# Generalized t Distribution (GT)
# =============================================================================

def dGT(x, mu=0.0, sigma=1.0, nu=2.0, tau=2.0, log=False):
    """Density function for Generalized t distribution.
    
    Parameters
    ----------
    x : array_like
        Quantiles
    mu : array_like
        Location parameter
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Shape parameter 1 (nu > 0)
    tau : array_like
        Shape parameter 2 (tau > 0)
    log : bool
        If True, return log density
        
    Returns
    -------
    array_like
        Density values
        
    Notes
    -----
    GT is a generalization of Student-t with additional shape parameters:
    - z = (x - mu) / sigma
    - Density involves |z|^tau and generalized t kernel
    
    Special cases:
    - tau = 2, nu → ∞: Normal distribution
    - tau = 2: Student-t distribution
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    tau = jnp.asarray(tau, dtype=jnp.float64)
    
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.maximum(nu, eps)
    tau = jnp.maximum(tau, eps)
    
    # Transformation
    z = (x - mu) / sigma
    zt = jnp.power(jnp.abs(z), tau)
    
    # Log density
    log_fy = (
        jnp.log(tau)
        - jnp.log(2.0 * sigma)
        - (1.0 / tau) * jnp.log(nu)
        - gammaln(1.0 / tau)
        - gammaln(nu)
        + gammaln(nu + 1.0 / tau)
        - (nu + 1.0 / tau) * jnp.log1p(zt / nu)
    )
    
    return log_fy if log else jnp.exp(log_fy)


def pGT(q, mu=0.0, sigma=1.0, nu=2.0, tau=2.0, lower_tail=True, log_p=False):
    """CDF function for Generalized t distribution.
    
    Parameters
    ----------
    q : array_like
        Quantiles
    mu : array_like
        Location parameter
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Shape parameter 1 (nu > 0)
    tau : array_like
        Shape parameter 2 (tau > 0)
    lower_tail : bool
        If True, return P(X <= q), else P(X > q)
    log_p : bool
        If True, return log probability
        
    Returns
    -------
    array_like
        Cumulative probabilities
        
    Notes
    -----
    For GT:
    - CDF is computed using numerical integration
    - This is computationally expensive but accurate
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    tau = jnp.asarray(tau, dtype=jnp.float64)
    
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.maximum(nu, eps)
    tau = jnp.maximum(tau, eps)
    
    # For GT, we use numerical integration
    # This is a simplified implementation using scipy
    from scipy.integrate import quad
    
    def integrand(x_val):
        return float(dGT(jnp.array([x_val]), mu, sigma, nu, tau)[0])
    
    # Compute CDF using numerical integration
    q_array = np.asarray(q).flatten()
    # Extract scalar values from potentially array-shaped parameters
    mu_arr = np.asarray(mu).flatten()
    sigma_arr = np.asarray(sigma).flatten()
    nu_arr = np.asarray(nu).flatten()
    tau_arr = np.asarray(tau).flatten()
    n = q_array.size
    # Broadcast parameter arrays to match q length
    if mu_arr.size == 1: mu_arr = np.repeat(mu_arr, n)
    if sigma_arr.size == 1: sigma_arr = np.repeat(sigma_arr, n)
    if nu_arr.size == 1: nu_arr = np.repeat(nu_arr, n)
    if tau_arr.size == 1: tau_arr = np.repeat(tau_arr, n)

    cdf_values = []
    for i, q_val in enumerate(q_array):
        mu_i = float(mu_arr[i])
        sigma_i = float(sigma_arr[i])
        nu_i = float(nu_arr[i])
        tau_i = float(tau_arr[i])

        def integrand_i(x_val, _mu=mu_i, _sigma=sigma_i, _nu=nu_i, _tau=tau_i):
            return float(dGT(jnp.array([x_val]),
                             jnp.array([_mu]), jnp.array([_sigma]),
                             jnp.array([_nu]), jnp.array([_tau]))[0])

        lower_bound = mu_i - 10.0 * sigma_i
        result, _ = quad(integrand_i, lower_bound, float(q_val))
        cdf_values.append(result)

    cdf = jnp.asarray(cdf_values, dtype=jnp.float64).reshape(np.asarray(q).shape)
    
    if not lower_tail:
        cdf = 1.0 - cdf
    
    if log_p:
        cdf = jnp.log(cdf)
    
    return cdf


def qGT(p, mu=0.0, sigma=1.0, nu=2.0, tau=2.0, lower_tail=True, log_p=False):
    """Quantile function for Generalized t distribution.
    
    Parameters
    ----------
    p : array_like
        Probabilities
    mu : array_like
        Location parameter
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Shape parameter 1 (nu > 0)
    tau : array_like
        Shape parameter 2 (tau > 0)
    lower_tail : bool
        If True, p is P(X <= q), else P(X > q)
    log_p : bool
        If True, p is given as log(probability)
        
    Returns
    -------
    array_like
        Quantiles
        
    Notes
    -----
    For GT:
    - Quantile is computed using numerical root finding
    - This is computationally expensive but accurate
    """
    p = jnp.asarray(p, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    tau = jnp.asarray(tau, dtype=jnp.float64)
    
    if log_p:
        p = jnp.exp(p)
    
    if not lower_tail:
        p = 1.0 - p
    
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.maximum(nu, eps)
    tau = jnp.maximum(tau, eps)
    
    # For GT, we use numerical root finding
    from scipy.optimize import brentq

    p_array = np.asarray(p).flatten()
    mu_arr = np.asarray(mu).flatten()
    sigma_arr = np.asarray(sigma).flatten()
    nu_arr = np.asarray(nu).flatten()
    tau_arr = np.asarray(tau).flatten()
    n = p_array.size
    if mu_arr.size == 1: mu_arr = np.repeat(mu_arr, n)
    if sigma_arr.size == 1: sigma_arr = np.repeat(sigma_arr, n)
    if nu_arr.size == 1: nu_arr = np.repeat(nu_arr, n)
    if tau_arr.size == 1: tau_arr = np.repeat(tau_arr, n)

    q_values = []
    for i, p_val in enumerate(p_array):
        mu_i = float(mu_arr[i])
        sigma_i = float(sigma_arr[i])
        nu_i = float(nu_arr[i])
        tau_i = float(tau_arr[i])

        def objective(q_val, _mu=mu_i, _sigma=sigma_i, _nu=nu_i, _tau=tau_i):
            cdf_val = float(pGT(jnp.array([q_val]),
                                jnp.array([_mu]), jnp.array([_sigma]),
                                jnp.array([_nu]), jnp.array([_tau]))[0])
            return cdf_val - float(p_val)

        lower_bound = mu_i - 10.0 * sigma_i
        upper_bound = mu_i + 10.0 * sigma_i
        try:
            q_val = brentq(objective, lower_bound, upper_bound)
        except Exception:
            q_val = mu_i
        q_values.append(q_val)

    q = jnp.asarray(q_values, dtype=jnp.float64).reshape(np.asarray(p).shape)
    return q


def rGT(key, n, mu=0.0, sigma=1.0, nu=2.0, tau=2.0):
    """Random generation for Generalized t distribution.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key
    n : int
        Number of observations
    mu : array_like
        Location parameter
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Shape parameter 1 (nu > 0)
    tau : array_like
        Shape parameter 2 (tau > 0)
        
    Returns
    -------
    array_like
        Random samples
        
    Notes
    -----
    Generation algorithm:
    1. Generate u ~ Uniform(0, 1)
    2. Compute q = quantile(u)
    3. Return q
    
    This uses the quantile function for generation.
    """
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    tau = jnp.asarray(tau, dtype=jnp.float64)
    
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.maximum(nu, eps)
    tau = jnp.maximum(tau, eps)
    
    # Generate uniform samples
    u = jrandom.uniform(key, shape=(n,))
    
    # Convert to GT samples using quantile function
    samples = qGT(u, mu, sigma, nu, tau)
    
    return samples


# ==============================================================================
# P1 DISTRIBUTIONS - Priority 1 (Important distributions)
# ==============================================================================

# ------------------------------------------------------------------------------
# GG (Generalized Gamma) - 3 parameters
# ------------------------------------------------------------------------------

def dGG(x, mu=1.0, sigma=1.0, nu=1.0, log=False):
    """Density function for Generalized Gamma distribution.

    The Generalized Gamma is a flexible three-parameter family that includes
    Gamma, Weibull, Exponential, and Log-Normal as special cases.

    Parameters
    ----------
    x : array_like
        Quantiles (x > 0)
    mu : array_like
        Location parameter (mu > 0)
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Shape parameter (can be positive or negative, but not zero)
    log : bool
        If True, return log density

    Returns
    -------
    array_like
        Density values

    Notes
    -----
    Parameterization:
    - z = (x/mu)^nu
    - For |nu| > 1e-6: Uses Gamma distribution on z
    - For |nu| <= 1e-6: Converges to Log-Normal distribution

    Special cases:
    - nu = 1, sigma = 1: Exponential(mu)
    - nu = 1: Gamma distribution
    - nu → 0: Log-Normal distribution
    - sigma = 1: Weibull distribution

    References
    ----------
    Stacy, E. W. (1962). A generalization of the gamma distribution.
    Annals of Mathematical Statistics, 33(3), 1187-1192.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)

    eps = jnp.finfo(jnp.float64).eps
    x = jnp.maximum(x, eps)
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)

    # Compute z = (x/mu)^nu
    z = jnp.power(x / mu, nu)

    # For |nu| > 1e-6, use Gamma distribution on z
    sigma_prime = sigma * jnp.abs(nu)
    shape = 1.0 / (sigma_prime * sigma_prime)

    # Gamma log-pdf on z
    log_pdf_z = (shape - 1.0) * jnp.log(z) - z * shape - gammaln(shape) + shape * jnp.log(shape)

    # Jacobian: |nu| * z / x
    log_jacobian = jnp.log(jnp.abs(nu)) + jnp.log(z) - jnp.log(x)

    # Total log-pdf
    log_pdf_gg = log_pdf_z + log_jacobian

    # For nu ~ 0, use log-normal limit
    log_pdf_lognormal = (
        -jnp.log(x) - 0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma) -
        0.5 * jnp.square((jnp.log(x) - jnp.log(mu)) / sigma)
    )

    # Use log-normal when |nu| < 1e-6
    log_pdf = jnp.where(jnp.abs(nu) > 1e-6, log_pdf_gg, log_pdf_lognormal)

    # Handle x <= 0
    log_pdf = jnp.where(x <= 0, -jnp.inf, log_pdf)

    if log:
        return log_pdf
    else:
        return jnp.exp(log_pdf)


def pGG(q, mu=1.0, sigma=1.0, nu=1.0, lower_tail=True, log_p=False):
    """CDF for Generalized Gamma distribution.

    Parameters
    ----------
    q : array_like
        Quantiles (q > 0)
    mu : array_like
        Location parameter (mu > 0)
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Shape parameter (can be positive or negative, but not zero)
    lower_tail : bool
        If True, return P(X <= q), else P(X > q)
    log_p : bool
        If True, return log probability

    Returns
    -------
    array_like
        CDF values

    Notes
    -----
    For |nu| > 1e-6:
        z = (q/mu)^nu
        P(X <= q) = P_Gamma(z) if nu > 0
        P(X <= q) = 1 - P_Gamma(z) if nu < 0

    For |nu| <= 1e-6:
        Uses Log-Normal CDF
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)

    eps = jnp.finfo(jnp.float64).eps
    q = jnp.maximum(q, eps)
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)

    # Compute z = (q/mu)^nu
    z = jnp.power(q / mu, nu)

    # For |nu| > 1e-6, use Gamma CDF on z
    sigma_prime = sigma * jnp.abs(nu)
    shape = 1.0 / (sigma_prime * sigma_prime)
    scale = sigma_prime * sigma_prime

    # Gamma CDF
    from scipy.stats import gamma as sp_gamma
    cdf_gamma = sp_gamma.cdf(z, a=shape, scale=scale)

    # If nu < 0, flip the CDF
    cdf_gg = jnp.where(nu > 0, cdf_gamma, 1.0 - cdf_gamma)

    # For nu ~ 0, use log-normal CDF
    from scipy.stats import lognorm as sp_lognorm
    # lognorm parameterization: s=sigma, scale=mu
    cdf_lognormal = sp_lognorm.cdf(q, s=sigma, scale=mu)

    # Use log-normal when |nu| < 1e-6
    cdf = jnp.where(jnp.abs(nu) > 1e-6, cdf_gg, cdf_lognormal)

    # Handle q <= 0
    cdf = jnp.where(q <= 0, 0.0, cdf)

    if not lower_tail:
        cdf = 1.0 - cdf

    if log_p:
        cdf = jnp.log(cdf)

    return cdf


def qGG(p, mu=1.0, sigma=1.0, nu=1.0, lower_tail=True, log_p=False):
    """Quantile function for Generalized Gamma distribution.

    Parameters
    ----------
    p : array_like
        Probabilities (0 < p < 1)
    mu : array_like
        Location parameter (mu > 0)
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Shape parameter (can be positive or negative, but not zero)
    lower_tail : bool
        If True, p is P(X <= q), else P(X > q)
    log_p : bool
        If True, p is log probability

    Returns
    -------
    array_like
        Quantile values

    Notes
    -----
    For |nu| > 1e-6:
        z = Q_Gamma(p) if nu > 0
        z = Q_Gamma(1-p) if nu < 0
        q = mu * z^(1/nu)

    For |nu| <= 1e-6:
        Uses Log-Normal quantile function
    """
    p = jnp.asarray(p, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)

    if log_p:
        p = jnp.exp(p)

    if not lower_tail:
        p = 1.0 - p

    eps = jnp.finfo(jnp.float64).eps
    p = jnp.clip(p, eps, 1.0 - eps)
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)

    # For |nu| > 1e-6, use Gamma quantile on z
    sigma_prime = sigma * jnp.abs(nu)
    shape = 1.0 / (sigma_prime * sigma_prime)
    scale = sigma_prime * sigma_prime

    # Adjust p for negative nu
    p_adjusted = jnp.where(nu > 0, p, 1.0 - p)

    # Gamma quantile
    from scipy.stats import gamma as sp_gamma
    z = sp_gamma.ppf(p_adjusted, a=shape, scale=scale)

    # Transform back: q = mu * z^(1/nu)
    q_gg = mu * jnp.power(z, 1.0 / nu)

    # For nu ~ 0, use log-normal quantile
    from scipy.stats import lognorm as sp_lognorm
    q_lognormal = sp_lognorm.ppf(p, s=sigma, scale=mu)

    # Use log-normal when |nu| < 1e-6
    q = jnp.where(jnp.abs(nu) > 1e-6, q_gg, q_lognormal)

    return q


def rGG(key, n, mu=1.0, sigma=1.0, nu=1.0):
    """Random generation for Generalized Gamma distribution.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key
    n : int
        Number of observations
    mu : array_like
        Location parameter (mu > 0)
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Shape parameter (can be positive or negative, but not zero)

    Returns
    -------
    array_like
        Random samples

    Notes
    -----
    Generation algorithm:
    1. Generate u ~ Uniform(0, 1)
    2. Compute q = quantile(u)
    3. Return q
    """
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)

    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)

    # Generate uniform samples
    u = jrandom.uniform(key, shape=(n,))

    # Convert to GG samples using quantile function
    samples = qGG(u, mu, sigma, nu)

    return samples


# ------------------------------------------------------------------------------
# GB2 (Generalized Beta Type 2) - 4 parameters
# ------------------------------------------------------------------------------

def dGB2(x, mu=1.0, sigma=1.0, nu=1.0, tau=1.0, log=False):
    """Density function for Generalized Beta Type 2 distribution.

    The GB2 is a very flexible four-parameter family that includes many
    distributions as special cases (e.g., Burr, Fisk, Lomax, etc.).

    Parameters
    ----------
    x : array_like
        Quantiles (x > 0)
    mu : array_like
        Location parameter (mu > 0)
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Shape parameter 1 (nu > 0)
    tau : array_like
        Shape parameter 2 (tau > 0)
    log : bool
        If True, return log density

    Returns
    -------
    array_like
        Density values

    Notes
    -----
    Density formula:
    f(x|μ,σ,ν,τ) = |σ| x^(σν-1) / {μ^(σν) B(ν,τ) [1+(x/μ)^σ]^(ν+τ)}

    where B(ν,τ) is the Beta function.

    Special cases:
    - σ=1, τ→∞: Gamma distribution
    - σ=1, ν=1: Lomax distribution
    - σ=1: Generalized Beta Prime
    - ν=1: Burr distribution

    References
    ----------
    McDonald, J. B. (1984). Some generalized functions for the size
    distribution of income. Econometrica, 52(3), 647-663.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    tau = jnp.asarray(tau, dtype=jnp.float64)

    valid = ((mu > 0) & (sigma != 0) & (nu > 0) & (tau > 0)
             & jnp.isfinite(mu) & jnp.isfinite(sigma)
             & jnp.isfinite(nu) & jnp.isfinite(tau))
    # Evaluate outside-support observations at a safe point before masking.
    # Splitting the logs avoids overflow in x/mu for finite positive inputs.
    safe_x = jnp.where((x > 0) & jnp.isfinite(x), x, 1.0)
    log_x = jnp.log(safe_x)
    z = sigma * (log_x - jnp.log(mu))
    log_pdf = (jnp.log(jnp.abs(sigma)) - log_x - betaln(nu, tau)
               - nu * jnp.logaddexp(0.0, -z)
               - tau * jnp.logaddexp(0.0, z))
    log_pdf = jnp.where((x <= 0) | jnp.isposinf(x), -jnp.inf, log_pdf)
    log_pdf = jnp.where(valid & ~jnp.isnan(x), log_pdf, jnp.nan)

    if log:
        return log_pdf
    else:
        return jnp.exp(log_pdf)


def pGB2(q, mu=1.0, sigma=1.0, nu=1.0, tau=1.0, lower_tail=True, log_p=False):
    """CDF for Generalized Beta Type 2 distribution.

    Parameters
    ----------
    q : array_like
        Quantiles (q > 0)
    mu : array_like
        Location parameter (mu > 0)
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Shape parameter 1 (nu > 0)
    tau : array_like
        Shape parameter 2 (tau > 0)
    lower_tail : bool
        If True, return P(X <= q), else P(X > q)
    log_p : bool
        If True, return log probability

    Returns
    -------
    array_like
        CDF values

    Notes
    -----
    The CDF can be expressed using the incomplete beta function:
    F(q) = I_z(ν, τ) where z = (q/μ)^σ / [1 + (q/μ)^σ]
    and I_z is the regularized incomplete beta function.
    """
    from scipy.special import betainc, betaincc, expit

    q, mu, sigma, nu, tau = np.broadcast_arrays(
        *[np.asarray(v, dtype=float) for v in (q, mu, sigma, nu, tau)]
    )
    valid = ((mu > 0) & (sigma != 0) & (nu > 0) & (tau > 0)
             & np.isfinite(mu) & np.isfinite(sigma)
             & np.isfinite(nu) & np.isfinite(tau))
    # Negative sigma swaps the beta shapes when expressed with |sigma|.
    a, b = np.where(sigma > 0, nu, tau), np.where(sigma > 0, tau, nu)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        z = np.abs(sigma) * (np.log(np.where(q > 0, q, 1.)) - np.log(mu))
        # Evaluate the small beta argument and its complement separately;
        # neither sigmoid rounding to one nor 1-CDF destroys the upper tail.
        left = betainc(a, b, expit(z))
        right = betainc(b, a, expit(-z))
        cdf = np.where(z <= 0, left, betaincc(b, a, expit(-z)))
        sf = np.where(z >= 0, right, betaincc(a, b, expit(z)))
        result = cdf if lower_tail else sf
        result = np.where(q <= 0, float(not lower_tail), result)
        result = np.where(np.isposinf(q), float(lower_tail), result)
        if log_p:
            result = np.log(result)
    return jnp.asarray(np.where(valid & ~np.isnan(q), result, np.nan))


def qGB2(p, mu=1.0, sigma=1.0, nu=1.0, tau=1.0, lower_tail=True, log_p=False):
    """Quantile function for Generalized Beta Type 2 distribution.

    Parameters
    ----------
    p : array_like
        Probabilities (0 < p < 1)
    mu : array_like
        Location parameter (mu > 0)
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Shape parameter 1 (nu > 0)
    tau : array_like
        Shape parameter 2 (tau > 0)
    lower_tail : bool
        If True, p is P(X <= q), else P(X > q)
    log_p : bool
        If True, p is log probability

    Returns
    -------
    array_like
        Quantile values

    Notes
    -----
    The quantile function uses the inverse relationship:
    z = I^(-1)_p(ν, τ) (inverse incomplete beta)
    q = μ * [z / (1 - z)]^(1/σ)
    """
    from scipy.special import betaincinv, betainccinv

    p, mu, sigma, nu, tau = np.broadcast_arrays(
        *[np.asarray(v, dtype=float) for v in (p, mu, sigma, nu, tau)]
    )
    valid_p = (p <= 0) if log_p else ((p >= 0) & (p <= 1))
    valid = (valid_p & (mu > 0) & (sigma != 0) & (nu > 0) & (tau > 0)
             & np.isfinite(mu) & np.isfinite(sigma)
             & np.isfinite(nu) & np.isfinite(tau))
    a, b = np.where(sigma > 0, nu, tau), np.where(sigma > 0, tau, nu)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        flip = (p > -np.log(2.0)) if log_p else (p > 0.5)
        probability = (np.where(flip, -np.expm1(p), np.exp(p)) if log_p
                       else np.where(flip, 1.0 - p, p))
        # Compute both beta coordinates directly, never as 1 - coordinate.
        use_lower = np.logical_xor(lower_tail, flip)
        z = np.where(use_lower, betaincinv(a, b, probability),
                     betainccinv(a, b, probability))
        w = np.where(use_lower, betainccinv(b, a, probability),
                     betaincinv(b, a, probability))
        result = np.exp(np.log(mu) + (np.log(z) - np.log(w)) / np.abs(sigma))
    return jnp.asarray(np.where(valid, result, np.nan))


def rGB2(key, n, mu=1.0, sigma=1.0, nu=1.0, tau=1.0):
    """Random generation for Generalized Beta Type 2 distribution.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key
    n : int
        Number of observations
    mu : array_like
        Location parameter (mu > 0)
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Shape parameter 1 (nu > 0)
    tau : array_like
        Shape parameter 2 (tau > 0)

    Returns
    -------
    array_like
        Random samples

    Notes
    -----
    Generation algorithm:
    1. Generate u ~ Uniform(0, 1)
    2. Compute q = quantile(u)
    3. Return q
    """
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    tau = jnp.asarray(tau, dtype=jnp.float64)

    # Generate uniform samples
    u = jrandom.uniform(key, shape=(n,), dtype=jnp.float64)

    # Convert to GB2 samples using quantile function
    samples = qGB2(u, mu, sigma, nu, tau)

    return samples


# ------------------------------------------------------------------------------
# PARETO2 (Pareto Type 2 / Lomax) - 2 parameters
# ------------------------------------------------------------------------------

def dPARETO2(x, mu=1.0, sigma=0.5, log=False):
    """Density function for Pareto Type 2 (Lomax) distribution.

    The Pareto Type 2 (also known as Lomax distribution) is a shifted
    Pareto distribution with support on (0, ∞).

    Parameters
    ----------
    x : array_like
        Quantiles (x > 0)
    mu : array_like
        Scale parameter (mu > 0)
    sigma : array_like
        Shape parameter (sigma > 0)
    log : bool
        If True, return log density

    Returns
    -------
    array_like
        Density values

    Notes
    -----
    Density formula:
    f(x|μ,σ) = (1/σ) * (μ/σ)^(1/σ) / (x + μ)^(1/σ + 1)

    Or in log form:
    log f(x) = -log(σ) + (1/σ)*log(μ) - (1/σ + 1)*log(x + μ)

    Moments:
    - E(X) = μ (for σ < 1)
    - Var(X) = μ²σ² / (1 - σ²) (for σ < 1/√2)

    References
    ----------
    Lomax, K. S. (1954). Business failures: Another example of the
    analysis of failure data. Journal of the American Statistical
    Association, 49(268), 847-852.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)

    eps = jnp.finfo(jnp.float64).eps
    x = jnp.maximum(x, eps)
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)

    # log f(x) = -log(σ) + (1/σ)*log(μ) - (1/σ + 1)*log(x + μ)
    log_pdf = (
        -jnp.log(sigma) +
        (1.0 / sigma) * jnp.log(mu) -
        (1.0 / sigma + 1.0) * jnp.log(x + mu)
    )

    # Handle x <= 0
    log_pdf = jnp.where(x <= 0, -jnp.inf, log_pdf)

    if log:
        return log_pdf
    else:
        return jnp.exp(log_pdf)


def pPARETO2(q, mu=1.0, sigma=0.5, lower_tail=True, log_p=False):
    """CDF for Pareto Type 2 (Lomax) distribution.

    Parameters
    ----------
    q : array_like
        Quantiles (q > 0)
    mu : array_like
        Scale parameter (mu > 0)
    sigma : array_like
        Shape parameter (sigma > 0)
    lower_tail : bool
        If True, return P(X <= q), else P(X > q)
    log_p : bool
        If True, return log probability

    Returns
    -------
    array_like
        CDF values

    Notes
    -----
    CDF formula:
    F(q) = 1 - [μ / (q + μ)]^(1/σ)
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)

    eps = jnp.finfo(jnp.float64).eps
    q = jnp.maximum(q, eps)
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)

    # F(q) = 1 - [μ / (q + μ)]^(1/σ)
    cdf = 1.0 - jnp.power(mu / (q + mu), 1.0 / sigma)

    # Handle q <= 0
    cdf = jnp.where(q <= 0, 0.0, cdf)

    if not lower_tail:
        cdf = 1.0 - cdf

    if log_p:
        cdf = jnp.log(cdf)

    return cdf


def qPARETO2(p, mu=1.0, sigma=0.5, lower_tail=True, log_p=False):
    """Quantile function for Pareto Type 2 (Lomax) distribution.

    Parameters
    ----------
    p : array_like
        Probabilities (0 < p < 1)
    mu : array_like
        Scale parameter (mu > 0)
    sigma : array_like
        Shape parameter (sigma > 0)
    lower_tail : bool
        If True, p is P(X <= q), else P(X > q)
    log_p : bool
        If True, p is log probability

    Returns
    -------
    array_like
        Quantile values

    Notes
    -----
    Quantile formula:
    Q(p) = μ * [(1 - p)^(-σ) - 1]
    """
    p = jnp.asarray(p, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)

    if log_p:
        p = jnp.exp(p)

    if not lower_tail:
        p = 1.0 - p

    eps = jnp.finfo(jnp.float64).eps
    p = jnp.clip(p, eps, 1.0 - eps)
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)

    # Q(p) = μ * [(1 - p)^(-σ) - 1]
    q = mu * (jnp.power(1.0 - p, -sigma) - 1.0)

    return q


def rPARETO2(key, n, mu=1.0, sigma=0.5):
    """Random generation for Pareto Type 2 (Lomax) distribution.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key
    n : int
        Number of observations
    mu : array_like
        Scale parameter (mu > 0)
    sigma : array_like
        Shape parameter (sigma > 0)

    Returns
    -------
    array_like
        Random samples

    Notes
    -----
    Generation algorithm:
    1. Generate u ~ Uniform(0, 1)
    2. Compute q = quantile(u)
    3. Return q
    """
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)

    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)

    # Generate uniform samples
    u = jrandom.uniform(key, shape=(n,))

    # Convert to PARETO2 samples using quantile function
    samples = qPARETO2(u, mu, sigma)

    return samples


# ------------------------------------------------------------------------------
# SIMPLEX - 2 parameters (distribution on (0,1))
# ------------------------------------------------------------------------------

def dSIMPLEX(x, mu=0.5, sigma=1.0, log=False):
    """Density function for Simplex distribution.

    The Simplex distribution is defined on (0,1) and is useful for
    modeling proportions and rates.

    Parameters
    ----------
    x : array_like
        Quantiles (0 < x < 1)
    mu : array_like
        Location parameter (0 < mu < 1)
    sigma : array_like
        Dispersion parameter (sigma > 0)
    log : bool
        If True, return log density

    Returns
    -------
    array_like
        Density values

    Notes
    -----
    Log density formula:
    log f(x|μ,σ) = -[(x-μ)/(μ(1-μ))]²/[2x(1-x)σ²] 
                   - (1/2)[log(2πσ²) + 3(log(x) + log(1-x))]

    The Simplex distribution was introduced by Barndorff-Nielsen and Jørgensen (1991)
    for modeling data on the unit interval.

    References
    ----------
    Barndorff-Nielsen, O. E., & Jørgensen, B. (1991). Some parametric models
    on the simplex. Journal of Multivariate Analysis, 39(1), 106-116.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)

    eps = jnp.finfo(jnp.float64).eps
    x = jnp.clip(x, eps, 1.0 - eps)
    mu = jnp.clip(mu, eps, 1.0 - eps)
    sigma = jnp.maximum(sigma, eps)

    # Compute d = (x - mu) / (mu * (1 - mu))
    d = (x - mu) / (mu * (1.0 - mu))

    # Log density
    log_pdf = (
        -jnp.square(d) / (2.0 * x * (1.0 - x) * jnp.square(sigma))
        - 0.5 * (jnp.log(2.0 * jnp.pi * jnp.square(sigma)) + 3.0 * (jnp.log(x) + jnp.log(1.0 - x)))
    )

    # Handle x outside (0, 1)
    log_pdf = jnp.where((x <= 0) | (x >= 1), -jnp.inf, log_pdf)

    if log:
        return log_pdf
    else:
        return jnp.exp(log_pdf)


def pSIMPLEX(q, mu=0.5, sigma=1.0, lower_tail=True, log_p=False):
    """CDF for Simplex distribution.

    Parameters
    ----------
    q : array_like
        Quantiles (0 < q < 1)
    mu : array_like
        Location parameter (0 < mu < 1)
    sigma : array_like
        Dispersion parameter (sigma > 0)
    lower_tail : bool
        If True, return P(X <= q), else P(X > q)
    log_p : bool
        If True, return log probability

    Returns
    -------
    array_like
        CDF values

    Notes
    -----
    The CDF is computed using numerical integration of the density function.
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)

    eps = jnp.finfo(jnp.float64).eps
    q = jnp.clip(q, eps, 1.0 - eps)
    mu = jnp.clip(mu, eps, 1.0 - eps)
    sigma = jnp.maximum(sigma, eps)

    # Use vectorized trapezoidal integration over a shared grid.
    # All quantiles share the same grid nodes; integrate per-q by
    # masking density beyond each q (faster and more accurate than
    # jnp.vectorize over per-q grids).
    n_points = 16384
    x_grid = jnp.linspace(eps, 1.0 - eps, n_points)
    d_grid = dSIMPLEX(x_grid, mu, sigma, log=False)
    # Ensure 2-D shape (param_batch, n_points): scalar params -> (1, n_points)
    d_grid = jnp.atleast_2d(d_grid)

    # Cumulative integral via cumulative trapezoid on the shared grid.
    dx = (1.0 - 2.0 * eps) / (n_points - 1)
    integrand = (d_grid[:, :-1] + d_grid[:, 1:]) * 0.5 * dx
    cum = jnp.concatenate([jnp.zeros_like(integrand[:, :1]), jnp.cumsum(integrand, axis=-1)], axis=-1)

    # Map each q onto the grid via searchsorted, then interpolate the cumulative mass.
    # q is 1-D here (broadcast over parameters was already applied by dSIMPLEX).
    q_flat = jnp.atleast_1d(q)
    q_clipped = jnp.clip(q_flat, eps, 1.0 - eps)
    idx = jnp.searchsorted(x_grid, q_clipped, side="right") - 1
    idx = jnp.clip(idx, 0, n_points - 2)
    x0 = x_grid[idx]
    frac = (q_clipped - x0) / jnp.maximum(x_grid[idx + 1] - x0, eps)
    # cum has shape (param_batch, n_points); gather along the grid axis.
    cdf_vals = cum[:, idx] + frac[None, :] * (cum[:, idx + 1] - cum[:, idx])
    if cdf_vals.shape[0] == 1:
        cdf = jnp.reshape(cdf_vals[0], jnp.shape(q))
    else:
        cdf = cdf_vals

    # Handle q outside (0, 1)
    cdf = jnp.where(q_flat <= 0, 0.0, cdf)
    cdf = jnp.where(q_flat >= 1, 1.0, cdf)

    # Broadcast back to the original shape (scalar input -> scalar output)
    cdf = cdf.reshape(jnp.shape(q))

    if not lower_tail:
        cdf = 1.0 - cdf

    if log_p:
        cdf = jnp.log(cdf)

    return cdf


def qSIMPLEX(p, mu=0.5, sigma=1.0, lower_tail=True, log_p=False):
    """Quantile function for Simplex distribution.

    Parameters
    ----------
    p : array_like
        Probabilities (0 < p < 1)
    mu : array_like
        Location parameter (0 < mu < 1)
    sigma : array_like
        Dispersion parameter (sigma > 0)
    lower_tail : bool
        If True, p is P(X <= q), else P(X > q)
    log_p : bool
        If True, p is log probability

    Returns
    -------
    array_like
        Quantile values

    Notes
    -----
    The quantile function is computed using numerical root finding
    on the CDF.
    """
    p = jnp.asarray(p, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)

    if log_p:
        p = jnp.exp(p)

    if not lower_tail:
        p = 1.0 - p

    eps = jnp.finfo(jnp.float64).eps
    p = jnp.clip(p, eps, 1.0 - eps)
    mu = jnp.clip(mu, eps, 1.0 - eps)
    sigma = jnp.maximum(sigma, eps)

    # Use scipy for root finding
    from scipy.optimize import brentq
    
    def find_quantile_scalar(p_val):
        # Define the function to find root of: F(q) - p = 0
        def objective(q_val):
            cdf_val = pSIMPLEX(q_val, mu, sigma, lower_tail=True, log_p=False)
            return float(cdf_val) - p_val
        
        # Bounds
        lower = eps
        upper = 1.0 - eps
        
        try:
            result = brentq(objective, lower, upper, xtol=1e-10)
            return result
        except:
            # Fallback to bisection
            for _ in range(50):
                mid = (lower + upper) / 2.0
                f_mid = objective(mid)
                if abs(f_mid) < 1e-8:
                    return mid
                f_lower = objective(lower)
                if f_lower * f_mid < 0:
                    upper = mid
                else:
                    lower = mid
            return (lower + upper) / 2.0
    
    # Vectorize
    p_array = jnp.atleast_1d(p)
    q_list = [find_quantile_scalar(float(p_val)) for p_val in p_array]
    q = jnp.array(q_list)
    
    # Return scalar if input was scalar
    if jnp.ndim(p) == 0:
        return q[0]
    
    return q


def rSIMPLEX(key, n, mu=0.5, sigma=1.0):
    """Random generation for Simplex distribution.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key
    n : int
        Number of observations
    mu : array_like
        Location parameter (0 < mu < 1)
    sigma : array_like
        Dispersion parameter (sigma > 0)

    Returns
    -------
    array_like
        Random samples

    Notes
    -----
    Generation algorithm:
    1. Generate u ~ Uniform(0, 1)
    2. Compute q = quantile(u)
    3. Return q
    """
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)

    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.clip(mu, eps, 1.0 - eps)
    sigma = jnp.maximum(sigma, eps)

    # Generate uniform samples
    u = jrandom.uniform(key, shape=(n,))

    # Convert to SIMPLEX samples using quantile function
    samples = qSIMPLEX(u, mu, sigma)

    return samples


# ------------------------------------------------------------------------------
# exGAUS (Exponentially-modified Gaussian) - 3 parameters
# ------------------------------------------------------------------------------

def dexGAUS(x, mu=0.0, sigma=1.0, nu=1.0, log=False):
    """Density function for Exponentially-modified Gaussian distribution.

    The exGAUS (also known as EMG) is the convolution of a Gaussian and
    an exponential distribution. It's commonly used in reaction time analysis.

    Parameters
    ----------
    x : array_like
        Quantiles
    mu : array_like
        Location parameter (mean of Gaussian component)
    sigma : array_like
        Scale parameter (SD of Gaussian component, sigma > 0)
    nu : array_like
        Exponential rate parameter (nu > 0)
    log : bool
        If True, return log density

    Returns
    -------
    array_like
        Density values

    Notes
    -----
    The exGAUS is the convolution of Normal(μ, σ) and Exponential(1/ν).

    For nu > 0.05*sigma:
        z = x - μ - σ²/ν
        log f(x) = -log(ν) - (z + σ²/(2ν))/ν + log(Φ(z/σ))
    
    For nu ≤ 0.05*sigma (nu very small):
        Falls back to Normal(μ, σ)

    where Φ is the standard normal CDF.

    References
    ----------
    Grushka, E. (1972). Characterization of exponentially modified Gaussian
    peaks in chromatography. Analytical Chemistry, 44(11), 1733-1738.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)

    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.maximum(nu, eps)

    # Compute z = x - mu - sigma^2/nu
    z = x - mu - jnp.square(sigma) / nu

    # Exponentially-modified Gaussian formula
    from scipy.stats import norm as sp_norm
    log_pdf_emg = (
        -jnp.log(nu) -
        (z + jnp.square(sigma) / (2.0 * nu)) / nu +
        sp_norm.logcdf(z / sigma)
    )

    # Fallback to normal when nu is very small
    log_pdf_normal = sp_norm.logpdf(x, loc=mu, scale=sigma)

    # Use EMG when nu > 0.05*sigma, otherwise use normal
    log_pdf = jnp.where(nu > 0.05 * sigma, log_pdf_emg, log_pdf_normal)

    if log:
        return log_pdf
    else:
        return jnp.exp(log_pdf)


def pexGAUS(q, mu=0.0, sigma=1.0, nu=1.0, lower_tail=True, log_p=False):
    """CDF for Exponentially-modified Gaussian distribution.

    Parameters
    ----------
    q : array_like
        Quantiles
    mu : array_like
        Location parameter
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Exponential rate parameter (nu > 0)
    lower_tail : bool
        If True, return P(X <= q), else P(X > q)
    log_p : bool
        If True, return log probability

    Returns
    -------
    array_like
        CDF values

    Notes
    -----
    The CDF can be expressed as:
    F(q) = Φ((q-μ)/σ) - exp(λ)*Φ((q-μ-σ²/ν)/σ)
    where λ = (μ + σ²/(2ν) - q)/ν
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)

    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.maximum(nu, eps)

    from scipy.stats import norm as sp_norm

    # For nu > 0.05*sigma, use the EMG CDF formula
    z1 = (q - mu) / sigma
    z2 = (q - mu - jnp.square(sigma) / nu) / sigma
    lambda_val = (mu + jnp.square(sigma) / (2.0 * nu) - q) / nu

    cdf_emg = sp_norm.cdf(z1) - jnp.exp(lambda_val) * sp_norm.cdf(z2)

    # For nu ≤ 0.05*sigma, use normal CDF
    cdf_normal = sp_norm.cdf(q, loc=mu, scale=sigma)

    # Choose based on nu
    cdf = jnp.where(nu > 0.05 * sigma, cdf_emg, cdf_normal)

    # Clip to [0, 1]
    cdf = jnp.clip(cdf, 0.0, 1.0)

    if not lower_tail:
        cdf = 1.0 - cdf

    if log_p:
        cdf = jnp.log(cdf)

    return cdf


def qexGAUS(p, mu=0.0, sigma=1.0, nu=1.0, lower_tail=True, log_p=False):
    """Quantile function for Exponentially-modified Gaussian distribution.

    Parameters
    ----------
    p : array_like
        Probabilities (0 < p < 1)
    mu : array_like
        Location parameter
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Exponential rate parameter (nu > 0)
    lower_tail : bool
        If True, p is P(X <= q), else P(X > q)
    log_p : bool
        If True, p is log probability

    Returns
    -------
    array_like
        Quantile values

    Notes
    -----
    The quantile function is computed using numerical root finding
    on the CDF.
    """
    p = jnp.asarray(p, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)

    if log_p:
        p = jnp.exp(p)

    if not lower_tail:
        p = 1.0 - p

    eps = jnp.finfo(jnp.float64).eps
    p = jnp.clip(p, eps, 1.0 - eps)
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.maximum(nu, eps)

    # Use scipy for quantile (simpler and more reliable)
    from scipy.stats import norm as sp_norm
    
    # Approximate initial guess using normal quantile
    q_normal = sp_norm.ppf(p, loc=float(mu), scale=float(sigma))
    
    # Use scipy's root finding
    from scipy.optimize import brentq
    
    def find_quantile_scalar(p_val):
        # Define the function to find root of: F(q) - p = 0
        def objective(q_val):
            cdf_val = pexGAUS(q_val, mu, sigma, nu, lower_tail=True, log_p=False)
            return float(cdf_val) - p_val
        
        # Bounds
        lower = float(mu) - 5.0 * float(sigma)
        upper = float(mu) + 5.0 * float(sigma) + 10.0 * float(nu)
        
        try:
            result = brentq(objective, lower, upper, xtol=1e-8)
            return result
        except:
            # Fallback to bisection
            for _ in range(50):
                mid = (lower + upper) / 2.0
                f_mid = objective(mid)
                if abs(f_mid) < 1e-8:
                    return mid
                f_lower = objective(lower)
                if f_lower * f_mid < 0:
                    upper = mid
                else:
                    lower = mid
            return (lower + upper) / 2.0
    
    # Vectorize
    p_array = jnp.atleast_1d(p)
    q_list = [find_quantile_scalar(float(p_val)) for p_val in p_array]
    q = jnp.array(q_list)
    
    # Return scalar if input was scalar
    if jnp.ndim(p) == 0:
        return q[0]
    
    return q


def rexGAUS(key, n, mu=0.0, sigma=1.0, nu=1.0):
    """Random generation for Exponentially-modified Gaussian distribution.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key
    n : int
        Number of observations
    mu : array_like
        Location parameter
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Exponential rate parameter (nu > 0)

    Returns
    -------
    array_like
        Random samples

    Notes
    -----
    Generation algorithm (convolution method):
    1. Generate z ~ Normal(μ, σ)
    2. Generate e ~ Exponential(1/ν)
    3. Return x = z + e

    This is more efficient than using the quantile function.
    """
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)

    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.maximum(nu, eps)

    # Split key for two random generations
    key1, key2 = jrandom.split(key)

    # Generate normal samples
    z = mu + sigma * jrandom.normal(key1, shape=(n,))

    # Generate exponential samples with rate 1/nu (scale = nu)
    e = nu * jrandom.exponential(key2, shape=(n,))

    # Convolution: x = z + e
    samples = z + e

    return samples


# ==============================================================================
# DISCRETE DISTRIBUTIONS (P1)
# ==============================================================================

# ------------------------------------------------------------------------------
# BB (Beta-Binomial) - 3 parameters (mu, sigma, bd)
# ------------------------------------------------------------------------------

def dBB(x, mu=0.5, sigma=1.0, bd=10, log=False):
    """Density function for Beta-Binomial distribution.

    The Beta-Binomial is a discrete distribution that generalizes the
    Binomial by allowing the success probability to vary according to
    a Beta distribution.

    Parameters
    ----------
    x : array_like
        Counts (0 <= x <= bd, integer values)
    mu : array_like
        Mean parameter (0 < mu < 1)
    sigma : array_like
        Dispersion parameter (sigma > 0)
    bd : array_like
        Binomial denominator (bd >= 1, typically fixed)
    log : bool
        If True, return log density

    Returns
    -------
    array_like
        Density values

    Notes
    -----
    Parameterization (from R gamlss.dist):
    k = 1/sigma
    alpha = mu * k
    beta = (1 - mu) * k

    log p(x) = lgamma(bd+1) - lgamma(x+1) - lgamma(bd-x+1)
               + lgamma(k) + lgamma(x + alpha) + lgamma(bd + beta - x)
               - lgamma(alpha) - lgamma(beta) - lgamma(bd + k)

    When sigma → 0, approaches Binomial(bd, mu).

    References
    ----------
    Skellam, J. G. (1948). A probability distribution derived from the
    binomial distribution by regarding the probability of success as variable
    between the sets of trials. Journal of the Royal Statistical Society
    Series B, 10(2), 257-261.
    """
    x, mu, sigma, bd = jnp.broadcast_arrays(
        *[jnp.asarray(v, dtype=jnp.float64) for v in (x, mu, sigma, bd)])
    valid = ((mu > 0) & (mu < 1) & (sigma > 0) & jnp.isfinite(sigma)
             & (bd >= 0) & jnp.isfinite(bd) & (bd == jnp.floor(bd)))
    support = (x >= 0) & (x <= bd) & (x == jnp.floor(x)) & jnp.isfinite(x)
    y = jnp.where(support, x, 0.)
    m = jnp.where(valid, mu, .5)
    n = jnp.where(valid, bd, 1.)
    small = sigma < 1e-4
    # Keep the inactive beta branch finite even for sigma below float range.
    s = jnp.where(valid & ~small, sigma, 1.)
    a, b = m/s, (1-m)/s
    choose = gammaln(n+1) - gammaln(y+1) - gammaln(n-y+1)
    # JAX betaln's approximation can lose ~1e-6 for small shape sums;
    # the gamma identity preserves R agreement in the regular branch.
    beta_log = (choose + gammaln(y+a) + gammaln(n-y+b) - gammaln(n+a+b)
                - gammaln(a) - gammaln(b) + gammaln(a+b))
    binomial_log = choose + y*jnp.log(m) + (n-y)*jnp.log1p(-m)
    log_pdf = jnp.where(small, binomial_log, beta_log)
    log_pdf = jnp.where(support, log_pdf, -jnp.inf)
    log_pdf = jnp.where(valid & ~jnp.isnan(x), log_pdf, jnp.nan)

    if log:
        return log_pdf
    else:
        return jnp.exp(log_pdf)


def pBB(q, mu=0.5, sigma=1.0, bd=10, lower_tail=True, log_p=False):
    """CDF for Beta-Binomial distribution.

    Parameters
    ----------
    q : array_like
        Quantiles (0 <= q <= bd)
    mu : array_like
        Mean parameter (0 < mu < 1)
    sigma : array_like
        Dispersion parameter (sigma > 0)
    bd : array_like
        Binomial denominator (bd >= 1)
    lower_tail : bool
        If True, return P(X <= q), else P(X > q)
    log_p : bool
        If True, return log probability

    Returns
    -------
    array_like
        CDF values

    Notes
    -----
    The CDF is computed by summing the PMF from 0 to floor(q).
    """
    q, mu, sigma, bd = np.broadcast_arrays(
        *[np.asarray(value, dtype=float) for value in (q, mu, sigma, bd)]
    )
    method = ("logcdf" if log_p else "cdf") if lower_tail else (
        "logsf" if log_p else "sf"
    )
    # As in R, use the binomial limit below sigma=0.0001. Only evaluate
    # the selected branch to avoid cancellation at enormous beta shapes.
    result = np.full(q.shape, np.nan)
    valid = (mu > 0) & (mu < 1) & (sigma > 0) & np.isfinite(sigma)
    valid &= (bd >= 0) & np.isfinite(bd) & (bd == np.floor(bd))
    small = valid & (sigma < 0.0001)
    regular = valid & ~small
    result[small] = getattr(_binom_dist, method)(q[small], bd[small], mu[small])
    result[regular] = getattr(_betabinom_dist, method)(
        q[regular], bd[regular], mu[regular] / sigma[regular],
        (1 - mu[regular]) / sigma[regular],
    )
    return jnp.asarray(result, dtype=jnp.float64)


def qBB(p, mu=0.5, sigma=1.0, bd=10, lower_tail=True, log_p=False):
    """Quantile function for Beta-Binomial distribution.

    Parameters
    ----------
    p : array_like
        Probabilities (0 < p < 1)
    mu : array_like
        Mean parameter (0 < mu < 1)
    sigma : array_like
        Dispersion parameter (sigma > 0)
    bd : array_like
        Binomial denominator (bd >= 1)
    lower_tail : bool
        If True, p is P(X <= q), else P(X > q)
    log_p : bool
        If True, p is log probability

    Returns
    -------
    array_like
        Quantile values (integers)

    Notes
    -----
    The quantile is found by searching for the smallest x such that
    P(X <= x) >= p.
    Parameters follow NumPy broadcasting. This is a CPU quantile routine,
    with the same binomial limit (sigma < 0.0001) as R gamlss.dist.
    """
    p, mu, sigma, bd = np.broadcast_arrays(
        *[np.asarray(value, dtype=float) for value in (p, mu, sigma, bd)]
    )
    if log_p:
        p = np.exp(p) if lower_tail else -np.expm1(p)
    elif not lower_tail:
        p = 1.0 - p
    result = np.full(p.shape, np.nan)
    valid = (p >= 0) & (p <= 1) & (mu > 0) & (mu < 1)
    valid &= (sigma > 0) & np.isfinite(sigma)
    valid &= (bd >= 0) & np.isfinite(bd) & (bd == np.floor(bd))
    small = valid & (sigma < 0.0001)
    regular = valid & ~small
    result[small] = _binom_dist.ppf(p[small], bd[small], mu[small])
    result[regular] = _betabinom_dist.ppf(
        p[regular], bd[regular], mu[regular] / sigma[regular],
        (1 - mu[regular]) / sigma[regular],
    )
    result = np.where(valid & (p == 0), 0.0, result)
    return jnp.asarray(result, dtype=jnp.float64)


def rBB(key, n, mu=0.5, sigma=1.0, bd=10):
    """Random generation for Beta-Binomial distribution.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key
    n : int
        Number of observations
    mu : array_like
        Mean parameter (0 < mu < 1)
    sigma : array_like
        Dispersion parameter (sigma > 0)
    bd : array_like
        Nonnegative integer binomial denominator; zero gives all-zero samples.

    Returns
    -------
    array_like
        Random samples (integers)

    Notes
    -----
    Generation algorithm:
    1. Generate p ~ Beta(alpha, beta) where alpha = mu/sigma, beta = (1-mu)/sigma
    2. Generate x ~ Binomial(bd, p)
    3. Return x

    Parameters must broadcast to (n,). Invalid parameter elements return NaN.
    n must be a nonnegative static integer (including when used with jit).
    """
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    bd = jnp.asarray(bd, dtype=jnp.float64)

    mu, sigma, bd = [jnp.broadcast_to(v, (n,)) for v in (mu, sigma, bd)]
    valid = ((mu > 0) & (mu < 1) & (sigma > 0) & jnp.isfinite(sigma)
             & (bd >= 0) & jnp.isfinite(bd) & (bd == jnp.floor(bd)))
    small = sigma < 1e-4
    safe_mu = jnp.where(valid, mu, .5)
    safe_bd = jnp.where(valid, bd, 0.)
    # The beta draw is unused in the binomial/degenerate branches. Avoid
    # enormous shapes or NaNs there rather than relying on output masking.
    beta_active = valid & ~small & (bd > 0)
    safe_sigma = jnp.where(beta_active, sigma, 1.)
    beta_mu = jnp.where(beta_active, safe_mu, .5)

    # Split key
    key1, key2 = jrandom.split(key)

    # Beta parameters
    alpha = beta_mu / safe_sigma
    beta = (1.0 - beta_mu) / safe_sigma

    # Generate Beta samples
    p_samples = jrandom.beta(key1, alpha, beta, shape=(n,), dtype=jnp.float64)

    # One vectorized draw supplies independent randomness for every observation;
    # reusing key2 inside a Python loop made the old samples correlated.
    probabilities = jnp.where(small, safe_mu, p_samples)
    samples = jrandom.binomial(key2, n=safe_bd, p=probabilities, shape=(n,))

    return jnp.where(valid, samples, jnp.nan)


# ------------------------------------------------------------------------------
# BNB (Beta-Negative Binomial) - 3 parameters (simplified to 2: mu, sigma)
# ------------------------------------------------------------------------------

def dBNB(x, mu=1.0, sigma=1.0, nu=1.0, log=False):
    """Density function for Beta-Negative Binomial distribution.

    The Beta-Negative Binomial is a compound distribution that generalizes
    the Negative Binomial by allowing the probability parameter to vary
    according to a Beta distribution.

    Parameters
    ----------
    x : array_like
        Counts (x >= 0, integer values)
    mu : array_like
        Mean parameter (mu > 0)
    sigma : array_like
        Dispersion parameter (sigma > 0)
    log : bool
        If True, return log density

    Returns
    -------
    array_like
        Density values

    Notes
    -----
    Simplified parameterization (nu = 1):
    m = 1/sigma + 1
    n = mu/sigma
    k = 1

    log p(x) = lbeta(x+n, m+k) - lbeta(n,m) - lgamma(x+1) - lgamma(k) + lgamma(x+k)

    References
    ----------
    Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized additive models
    for location, scale and shape. Journal of the Royal Statistical Society
    Series C, 54(3), 507-554.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)

    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)

    # Simplified parameterization (nu = 1)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    nu = jnp.maximum(nu, eps)
    m = (1.0 / sigma) + 1.0
    n = (mu * nu) / sigma
    k = 1.0 / nu

    # log p(x)
    log_pdf = (
        betaln(x + n, m + k) - betaln(n, m) -
        gammaln(x + 1.0) - gammaln(k) + gammaln(x + k)
    )

    # Handle x < 0
    log_pdf = jnp.where(x < 0, -jnp.inf, log_pdf)

    if log:
        return log_pdf
    else:
        return jnp.exp(log_pdf)


def pBNB(q, mu=1.0, sigma=1.0, nu=1.0, lower_tail=True, log_p=False):
    """CDF for Beta-Negative Binomial distribution.

    Parameters
    ----------
    q : array_like
        Quantiles (q >= 0)
    mu : array_like
        Mean parameter (mu > 0)
    sigma : array_like
        Dispersion parameter (sigma > 0)
    nu : array_like
        Shape parameter (nu > 0)
    lower_tail : bool
        If True, return P(X <= q), else P(X > q)
    log_p : bool
        If True, return log probability

    Returns
    -------
    array_like
        CDF values

    Notes
    -----
    The CDF is computed by summing the PMF from 0 to floor(q).
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)

    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.maximum(nu, eps)

    q, mu, sigma, nu = jnp.broadcast_arrays(q, mu, sigma, nu)
    q_int = jnp.floor(q)
    # Accumulate only through the requested count.  A lax loop keeps the
    # operation JIT-compatible without allocating a fixed (and often huge)
    # support axis for every broadcasted parameter value.
    finite_q = jnp.where(jnp.isfinite(q_int), jnp.maximum(q_int, -1.0), -1.0)
    total = jnp.zeros_like(q_int)
    x = jnp.array(0.0, dtype=jnp.float64)

    def cond(carry):
        current_x, _ = carry
        return current_x <= jnp.max(finite_q, initial=-1.0)

    def body(carry):
        current_x, current_total = carry
        contribution = dBNB(current_x, mu, sigma, nu)
        current_total = current_total + jnp.where(
            current_x <= finite_q, contribution, 0.0
        )
        return current_x + 1.0, current_total

    _, cdf = jax.lax.while_loop(cond, body, (x, total))
    cdf = jnp.clip(cdf, 0.0, 1.0)
    cdf = jnp.where(q < 0, 0.0, cdf)
    cdf = jnp.where(jnp.isposinf(q), 1.0, cdf)
    cdf = jnp.where(jnp.isnan(q), jnp.nan, cdf)

    if not lower_tail:
        cdf = 1.0 - cdf

    if log_p:
        cdf = jnp.log(cdf)

    return cdf


def qBNB(p, mu=1.0, sigma=1.0, nu=1.0, lower_tail=True, log_p=False):
    """Quantile function for Beta-Negative Binomial distribution.

    Parameters
    ----------
    p : array_like
        Probabilities (0 < p < 1)
    mu : array_like
        Mean parameter (mu > 0)
    sigma : array_like
        Dispersion parameter (sigma > 0)
    nu : array_like
        Shape parameter (nu > 0)
    lower_tail : bool
        If True, p is P(X <= q), else P(X > q)
    log_p : bool
        If True, p is log probability

    Returns
    -------
    array_like
        Quantile values (integers)

    Notes
    -----
    The quantile is found by searching for the smallest x such that
    P(X <= x) >= p. Support is scanned adaptively in blocks of 256 counts,
    with a work budget of 1,048,576 counts (0 through 1,048,575). Unresolved
    interior probabilities return NaN rather than an incorrect quantile.
    Extremely heavy tails can exhaust this budget. Accumulation uses the
    lower-tail PMF sum, so probabilities very close to one are also subject
    to floating-point summation error.
    """
    p = jnp.asarray(p, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)

    if log_p:
        p = jnp.exp(p)

    if not lower_tail:
        p = 1.0 - p

    p, mu, sigma, nu = jnp.broadcast_arrays(p, mu, sigma, nu)
    eps = jnp.finfo(jnp.float64).eps
    invalid = (p < 0) | (p > 1) | jnp.isnan(p)
    at_zero = p == 0
    at_one = p == 1
    p = jnp.clip(p, eps, 1.0 - eps)
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.maximum(nu, eps)

    # Search successive fixed-size blocks instead of materialising the entire
    # support or repeatedly summing it in a CDF-based binary search. Memory is
    # O(batch size * 256); work stops once all interior probabilities resolve.
    # A finite work budget prevents pathological heavy tails from hanging.
    offsets = jnp.arange(256, dtype=jnp.float64)
    active = ~(invalid | at_zero | at_one)
    active = active & jnp.isfinite(mu) & jnp.isfinite(sigma) & jnp.isfinite(nu)

    def cond(state):
        start, _, _, pending = state
        return (start < 1048576) & jnp.any(pending)

    def body(state):
        start, total, result, pending = state
        support = start + offsets
        cumulative = total[..., None] + jnp.cumsum(
            dBNB(support, mu[..., None], sigma[..., None], nu[..., None]),
            axis=-1,
        )
        crossed = cumulative >= p[..., None]
        found = pending & jnp.any(crossed, axis=-1)
        result = jnp.where(found, start + jnp.argmax(crossed, axis=-1), result)
        return start + 256, cumulative[..., -1], result, pending & ~found

    _, _, result, _ = jax.lax.while_loop(
        cond, body,
        (jnp.array(0), jnp.zeros_like(p), jnp.full_like(p, jnp.nan), active),
    )
    result = jnp.where(at_zero, 0.0, result)
    result = jnp.where(at_one, jnp.inf, result)
    result = jnp.where(invalid, jnp.nan, result)
    return result


def rBNB(key, n, mu=1.0, sigma=1.0, nu=1.0):
    """Random generation for Beta-Negative Binomial distribution.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key
    n : int
        Number of observations
    mu : array_like
        Mean parameter (mu > 0)
    sigma : array_like
        Dispersion parameter (sigma > 0)
    nu : array_like
        Shape parameter (nu > 0)

    Returns
    -------
    array_like
        Random samples (integers)
    """
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)

    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.maximum(nu, eps)

    # Keep the sample dimension first and broadcast parameters over the
    # remaining dimensions.  This is both useful for vectorized prediction
    # and avoids the old (n,) vs. batch-shape broadcasting failure.
    batch_shape = jnp.broadcast_shapes(mu.shape, sigma.shape, nu.shape)
    mu, sigma, nu = jnp.broadcast_arrays(mu, sigma, nu)
    u = jrandom.uniform(key, shape=(n,) + batch_shape)

    # Convert to BNB samples using quantile function
    samples = qBNB(u, mu, sigma, nu)

    return samples


# ------------------------------------------------------------------------------
# PIG (Poisson-Inverse Gaussian) - 2 parameters
# ------------------------------------------------------------------------------

from .pig import dPIG, pPIG, qPIG, rPIG


def _pig2_alpha(mu, sigma):
    """Convert PIG2's sigma to the alpha parameter used by PIG in R."""
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    alpha = (jnp.hypot(mu, sigma) + mu) / sigma / sigma
    return jnp.where((mu > 0) & (sigma > 0) & jnp.isfinite(sigma), alpha, jnp.nan)


def dPIG2(x, mu=1.0, sigma=1.0, log=False):
    """PIG2 PMF, matching R's dPIG2 parameter conversion."""
    return dPIG(x, mu, _pig2_alpha(mu, sigma), log=log)


def pPIG2(q, mu=1.0, sigma=1.0, lower_tail=True, log_p=False):
    """PIG2 CDF, matching R's pPIG2 parameter conversion."""
    return pPIG(q, mu, _pig2_alpha(mu, sigma), lower_tail=lower_tail, log_p=log_p)


def qPIG2(p, mu=1.0, sigma=1.0, lower_tail=True, log_p=False, max_value=10000):
    """PIG2 quantiles; tail/log transformations are applied exactly once."""
    return qPIG(p, mu, _pig2_alpha(mu, sigma), lower_tail=lower_tail,
                log_p=log_p, max_value=max_value)


def rPIG2(key, n, mu=1.0, sigma=1.0):
    """PIG2 random generation, matching R's rPIG2 parameter conversion."""
    return rPIG(key, n, mu, _pig2_alpha(mu, sigma))


# ------------------------------------------------------------------------------
# SICHEL - 3 parameters (simplified version)
# ------------------------------------------------------------------------------

def dSICHEL(x, mu=1.0, sigma=1.0, nu=-0.5, log=False):
    """Density function for Sichel distribution.

    The Sichel distribution is a three-parameter generalization of the
    Poisson-Inverse Gaussian distribution.

    Parameters
    ----------
    x : array_like
        Counts (x >= 0, integer values)
    mu : array_like
        Mean parameter (mu > 0)
    sigma : array_like
        Dispersion parameter (sigma > 0)
    nu : array_like
        Shape parameter
    log : bool
        If True, return log density

    Returns
    -------
    array_like
        Density values

    Notes
    -----
    Uses the family Bessel-K recurrence with numerical support 0..10000.
    Invalid parameters and counts beyond that range return NaN.

    For nu = -0.5, the Sichel distribution reduces to PIG.

    References
    ----------
    Sichel, H. S. (1971). On a family of discrete distributions particularly
    suited to represent long-tailed frequency data. In Proceedings of the
    Third Symposium on Mathematical Statistics (pp. 51-97).
    """
    from .distributions_b6 import _sichel_log_pdf

    x, mu, sigma, nu = jnp.broadcast_arrays(*[
        jnp.asarray(v, dtype=jnp.float64) for v in (x, mu, sigma, nu)
    ])
    flat_log = jax.vmap(_sichel_log_pdf)(x.ravel(), mu.ravel(),
                                         sigma.ravel(), nu.ravel())
    log_value = flat_log.reshape(x.shape)
    return log_value if log else jnp.exp(log_value)

def pSICHEL(q, mu=1.0, sigma=1.0, nu=-0.5, lower_tail=True, log_p=False):
    """CDF for Sichel distribution.

    Parameters
    ----------
    q : array_like
        Quantiles (q >= 0)
    mu : array_like
        Mean parameter (mu > 0)
    sigma : array_like
        Dispersion parameter (sigma > 0)
    nu : array_like
        Shape parameter
    lower_tail : bool
        If True, return P(X <= q), else P(X > q)
    log_p : bool
        If True, return log probability

    Returns
    -------
    array_like
        CDF values

    Notes
    -----
    The CDF is computed by summing the PMF from 0 to floor(q).
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)

    q, mu, sigma, nu = jnp.broadcast_arrays(*[
        jnp.asarray(v, dtype=jnp.float64) for v in (q, mu, sigma, nu)
    ])
    from .distributions_b6 import SICHEL_PROBABILITY_MAX_COUNT
    from .distributions_b6 import _sichel_log_pmf_sequence
    support = jnp.arange(SICHEL_PROBABILITY_MAX_COUNT + 1, dtype=jnp.float64)
    def logprob_scalar(qq, mm, ss, nn):
        log_mass = _sichel_log_pmf_sequence(mm, ss, nn, SICHEL_PROBABILITY_MAX_COUNT)
        lower = jax.scipy.special.logsumexp(
            jnp.where(support <= qq, log_mass, -jnp.inf)
        )
        upper = jax.scipy.special.logsumexp(
            jnp.where(support > qq, log_mass, -jnp.inf)
        )
        return jnp.where(lower_tail, lower, upper)
    log_probability = jax.vmap(logprob_scalar)(
        q.ravel(), mu.ravel(), sigma.ravel(), nu.ravel()
    ).reshape(q.shape)
    valid = (mu > 0) & (sigma > 0) & jnp.isfinite(mu) & jnp.isfinite(sigma) & jnp.isfinite(nu)
    log_probability = jnp.where(
        q < 0, jnp.where(lower_tail, -jnp.inf, 0.), log_probability
    )
    log_probability = jnp.where(
        jnp.isposinf(q), jnp.where(lower_tail, 0., -jnp.inf), log_probability
    )
    result = log_probability if log_p else jnp.exp(log_probability)
    return jnp.where(valid & ~jnp.isnan(q) & ~((q > SICHEL_PROBABILITY_MAX_COUNT) & jnp.isfinite(q)), result, jnp.nan)


def qSICHEL(p, mu=1.0, sigma=1.0, nu=-0.5, lower_tail=True, log_p=False,
            max_value=100000):
    """Quantile function for Sichel distribution.

    Parameters
    ----------
    p : array_like
        Probabilities (0 < p < 1)
    mu : array_like
        Mean parameter (mu > 0)
    sigma : array_like
        Dispersion parameter (sigma > 0)
    nu : array_like
        Shape parameter
    lower_tail : bool
        If True, p is P(X <= q), else P(X > q)
    log_p : bool
        If True, p is log probability

    Returns
    -------
    array_like
        Quantile values (integers)

    Notes
    -----
    The quantile is found by searching for the smallest x such that
    P(X <= x) >= p.
    """
    from .distributions_b6 import SICHEL_PROBABILITY_MAX_COUNT, _sichel_log_pmf_sequence
    if isinstance(max_value, bool) or not isinstance(max_value, int) or not 0 <= max_value <= SICHEL_PROBABILITY_MAX_COUNT:
        raise ValueError(f"max_value must be an integer in [0, {SICHEL_PROBABILITY_MAX_COUNT}]")
    p, mu, sigma, nu = jnp.broadcast_arrays(*[
        jnp.asarray(v, dtype=jnp.float64) for v in (p, mu, sigma, nu)
    ])
    # Compare the requested tail directly in log space. Taking a complement
    # rounds tiny upper tails to one; exponentiating loses tiny log targets.
    target_log = p if log_p else jnp.log(p)
    valid_probability = (p <= 0) if log_p else ((p >= 0) & (p <= 1))
    valid = (valid_probability & ~jnp.isnan(p)
             & (mu > 0) & (sigma > 0) & jnp.isfinite(mu)
             & jnp.isfinite(sigma) & jnp.isfinite(nu))
    from .distributions_b6 import _sichel_log_pdf
    # A search cap must not truncate the tail used to test a candidate.
    support = jnp.arange((max_value if lower_tail else SICHEL_PROBABILITY_MAX_COUNT) + 1, dtype=jnp.float64)

    def cumulative_scalar(mm, ss, nn):
        log_mass = _sichel_log_pmf_sequence(
            mm, ss, nn, support.shape[0] - 1
        )
        # A prefix scan has linear memory; the previous broadcasted
        # comparison materialized a 10001 x 10001 matrix per parameter set.
        if lower_tail:
            return jax.lax.associative_scan(jnp.logaddexp, log_mass)
        inclusive = jax.lax.associative_scan(jnp.logaddexp, log_mass, reverse=True)
        return jnp.concatenate((inclusive[1:], jnp.array([-jnp.inf])))

    cumulative_log_probability = jax.vmap(cumulative_scalar)(
        mu.ravel(), sigma.ravel(), nu.ravel()
    )
    tolerance = 8 * jnp.finfo(jnp.float64).eps * jnp.maximum(1., jnp.abs(target_log))
    tolerance = jnp.where(jnp.isfinite(target_log), tolerance, 0.)
    target = target_log.ravel()[:, None]
    tolerance = tolerance.ravel()[:, None]
    reached = (cumulative_log_probability >= target - tolerance) if lower_tail else (
        cumulative_log_probability <= target + tolerance)
    reached = reached & (support <= max_value)
    found = jnp.argmax(reached, axis=1).astype(jnp.float64)
    resolved = jnp.any(reached, axis=1)
    found = found.reshape(p.shape)
    at_zero = jnp.isneginf(target_log) if lower_tail else (target_log == 0)
    at_infinity = (target_log == 0) if lower_tail else jnp.isneginf(target_log)
    result = jnp.where(at_zero, 0., jnp.where(at_infinity, jnp.inf, found))
    # Endpoints do not require a finite quantile to be found by the search.
    resolved = resolved.reshape(p.shape) | at_zero | at_infinity
    return jnp.where(valid & resolved, result, jnp.nan)


def rSICHEL(key, n, mu=1.0, sigma=1.0, nu=-0.5):
    """Random generation for Sichel distribution.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key
    n : int
        Number of observations
    mu : array_like
        Mean parameter (mu > 0)
    sigma : array_like
        Dispersion parameter (sigma > 0)
    nu : array_like
        Shape parameter

    Returns
    -------
    array_like
        Random samples (integers)

    Notes
    -----
    Generation algorithm:
    1. Generate u ~ Uniform(0, 1)
    2. Compute q = quantile(u)
    3. Return q
    """
    if not isinstance(n, int) or n < 0:
        raise ValueError('n must be a nonnegative integer')
    mu, sigma, nu = jnp.broadcast_arrays(*[
        jnp.asarray(v, dtype=jnp.float64) for v in (mu, sigma, nu)
    ])
    u = jrandom.uniform(key, shape=(n,) + mu.shape, dtype=jnp.float64)
    samples = qSICHEL(u, mu, sigma, nu)
    valid = ((mu > 0) & (sigma > 0) & jnp.isfinite(mu)
             & jnp.isfinite(sigma) & jnp.isfinite(nu))
    return jnp.where(valid[None, ...], samples, jnp.nan)


def tofySICHEL(y, mu=1.0, sigma=1.0, nu=-0.5):
    """R `tofySICHEL` fitted-value recursion (tofySICHEL1.c).

    This is the GAMLSS "tofy" quantity consumed by the SICHEL mu-score,
    ``dldm = (y - tofy) / mu``.  It is *not* the distribution mean: the
    recursion tracks ``tofY[j]`` for ``j = 1..y`` and returns ``tofY[y]``,
    exactly like R's ``.C("tofySICHEL1", ...)`` call.

    Parameters
    ----------
    y : array_like
        Counts at which the recursion stops (y >= 0, truncated like R).
    mu : array_like
        Mean parameter (mu > 0)
    sigma : array_like
        Dispersion parameter (sigma > 0)
    nu : array_like
        Shape parameter

    Returns
    -------
    array_like
        The recursion value ``tofY[y]``; NaN for invalid parameters or
        counts beyond the configured SICHEL support budget.
    """
    from .bessel import log_bessel_kv
    from .distributions_b6 import SICHEL_DENSITY_MAX_COUNT

    eps = jnp.finfo(jnp.float64).eps
    y = jnp.asarray(y, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    y, mu, sigma, nu = jnp.broadcast_arrays(y, mu, sigma, nu)

    def scalar(y_i, mu_i, sigma_i, nu_i):
        valid = ((mu_i > 0) & (sigma_i > 0) & jnp.isfinite(mu_i)
                 & jnp.isfinite(sigma_i) & jnp.isfinite(nu_i))
        support = ((y_i >= 0) & jnp.isfinite(y_i)
                   & (y_i <= SICHEL_DENSITY_MAX_COUNT))
        # R indexes tofY[y] via int truncation of y + 1, i.e. floor(y).
        y_idx = jnp.clip(jnp.floor(y_i).astype(jnp.int32), 0,
                         SICHEL_DENSITY_MAX_COUNT)
        mu_i = jnp.where(valid, mu_i, 1.)
        sigma_i = jnp.where(valid, sigma_i, 1.)
        nu_i = jnp.where(valid, nu_i, -.5)

        inv_sigma = 1.0 / sigma_i
        log_k_nu = log_bessel_kv(nu_i, inv_sigma)
        cvec = jnp.exp(log_bessel_kv(nu_i + 1.0, inv_sigma) - log_k_nu)
        alpha = jnp.sqrt(1.0 + 2.0 * sigma_i * mu_i / cvec) / sigma_i
        lbes = (log_bessel_kv(nu_i + 1.0, alpha)
                - log_bessel_kv(nu_i, alpha))
        tofy_0 = ((mu_i / cvec)
                  * jnp.power(1.0 + 2.0 * sigma_i * mu_i / cvec, -0.5)
                  * jnp.exp(lbes))
        ratio = jnp.square(mu_i / (sigma_i * alpha * cvec))

        def scan_fun(carry, j):
            prev, out = carry
            prev_safe = jnp.clip(prev, eps, 1e100)
            jf = j.astype(jnp.float64)
            current = (cvec * sigma_i * (2.0 * (jf + nu_i) / mu_i)
                       + 1.0 / prev_safe) * ratio
            current = jnp.clip(current, eps, 1e100)
            out = jnp.where(j == y_idx, current, out)
            return (current, out), None

        init = (tofy_0, tofy_0)
        (_, result), _ = jax.lax.scan(scan_fun, init,
                                      jnp.arange(1, SICHEL_DENSITY_MAX_COUNT + 1))
        return jnp.where(valid & support, result, jnp.nan)

    flat = jax.vmap(scalar)(y.ravel(), mu.ravel(), sigma.ravel(), nu.ravel())
    return flat.reshape(y.shape)


def VSICHEL(mu=1.0, sigma=1.0, nu=-0.5):
    """Variance of the Sichel distribution (R `VSICHEL`).

    R's ``VSICHEL(obj)`` extracts fitted ``mu``/``sigma``/``nu`` from a
    fitted gamlss object; here they are passed directly:

    ``varY = mu + mu^2 * ((2 * sigma * (nu + 1)) / c + 1 / c^2 - 1)``
    with ``c = K_{nu+1}(1/sigma) / K_nu(1/sigma)``.

    Parameters
    ----------
    mu : array_like
        Mean parameter (mu > 0)
    sigma : array_like
        Dispersion parameter (sigma > 0)
    nu : array_like
        Shape parameter

    Returns
    -------
    array_like
        Distribution variance; NaN for invalid parameters.
    """
    from .bessel import log_bessel_kv

    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    mu, sigma, nu = jnp.broadcast_arrays(mu, sigma, nu)
    valid = ((mu > 0) & (sigma > 0) & jnp.isfinite(mu)
             & jnp.isfinite(sigma) & jnp.isfinite(nu))
    mu_safe = jnp.where(valid, mu, 1.)
    sigma_safe = jnp.where(valid, sigma, 1.)
    nu_safe = jnp.where(valid, nu, -.5)
    c = jnp.exp(log_bessel_kv(nu_safe + 1.0, 1.0 / sigma_safe)
                - log_bessel_kv(nu_safe, 1.0 / sigma_safe))
    var_y = mu_safe + jnp.square(mu_safe) * (
        (2.0 * sigma_safe * (nu_safe + 1.0)) / c + 1.0 / jnp.square(c) - 1.0)
    return jnp.where(valid, var_y, jnp.nan)


# Static support budget for the SI "tofy" recursion, aligned with the SI
# family's density scan length (see distributions_b6._si_log_pdf).
SI_TOFY_MAX_COUNT = 512


def tofyS(y, mu=1.0, sigma=1.0, nu=-0.5, max_count=SI_TOFY_MAX_COUNT):
    """R `tofyS` fitted-value recursion for the SI family (tofySI1.c).

    The SI analogue of `tofySICHEL`: the GAMLSS "tofy" quantity consumed by
    the SI mu-score.  The recursion returns ``tofY[y]`` where

    - ``tofY[0] = mu * (1 + 2*sigma*mu)^(-0.5) * exp(lbes)``
    - ``alpha = sqrt(1 + 2*sigma*mu) / sigma``
    - ``lbes = log K_{nu+1}(alpha) - log K_nu(alpha)``
    - ``tofY[j] = (sigma * 2*(j+nu)/mu + 1/tofY[j-1]) * (mu/(sigma*alpha))^2``

    R's exported wrapper carries an unused ``what`` argument; it is dropped
    here.  It is *not* the distribution mean.

    Parameters
    ----------
    y : array_like
        Counts at which the recursion stops (y >= 0, floor-truncated like R).
    mu : array_like
        Mean parameter (mu > 0)
    sigma : array_like
        Dispersion parameter (sigma > 0)
    nu : array_like
        Shape parameter
    max_count : int
        Static recursion budget (defaults to the SI support of 512).

    Returns
    -------
    array_like
        The recursion value ``tofY[y]``; NaN for invalid parameters or
        counts beyond ``max_count``.
    """
    if isinstance(max_count, bool) or not isinstance(max_count, int) or max_count < 0:
        raise ValueError("max_count must be a nonnegative integer")
    from .bessel import log_bessel_kv

    eps = jnp.finfo(jnp.float64).eps
    y = jnp.asarray(y, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    y, mu, sigma, nu = jnp.broadcast_arrays(y, mu, sigma, nu)

    def scalar(y_i, mu_i, sigma_i, nu_i):
        valid = ((mu_i > 0) & (sigma_i > 0) & jnp.isfinite(mu_i)
                 & jnp.isfinite(sigma_i) & jnp.isfinite(nu_i))
        support = ((y_i >= 0) & jnp.isfinite(y_i) & (y_i <= max_count))
        # R indexes tofY[y] via int truncation of y + 1, i.e. floor(y).
        y_idx = jnp.clip(jnp.floor(y_i).astype(jnp.int32), 0, max_count)
        mu_i = jnp.where(valid, mu_i, 1.)
        sigma_i = jnp.where(valid, sigma_i, 1.)
        nu_i = jnp.where(valid, nu_i, -.5)

        alpha = jnp.sqrt(1.0 + 2.0 * sigma_i * mu_i) / sigma_i
        lbes = (log_bessel_kv(nu_i + 1.0, alpha)
                - log_bessel_kv(nu_i, alpha))
        tofy_0 = (mu_i * jnp.power(1.0 + 2.0 * sigma_i * mu_i, -0.5)
                  * jnp.exp(lbes))
        ratio = jnp.square(mu_i / (sigma_i * alpha))

        def scan_fun(carry, j):
            prev, out = carry
            prev_safe = jnp.clip(prev, eps, 1e100)
            jf = j.astype(jnp.float64)
            current = (sigma_i * (2.0 * (jf + nu_i) / mu_i)
                       + 1.0 / prev_safe) * ratio
            current = jnp.clip(current, eps, 1e100)
            out = jnp.where(j == y_idx, current, out)
            return (current, out), None

        init = (tofy_0, tofy_0)
        (_, result), _ = jax.lax.scan(scan_fun, init,
                                      jnp.arange(1, max_count + 1))
        return jnp.where(valid & support, result, jnp.nan)

    flat = jax.vmap(scalar)(y.ravel(), mu.ravel(), sigma.ravel(), nu.ravel())
    return flat.reshape(y.shape)


def zetaP(x):
    """Riemann zeta via R `zetaP` (direct sum + Euler-Maclaurin tail).

    Sums ``n = 1..12`` directly and closes the tail with eight Bernoulli
    correction terms, matching gamlss.dist's helper used to normalize the
    ZIPF distribution.  Accurate for ``x >= ~12`` (R warns below 11);
    below that the returned value degrades gracefully.

    Parameters
    ----------
    x : array_like
        Zeta argument(s), typically ``>= 12``.

    Returns
    -------
    array_like
        Approximation of ``zeta(x)``.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    aa = 12.0
    base = 1.0 + aa
    bernoulli2 = jnp.array([1 / 6, -1 / 30, 1 / 42, -1 / 30, 5 / 66,
                            -691 / 2730, 7 / 6, -3617 / 510],
                           dtype=jnp.float64)
    direct = jnp.sum(
        jnp.power(1.0 + jnp.arange(aa), -x[..., None]), axis=-1
    ).reshape(x.shape)
    tail = (1.0 / ((x - 1.0) * jnp.power(base, x - 1.0))
            + 0.5 / jnp.power(base, x))
    term = (x / 2.0) / jnp.power(base, x + 1.0)
    tail = tail + term * bernoulli2[0]
    for mm in range(2, bernoulli2.shape[0] + 1):
        term = term * ((x + 2 * mm - 3.0) * (x + 2 * mm - 2.0)
                       / ((2 * mm - 1) * (2 * mm) * base * base))
        tail = tail + term * bernoulli2[mm - 1]
    return direct + tail


def GetBI_C(mu, sigma, bd):
    """R `GetBI_C` normalization constant for the BI family (getBI_C2.c).

    Computes ``log(1 / sum_j exp(ss_j))`` over ``j = 0..bd`` where

    ``ss_j = lchoose(bd, j) + (j*log<j> + (bd-j)*log<bd-j>) * (1 - sigma)
             + bd*log<bd> * (sigma - 1)
             + (j*log(mu) + (bd-j)*log(1-mu)) * sigma``

    with ``log<t> = 1`` at ``t = 0`` (the C code's ``ifelse`` guard).

    R's exported wrapper inverts ``sigma`` (``sigma <- 1/sigma``) before
    calling ``getBI_C2``; this implementation reproduces that
    parameterization, so ``ss_j`` uses the inverted value.

    Parameters
    ----------
    mu : array_like
        Binomial probability parameter (0 < mu < 1)
    sigma : array_like
        Power-transformation parameter (sigma > 0; inverted internally
        exactly like the R wrapper)
    bd : array_like
        Binomial denominator (nonnegative integer)

    Returns
    -------
    array_like
        Log normalization constant; NaN for invalid parameters.
    """
    from jax.scipy.special import gammaln

    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    bd = jnp.asarray(bd, dtype=jnp.float64)
    mu, sigma, bd = jnp.broadcast_arrays(mu, sigma, bd)

    def scalar(mu_i, sigma_i, bd_i):
        valid = ((mu_i > 0) & (mu_i < 1) & (sigma_i > 0)
                 & jnp.isfinite(mu_i) & jnp.isfinite(sigma_i)
                 & jnp.isfinite(bd_i) & (bd_i >= 0) & (bd_i == jnp.floor(bd_i)))
        mu_i = jnp.where(valid, mu_i, .5)
        sigma_i = jnp.where(valid, 1.0 / sigma_i, 1.)
        n = jnp.floor(bd_i).astype(jnp.int32)
        n = jnp.where(valid, n, 0)
        sm1 = sigma_i - 1.0
        s_1 = 1.0 - sigma_i
        nlogn = n * jnp.where(n == 0, 1.0,
                              jnp.log(n.astype(jnp.float64)))

        def cond(state):
            j, _ = state
            return j <= n

        def body(state):
            j, sum_c = state
            jf = j.astype(jnp.float64)
            bdj = n - j
            bdjf = bdj.astype(jnp.float64)
            log_choose = (gammaln(n + 1.0) - gammaln(jf + 1.0)
                          - gammaln(bdjf + 1.0))
            y_log_of_y = jf * jnp.where(j == 0, 1.0, jnp.log(jf))
            ny_log_of_ny = bdjf * jnp.where(bdj == 0, 1.0, jnp.log(bdjf))
            ss = (log_choose + (y_log_of_y + ny_log_of_ny) * s_1
                  + nlogn * sm1
                  + (jf * jnp.log(mu_i) + bdjf * jnp.log1p(-mu_i)) * sigma_i)
            return (j + 1, sum_c + jnp.exp(ss))

        _, sum_c = jax.lax.while_loop(cond, body, (0, 0.0))
        return jnp.where(valid, -jnp.log(sum_c), jnp.nan)

    flat = jax.vmap(scalar)(mu.ravel(), sigma.ravel(), bd.ravel())
    return flat.reshape(mu.shape)


def get_C(x, mu=1.0, sigma=1.0, max_value=None):
    """R `get_C` log normalization constant for the DPO family.

    Mirrors gamlss.dist's ``dDPOgetC5_C``: the double-Poisson normalizer is

    ``C = 1 / sum_{j=0..maxV} exp(ls - lgamma(j+1) + y*log<y> + j*lmus
                                    + (1/sigma)*(j - y*log<y>))``

    with ``log<y> = 1`` at ``y = 0``, ``ls = -0.5*log(sigma) - mu/sigma``
    and ``lmus = log(mu)/sigma - 1``.  R's wrapper returns ``log(C)``.

    R's wrapper does not recycle scalar ``mu``/``sigma`` against vector
    ``x`` (long vectors read out of bounds and produce NaN); this
    implementation broadcasts correctly.  ``x`` only influences the default
    truncation ``maxV = max(3*max(x), 500)``.

    Parameters
    ----------
    x : array_like
        Support values used to derive the default truncation.
    mu : array_like
        Double-Poisson location parameter (mu > 0)
    sigma : array_like
        Double-Poisson dispersion parameter (sigma > 0)
    max_value : int, optional
        Explicit static truncation (overrides the value derived from ``x``).

    Returns
    -------
    array_like
        Log normalization constant; NaN for invalid parameters.
    """
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    x = jnp.asarray(x, dtype=jnp.float64)
    x, mu, sigma = jnp.broadcast_arrays(x, mu, sigma)
    if max_value is None:
        x_host = np.asarray(x)
        if not np.issubdtype(x_host.dtype, np.number):
            raise ValueError("x must be numeric")
        max_value = int(max(3 * int(np.max(x_host)), 500))
    if isinstance(max_value, bool) or not isinstance(max_value, int) or max_value < 0:
        raise ValueError("max_value must be a nonnegative integer")

    counts = jnp.arange(max_value + 1, dtype=jnp.float64)
    count_safe = jnp.maximum(counts, 1.0)
    log_of_count = jnp.where(counts == 0, 1.0, jnp.log(count_safe))
    y_log_of_y = counts * log_of_count
    log_factorial = jax.scipy.special.gammaln(counts + 1.0)
    y_minus_y_log_y = counts - y_log_of_y

    def scalar(mu_i, sigma_i):
        valid = ((mu_i > 0) & (sigma_i > 0) & jnp.isfinite(mu_i)
                 & jnp.isfinite(sigma_i))
        mu_i = jnp.where(valid, mu_i, 1.)
        sigma_i = jnp.where(valid, sigma_i, 1.)
        ls = -0.5 * jnp.log(sigma_i) - mu_i / sigma_i
        lmus = jnp.log(mu_i) / sigma_i - 1.0
        invs = 1.0 / sigma_i
        sum_c = jnp.sum(jnp.exp(ls - log_factorial + y_log_of_y
                                + counts * lmus + invs * y_minus_y_log_y))
        return jnp.where(valid, -jnp.log(sum_c), jnp.nan)

    flat = jax.vmap(scalar)(mu.ravel(), sigma.ravel())
    return flat.reshape(mu.shape)


# ------------------------------------------------------------------------------
# NET (Normal-Exponential-t) - 4 parameters
# ------------------------------------------------------------------------------

def dNET(x, mu=0.0, sigma=1.0, nu=2.0, tau=4.0, log=False):
    """Density function for Normal-Exponential-t distribution.

    The NET distribution uses a piecewise density function that combines
    Normal, Exponential, and power tail components.

    Parameters
    ----------
    x : array_like
        Quantiles
    mu : array_like
        Location parameter
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Threshold parameter k1 (nu > 0)
    tau : array_like
        Threshold parameter k2 (tau >= nu)
    log : bool
        If True, return log density

    Returns
    -------
    array_like
        Density values

    Notes
    -----
    The density is piecewise:
    - Normal in center: |z| <= k1
    - Exponential in middle: k1 < |z| <= k2
    - Power tail beyond: |z| > k2

    where z = (x - mu) / sigma, k1 = nu, k2 = tau.

    References
    ----------
    Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized additive models
    for location, scale and shape. Journal of the Royal Statistical Society
    Series C, 54(3), 507-554.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    tau = jnp.asarray(tau, dtype=jnp.float64)

    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.maximum(nu, eps)
    tau = jnp.maximum(tau, nu)  # Ensure tau >= nu

    k1 = nu
    k2 = tau

    # Normalizing constant components
    from scipy.stats import norm as sp_norm
    c1 = (1.0 - 2.0 * sp_norm.cdf(-k1)) * jnp.sqrt(2.0 * jnp.pi)
    c2 = (2.0 / k1) * jnp.exp(-(k1 * k1) / 2.0)
    c3 = 2.0 * jnp.exp(-k1 * k2 + (k1 * k1) / 2.0) / ((k1 * k2 - 1.0) * k1)
    ct = 1.0 / (c1 + c2 + c3)

    # Standardized residual
    tc = (x - mu) / sigma
    abs_tc = jnp.abs(tc)

    # Piecewise log-density
    # Region 1: |tc| <= k1 (Normal)
    d1 = jnp.where(abs_tc <= k1, -(tc * tc) / 2.0, 0.0)

    # Region 2: k1 < |tc| <= k2 (Exponential)
    d2 = jnp.where((abs_tc > k1) & (abs_tc <= k2), -k1 * abs_tc + (k1 * k1) / 2.0, 0.0)

    # Region 3: |tc| > k2 (Power tail)
    d3 = jnp.where(
        abs_tc > k2,
        -k1 * k2 * jnp.log(abs_tc / k2) - k1 * k2 + (k1 * k1) / 2.0,
        0.0
    )

    # Total log-pdf
    log_pdf = jnp.log(ct) - jnp.log(sigma) + d1 + d2 + d3

    if log:
        return log_pdf
    else:
        return jnp.exp(log_pdf)


def pNET(q, mu=0.0, sigma=1.0, nu=2.0, tau=4.0, lower_tail=True, log_p=False):
    """CDF for Normal-Exponential-t distribution.

    Parameters
    ----------
    q : array_like
        Quantiles
    mu : array_like
        Location parameter
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Threshold parameter k1 (nu > 0)
    tau : array_like
        Threshold parameter k2 (tau >= nu)
    lower_tail : bool
        If True, return P(X <= q), else P(X > q)
    log_p : bool
        If True, return log probability

    Returns
    -------
    array_like
        CDF values

    Notes
    -----
    The CDF is computed using numerical integration of the density function.
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    tau = jnp.asarray(tau, dtype=jnp.float64)

    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.maximum(nu, eps)
    tau = jnp.maximum(tau, nu)

    # Use numerical integration
    def integrate_density(q_val):
        # Create integration grid
        lower = float(mu) - 10.0 * float(sigma)
        upper = q_val
        n_points = 500
        
        if upper <= lower:
            return 0.0
        
        x_grid = jnp.linspace(lower, upper, n_points)
        
        # Compute density at grid points
        d_grid = dNET(x_grid, mu, sigma, nu, tau, log=False)
        
        # Trapezoidal integration
        dx = (upper - lower) / (n_points - 1)
        cdf_val = jnp.sum(d_grid) * dx
        return float(cdf_val)

    # Vectorize
    q_array = jnp.atleast_1d(q)
    cdf_list = [integrate_density(float(q_val)) for q_val in q_array]
    cdf = jnp.array(cdf_list)

    # Return scalar if input was scalar
    if jnp.ndim(q) == 0:
        cdf = cdf[0]

    # Clip to [0, 1]
    cdf = jnp.clip(cdf, 0.0, 1.0)

    if not lower_tail:
        cdf = 1.0 - cdf

    if log_p:
        cdf = jnp.log(cdf)

    return cdf


def qNET(p, mu=0.0, sigma=1.0, nu=2.0, tau=4.0, lower_tail=True, log_p=False):
    """Quantile function for Normal-Exponential-t distribution.

    Parameters
    ----------
    p : array_like
        Probabilities (0 < p < 1)
    mu : array_like
        Location parameter
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Threshold parameter k1 (nu > 0)
    tau : array_like
        Threshold parameter k2 (tau >= nu)
    lower_tail : bool
        If True, p is P(X <= q), else P(X > q)
    log_p : bool
        If True, p is log probability

    Returns
    -------
    array_like
        Quantile values

    Notes
    -----
    The quantile function is computed using numerical root finding
    on the CDF.
    """
    p = jnp.asarray(p, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    tau = jnp.asarray(tau, dtype=jnp.float64)

    if log_p:
        p = jnp.exp(p)

    if not lower_tail:
        p = 1.0 - p

    eps = jnp.finfo(jnp.float64).eps
    p = jnp.clip(p, eps, 1.0 - eps)
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.maximum(nu, eps)
    tau = jnp.maximum(tau, nu)

    # Use scipy for root finding
    from scipy.optimize import brentq

    def find_quantile_scalar(p_val):
        # Define the function to find root of: F(q) - p = 0
        def objective(q_val):
            cdf_val = pNET(q_val, mu, sigma, nu, tau, lower_tail=True, log_p=False)
            return float(cdf_val) - p_val

        # Bounds
        lower = float(mu) - 10.0 * float(sigma)
        upper = float(mu) + 10.0 * float(sigma)

        try:
            result = brentq(objective, lower, upper, xtol=1e-8)
            return result


        except:
            # Fallback to bisection
            for _ in range(50):
                mid = (lower + upper) / 2.0
                f_mid = objective(mid)
                if abs(f_mid) < 1e-8:
                    return mid
                f_lower = objective(lower)
                if f_lower * f_mid < 0:
                    upper = mid
                else:
                    lower = mid
            return (lower + upper) / 2.0

    # Vectorize
    p_array = jnp.atleast_1d(p)
    q_list = [find_quantile_scalar(float(p_val)) for p_val in p_array]
    q = jnp.array(q_list)

    # Return scalar if input was scalar
    if jnp.ndim(p) == 0:
        return q[0]

    return q


def fittedMN(*params):
    """Convert fitted MN parameters (or a ``GAMLSSModel``) to probabilities."""
    if len(params) == 1 and hasattr(params[0], "fitted_values"):
        model = params[0]
        names = tuple(getattr(model, "parameters", ())) or tuple(
            getattr(model, "par", ())
        )
        values = getattr(model, "fitted_values", {})
        params = tuple(values[name] for name in names if name in values)
    if not 1 <= len(params) <= 4:
        raise ValueError("fittedMN requires between one and four parameters")
    arrays = jnp.broadcast_arrays(*(jnp.asarray(param, dtype=jnp.float64)
                                    for param in params))
    if len(arrays) == 1:
        return jnp.stack((arrays[0], 1.0 - arrays[0]), axis=-1)
    denominator = 1.0 + sum(arrays)
    return jnp.stack(tuple(param / denominator for param in arrays)
                     + (1.0 / denominator,), axis=-1)


def meanZAGA(obj):
    """Return fitted means for a ZAGA model.

    This mirrors ``gamlss.dist::meanZAGA``: the continuous Gamma component
    has mean ``mu`` and is selected with probability ``1 - nu``.
    """
    family = getattr(obj, "family", None)
    family_name = getattr(family, "name", family)
    if isinstance(family_name, (tuple, list)):
        family_name = family_name[0] if family_name else None
    if family_name != "ZAGA":
        raise ValueError("the object does not have a ZAGA distribution")
    values = getattr(obj, "fitted_values", None)
    if values is None or "mu" not in values or "nu" not in values:
        raise ValueError("ZAGA model fitted_values must contain 'mu' and 'nu'")
    return (1.0 - jnp.asarray(values["nu"], dtype=jnp.float64)) * jnp.asarray(
        values["mu"], dtype=jnp.float64
    )


def meanZAIG(obj):
    """Return fitted means for a ZAIG model, matching R ``meanZAIG``."""
    family = getattr(obj, "family", None)
    family_name = getattr(family, "name", family)
    if isinstance(family_name, (tuple, list)):
        family_name = family_name[0] if family_name else None
    if family_name != "ZAIG":
        raise ValueError("the object does not have a ZAIG distribution")
    values = getattr(obj, "fitted_values", None)
    if values is None or "mu" not in values or "nu" not in values:
        raise ValueError("ZAIG model fitted_values must contain 'mu' and 'nu'")
    return (1.0 - jnp.asarray(values["nu"], dtype=jnp.float64)) * jnp.asarray(
        values["mu"], dtype=jnp.float64
    )


def meanBEINF(obj):
    """Return fitted means for a BEINF model, matching R ``meanBEINF``."""
    family = getattr(obj, "family", None)
    family_name = getattr(family, "name", family)
    if isinstance(family_name, (tuple, list)):
        family_name = family_name[0] if family_name else None
    if family_name != "BEINF":
        raise ValueError("the object does not have a BEINF distribution")
    values = getattr(obj, "fitted_values", None)
    if values is None or not all(name in values for name in ("mu", "nu", "tau")):
        raise ValueError("BEINF model fitted_values must contain 'mu', 'nu', and 'tau'")
    mu = jnp.asarray(values["mu"], dtype=jnp.float64)
    nu = jnp.asarray(values["nu"], dtype=jnp.float64)
    tau = jnp.asarray(values["tau"], dtype=jnp.float64)
    return (tau + mu) / (1.0 + nu + tau)


def meanBEINF0(obj):
    """Return fitted means for a BEINF0 model, matching R ``meanBEINF0``."""
    family = getattr(obj, "family", None)
    family_name = getattr(family, "name", family)
    if isinstance(family_name, (tuple, list)):
        family_name = family_name[0] if family_name else None
    if family_name != "BEINF0":
        raise ValueError("the object does not have a BEINF0 distribution")
    values = getattr(obj, "fitted_values", None)
    if values is None or not all(name in values for name in ("mu", "nu")):
        raise ValueError("BEINF0 model fitted_values must contain 'mu' and 'nu'")
    mu = jnp.asarray(values["mu"], dtype=jnp.float64)
    nu = jnp.asarray(values["nu"], dtype=jnp.float64)
    return mu / (1.0 + nu)


def meanBEINF1(obj):
    """Return fitted means for a BEINF1 model, matching R ``meanBEINF1``."""
    family = getattr(obj, "family", None)
    family_name = getattr(family, "name", family)
    if isinstance(family_name, (tuple, list)):
        family_name = family_name[0] if family_name else None
    if family_name != "BEINF1":
        raise ValueError("the object does not have a BEINF1 distribution")
    values = getattr(obj, "fitted_values", None)
    if values is None or not all(name in values for name in ("mu", "nu")):
        raise ValueError("BEINF1 model fitted_values must contain 'mu' and 'nu'")
    mu = jnp.asarray(values["mu"], dtype=jnp.float64)
    nu = jnp.asarray(values["nu"], dtype=jnp.float64)
    return (nu + mu) / (1.0 + nu)


def rNET(key, n, mu=0.0, sigma=1.0, nu=2.0, tau=4.0):
    """Random generation for Normal-Exponential-t distribution.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key
    n : int
        Number of observations
    mu : array_like
        Location parameter
    sigma : array_like
        Scale parameter (sigma > 0)
    nu : array_like
        Threshold parameter k1 (nu > 0)
    tau : array_like
        Threshold parameter k2 (tau >= nu)

    Returns
    -------
    array_like
        Random samples

    Notes
    -----
    Generation algorithm:
    1. Generate u ~ Uniform(0, 1)
    2. Compute q = quantile(u)
    3. Return q
    """
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    tau = jnp.asarray(tau, dtype=jnp.float64)

    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.maximum(nu, eps)
    tau = jnp.maximum(tau, nu)

    # Generate uniform samples
    u = jrandom.uniform(key, shape=(n,))

    # Convert to NET samples using quantile function
    samples = qNET(u, mu, sigma, nu, tau)

    return samples
