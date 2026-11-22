"""GAMLSS distributions: Batch 9 — Additional discrete and continuous families.

This batch includes distributions that were missing from the initial implementation:
- LG (Logarithmic)
- ZIPF (Zipf)
- JSU (Johnson's SU)
- GIG (Generalized Inverse Gaussian)
- GB1 (Generalized Beta Type 1)
- And zero-inflated/altered variants
"""

from __future__ import annotations

# Enable float64 precision for numerical accuracy

from dataclasses import dataclass
import math
import numpy as np
from scipy.integrate import quad

import jax.numpy as jnp
import jax.random as jrandom
from jax.scipy.special import zeta

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

# ------------------------------------------------------------------
# 1. LG — Logarithmic Distribution
#    R: dLG(x, mu)
#    Support: {1, 2, 3, ...}
#    Parameter: mu ∈ (0, 1)
#    PMF: P(X=x) = mu^x / (x * (-log(1-mu)))
# ------------------------------------------------------------------


@dataclass(frozen=True)
class LogarithmicFamily(FamilyDefinition):
    """Logarithmic distribution (`LG`).

    Support: {1, 2, 3, ...}
    Parameter: mu ∈ (0, 1)

    The logarithmic distribution is a discrete probability distribution
    derived from the Maclaurin series expansion of -log(1-mu).

    PMF: P(X=x) = mu^x / (x * (-log(1-mu)))

    Mean: -mu / ((1-mu) * log(1-mu))
    Variance: -mu * (1 + mu/(-log(1-mu))) / ((1-mu)^2 * log(1-mu))

    Common uses:
    - Species abundance models
    - Accident counts
    - Rare event modeling
    - Truncated geometric processes
    """


def _lg_log_pdf(y, mu):
    """Log-PDF of logarithmic distribution.

    Args:
        y: Response variable (positive integers)
        mu: Parameter in (0, 1)

    Returns:
        Log-probability
    """
    y, mu = jnp.broadcast_arrays(
        jnp.asarray(y, dtype=jnp.float64), jnp.asarray(mu, dtype=jnp.float64)
    )
    valid = (mu > 0) & (mu < 1)
    mu = jnp.where(valid, mu, 0.5)
    # Evaluate the positive-support formula only at safe finite points.
    y_safe = jnp.where((y >= 1) & jnp.isfinite(y), y, 1.0)

    # Log-PDF: log(mu^y / (y * (-log(1-mu))))
    # = y*log(mu) - log(y) - log(-log(1-mu))
    log_pdf = y_safe * jnp.log(mu) - jnp.log(y_safe) - jnp.log(-jnp.log1p(-mu))

    # LG starts at one; positive infinity has zero limiting mass.
    log_pdf = jnp.where((y < 1) | jnp.isposinf(y), -jnp.inf, log_pdf)

    return jnp.where(valid & ~jnp.isnan(y), log_pdf, jnp.nan)


def _lg_d(x, mu=0.5, log=False):
    value = _lg_log_pdf(x, mu)
    return value if log else jnp.exp(value)


def _lg_p(q, mu=0.5, lower_tail=True, log_p=False):
    """CPU logarithmic CDF using the shared infinite-tail kernel."""
    return _logarithmic_p(q, mu, 0.0, lower_tail, log_p)


def _lg_q(p, mu=0.5, lower_tail=True, log_p=False, max_value=None):
    """Dynamic logarithmic quantile; unresolved explicit caps return NaN."""
    return _logarithmic_q(p, mu, 0.0, lower_tail, log_p, max_value)


def _lg_r(key, n, mu=0.5):
    import jax.numpy as _jnp
    import jax.random as _jr

    u = _jr.uniform(key, shape=(n,), dtype=_jnp.float64)
    return _lg_q(u, mu)


def LG() -> LogarithmicFamily:
    """Logarithmic distribution.

    Support: {1, 2, 3, ...}
    Parameter: mu ∈ (0, 1)

    Returns:
        LogarithmicFamily instance

    Example:
        >>> from omnilss import gamlss
        >>> from omnilss.distributions import LG
        >>>
        >>> # Fit logarithmic distribution
        >>> model = gamlss("y ~ x", family=LG(), data=data)

    Notes:
        - Default link for mu is logit
        - mu must be in (0, 1)
        - y must be positive integers
        - Mean: -mu / ((1-mu) * log(1-mu))
        - Variance: -mu * (1 + mu/(-log(1-mu))) / ((1-mu)^2 * log(1-mu))

    References:
        - Johnson, N. L., Kemp, A. W., & Kotz, S. (2005).
          Univariate Discrete Distributions (3rd ed.). Wiley.
        - R gamlss.dist::LG
    """
    return build_ad_family(
        family_class=LogarithmicFamily,
        name="LG",
        parameters=("mu",),
        log_pdf_func=_lg_log_pdf,
        type_="Discrete",
        d=_lg_d,
        p=_lg_p,
        q=_lg_q,
        r=_lg_r,
        links={"mu": "logit"},
        link_functions={"mu": logit_link},
        link_inverses={"mu": logit_inverse},
        link_derivatives={"mu": logit_derivative},
    )


# ------------------------------------------------------------------
# 2. ZIPF — Zipf Distribution
#    R: dZIPF(x, mu)
#    Support: {1, 2, 3, ...}
#    Parameter: mu > 0 (shape parameter, often called s or α)
#    PMF: P(X=x) = x^(-(mu+1)) / ζ(mu+1)
#    where ζ is the Riemann zeta function
# ------------------------------------------------------------------


@dataclass(frozen=True)
class ZipfFamily(FamilyDefinition):
    """Zipf distribution (`ZIPF`).

    Support: {1, 2, 3, ...}
    Parameter: mu > 0 (shape parameter)

    The Zipf distribution is a discrete power-law probability distribution.
    It's commonly used to model word frequencies, city populations, and
    other phenomena following Zipf's law.

    PMF: P(X=x) = x^(-(mu+1)) / ζ(mu+1)
    where ζ is the Riemann zeta function

    Mean: ζ(mu) / ζ(mu+1) if mu > 1, else Inf
    Variance: [ζ(mu+1)*ζ(mu-1) - ζ(mu)^2] / ζ(mu+1)^2 if mu > 2, else Inf

    Common uses:
    - Word frequency analysis
    - City population distributions
    - Website traffic patterns
    - Power-law phenomena
    """


def _zipf_log_pdf(y, mu):
    """Log-PDF of the R-compatible Zipf distribution."""
    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.maximum(jnp.asarray(mu, dtype=jnp.float64), eps)
    y = jnp.asarray(y, dtype=jnp.float64)
    y_safe = jnp.maximum(y, 1.0)
    log_zeta = jnp.log(zeta(mu + 1.0, 1.0))
    log_pdf = -(mu + 1.0) * jnp.log(y_safe) - log_zeta
    return jnp.where(y < 1.0, -jnp.inf, log_pdf)


def _zipf_d(x, mu=1.0, log=False):
    """R ``dZIPF``: the support starts at one, but accepts real x >= 1."""
    x = jnp.asarray(x, dtype=jnp.float64)
    log_mass = _zipf_log_pdf(x, mu)
    # Match the explicit boundary assignments in R's dZIPF implementation.
    log_mass = jnp.where((x <= 0.0) | jnp.isposinf(x), -jnp.inf, log_mass)
    return log_mass if log else jnp.exp(log_mass)


def _zipf_tail(shape, a):
    """R ``pZIPF``'s Euler--Maclaurin tail approximation after ``a``."""
    bernoulli = jnp.asarray(
        [
            1.0 / 6.0,
            -1.0 / 30.0,
            1.0 / 42.0,
            -1.0 / 30.0,
            5.0 / 66.0,
            -691.0 / 2730.0,
            7.0 / 6.0,
            -3617.0 / 510.0,
        ],
        dtype=jnp.float64,
    )
    # R calls this helper with ``a = floor(q) + 1`` and uses ``1 + a``.
    a = a + 1.0
    result = 1.0 / ((shape - 1.0) * a ** (shape - 1.0))
    result = result + 0.5 / a**shape
    term = (shape / 2.0) / a ** (shape + 1.0)
    result = result + term * bernoulli[0]
    for index in range(1, 8):
        order = index + 1
        term = term * (shape + 2.0 * order - 3.0) * (shape + 2.0 * order - 2.0)
        term = term / ((2.0 * order - 1.0) * 2.0 * order * a**2)
        result = result + term * bernoulli[index]
    return result


def _zipf_cdf(q, mu):
    """Return R's lower-tail CDF before optional tail/log transformations."""
    q, mu = jnp.broadcast_arrays(
        jnp.asarray(q, dtype=jnp.float64), jnp.asarray(mu, dtype=jnp.float64)
    )
    shape = mu + 1.0
    k = jnp.arange(1, 12, dtype=jnp.float64)
    head = jnp.sum(
        jnp.where(k <= jnp.floor(q)[..., None], k ** (-shape[..., None]), 0.0), axis=-1
    )
    qfloor = jnp.floor(q)
    tail = _zipf_tail(shape, qfloor + 1.0)
    cdf = jnp.where(qfloor >= 11.0, zeta(shape, 1.0) - tail, head)
    cdf = cdf / zeta(shape, 1.0)
    cdf = jnp.where(q < 1.0, 0.0, cdf)
    cdf = jnp.where(jnp.isposinf(q), 1.0, cdf)
    return cdf


def _zipf_p(q, mu=1.0, lower_tail=True, log_p=False):
    """R ``pZIPF`` using the floor of q for the discrete CDF."""
    cdf = _zipf_cdf(q, mu)
    if not lower_tail:
        cdf = 1.0 - cdf
    return jnp.log(cdf) if log_p else cdf


def _zipf_q(p, mu=1.0, lower_tail=True, log_p=False, max_value=10000):
    """R ``qZIPF``: first exact CDF crossing, capped at ``max_value``."""
    p, mu = jnp.broadcast_arrays(
        jnp.asarray(p, dtype=jnp.float64), jnp.asarray(mu, dtype=jnp.float64)
    )
    if log_p:
        p = jnp.exp(p)
    if not lower_tail:
        p = 1.0 - p
    valid = (p >= 0.0) & (p <= 1.0)
    k = jnp.arange(1, max_value + 1, dtype=jnp.float64)
    # qZIPF calls pZIPF for each integer; use the same R tail approximation.
    cdf = _zipf_cdf(k, mu[..., None])
    crossed = cdf >= p[..., None]
    crossing = jnp.argmax(crossed, axis=-1) + 1.0
    result = jnp.where(jnp.any(crossed, axis=-1), crossing, float(max_value))
    result = jnp.where(p >= 1.0, jnp.inf, result)
    # R's generalized inverse returns zero at p == 0, before the first
    # positive support point.
    result = jnp.where(p == 0.0, 0.0, result)
    return jnp.where(valid, result, jnp.nan)


def _zipf_r(key, n, mu=1.0):
    """Generate Zipf variates by inverse CDF, matching R's qZIPF limit."""
    return _zipf_q(jrandom.uniform(key, shape=(n,), dtype=jnp.float64), mu)


def ZIPF() -> ZipfFamily:
    """Zipf distribution.

    Support: {1, 2, 3, ...}
    Parameter: mu > 0 (shape parameter)

    Returns:
        ZipfFamily instance

    Example:
        >>> from omnilss import gamlss
        >>> from omnilss.distributions import ZIPF
        >>>
        >>> # Fit Zipf distribution
        >>> model = gamlss("y ~ x", family=ZIPF(), data=data)

    Notes:
        - Default link for mu is log
        - mu must be positive
        - y must be positive integers
        - Mean: ζ(mu) / ζ(mu+1) if mu > 1
        - Variance: [ζ(mu+1)*ζ(mu-1) - ζ(mu)^2] / ζ(mu+1)^2 if mu > 2
        - Current implementation uses approximation for zeta function
        - For production use, consider using scipy.special.zeta

    References:
        - Zipf, G. K. (1949). Human Behavior and the Principle of Least Effort.
        - Newman, M. E. J. (2005). Power laws, Pareto distributions and Zipf's law.
        - R gamlss.dist::ZIPF
    """
    return build_ad_family(
        family_class=ZipfFamily,
        name="ZIPF",
        parameters=("mu",),
        log_pdf_func=_zipf_log_pdf,
        type_="Discrete",
        d=_zipf_d,
        p=_zipf_p,
        q=_zipf_q,
        r=_zipf_r,
        links={"mu": "log"},
        link_functions={"mu": log_link},
        link_inverses={"mu": log_inverse},
        link_derivatives={"mu": log_derivative},
    )


# ------------------------------------------------------------------
# Helper function to create zero-altered distributions
# ------------------------------------------------------------------


def create_zero_altered_family(base_family_func, base_name, new_name):
    """Create a zero-altered version of a distribution.

    Zero-altered (hurdle) model:
    - Pr(Y=0) = nu
    - Pr(Y=y|y>0) = (1-nu) * f(y) / [1 - f(0)]

    Args:
        base_family_func: Function that returns base family
        base_name: Name of base distribution
        new_name: Name of zero-altered distribution

    Returns:
        Zero-altered family function

    Example:
        >>> ZALG = create_zero_altered_family(LG, "LG", "ZALG")
    """
    # This is a template - full implementation would require
    # wrapping the base distribution's log_pdf with zero-altered logic
    # Similar to how ZAGA wraps GA
    pass


# ------------------------------------------------------------------
# Helper function to create zero-inflated distributions
# ------------------------------------------------------------------


def create_zero_inflated_family(base_family_func, base_name, new_name):
    """Create a zero-inflated version of a distribution.

    Zero-inflated model:
    - Pr(Y=0) = nu + (1-nu) * f(0)
    - Pr(Y=y|y>0) = (1-nu) * f(y)

    Args:
        base_family_func: Function that returns base family
        base_name: Name of base distribution
        new_name: Name of zero-inflated distribution

    Returns:
        Zero-inflated family function

    Example:
        >>> ZILG = create_zero_inflated_family(LG, "LG", "ZILG")
    """
    raise NotImplementedError("Generic zero-inflated construction is not yet supported")


# ------------------------------------------------------------------
# Zero-Altered Logarithmic (ZALG)
# ------------------------------------------------------------------


@dataclass(frozen=True)
class ZeroAlteredLogarithmicFamily(FamilyDefinition):
    """Zero-Altered Logarithmic distribution (ZALG)."""


def _zalg_log_pdf(y, mu, sigma):
    """Zero-adjusted LG with the same parameter validity as p/q/r."""
    y, mu, sigma = jnp.broadcast_arrays(
        *[jnp.asarray(v, dtype=jnp.float64) for v in (y, mu, sigma)]
    )
    valid = (mu > 0) & (mu < 1) & (sigma > 0) & (sigma < 1)
    safe_sigma = jnp.where(valid, sigma, 0.5)
    # At the atom, avoid evaluating log(0) in an unselected AD branch.
    log_lg = _lg_log_pdf(jnp.where(y == 0, 1.0, y), mu)
    result = jnp.where(y == 0, jnp.log(safe_sigma), jnp.log1p(-safe_sigma) + log_lg)
    return jnp.where(valid & ~jnp.isnan(y), result, jnp.nan)


def _logarithmic_logsf(q, mu, sigma):
    """Scalar log survival via an integral of the infinite logarithmic tail.

    For m=floor(q)+1, sum(mu**k/k, k=m..inf) equals
    mu**m/m * integral(exp(-u)/(1-mu*exp(-u/m)), u=0..inf).
    This avoids allocating support grids and retains underflowed log tails.
    """
    if not (0 < mu < 1 and 0 <= sigma < 1) or np.isnan(q):
        return np.nan
    if q < 0:
        return 0.0
    if np.isposinf(q):
        return -np.inf
    if q < 1:
        return math.log1p(-sigma)
    m = math.floor(q) + 1
    log_mu = math.log(mu)
    integral, _ = quad(
        lambda u: math.exp(-u) / -math.expm1(log_mu - u / m),
        0,
        np.inf,
        epsabs=1e-11,
        epsrel=1e-11,
        limit=200,
    )
    return (
        math.log1p(-sigma)
        + m * log_mu
        - math.log(m)
        + math.log(integral)
        - math.log(-math.log1p(-mu))
    )


def _logarithmic_p(q, mu, sigma, lower_tail=True, log_p=False):
    """CPU CDF/survival with an untruncated logarithmic tail."""
    q, mu, sigma = np.broadcast_arrays(
        *[np.asarray(v, dtype=float) for v in (q, mu, sigma)]
    )
    logs = np.array(
        [_logarithmic_logsf(x, m, s) for x, m, s in zip(q.flat, mu.flat, sigma.flat)]
    ).reshape(q.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = -np.expm1(logs) if lower_tail else np.exp(logs)
        if log_p:
            result = np.log(result) if lower_tail else logs
    return jnp.asarray(result)


def _logarithmic_q(p, mu, sigma, lower_tail=True, log_p=False, max_value=None):
    """CPU inverse CDF with doubling/bisection; unresolved capped searches yield NaN.

    The default searches up to 2**53-1, the exact integer range of float64.
    An explicit max_value must be a nonnegative integer in this range.
    """
    if max_value is not None and (
        isinstance(max_value, (bool, np.bool_))
        or not isinstance(max_value, (int, np.integer))
        or not 0 <= max_value <= 2**53 - 1
    ):
        raise ValueError("max_value must be an integer in [0, 2**53-1] or None")
    cap = 2**53 - 1 if max_value is None else int(max_value)
    p, mu, sigma = np.broadcast_arrays(
        *[np.asarray(v, dtype=float) for v in (p, mu, sigma)]
    )
    out = np.full(p.shape, np.nan)
    for index in np.ndindex(p.shape):
        prob, m, s = p[index], mu[index], sigma[index]
        if not (0 < m < 1 and 0 <= s < 1) or np.isnan(prob):
            continue
        if (log_p and prob > 0) or (not log_p and not 0 <= prob <= 1):
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            target = (
                (np.log(-np.expm1(prob)) if lower_tail else prob)
                if log_p
                else (np.log1p(-prob) if lower_tail else np.log(prob))
            )
        # LG has no atom at zero: positive probabilities below the floating
        # exponential range still map to its first support point, not zero.
        if s == 0 and log_p and lower_tail and np.isfinite(prob) and target == 0:
            out[index] = 1.0 if cap >= 1 else np.nan
            continue
        if target >= math.log1p(-s):
            out[index] = 0
            continue
        if np.isneginf(target):
            out[index] = np.inf
            continue
        lo, hi = 0, min(1, cap)
        while hi < cap and _logarithmic_logsf(hi, m, s) > target:
            lo, hi = hi, min(2 * hi, cap)
        if _logarithmic_logsf(hi, m, s) > target:
            continue
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if _logarithmic_logsf(mid, m, s) <= target:
                hi = mid
            else:
                lo = mid
        out[index] = hi
    return jnp.asarray(out)


def _zalg_d(x, mu=0.5, sigma=0.1, log=False):
    value = _zalg_log_pdf(x, mu, sigma)
    return value if log else jnp.exp(value)


def _zalg_p(q, mu=0.5, sigma=0.1, lower_tail=True, log_p=False):
    sigma = np.asarray(sigma, dtype=float)
    sigma = np.where((sigma > 0) & (sigma < 1), sigma, np.nan)
    return _logarithmic_p(q, mu, sigma, lower_tail, log_p)


def _zalg_q(p, mu=0.5, sigma=0.1, lower_tail=True, log_p=False, max_value=None):
    sigma = np.asarray(sigma, dtype=float)
    sigma = np.where((sigma > 0) & (sigma < 1), sigma, np.nan)
    return _logarithmic_q(p, mu, sigma, lower_tail, log_p, max_value)


def _zalg_r(key, n, mu=0.5, sigma=0.1):
    return _zalg_q(jrandom.uniform(key, shape=(n,), dtype=jnp.float64), mu, sigma)


def ZALG() -> ZeroAlteredLogarithmicFamily:
    """Zero-Altered Logarithmic distribution."""
    return build_ad_family(
        family_class=ZeroAlteredLogarithmicFamily,
        name="ZALG",
        parameters=("mu", "sigma"),
        log_pdf_func=_zalg_log_pdf,
        type_="Discrete",
        d=_zalg_d,
        p=_zalg_p,
        q=_zalg_q,
        r=_zalg_r,
        links={"mu": "logit", "sigma": "logit"},
        link_functions={"mu": logit_link, "sigma": logit_link},
        link_inverses={"mu": logit_inverse, "sigma": logit_inverse},
        link_derivatives={"mu": logit_derivative, "sigma": logit_derivative},
    )


# ------------------------------------------------------------------
# Zero-Inflated Logarithmic (ZILG)
# ------------------------------------------------------------------


@dataclass(frozen=True)
class ZeroInflatedLogarithmicFamily(FamilyDefinition):
    """Zero-inflated logarithmic distribution (R ``gamlss.dist::ZILG``)."""


def _zilg_log_pdf(y, mu, nu):
    """ZILG mass: P(Y=0)=nu and P(Y=y)=(1-nu) dLG(y; mu)."""
    return _zalg_log_pdf(y, mu, nu)


def _zilg_p(q, mu=0.5, nu=0.1, lower_tail=True, log_p=False):
    return _zalg_p(q, mu, nu, lower_tail=lower_tail, log_p=log_p)


def _zilg_q(p, mu=0.5, nu=0.1, lower_tail=True, log_p=False, max_value=10000):
    return _zalg_q(p, mu, nu, lower_tail=lower_tail, log_p=log_p, max_value=max_value)


def _zilg_r(key, n, mu=0.5, nu=0.1):
    return _zalg_r(key, n, mu, nu)


def ZILG() -> ZeroInflatedLogarithmicFamily:
    """Zero-inflated logarithmic distribution with R parameters ``mu, nu``."""
    return build_ad_family(
        family_class=ZeroInflatedLogarithmicFamily,
        name="ZILG",
        parameters=("mu", "nu"),
        log_pdf_func=_zilg_log_pdf,
        type_="Discrete",
        p=_zilg_p,
        q=_zilg_q,
        r=_zilg_r,
        links={"mu": "logit", "nu": "logit"},
        link_functions={"mu": logit_link, "nu": logit_link},
        link_inverses={"mu": logit_inverse, "nu": logit_inverse},
        link_derivatives={"mu": logit_derivative, "nu": logit_derivative},
    )


# ------------------------------------------------------------------
# Additional simple distributions from gamlss.dist
# ------------------------------------------------------------------

# WEI2, WEI3 - Weibull variants
# GEOMo - Geometric original
# NOF, GAF - Normal/Gamma families
# etc.

# These will be added incrementally based on priority
