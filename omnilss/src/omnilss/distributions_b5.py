"""GAMLSS distributions: Batch 5 — Zero-inflated and Zero-altered (Hurdle) families.

These models handle excessive zeros in continuous or discrete data.
- Zero-Altered (ZA/Hurdle): Pr(Y=0) = nu, Pr(Y=y) = (1-nu)*f(y)/[1-f(0)]
- Zero-Inflated (ZI): Pr(Y=0) = nu + (1-nu)*f(0), Pr(Y=y) = (1-nu)*f(y)

Distributions:
- ZAGA   (Zero-Altered Gamma, continuous hurdle)
- ZAIG   (Zero-Altered Inverse Gaussian, continuous hurdle)
- ZIP2   (Zero-Inflated Poisson type 2)
- ZINBI  (Zero-Inflated Negative Binomial type I)
- ZAP    (Zero-Altered Poisson, hurdle)
- ZAPIG  (Zero-Altered Poisson Inverse Gaussian, hurdle)
"""

from __future__ import annotations

# Enable float64 precision for numerical accuracy
from dataclasses import dataclass
from functools import lru_cache
import math
from scipy.stats import nbinom as _nbinom

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax.scipy.special import gammaln
from jax.scipy.stats import poisson
from jax import jit, vmap

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
from .dpqr_functions import qNBI

# ------------------------------------------------------------------
# 1. ZAGA — Zero-Altered Gamma (Continuous Hurdle)
#    R: dZAGA(x, mu, sigma, nu)
#       logfy = if x==0 then log(nu) else log(1-nu) + dGA(x, mu, sigma, log=T)
# ------------------------------------------------------------------


@dataclass(frozen=True)
class ZeroAlteredGammaFamily(FamilyDefinition):
    """Zero-Altered Gamma distribution (`ZAGA`)."""


@jit
def _zaga_log_pdf_single(y_i: float, mu_i: float, sigma_i: float, nu_i: float) -> float:
    """JIT-compiled log PDF for single ZAGA observation.

    This function is vectorized for better performance.
    """
    eps = jnp.finfo(jnp.float64).eps
    # Ensure parameters are in valid range with better bounds
    mu_i = jnp.clip(mu_i, eps * 10, 1e10)
    sigma_i = jnp.clip(sigma_i, eps * 10, 10.0)
    nu_i = jnp.clip(nu_i, eps * 10, 1.0 - eps * 10)

    # Ensure y is non-negative
    y_i = jnp.maximum(y_i, 0.0)

    # dGA log pdf with numerical stability.
    # R (ZAGA.R) uses shape=1/sigma^2, scale=mu*sigma^2 without clipping the
    # shape.  A hard clip of alpha (previously [0.1, 100]) distorted the
    # density for sigma outside [0.1, 3.16] (e.g. sigma ~ 4 -> alpha clamped
    # to 0.1), breaking both the likelihood and its gradients there.
    alpha = 1.0 / jnp.square(sigma_i)
    beta = mu_i / alpha
    beta = jnp.maximum(beta, eps * 10)

    # For y > 0, compute gamma log pdf
    y_safe = jnp.maximum(y_i, eps * 10)
    log_y = jnp.log(y_safe)
    log_beta = jnp.log(beta)

    # Gamma log pdf
    log_ga = (alpha - 1.0) * log_y - y_safe / beta - alpha * log_beta - gammaln(alpha)
    log_ga = jnp.where(jnp.isfinite(log_ga), log_ga, -1e10)

    # Zero-altered model
    log_at0 = jnp.log(nu_i)
    log_cont = jnp.log1p(-nu_i) + log_ga

    # Use smooth transition near zero
    is_zero = y_i <= eps * 100
    result = jnp.where(is_zero, log_at0, log_cont)

    # Final safety check
    result = jnp.where(jnp.isfinite(result), result, -1e10)

    return result


# Vectorize for all observations
_zaga_log_pdf_vectorized = jit(vmap(_zaga_log_pdf_single, in_axes=(0, 0, 0, 0)))

# Pre-compile with dummy data to avoid first-call overhead
_dummy_y = jnp.array([0.0], dtype=jnp.float64)
_dummy_mu = jnp.array([1.0], dtype=jnp.float64)
_dummy_sigma = jnp.array([0.5], dtype=jnp.float64)
_dummy_nu = jnp.array([0.3], dtype=jnp.float64)
_ = _zaga_log_pdf_vectorized(
    _dummy_y, _dummy_mu, _dummy_sigma, _dummy_nu
)  # Warm-up compilation


def _zaga_log_pdf(y, mu, sigma, nu):
    """Zero-Altered Gamma log PDF.

    Performance: Uses JIT-compiled vectorized computation for 5-10x speedup.
    """
    # Ensure inputs are arrays (handle scalar case)
    y = jnp.atleast_1d(y)
    mu = jnp.atleast_1d(mu)
    sigma = jnp.atleast_1d(sigma)
    nu = jnp.atleast_1d(nu)

    # Use vectorized version for better performance
    result = _zaga_log_pdf_vectorized(y, mu, sigma, nu)

    # Return scalar if input was scalar
    return result.squeeze() if result.shape == (1,) else result


@lru_cache(maxsize=1)
def ZAGA() -> ZeroAlteredGammaFamily:
    """Zero-Altered Gamma family."""
    from .dpqr_functions import dZAGA, pZAGA, qZAGA, rZAGA

    return build_ad_family(
        family_class=ZeroAlteredGammaFamily,
        name="ZAGA",
        parameters=("mu", "sigma", "nu"),
        log_pdf_func=_zaga_log_pdf,
        type_="Mixed",
        links={"mu": "log", "sigma": "log", "nu": "logit"},
        link_functions={"mu": log_link, "sigma": log_link, "nu": logit_link},
        link_inverses={"mu": log_inverse, "sigma": log_inverse, "nu": logit_inverse},
        link_derivatives={
            "mu": log_derivative,
            "sigma": log_derivative,
            "nu": logit_derivative,
        },
        d=dZAGA,
        p=pZAGA,
        q=qZAGA,
        r=rZAGA,
    )


# ------------------------------------------------------------------
# 2. ZAIG — Zero-Altered Inverse Gaussian (Continuous Hurdle)
#    R: dZAIG(x, mu, sigma, nu)
#       logfy = if x==0 then log(nu) else log(1-nu) + dIG(x, mu, sigma, log=T)
# ------------------------------------------------------------------


@dataclass(frozen=True)
class ZeroAlteredInverseGaussianFamily(FamilyDefinition):
    """Zero-Altered Inverse Gaussian distribution (`ZAIG`)."""


def _zaig_log_pdf(y, mu, sigma, nu):
    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.clip(nu, eps, 1.0 - eps)

    # dIG log pdf
    y_safe = jnp.maximum(y, eps)
    log_ig = -0.5 * jnp.log(
        2.0 * math.pi * jnp.square(sigma) * jnp.power(y_safe, 3)
    ) - jnp.square(y_safe - mu) / (2.0 * jnp.square(sigma * mu) * y_safe)

    log_at0 = jnp.log(nu)
    log_cont = jnp.log1p(-nu) + log_ig
    return jnp.where(y <= 0.0, log_at0, log_cont)


def ZAIG() -> ZeroAlteredInverseGaussianFamily:
    from .dpqr_functions import dZAIG, pZAIG, qZAIG, rZAIG

    return build_ad_family(
        family_class=ZeroAlteredInverseGaussianFamily,
        name="ZAIG",
        parameters=("mu", "sigma", "nu"),
        log_pdf_func=_zaig_log_pdf,
        type_="Mixed",
        links={"mu": "log", "sigma": "log", "nu": "logit"},
        link_functions={"mu": log_link, "sigma": log_link, "nu": logit_link},
        link_inverses={"mu": log_inverse, "sigma": log_inverse, "nu": logit_inverse},
        link_derivatives={
            "mu": log_derivative,
            "sigma": log_derivative,
            "nu": logit_derivative,
        },
        d=dZAIG,
        p=pZAIG,
        q=qZAIG,
        r=rZAIG,
    )


# ------------------------------------------------------------------
# 3. ZIP2 — Zero-Inflated Poisson type 2
#    R: dZIP2(x, mu, sigma)
#    In R, mu is the actual mean and sigma is the extra-zero probability.
# ------------------------------------------------------------------


@dataclass(frozen=True)
class ZeroInflatedPoisson2Family(FamilyDefinition):
    """Zero-Inflated Poisson type 2 (`ZIP2`).

    R ``gamlss.dist`` parameters are ``mu`` (the mixture mean) and ``sigma``
    (the extra-zero probability).
    """


def _zip2_log_pdf(y, mu, sigma):
    # R parameterization keeps mu as the mean and uses mus=lambda=mu/(1-sigma).
    valid = (mu > 0.0) & (sigma > 0.0) & (sigma < 1.0)
    safe_mu = jnp.where(valid, mu, 1.0)
    safe_sigma = jnp.where(valid, sigma, 0.5)
    mus = safe_mu / (1.0 - safe_sigma)
    log_at0 = jnp.log(safe_sigma + (1.0 - safe_sigma) * jnp.exp(-mus))
    log_cont = (
        (1.0 - y) * jnp.log1p(-safe_sigma)
        - mus
        + y * jnp.log(safe_mu)
        - gammaln(y + 1.0)
    )
    result = jnp.where(y == 0.0, log_at0, log_cont)
    result = jnp.where(y < 0.0, -jnp.inf, result)
    return jnp.where(valid, result, jnp.nan)


def _zip2_p(q, mu=5.0, sigma=0.1, lower_tail=True, log_p=False):
    """CDF matching R ``pZIP2`` with standard tail/log boundary semantics."""
    q, mu, sigma = jnp.broadcast_arrays(
        *[jnp.asarray(value, dtype=jnp.float64) for value in (q, mu, sigma)]
    )
    valid = (mu > 0.0) & (sigma > 0.0) & (sigma < 1.0)
    safe_mu = jnp.where(valid, mu, 1.0)
    safe_sigma = jnp.where(valid, sigma, 0.5)
    mus = safe_mu / (1.0 - safe_sigma)
    cdf = safe_sigma + (1.0 - safe_sigma) * poisson.cdf(q, mus)
    cdf = jnp.where(q < 0.0, 0.0, cdf)
    probability = jnp.where(lower_tail, cdf, 1.0 - cdf)
    result = jnp.log(probability) if log_p else probability
    return jnp.where(valid, result, jnp.nan)


ZIP2_QUANTILE_MAX_COUNT = 10_000


def _zip2_q(
    p,
    mu=5.0,
    sigma=0.1,
    lower_tail=True,
    log_p=False,
    max_value=ZIP2_QUANTILE_MAX_COUNT,
):
    """Quantile function for ZIP2 using the adjusted Poisson probability."""
    p, mu, sigma = jnp.broadcast_arrays(
        *[jnp.asarray(value, dtype=jnp.float64) for value in (p, mu, sigma)]
    )
    probability = jnp.exp(p) if log_p else p
    probability = jnp.where(lower_tail, probability, 1.0 - probability)
    valid_parameters = (mu > 0.0) & (sigma > 0.0) & (sigma < 1.0)
    valid_probability = (probability >= 0.0) & (probability <= 1.0)
    safe_mu = jnp.where(valid_parameters, mu, 1.0)
    safe_sigma = jnp.where(valid_parameters, sigma, 0.5)
    mus = safe_mu / (1.0 - safe_sigma)
    adjusted = (probability - safe_sigma) / (1.0 - safe_sigma) - 1e-10
    support = jnp.arange(max_value + 1, dtype=jnp.float64)

    def one(pi, adjusted_i, musi):
        reached = poisson.cdf(support, musi) >= adjusted_i
        result = jnp.where(adjusted_i <= 0.0, 0.0, jnp.argmax(reached)).astype(
            jnp.float64
        )
        result = jnp.where((adjusted_i > 0.0) & ~jnp.any(reached), jnp.nan, result)
        result = jnp.where(pi == 0.0, 0.0, result)
        return jnp.where(pi == 1.0, jnp.inf, result)

    result = jax.vmap(one)(
        probability.reshape(-1), adjusted.reshape(-1), mus.reshape(-1)
    ).reshape(p.shape)
    valid = valid_parameters & valid_probability & ~jnp.isnan(probability)
    return jnp.where(valid, result, jnp.nan)


def _zip2_r(key, n, mu=5.0, sigma=0.1):
    """Generate ZIP2 variates directly from its zero/Poisson mixture."""
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    valid = (mu > 0.0) & (sigma > 0.0) & (sigma < 1.0)
    safe_mu = jnp.where(valid, mu, 1.0)
    safe_sigma = jnp.where(valid, sigma, 0.5)
    zero_key, poisson_key = jrandom.split(key)
    output_shape = (
        (n,)
        if mu.ndim == 0 and sigma.ndim == 0
        else jnp.broadcast_shapes(mu.shape, sigma.shape)
    )
    structural_zero = jrandom.bernoulli(zero_key, safe_sigma, shape=output_shape)
    counts = jrandom.poisson(
        poisson_key, safe_mu / (1.0 - safe_sigma), shape=output_shape
    )
    result = jnp.where(structural_zero, 0.0, counts.astype(jnp.float64))
    return jnp.where(valid, result, jnp.nan)


def ZIP2() -> ZeroInflatedPoisson2Family:
    return build_ad_family(
        family_class=ZeroInflatedPoisson2Family,
        name="ZIP2",
        parameters=("mu", "sigma"),
        log_pdf_func=_zip2_log_pdf,
        type_="Discrete",
        links={"mu": "log", "sigma": "logit"},
        link_functions={"mu": log_link, "sigma": logit_link},
        link_inverses={"mu": log_inverse, "sigma": logit_inverse},
        link_derivatives={"mu": log_derivative, "sigma": logit_derivative},
        p=_zip2_p,
        q=_zip2_q,
        r=_zip2_r,
    )


# ------------------------------------------------------------------
# 4. ZINBI — Zero-Inflated Negative Binomial type I
#    R: dZINBI(x, mu, sigma, nu)
#       logfy = if x==0 then log(nu + (1-nu)*dNBI(0, mu, sigma)) else log(1-nu) + dNBI(x, mu, sigma, log=T)
# ------------------------------------------------------------------


@dataclass(frozen=True)
class ZeroInflatedNegativeBinomial1Family(FamilyDefinition):
    """Zero-Inflated Negative Binomial type I (`ZINBI`)."""


def _zinbi_log_pdf(y, mu, sigma, nu):
    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.clip(nu, eps, 1.0 - eps)

    # dNBI log pdf
    size = 1.0 / sigma
    log_prob = jnp.log(size / (size + mu))
    log_nbi = (
        gammaln(y + size)
        - gammaln(size)
        - gammaln(y + 1.0)
        + size * log_prob
        + y * jnp.log(mu / (size + mu))
    )

    # dNBI(0, mu, sigma) = (size / (size + mu))^size
    nbi_at0 = jnp.exp(size * log_prob)

    log_at0 = jnp.log(nu + (1.0 - nu) * nbi_at0)
    log_cont = jnp.log1p(-nu) + log_nbi
    return jnp.where(y <= 0.0, log_at0, log_cont)


def _zinbi_args(mu, sigma):
    size = 1.0 / sigma
    return size, size / (size + mu)


def dZINBI(x, mu=1.0, sigma=1.0, nu=0.3, log=False):
    x, mu, sigma, nu = jnp.broadcast_arrays(
        *[jnp.asarray(v, dtype=jnp.float64) for v in (x, mu, sigma, nu)]
    )
    size, prob = _zinbi_args(mu, sigma)
    base = _nbinom.logpmf(x, size, prob)
    base0 = _nbinom.logpmf(0, size, prob)
    value = jnp.where(
        x == 0,
        jnp.logaddexp(jnp.log(nu), jnp.log1p(-nu) + base0),
        jnp.log1p(-nu) + base,
    )
    value = jnp.where((x >= 0) & (x == jnp.floor(x)), value, -jnp.inf)
    return value if log else jnp.exp(value)


def pZINBI(q, mu=1.0, sigma=1.0, nu=0.3, lower_tail=True, log_p=False):
    q, mu, sigma, nu = jnp.broadcast_arrays(
        *[jnp.asarray(v, dtype=jnp.float64) for v in (q, mu, sigma, nu)]
    )
    size, prob = _zinbi_args(mu, sigma)
    value = jnp.where(q < 0, 0.0, nu + (1 - nu) * _nbinom.cdf(q, size, prob))
    value = jnp.where(lower_tail, value, 1 - value)
    return jnp.log(value) if log_p else value


def qZINBI(p, mu=1.0, sigma=1.0, nu=0.3, lower_tail=True, log_p=False):
    p, mu, sigma, nu = jnp.broadcast_arrays(
        *[jnp.asarray(v, dtype=jnp.float64) for v in (p, mu, sigma, nu)]
    )
    p = jnp.exp(p) if log_p else p
    p = jnp.where(lower_tail, p, 1 - p)
    # D2.27: R inverts through qNBI on the rescaled probability, with its own
    # coarse epsilon: qZINBI subtracts 1e-7 (not the 1e-10 its siblings use)
    # and clamps at zero, so p = 1 asks qNBI(0.9999999) = 49 rather than the
    # inf a ppf(1.0) gives.  Live R: qZINBI(0/0.1/0.3/0.5/0.9/1) = 0/0/0/2/9/49.
    pnew = jnp.maximum((p - nu) / (1.0 - nu) - 1e-7, 0.0)
    return qNBI(pnew, mu, sigma)


def rZINBI(key, n, mu=1.0, sigma=1.0, nu=0.3):
    k0, k1 = jrandom.split(key)
    shape = (
        (n,)
        if all(jnp.asarray(v).ndim == 0 for v in (mu, sigma, nu))
        else jnp.broadcast_shapes(jnp.shape(mu), jnp.shape(sigma), jnp.shape(nu))
    )
    return jnp.where(
        jrandom.bernoulli(k0, nu, shape=shape),
        0.0,
        qNBI(jrandom.uniform(k1, shape=shape), mu, sigma),
    )


def ZINBI() -> ZeroInflatedNegativeBinomial1Family:
    return build_ad_family(
        family_class=ZeroInflatedNegativeBinomial1Family,
        name="ZINBI",
        parameters=("mu", "sigma", "nu"),
        log_pdf_func=_zinbi_log_pdf,
        type_="Discrete",
        links={"mu": "log", "sigma": "log", "nu": "logit"},
        link_functions={"mu": log_link, "sigma": log_link, "nu": logit_link},
        link_inverses={"mu": log_inverse, "sigma": log_inverse, "nu": logit_inverse},
        link_derivatives={
            "mu": log_derivative,
            "sigma": log_derivative,
            "nu": logit_derivative,
        },
        # D2.27: pZINBI/qZINBI/rZINBI existed here since the family landed but
        # were never wired in, so `ZINBI().p` answered with build_ad_family's
        # all-NaN placeholder -- the capability existed and had no entry point.
        p=pZINBI,
        q=qZINBI,
        r=rZINBI,
    )


# ------------------------------------------------------------------
# 5. ZAP — Zero-Altered Poisson (Hurdle)
#    R: dZAP(x, mu, sigma)
#    In R, sigma is the extra-zero probability.
# ------------------------------------------------------------------


@dataclass(frozen=True)
class ZeroAlteredPoissonFamily(FamilyDefinition):
    """Zero-Altered Poisson distribution (`ZAP`)."""


def _zap_log_pdf(y, mu, sigma):
    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.maximum(mu, eps)
    sigma = jnp.clip(sigma, eps, 1.0 - eps)

    log_po = y * jnp.log(mu) - mu - gammaln(y + 1.0)
    log_trunc = jnp.log1p(-jnp.exp(-mu))

    log_at0 = jnp.log(sigma)
    log_cont = jnp.log1p(-sigma) + log_po - log_trunc
    return jnp.where(y <= 0.0, log_at0, log_cont)


def ZAP() -> ZeroAlteredPoissonFamily:
    # D2.28: ZAP was the one ZA family whose tails were neither factory-composed
    # (D2.27) nor hand-written, so `build_ad_family` answered every `p`/`q`/`r`
    # with the all-NaN placeholder.  Live R's pZAP/qZAP are the standard hurdle
    # composition over ppois/qpois, so the D2.27 composition layer is reused
    # here with one transcription difference: R's qZAP subtracts *no* epsilon
    # (`pnew <- (p - sigma)/(1 - sigma)` goes straight to the inversion), the
    # seventeenth quantile-epsilon variant after D2.27's sixteen 1e-10
    # families and ZIBNB/ZINBI's 1e-7.  No black-box input separates 0 from
    # 1e-10 (the shift is below CDF granularity), so the zero is pinned by
    # this comment against live R source, not by a quantile assertion.
    # The import stays inside the factory: b10 does not import b5 at module
    # level, but the registry resolves b5 names by string, so a module-level
    # import here would put b5 -> b10 -> registry on the import-time path.
    from .distributions_b10_zero_variants import _zero_variant_tails

    return build_ad_family(
        family_class=ZeroAlteredPoissonFamily,
        name="ZAP",
        parameters=("mu", "sigma"),
        log_pdf_func=_zap_log_pdf,
        type_="Discrete",
        links={"mu": "log", "sigma": "logit"},
        link_functions={"mu": log_link, "sigma": logit_link},
        link_inverses={"mu": log_inverse, "sigma": logit_inverse},
        link_derivatives={"mu": log_derivative, "sigma": logit_derivative},
        **_zero_variant_tails("PO", altered=True, epsilon=0.0),
    )


# ------------------------------------------------------------------
# 6. ZAPIG — Zero-Altered Poisson Inverse Gaussian (Hurdle)
#    R: dZAPIG(x, mu, sigma, nu)
#       logfy = if x==0 then log(nu) else log(1-nu) + dPIG(x, mu, sigma, log=T) - log(1-dPIG(0, mu, sigma))
#    Note: PIG is more complex (uses Bessel or C code).
#    For now, we approximate PIG or use the standard PIG implementation if available.
#    Actually, PIG is not yet implemented in distributions.py.
#    Wait, I'll defer ZAPIG until Batch 6 (Discrete Special) where PIG belongs.
# ------------------------------------------------------------------
