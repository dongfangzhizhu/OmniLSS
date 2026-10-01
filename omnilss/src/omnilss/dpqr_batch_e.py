"""D1-E batch kernels: real d/p/q/r for families that were registered as placeholders.

Each kernel mirrors the corresponding ``gamlss.dist`` definition (vendored under
``R/gamlss.dist/R``) so the Python side can be compared point-by-point against R.
Conventions follow the rest of the package:

``d(x, mu, ..., log=False)``
``p(q, mu, ..., lower_tail=True, log_p=False)``
``q(p, mu, ..., lower_tail=True, log_p=False)``
``r(key, n, mu, ...)``

Where R's own ``gamlss.dist`` implementation contains an outright bug (for
example ``pPARETO`` silently ignores ``log.p`` because of the R typo
``cdf < - log(cdf)``, and ``qPARETO`` never applies ``lower.tail``) this module
implements the mathematically correct branch instead of reproducing the bug.
Every such deviation is listed in ``docs/development/task-reports/`` and asserted
explicitly in the parity tests, so the difference is documented rather than
silently absorbed.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpy as np

#: Support endpoints used when a probability lands exactly on 0 or 1.
_INF = jnp.inf
_NAN = jnp.nan


def _safe_log(value):
    """``log`` that maps 0 to ``-inf`` instead of ``NaN``."""
    positive = value > 0
    return jnp.where(positive, jnp.log(jnp.where(positive, value, 1.0)), -_INF)


def _apply_tail(value, lower_tail: bool, log_p: bool):
    """Apply ``lower_tail`` / ``log_p`` in a numerically safe way."""
    result = value if lower_tail else 1.0 - value
    if log_p:
        return _safe_log(result)
    return result


def _prob_from_p(p, lower_tail: bool, log_p: bool):
    """Normalise a quantile-function probability onto the lower tail."""
    prob = jnp.exp(p) if log_p else jnp.asarray(p, dtype=jnp.float64)
    return prob if lower_tail else 1.0 - prob


def _quantile_from_prob(prob, value_fn, low, high):
    """Evaluate ``value_fn`` and clamp the degenerate probabilities."""
    finite = (prob > 0.0) & (prob < 1.0)
    safe = jnp.where(finite, prob, 0.5)
    value = value_fn(safe)
    value = jnp.where(prob <= 0.0, low, value)
    value = jnp.where(prob >= 1.0, high, value)
    return jnp.where(jnp.isnan(prob), _NAN, value)


# ---------------------------------------------------------------------------
# PARETO - one-parameter Pareto, y > 1, mu > 0
# ---------------------------------------------------------------------------


def dPARETO(x, mu=1.0, log=False):
    """Density of the one-parameter Pareto distribution (support ``y > 1``)."""
    x, mu = jnp.broadcast_arrays(jnp.asarray(x, dtype=jnp.float64), jnp.asarray(mu, dtype=jnp.float64))
    inside = (x > 1.0) & (mu > 0.0)
    xs = jnp.where(inside, x, 2.0)
    value = jnp.where(inside, jnp.log(mu) - (mu + 1.0) * jnp.log(xs), -_INF)
    value = jnp.where(mu > 0.0, value, _NAN)
    return value if log else jnp.exp(value)


def pPARETO(q, mu=1.0, lower_tail=True, log_p=False):
    """CDF of the one-parameter Pareto distribution (support ``y > 1``)."""
    q, mu = jnp.broadcast_arrays(jnp.asarray(q, dtype=jnp.float64), jnp.asarray(mu, dtype=jnp.float64))
    inside = q > 1.0
    qs = jnp.where(inside, q, 2.0)
    cdf = jnp.where(inside, -jnp.expm1(-mu * jnp.log(qs)), 0.0)
    value = _apply_tail(cdf, lower_tail, log_p)
    return jnp.where(mu > 0.0, value, _NAN)


def qPARETO(p, mu=1.0, lower_tail=True, log_p=False):
    """Quantile of the one-parameter Pareto distribution (support ``y > 1``)."""
    p, mu = jnp.broadcast_arrays(jnp.asarray(p, dtype=jnp.float64), jnp.asarray(mu, dtype=jnp.float64))
    prob = _prob_from_p(p, lower_tail, log_p)

    def value_fn(pr):
        return jnp.exp(-jnp.log1p(-pr) / mu)

    value = _quantile_from_prob(prob, value_fn, low=1.0, high=_INF)
    return jnp.where(mu > 0.0, value, _NAN)


def rPARETO(key, n, mu=1.0):
    """Inverse-transform sampler for the one-parameter Pareto distribution."""
    return qPARETO(jr.uniform(key, (int(n),), dtype=jnp.float64), mu)


# ---------------------------------------------------------------------------
# PARETO1 - one-parameter Pareto, y > 0, mu > 0
# ---------------------------------------------------------------------------


def dPARETO1(x, mu=1.0, log=False):
    """Density of the one-parameter Pareto distribution (support ``y > 0``)."""
    x, mu = jnp.broadcast_arrays(jnp.asarray(x, dtype=jnp.float64), jnp.asarray(mu, dtype=jnp.float64))
    inside = (x > 0.0) & (mu > 0.0)
    xs = jnp.where(inside, x, 1.0)
    value = jnp.where(inside, jnp.log(mu) - (mu + 1.0) * jnp.log1p(xs), -_INF)
    value = jnp.where(mu > 0.0, value, _NAN)
    return value if log else jnp.exp(value)


def pPARETO1(q, mu=1.0, lower_tail=True, log_p=False):
    """CDF of the one-parameter Pareto distribution (support ``y > 0``)."""
    q, mu = jnp.broadcast_arrays(jnp.asarray(q, dtype=jnp.float64), jnp.asarray(mu, dtype=jnp.float64))
    inside = q > 0.0
    qs = jnp.where(inside, q, 1.0)
    cdf = jnp.where(inside, -jnp.expm1(-mu * jnp.log1p(qs)), 0.0)
    value = _apply_tail(cdf, lower_tail, log_p)
    return jnp.where(mu > 0.0, value, _NAN)


def qPARETO1(p, mu=1.0, lower_tail=True, log_p=False):
    """Quantile of the one-parameter Pareto distribution (support ``y > 0``)."""
    p, mu = jnp.broadcast_arrays(jnp.asarray(p, dtype=jnp.float64), jnp.asarray(mu, dtype=jnp.float64))
    prob = _prob_from_p(p, lower_tail, log_p)

    def value_fn(pr):
        return jnp.expm1(-jnp.log1p(-pr) / mu)

    value = _quantile_from_prob(prob, value_fn, low=0.0, high=_INF)
    return jnp.where(mu > 0.0, value, _NAN)


def rPARETO1(key, n, mu=1.0):
    """Inverse-transform sampler for the one-parameter Pareto distribution."""
    return qPARETO1(jr.uniform(key, (int(n),), dtype=jnp.float64), mu)


# ---------------------------------------------------------------------------
# WEI2 - Weibull type 2.  R: dweibull(x, scale = mu^(-1/sigma), shape = sigma)
# ---------------------------------------------------------------------------


def _wei2_scale(mu, sigma):
    """R's ``muinv``: the Weibull scale implied by gamlss' (mu, sigma)."""
    return jnp.exp(-jnp.log(mu) / sigma)


def _wei2_log_pdf(x, mu, sigma):
    scale = _wei2_scale(mu, sigma)
    xs = jnp.where(x > 0.0, x, 1.0)
    inside = (x > 0.0) & (mu > 0.0) & (sigma > 0.0)
    value = (
        jnp.log(sigma)
        - jnp.log(scale)
        + (sigma - 1.0) * (jnp.log(xs) - jnp.log(scale))
        - jnp.power(xs / scale, sigma)
    )
    value = jnp.where(inside, value, -_INF)
    return jnp.where((mu > 0.0) & (sigma > 0.0), value, _NAN)


def dWEI2(x, mu=1.0, sigma=1.0, log=False):
    """Density of the Weibull type 2 distribution."""
    x, mu, sigma = jnp.broadcast_arrays(
        jnp.asarray(x, dtype=jnp.float64),
        jnp.asarray(mu, dtype=jnp.float64),
        jnp.asarray(sigma, dtype=jnp.float64),
    )
    value = _wei2_log_pdf(x, mu, sigma)
    return value if log else jnp.exp(value)


def pWEI2(q, mu=1.0, sigma=1.0, lower_tail=True, log_p=False):
    """CDF of the Weibull type 2 distribution."""
    q, mu, sigma = jnp.broadcast_arrays(
        jnp.asarray(q, dtype=jnp.float64),
        jnp.asarray(mu, dtype=jnp.float64),
        jnp.asarray(sigma, dtype=jnp.float64),
    )
    scale = _wei2_scale(mu, sigma)
    qs = jnp.where(q > 0.0, q, 1.0)
    cdf = jnp.where(q > 0.0, -jnp.expm1(-jnp.power(qs / scale, sigma)), 0.0)
    value = _apply_tail(cdf, lower_tail, log_p)
    return jnp.where((mu > 0.0) & (sigma > 0.0), value, _NAN)


def qWEI2(p, mu=1.0, sigma=1.0, lower_tail=True, log_p=False):
    """Quantile of the Weibull type 2 distribution."""
    p, mu, sigma = jnp.broadcast_arrays(
        jnp.asarray(p, dtype=jnp.float64),
        jnp.asarray(mu, dtype=jnp.float64),
        jnp.asarray(sigma, dtype=jnp.float64),
    )
    scale = _wei2_scale(mu, sigma)
    prob = _prob_from_p(p, lower_tail, log_p)

    def value_fn(pr):
        return scale * jnp.power(-jnp.log1p(-pr), 1.0 / sigma)

    value = _quantile_from_prob(prob, value_fn, low=0.0, high=_INF)
    return jnp.where((mu > 0.0) & (sigma > 0.0), value, _NAN)


def rWEI2(key, n, mu=1.0, sigma=1.0):
    """Inverse-transform sampler for the Weibull type 2 distribution."""
    return qWEI2(jr.uniform(key, (int(n),), dtype=jnp.float64), mu, sigma)


# ---------------------------------------------------------------------------
# RGE - Reverse Generalized Extreme, y > mu - sigma/nu
# ---------------------------------------------------------------------------

#: R switches to the reverse-Gumbel limit below this |nu| threshold.
_RGE_NU_LIMIT = 1e-3


def _rge_scaled(x, mu, sigma, nu):
    return nu * (x - mu) / sigma


def _rge_log_pdf(x, mu, sigma, nu):
    y = _rge_scaled(x, mu, sigma, nu)
    one_plus = 1.0 + y
    log_one_plus = jnp.where(one_plus > 0.0, jnp.log(jnp.where(one_plus > 0.0, one_plus, 1.0)), 0.0)
    general = -jnp.log(sigma) + (1.0 / nu - 1.0) * log_one_plus - jnp.power(
        jnp.where(one_plus > 0.0, one_plus, 1.0), 1.0 / nu
    )
    z = (x - mu) / sigma
    limit = -jnp.log(sigma) + z - jnp.exp(z)
    value = jnp.where(jnp.abs(nu) < _RGE_NU_LIMIT, limit, general)
    value = jnp.where(one_plus > 0.0, value, -_INF)
    return jnp.where((sigma > 0.0) & (nu != 0.0), value, _NAN)


def dRGE(x, mu=1.0, sigma=0.1, nu=1.0, log=False):
    """Density of the Reverse Generalized Extreme distribution."""
    x, mu, sigma, nu = jnp.broadcast_arrays(
        jnp.asarray(x, dtype=jnp.float64),
        jnp.asarray(mu, dtype=jnp.float64),
        jnp.asarray(sigma, dtype=jnp.float64),
        jnp.asarray(nu, dtype=jnp.float64),
    )
    value = _rge_log_pdf(x, mu, sigma, nu)
    return value if log else jnp.exp(value)


def pRGE(q, mu=1.0, sigma=0.1, nu=1.0, lower_tail=True, log_p=False):
    """CDF of the Reverse Generalized Extreme distribution."""
    q, mu, sigma, nu = jnp.broadcast_arrays(
        jnp.asarray(q, dtype=jnp.float64),
        jnp.asarray(mu, dtype=jnp.float64),
        jnp.asarray(sigma, dtype=jnp.float64),
        jnp.asarray(nu, dtype=jnp.float64),
    )
    y = _rge_scaled(q, mu, sigma, nu)
    one_plus = 1.0 + y
    safe = jnp.where(one_plus > 0.0, one_plus, 1.0)
    general = -jnp.expm1(-jnp.power(safe, 1.0 / nu))
    z = (q - mu) / sigma
    limit = -jnp.expm1(-jnp.exp(z))
    cdf = jnp.where(jnp.abs(nu) < _RGE_NU_LIMIT, limit, general)
    cdf = jnp.where(one_plus > 0.0, cdf, 0.0)
    value = _apply_tail(cdf, lower_tail, log_p)
    return jnp.where((sigma > 0.0) & (nu != 0.0), value, _NAN)


def qRGE(p, mu=1.0, sigma=0.1, nu=1.0, lower_tail=True, log_p=False):
    """Quantile of the Reverse Generalized Extreme distribution."""
    p, mu, sigma, nu = jnp.broadcast_arrays(
        jnp.asarray(p, dtype=jnp.float64),
        jnp.asarray(mu, dtype=jnp.float64),
        jnp.asarray(sigma, dtype=jnp.float64),
        jnp.asarray(nu, dtype=jnp.float64),
    )
    prob = _prob_from_p(p, lower_tail, log_p)
    # Lower endpoint of the support: mu - sigma/nu (nu > 0); R returns 0 here,
    # which is not the support boundary - see the module docstring.
    low = mu - sigma / nu

    def value_fn(pr):
        t = -jnp.log1p(-pr)
        general = mu + (sigma / nu) * (jnp.power(t, nu) - 1.0)
        limit = mu + sigma * jnp.log(t)
        return jnp.where(jnp.abs(nu) < _RGE_NU_LIMIT, limit, general)

    value = _quantile_from_prob(prob, value_fn, low=low, high=_INF)
    return jnp.where((sigma > 0.0) & (nu != 0.0), value, _NAN)


def rRGE(key, n, mu=1.0, sigma=0.1, nu=1.0):
    """Inverse-transform sampler for the Reverse Generalized Extreme distribution."""
    return qRGE(jr.uniform(key, (int(n),), dtype=jnp.float64), mu, sigma, nu)


# ---------------------------------------------------------------------------
# NO2 - normal parameterised by the variance (R: dNO2 uses sd = sqrt(sigma))
# ---------------------------------------------------------------------------


def _norm_cdf(z):
    from jax.scipy.special import ndtr

    return ndtr(z)


def _norm_quantile(pr):
    from jax.scipy.special import ndtri

    return ndtri(pr)


def dNO2(x, mu=0.0, sigma=1.0, log=False):
    """Density of the normal distribution parameterised by the variance ``sigma``."""
    x, mu, sigma = jnp.broadcast_arrays(
        jnp.asarray(x, dtype=jnp.float64),
        jnp.asarray(mu, dtype=jnp.float64),
        jnp.asarray(sigma, dtype=jnp.float64),
    )
    z = (x - mu) / jnp.sqrt(jnp.where(sigma > 0.0, sigma, 1.0))
    value = -0.5 * jnp.log(2.0 * np.pi) - 0.5 * jnp.log(sigma) - 0.5 * z * z
    value = jnp.where(sigma > 0.0, value, _NAN)
    return value if log else jnp.exp(value)


def pNO2(q, mu=0.0, sigma=1.0, lower_tail=True, log_p=False):
    """CDF of the normal distribution parameterised by the variance ``sigma``."""
    q, mu, sigma = jnp.broadcast_arrays(
        jnp.asarray(q, dtype=jnp.float64),
        jnp.asarray(mu, dtype=jnp.float64),
        jnp.asarray(sigma, dtype=jnp.float64),
    )
    sd = jnp.sqrt(jnp.where(sigma > 0.0, sigma, 1.0))
    value = _apply_tail(_norm_cdf((q - mu) / sd), lower_tail, log_p)
    return jnp.where(sigma > 0.0, value, _NAN)


def qNO2(p, mu=0.0, sigma=1.0, lower_tail=True, log_p=False):
    """Quantile of the normal distribution parameterised by the variance ``sigma``."""
    p, mu, sigma = jnp.broadcast_arrays(
        jnp.asarray(p, dtype=jnp.float64),
        jnp.asarray(mu, dtype=jnp.float64),
        jnp.asarray(sigma, dtype=jnp.float64),
    )
    sd = jnp.sqrt(jnp.where(sigma > 0.0, sigma, 1.0))
    prob = _prob_from_p(p, lower_tail, log_p)

    def value_fn(pr):
        return mu + sd * _norm_quantile(pr)

    value = _quantile_from_prob(prob, value_fn, low=-_INF, high=_INF)
    return jnp.where(sigma > 0.0, value, _NAN)


def rNO2(key, n, mu=0.0, sigma=1.0):
    """Sampler for the normal distribution parameterised by the variance ``sigma``."""
    return qNO2(jr.uniform(key, (int(n),), dtype=jnp.float64), mu, sigma)
