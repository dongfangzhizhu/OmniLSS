"""D1-E batch 3 kernels: ``LQNO`` plus the categorical ``MN3``/``MN4``/``MN5``.

Conventions follow ``dpqr_batch_e``:

``d(x, mu, ..., log=False)``
``p(q, mu, ..., lower_tail=True, log_p=False)``
``q(p, mu, ..., lower_tail=True, log_p=False)``
``r(key, n, mu, ...)``

Two things are worth calling out because they are easy to get wrong:

``LQNO``
    R's ``LQNO`` is a **two**-parameter family: ``dnorm(x, mu, sqrt(mu * (1 + sigma * mu)))``
    with ``mu.link = sigma.link = "log"`` (see ``R/gamlss.dist/R/LQNO.R`` and the installed
    ``gamlss.dist`` 6.1.1).  The Python side previously shipped a three-parameter
    ``tanh``-transformed "log quantile normal" that has nothing to do with it; that
    parameterisation is gone and the R definition is implemented verbatim.

``MN3`` / ``MN4`` / ``MN5``
    Categorical families over ``1..k`` with ``k-1`` positive parameters and a
    reference category ``k`` whose weight is fixed at 1.  R's quantile functions
    (``qMN3`` etc.) ignore both ``lower.tail`` and ``log.p`` entirely; we honour
    them, which is the same documented-deviation policy used by ``dpqr_batch_e``.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax.scipy.special import gammaln, ndtr, ndtri
from scipy.stats import f as _scipy_f

_INF = jnp.inf
_NAN = jnp.nan

# Inventory of R ``gamlss.dist`` quirks we deliberately do **not** reproduce.
# ``tests/test_dpqr_batch_f.py::test_r_quirk_inventory_is_explicit`` asserts that
# every family listed here is part of the parity sweep.
R_QUIRKS: dict[str, str] = {
    "MN3": "qMN3 完全忽略 lower.tail 与 log.p",
    "MN4": "qMN4 完全忽略 lower.tail 与 log.p",
    "MN5": "qMN5 完全忽略 lower.tail 与 log.p",
}


def _prob_from_p(p, lower_tail: bool, log_p: bool):
    """Normalise a quantile-function probability onto the lower tail."""
    prob = jnp.exp(p) if log_p else jnp.asarray(p, dtype=jnp.float64)
    return prob if lower_tail else 1.0 - prob


def _safe_log(value):
    """``log`` that maps 0 to ``-inf`` instead of ``NaN``."""
    positive = value > 0.0
    return jnp.where(positive, jnp.log(jnp.where(positive, value, 1.0)), -_INF)


def _apply_tail(value, lower_tail: bool, log_p: bool):
    """Apply ``lower_tail`` / ``log_p`` in a numerically safe way."""
    result = value if lower_tail else 1.0 - value
    if log_p:
        positive = result > 0.0
        result = jnp.where(positive, jnp.log(jnp.where(positive, result, 1.0)), -_INF)
    return result


def _broadcast(values):
    """Broadcast every argument to a common shape as float64."""
    return jnp.broadcast_arrays(*[jnp.asarray(v, dtype=jnp.float64) for v in values])


def _all_positive(*values):
    """``True`` where every parameter satisfies R's strict positivity check."""
    mask = jnp.ones_like(values[0], dtype=bool)
    for value in values:
        mask = mask & (value > 0.0)
    return mask


# ---------------------------------------------------------------------------
# LQNO - Normal(mu, sqrt(mu * (1 + sigma * mu))), mu > 0, sigma > 0
# ---------------------------------------------------------------------------


def _lqno_scale(mu, sigma):
    """Standard deviation ``sqrt(mu * (1 + sigma * mu))``."""
    return jnp.sqrt(mu * (1.0 + sigma * mu))


def dLQNO(x, mu=1.0, sigma=1.0, log=False):
    """Density of ``LQNO``: a Normal with a mean-dependent variance."""
    x, mu, sigma = _broadcast((x, mu, sigma))
    valid = _all_positive(mu, sigma)
    safe_mu = jnp.where(mu > 0.0, mu, 1.0)
    safe_sigma = jnp.where(sigma > 0.0, sigma, 1.0)
    scale = _lqno_scale(safe_mu, safe_sigma)
    z = (x - safe_mu) / scale
    value = -0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(scale) - 0.5 * z * z
    value = jnp.where(valid, value, _NAN)
    return value if log else jnp.exp(value)


def pLQNO(q, mu=1.0, sigma=1.0, lower_tail=True, log_p=False):
    """CDF of ``LQNO``.

    The upper tail is evaluated as ``ndtr(-z)`` rather than ``1 - ndtr(z)`` so
    that ``log_p=True`` stays accurate deep in the tail (R's ``pnorm`` does the
    same) instead of losing every significant digit to cancellation.
    """
    q, mu, sigma = _broadcast((q, mu, sigma))
    valid = _all_positive(mu, sigma)
    safe_mu = jnp.where(mu > 0.0, mu, 1.0)
    safe_sigma = jnp.where(sigma > 0.0, sigma, 1.0)
    z = (q - safe_mu) / _lqno_scale(safe_mu, safe_sigma)
    z_tail = z if lower_tail else -z
    value = _safe_log(ndtr(z_tail)) if log_p else ndtr(z_tail)
    return jnp.where(valid, value, _NAN)


def qLQNO(p, mu=1.0, sigma=1.0, lower_tail=True, log_p=False):
    """Quantile of ``LQNO``."""
    p, mu, sigma = _broadcast((p, mu, sigma))
    valid = _all_positive(mu, sigma)
    safe_mu = jnp.where(mu > 0.0, mu, 1.0)
    safe_sigma = jnp.where(sigma > 0.0, sigma, 1.0)
    scale = _lqno_scale(safe_mu, safe_sigma)
    prob = _prob_from_p(p, lower_tail, log_p)
    inside = (prob > 0.0) & (prob < 1.0)
    z = ndtri(jnp.where(inside, prob, 0.5))
    value = jnp.where(prob <= 0.0, -_INF, jnp.where(prob >= 1.0, _INF, safe_mu + scale * z))
    return jnp.where(valid, value, _NAN)


def rLQNO(key, n, mu=1.0, sigma=1.0):
    """Inverse-transform sampler for ``LQNO``."""
    return qLQNO(jr.uniform(key, (int(n),), dtype=jnp.float64), mu, sigma)


# ---------------------------------------------------------------------------
# MN3 / MN4 / MN5 - categorical families over 1..k with a reference category k
# ---------------------------------------------------------------------------


def _mn_log_prob(y, params):
    """Log-probability of the categorical MN families.

    ``params`` holds the weights of categories ``1..k-1``; category ``k`` carries
    weight 1.  Mirrors ``dMN3``/``dMN4``/``dMN5`` exactly, including the fact
    that any ``y`` outside ``1..k-1`` falls through to the reference category.
    """
    normalizer = 1.0
    for value in params:
        normalizer = normalizer + value
    log_normalizer = jnp.log(normalizer)
    result = -log_normalizer
    for index, value in enumerate(params, start=1):
        result = jnp.where(y == index, jnp.log(value) - log_normalizer, result)
    return result


def _mn_cdf(y, params):
    """Cumulative probability at ``y`` for the categorical MN families."""
    k = len(params) + 1
    normalizer = 1.0
    for value in params:
        normalizer = normalizer + value
    cumulative = 0.0
    result = jnp.zeros_like(y)
    for index, value in enumerate(params, start=1):
        cumulative = cumulative + value
        result = jnp.where(y == index, cumulative / normalizer, result)
    return jnp.where(y == k, 1.0, result)


def _mn_quantile(prob, params):
    """Smallest category whose cumulative probability is not below ``prob``."""
    normalizer = 1.0
    for value in params:
        normalizer = normalizer + value
    cumulative = 0.0
    result = jnp.ones_like(prob)
    for index, value in enumerate(params, start=1):
        cumulative = cumulative + value
        result = jnp.where(prob >= cumulative / normalizer, index + 1, result)
    return result


def _mn_d(y, params, log):
    y, *params = _broadcast((y, *params))
    valid = _all_positive(*params)
    value = _mn_log_prob(y, params)
    value = jnp.where(valid, value, _NAN)
    return value if log else jnp.exp(value)


def _mn_p(q, params, lower_tail, log_p):
    q, *params = _broadcast((q, *params))
    valid = _all_positive(*params)
    value = _apply_tail(_mn_cdf(q, params), lower_tail, log_p)
    return jnp.where(valid, value, _NAN)


def _mn_q(p, params, lower_tail, log_p):
    p, *params = _broadcast((p, *params))
    valid = _all_positive(*params)
    # R's qMN* ignores ``lower.tail``/``log.p``; we honour them (see ``R_QUIRKS``).
    prob = _prob_from_p(p, lower_tail, log_p)
    prob = jnp.clip(prob, 0.0, 1.0)
    value = _mn_quantile(prob, params)
    return jnp.where(valid, value, _NAN)


def dMN3(x, mu=1.0, sigma=1.0, log=False):
    """Probability mass of the 3-category ``MN3`` family."""
    return _mn_d(x, (mu, sigma), log)


def pMN3(q, mu=1.0, sigma=1.0, lower_tail=True, log_p=False):
    """CDF of the 3-category ``MN3`` family."""
    return _mn_p(q, (mu, sigma), lower_tail, log_p)


def qMN3(p, mu=1.0, sigma=1.0, lower_tail=True, log_p=False):
    """Quantile of the 3-category ``MN3`` family."""
    return _mn_q(p, (mu, sigma), lower_tail, log_p)


def rMN3(key, n, mu=1.0, sigma=1.0):
    """Sampler for ``MN3`` — inverse transform on a uniform draw (as R does)."""
    return qMN3(jr.uniform(key, (int(n),), dtype=jnp.float64), mu, sigma)


def dMN4(x, mu=1.0, sigma=1.0, nu=1.0, log=False):
    """Probability mass of the 4-category ``MN4`` family."""
    return _mn_d(x, (mu, sigma, nu), log)


def pMN4(q, mu=1.0, sigma=1.0, nu=1.0, lower_tail=True, log_p=False):
    """CDF of the 4-category ``MN4`` family."""
    return _mn_p(q, (mu, sigma, nu), lower_tail, log_p)


def qMN4(p, mu=1.0, sigma=1.0, nu=1.0, lower_tail=True, log_p=False):
    """Quantile of the 4-category ``MN4`` family."""
    return _mn_q(p, (mu, sigma, nu), lower_tail, log_p)


def rMN4(key, n, mu=1.0, sigma=1.0, nu=1.0):
    """Sampler for ``MN4``."""
    return qMN4(jr.uniform(key, (int(n),), dtype=jnp.float64), mu, sigma, nu)


def dMN5(x, mu=1.0, sigma=1.0, nu=1.0, tau=1.0, log=False):
    """Probability mass of the 5-category ``MN5`` family."""
    return _mn_d(x, (mu, sigma, nu, tau), log)


def pMN5(q, mu=1.0, sigma=1.0, nu=1.0, tau=1.0, lower_tail=True, log_p=False):
    """CDF of the 5-category ``MN5`` family."""
    return _mn_p(q, (mu, sigma, nu, tau), lower_tail, log_p)


def qMN5(p, mu=1.0, sigma=1.0, nu=1.0, tau=1.0, lower_tail=True, log_p=False):
    """Quantile of the 5-category ``MN5`` family."""
    return _mn_q(p, (mu, sigma, nu, tau), lower_tail, log_p)


def rMN5(key, n, mu=1.0, sigma=1.0, nu=1.0, tau=1.0):
    """Sampler for ``MN5``."""
    return qMN5(jr.uniform(key, (int(n),), dtype=jnp.float64), mu, sigma, nu, tau)


# ---------------------------------------------------------------------------
# exGAUS - exponentially modified Gaussian, mu real, sigma > 0, nu > 0
# ---------------------------------------------------------------------------


def _exgaus_z(x, mu, sigma, nu):
    """R's ``z <- x - mu - sigma^2/nu``."""
    return x - mu - sigma * sigma / nu


def _exgaus_log_density(x, mu, sigma, nu):
    """R's ``dexGAUS`` log branch, reproduced term for term."""
    z = _exgaus_z(x, mu, sigma, nu)
    emg = (
        -jnp.log(nu)
        - (z + sigma * sigma / (2.0 * nu)) / nu
        + _safe_log(ndtr(z / sigma))
    )
    normal = -0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma) - 0.5 * ((x - mu) / sigma) ** 2
    return jnp.where(nu > 0.05 * sigma, emg, normal)


def dexGAUS(x, mu=5.0, sigma=1.0, nu=1.0, log=False):
    """Density of the ex-Gaussian distribution ``N(mu, sigma) + Exp(mean=nu)``."""
    x, mu, sigma, nu = _broadcast((x, mu, sigma, nu))
    valid = (sigma > 0.0) & (nu > 0.0)
    safe_sigma = jnp.where(sigma > 0.0, sigma, 1.0)
    safe_nu = jnp.where(nu > 0.0, nu, 1.0)
    value = jnp.where(valid, _exgaus_log_density(x, mu, safe_sigma, safe_nu), _NAN)
    return value if log else jnp.exp(value)


def _exgaus_cdf_scalar(q, mu, sigma, nu):
    """R's ``pexGAUS`` body, reproduced term for term (no ``sigma < 0`` branch)."""
    z = _exgaus_z(q, mu, sigma, nu)
    emg = ndtr((q - mu) / sigma) - ndtr(z / sigma) * jnp.exp(
        ((mu + sigma * sigma / nu) ** 2 - mu * mu - 2.0 * q * (sigma * sigma / nu))
        / (2.0 * sigma * sigma)
    )
    normal = ndtr((q - mu) / sigma)
    return jnp.where(nu > 0.05 * sigma, emg, normal)


def pexGAUS(q, mu=5.0, sigma=1.0, nu=1.0, lower_tail=True, log_p=False):
    """CDF of the ex-Gaussian distribution."""
    q, mu, sigma, nu = _broadcast((q, mu, sigma, nu))
    valid = (sigma > 0.0) & (nu > 0.0)
    safe_sigma = jnp.where(sigma > 0.0, sigma, 1.0)
    safe_nu = jnp.where(nu > 0.0, nu, 1.0)
    cdf = jnp.clip(_exgaus_cdf_scalar(q, mu, safe_sigma, safe_nu), 0.0, 1.0)
    value = _apply_tail(cdf, lower_tail, log_p)
    return jnp.where(valid, value, _NAN)


def _bisection_invert(prob, cdf_fn, lo, hi, *, expansions=24, iters=90):
    """Vectorised bracket-then-bisect inversion of a monotone increasing CDF.

    ``exGAUS`` (and later ``GIG``) have no closed-form quantile, so R uses
    ``uniroot``.  A fixed-iteration bisection is deterministic, JIT-safe and
    reaches full double precision long before the iteration cap.
    """
    for _ in range(expansions):
        width = hi - lo
        lo = jnp.where(cdf_fn(lo) > prob, lo - width, lo)
        hi = jnp.where(cdf_fn(hi) < prob, hi + width, hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        below = cdf_fn(mid) < prob
        lo = jnp.where(below, mid, lo)
        hi = jnp.where(below, hi, mid)
    return 0.5 * (lo + hi)


def qexGAUS(p, mu=5.0, sigma=1.0, nu=1.0, lower_tail=True, log_p=False):
    """Quantile of the ex-Gaussian distribution (bisection, like R's ``uniroot``)."""
    p, mu, sigma, nu = _broadcast((p, mu, sigma, nu))
    valid = (sigma > 0.0) & (nu > 0.0)
    safe_sigma = jnp.where(sigma > 0.0, sigma, 1.0)
    safe_nu = jnp.where(nu > 0.0, nu, 1.0)
    prob = _prob_from_p(p, lower_tail, log_p)
    prob = jnp.clip(prob, 0.0, 1.0)
    span = safe_sigma + safe_nu
    lo = mu - 64.0 * span
    hi = mu + 64.0 * span

    def cdf_fn(v):
        return jnp.clip(_exgaus_cdf_scalar(v, mu, safe_sigma, safe_nu), 0.0, 1.0)

    value = _bisection_invert(prob, cdf_fn, lo, hi)
    value = jnp.where(prob <= 0.0, -_INF, value)
    value = jnp.where(prob >= 1.0, _INF, value)
    return jnp.where(valid, value, _NAN)


def rexGAUS(key, n, mu=5.0, sigma=1.0, nu=1.0):
    """Sampler for the ex-Gaussian distribution.

    ``exGAUS`` is *defined* as ``N(mu, sigma) + Exp(mean=nu)``, so convolving two
    exact draws is an exact sampler — and far cheaper than R's route of pushing a
    uniform through the numerically inverted ``qexGAUS``.
    """
    z = jr.normal(key, (int(n),), dtype=jnp.float64)
    return mu + sigma * z + nu * jr.exponential(key, (int(n),), dtype=jnp.float64)


# ---------------------------------------------------------------------------
# EGB2 - exponential generalized beta type 2, mu real, sigma/nu/tau > 0
# ---------------------------------------------------------------------------


def dEGB2(x, mu=0.0, sigma=1.0, nu=1.0, tau=0.5, log=False):
    """Density of the exponential generalized beta type 2 distribution.

    Mirrors R ``dEGB2`` term for term: ``z = (x-mu)/sigma`` and
    ``log f = nu*z - log(sigma) - lgamma(nu) - lgamma(tau) + lgamma(nu+tau)
    - (nu+tau)*log(1+exp(z))``.  ``log(1+exp(z))`` is evaluated as
    ``logaddexp(0, z)`` so the upper tail does not overflow.
    """
    x, mu, sigma, nu, tau = _broadcast((x, mu, sigma, nu, tau))
    valid = (sigma > 0.0) & (nu > 0.0) & (tau > 0.0)
    safe_sigma = jnp.where(sigma > 0.0, sigma, 1.0)
    safe_nu = jnp.where(nu > 0.0, nu, 1.0)
    safe_tau = jnp.where(tau > 0.0, tau, 1.0)
    z = (x - mu) / safe_sigma
    value = (
        safe_nu * z
        - jnp.log(safe_sigma)
        - gammaln(safe_nu)
        - gammaln(safe_tau)
        + gammaln(safe_nu + safe_tau)
        - (safe_nu + safe_tau) * jnp.logaddexp(0.0, z)
    )
    value = jnp.where(valid, value, _NAN)
    return value if log else jnp.exp(value)


def pEGB2(q, mu=0.0, sigma=1.0, nu=1.0, tau=0.5, lower_tail=True, log_p=False):
    """CDF of the exponential generalized beta type 2 distribution (F-based)."""
    q, mu, sigma, nu, tau = _broadcast((q, mu, sigma, nu, tau))
    valid = (sigma > 0.0) & (nu > 0.0) & (tau > 0.0)
    safe_sigma = jnp.where(sigma > 0.0, sigma, 1.0)
    safe_nu = jnp.where(nu > 0.0, nu, 1.0)
    safe_tau = jnp.where(tau > 0.0, tau, 1.0)
    w = (safe_tau / safe_nu) * jnp.exp((q - mu) / safe_sigma)
    w = jnp.where(jnp.isfinite(w), w, 1.0)
    cdf = jnp.asarray(
        _scipy_f.cdf(np.asarray(w, dtype=np.float64), 2.0 * np.asarray(safe_nu, np.float64),
                     2.0 * np.asarray(safe_tau, np.float64)),
        dtype=jnp.float64,
    )
    cdf = jnp.where(jnp.isfinite(w), cdf, jnp.where(q > mu, 1.0, 0.0))
    value = _apply_tail(cdf, lower_tail, log_p)
    return jnp.where(valid, value, _NAN)


def qEGB2(p, mu=0.0, sigma=1.0, nu=1.0, tau=0.5, lower_tail=True, log_p=False):
    """Quantile of the exponential generalized beta type 2 distribution."""
    p, mu, sigma, nu, tau = _broadcast((p, mu, sigma, nu, tau))
    valid = (sigma > 0.0) & (nu > 0.0) & (tau > 0.0)
    safe_sigma = jnp.where(sigma > 0.0, sigma, 1.0)
    safe_nu = jnp.where(nu > 0.0, nu, 1.0)
    safe_tau = jnp.where(tau > 0.0, tau, 1.0)
    prob = _prob_from_p(p, lower_tail, log_p)
    prob = jnp.clip(jnp.where(jnp.isnan(prob), 0.5, prob), 0.0, 1.0)
    inside = (prob > 0.0) & (prob < 1.0)
    prob_s = jnp.where(inside, prob, 0.5)
    w = jnp.asarray(
        _scipy_f.ppf(np.asarray(prob_s, dtype=np.float64), 2.0 * np.asarray(safe_nu, np.float64),
                     2.0 * np.asarray(safe_tau, np.float64)),
        dtype=jnp.float64,
    )
    value = mu + safe_sigma * jnp.log((safe_nu / safe_tau) * w)
    value = jnp.where(prob <= 0.0, -_INF, value)
    value = jnp.where(prob >= 1.0, _INF, value)
    return jnp.where(valid, value, _NAN)


def rEGB2(key, n, mu=0.0, sigma=1.0, nu=1.0, tau=0.5):
    """Inverse-transform sampler for the EGB2 distribution (as R does)."""
    return qEGB2(jr.uniform(key, (int(n),), dtype=jnp.float64), mu, sigma, nu, tau)
