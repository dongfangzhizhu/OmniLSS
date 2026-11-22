"""Finite-support zero-adjusted BB with direct positive-mass normalization."""

import numpy as np
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import logsumexp
from .dpqr_functions import dBB


def _check_bd(bd):
    if (
        isinstance(bd, (bool, np.bool_))
        or not isinstance(bd, (int, np.integer))
        or bd < 1
    ):
        raise ValueError("ZABB bd must be a positive static integer")


def _positive(mu, sigma, bd):
    _check_bd(bd)
    mu, sigma = jnp.broadcast_arrays(
        jnp.asarray(mu, dtype=jnp.float64), jnp.asarray(sigma, dtype=jnp.float64)
    )
    mass = dBB(jnp.arange(1, bd + 1), mu[..., None], sigma[..., None], bd, log=True)
    return mass, logsumexp(mass, axis=-1)


def dZABB(x, mu=0.5, sigma=0.1, nu=0.1, bd=10, log=False):
    _, norm = _positive(mu, sigma, bd)
    x, nu = jnp.asarray(x, dtype=jnp.float64), jnp.asarray(nu, dtype=jnp.float64)
    value = jnp.where(
        x == 0, jnp.log(nu), jnp.log1p(-nu) + dBB(x, mu, sigma, bd, log=True) - norm
    )
    value = jnp.where((nu > 0) & (nu < 1) & jnp.isfinite(norm), value, jnp.nan)
    return value if log else jnp.exp(value)


def _log_table(mu, sigma, nu, bd):
    mu, sigma, nu = jnp.broadcast_arrays(
        *[jnp.asarray(v, dtype=jnp.float64) for v in (mu, sigma, nu)]
    )
    mass, norm = _positive(mu, sigma, bd)
    table = jnp.concatenate(
        (jnp.log(nu)[..., None], jnp.log1p(-nu)[..., None] + mass - norm[..., None]),
        axis=-1,
    )
    return jnp.where(
        ((nu > 0) & (nu < 1) & jnp.isfinite(norm))[..., None], table, jnp.nan
    )


def pZABB(q, mu=0.5, sigma=0.1, nu=0.1, bd=10, lower_tail=True, log_p=False):
    q, mu, sigma, nu = jnp.broadcast_arrays(
        *[jnp.asarray(v, dtype=jnp.float64) for v in (q, mu, sigma, nu)]
    )
    table = _log_table(mu, sigma, nu, bd)
    mask = jnp.arange(bd + 1) <= q[..., None]
    if not lower_tail:
        mask = ~mask
    value = logsumexp(jnp.where(mask, table, -jnp.inf), axis=-1)
    value = jnp.minimum(value, 0.0)
    value = jnp.where(q < 0, -jnp.inf if lower_tail else 0.0, value)
    value = jnp.where(q >= bd, 0.0 if lower_tail else -jnp.inf, value)
    value = jnp.where(jnp.isnan(q) | jnp.any(jnp.isnan(table), axis=-1), jnp.nan, value)
    return value if log_p else jnp.exp(value)


def qZABB(p, mu=0.5, sigma=0.1, nu=0.1, bd=10, lower_tail=True, log_p=False):
    p, mu, sigma, nu = np.broadcast_arrays(
        *[np.asarray(v, dtype=float) for v in (p, mu, sigma, nu)]
    )
    table = np.asarray(_log_table(mu, sigma, nu, bd))
    with np.errstate(all="ignore"):
        valid = (p <= 0) if log_p else ((p >= 0) & (p <= 1))
        target = (
            (p if lower_tail else np.log(-np.expm1(p)))
            if log_p
            else (np.log(p) if lower_tail else np.log1p(-p))
        )
        cdf = np.logaddexp.accumulate(table, axis=-1)
        cdf[..., -1] = 0.0
        crossed = cdf >= target[..., None]
        value = np.argmax(crossed, axis=-1).astype(float)
        value = np.where(target == 0, bd, value)
    return jnp.asarray(
        np.where(
            valid & ~np.isnan(p) & ~np.any(np.isnan(table), axis=-1), value, np.nan
        )
    )


def rZABB(key, n, mu=0.5, sigma=0.1, nu=0.1, bd=10):
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    return qZABB(jr.uniform(key, (n,), dtype=jnp.float64), mu, sigma, nu, bd)
