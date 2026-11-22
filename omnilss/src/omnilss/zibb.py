"""Zero-inflated beta-binomial kernels, with explicit fixed denominator."""

import jax.numpy as jnp
import jax.random as jr
from .dpqr_functions import dBB, pBB, rBB
import numpy as np


def dZIBB(x, mu=0.5, sigma=0.5, nu=0.1, bd=10, log=False):
    x, nu = jnp.broadcast_arrays(
        jnp.asarray(x, dtype=jnp.float64), jnp.asarray(nu, dtype=jnp.float64)
    )
    base = dBB(x, mu, sigma, bd, log=True)
    value = jnp.where(
        x == 0, jnp.logaddexp(jnp.log(nu), jnp.log1p(-nu) + base), jnp.log1p(-nu) + base
    )
    value = jnp.where((nu > 0) & (nu < 1), value, jnp.nan)
    return value if log else jnp.exp(value)


def pZIBB(q, mu=0.5, sigma=0.5, nu=0.1, bd=10, lower_tail=True, log_p=False):
    q, nu = np.broadcast_arrays(np.asarray(q, dtype=float), np.asarray(nu, dtype=float))
    logs = np.asarray(pBB(q, mu, sigma, bd, lower_tail=False, log_p=True))
    with np.errstate(all="ignore"):
        logs = np.log1p(-nu) + logs
        logs = np.where(q < 0, 0.0, logs)
        value = -np.expm1(logs) if lower_tail else np.exp(logs)
        if log_p:
            value = np.log(value) if lower_tail else logs
    # Preserve invalid base parameters even below the support.
    valid = (nu > 0) & (nu < 1) & ~np.isnan(np.asarray(pBB(q, mu, sigma, bd)))
    return jnp.asarray(np.where(valid, value, np.nan))


def qZIBB(p, mu=0.5, sigma=0.5, nu=0.1, bd=10, lower_tail=True, log_p=False):
    p, mu, sigma, nu, bd = np.broadcast_arrays(
        *[np.asarray(v, dtype=float) for v in (p, mu, sigma, nu, bd)]
    )
    out = np.full(p.shape, np.nan)
    for index in np.ndindex(p.shape):
        prob, m, s, z, n = (v[index] for v in (p, mu, sigma, nu, bd))
        if not (
            0 < m < 1
            and 0 < s < np.inf
            and 0 < z < 1
            and 0 <= n < np.inf
            and n == np.floor(n)
        ):
            continue
        if np.isnan(prob) or (prob > 0 if log_p else not 0 <= prob <= 1):
            continue
        with np.errstate(all="ignore"):
            target = (
                (np.log(-np.expm1(prob)) if lower_tail else prob)
                if log_p
                else (np.log1p(-prob) if lower_tail else np.log(prob))
            )
        lo, hi = -1, int(n)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            sf = float(pZIBB(mid, m, s, z, n, lower_tail=False, log_p=True))
            if sf <= target:
                hi = mid
            else:
                lo = mid
        out[index] = hi
    return jnp.asarray(out)


def rZIBB(key, n, mu=0.5, sigma=0.5, nu=0.1, bd=10):
    base_key, zero_key = jr.split(key)
    base = rBB(base_key, n, mu, sigma, bd)
    nu = jnp.broadcast_to(jnp.asarray(nu, dtype=jnp.float64), (n,))
    sample = jnp.where(jr.uniform(zero_key, (n,), dtype=jnp.float64) < nu, 0.0, base)
    return jnp.where((nu > 0) & (nu < 1) & jnp.isfinite(base), sample, jnp.nan)
