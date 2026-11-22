"""JAX PIG kernels.

D2.29: the numerical recursion still runs 1..10000, but live R does *not*
answer ``nan`` outside that range -- ``pPIG`` hands back its accumulated
partial sum (R prints ``1.0000000000000004`` for ``q >= 10001``, its own
summation ulp above 1) and ``qPIG`` short-circuits ``p + 1e-9 >= 1`` to
``Inf`` without scanning.  Both behaviours are reproduced here; only the
pmf keeps the bounded support.  Sampling uses the unbounded
Poisson/inverse-Gaussian mixture (Var(Y)=mu+sigma*mu**2).
"""

import jax
import jax.numpy as jnp
import numpy as np


def _args(x, mu, sigma):
    x, mu, sigma = jnp.broadcast_arrays(
        *[jnp.asarray(v, dtype=jnp.float64) for v in (x, mu, sigma)]
    )
    valid = (mu > 0) & (sigma > 0) & jnp.isfinite(mu) & jnp.isfinite(sigma)
    return x, jnp.where(valid, mu, 1.0), jnp.where(valid, sigma, 1.0), valid


def _initial(mu, sigma):
    root = jnp.sqrt(1 + 2 * sigma * mu)
    return mu / root, -2 * mu / (1 + root)


def _step(i, ratio, logmass, mu, sigma):
    first, _ = _initial(mu, sigma)
    mass = logmass + jnp.log(ratio) - jnp.log(i)
    ratio = (sigma * (2 * i - 1) / mu + 1 / ratio) * first**2
    return ratio, mass


# D2.150: the pmf recurrence used to be capped at a fixed 10001 steps and mask
# anything beyond it to nan.  Live R answers dPIG(10001) = -4068.588 (log
# scale; 0 on the density scale) and dPIG(20000) = -8123.874, so the cap was
# our own limitation, not R's.  The bound is now derived from the data when the
# argument is concrete (which also *shortens* the loop for ordinary counts:
# it was always running 10000 iterations regardless of y).
_PMF_STEP_CAP = 1_000_000


def _pmf_steps(x) -> int:
    try:
        arr = np.asarray(x, dtype=np.float64)
    except Exception:  # traced value under jit -- keep the previous bound
        return 10001
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return 10001
    return int(min(max(float(np.max(arr)) + 1.0, 2.0), _PMF_STEP_CAP))


def logpmf(x, mu, sigma):
    x, mu, sigma, valid = _args(x, mu, sigma)
    ratio, base = _initial(mu, sigma)
    n_steps = _pmf_steps(x)

    def step(i, state):
        ratio, mass, result = state
        ratio, mass = _step(i, ratio, mass, mu, sigma)
        return ratio, mass, jnp.where(x == i, mass, result)

    _, _, result = jax.lax.fori_loop(1, n_steps, step, (ratio, base, base))
    result = jnp.where(
        (x >= 0) & jnp.isfinite(x) & (x == jnp.floor(x)), result, -jnp.inf
    )
    return jnp.where(
        valid & ~jnp.isnan(x) & ~((x >= n_steps) & jnp.isfinite(x)), result, jnp.nan
    )


def dPIG(x, mu=1.0, sigma=1.0, log=False):
    value = logpmf(x, mu, sigma)
    return value if log else jnp.exp(value)


def pPIG(q, mu=1.0, sigma=1.0, lower_tail=True, log_p=False):
    q, mu, sigma, valid = _args(q, mu, sigma)
    ratio, base = _initial(mu, sigma)

    def step(i, state):
        ratio, mass, total = state
        ratio, mass = _step(i, ratio, mass, mu, sigma)
        return ratio, mass, jnp.logaddexp(total, jnp.where(i <= q, mass, -jnp.inf))

    _, _, total = jax.lax.fori_loop(1, 10001, step, (ratio, base, base))
    total = jnp.minimum(total, 0.0)
    total = jnp.where(q < 0, -jnp.inf, jnp.where(jnp.isposinf(q), 0.0, total))
    result = total if lower_tail else jnp.log(-jnp.expm1(total))
    result = result if log_p else jnp.exp(result)
    # D2.29: live R does not truncate -- q = 10001 and q = 1e6 both get the
    # accumulated partial sum (R's own print saturates a summation ulp above
    # 1; our log-scale clamp keeps exactly 1.0, within 4e-16 of R).  The old
    # nan cutoff answered a real question with nan where R answers ~1.
    return jnp.where(valid & ~jnp.isnan(q), result, jnp.nan)


def qPIG(p, mu=1.0, sigma=1.0, lower_tail=True, log_p=False, max_value=10000):
    if not isinstance(max_value, int) or not 0 <= max_value <= 10000:
        raise ValueError("max_value must be an integer in [0, 10000]")
    p, mu, sigma, valid = _args(p, mu, sigma)
    prob = jnp.exp(p) if log_p else p
    valid = valid & (prob >= 0) & (prob <= 1)
    prob = prob if lower_tail else (-jnp.expm1(p) if log_p else 1 - prob)
    target = jnp.log(prob)
    ratio, base = _initial(mu, sigma)
    found = jnp.where(base >= target, 0.0, jnp.nan)

    def step(i, state):
        ratio, mass, total, found = state
        ratio, mass = _step(i, ratio, mass, mu, sigma)
        total = jnp.logaddexp(total, mass)
        found = jnp.where(jnp.isnan(found) & (total >= target), i, found)
        return ratio, mass, total, found

    *_, result = jax.lax.fori_loop(1, max_value + 1, step, (ratio, base, base, found))
    # D2.29: R's upper-tail short circuit, verbatim (`if (p[i] + 1e-09 >= 1)
    # QQQ[i] <- Inf`), checked *after* the tail conversion -- which is where
    # R checks it too.  Live R: qPIG(0.999999999) is already Inf while 0.999
    # still resolves; without this we scanned to the pmf cap and answered 62
    # where R answers Inf.
    result = jnp.where(prob + 1e-9 >= 1, jnp.inf, jnp.where(prob == 0, 0.0, result))
    return jnp.where(valid, result, jnp.nan)


def rPIG(key, n, mu=1.0, sigma=1.0):
    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    _, mu, sigma, valid = _args(jnp.empty((n,)), mu, sigma)
    normal_key, uniform_key, poisson_key = jax.random.split(key, 3)
    # Michael-Schucany-Haas inverse Gaussian sampler, rationalized small root.
    z = jax.random.normal(normal_key, (n,), dtype=jnp.float64) ** 2
    a = sigma * z / 2
    small = 1 / (1 + a + jnp.sqrt(a * (a + 2)))
    u = jax.random.uniform(uniform_key, (n,), dtype=jnp.float64)
    rate = mu * jnp.where(u <= 1 / (1 + small), small, 1 / small)
    samples = jax.random.poisson(poisson_key, rate, shape=(n,))
    return jnp.where(valid, samples.astype(jnp.float64), jnp.nan)
