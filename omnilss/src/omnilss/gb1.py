"""GB1 kernels using gamlss.dist's bounded, transformed-Beta parameterization."""
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import betaln
from scipy.special import betainc, betaincc, betaincinv, betainccinv, expit


def _valid(mu, sigma, nu, tau):
    return ((mu > 0) & (mu < 1) & (sigma > 0) & (sigma < 1)
            & (nu > 0) & (tau > 0) & jnp.isfinite(nu) & jnp.isfinite(tau))


def log_density(x, mu, sigma, nu, tau):
    x, mu, sigma, nu, tau = jnp.broadcast_arrays(
        *[jnp.asarray(v, dtype=jnp.float64) for v in (x, mu, sigma, nu, tau)])
    valid = _valid(mu, sigma, nu, tau)
    y = jnp.where((x > 0) & (x < 1), x, .5)
    a = mu * (1 / sigma**2 - 1)
    b = (1-mu) * (1 / sigma**2 - 1)
    log_t = tau * jnp.log(y)
    log_w = jnp.log(-jnp.expm1(log_t))
    log_den = jnp.logaddexp(log_t, jnp.log(nu) + log_w)
    value = (jnp.log(tau) + b*jnp.log(nu) + (tau*a-1)*jnp.log(y)
             + (b-1)*log_w - betaln(a, b) - (a+b)*log_den)
    value = jnp.where((x <= 0) | (x >= 1), -jnp.inf, value)
    return jnp.where(valid & ~jnp.isnan(x), value, jnp.nan)


def dGB1(x, mu=.5, sigma=.4, nu=1., tau=1., log=False):
    value = log_density(x, mu, sigma, nu, tau)
    return value if log else jnp.exp(value)


def _arrays(x, mu, sigma, nu, tau):
    return np.broadcast_arrays(*[np.asarray(v, dtype=float) for v in (x, mu, sigma, nu, tau)])


def pGB1(q, mu=.5, sigma=.4, nu=1., tau=1., lower_tail=True, log_p=False):
    q, mu, sigma, nu, tau = _arrays(q, mu, sigma, nu, tau)
    with np.errstate(all="ignore"):
        a, b = mu*(1/sigma**2-1), (1-mu)*(1/sigma**2-1)
        t = tau*np.log(np.where((q > 0) & (q < 1), q, .5))
        z = t - np.log(nu) - np.log(-np.expm1(t))
        lower = np.where(z <= 0, betainc(a, b, expit(z)), betaincc(b, a, expit(-z)))
        upper = np.where(z >= 0, betainc(b, a, expit(-z)), betaincc(a, b, expit(z)))
        value = lower if lower_tail else upper
        value = np.where(q <= 0, float(not lower_tail), value)
        value = np.where(q >= 1, float(lower_tail), value)
        if log_p:
            value = np.log(value)
    return jnp.asarray(np.where(np.asarray(_valid(mu, sigma, nu, tau)) & ~np.isnan(q), value, np.nan))


def qGB1(p, mu=.5, sigma=.4, nu=1., tau=1., lower_tail=True, log_p=False):
    p, mu, sigma, nu, tau = _arrays(p, mu, sigma, nu, tau)
    valid_p = (p <= 0) if log_p else ((p >= 0) & (p <= 1))
    with np.errstate(all="ignore"):
        flip = p > (-np.log(2) if log_p else .5)
        prob = np.where(flip, -np.expm1(p), np.exp(p)) if log_p else np.where(flip, 1-p, p)
        lower = np.logical_xor(lower_tail, flip)
        a, b = mu*(1/sigma**2-1), (1-mu)*(1/sigma**2-1)
        z = np.where(lower, betaincinv(a,b,prob), betainccinv(a,b,prob))
        w = np.where(lower, betainccinv(b,a,prob), betaincinv(b,a,prob))
        log_t = -np.logaddexp(0., np.log(w)-np.log(nu)-np.log(z))
        value = np.exp(log_t/tau)
    return jnp.asarray(np.where(np.asarray(_valid(mu,sigma,nu,tau)) & valid_p, value, np.nan))


def rGB1(key, n, mu=.5, sigma=.4, nu=1., tau=1.):
    return qGB1(jr.uniform(key, (n,), dtype=jnp.float64), mu, sigma, nu, tau)
