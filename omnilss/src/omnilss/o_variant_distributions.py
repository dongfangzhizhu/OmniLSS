"""Public dpqr functions for the R 'o' (original) parameterisation families.

R gamlss.dist exports dBCCGo / dBCPEo / dBCTo / dBEo / dGEOMo / dJSUo /
dPARETO2o / dSHASHo and the matching p/q/r helpers.  The kernels already
existed privately in the distribution modules and the family registry; these
wrappers expose them with R's argument names and defaults.

Verification: every d/p/q function is compared against live R 4.6.0 +
gamlss.dist 6.1.1 on the audit grid; the maximum absolute difference is at
most 4.1e-14 (BCT quantile) and typically below 1e-15.  The r functions draw
uniforms through the quantile function exactly like R, so the distribution
matches while the random stream necessarily does not: pass an explicit JAX
PRNG key for reproducibility, or leave it None for an R-like fresh draw.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .distributions_b12 import _beo_log_pdf, _geomo_log_pdf, _pareto2o_log_pdf
from .distributions_b14 import _bcpeo_log_pdf, _bcto_log_pdf, _jsuo_log_pdf
from .distributions_b16 import _bccgo_log_pdf
from .distributions_b3 import _shasho_log_pdf


def _emit(log_value: Any, log: bool) -> Any:
    return log_value if log else jnp.exp(log_value)


def _family(name: str):
    """Resolve lazily: the registry imports the distribution modules."""
    from .distribution_registry import resolve

    return resolve(name)


def _resolve_key(key: Any) -> Any:
    if key is not None:
        return key
    return jax.random.PRNGKey(int(np.random.randint(0, 2**31 - 1)))


def _check_n(n: Any) -> int:
    n_int = int(np.ceil(float(np.max(np.asarray(n, dtype=float)))))
    if np.any(np.asarray(n, dtype=float) <= 0):
        raise ValueError("n must be a positive integer")
    return n_int


# ---------------------------------------------------------------------------
# Densities
# ---------------------------------------------------------------------------


def dBCCGo(x, mu=1.0, sigma=0.1, nu=1.0, log: bool = False):
    """R dBCCGo: Box-Cox-Cole-Green (original parameterisation)."""
    return _emit(_bccgo_log_pdf(x, mu, sigma, nu), log)


def dBCPEo(x, mu=5.0, sigma=0.1, nu=1.0, tau=2.0, log: bool = False):
    """R dBCPEo: Box-Cox power exponential (original parameterisation)."""
    return _emit(_bcpeo_log_pdf(x, mu, sigma, nu, tau), log)


def dBCTo(x, mu=5.0, sigma=0.1, nu=1.0, tau=2.0, log: bool = False):
    """R dBCTo: Box-Cox t (original parameterisation)."""
    return _emit(_bcto_log_pdf(x, mu, sigma, nu, tau), log)


def dBEo(x, mu=0.5, sigma=0.2, log: bool = False):
    """R dBEo: beta (original parameterisation, mu = shape1)."""
    return _emit(_beo_log_pdf(x, mu, sigma), log)


def dGEOMo(x, mu=0.5, log: bool = False):
    """R dGEOMo: geometric (original parameterisation)."""
    return _emit(_geomo_log_pdf(x, mu), log)


def dJSUo(x, mu=0.0, sigma=1.0, nu=0.0, tau=1.0, log: bool = False):
    """R dJSUo: Johnson SU (original parameterisation)."""
    return _emit(_jsuo_log_pdf(x, mu, sigma, nu, tau), log)


def dPARETO2o(x, mu=1.0, sigma=0.5, log: bool = False):
    """R dPARETO2o: Pareto type 2 (original parameterisation)."""
    return _emit(_pareto2o_log_pdf(x, mu, sigma), log)


def dSHASHo(x, mu=0.0, sigma=1.0, nu=0.0, tau=1.0, log: bool = False):
    """R dSHASHo: sinh-arcsinh (original parameterisation)."""
    return _emit(_shasho_log_pdf(x, mu, sigma, nu, tau), log)


# ---------------------------------------------------------------------------
# Distribution functions.  R defines pBCCGo == pBCCG, pBCPEo == pBCPE and
# pBCTo == pBCT, so delegating to the registry family reproduces R exactly.
# ---------------------------------------------------------------------------


def pBCCGo(q, mu=1.0, sigma=0.1, nu=1.0, lower_tail=True, log_p=False):
    """R pBCCGo: CDF of the Box-Cox-Cole-Green (original)."""
    return _family("BCCGO").p(q, mu, sigma, nu, lower_tail=lower_tail, log_p=log_p)


def pBCPEo(q, mu=5.0, sigma=0.1, nu=1.0, tau=2.0, lower_tail=True, log_p=False):
    """R pBCPEo: CDF of the Box-Cox power exponential (original)."""
    return _family("BCPEO").p(q, mu, sigma, nu, tau, lower_tail=lower_tail, log_p=log_p)


def pBCTo(q, mu=5.0, sigma=0.1, nu=1.0, tau=2.0, lower_tail=True, log_p=False):
    """R pBCTo: CDF of the Box-Cox t (original)."""
    return _family("BCTO").p(q, mu, sigma, nu, tau, lower_tail=lower_tail, log_p=log_p)


def pBEo(q, mu=0.5, sigma=0.2, lower_tail=True, log_p=False):
    """R pBEo: CDF of the beta (original)."""
    return _family("BEO").p(q, mu, sigma, lower_tail=lower_tail, log_p=log_p)


def pGEOMo(q, mu=0.5, lower_tail=True, log_p=False):
    """R pGEOMo: CDF of the geometric (original)."""
    return _family("GEOMO").p(q, mu, lower_tail=lower_tail, log_p=log_p)


def pJSUo(q, mu=0.0, sigma=1.0, nu=0.0, tau=1.0, lower_tail=True, log_p=False):
    """R pJSUo: CDF of the Johnson SU (original)."""
    return _family("JSUO").p(q, mu, sigma, nu, tau, lower_tail=lower_tail, log_p=log_p)


def pPARETO2o(q, mu=1.0, sigma=0.5, lower_tail=True, log_p=False):
    """R pPARETO2o: CDF of the Pareto type 2 (original).

    Documented deviation: live gamlss.dist 6.1.1 writes

        if (log.p == FALSE) cdf <- cdf
        else cdf < - log(cdf)

    and R parses cdf < - log(cdf) as the comparison cdf < (-log(cdf)), so the
    log.p branch is a no-op and R silently returns the probability.  We
    implement the evident intent (log of the CDF) instead, matching the
    internal _pareto2o_p kernel used by the fitting engine.  With log_p=False
    the numbers are identical to R.
    """
    return _family("PARETO2O").p(q, mu, sigma, lower_tail=lower_tail, log_p=log_p)


def pSHASHo(q, mu=0.0, sigma=1.0, nu=0.0, tau=1.0, lower_tail=True, log_p=False):
    """R pSHASHo: CDF of the sinh-arcsinh (original)."""
    return _family("SHASHO").p(
        q, mu, sigma, nu, tau, lower_tail=lower_tail, log_p=log_p
    )


# ---------------------------------------------------------------------------
# Quantile functions
# ---------------------------------------------------------------------------


def qBCCGo(p, mu=1.0, sigma=0.1, nu=1.0, lower_tail=True, log_p=False):
    """R qBCCGo: quantiles of the Box-Cox-Cole-Green (original)."""
    return _family("BCCGO").q(p, mu, sigma, nu, lower_tail=lower_tail, log_p=log_p)


def qBCPEo(p, mu=5.0, sigma=0.1, nu=1.0, tau=2.0, lower_tail=True, log_p=False):
    """R qBCPEo: quantiles of the Box-Cox power exponential (original)."""
    return _family("BCPEO").q(p, mu, sigma, nu, tau, lower_tail=lower_tail, log_p=log_p)


def qBCTo(p, mu=5.0, sigma=0.1, nu=1.0, tau=2.0, lower_tail=True, log_p=False):
    """R qBCTo: quantiles of the Box-Cox t (original)."""
    return _family("BCTO").q(p, mu, sigma, nu, tau, lower_tail=lower_tail, log_p=log_p)


def qBEo(p, mu=0.5, sigma=0.2, lower_tail=True, log_p=False):
    """R qBEo: quantiles of the beta (original)."""
    return _family("BEO").q(p, mu, sigma, lower_tail=lower_tail, log_p=log_p)


def qGEOMo(p, mu=0.5, lower_tail=True, log_p=False):
    """R qGEOMo: quantiles of the geometric (original)."""
    return _family("GEOMO").q(p, mu, lower_tail=lower_tail, log_p=log_p)


def qJSUo(p, mu=0.0, sigma=1.0, nu=0.0, tau=1.0, lower_tail=True, log_p=False):
    """R qJSUo: quantiles of the Johnson SU (original)."""
    return _family("JSUO").q(p, mu, sigma, nu, tau, lower_tail=lower_tail, log_p=log_p)


def qPARETO2o(p, mu=1.0, sigma=0.5, lower_tail=True, log_p=False):
    """R qPARETO2o: quantiles of the Pareto type 2 (original)."""
    return _family("PARETO2O").q(p, mu, sigma, lower_tail=lower_tail, log_p=log_p)


def qSHASHo(p, mu=0.0, sigma=1.0, nu=0.0, tau=1.0, lower_tail=True, log_p=False):
    """R qSHASHo: quantiles of the sinh-arcsinh (original)."""
    return _family("SHASHO").q(
        p, mu, sigma, nu, tau, lower_tail=lower_tail, log_p=log_p
    )


# ---------------------------------------------------------------------------
# Random generation.  R draws runif(n) and inverts the CDF; we do the same
# with a JAX PRNG key, so the law matches but the stream cannot.
# ---------------------------------------------------------------------------


def rBCCGo(n, mu=1.0, sigma=0.1, nu=1.0, key=None):
    """R rBCCGo: random draws (pass key for reproducibility)."""
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be positive")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    size = _check_n(n)
    u = jax.random.uniform(_resolve_key(key), shape=(size,))
    return qBCCGo(u, mu, sigma, nu)


def rBCPEo(n, mu=5.0, sigma=0.1, nu=1.0, tau=2.0, key=None):
    """R rBCPEo: random draws (pass key for reproducibility)."""
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be positive")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    if np.any(np.asarray(tau) <= 0):
        raise ValueError("tau must be positive")
    size = _check_n(n)
    u = jax.random.uniform(_resolve_key(key), shape=(size,))
    return qBCPEo(u, mu, sigma, nu, tau)


def rBCTo(n, mu=5.0, sigma=0.1, nu=1.0, tau=2.0, key=None):
    """R rBCTo: random draws (pass key for reproducibility)."""
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be positive")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    if np.any(np.asarray(tau) <= 0):
        raise ValueError("tau must be positive")
    size = _check_n(n)
    u = jax.random.uniform(_resolve_key(key), shape=(size,))
    return qBCTo(u, mu, sigma, nu, tau)


def rBEo(n, mu=0.5, sigma=0.2, key=None):
    """R rBEo: random draws.  R inverts qbeta(p, shape1=mu, shape2=sigma)."""
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be positive")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    size = _check_n(n)
    u = jax.random.uniform(_resolve_key(key), shape=(size,))
    return qBEo(u, mu, sigma)


def rGEOMo(n, mu=0.5, key=None):
    """R rGEOMo: random draws (integer valued, like R's as.integer)."""
    if np.any(np.asarray(mu) < 0):
        raise ValueError("mu must be > 0")
    size = _check_n(n)
    u = jax.random.uniform(_resolve_key(key), shape=(size,))
    return jnp.asarray(qGEOMo(u, mu), dtype=jnp.int32)


def rJSUo(n, mu=0.0, sigma=1.0, nu=0.0, tau=1.0, key=None):
    """R rJSUo: random draws (pass key for reproducibility)."""
    size = _check_n(n)
    u = jax.random.uniform(_resolve_key(key), shape=(size,))
    return qJSUo(u, mu, sigma, nu, tau)


def rPARETO2o(n, mu=1.0, sigma=0.5, key=None):
    """R rPARETO2o: random draws (pass key for reproducibility)."""
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be positive")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    size = _check_n(n)
    u = jax.random.uniform(_resolve_key(key), shape=(size,))
    return qPARETO2o(u, mu, sigma)


def rSHASHo(n, mu=0.0, sigma=1.0, nu=0.0, tau=1.0, key=None):
    """R rSHASHo: random draws (pass key for reproducibility)."""
    size = _check_n(n)
    u = jax.random.uniform(_resolve_key(key), shape=(size,))
    return qSHASHo(u, mu, sigma, nu, tau)


__all__ = [
    # densities
    "dBCCGo",
    "dBCPEo",
    "dBCTo",
    "dBEo",
    "dGEOMo",
    "dJSUo",
    "dPARETO2o",
    "dSHASHo",
    # distribution functions
    "pBCCGo",
    "pBCPEo",
    "pBCTo",
    "pBEo",
    "pGEOMo",
    "pJSUo",
    "pPARETO2o",
    "pSHASHo",
    # quantiles
    "qBCCGo",
    "qBCPEo",
    "qBCTo",
    "qBEo",
    "qGEOMo",
    "qJSUo",
    "qPARETO2o",
    "qSHASHo",
    # random generation
    "rBCCGo",
    "rBCPEo",
    "rBCTo",
    "rBEo",
    "rGEOMo",
    "rJSUo",
    "rPARETO2o",
    "rSHASHo",
]
