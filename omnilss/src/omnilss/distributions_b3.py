"""GAMLSS distributions: Batch 3 — Heavy-tail / Skewed families.

Using JAX AD factory for:
- SHASH   (Sinh-Arcsinh, 4-parameter, Jones & Pewsey 2009)
- SHASHo  (Sinh-Arcsinh original parameterization, 4-parameter)
- SN1     (Skew Normal type 1, 3-parameter)
- SN2     (Skew-Normal type 2 / 'two-piece' Normal, 3-parameter)
- GT      (Generalised t, 4-parameter)
"""

from __future__ import annotations

# Enable float64 precision for numerical accuracy

from dataclasses import dataclass
import math
import numpy as np

import jax.numpy as jnp
from jax.scipy.special import gammaln
from jax.scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import skewnorm as scipy_skewnorm

from .ad import build_ad_family
from .families import FamilyDefinition
from .links import (
    identity_derivative,
    identity_inverse,
    identity_link,
    log_derivative,
    log_inverse,
    log_link,
)
from .dpqr_functions import dSHASH, pSHASH, qSHASH, rSHASH, dGT, pGT, qGT, rGT


def _broadcast_jax_args(args):
    arrays = [jnp.asarray(arg, dtype=jnp.float64) for arg in args]
    target_size = max(arr.size if arr.ndim > 0 else 1 for arr in arrays)
    scalar_output = target_size == 1 and all(arr.ndim == 0 for arr in arrays)
    prepared = []
    for arr in arrays:
        flat = jnp.reshape(arr, (-1,)) if arr.ndim > 0 else jnp.reshape(arr, (1,))
        if flat.size == target_size:
            prepared.append(flat)
        elif flat.size == 1:
            prepared.append(jnp.repeat(flat, target_size))
        else:
            raise ValueError("Arguments must be broadcastable to a common length")
    return prepared, scalar_output


def _shasho_p(x, mu=0.0, sigma=1.0, nu=0.0, tau=1.0, lower_tail=True, log_p=False):
    x, mu, sigma, nu, tau = [jnp.asarray(v, dtype=jnp.float64) for v in (x, mu, sigma, nu, tau)]
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    tau = jnp.maximum(tau, eps)
    z = (x - mu) / sigma
    r = jnp.sinh(tau * jnp.arcsinh(z) - nu)
    prob = norm.cdf(r)
    if not lower_tail:
        prob = 1.0 - prob
    return jnp.log(prob) if log_p else prob


def _shasho_q(p, mu=0.0, sigma=1.0, nu=0.0, tau=1.0, lower_tail=True, log_p=False):
    """Quantile of the original-parameterization sinh-arcsinh (``qSHASHo``).

    Boundary contract follows the vendored 6.1-8 source::

        q[p == 0] <- -Inf ; q[p == 1] <- Inf
        q[p <  0] <- NaN  ; q[p >  1] <- NaN

    while the *installed* 6.1.1 instead stops for any ``p`` outside the
    open interval (0, 1) -- the same install-vs-vendored split SHASHo2's
    quantile has (pinned on both sides in
    ``tests/test_r_consistency_shasho.py``).  We follow vendored: the
    support is the whole real line, so ±Inf are the continuous extensions
    at ``p = 0`` / ``p = 1``.  The previous implementation clipped ``p``
    into ``(eps, 1 - eps)``, which returned finite values there -- neither
    vendored's ±Inf nor live R's stop.
    """
    p, mu, sigma, nu, tau = [jnp.asarray(v, dtype=jnp.float64) for v in (p, mu, sigma, nu, tau)]
    if log_p:
        p = jnp.exp(p)
    if not lower_tail:
        p = 1.0 - p
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    tau = jnp.maximum(tau, eps)
    r = norm.ppf(p)
    value = mu + sigma * jnp.sinh((jnp.arcsinh(r) + nu) / tau)
    value = jnp.where(p == 0.0, -jnp.inf, value)
    value = jnp.where(p == 1.0, jnp.inf, value)
    value = jnp.where((p < 0.0) | (p > 1.0), jnp.nan, value)
    return value


def _shasho_r(key, n, mu=0.0, sigma=1.0, nu=0.0, tau=1.0):
    """Inverse-transform sampler -- R's ``rSHASHo`` construction exactly.

    R's source is ``p <- runif(n); qSHASHo(p, mu, sigma, nu, tau)``; using
    the same construction here (instead of a dedicated sampler) is what
    makes sampler/CDF drift structurally impossible.  ``runif`` returns
    [0, 1), so a draw can land on exactly 0 -- which maps to -Inf, the
    same as R's own behaviour.
    """
    import jax.random as random

    return _shasho_q(
        random.uniform(key, shape=(int(n),), dtype=jnp.float64), mu, sigma, nu, tau
    )


def _sn1_p(x, mu=0.0, sigma=1.0, nu=0.0, lower_tail=True, log_p=False):
    args, scalar_output = _broadcast_jax_args((x, mu, sigma, nu))
    vals = scipy_skewnorm.cdf(np.asarray(args[0]), np.asarray(args[3]), loc=np.asarray(args[1]), scale=np.asarray(args[2]))
    vals = 1.0 - vals if not lower_tail else vals
    vals = np.log(vals) if log_p else vals
    if scalar_output:
        return jnp.asarray(vals[0], dtype=jnp.float64)
    return jnp.asarray(vals, dtype=jnp.float64)


def _sn1_q(p, mu=0.0, sigma=1.0, nu=0.0, lower_tail=True, log_p=False):
    args, scalar_output = _broadcast_jax_args((p, mu, sigma, nu))
    probs = np.asarray(args[0])
    if log_p:
        probs = np.exp(probs)
    if not lower_tail:
        probs = 1.0 - probs
    probs = np.clip(probs, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    vals = scipy_skewnorm.ppf(probs, np.asarray(args[3]), loc=np.asarray(args[1]), scale=np.asarray(args[2]))
    if scalar_output:
        return jnp.asarray(vals[0], dtype=jnp.float64)
    return jnp.asarray(vals, dtype=jnp.float64)


def _sn1_r(key, n, mu=0.0, sigma=1.0, nu=0.0):
    """Generate SN1 variates using inverse-CDF sampling."""
    import jax.random as random

    return _sn1_q(random.uniform(key, shape=(n,), dtype=jnp.float64), mu, sigma, nu)


def _attach_numeric_real_line_dpq(family_class: type[FamilyDefinition], log_pdf_func):
    """Attach simple numerical d/p/q helpers for continuous real-line families."""

    def _broadcast_numpy_args(args):
        arrays = [np.asarray(arg, dtype=float) for arg in args]
        target_size = max(arr.size if arr.ndim > 0 else 1 for arr in arrays)
        scalar_output = target_size == 1 and all(arr.ndim == 0 for arr in arrays)

        prepared = []
        for arr in arrays:
            flat = arr.reshape(-1) if arr.ndim > 0 else arr.reshape(1)
            if flat.size == target_size:
                prepared.append(flat)
            elif flat.size == 1:
                prepared.append(np.repeat(flat, target_size))
            else:
                raise ValueError(
                    "Arguments must be broadcastable to a common length; "
                    f"got sizes {[a.size if a.ndim > 0 else 1 for a in arrays]}"
                )
        return prepared, scalar_output

    def _pdf_scalar(x, params):
        args = [jnp.asarray(float(x), dtype=jnp.float64)]
        args.extend(jnp.asarray(float(param), dtype=jnp.float64) for param in params)
        return float(jnp.exp(log_pdf_func(*args)))

    def _cdf_scalar(x, params):
        x = float(x)
        lo, hi = _guess_bounds(params)
        if x <= lo:
            return 0.0
        if x >= hi:
            return 1.0

        def safe_pdf(t):
            value = _pdf_scalar(t, params)
            return value if np.isfinite(value) else 0.0

        numerator = quad(safe_pdf, lo, x, limit=200)[0]
        denominator = quad(safe_pdf, lo, hi, limit=200)[0]
        if not np.isfinite(denominator) or denominator <= 0.0:
            return np.nan
        return numerator / denominator

    def _guess_bounds(params):
        mu = float(params[0]) if params else 0.0
        scale = abs(float(params[1])) if len(params) > 1 else 1.0
        scale = max(scale, 1.0)
        return mu - 8.0 * scale - 8.0, mu + 8.0 * scale + 8.0

    def _d(self, x, *params):
        args, scalar_output = _broadcast_numpy_args((x, *params))
        values = np.asarray(
            [_pdf_scalar(args[0][i], [arg[i] for arg in args[1:]]) for i in range(args[0].size)],
            dtype=float,
        )
        if scalar_output:
            return values[0]
        return jnp.asarray(values, dtype=jnp.float64)

    def _p(self, x, *params):
        args, scalar_output = _broadcast_numpy_args((x, *params))
        values = np.asarray(
            [_cdf_scalar(args[0][i], [arg[i] for arg in args[1:]]) for i in range(args[0].size)],
            dtype=float,
        )
        values = np.clip(values, 0.0, 1.0)
        if scalar_output:
            return values[0]
        return jnp.asarray(values, dtype=jnp.float64)

    def _q(self, probs, *params):
        args, scalar_output = _broadcast_numpy_args((probs, *params))
        outputs = []
        for i in range(args[0].size):
            prob = float(np.clip(args[0][i], 0.0, 1.0))
            param_values = [arg[i] for arg in args[1:]]
            if prob <= 0.0:
                outputs.append(-np.inf)
                continue
            if prob >= 1.0:
                outputs.append(np.inf)
                continue

            lo, hi = _guess_bounds(param_values)
            cdf_lo = _cdf_scalar(lo, param_values)
            cdf_hi = _cdf_scalar(hi, param_values)
            expand = 0
            while cdf_lo > prob and expand < 12:
                lo -= max(1.0, abs(lo - hi))
                cdf_lo = _cdf_scalar(lo, param_values)
                expand += 1
            while cdf_hi < prob and expand < 24:
                hi += max(1.0, abs(hi - lo))
                cdf_hi = _cdf_scalar(hi, param_values)
                expand += 1

            try:
                outputs.append(brentq(lambda value: _cdf_scalar(value, param_values) - prob, lo, hi, maxiter=200))
            except ValueError:
                mu = float(param_values[0]) if param_values else 0.0
                scale = max(abs(float(param_values[1])) if len(param_values) > 1 else 1.0, 1e-6)
                outputs.append(mu + scale * norm.ppf(prob))

        values = np.asarray(outputs, dtype=float)
        if scalar_output:
            return values[0]
        return jnp.asarray(values, dtype=jnp.float64)

    setattr(family_class, "d", _d)
    setattr(family_class, "p", _p)
    setattr(family_class, "q", _q)


# ------------------------------------------------------------------
# 1. SHASH — Sinh-Arcsinh distribution (Jones & Pewsey 2009)
#    4-parameter: mu (loc), sigma (scale), nu (skew), tau (kurtosis)
#    R: z = (x-mu)/sigma
#       r = 0.5*(exp(tau*asinh(z)) - exp(-nu*asinh(z)))
#       c = 0.5*(tau*exp(tau*asinh(z)) + nu*exp(-nu*asinh(z)))
#       loglik = -log(sigma) - log(2pi)/2 - log(1+z^2)/2 + log(c) - r^2/2
# ------------------------------------------------------------------

@dataclass(frozen=True)
class SinhArcsinhFamily(FamilyDefinition):
    """Sinh-Arcsinh distribution (`SHASH`)."""


def _shash_log_pdf(y, mu, sigma, nu, tau):
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    tau = jnp.maximum(tau, eps)
    z = (y - mu) / sigma
    asinh_z = jnp.arcsinh(z)
    r = 0.5 * (jnp.exp(tau * asinh_z) - jnp.exp(-nu * asinh_z))
    c = 0.5 * (tau * jnp.exp(tau * asinh_z) + nu * jnp.exp(-nu * asinh_z))
    return -jnp.log(sigma) - 0.5 * math.log(2.0 * math.pi) - 0.5 * jnp.log1p(z**2) + jnp.log(jnp.maximum(c, eps)) - 0.5 * r**2


def SHASH() -> SinhArcsinhFamily:
    _attach_numeric_real_line_dpq(SinhArcsinhFamily, _shash_log_pdf)
    return build_ad_family(
        family_class=SinhArcsinhFamily,
        name="SHASH",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_shash_log_pdf,
        type_="Continuous",
        links={"mu": "identity", "sigma": "log", "nu": "log", "tau": "log"},
        link_functions={"mu": identity_link, "sigma": log_link, "nu": log_link, "tau": log_link},
        link_inverses={"mu": identity_inverse, "sigma": log_inverse, "nu": log_inverse, "tau": log_inverse},
        link_derivatives={"mu": identity_derivative, "sigma": log_derivative, "nu": log_derivative, "tau": log_derivative},
        d=dSHASH,
        p=pSHASH,
        q=qSHASH,
        r=rSHASH,
    )


# ------------------------------------------------------------------
# 2. SHASHo — Sinh-Arcsinh (original parameterization)
#    R: z=(x-mu)/sigma, c=cosh(tau*asinh(z)-nu), r=sinh(tau*asinh(z)-nu)
#       loglik = -log(sigma)+log(tau)-log(2pi)/2 -log(1+z^2)/2 + log(c) - r^2/2
# ------------------------------------------------------------------

@dataclass(frozen=True)
class SinhArcsinhOFamily(FamilyDefinition):
    """Sinh-Arcsinh distribution (original parameterization) (`SHASHo`)."""


def _shasho_log_pdf(y, mu, sigma, nu, tau):
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    tau = jnp.maximum(tau, eps)
    z = (y - mu) / sigma
    arg = tau * jnp.arcsinh(z) - nu
    c = jnp.cosh(arg)
    r = jnp.sinh(arg)
    return -jnp.log(sigma) + jnp.log(tau) - 0.5 * math.log(2.0 * math.pi) - 0.5 * jnp.log1p(z**2) + jnp.log(jnp.maximum(c, eps)) - 0.5 * r**2


def SHASHo() -> SinhArcsinhOFamily:
    return build_ad_family(
        family_class=SinhArcsinhOFamily,
        name="SHASHo",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_shasho_log_pdf,
        type_="Continuous",
        links={"mu": "identity", "sigma": "log", "nu": "identity", "tau": "log"},
        link_functions={"mu": identity_link, "sigma": log_link, "nu": identity_link, "tau": log_link},
        link_inverses={"mu": identity_inverse, "sigma": log_inverse, "nu": identity_inverse, "tau": log_inverse},
        link_derivatives={"mu": identity_derivative, "sigma": log_derivative, "nu": identity_derivative, "tau": log_derivative},
        p=_shasho_p,
        q=_shasho_q,
        r=_shasho_r,
    )


# ------------------------------------------------------------------
# 3. SN2 — Two-piece Normal / Skew-Normal type 2
#    R: loglik = ifelse(x<mu, -0.5*(nu*|z|)^2, -0.5*(|z|/nu)^2)
#                -log(sigma)+log(nu)-log(1+nu^2) - 0.5*log(2) - lgamma(3/2)
# ------------------------------------------------------------------

@dataclass(frozen=True)
class SkewNormal2Family(FamilyDefinition):
    """Two-piece Skew-Normal distribution (`SN2`)."""


def _sn2_log_pdf(y, mu, sigma, nu):
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.maximum(nu, eps)
    z = (y - mu) / sigma
    loglik_left = -0.5 * jnp.square(nu * jnp.abs(z))
    loglik_right = -0.5 * jnp.square(jnp.abs(z) / nu)
    loglik = jnp.where(y < mu, loglik_left, loglik_right)
    return loglik - jnp.log(sigma) + jnp.log(nu) - jnp.log(1.0 + jnp.square(nu)) - 0.5 * math.log(2.0) - gammaln(1.5)


def _sn2_p(x, mu=0.0, sigma=1.0, nu=1.0, lower_tail=True, log_p=False):
    """CDF of the two-piece normal, matching R ``pSN2``."""
    x, mu, sigma, nu = [jnp.asarray(v, dtype=jnp.float64) for v in (x, mu, sigma, nu)]
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.maximum(nu, eps)
    z = (x - mu) / sigma
    left = 2.0 * norm.cdf(nu * z) / (1.0 + nu * nu)
    # The left-hand mass is 1/(1+nu^2); preserve it at the join x=mu.
    right = (1.0 - nu * nu + 2.0 * nu * nu * norm.cdf(z / nu)) / (1.0 + nu * nu)
    result = jnp.where(x < mu, left, right)
    # R's implementation also bounds the numerical CDF to its probability
    # domain when the limiting ``nu=0`` smoke-test value is supplied.
    result = jnp.clip(result, 0.0, 1.0)
    result = jnp.where(lower_tail, result, 1.0 - result)
    return jnp.log(result) if log_p else result


def _sn2_q(p, mu=0.0, sigma=1.0, nu=1.0, lower_tail=True, log_p=False):
    """Quantile of the two-piece normal, matching R ``qSN2``."""
    p, mu, sigma, nu = [jnp.asarray(v, dtype=jnp.float64) for v in (p, mu, sigma, nu)]
    if log_p:
        p = jnp.exp(p)
    p = jnp.where(lower_tail, p, 1.0 - p)
    eps = jnp.finfo(jnp.float64).eps
    p = jnp.clip(p, eps, 1.0 - eps)
    nu = jnp.maximum(nu, eps)
    threshold = 1.0 / (1.0 + nu * nu)
    left_p = p * (1.0 + nu * nu) / 2.0
    right_p = (p * (1.0 + nu * nu) - (1.0 - nu * nu)) / (2.0 * nu * nu)
    z = jnp.where(p < threshold, norm.ppf(left_p) / nu, nu * norm.ppf(right_p))
    return mu + sigma * z


def _sn2_r(key, n, mu=0.0, sigma=1.0, nu=1.0):
    """Generate SN2 variates using inverse-CDF sampling."""
    import jax.random as random

    return _sn2_q(random.uniform(key, shape=(n,), dtype=jnp.float64), mu, sigma, nu)


def SN2() -> SkewNormal2Family:
    return build_ad_family(
        family_class=SkewNormal2Family,
        name="SN2",
        parameters=("mu", "sigma", "nu"),
        log_pdf_func=_sn2_log_pdf,
        type_="Continuous",
        links={"mu": "identity", "sigma": "log", "nu": "log"},
        link_functions={"mu": identity_link, "sigma": log_link, "nu": log_link},
        link_inverses={"mu": identity_inverse, "sigma": log_inverse, "nu": log_inverse},
        link_derivatives={"mu": identity_derivative, "sigma": log_derivative, "nu": log_derivative},
        p=_sn2_p,
        q=_sn2_q,
        r=_sn2_r,
    )


# ------------------------------------------------------------------
# 4. GT — Generalised t distribution (McDonald & Newey, 1988)
#    4-parameter: mu, sigma, nu (df), tau (shape exponent)
#    R: zt=|z|^tau
#       loglik = log(tau) - log(2σ) - (1/tau)*log(nu) - lgamma(1/tau) - lgamma(nu)
#               + lgamma(nu + 1/tau) - (nu+1/tau)*log(1 + zt/nu)
# ------------------------------------------------------------------

@dataclass(frozen=True)
class GeneralisedTFamily(FamilyDefinition):
    """Generalised t distribution (`GT`)."""


def _gt_log_pdf(y, mu, sigma, nu, tau):
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    nu = jnp.maximum(nu, eps)
    tau = jnp.maximum(tau, eps)
    z = (y - mu) / sigma
    zt = jnp.power(jnp.abs(z), tau)
    return (
        jnp.log(tau)
        - jnp.log(2.0 * sigma)
        - (1.0 / tau) * jnp.log(nu)
        - gammaln(1.0 / tau)
        - gammaln(nu)
        + gammaln(nu + 1.0 / tau)
        - (nu + 1.0 / tau) * jnp.log1p(zt / nu)
    )


def GT() -> GeneralisedTFamily:
    _attach_numeric_real_line_dpq(GeneralisedTFamily, _gt_log_pdf)
    # _attach_numeric_real_line_dpq sets instance methods (with self).
    # Wrap them as standalone functions for build_ad_family.
    _p_method = GeneralisedTFamily.p
    _q_method = GeneralisedTFamily.q

    def _p_standalone(x, *params, lower_tail=True, log_p=False):
        return _p_method(None, x, *params)

    def _q_standalone(probs, *params, lower_tail=True, log_p=False):
        return _q_method(None, probs, *params)

    return build_ad_family(
        family_class=GeneralisedTFamily,
        name="GT",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_gt_log_pdf,
        type_="Continuous",
        links={"mu": "identity", "sigma": "log", "nu": "log", "tau": "log"},
        link_functions={"mu": identity_link, "sigma": log_link, "nu": log_link, "tau": log_link},
        link_inverses={"mu": identity_inverse, "sigma": log_inverse, "nu": log_inverse, "tau": log_inverse},
        link_derivatives={"mu": identity_derivative, "sigma": log_derivative, "nu": log_derivative, "tau": log_derivative},
        d=dGT,
        p=_p_standalone,
        q=_q_standalone,
        r=rGT,
    )


# ------------------------------------------------------------------
# 5. SN1 — Skew Normal type 1 (Azzalini 1985)
#    R uses an exact but complex gamma-based CDF formulation.
#    We use the standard JAX norm CDF approach: logpdf = log(2) + norm.logpdf(z) + norm.logcdf(nu*z)
# ------------------------------------------------------------------

@dataclass(frozen=True)
class SkewNormal1Family(FamilyDefinition):
    """Skew-Normal type 1 distribution (`SN1`)."""


def _sn1_log_pdf(y, mu, sigma, nu):
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    z = (y - mu) / sigma
    # Standard SN1: f(z) = 2 * phi(z) * Phi(nu*z), log-version:
    return math.log(2.0) - jnp.log(sigma) + norm.logpdf(z) + norm.logcdf(nu * z)


def SN1() -> SkewNormal1Family:
    return build_ad_family(
        family_class=SkewNormal1Family,
        name="SN1",
        parameters=("mu", "sigma", "nu"),
        log_pdf_func=_sn1_log_pdf,
        type_="Continuous",
        links={"mu": "identity", "sigma": "log", "nu": "identity"},
        link_functions={"mu": identity_link, "sigma": log_link, "nu": identity_link},
        link_inverses={"mu": identity_inverse, "sigma": log_inverse, "nu": identity_inverse},
        link_derivatives={"mu": identity_derivative, "sigma": log_derivative, "nu": identity_derivative},
        p=_sn1_p,
        q=_sn1_q,
        r=_sn1_r,
    )
