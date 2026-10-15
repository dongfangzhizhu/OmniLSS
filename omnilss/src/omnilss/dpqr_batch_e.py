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

Batch 2 (PE2 / NOF / GAF) adds two more documented deviations:

* ``qNOF`` calls ``qnorm`` without forwarding ``log.p``, so R silently treats a
  log-scale probability as if it were on the probability scale.  We honour
  ``log_p``.
* ``dGAF`` applies ``ifelse(x <= 0, 0, fy)`` *after* the ``log`` branch, so R's
  log-density for ``x <= 0`` is ``0`` (i.e. density 1).  We return ``-inf``.

Batch 3 (PARETO1o, the fixed-mu Pareto Type 1) adds four, all verified on live
R 6.1.1 against the vendored 6.1-8 sources:

* ``pPARETO1o`` ignores ``log.p`` (the same ``cdf < - log(cdf)`` typo, still
  present in the installed package).  We honour it.
* ``pPARETO1o`` applies ``ifelse(q <= mu, 0, cdf)`` *after* the
  ``lower.tail`` flip, so R's upper tail is ``0`` at and below the support
  edge.  We return the mathematically correct ``1``.
* ``dPARETO1o`` returns ``0`` on the log scale outside the support (the
  ``dGAF`` pattern again).  We return ``-inf``; the family's ``g_dev_inc``
  reproduces R's ``-2 * 0`` on that path.
* Live ``qPARETO1o`` *stops* for ``p`` outside ``(0, 1)``; the vendored
  source instead assigns ``q[p == 0] <- 0``, contradicting its own support
  (infimum ``mu``).  We return the support infimum at ``p <= 0``.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpy as np

#: Support endpoints used when a probability lands exactly on 0 or 1.
_INF = jnp.inf
_NAN = jnp.nan

# Inventory of R ``gamlss.dist`` quirks we deliberately do **not** reproduce.
# Kept next to the kernels so the docs and the tests stay in sync:
# ``tests/test_dpqr_batch_e.py::test_r_quirk_inventory_is_explicit`` asserts that
# every family listed here is part of the parity sweep.
R_QUIRKS: dict[str, str] = {
    "PARETO": "pPARETO 忽略 log.p（R 源码 `cdf < - log(cdf)` 笔误）；qPARETO 忽略 lower.tail",
    "PARETO1": "pPARETO1 忽略 log.p（同上笔误）；qPARETO1 忽略 lower.tail",
    "PARETO1o": (
        "pPARETO1o 忽略 log.p（同一笔误，活 R 6.1.1 实测仍在）；"
        "pPARETO1o 的 lower.tail 翻转之后才 ifelse(q<=mu, 0, ...)，上尾在 support 边缘被 clamp 到 0（数学上为 1）；"
        "dPARETO1o 的 log 密度在 support 外返回 0（ifelse 在 log 分支之后）；"
        "活 R qPARETO1o 对 p∉(0,1) stop，vendored 的 q[p==0]<-0 与自身 support（下确界 mu）矛盾"
    ),
    "NOF": "qNOF 未把 log.p 传给 qnorm",
    "GAF": "dGAF 在 log 分支之后才做 ifelse(x<=0, 0, fy)，导致 log 密度返回 0",
}


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


def _lgamma(value):
    """``gamlss``' ``lgamma`` / R's ``lgamma``."""
    from jax.scipy.special import gammaln

    return gammaln(value)


def _pgamma_std(x, shape):
    """R's ``pgamma(x, shape=shape, scale=1)`` (regularised lower incomplete gamma)."""
    from jax.scipy.special import gammainc

    return gammainc(shape, x)


def _qgamma_std(prob, shape):
    """R's ``qgamma(prob, shape=shape, scale=1)``.

    JAX has no incomplete-gamma inverse, so this falls back to SciPy — the same
    route ``dpqr_functions`` already uses for BCPE.  SciPy's ``gammaincinv``
    agrees with R's ``qgamma`` to ~1e-14 relative, well inside the 1e-8 parity
    tolerance used by the R comparison tests.
    """
    from scipy.special import gammaincinv

    prob_np = np.asarray(prob, dtype=np.float64)
    shape_np = np.asarray(shape, dtype=np.float64)
    return jnp.asarray(gammaincinv(shape_np, prob_np), dtype=jnp.float64)


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
# PARETO1o - Pareto Type 1 with mu FIXED (2-parameter, R PARETO1_mu_fixed.R)
# ---------------------------------------------------------------------------
#
# R's family object marks ``mu`` non-estimable (``parameters = list(mu = FALSE,
# sigma = TRUE)``) and fixes it at ``mu.initial = rep(min(y), length(y))``;
# only ``sigma`` is fitted.  The d/p/q/r functions still take ``mu`` as a
# regular positional argument -- "fixed" is a *fitting-level* property, the
# same split DBI's ``bd`` and NET's ``nu``/``tau`` already have.


def dPARETO1o(x, mu=1.0, sigma=0.5, log=False):
    """Density of the fixed-mu Pareto Type 1 (support ``y > mu``).

    R's ``dPARETO1o`` computes ``lfy`` unconditionally and only then applies
    ``fy <- ifelse(x <= mu, 0, fy)`` -- *after* the ``log`` branch.  So on the
    log scale R returns ``0`` (not ``-Inf``) outside the support, the same
    quirk ``dGAF`` has (see ``R_QUIRKS``).  We return ``-inf``; the family's
    ``g_dev_inc`` reproduces R's ``-2 * 0`` separately (see
    ``distributions_b16._pareto1o_log_pdf``).
    """
    x, mu, sigma = jnp.broadcast_arrays(
        jnp.asarray(x, dtype=jnp.float64),
        jnp.asarray(mu, dtype=jnp.float64),
        jnp.asarray(sigma, dtype=jnp.float64),
    )
    inside = (x > mu) & (mu > 0.0) & (sigma > 0.0)
    xs = jnp.where(inside, x, 2.0)
    value = jnp.where(
        inside,
        jnp.log(sigma) + sigma * jnp.log(mu) - (sigma + 1.0) * jnp.log(xs),
        -_INF,
    )
    value = jnp.where((mu > 0.0) & (sigma > 0.0), value, _NAN)
    return value if log else jnp.exp(value)


def pPARETO1o(q, mu=1.0, sigma=0.5, lower_tail=True, log_p=False):
    """CDF of the fixed-mu Pareto Type 1 (support ``y > mu``).

    ``cdf = 1 - (mu/q)^sigma`` evaluated as ``-expm1(sigma*log(mu/q))`` so the
    cancellation near ``q = mu`` keeps full relative precision (R's literal
    ``1 - (mu/q)^sigma`` loses ~9 digits a decade inside the boundary).

    R applies ``ifelse(q <= mu, 0, cdf)`` *after* the ``lower.tail`` flip, so
    R's upper tail is 0 (not 1) at and below the support edge; we return the
    mathematically correct 1 there (documented in ``R_QUIRKS``).  R also
    ignores ``log.p`` outright (the ``cdf < - log(cdf)`` typo, confirmed on
    live R 6.1.1); we honour it.
    """
    q, mu, sigma = jnp.broadcast_arrays(
        jnp.asarray(q, dtype=jnp.float64),
        jnp.asarray(mu, dtype=jnp.float64),
        jnp.asarray(sigma, dtype=jnp.float64),
    )
    valid = (mu > 0.0) & (sigma > 0.0)
    inside = (q > mu) & valid
    qs = jnp.where(inside, q, 2.0)
    mus = jnp.where(valid, mu, 1.0)
    cdf = jnp.where(inside, -jnp.expm1(sigma * (jnp.log(mus) - jnp.log(qs))), 0.0)
    value = _apply_tail(cdf, lower_tail, log_p)
    return jnp.where(valid, value, _NAN)


def qPARETO1o(p, mu=1.0, sigma=0.5, lower_tail=True, log_p=False):
    """Quantile of the fixed-mu Pareto Type 1 (support ``y > mu``).

    Live R 6.1.1 *stops* for any ``p`` outside the open interval (0, 1)
    (``p must be between 0 and 1``); the vendored 6.1-8 source instead
    assigns ``q[p == 0] <- 0`` -- which contradicts its own support, whose
    infimum is ``mu``, not 0 -- plus ``Inf`` at 1 and ``NaN`` outside
    [0, 1].  We keep the house convention (``qPARETO`` above): the support
    infimum at ``p <= 0``, ``Inf`` at ``p >= 1``, ``NaN`` for ``NaN``.
    """
    p, mu, sigma = jnp.broadcast_arrays(
        jnp.asarray(p, dtype=jnp.float64),
        jnp.asarray(mu, dtype=jnp.float64),
        jnp.asarray(sigma, dtype=jnp.float64),
    )
    prob = _prob_from_p(p, lower_tail, log_p)

    def value_fn(pr):
        return mu * jnp.power(1.0 - pr, -1.0 / sigma)

    value = _quantile_from_prob(prob, value_fn, low=mu, high=_INF)
    return jnp.where((mu > 0.0) & (sigma > 0.0), value, _NAN)


def rPARETO1o(key, n, mu=1.0, sigma=0.5):
    """Inverse-transform sampler for the fixed-mu Pareto Type 1."""
    return qPARETO1o(jr.uniform(key, (int(n),), dtype=jnp.float64), mu, sigma)


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


# ---------------------------------------------------------------------------
# PE2 - Power Exponential type 2 (R: PE2.R)
#   z = (x-mu)/sigma, s = |z|^nu
#   cdf = 0.5*(1 + pgamma(s, shape=1/nu, scale=1)*sign(z))
# ---------------------------------------------------------------------------

#: R replaces the gamma CDF by the uniform limit above this nu.
_PE2_NU_LIMIT = 10000.0


def _pe2_s(x, mu, sigma, nu):
    return jnp.power(jnp.abs((x - mu) / sigma), nu)


def dPE2(x, mu=0.0, sigma=1.0, nu=2.0, log=False):
    """Density of the Power Exponential type 2 distribution."""
    x, mu, sigma, nu = jnp.broadcast_arrays(
        jnp.asarray(x, dtype=jnp.float64),
        jnp.asarray(mu, dtype=jnp.float64),
        jnp.asarray(sigma, dtype=jnp.float64),
        jnp.asarray(nu, dtype=jnp.float64),
    )
    valid = (sigma > 0.0) & (nu > 0.0)
    safe_sigma = jnp.where(sigma > 0.0, sigma, 1.0)
    safe_nu = jnp.where(nu > 0.0, nu, 1.0)
    value = (
        -jnp.log(safe_sigma)
        + jnp.log(safe_nu)
        - jnp.log(2.0)
        - _pe2_s(x, mu, safe_sigma, safe_nu)
        - _lgamma(1.0 / safe_nu)
    )
    value = jnp.where(valid, value, _NAN)
    return value if log else jnp.exp(value)


def pPE2(q, mu=0.0, sigma=1.0, nu=2.0, lower_tail=True, log_p=False):
    """CDF of the Power Exponential type 2 distribution."""
    q, mu, sigma, nu = jnp.broadcast_arrays(
        jnp.asarray(q, dtype=jnp.float64),
        jnp.asarray(mu, dtype=jnp.float64),
        jnp.asarray(sigma, dtype=jnp.float64),
        jnp.asarray(nu, dtype=jnp.float64),
    )
    valid = (sigma > 0.0) & (nu > 0.0)
    safe_sigma = jnp.where(sigma > 0.0, sigma, 1.0)
    safe_nu = jnp.where(nu > 0.0, nu, 1.0)
    z = (q - mu) / safe_sigma
    cdf = 0.5 * (1.0 + _pgamma_std(_pe2_s(q, mu, safe_sigma, safe_nu), 1.0 / safe_nu) * jnp.sign(z))
    # R's uniform limit for nu > 10000 (PE2.R).
    cdf = jnp.where(nu > _PE2_NU_LIMIT, (q - (mu - safe_sigma)) / (2.0 * safe_sigma), cdf)
    value = _apply_tail(cdf, lower_tail, log_p)
    return jnp.where(valid, value, _NAN)


def qPE2(p, mu=0.0, sigma=1.0, nu=2.0, lower_tail=True, log_p=False):
    """Quantile of the Power Exponential type 2 distribution."""
    p, mu, sigma, nu = jnp.broadcast_arrays(
        jnp.asarray(p, dtype=jnp.float64),
        jnp.asarray(mu, dtype=jnp.float64),
        jnp.asarray(sigma, dtype=jnp.float64),
        jnp.asarray(nu, dtype=jnp.float64),
    )
    valid = (sigma > 0.0) & (nu > 0.0)
    safe_sigma = jnp.where(sigma > 0.0, sigma, 1.0)
    safe_nu = jnp.where(nu > 0.0, nu, 1.0)
    prob = _prob_from_p(p, lower_tail, log_p)

    def value_fn(pr):
        sign_p = jnp.sign(pr - 0.5)
        arg = (2.0 * pr - 1.0) * sign_p
        s = _qgamma_std(arg, 1.0 / safe_nu)
        return mu + safe_sigma * sign_p * jnp.power(s, 1.0 / safe_nu)

    value = _quantile_from_prob(prob, value_fn, low=-_INF, high=_INF)
    return jnp.where(valid, value, _NAN)


def rPE2(key, n, mu=0.0, sigma=1.0, nu=2.0):
    """Inverse-transform sampler for the Power Exponential type 2 distribution."""
    return qPE2(jr.uniform(key, (int(n),), dtype=jnp.float64), mu, sigma, nu)


# ---------------------------------------------------------------------------
# NOF - normal family with variance function sigma^2 * |mu|^nu (R: NOF-13-3-13.R)
#   sigma1 = sigma * |mu|^(nu/2);  y ~ Normal(mu, sigma1)
# ---------------------------------------------------------------------------


def _nof_sd(mu, sigma, nu):
    return sigma * jnp.power(jnp.abs(mu), 0.5 * nu)


def dNOF(x, mu=0.0, sigma=1.0, nu=0.0, log=False):
    """Density of the NOF (normal family) distribution."""
    x, mu, sigma, nu = jnp.broadcast_arrays(
        jnp.asarray(x, dtype=jnp.float64),
        jnp.asarray(mu, dtype=jnp.float64),
        jnp.asarray(sigma, dtype=jnp.float64),
        jnp.asarray(nu, dtype=jnp.float64),
    )
    sd = _nof_sd(mu, sigma, nu)
    safe = jnp.where(sd > 0.0, sd, 1.0)
    z = (x - mu) / safe
    value = -0.5 * jnp.log(2.0 * np.pi) - jnp.log(safe) - 0.5 * z * z
    # R: dnorm(x, mean=mu, sd=0) is a point mass at mu.
    value = jnp.where(sd > 0.0, value, jnp.where(x == mu, _INF, -_INF))
    value = jnp.where(sigma > 0.0, value, _NAN)
    return value if log else jnp.exp(value)


def pNOF(q, mu=0.0, sigma=1.0, nu=0.0, lower_tail=True, log_p=False):
    """CDF of the NOF (normal family) distribution."""
    q, mu, sigma, nu = jnp.broadcast_arrays(
        jnp.asarray(q, dtype=jnp.float64),
        jnp.asarray(mu, dtype=jnp.float64),
        jnp.asarray(sigma, dtype=jnp.float64),
        jnp.asarray(nu, dtype=jnp.float64),
    )
    sd = _nof_sd(mu, sigma, nu)
    safe = jnp.where(sd > 0.0, sd, 1.0)
    cdf = jnp.where(sd > 0.0, _norm_cdf((q - mu) / safe), jnp.where(q >= mu, 1.0, 0.0))
    value = _apply_tail(cdf, lower_tail, log_p)
    return jnp.where(sigma > 0.0, value, _NAN)


def qNOF(p, mu=0.0, sigma=1.0, nu=0.0, lower_tail=True, log_p=False):
    """Quantile of the NOF (normal family) distribution."""
    p, mu, sigma, nu = jnp.broadcast_arrays(
        jnp.asarray(p, dtype=jnp.float64),
        jnp.asarray(mu, dtype=jnp.float64),
        jnp.asarray(sigma, dtype=jnp.float64),
        jnp.asarray(nu, dtype=jnp.float64),
    )
    sd = _nof_sd(mu, sigma, nu)
    safe = jnp.where(sd > 0.0, sd, 1.0)
    prob = _prob_from_p(p, lower_tail, log_p)

    def value_fn(pr):
        return mu + safe * _norm_quantile(pr)

    value = _quantile_from_prob(prob, value_fn, low=-_INF, high=_INF)
    # Degenerate sd == 0 collapses the family onto the point mass at mu.
    value = jnp.where(sd > 0.0, value, mu)
    return jnp.where(sigma > 0.0, value, _NAN)


def rNOF(key, n, mu=0.0, sigma=1.0, nu=0.0):
    """Sampler for the NOF (normal family) distribution."""
    return qNOF(jr.uniform(key, (int(n),), dtype=jnp.float64), mu, sigma, nu)


# ---------------------------------------------------------------------------
# GAF - gamma family with variance function sigma^2 * mu^nu (R: GAF.R)
#   sigma1 = sigma * mu^(nu/2 - 1)
#   y ~ Gamma(shape = 1/sigma1^2, scale = mu*sigma1^2)
# ---------------------------------------------------------------------------


def _gaf_sigma1(mu, sigma, nu):
    return sigma * jnp.power(mu, 0.5 * nu - 1.0)


def _gaf_shape_scale(mu, sigma, nu):
    sigma1 = _gaf_sigma1(mu, sigma, nu)
    sigma1_sq = sigma1 * sigma1
    return sigma1, 1.0 / sigma1_sq, mu * sigma1_sq


def dGAF(x, mu=1.0, sigma=1.0, nu=2.0, log=False):
    """Density of the GAF (gamma family) distribution."""
    x, mu, sigma, nu = jnp.broadcast_arrays(
        jnp.asarray(x, dtype=jnp.float64),
        jnp.asarray(mu, dtype=jnp.float64),
        jnp.asarray(sigma, dtype=jnp.float64),
        jnp.asarray(nu, dtype=jnp.float64),
    )
    valid = (mu > 0.0) & (sigma > 0.0)
    safe_mu = jnp.where(mu > 0.0, mu, 1.0)
    safe_sigma = jnp.where(sigma > 0.0, sigma, 1.0)
    _sigma1, shape, scale = _gaf_shape_scale(safe_mu, safe_sigma, nu)
    xs = jnp.where(x > 0.0, x, 1.0)
    value = shape * jnp.log(xs / scale) - xs / scale - jnp.log(xs) - _lgamma(shape)
    # R returns 0 for x <= 0 even on the log scale; see R_QUIRKS["GAF"].
    value = jnp.where(x > 0.0, value, -_INF)
    value = jnp.where(valid, value, _NAN)
    return value if log else jnp.exp(value)


def pGAF(q, mu=1.0, sigma=1.0, nu=2.0, lower_tail=True, log_p=False):
    """CDF of the GAF (gamma family) distribution."""
    q, mu, sigma, nu = jnp.broadcast_arrays(
        jnp.asarray(q, dtype=jnp.float64),
        jnp.asarray(mu, dtype=jnp.float64),
        jnp.asarray(sigma, dtype=jnp.float64),
        jnp.asarray(nu, dtype=jnp.float64),
    )
    valid = (mu > 0.0) & (sigma > 0.0)
    safe_mu = jnp.where(mu > 0.0, mu, 1.0)
    safe_sigma = jnp.where(sigma > 0.0, sigma, 1.0)
    _sigma1, shape, scale = _gaf_shape_scale(safe_mu, safe_sigma, nu)
    z = jnp.where(q > 0.0, q / scale, 0.0)
    cdf = jnp.where(q > 0.0, _pgamma_std(z, shape), 0.0)
    value = _apply_tail(cdf, lower_tail, log_p)
    return jnp.where(valid, value, _NAN)


def qGAF(p, mu=1.0, sigma=1.0, nu=2.0, lower_tail=True, log_p=False):
    """Quantile of the GAF (gamma family) distribution."""
    p, mu, sigma, nu = jnp.broadcast_arrays(
        jnp.asarray(p, dtype=jnp.float64),
        jnp.asarray(mu, dtype=jnp.float64),
        jnp.asarray(sigma, dtype=jnp.float64),
        jnp.asarray(nu, dtype=jnp.float64),
    )
    valid = (mu > 0.0) & (sigma > 0.0)
    safe_mu = jnp.where(mu > 0.0, mu, 1.0)
    safe_sigma = jnp.where(sigma > 0.0, sigma, 1.0)
    _sigma1, shape, scale = _gaf_shape_scale(safe_mu, safe_sigma, nu)
    prob = _prob_from_p(p, lower_tail, log_p)

    def value_fn(pr):
        return _qgamma_std(pr, shape) * scale

    value = _quantile_from_prob(prob, value_fn, low=0.0, high=_INF)
    return jnp.where(valid, value, _NAN)


def rGAF(key, n, mu=1.0, sigma=1.0, nu=2.0):
    """Inverse-transform sampler for the GAF (gamma family) distribution."""
    return qGAF(jr.uniform(key, (int(n),), dtype=jnp.float64), mu, sigma, nu)
