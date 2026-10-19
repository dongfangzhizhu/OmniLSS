"""DBURR12 -- the discretised Burr XII distribution (R ``gamlss.dist``).

R reference: ``gamlss.dist/R/discBURR12.R`` -- ``DBURR12`` plus ``dDBURR12``,
``pDBURR12``, ``qDBURR12`` and ``rDBURR12``.

History
-------
This module used to be a *compatibility shell*: ``DBURR12()`` returned the
SICHEL kernel under the DBURR12 name.  The two are different distributions.
The discretised Burr XII has survival

    Sur(y) = (1 + (y/mu)^sigma)^(-nu),      f(y) = Sur(y) - Sur(y+1)

with ``nu > 0`` required and all three parameters on the ``log`` link, while
SICHEL accepts any real ``nu`` and links ``nu`` with ``identity``.  Measured
against live R 6.1.1 at ``(mu, sigma, nu) = (5, 2, 2)``::

    dDBURR12(0..3) = 0.07544379, 0.1813933, 0.20168,  0.2025055
    dSICHEL (0..3) = 0.01322800, 0.02948999, 0.04537722, 0.05832936

-- different densities, not a rescaling.  The parity scanner never caught the
mismatch because the shell inherited SICHEL's ``nu.link = identity``: the
probe grid contained negative ``nu``, live R refused it (``nu must be
greater than 0``) and gate 1 reported ``skip`` instead of ``fail`` (D2.44
finding).  Declaring R's own links makes the same gate probe ``nu > 0`` and
actually compare values.

Two R contracts worth pinning (both measured against live R)
------------------------------------------------------------
1. ``dDBURR12`` does **not** enforce an integer argument: ``d(1.5)`` returns
   ``Sur(1.5) - Sur(2.5)``.  A discrete PMF has no mass there, so our kernel
   returns 0 (``-inf`` logged).  Same precedent as PIG's non-integer counts:
   reproduce R in-domain, do not reproduce it off-support.  Two more R
   quirks in the same family, both clamps applied *after* the log / tail
   transform: ``d`` sets ``prob[x < 0] <- 0`` so R's logged density at a
   negative count is ``0`` (ours is ``-inf``, the honest log of a zero
   mass), and ``p`` sets ``cdf[q < 0] <- 0`` after the upper-tail flip, so
   R's survival at ``q < 0`` is ``0`` where it should be ``1`` (ours is 1).
   Both are pinned in ``test_r_consistency_dburr12``.
2. ``qDBURR12`` differs between the vendored source and the install -- the
   same install-vs-vendored split PARETO1o, SHASHo and SHASHo2 have.
   Vendored ``discBURR12.R`` (6.1-8) assigns
   ``trunc(w - 1)`` with ``q[p==0] <- 0; q[p==1] <- Inf;
   q[p<0 | p>1] <- NaN`` and checks nothing; the **installed 6.1.1**
   (dumped from the namespace) is

       if (any(p <= 0) | any(p >= 1)) stop("p must be between 0 and 1")
       q <- round(w - 1);  q <- ifelse(q < 0, 0, q)

   with ``w = mu * ((1-p)^(-1/nu) - 1)^(1/sigma)``.  ``round`` is
   round-half-to-even, and it is *not* the CDF inverse --
   ``p = 0.25`` at the default profile gives ``w - 1 = 0.9666``, which R
   rounds to 1 while the smallest count with ``F >= 0.25`` is 1; but
   ``p = 0.5`` gives ``w - 1 = 2.218``, rounded to 2 while ``F(2) = 0.4593``
   is still below 0.5.  We reproduce the installed body (round + clamp),
   which is the family's real behaviour in a ``gamlss`` fit, and take the
   house boundary convention (``p <= 0`` -> 0, ``p >= 1`` -> ``Inf``,
   otherwise NaN); the tests pin R's stop.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .ad import _prepare_jax_args, build_ad_family
from .families import FamilyDefinition
from .links import log_derivative, log_inverse, log_link

#: R's argument defaults: ``dDBURR12(x, mu = 5, sigma = 2, nu = 2)``.
DEFAULT_MU = 5.0
DEFAULT_SIGMA = 2.0
DEFAULT_NU = 2.0


def _dburr12_parameters(mu, sigma, nu):
    """Broadcast and validate, returning safe stand-ins for invalid entries.

    R stops outright on ``mu/sigma/nu <= 0``; the library convention is to
    return NaN for invalid parameters, so the invalid entries are masked out
    by the caller instead of raising.

    ``jnp.broadcast_arrays``, *not* ``_prepare_jax_args``: the latter flattens
    every argument to 1-D, so a scalar kernel fed scalar parameters would
    return shape ``(1,)`` instead of ``()``.  Under ``vmap`` that becomes an
    ``(n, 1)`` result, and ``g_dev_inc`` then broadcasts against the weights
    into an ``n``-times-too-large deviance (measured: 3350.48 vs 111.68 on a
    30-observation fit).
    """
    mu, sigma, nu = jnp.broadcast_arrays(
        jnp.asarray(mu, dtype=jnp.float64),
        jnp.asarray(sigma, dtype=jnp.float64),
        jnp.asarray(nu, dtype=jnp.float64),
    )
    valid = (
        (mu > 0) & (sigma > 0) & (nu > 0)
        & jnp.isfinite(mu) & jnp.isfinite(sigma) & jnp.isfinite(nu)
    )
    mu_s = jnp.where(valid, mu, 1.0)
    sigma_s = jnp.where(valid, sigma, 1.0)
    nu_s = jnp.where(valid, nu, 1.0)
    return (mu_s, sigma_s, nu_s), valid


def _dburr12_log_pdf(y, mu, sigma, nu):
    """Scalar log-PMF: ``log(Sur(y) - Sur(y+1))``.

    Written as ``log Sur(y) + log(1 - exp(delta))`` with
    ``delta = log Sur(y+1) - log Sur(y) <= 0`` so the subtraction never
    cancels: ``log(-expm1(delta))`` keeps full precision as ``delta -> 0``
    (large counts), where R's literal ``Sur(y) - Sur(y+1)`` loses digits.
    """
    y = jnp.asarray(y, dtype=jnp.float64)
    (mu_s, sigma_s, nu_s), valid = _dburr12_parameters(mu, sigma, nu)
    support = (y >= 0) & jnp.isfinite(y) & (y == jnp.floor(y))
    y_s = jnp.where(support, y, 0.0)
    r_y = jnp.power(y_s / mu_s, sigma_s)
    r_y1 = jnp.power((y_s + 1.0) / mu_s, sigma_s)
    log_sur_y = -nu_s * jnp.log1p(r_y)
    delta = -nu_s * (jnp.log1p(r_y1) - jnp.log1p(r_y))
    log_pdf = log_sur_y + jnp.log(-jnp.expm1(delta))
    log_pdf = jnp.where(support, log_pdf, -jnp.inf)
    return jnp.where(valid & ~jnp.isnan(y), log_pdf, jnp.nan)


def _dburr12_cdf_scalar(q, mu, sigma, nu):
    """Scalar CDF: ``1 - (1 + ((q+1)/mu)^sigma)^(-nu)``.

    R evaluates ``q`` as a real number (no floor), so this is the continuous
    extension; at integer ``q`` it is exactly the discrete CDF.  ``q < 0``
    clamps to 0 and ``q = Inf`` to 1, as R does.
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    (mu_s, sigma_s, nu_s), valid = _dburr12_parameters(mu, sigma, nu)
    q_s = jnp.where(jnp.isfinite(q), q, 0.0)
    z = (q_s + 1.0) / mu_s
    # 1 - (1+z^sigma)^(-nu) as -expm1(...) keeps small tail probabilities
    # relative-accurate; R's literal 1 - ... cancels.
    cdf = -jnp.expm1(-nu_s * jnp.log1p(jnp.power(z, sigma_s)))
    cdf = jnp.where(q < 0.0, 0.0, cdf)
    cdf = jnp.where(q == jnp.inf, 1.0, cdf)
    return jnp.where(valid & ~jnp.isnan(q), cdf, jnp.nan)


def _dburr12_quantile_scalar(p, mu, sigma, nu):
    """Scalar quantile, the *installed* R construction: ``round(w - 1)``.

    ``p`` arrives already transformed (``exp`` for ``log_p``, complemented
    for the upper tail) -- R applies those transforms before clamping, and so
    does the vector wrapper.  The boundary assignments stay in the wrapper.

    ``round`` is R's, i.e. round-half-to-even; ``jnp.round`` is the same
    IEC-60559 rule.  R then clamps negative results to 0 with
    ``ifelse(q < 0, 0, q)``, reproduced here.
    """
    p = jnp.asarray(p, dtype=jnp.float64)
    (mu_s, sigma_s, nu_s), valid = _dburr12_parameters(mu, sigma, nu)
    p_s = jnp.where((p >= 0.0) & (p <= 1.0), p, 0.0)
    # exp(-log(1-p)/nu) - 1 = expm1(-log1p(-p)/nu); log1p(-p) is the stable
    # spelling of R's log(1-p) (<= 1 ulp apart for p in (0, 1)).
    w = mu_s * jnp.power(jnp.expm1(-jnp.log1p(-p_s) / nu_s), 1.0 / sigma_s)
    q = jnp.round(w - 1.0)
    q = jnp.where(q < 0.0, 0.0, q)
    return jnp.where(valid & jnp.isfinite(p) & (p >= 0.0) & (p <= 1.0),
                     q, jnp.nan)


# ---------------------------------------------------------------------------
# Public d/p/q/r (R signatures, R defaults)
# ---------------------------------------------------------------------------


def dDBURR12(x, mu=DEFAULT_MU, sigma=DEFAULT_SIGMA, nu=DEFAULT_NU, log=False):
    """PMF of the discretised Burr XII (R ``dDBURR12``).

    Non-integer ``x`` is outside the support and returns 0 (``-inf`` on the
    log scale); see the module docstring for why R's continuous value is not
    reproduced there.
    """
    x, mu, sigma, nu = jnp.broadcast_arrays(*[
        jnp.asarray(v, dtype=jnp.float64) for v in (x, mu, sigma, nu)
    ])
    flat_log = jax.vmap(_dburr12_log_pdf)(x.ravel(), mu.ravel(),
                                          sigma.ravel(), nu.ravel())
    log_value = flat_log.reshape(x.shape)
    return log_value if log else jnp.exp(log_value)


def pDBURR12(q, mu=DEFAULT_MU, sigma=DEFAULT_SIGMA, nu=DEFAULT_NU,
             lower_tail=True, log_p=False):
    """CDF of the discretised Burr XII (R ``pDBURR12``).

    ``lower_tail``/``log_p`` are honoured.  R applies its ``q < 0`` /
    ``q == Inf`` clamps *after* the ``log`` transform, so R's logged CDF at
    ``q < 0`` is ``0`` (a probability of one) rather than ``-inf``; that
    quirk is pinned in the tests, not copied here.
    """
    q, mu, sigma, nu = jnp.broadcast_arrays(*[
        jnp.asarray(v, dtype=jnp.float64) for v in (q, mu, sigma, nu)
    ])
    cdf = jax.vmap(_dburr12_cdf_scalar)(q.ravel(), mu.ravel(),
                                        sigma.ravel(), nu.ravel()).reshape(q.shape)
    if not lower_tail:
        cdf = 1.0 - cdf
    if log_p:
        cdf = jnp.log(cdf)
    return cdf


def qDBURR12(p, mu=DEFAULT_MU, sigma=DEFAULT_SIGMA, nu=DEFAULT_NU,
             lower_tail=True, log_p=False):
    """Quantile of the discretised Burr XII (R ``qDBURR12``).

    Body: the installed R's ``round(w - 1)`` with its negative clamp (module
    docstring, contract 2).  Boundaries follow the house convention shared
    with ``qSICHEL`` -- ``p == 0`` -> 0 (the support infimum), ``p == 1`` ->
    ``Inf``, any ``p`` outside ``[0, 1]`` -> NaN -- while live R *stops* in
    all four cases (``p must be between 0 and 1``); the stop is pinned in the
    tests.
    """
    p, mu, sigma, nu = jnp.broadcast_arrays(*[
        jnp.asarray(v, dtype=jnp.float64) for v in (p, mu, sigma, nu)
    ])
    target = jnp.exp(p) if log_p else p
    if not lower_tail:
        target = 1.0 - target
    q = jax.vmap(_dburr12_quantile_scalar)(
        target.ravel(), mu.ravel(), sigma.ravel(), nu.ravel()
    ).reshape(p.shape)
    params_ok = ((mu > 0) & (sigma > 0) & (nu > 0)
                 & jnp.isfinite(mu) & jnp.isfinite(sigma) & jnp.isfinite(nu))
    q = jnp.where(params_ok & (target <= 0.0), 0.0, q)
    q = jnp.where(params_ok & (target >= 1.0), jnp.inf, q)
    outside = (target < 0.0) | (target > 1.0) | jnp.isnan(target)
    return jnp.where(outside, jnp.nan, q)


def rDBURR12(key, n, mu=DEFAULT_MU, sigma=DEFAULT_SIGMA, nu=DEFAULT_NU):
    """Random draws (R ``rDBURR12``: ``q(runif(n))``, same construction)."""
    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    mu, sigma, nu = jnp.broadcast_arrays(*[
        jnp.asarray(v, dtype=jnp.float64) for v in (mu, sigma, nu)
    ])
    uniform = jax.random.uniform(key, (n,) + mu.shape, dtype=jnp.float64)
    return qDBURR12(uniform, mu, sigma, nu)


# ---------------------------------------------------------------------------
# R's closed-form derivative slots (discBURR12.R), transcribed verbatim.
#
# Every slot is a survival-difference quotient
#     dldX = (dSurX(y) - dSurX(y+1)) / (Sur(y) - Sur(y+1))
# with the numerator zero at y < 1 (R's ifelse guard), and every d2ldX2 is
# the plain -score^2 -- no -1e-15 clamp here, unlike the Sichel cluster.
# ---------------------------------------------------------------------------

def _dburr12_sur_difference(y, mu, sigma, nu):
    """``Sur(y) - Sur(y+1)`` -- R's denominator, same order of operations."""
    r_y = jnp.power(y / mu, sigma)
    r_y1 = jnp.power((y + 1.0) / mu, sigma)
    return jnp.power(1.0 + r_y, -nu) - jnp.power(1.0 + r_y1, -nu)


def _dburr12_dsur_dmu(y, mu, sigma, nu):
    """R's ``dSdmu``: zero below 1, else ``sigma*nu*(y/mu)^sigma/mu*Sur^(nu+1)``."""
    r_y = jnp.power(y / mu, sigma)
    return jnp.where(
        y >= 1.0,
        sigma * nu * (r_y / mu) * jnp.power(1.0 + r_y, -nu - 1.0),
        0.0,
    )


def _dburr12_dsur_dsigma(y, mu, sigma, nu):
    """R's ``dSdsigma``: zero below 1, else ``-nu*log(y/mu)*(y/mu)^sigma*Sur^(nu+1)``."""
    ratio = y / mu
    r_y = jnp.power(ratio, sigma)
    return jnp.where(
        y >= 1.0,
        -nu * jnp.log(ratio) * r_y * jnp.power(1.0 + r_y, -nu - 1.0),
        0.0,
    )


def _dburr12_dsur_dnu(y, mu, sigma, nu):
    """R's ``dSdnu``: zero below 1, else ``-log(1+(y/mu)^sigma)*Sur``."""
    r_y = jnp.power(y / mu, sigma)
    return jnp.where(
        y >= 1.0,
        -jnp.log1p(r_y) * jnp.power(1.0 + r_y, -nu),
        0.0,
    )


def _dburr12_slot(dsur, y, mu, sigma, nu):
    """Shared body of the three score slots."""
    (y, mu, sigma, nu), scalar = _prepare_jax_args((y, mu, sigma, nu))
    numerator = dsur(y, mu, sigma, nu) - dsur(y + 1.0, mu, sigma, nu)
    out = numerator / _dburr12_sur_difference(y, mu, sigma, nu)
    return out[0] if scalar else out


def _dburr12_dldm(y, mu, sigma, nu):
    """``dldm`` = (dSdmu(y) - dSdmu(y+1)) / (Sur(y) - Sur(y+1))."""
    return _dburr12_slot(_dburr12_dsur_dmu, y, mu, sigma, nu)


def _dburr12_dldd(y, mu, sigma, nu):
    """``dldd`` = (dSdsigma(y) - dSdsigma(y+1)) / (Sur(y) - Sur(y+1))."""
    return _dburr12_slot(_dburr12_dsur_dsigma, y, mu, sigma, nu)


def _dburr12_dldv(y, mu, sigma, nu):
    """``dldv`` = (dSdnu(y) - dSdnu(y+1)) / (Sur(y) - Sur(y+1))."""
    return _dburr12_slot(_dburr12_dsur_dnu, y, mu, sigma, nu)


def _dburr12_negative_information(score_fn):
    """R's ``d2ldX2 = -dldX^2`` -- no floor: DBURR12 has no -1e-15 clamp."""

    def hessian(y, mu, sigma, nu):
        score = score_fn(y, mu, sigma, nu)
        return -score * score

    hessian.__name__ = f"d2_{getattr(score_fn, '__name__', 'slot')}"
    return hessian


_DBURR12_SCORES = {
    "mu": _dburr12_dldm,
    "sigma": _dburr12_dldd,
    "nu": _dburr12_dldv,
}
_DBURR12_HESSIANS = {
    "mu": _dburr12_negative_information(_dburr12_dldm),
    "sigma": _dburr12_negative_information(_dburr12_dldd),
    "nu": _dburr12_negative_information(_dburr12_dldv),
}


@dataclass(frozen=True)
class DBURR12Family(FamilyDefinition):
    """Discretised Burr XII family (R ``DBURR12``)."""


def DBURR12() -> FamilyDefinition:
    """Discrete Burr XII family, R's links / initials / valid set.

    R's family object: all three ``*.link = 'log'``,
    ``mu.initial = (y + mean(y))/2``, ``sigma.initial = rep(2)``,
    ``nu.initial = rep(2)``, ``nu.valid`` requires ``nu > 0``.  The sigma and
    nu starts live in ``_fitting_init`` (the house home for R's initial
    expressions); mu's roll-out default is already R's expression.
    """
    return build_ad_family(
        family_class=DBURR12Family,
        name="DBURR12",
        parameters=("mu", "sigma", "nu"),
        log_pdf_func=_dburr12_log_pdf,
        type_="Discrete",
        links={"mu": "log", "sigma": "log", "nu": "log"},
        link_functions={"mu": log_link, "sigma": log_link, "nu": log_link},
        link_inverses={"mu": log_inverse, "sigma": log_inverse,
                       "nu": log_inverse},
        link_derivatives={"mu": log_derivative, "sigma": log_derivative,
                          "nu": log_derivative},
        score_overrides=_DBURR12_SCORES,
        hessian_overrides=_DBURR12_HESSIANS,
        d=dDBURR12,
        p=pDBURR12,
        q=qDBURR12,
        r=rDBURR12,
    )
