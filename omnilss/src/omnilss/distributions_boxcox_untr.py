"""Untruncated Box-Cox families -- R ``BCTuntr`` / ``BCCGuntr`` / ``BCPEuntr``.

gamlss.dist ships these as three families in their own right.  They differ from
``BCT`` / ``BCCG`` / ``BCPE`` in exactly one respect: the upper tail beyond
``1/(sigma*|nu|)`` is **not** renormalised away.  Concretely, the truncated
families

* subtract ``log(F(1/(sigma*|nu|)))`` from the log-density, and
* carry the matching hazard ``h = f(1/(sigma*|nu|)) / F(1/(sigma*|nu|))`` in
  ``dldd`` / ``dldv`` / ``dldt``.

The untruncated families drop both, and keep everything else -- kernels,
standardisation constant ``c``, links -- byte-for-byte.

Why this module exists separately from ``distributions.py``
----------------------------------------------------------
``BCT`` / ``BCCG`` / ``BCPE`` each wrap a hand-transcribed score/Hessian pair
that is *specific to the truncated density*.  The untruncated scores are a
different set of formulas (same algebra, hazard and finite-difference terms
removed), so sharing the closures would mean threading a truncation flag
through every one of them.  Three small standalone families are easier to
read against R's source than one parameterised family would be.

Parity note
-----------
R's untruncated families expose **no** ``d``/``p``/``q`` functions at all --
only ``G.dev.incr`` plus a warning when ``F(1/(sigma*|nu|)) < 0.99``.  So the
log-densities and scores below are transcribed from ``G.dev.incr`` and the
``dld*``/``d2ld*`` slots R does ship, while ``p``/``q``/``r`` are derived from
the same kernels (a distribution's CDF is determined by its density; the
absence of a named R function is not the absence of a definition).

``BCTo`` / ``BCCGo`` / ``BCPEo`` are a **different** axis: their densities are
byte-identical to the truncated families' (``identical(deparse(body(dBCCGo)),
deparse(body(dBCCG)))`` is ``TRUE`` in R), and only ``mu.link`` swaps between
``log`` and ``identity``.  Do not confuse the two groups -- the ``o`` suffix
does not mean "untruncated".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax.numpy as jnp
from jax.scipy.special import digamma as _digamma
from jax.scipy.special import gammaln, polygamma as _polygamma

from .dpqr_functions import (
    _bcpe_c,
    dBCCGuntr,
    dBCPEuntr,
    dBCTuntr,
    pBCCGuntr,
    pBCPEuntr,
    pBCTuntr,
    qBCCGuntr,
    qBCPEuntr,
    qBCTuntr,
    rBCCGuntr,
    rBCPEuntr,
    rBCTuntr,
)
from .families import FamilyDefinition
from .links import (
    identity_derivative,
    identity_inverse,
    identity_link,
    log_derivative,
    log_inverse,
    log_link,
)

_EPS = jnp.finfo(jnp.float64).eps


@dataclass(frozen=True)
class BCTuntrFamily(FamilyDefinition):
    """Untruncated Box-Cox t (R ``BCTuntr``)."""

    pass


@dataclass(frozen=True)
class BCCGuntrFamily(FamilyDefinition):
    """Untruncated Box-Cox Cole-Green (R ``BCCGuntr``)."""

    pass


@dataclass(frozen=True)
class BCPEuntrFamily(FamilyDefinition):
    """Untruncated Box-Cox power exponential (R ``BCPEuntr``)."""

    pass


def _boxcox_z(y, mu, sigma, nu):
    """``z = ((y/mu)^nu - 1)/(nu*sigma)``, falling back to ``log(y/mu)/sigma``.

    R writes this as ``ifelse(nu != 0, 1, 0) * (...) + ifelse(nu == 0, 1, 0) *
    log(y/mu) * k`` so that both branches stay finite even when ``nu`` is
    exactly zero.  ``jnp.where`` evaluates both sides too, so the 1/nu
    blow-up is masked away rather than avoided.
    """
    y = jnp.maximum(jnp.asarray(y, dtype=jnp.float64), _EPS)
    mu = jnp.maximum(jnp.asarray(mu, dtype=jnp.float64), _EPS)
    sigma = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float64), _EPS)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    ratio = y / mu
    return jnp.where(
        jnp.abs(nu) < 1e-6,
        jnp.log(ratio) / sigma,
        (jnp.power(ratio, nu) - 1.0) / (nu * sigma),
    )


def _gammaln_array(x):
    """``lgamma`` elementwise.

    ``jax.scipy.special.gammaln`` is already vectorised, but ``distributions.py``
    uses a ``math.lgamma``-loop in its BCT family; matching that here would be
    gratuitous.  The library function is exact and traced-safe, so use it.
    """
    return gammaln(jnp.asarray(x, dtype=jnp.float64))


# ---------------------------------------------------------------------------
# BCTuntr
# ---------------------------------------------------------------------------


def _bctuntr_log_density(y, mu, sigma, nu, tau):
    """R ``BCTuntr()$G.dev.incr`` / -2.

    Same body as :func:`dBCTuntr`; kept as a named local so the family object
    and the exported ``d`` cannot drift apart.
    """
    return dBCTuntr(y, mu, sigma, nu, tau, log=True)


def _bctuntr_score(parameter, y, mu, sigma, nu, tau):
    """R ``BCTuntr()``'s ``dldm`` / ``dldd`` / ``dldv`` / ``dldt``.

    Transcribed from ``gamlss.dist::BCTuntr()``.  The difference from the
    truncated ``BCT`` scores is that ``dldd`` / ``dldv`` drop the hazard term
    and ``dldt`` drops the 0.01 forward difference of ``log pt`` -- both exist
    only to differentiate the truncation constant.
    """
    y = jnp.maximum(jnp.asarray(y, dtype=jnp.float64), _EPS)
    mu = jnp.maximum(jnp.asarray(mu, dtype=jnp.float64), _EPS)
    sigma = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float64), _EPS)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    tau = jnp.maximum(jnp.asarray(tau, dtype=jnp.float64), _EPS)

    k = 1.0 / sigma
    z = _boxcox_z(y, mu, sigma, nu)
    u = (tau + 1.0) / (tau + jnp.square(z))
    log_ratio = jnp.log(y / mu)

    if parameter == "mu":
        # R: k * u * (z/mu) + (nu/mu) * (u * (z*z) - 1)
        return k * u * (z / mu) + (nu / mu) * (u * jnp.square(z) - 1.0)
    if parameter == "sigma":
        # R: k * (u * (z*z) - 1)
        return k * (u * jnp.square(z) - 1.0)
    if parameter == "nu":
        # R: ((u*z)/nu) * (z - lnybym*k) - lnybym * (u*z*z - 1)
        return ((u * z) / nu) * (z - log_ratio * k) - log_ratio * (
            u * jnp.square(z) - 1.0
        )
    return (
        -0.5 * jnp.log1p(jnp.square(z) / tau)
        + 0.5 * u * jnp.square(z) / tau
        + 0.5 * _digamma((tau + 1.0) / 2.0)
        - 0.5 * _digamma(tau / 2.0)
        - 1.0 / (2.0 * tau)
    )


def _bctuntr_hessian(parameter, y, mu, sigma, nu, tau):
    """R ``BCTuntr()``'s diagonal second derivatives.

    ``d2ldm2`` / ``d2ldd2`` / ``d2ldv2`` / ``d2ldt2`` are identical to the
    truncated ``BCT`` family (verified with ``identical(deparse(body(...)))``
    against live R) -- none of them involved the truncation constant.  They are
    coarse (``d2ldv2`` is literally ``-7*sigma^2/4``) but that is what R feeds
    its IRLS.
    """
    mu = jnp.maximum(jnp.asarray(mu, dtype=jnp.float64), _EPS)
    sigma = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float64), _EPS)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    tau = jnp.maximum(jnp.asarray(tau, dtype=jnp.float64), _EPS)

    if parameter == "mu":
        return -(tau + 2.0 * jnp.square(nu) * jnp.square(sigma) * tau + 1.0) / (
            tau + 3.0
        ) / (jnp.square(mu) * jnp.square(sigma))
    if parameter == "sigma":
        return -2.0 * tau / (jnp.square(sigma) * (tau + 3.0))
    if parameter == "nu":
        return -7.0 * jnp.square(sigma) / 4.0

    value = (
        _polygamma(1, (tau + 1.0) / 2.0)
        - _polygamma(1, tau / 2.0)
        + 2.0 * (tau + 5.0) / (tau * (tau + 1.0) * (tau + 3.0))
    ) / 4.0
    return jnp.where(value < -1e-15, value, -1e-15)


def _tau_link(x):
    return jnp.log(jnp.maximum(jnp.asarray(x, dtype=jnp.float64), _EPS))


def _tau_inverse(eta):
    return jnp.maximum(jnp.exp(jnp.asarray(eta, dtype=jnp.float64)), _EPS)


def BCTuntr() -> BCTuntrFamily:
    """Untruncated Box-Cox t with parameters `mu`, `sigma`, `nu`, `tau`.

    R's ``mu.link`` is ``identity`` -- *the same as* the truncated ``BCT``, not
    ``log``.  The ``log`` link belongs to ``BCTo``, which is a different family
    that happens to share this family's density.
    """
    return BCTuntrFamily(
        name="BCTUNTR",
        parameters=("mu", "sigma", "nu", "tau"),
        g_dev_inc=lambda y, mu, sigma, nu, tau: -2.0
        * _bctuntr_log_density(y, mu, sigma, nu, tau),
        type="Continuous",
        links={"mu": "identity", "sigma": "log", "nu": "identity", "tau": "log"},
        link_functions={
            "mu": identity_link,
            "sigma": log_link,
            "nu": identity_link,
            "tau": _tau_link,
        },
        link_inverses={
            "mu": identity_inverse,
            "sigma": log_inverse,
            "nu": identity_inverse,
            "tau": _tau_inverse,
        },
        link_derivatives={
            "mu": identity_derivative,
            "sigma": log_derivative,
            "nu": identity_derivative,
            "tau": _tau_inverse,
        },
        score_functions={
            name: (lambda p: lambda y, mu, sigma, nu, tau: _bctuntr_score(
                p, y, mu, sigma, nu, tau
            ))(name)
            for name in ("mu", "sigma", "nu", "tau")
        },
        hessian_functions={
            name: (lambda p: lambda y, mu, sigma, nu, tau: _bctuntr_hessian(
                p, y, mu, sigma, nu, tau
            ))(name)
            for name in ("mu", "sigma", "nu", "tau")
        },
        d=dBCTuntr,
        p=pBCTuntr,
        q=qBCTuntr,
        r=rBCTuntr,
    )


# ---------------------------------------------------------------------------
# BCCGuntr
# ---------------------------------------------------------------------------


def _bccguntr_score(parameter, y, mu, sigma, nu):
    """R ``BCCGuntr()``'s ``dldm`` / ``dldd`` / ``dldv`` -- no hazard term."""
    y = jnp.maximum(jnp.asarray(y, dtype=jnp.float64), _EPS)
    mu = jnp.maximum(jnp.asarray(mu, dtype=jnp.float64), _EPS)
    sigma = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float64), _EPS)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    z = _boxcox_z(y, mu, sigma, nu)

    if parameter == "mu":
        return ((z / sigma) + nu * (jnp.square(z) - 1.0)) / mu
    if parameter == "sigma":
        return (jnp.square(z) - 1.0) / sigma
    log_ratio = jnp.log(y / mu)
    return (z - (log_ratio / sigma)) * (z / nu) - log_ratio * (
        jnp.square(z) - 1.0
    )


def _bccguntr_hessian(parameter, y, mu, sigma, nu):
    """R ``BCCGuntr()``'s diagonal second derivatives (identical to ``BCCG``'s)."""
    mu = jnp.maximum(jnp.asarray(mu, dtype=jnp.float64), _EPS)
    sigma = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float64), _EPS)
    nu = jnp.asarray(nu, dtype=jnp.float64)

    if parameter == "mu":
        return -(1.0 + 2.0 * jnp.square(nu) * jnp.square(sigma)) / (
            jnp.square(mu) * jnp.square(sigma)
        )
    if parameter == "sigma":
        return -2.0 / jnp.square(sigma)
    return -7.0 * jnp.square(sigma) / 4.0


def BCCGuntr() -> BCCGuntrFamily:
    """Untruncated Box-Cox Cole-Green with parameters `mu`, `sigma`, `nu`."""
    return BCCGuntrFamily(
        name="BCCGUNTR",
        parameters=("mu", "sigma", "nu"),
        g_dev_inc=lambda y, mu, sigma, nu: -2.0
        * dBCCGuntr(y, mu, sigma, nu, log=True),
        type="Continuous",
        links={"mu": "identity", "sigma": "log", "nu": "identity"},
        link_functions={
            "mu": identity_link,
            "sigma": log_link,
            "nu": identity_link,
        },
        link_inverses={
            "mu": identity_inverse,
            "sigma": log_inverse,
            "nu": identity_inverse,
        },
        link_derivatives={
            "mu": identity_derivative,
            "sigma": log_derivative,
            "nu": identity_derivative,
        },
        score_functions={
            name: (lambda p: lambda y, mu, sigma, nu: _bccguntr_score(
                p, y, mu, sigma, nu
            ))(name)
            for name in ("mu", "sigma", "nu")
        },
        hessian_functions={
            name: (lambda p: lambda y, mu, sigma, nu: _bccguntr_hessian(
                p, y, mu, sigma, nu
            ))(name)
            for name in ("mu", "sigma", "nu")
        },
        d=dBCCGuntr,
        p=pBCCGuntr,
        q=qBCCGuntr,
        r=rBCCGuntr,
    )


# ---------------------------------------------------------------------------
# BCPEuntr
# ---------------------------------------------------------------------------


def _bcpeuntr_score(parameter, y, mu, sigma, nu, tau):
    """R ``BCPEuntr()``'s ``dldm`` / ``dldd`` / ``dldv`` / ``dldt``.

    Identical to ``BCPE``'s scores minus (a) the hazard in ``dldd`` / ``dldv``
    and (b) the 0.001 forward difference of ``log F.T`` in ``dldt``.

    Unlike ``BCPE``'s ``dldm``, this one is written out in R's own
    ``ifelse(nu != 0, ..., log(y/mu)/sigma)`` form; the two branches agree
    numerically, and R's version is the one being transcribed.
    """
    y = jnp.maximum(jnp.asarray(y, dtype=jnp.float64), _EPS)
    mu = jnp.maximum(jnp.asarray(mu, dtype=jnp.float64), _EPS)
    sigma = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float64), _EPS)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    tau = jnp.maximum(jnp.asarray(tau, dtype=jnp.float64), _EPS)

    z = _boxcox_z(y, mu, sigma, nu)
    c = _bcpe_c(tau)

    if parameter == "mu":
        return (
            tau / (2.0 * mu * sigma * jnp.square(c))
        ) * (z + sigma * nu * jnp.square(z)) * jnp.power(
            jnp.abs(z / c), tau - 2.0
        ) - (nu / mu)
    if parameter == "sigma":
        return (1.0 / sigma) * (
            (tau / 2.0) * jnp.power(jnp.abs(z / c), tau) - 1.0
        )
    if parameter == "nu":
        value = (
            -(tau / (2.0 * nu * jnp.square(c)))
            * jnp.power(jnp.abs(z / c), tau - 2.0)
            * z
            * ((nu * z + 1.0 / sigma) * jnp.log(y / mu) - z)
        )
        return value + jnp.log(y / mu)

    d_log_c_dt = (1.0 / (2.0 * tau * tau)) * (
        2.0 * math.log(2.0) - _digamma(1.0 / tau) + 3.0 * _digamma(3.0 / tau)
    )
    return (
        1.0 / tau
        - 0.5 * jnp.log(jnp.abs(z / c)) * jnp.power(jnp.abs(z / c), tau)
        + (1.0 / (tau * tau)) * (math.log(2.0) + _digamma(1.0 / tau))
        + ((tau / 2.0) * jnp.power(jnp.abs(z / c), tau) - 1.0) * d_log_c_dt
    )


def _bcpeuntr_hessian(parameter, y, mu, sigma, nu, tau):
    """R ``BCPEuntr()``'s diagonal second derivatives.

    One real difference from ``BCPE``: ``d2ldm2`` here has **no**
    ``if (any(tau < 1.05)) -dldm^2`` branch -- the untruncated family keeps the
    closed form unconditionally.  (Verified with
    ``identical(deparse(body(BCPE()$d2ldm2)), deparse(body(BCPEuntr()$d2ldm2)))``
    -> ``FALSE``, while the bodies differ *only* in that branch.)
    """
    mu = jnp.maximum(jnp.asarray(mu, dtype=jnp.float64), _EPS)
    sigma = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float64), _EPS)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    tau = jnp.maximum(jnp.asarray(tau, dtype=jnp.float64), _EPS)

    if parameter == "mu":
        return (
            -(tau * tau)
            * jnp.exp(
                gammaln(2.0 - 1.0 / tau)
                + gammaln(3.0 / tau)
                - 2.0 * gammaln(1.0 / tau)
            )
            / (jnp.square(mu) * jnp.square(sigma))
            - (tau * jnp.square(nu)) / jnp.square(mu)
        )
    if parameter == "sigma":
        return -tau / jnp.square(sigma)
    if parameter == "nu":
        return -jnp.square(sigma) * (3.0 * tau + 1.0) / 4.0

    p = (tau + 1.0) / tau
    d_log_c_dt = (1.0 / (2.0 * tau * tau)) * (
        2.0 * math.log(2.0) - _digamma(1.0 / tau) + 3.0 * _digamma(3.0 / tau)
    )
    part1 = p * _polygamma(1, p) + 2.0 * jnp.square(_digamma(p))
    part2 = _digamma(p) * (
        math.log(2.0) + 3.0 - 3.0 * _digamma(3.0 / tau) - tau
    )
    part3 = -3.0 * _digamma(3.0 / tau) * (1.0 + math.log(2.0))
    part4 = -(tau + math.log(2.0)) * math.log(2.0)
    part5 = -tau + (tau ** 4) * d_log_c_dt ** 2
    value = -(part1 + part2 + part3 + part4 + part5) / (tau ** 3)
    return jnp.where(value < -1e-15, value, -1e-15)


def BCPEuntr() -> BCPEuntrFamily:
    """Untruncated Box-Cox power exponential with `mu`, `sigma`, `nu`, `tau`."""
    return BCPEuntrFamily(
        name="BCPEUNTR",
        parameters=("mu", "sigma", "nu", "tau"),
        g_dev_inc=lambda y, mu, sigma, nu, tau: -2.0
        * dBCPEuntr(y, mu, sigma, nu, tau, log=True),
        type="Continuous",
        links={"mu": "identity", "sigma": "log", "nu": "identity", "tau": "log"},
        link_functions={
            "mu": identity_link,
            "sigma": log_link,
            "nu": identity_link,
            "tau": _tau_link,
        },
        link_inverses={
            "mu": identity_inverse,
            "sigma": log_inverse,
            "nu": identity_inverse,
            "tau": _tau_inverse,
        },
        link_derivatives={
            "mu": identity_derivative,
            "sigma": log_derivative,
            "nu": identity_derivative,
            "tau": _tau_inverse,
        },
        score_functions={
            name: (lambda p: lambda y, mu, sigma, nu, tau: _bcpeuntr_score(
                p, y, mu, sigma, nu, tau
            ))(name)
            for name in ("mu", "sigma", "nu", "tau")
        },
        hessian_functions={
            name: (lambda p: lambda y, mu, sigma, nu, tau: _bcpeuntr_hessian(
                p, y, mu, sigma, nu, tau
            ))(name)
            for name in ("mu", "sigma", "nu", "tau")
        },
        d=dBCPEuntr,
        p=pBCPEuntr,
        q=qBCPEuntr,
        r=rBCPEuntr,
    )