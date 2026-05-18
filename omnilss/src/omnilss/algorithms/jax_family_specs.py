# SPDX-License-Identifier: GPL-3.0-or-later
"""Pure-function family specifications for JAX-native RS fitting.

Each ``FamilyJAXSpec`` bundles the mathematical primitives of a GAMLSS
distribution family as plain callables that JAX can trace through
``jax.jit``, ``jax.lax.while_loop``, and ``jax.lax.fori_loop``.

The key design constraint: **no Python objects inside the JIT boundary**.
All family-specific constants are captured via closures or
``functools.partial`` before the spec is constructed.

Supported families (Phase 1):
    NO  — Normal
    GA  — Gamma
    PO  — Poisson
    BI  — Binomial (Bernoulli when bd=1)
    WEI — Weibull
    TF  — Student-t (3-parameter)
"""

from __future__ import annotations

import functools
import math
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, digamma, polygamma

from ..links import (
    identity_link, identity_inverse, identity_derivative,
    log_link, log_inverse, log_derivative,
    logit_link, logit_inverse, logit_derivative,
)

# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

class FamilyJAXSpec(NamedTuple):
    """Immutable specification of a GAMLSS family for JAX-native fitting.

    All callables must be pure functions compatible with ``jax.jit``.
    Tuple ordering of ``param_names`` determines the axis-0 ordering of
    the ``params`` array (shape ``[n_params, n_obs]``) used inside the
    JAX RS core.

    Attributes
    ----------
    name : str
        Family name, e.g. ``"NO"``.
    param_names : tuple[str, ...]
        Estimable parameter names in canonical order.
    link_fns : tuple[Callable, ...]
        Forward link functions ``g(theta) -> eta``, one per parameter.
    link_inv_fns : tuple[Callable, ...]
        Inverse link functions ``g^{-1}(eta) -> theta``, one per parameter.
    link_deriv_fns : tuple[Callable, ...]
        Derivatives of the inverse link ``dtheta/deta``, one per parameter.
    score_fns : tuple[Callable, ...]
        Score functions ``dl/dtheta_k(y, *params) -> [n]``, one per parameter.
        Signature: ``(y, p0, p1, ...) -> array`` where ``p0, p1, ...`` are
        the parameter arrays in ``param_names`` order.
    hessian_fns : tuple[Callable, ...]
        Diagonal Hessian functions ``d2l/dtheta_k^2(y, *params) -> [n]``.
        Same signature as ``score_fns``.
    loglik_fn : Callable
        Per-observation log-likelihood ``log p(y | p0, p1, ...) -> [n]``.
    eta_bounds : tuple[tuple[float, float], ...]
        (lo, hi) clip bounds for the linear predictor of each parameter.
        Prevents overflow in link_inv during IRLS.  Defaults to (-10, 10)
        for log/logit links and (-1e6, 1e6) for identity links.
    """

    name: str
    param_names: tuple[str, ...]
    link_fns: tuple[Callable, ...]
    link_inv_fns: tuple[Callable, ...]
    link_deriv_fns: tuple[Callable, ...]
    score_fns: tuple[Callable, ...]
    hessian_fns: tuple[Callable, ...]
    loglik_fn: Callable
    eta_bounds: tuple = ((-10.0, 10.0),)  # overridden per family
    init_etas: tuple[float, ...] = ()  # cold-start eta defaults; NaN means data-derived


# ---------------------------------------------------------------------------
# Helper: build score/hessian via JAX autodiff from a log-pdf
# ---------------------------------------------------------------------------

def _make_ad_score_hessian(
    logpdf_fn: Callable,
    param_idx: int,
    n_params: int,
) -> tuple[Callable, Callable]:
    """Return (score_fn, hessian_fn) for parameter at ``param_idx``.

    ``logpdf_fn`` must have signature ``(y, p0, p1, ...) -> scalar``.
    The returned functions have signature ``(y, p0, p1, ...) -> [n]``.
    """
    # argnums offset by 1 because y is arg 0
    argnums = param_idx + 1

    _grad = jax.grad(logpdf_fn, argnums=argnums)
    _hess = jax.grad(_grad, argnums=argnums)

    # Vectorise over observations
    _score_vmap = jax.vmap(
        lambda *args: _grad(*args),
        in_axes=(0,) + (0,) * n_params,
    )
    _hess_vmap = jax.vmap(
        lambda *args: _hess(*args),
        in_axes=(0,) + (0,) * n_params,
    )

    def score_fn(y, *params):
        return _score_vmap(y, *params)

    def hessian_fn(y, *params):
        return _hess_vmap(y, *params)

    return score_fn, hessian_fn


# ---------------------------------------------------------------------------
# NO — Normal distribution
# ---------------------------------------------------------------------------

def make_no_spec() -> FamilyJAXSpec:
    """Build ``FamilyJAXSpec`` for the Normal (NO) distribution.

    Parameters: mu (identity link), sigma (log link).
    Score and Hessian are analytic (closed-form).
    """
    eps = jnp.finfo(jnp.float64).eps

    def loglik(y, mu, sigma):
        sigma = jnp.maximum(sigma, eps)
        return (
            -0.5 * jnp.log(2.0 * math.pi)
            - jnp.log(sigma)
            - 0.5 * jnp.square((y - mu) / sigma)
        )

    # Analytic score
    def score_mu(y, mu, sigma):
        sigma = jnp.maximum(sigma, eps)
        return (y - mu) / jnp.square(sigma)

    def score_sigma(y, mu, sigma):
        sigma = jnp.maximum(sigma, eps)
        return -1.0 / sigma + jnp.square(y - mu) / jnp.power(sigma, 3.0)

    # Analytic (expected) Hessian — per-observation (constant for NO)
    def hess_mu(y, mu, sigma):
        sigma = jnp.maximum(sigma, eps)
        return -jnp.ones_like(y) / jnp.square(sigma)

    def hess_sigma(y, mu, sigma):
        sigma = jnp.maximum(sigma, eps)
        return -2.0 * jnp.ones_like(y) / jnp.square(sigma)

    return FamilyJAXSpec(
        name="NO",
        param_names=("mu", "sigma"),
        link_fns=(identity_link, log_link),
        link_inv_fns=(identity_inverse, log_inverse),
        link_deriv_fns=(identity_derivative, log_derivative),
        score_fns=(score_mu, score_sigma),
        hessian_fns=(hess_mu, hess_sigma),
        loglik_fn=loglik,
        # eta bounds: identity mu (wide); log sigma: exp(-6)≈0.002, exp(6)≈400
        eta_bounds=((-1e6, 1e6), (-6.0, 6.0)),
        init_etas=(math.nan, math.nan),
    )


# ---------------------------------------------------------------------------
# GA — Gamma distribution
# ---------------------------------------------------------------------------

def make_ga_spec() -> FamilyJAXSpec:
    """Build ``FamilyJAXSpec`` for the Gamma (GA) distribution.

    Parameters: mu (log link), sigma (log link).
    Parameterisation: E[Y]=mu, Var[Y]=sigma^2 * mu^2.
    """
    eps = jnp.finfo(jnp.float64).eps

    def loglik(y, mu, sigma):
        y = jnp.maximum(y, eps)
        mu = jnp.maximum(mu, eps)
        sigma = jnp.maximum(sigma, eps)
        shape = 1.0 / jnp.square(sigma)
        scale = mu * jnp.square(sigma)
        return (
            (shape - 1.0) * jnp.log(y)
            - y / scale
            - gammaln(shape)
            - shape * jnp.log(scale)
        )

    def score_mu(y, mu, sigma):
        mu = jnp.maximum(mu, eps)
        sigma = jnp.maximum(sigma, eps)
        return (y - mu) / (jnp.square(mu) * jnp.square(sigma))

    def score_sigma(y, mu, sigma):
        mu = jnp.maximum(mu, eps)
        sigma = jnp.maximum(sigma, eps)
        y = jnp.maximum(y, eps)
        shape = 1.0 / jnp.square(sigma)
        return (2.0 / jnp.power(sigma, 3.0)) * (
            (y / mu) - jnp.log(y) + jnp.log(mu)
            + jnp.log(jnp.square(sigma)) - 1.0 + digamma(shape)
        )

    def hess_mu(y, mu, sigma):
        mu = jnp.maximum(mu, eps)
        sigma = jnp.maximum(sigma, eps)
        return -jnp.ones_like(y) / (jnp.square(mu) * jnp.square(sigma))

    def hess_sigma(y, mu, sigma):
        sigma = jnp.maximum(sigma, eps)
        shape = 1.0 / jnp.square(sigma)
        return (4.0 / jnp.power(sigma, 4.0)) - (4.0 / jnp.power(sigma, 6.0)) * polygamma(1, shape)

    return FamilyJAXSpec(
        name="GA",
        param_names=("mu", "sigma"),
        link_fns=(log_link, log_link),
        link_inv_fns=(log_inverse, log_inverse),
        link_deriv_fns=(log_derivative, log_derivative),
        score_fns=(score_mu, score_sigma),
        hessian_fns=(hess_mu, hess_sigma),
        loglik_fn=loglik,
        # eta bounds: log mu: exp(-6)≈0.002, exp(10)≈22000; log sigma: exp(-6)..exp(3)≈20
        eta_bounds=((-6.0, 10.0), (-6.0, 3.0)),
        init_etas=(math.nan, 0.0),
    )


# ---------------------------------------------------------------------------
# PO — Poisson distribution
# ---------------------------------------------------------------------------

def make_po_spec() -> FamilyJAXSpec:
    """Build ``FamilyJAXSpec`` for the Poisson (PO) distribution.

    Single parameter: mu (log link).
    """
    eps = jnp.finfo(jnp.float64).eps

    def loglik(y, mu):
        mu = jnp.maximum(mu, eps)
        return y * jnp.log(mu) - mu - gammaln(y + 1.0)

    def score_mu(y, mu):
        mu = jnp.maximum(mu, eps)
        return y / mu - 1.0

    def hess_mu(y, mu):
        mu = jnp.maximum(mu, eps)
        # Use expected (Fisher) Hessian: E[-d2l/dmu2] = 1/mu
        # This is more stable than the observed Hessian -y/mu^2
        return -jnp.ones_like(y) / mu

    return FamilyJAXSpec(
        name="PO",
        param_names=("mu",),
        link_fns=(log_link,),
        link_inv_fns=(log_inverse,),
        link_deriv_fns=(log_derivative,),
        score_fns=(score_mu,),
        hessian_fns=(hess_mu,),
        loglik_fn=loglik,
        # eta bounds: log mu: exp(-6)≈0.002, exp(8)≈3000
        eta_bounds=((-6.0, 8.0),),
        init_etas=(math.nan,),
    )


# ---------------------------------------------------------------------------
# BI — Binomial distribution
# ---------------------------------------------------------------------------

def make_bi_spec(bd: float = 1.0) -> FamilyJAXSpec:
    """Build ``FamilyJAXSpec`` for the Binomial (BI) distribution.

    Parameters
    ----------
    bd : float
        Binomial denominator (number of trials).  Defaults to 1 (Bernoulli).
        For general binomial, pass the fixed denominator value.
        When ``bd`` varies per observation, use the NumPy RS path instead.

    Single estimable parameter: mu (logit link).
    """
    eps = jnp.finfo(jnp.float64).eps
    _bd = float(bd)

    def loglik(y, mu):
        mu = jnp.clip(mu, eps, 1.0 - eps)
        if _bd == 1.0:
            return y * jnp.log(mu) + (1.0 - y) * jnp.log(1.0 - mu)
        bd_arr = jnp.asarray(_bd, dtype=jnp.float64)
        y_counts = jnp.where(y <= 1.0, jnp.round(y * bd_arr), y)
        log_comb = (
            gammaln(bd_arr + 1.0)
            - gammaln(y_counts + 1.0)
            - gammaln(bd_arr - y_counts + 1.0)
        )
        return log_comb + y_counts * jnp.log(mu) + (bd_arr - y_counts) * jnp.log(1.0 - mu)

    def score_mu(y, mu):
        mu = jnp.clip(mu, eps, 1.0 - eps)
        if _bd == 1.0:
            return y / mu - (1.0 - y) / (1.0 - mu)
        bd_arr = jnp.asarray(_bd, dtype=jnp.float64)
        y_counts = jnp.where(y <= 1.0, jnp.round(y * bd_arr), y)
        return (y_counts - bd_arr * mu) / (mu * (1.0 - mu))

    def hess_mu(y, mu):
        mu = jnp.clip(mu, eps, 1.0 - eps)
        if _bd == 1.0:
            # Expected (Fisher) Hessian for Bernoulli: -1/(mu*(1-mu))
            return -jnp.ones_like(y) / (mu * (1.0 - mu))
        bd_arr = jnp.asarray(_bd, dtype=jnp.float64)
        return -bd_arr * jnp.ones_like(y) / (mu * (1.0 - mu))

    return FamilyJAXSpec(
        name="BI",
        param_names=("mu",),
        link_fns=(logit_link,),
        link_inv_fns=(logit_inverse,),
        link_deriv_fns=(logit_derivative,),
        score_fns=(score_mu,),
        hessian_fns=(hess_mu,),
        loglik_fn=loglik,
        # eta bounds: logit mu: sigmoid(-8)≈0.0003, sigmoid(8)≈0.9997
        eta_bounds=((-8.0, 8.0),),
        init_etas=(0.0,),
    )


# ---------------------------------------------------------------------------
# WEI — Weibull distribution
# ---------------------------------------------------------------------------

def make_wei_spec() -> FamilyJAXSpec:
    """Build ``FamilyJAXSpec`` for the Weibull (WEI) distribution.

    Parameters: mu (log link, scale λ), sigma (log link, shape k).
    Uses R's parameterisation: dweibull(x, scale=mu, shape=sigma).
    Hessian uses Fisher information (expected), matching R gamlss.
    """
    eps = jnp.finfo(jnp.float64).eps

    def loglik(y, mu, sigma):
        y = jnp.maximum(y, eps)
        mu = jnp.maximum(mu, eps)
        sigma = jnp.maximum(sigma, eps)
        return (
            jnp.log(sigma)
            - jnp.log(mu)
            + (sigma - 1.0) * (jnp.log(y) - jnp.log(mu))
            - jnp.power(y / mu, sigma)
        )

    # AD-based score (exact)
    def _loglik_scalar(y_s, mu_s, sigma_s):
        y_s = jnp.maximum(y_s, eps)
        mu_s = jnp.maximum(mu_s, eps)
        sigma_s = jnp.maximum(sigma_s, eps)
        return (
            jnp.log(sigma_s)
            - jnp.log(mu_s)
            + (sigma_s - 1.0) * (jnp.log(y_s) - jnp.log(mu_s))
            - jnp.power(y_s / mu_s, sigma_s)
        )

    _score_mu_scalar = jax.grad(_loglik_scalar, argnums=1)
    _score_sigma_scalar = jax.grad(_loglik_scalar, argnums=2)
    _score_mu_vmap = jax.vmap(_score_mu_scalar)
    _score_sigma_vmap = jax.vmap(_score_sigma_scalar)

    def score_mu(y, mu, sigma):
        return _score_mu_vmap(
            jnp.asarray(y, dtype=jnp.float64),
            jnp.asarray(mu, dtype=jnp.float64),
            jnp.asarray(sigma, dtype=jnp.float64),
        )

    def score_sigma(y, mu, sigma):
        return _score_sigma_vmap(
            jnp.asarray(y, dtype=jnp.float64),
            jnp.asarray(mu, dtype=jnp.float64),
            jnp.asarray(sigma, dtype=jnp.float64),
        )

    # Expected (Fisher) Hessian — matches R gamlss WEI
    def hess_mu(y, mu, sigma):
        mu = jnp.maximum(jnp.asarray(mu, dtype=jnp.float64), eps)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        val = -jnp.square(sigma) / jnp.square(mu)
        return jnp.where(val < -eps, val, -eps) * jnp.ones_like(y)

    def hess_sigma(y, mu, sigma):
        sigma = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float64), eps)
        # R uses constant 1.82368 ≈ trigamma(1) empirical value
        val = -1.82368 / jnp.square(sigma)
        return jnp.where(val < -eps, val, -eps) * jnp.ones_like(y)

    return FamilyJAXSpec(
        name="WEI",
        param_names=("mu", "sigma"),
        link_fns=(log_link, log_link),
        link_inv_fns=(log_inverse, log_inverse),
        link_deriv_fns=(log_derivative, log_derivative),
        score_fns=(score_mu, score_sigma),
        hessian_fns=(hess_mu, hess_sigma),
        loglik_fn=loglik,
        # eta bounds: log mu: exp(-6)..exp(10); log sigma (shape): exp(-3)..exp(4)≈55
        eta_bounds=((-6.0, 10.0), (-3.0, 4.0)),
        init_etas=(math.nan, 0.0),
    )


# ---------------------------------------------------------------------------
# TF — Student-t distribution (3 parameters)
# ---------------------------------------------------------------------------

def make_tf_spec() -> FamilyJAXSpec:
    """Build ``FamilyJAXSpec`` for the Student-t (TF) distribution.

    Parameters: mu (identity), sigma (log), nu (log, degrees of freedom).
    Score and Hessian via JAX autodiff (vmap of grad).
    """
    eps = jnp.finfo(jnp.float64).eps

    def _loglik_scalar(y_s, mu_s, sigma_s, nu_s):
        sigma_s = jnp.maximum(sigma_s, eps)
        nu_s = jnp.maximum(nu_s, eps)
        z = (y_s - mu_s) / sigma_s
        return (
            gammaln((nu_s + 1.0) / 2.0)
            - gammaln(nu_s / 2.0)
            - 0.5 * jnp.log(nu_s * math.pi)
            - jnp.log(sigma_s)
            - ((nu_s + 1.0) / 2.0) * jnp.log1p(jnp.square(z) / nu_s)
        )

    def loglik(y, mu, sigma, nu):
        return jax.vmap(_loglik_scalar)(
            jnp.asarray(y, dtype=jnp.float64),
            jnp.asarray(mu, dtype=jnp.float64),
            jnp.asarray(sigma, dtype=jnp.float64),
            jnp.asarray(nu, dtype=jnp.float64),
        )

    # Vectorised score via vmap of grad
    _grad_mu    = jax.vmap(jax.grad(_loglik_scalar, argnums=1))
    _grad_sigma = jax.vmap(jax.grad(_loglik_scalar, argnums=2))
    _grad_nu    = jax.vmap(jax.grad(_loglik_scalar, argnums=3))
    _hess_mu    = jax.vmap(jax.grad(jax.grad(_loglik_scalar, argnums=1), argnums=1))
    _hess_sigma = jax.vmap(jax.grad(jax.grad(_loglik_scalar, argnums=2), argnums=2))
    _hess_nu    = jax.vmap(jax.grad(jax.grad(_loglik_scalar, argnums=3), argnums=3))

    def _cast(y, mu, sigma, nu):
        return (
            jnp.asarray(y, dtype=jnp.float64),
            jnp.asarray(mu, dtype=jnp.float64),
            jnp.maximum(jnp.asarray(sigma, dtype=jnp.float64), eps),
            jnp.maximum(jnp.asarray(nu, dtype=jnp.float64), eps),
        )

    def score_mu(y, mu, sigma, nu):
        y, mu, sigma, nu = _cast(y, mu, sigma, nu)
        return _grad_mu(y, mu, sigma, nu)

    def score_sigma(y, mu, sigma, nu):
        y, mu, sigma, nu = _cast(y, mu, sigma, nu)
        return _grad_sigma(y, mu, sigma, nu)

    def score_nu(y, mu, sigma, nu):
        y, mu, sigma, nu = _cast(y, mu, sigma, nu)
        return _grad_nu(y, mu, sigma, nu)

    def hess_mu(y, mu, sigma, nu):
        y, mu, sigma, nu = _cast(y, mu, sigma, nu)
        return _hess_mu(y, mu, sigma, nu)

    def hess_sigma(y, mu, sigma, nu):
        y, mu, sigma, nu = _cast(y, mu, sigma, nu)
        return _hess_sigma(y, mu, sigma, nu)

    def hess_nu(y, mu, sigma, nu):
        y, mu, sigma, nu = _cast(y, mu, sigma, nu)
        return _hess_nu(y, mu, sigma, nu)

    # TF nu link: log with clip to avoid extreme values
    def nu_link(nu):
        nu = jnp.maximum(jnp.asarray(nu, dtype=jnp.float64), eps)
        return jnp.log(nu)

    def nu_inverse(eta):
        return jnp.maximum(jnp.exp(jnp.clip(eta, -10.0, 10.0)), eps)

    def nu_derivative(eta):
        return jnp.maximum(jnp.exp(jnp.clip(eta, -10.0, 10.0)), eps)

    return FamilyJAXSpec(
        name="TF",
        param_names=("mu", "sigma", "nu"),
        link_fns=(identity_link, log_link, nu_link),
        link_inv_fns=(identity_inverse, log_inverse, nu_inverse),
        link_deriv_fns=(identity_derivative, log_derivative, nu_derivative),
        score_fns=(score_mu, score_sigma, score_nu),
        hessian_fns=(hess_mu, hess_sigma, hess_nu),
        loglik_fn=loglik,
        # eta bounds: identity mu (wide); log sigma: exp(-6)..exp(6); log nu: exp(-1)..exp(6)
        eta_bounds=((-1e6, 1e6), (-6.0, 6.0), (-1.0, 6.0)),
        init_etas=(math.nan, math.nan, math.log(7.0)),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_SPEC_BUILDERS: dict[str, Callable[[], FamilyJAXSpec]] = {
    "NO":  make_no_spec,
    "GA":  make_ga_spec,
    "PO":  make_po_spec,
    "BI":  make_bi_spec,
    "WEI": make_wei_spec,
    "TF":  make_tf_spec,
}

_SPEC_CACHE: dict[str, FamilyJAXSpec] = {}


def get_jax_spec(family_name: str, **kwargs) -> FamilyJAXSpec:
    """Return a cached ``FamilyJAXSpec`` for the given family name.

    Parameters
    ----------
    family_name : str
        One of ``"NO"``, ``"GA"``, ``"PO"``, ``"BI"``, ``"WEI"``, ``"TF"``.
    **kwargs
        Passed to the spec builder (e.g. ``bd=5`` for ``"BI"``).

    Raises
    ------
    KeyError
        If the family is not supported by the JAX RS core.
    """
    key = family_name if not kwargs else f"{family_name}_{kwargs}"
    if key not in _SPEC_CACHE:
        if family_name not in _SPEC_BUILDERS:
            supported = ", ".join(sorted(_SPEC_BUILDERS))
            raise KeyError(
                f"Family '{family_name}' is not supported by the JAX RS core. "
                f"Supported: {supported}. Use method='RS' for other families."
            )
        builder = _SPEC_BUILDERS[family_name]
        _SPEC_CACHE[key] = builder(**kwargs) if kwargs else builder()
    return _SPEC_CACHE[key]


def supported_families() -> list[str]:
    """Return the list of families supported by the JAX RS core."""
    return sorted(_SPEC_BUILDERS.keys())


__all__ = [
    "FamilyJAXSpec",
    "get_jax_spec",
    "make_no_spec",
    "make_ga_spec",
    "make_po_spec",
    "make_bi_spec",
    "make_wei_spec",
    "make_tf_spec",
    "supported_families",
]
