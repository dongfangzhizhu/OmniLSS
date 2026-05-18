# SPDX-License-Identifier: GPL-3.0-or-later
"""JAX-native RS (Rigby-Stasinopoulos) fitting core.

This module provides a fully JAX-traced implementation of the RS algorithm
that runs end-to-end on GPU/TPU without host-device round-trips.

Architecture
------------
The outer convergence loop uses ``jax.lax.while_loop`` so the entire
fitting procedure compiles to a single XLA computation.  The inner IRLS
step for each parameter uses ``jax.lax.fori_loop`` with a fixed iteration
count (``max_inner``), which avoids dynamic shapes inside the JIT boundary.

Key design decisions
--------------------
* **Params layout**: a single ``jnp.ndarray`` of shape ``[n_params, n]``
  holds all parameter arrays.  Axis 0 indexes parameters in the order
  defined by ``FamilyJAXSpec.param_names``.
* **Design matrices**: a tuple of arrays ``(X0, X1, ...)`` one per
  parameter.  Shapes are ``[n, p_k]`` where ``p_k`` may differ.
* **Static shapes**: ``p_k`` (columns per parameter) must be known at
  compile time.  They are passed as ``static_argnames`` to ``jax.jit``.
* **No Python objects inside JIT**: all family math is captured in
  ``FamilyJAXSpec`` closures before tracing.
* **No nested JIT**: IRLS logic is inlined directly into the while_loop
  body to avoid nested jit/while_loop issues.

Public API
----------
``jax_rs_fit_core``
    The main JIT-compiled fitting function.
``JaxRSResult``
    Named tuple returned by ``jax_rs_fit_core``.
"""

from __future__ import annotations

from typing import NamedTuple
import math

import jax
import jax.numpy as jnp

from .jax_family_specs import FamilyJAXSpec
from ..links import log_link, logit_link


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class JaxRSResult(NamedTuple):
    """Output of ``jax_rs_fit_core``.

    Attributes
    ----------
    params : jnp.ndarray
        Final parameter arrays, shape ``[n_params, n]``.
    etas : jnp.ndarray
        Final linear predictors, shape ``[n_params, n]``.
    betas : list[jnp.ndarray]
        Final coefficient vectors, one per parameter.
    g_dev : float
        Final global deviance (scalar).
    iterations : int
        Number of outer iterations performed.
    converged : bool
        Whether the outer loop converged within ``max_outer``.
    """

    params: jnp.ndarray
    etas: jnp.ndarray
    betas: list
    g_dev: float
    iterations: int
    converged: bool


# ---------------------------------------------------------------------------
# Core: one IRLS step for parameter k (pure function, no @jax.jit)
# ---------------------------------------------------------------------------

def _irls_step_inline(
    y, X_k, eta_k, params, obs_weights,
    score_fn, hessian_fn, link_fn, link_inv, link_deriv,
    param_idx, n_params, max_inner,
    eta_lo=-10.0, eta_hi=10.0,
    eta_clip_scale=3.0,
):
    """Run ``max_inner`` IRLS iterations for parameter ``param_idx``.

    This is a pure function (no @jax.jit decorator) designed to be called
    inside ``jax.lax.while_loop`` body.  All JAX transformations are
    applied by the outer while_loop JIT.

    Returns
    -------
    eta_new : [n]
    params_new : [n_params, n]
    """
    eps = jnp.finfo(jnp.float64).eps

    def body(i, carry):
        eta_c, params_c = carry

        # Unpack params into positional args
        param_list = [params_c[k] for k in range(n_params)]

        score = score_fn(y, *param_list)
        hess  = hessian_fn(y, *param_list)

        # Numerical safety
        score = jnp.where(jnp.isfinite(score), score, 0.0)
        hess  = jnp.where(hess < -eps, hess, -eps)

        # dtheta/deta (inverse link derivative)
        dtheta_deta = link_deriv(eta_c)
        dtheta_deta = jnp.where(jnp.abs(dtheta_deta) < eps, eps, dtheta_deta)
        deta_dtheta = 1.0 / dtheta_deta

        # Working weights: w = -hess * (dtheta/deta)^2
        ww = -hess * jnp.square(dtheta_deta)
        ww = jnp.clip(ww, 1e-10, 1e10)
        ww = jnp.where(jnp.isfinite(ww), ww, 1e-10)

        # Working response: z = eta + score / (deta/dtheta * ww)
        denom = deta_dtheta * ww
        denom = jnp.where(jnp.abs(denom) < eps, eps, denom)
        z = eta_c + score / denom
        z = jnp.where(jnp.isfinite(z), z, eta_c)

        # Weighted least squares: solve (X^T W X) beta = X^T W z
        W = ww * obs_weights
        sqrt_W = jnp.sqrt(jnp.maximum(W, 0.0))
        X_w = X_k * sqrt_W[:, None]   # [n, p]
        z_w = z * sqrt_W              # [n]

        beta, _, _, _ = jnp.linalg.lstsq(X_w, z_w, rcond=None)
        beta = jax.lax.cond(
            i == 0,
            lambda b: jnp.clip(b, -eta_clip_scale, eta_clip_scale),
            lambda b: b,
            beta,
        )

        # Update eta and the parameter value in params
        eta_new = X_k @ beta
        # Clip eta to the per-parameter safe range.
        # This prevents overflow in link_inv and keeps the IRLS stable.
        # The bounds are set conservatively in FamilyJAXSpec.eta_bounds.
        eta_new = jnp.clip(eta_new, eta_lo, eta_hi)
        theta_new = link_inv(eta_new)
        # Safety: replace non-finite params with previous values
        theta_new = jnp.where(jnp.isfinite(theta_new), theta_new, params_c[param_idx])
        params_new = params_c.at[param_idx].set(theta_new)
        # Recompute eta from clipped theta to keep eta/theta consistent
        eta_new = link_fn(theta_new)

        return eta_new, params_new

    eta_final, params_final = jax.lax.fori_loop(
        0, max_inner, body, (eta_k, params)
    )
    return eta_final, params_final




def _safe_log_value(value):
    eps = jnp.finfo(jnp.float64).eps
    return jnp.log(jnp.maximum(value, eps))


def _default_init_etas(y: jnp.ndarray, spec: FamilyJAXSpec) -> jnp.ndarray:
    """Build data-aware cold-start etas when callers do not provide them."""
    eps = jnp.finfo(jnp.float64).eps
    y_mean = jnp.mean(y)
    y_pos_mean = jnp.mean(jnp.maximum(y, eps))
    y_std = jnp.maximum(jnp.std(y), eps)
    etas = []
    for k, param in enumerate(spec.param_names):
        configured = spec.init_etas[k] if k < len(spec.init_etas) else jnp.nan
        if math.isnan(float(configured)):
            if param == "mu":
                if spec.link_fns[k] is log_link:
                    eta0 = _safe_log_value(y_pos_mean)
                elif spec.link_fns[k] is logit_link:
                    eta0 = jnp.asarray(0.0, dtype=jnp.float64)
                else:
                    eta0 = y_mean
            elif param == "sigma":
                eta0 = _safe_log_value(y_std)
            else:
                lo, hi = spec.eta_bounds[k] if k < len(spec.eta_bounds) else (-10.0, 10.0)
                eta0 = jnp.asarray((lo + hi) / 2.0, dtype=jnp.float64)
        else:
            eta0 = jnp.asarray(configured, dtype=jnp.float64)
        etas.append(jnp.full(y.shape, eta0, dtype=jnp.float64))
    return jnp.stack(etas, axis=0)

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def jax_rs_fit_core(
    y: jnp.ndarray,
    Xs: tuple,
    init_params: jnp.ndarray | None = None,
    init_etas: jnp.ndarray | None = None,
    obs_weights: jnp.ndarray | None = None,
    spec: FamilyJAXSpec | None = None,
    max_outer: int = 20,
    max_inner: int = 1,
    tol: float = 1e-4,
    eta_clip_scale: float = 3.0,
) -> JaxRSResult:
    """Fit a GAMLSS model using the JAX-native RS algorithm.

    This function compiles the entire fitting loop to XLA and runs it
    on the available device (CPU/GPU/TPU) without host-device round-trips.

    Parameters
    ----------
    y : jnp.ndarray, shape [n]
        Response variable.
    Xs : tuple of jnp.ndarray
        Design matrices, one per estimable parameter.  ``Xs[k]`` has shape
        ``[n, p_k]``.  Must be in the same order as ``spec.param_names``.
    init_params : jnp.ndarray, shape [n_params, n]
        Initial parameter values.  Row ``k`` corresponds to
        ``spec.param_names[k]``.
    init_etas : jnp.ndarray, shape [n_params, n]
        Initial linear predictors (``g(init_params[k])``).
    obs_weights : jnp.ndarray, shape [n]
        Observation weights (use ``jnp.ones(n)`` for unweighted).
    spec : FamilyJAXSpec
        Family specification from ``jax_family_specs.get_jax_spec``.
    max_outer : int, default 20
        Maximum outer RS iterations.
    max_inner : int, default 1
        Fixed inner IRLS iterations per parameter per outer step.
        The default intentionally matches the stable RS update cadence;
        larger values can oscillate for some families.
    tol : float, default 1e-4
        Convergence tolerance on absolute change in global deviance.

    Returns
    -------
    JaxRSResult
        Named tuple with ``params``, ``etas``, ``betas``, ``g_dev``,
        ``iterations``, ``converged``.

    Notes
    -----
    The first call triggers JIT compilation (cold time).  Subsequent calls
    with the same shapes and ``spec`` reuse the compiled XLA computation
    (warm time).

    For families not in ``supported_families()``, use ``method='RS'``
    (the NumPy-based path) instead.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from omnilss.algorithms.jax_family_specs import get_jax_spec
    >>> from omnilss.algorithms.jax_rs_core import jax_rs_fit_core
    >>>
    >>> spec = get_jax_spec("NO")
    >>> n = 1000
    >>> y = jnp.array(...)          # [n]
    >>> X = jnp.ones((n, 1))        # intercept-only
    >>> Xs = (X, X)                 # one per parameter (mu, sigma)
    >>> init_params = jnp.stack([jnp.full(n, y.mean()),
    ...                           jnp.full(n, y.std())])
    >>> init_etas = jnp.stack([spec.link_fns[0](init_params[0]),
    ...                         spec.link_fns[1](init_params[1])])
    >>> result = jax_rs_fit_core(y, Xs, init_params, init_etas,
    ...                          jnp.ones(n), spec)
    >>> print(result.g_dev)
    """
    if spec is None:
        raise ValueError("spec must be provided.")

    y = jnp.asarray(y, dtype=jnp.float64)
    n_params = len(spec.param_names)
    n_obs = y.shape[0]

    if obs_weights is None:
        obs_weights = jnp.ones(n_obs, dtype=jnp.float64)
    else:
        obs_weights = jnp.asarray(obs_weights, dtype=jnp.float64)

    if init_etas is None:
        init_etas = _default_init_etas(y, spec)
    else:
        init_etas = jnp.asarray(init_etas, dtype=jnp.float64)

    if init_params is None:
        init_params = jnp.stack([
            spec.link_inv_fns[k](init_etas[k]) for k in range(n_params)
        ])
    else:
        init_params = jnp.asarray(init_params, dtype=jnp.float64)

    if len(Xs) != n_params:
        raise ValueError(
            f"Expected {n_params} design matrices for spec '{spec.name}', "
            f"got {len(Xs)}."
        )
    if init_params.shape[0] != n_params:
        raise ValueError(
            f"init_params.shape[0]={init_params.shape[0]} != n_params={n_params}."
        )

    # Capture spec components as local variables for closure
    loglik_fn   = spec.loglik_fn
    score_fns   = spec.score_fns
    hessian_fns = spec.hessian_fns
    link_fns       = spec.link_fns
    link_inv_fns   = spec.link_inv_fns
    link_deriv_fns = spec.link_deriv_fns
    eta_bounds     = spec.eta_bounds  # tuple of (lo, hi) per parameter

    def compute_gdev(params):
        param_list = [params[k] for k in range(n_params)]
        ll = loglik_fn(y, *param_list)
        ll = jnp.where(jnp.isfinite(ll), ll, -1e6)
        return -2.0 * jnp.sum(obs_weights * ll)

    # Build the outer while_loop as a single @jax.jit function
    @jax.jit
    def _run(init_params, init_etas, obs_weights):
        init_gdev = compute_gdev(init_params)

        # State: (params, etas, g_dev, g_dev_old, iteration)
        init_state = (
            init_params,
            init_etas,
            init_gdev,
            init_gdev + 1.0,   # g_dev_old > g_dev to enter loop
            jnp.array(0, dtype=jnp.int32),
        )

        def cond_fn(state):
            _, _, g_dev, g_dev_old, it = state
            not_converged = jnp.abs(g_dev_old - g_dev) > tol
            not_maxed = it < max_outer
            return jnp.logical_and(not_converged, not_maxed)

        def body_fn(state):
            params, etas, g_dev, _, it = state

            new_params = params
            new_etas   = etas

            # Update each parameter in sequence (Python loop = static unroll)
            for k in range(n_params):
                # Get eta bounds for this parameter
                if k < len(eta_bounds):
                    eta_lo, eta_hi = eta_bounds[k]
                else:
                    eta_lo, eta_hi = -10.0, 10.0

                old_params_k = new_params
                old_etas_k = new_etas
                old_gdev_k = compute_gdev(old_params_k)

                eta_k_new, candidate_params = _irls_step_inline(
                    y=y,
                    X_k=Xs[k],
                    eta_k=old_etas_k[k],
                    params=old_params_k,
                    obs_weights=obs_weights,
                    score_fn=score_fns[k],
                    hessian_fn=hessian_fns[k],
                    link_fn=link_fns[k],
                    link_inv=link_inv_fns[k],
                    link_deriv=link_deriv_fns[k],
                    param_idx=k,
                    n_params=n_params,
                    max_inner=max_inner,
                    eta_lo=eta_lo,
                    eta_hi=eta_hi,
                    eta_clip_scale=jnp.where(it == 0, eta_clip_scale, 1e12),
                )

                # Per-parameter step-halving mirrors the NumPy RS path: a
                # single cold-start update can be too aggressive even when the
                # overall direction is useful.  Halve only this parameter's eta
                # until the global deviance no longer increases; otherwise keep
                # the previous parameter value.
                candidate_gdev = compute_gdev(candidate_params)

                def halve_cond(carry):
                    eta_try, params_try, gdev_try, count = carry
                    return jnp.logical_and(gdev_try > old_gdev_k + 1e-6, count < 8)

                def halve_body(carry):
                    eta_try, _params_try, _gdev_try, count = carry
                    eta_halved = 0.5 * (old_etas_k[k] + eta_try)
                    theta_halved = link_inv_fns[k](eta_halved)
                    params_halved = old_params_k.at[k].set(theta_halved)
                    gdev_halved = compute_gdev(params_halved)
                    return eta_halved, params_halved, gdev_halved, count + 1

                eta_accept, params_accept, gdev_accept, _ = jax.lax.while_loop(
                    halve_cond,
                    halve_body,
                    (
                        eta_k_new,
                        candidate_params,
                        candidate_gdev,
                        jnp.array(0, dtype=jnp.int32),
                    ),
                )

                def keep_old_param(_):
                    return old_params_k, old_etas_k

                def use_new_param(_):
                    return params_accept, old_etas_k.at[k].set(eta_accept)

                new_params, new_etas = jax.lax.cond(
                    gdev_accept > old_gdev_k + 1e-6,
                    keep_old_param,
                    use_new_param,
                    operand=None,
                )

            new_gdev = compute_gdev(new_params)

            # Step-halving: if deviance increased, average old and new etas
            # This prevents divergence when IRLS overshoots
            def halve_step(_):
                halved_etas   = (etas + new_etas) / 2.0
                halved_params = jnp.stack([
                    link_inv_fns[k](halved_etas[k]) for k in range(n_params)
                ])
                halved_gdev = compute_gdev(halved_params)
                # If halved step is still worse, keep old params (no update)
                def keep_old(_):
                    return params, etas, g_dev
                def use_halved(_):
                    return halved_params, halved_etas, halved_gdev
                return jax.lax.cond(
                    halved_gdev >= g_dev,
                    keep_old,
                    use_halved,
                    operand=None,
                )

            def keep_step(_):
                return new_params, new_etas, new_gdev

            final_params, final_etas, final_gdev = jax.lax.cond(
                new_gdev > g_dev + 1e-6,
                halve_step,
                keep_step,
                operand=None,
            )

            return (final_params, final_etas, final_gdev, g_dev, it + 1)

        return jax.lax.while_loop(cond_fn, body_fn, init_state)

    final_params, final_etas, final_gdev, _, final_it = _run(
        init_params, init_etas, obs_weights
    )

    # Extract beta vectors from final etas (one WLS solve per parameter)
    betas = []
    for k in range(n_params):
        X_k   = Xs[k]
        eta_k = final_etas[k]
        beta_k, _, _, _ = jnp.linalg.lstsq(X_k, eta_k, rcond=None)
        betas.append(beta_k)

    return JaxRSResult(
        params=final_params,
        etas=final_etas,
        betas=betas,
        g_dev=float(final_gdev),
        iterations=int(final_it),
        converged=int(final_it) < max_outer,
    )


__all__ = [
    "JaxRSResult",
    "jax_rs_fit_core",
]
