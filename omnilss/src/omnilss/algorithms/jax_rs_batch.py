# SPDX-License-Identifier: GPL-3.0-or-later
"""Batched entry points for JAX-native RS fitting.

The sequential single-model IRLS loop is not where GPUs gain most of their
advantage.  This module provides a batch-oriented API for independent model
fits (bootstrap, cross-validation, or family selection).  Same-family,
same-shape groups use a real ``jax.vmap`` kernel, while mixed-family or
shape-heterogeneous groups keep a CPU-safe sequential fallback without changing
the public API.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .jax_family_specs import FamilyJAXSpec
from .jax_rs_core import (
    JaxRSResult,
    _default_init_etas,
    _irls_step_inline,
    jax_rs_fit_core,
)


def _is_design_tuple(value: Any) -> bool:
    """Return True when ``value`` looks like one model's design tuple."""
    return isinstance(value, (tuple, list)) and bool(value) and all(
        hasattr(item, "shape") and getattr(item, "ndim", None) == 2 for item in value
    )


def _normalize_designs(Xs_per_model: Any, k_models: int) -> list[tuple[jnp.ndarray, ...]]:
    """Normalize a broadcast or per-model design specification."""
    if _is_design_tuple(Xs_per_model):
        design = tuple(jnp.asarray(x, dtype=jnp.float64) for x in Xs_per_model)
        return [design for _ in range(k_models)]

    if not isinstance(Xs_per_model, Sequence) or len(Xs_per_model) != k_models:
        raise ValueError(
            "Xs_per_model must be one design tuple broadcast to all models or "
            f"a sequence of {k_models} design tuples."
        )

    designs: list[tuple[jnp.ndarray, ...]] = []
    for idx, design in enumerate(Xs_per_model):
        if not _is_design_tuple(design):
            raise ValueError(f"Xs_per_model[{idx}] is not a valid design tuple.")
        designs.append(tuple(jnp.asarray(x, dtype=jnp.float64) for x in design))
    return designs


def _normalize_family_specs(
    family_specs: FamilyJAXSpec | Sequence[FamilyJAXSpec], k_models: int
) -> list[FamilyJAXSpec]:
    """Normalize one family spec or one spec per model."""
    if isinstance(family_specs, FamilyJAXSpec):
        return [family_specs for _ in range(k_models)]
    if not isinstance(family_specs, Sequence) or len(family_specs) != k_models:
        raise ValueError(
            "family_specs must be a FamilyJAXSpec or a sequence of one spec per model."
        )
    return list(family_specs)


def _normalize_weights(obs_weights: jnp.ndarray | None, ys: jnp.ndarray) -> jnp.ndarray:
    """Normalize observation weights to shape ``[K, n]``."""
    k_models, n_obs = ys.shape
    if obs_weights is None:
        return jnp.ones((k_models, n_obs), dtype=jnp.float64)

    weights = jnp.asarray(obs_weights, dtype=jnp.float64)
    if weights.ndim == 1:
        if weights.shape[0] != n_obs:
            raise ValueError(f"obs_weights length {weights.shape[0]} != n={n_obs}.")
        return jnp.broadcast_to(weights[None, :], (k_models, n_obs))
    if weights.ndim == 2 and weights.shape == (k_models, n_obs):
        return weights
    raise ValueError(
        "obs_weights must be None, shape [n], or shape [K, n]; "
        f"got shape {weights.shape}."
    )


def _validate_batch_inputs(
    ys: jnp.ndarray,
    designs: Sequence[tuple[jnp.ndarray, ...]],
    specs: Sequence[FamilyJAXSpec],
) -> None:
    """Validate batch dimensions before fitting."""
    _, n_obs = ys.shape
    for idx, (design, spec) in enumerate(zip(designs, specs, strict=True)):
        if len(design) != len(spec.param_names):
            raise ValueError(
                f"Model {idx} expected {len(spec.param_names)} design matrices for "
                f"family {spec.name}, got {len(design)}."
            )
        for param_idx, x_mat in enumerate(design):
            if x_mat.shape[0] != n_obs:
                raise ValueError(
                    f"Model {idx} design {param_idx} has n={x_mat.shape[0]}, "
                    f"expected {n_obs}."
                )


def _same_design_shapes(
    designs: Sequence[tuple[jnp.ndarray, ...]], indices: Sequence[int]
) -> bool:
    """Return True when a group can share one vmapped XLA program."""
    if not indices:
        return False
    reference = tuple(x.shape for x in designs[indices[0]])
    return all(tuple(x.shape for x in designs[idx]) == reference for idx in indices)


def _vmap_same_family_group(
    ys: jnp.ndarray,
    designs: Sequence[tuple[jnp.ndarray, ...]],
    weights: jnp.ndarray,
    spec: FamilyJAXSpec,
    indices: Sequence[int],
    *,
    max_outer: int,
    max_inner: int,
    tol: float,
) -> list[JaxRSResult]:
    """Fit a same-family/same-shape batch with a real ``jax.vmap`` kernel."""
    n_params = len(spec.param_names)
    y_batch = ys[jnp.asarray(indices)]
    weight_batch = weights[jnp.asarray(indices)]
    Xs_batch = tuple(
        jnp.stack([designs[idx][k] for idx in indices]) for k in range(n_params)
    )

    loglik_fn = spec.loglik_fn
    score_fns = spec.score_fns
    hessian_fns = spec.hessian_fns
    link_fns = spec.link_fns
    link_inv_fns = spec.link_inv_fns
    link_deriv_fns = spec.link_deriv_fns
    eta_bounds = spec.eta_bounds

    def _single_fit(y, obs_weights, *Xs):
        init_etas = _default_init_etas(y, spec)
        init_params = jnp.stack(
            [link_inv_fns[k](init_etas[k]) for k in range(n_params)]
        )

        def compute_gdev(params):
            param_list = [params[k] for k in range(n_params)]
            ll = loglik_fn(y, *param_list)
            ll = jnp.where(jnp.isfinite(ll), ll, -1e6)
            return -2.0 * jnp.sum(obs_weights * ll)

        init_gdev = compute_gdev(init_params)
        init_state = (
            init_params,
            init_etas,
            init_gdev,
            init_gdev + 1.0,
            jnp.array(0, dtype=jnp.int32),
        )

        def cond_fn(state):
            _, _, g_dev, g_dev_old, it = state
            return jnp.logical_and(jnp.abs(g_dev_old - g_dev) > tol, it < max_outer)

        def body_fn(state):
            params, etas, g_dev, _, it = state
            new_params = params
            new_etas = etas

            for k in range(n_params):
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
                    eta_clip_scale=jnp.where(it == 0, 3.0, 1e12),
                )
                candidate_gdev = compute_gdev(candidate_params)

                def halve_cond(carry):
                    _eta_try, _params_try, gdev_try, count = carry
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
                    (eta_k_new, candidate_params, candidate_gdev, jnp.array(0, dtype=jnp.int32)),
                )

                new_params, new_etas = jax.lax.cond(
                    gdev_accept > old_gdev_k + 1e-6,
                    lambda _: (old_params_k, old_etas_k),
                    lambda _: (params_accept, old_etas_k.at[k].set(eta_accept)),
                    operand=None,
                )

            new_gdev = compute_gdev(new_params)

            def halve_step(_):
                halved_etas = (etas + new_etas) / 2.0
                halved_params = jnp.stack(
                    [link_inv_fns[k](halved_etas[k]) for k in range(n_params)]
                )
                halved_gdev = compute_gdev(halved_params)
                return jax.lax.cond(
                    halved_gdev >= g_dev,
                    lambda _: (params, etas, g_dev),
                    lambda _: (halved_params, halved_etas, halved_gdev),
                    operand=None,
                )

            final_params, final_etas, final_gdev = jax.lax.cond(
                new_gdev > g_dev + 1e-6,
                halve_step,
                lambda _: (new_params, new_etas, new_gdev),
                operand=None,
            )
            return final_params, final_etas, final_gdev, g_dev, it + 1

        final_params, final_etas, final_gdev, _, final_it = jax.lax.while_loop(
            cond_fn, body_fn, init_state
        )
        betas = tuple(
            jnp.linalg.lstsq(Xs[k], final_etas[k], rcond=None)[0]
            for k in range(n_params)
        )
        return (
            final_params,
            final_etas,
            betas,
            final_gdev,
            final_it,
            final_it < max_outer,
        )

    @jax.jit
    def _run_batch(y_batch, weight_batch, Xs_batch):
        return jax.vmap(_single_fit, in_axes=(0, 0, *([0] * n_params)))(
            y_batch, weight_batch, *Xs_batch
        )

    params_b, etas_b, betas_b, gdev_b, it_b, converged_b = _run_batch(
        y_batch, weight_batch, Xs_batch
    )
    return [
        JaxRSResult(
            params=params_b[local_idx],
            etas=etas_b[local_idx],
            betas=[betas_b[k][local_idx] for k in range(n_params)],
            g_dev=float(gdev_b[local_idx]),
            iterations=int(it_b[local_idx]),
            converged=bool(converged_b[local_idx]),
        )
        for local_idx in range(len(indices))
    ]


def batch_jax_rs_fit(
    ys: jnp.ndarray,
    Xs_per_model: list,
    family_specs: list[FamilyJAXSpec] | FamilyJAXSpec,
    obs_weights: jnp.ndarray | None = None,
    max_outer: int = 20,
    max_inner: int = 1,
    tol: float = 1e-4,
) -> list[JaxRSResult]:
    """Fit K independent GAMLSS models with the JAX RS core.

    Parameters
    ----------
    ys : jnp.ndarray, shape [K, n]
        Batched response arrays.  A one-dimensional array is accepted and
        treated as a batch of size one.
    Xs_per_model : list
        Either one design tuple/list broadcast to all models, or a sequence of
        K design tuples/lists.  Each design tuple is ordered like the matching
        ``FamilyJAXSpec.param_names``.
    family_specs : FamilyJAXSpec or sequence[FamilyJAXSpec]
        One JAX family spec broadcast to all models, or one spec per model.
        Mixed-family batches are grouped by family for deterministic execution.
    obs_weights : jnp.ndarray, optional
        Observation weights with shape [n] or [K, n].
    max_outer, max_inner, tol
        Forwarded to :func:`jax_rs_fit_core`.

    Returns
    -------
    list[JaxRSResult]
        One result per model, in the same order as the input batch.

    Notes
    -----
    Same-family groups with identical design shapes are executed through a real
    ``jax.vmap`` kernel.  Mixed-family or shape-heterogeneous groups fall back
    to a deterministic per-model loop, preserving result semantics on CPU and
    accelerators.
    """
    ys = jnp.asarray(ys, dtype=jnp.float64)
    if ys.ndim == 1:
        ys = ys[None, :]
    if ys.ndim != 2:
        raise ValueError(f"ys must have shape [K, n] or [n], got {ys.shape}.")

    k_models = ys.shape[0]
    designs = _normalize_designs(Xs_per_model, k_models)
    specs = _normalize_family_specs(family_specs, k_models)
    weights = _normalize_weights(obs_weights, ys)
    _validate_batch_inputs(ys, designs, specs)

    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for idx, spec in enumerate(specs):
        grouped_indices[spec.name].append(idx)

    results: list[JaxRSResult | None] = [None] * k_models
    for _family_name, indices in grouped_indices.items():
        if len(indices) > 1 and _same_design_shapes(designs, indices):
            group_results = _vmap_same_family_group(
                ys,
                designs,
                weights,
                specs[indices[0]],
                indices,
                max_outer=max_outer,
                max_inner=max_inner,
                tol=tol,
            )
            for idx, result in zip(indices, group_results, strict=True):
                results[idx] = result
            continue

        for idx in indices:
            results[idx] = jax_rs_fit_core(
                y=ys[idx],
                Xs=designs[idx],
                obs_weights=weights[idx],
                spec=specs[idx],
                max_outer=max_outer,
                max_inner=max_inner,
                tol=tol,
            )

    return [result for result in results if result is not None]


__all__ = ["batch_jax_rs_fit"]
