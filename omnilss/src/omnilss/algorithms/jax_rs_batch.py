# SPDX-License-Identifier: GPL-3.0-or-later
"""Batched entry points for JAX-native RS fitting.

The sequential single-model IRLS loop is not where GPUs gain most of their
advantage.  This module provides a batch-oriented API for independent model
fits (bootstrap, cross-validation, or family selection).  The implementation is
shape- and family-aware and intentionally keeps a CPU-safe sequential fallback;
future GPU-specific kernels can optimize same-family groups without changing the
public API.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np

from .jax_family_specs import FamilyJAXSpec
from .jax_rs_core import JaxRSResult, jax_rs_fit_core


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
    The current implementation uses a deterministic per-family loop that works
    on CPU and GPU.  This preserves the public batch API and result semantics;
    same-family groups are explicit so a later GPU-specific ``jax.vmap`` kernel
    can be dropped in without changing callers.
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
