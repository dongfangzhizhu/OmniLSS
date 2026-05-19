"""Cross-parameter derivative helpers for Cole-Green development."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import jax
import jax.numpy as jnp

from omnilss.cg_derivatives import eta_score_hessian

Array = jnp.ndarray


def cross_hessian(
    log_lik_fn: Callable[[dict[str, Array]], Array],
    params: dict[str, Array],
) -> dict[tuple[str, str], Array]:
    """Return mixed second derivatives d²ll/(d_eta_j d_eta_k).

    Parameters are differentiated with respect to the *summed* scalar
    log-likelihood. The returned matrix entries are keyed by (row, col)
    parameter names.
    """

    keys = tuple(params.keys())

    def scalar_ll(flat_values: tuple[Array, ...]) -> Array:
        mapped = {k: v for k, v in zip(keys, flat_values, strict=True)}
        return jnp.sum(log_lik_fn(mapped))

    hess = jax.hessian(scalar_ll)(tuple(params[k] for k in keys))
    out: dict[tuple[str, str], Array] = {}
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            out[(ki, kj)] = hess[i][j]
    return out


def cross_hessian_from_family(
    y: Any,
    param_values: Mapping[str, Any],
    family: Any,
    parameter_order: tuple[str, ...] | None = None,
) -> dict[tuple[str, str], Array]:
    """Compute per-observation cross derivatives using family definitions.

    Returns entries keyed by parameter pair with values of shape ``(n,)``.
    """
    bundle = eta_score_hessian(
        y=y,
        param_values=param_values,
        family=family,
        parameter_order=parameter_order,
    )
    order = bundle.parameter_order
    hess = bundle.hessian
    return {
        (pi, pj): hess[:, i, j]
        for i, pi in enumerate(order)
        for j, pj in enumerate(order)
    }
