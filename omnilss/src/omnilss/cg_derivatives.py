"""Eta-scale derivative kernels for the Cole-Green algorithm.

The helpers in this module compute per-observation first and second derivatives
of the log-likelihood with respect to distribution-parameter linear predictors
(`eta_mu`, `eta_sigma`, ...).  They are intentionally independent of the RS
working-response code so they can be used both as CG correctness checks and as
building blocks for an eta-level cross-derivative CG backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class EtaDerivativeBundle:
    """Per-observation eta-scale CG derivatives.

    Attributes
    ----------
    parameter_order:
        Ordered estimable parameter names represented by the derivative axes.
    eta:
        Matrix of current linear predictors with shape ``(n, k)``.
    score:
        Per-observation score matrix ``d l_i / d eta_k`` with shape ``(n, k)``.
    hessian:
        Per-observation Hessian tensor ``d² l_i / d eta_k d eta_j`` with shape
        ``(n, k, k)``.  Off-diagonal entries are the CG cross derivatives.
    """

    parameter_order: tuple[str, ...]
    eta: jnp.ndarray
    score: jnp.ndarray
    hessian: jnp.ndarray


def _as_observation_vector(value: Any, n: int) -> jnp.ndarray:
    """Return ``value`` as a length-``n`` JAX vector."""
    arr = jnp.asarray(value, dtype=jnp.float64)
    if arr.ndim == 0:
        return jnp.full((n,), arr, dtype=jnp.float64)
    if arr.shape[0] != n:
        raise ValueError(
            f"expected parameter vector of length {n}, got shape {arr.shape}"
        )
    return arr


def _family_logpdf_from_eta(
    family: Any,
    parameter_order: Sequence[str],
    all_parameters: Sequence[str],
    eta_row: jnp.ndarray,
    fitted_row: jnp.ndarray,
    y_i: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate a scalar family log-density for one observation."""
    eta_index = {name: idx for idx, name in enumerate(parameter_order)}
    values = []
    for param_idx, parameter in enumerate(all_parameters):
        if parameter in eta_index:
            eta_value = eta_row[eta_index[parameter]]
            if (
                getattr(family, "link_inverses", None)
                and parameter in family.link_inverses
            ):
                values.append(family.link_inverses[parameter](eta_value))
            elif parameter in {"sigma", "tau"}:
                values.append(jnp.exp(eta_value))
            else:
                values.append(eta_value)
        else:
            values.append(fitted_row[param_idx])

    if getattr(family, "d", None) is None:
        raise AttributeError(
            f"Family {family.name!r} does not define a density function"
        )
    return jnp.squeeze(family.d(y_i, *values, log=True))


def eta_score_hessian(
    y: Any,
    param_values: Mapping[str, Any],
    family: Any,
    parameter_order: Sequence[str] | None = None,
) -> EtaDerivativeBundle:
    """Compute per-observation eta-scale scores and Hessians.

    Parameters
    ----------
    y:
        Response vector.
    param_values:
        Current fitted distribution-parameter values.  Every family parameter
        must be present; scalar fixed values are broadcast to the observation
        count.
    family:
        OmniLSS family definition.
    parameter_order:
        Estimable parameters to differentiate.  Defaults to
        ``family.estimable_parameters``.

    Returns
    -------
    EtaDerivativeBundle
        Score and Hessian tensors.  The Hessian off-diagonal blocks are the
        cross derivatives required by CG.
    """
    y_vec = jnp.asarray(y, dtype=jnp.float64)
    if y_vec.ndim == 0:
        y_vec = y_vec.reshape((1,))
    n = int(y_vec.shape[0])
    order = tuple(parameter_order or family.estimable_parameters)
    all_parameters = tuple(family.parameters)

    missing = [p for p in all_parameters if p not in param_values]
    if missing:
        raise KeyError(
            f"missing current fitted values for family parameters: {missing}"
        )

    fitted_columns = [
        _as_observation_vector(param_values[p], n) for p in all_parameters
    ]
    fitted_matrix = jnp.stack(fitted_columns, axis=1)

    eta_columns = []
    for parameter in order:
        values = _as_observation_vector(param_values[parameter], n)
        if (
            getattr(family, "link_functions", None)
            and parameter in family.link_functions
        ):
            eta_columns.append(family.link_functions[parameter](values))
        elif parameter in {"sigma", "tau"}:
            eta_columns.append(jnp.log(jnp.maximum(values, jnp.finfo(jnp.float64).eps)))
        else:
            eta_columns.append(values)
    eta_matrix = jnp.stack(eta_columns, axis=1)

    def logpdf_eta(
        eta_row: jnp.ndarray, fitted_row: jnp.ndarray, y_i: jnp.ndarray
    ) -> jnp.ndarray:
        return _family_logpdf_from_eta(
            family=family,
            parameter_order=order,
            all_parameters=all_parameters,
            eta_row=eta_row,
            fitted_row=fitted_row,
            y_i=y_i,
        )

    score_fn = jax.grad(logpdf_eta, argnums=0)
    hessian_fn = jax.hessian(logpdf_eta, argnums=0)
    score = jax.vmap(score_fn)(eta_matrix, fitted_matrix, y_vec)
    hess = jax.vmap(hessian_fn)(eta_matrix, fitted_matrix, y_vec)
    hess = 0.5 * (hess + jnp.swapaxes(hess, -1, -2))
    return EtaDerivativeBundle(
        parameter_order=order,
        eta=eta_matrix,
        score=jnp.where(jnp.isfinite(score), score, 0.0),
        hessian=jnp.where(jnp.isfinite(hess), hess, 0.0),
    )


def eta_cross_hessian(
    y: Any,
    param_values: Mapping[str, Any],
    family: Any,
    parameter_order: Sequence[str] | None = None,
) -> jnp.ndarray:
    """Return the per-observation eta Hessian tensor for CG cross derivatives."""
    return eta_score_hessian(
        y=y,
        param_values=param_values,
        family=family,
        parameter_order=parameter_order,
    ).hessian
