"""Adapters that let existing ``FamilyDefinition`` objects participate in the new protocol.

The adapter is a migration bridge only.  It prevents immediate large-scale file
moves while giving tests and new code a single target contract.
"""

from __future__ import annotations

from collections.abc import Mapping
import inspect
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ...families import FamilyDefinition
from ..constraints import Constraint
from ..links import Link
from ..params import Parameter, parameters_from_names


@dataclass(frozen=True)
class FamilyDistributionAdapter:
    """Stateless protocol adapter for legacy ``FamilyDefinition`` instances."""

    family: FamilyDefinition

    @property
    def name(self) -> str:
        return self.family.name

    def _estimable_parameter_names(self) -> tuple[str, ...]:
        """Return family parameters that participate in optimization.

        Some legacy families include fixed data parameters (for example the
        beta-binomial denominator ``bd``) in ``family.parameters`` because the
        density needs them.  The canonical parameter protocol only describes
        estimable model parameters, so fixed data parameters are accepted in
        ``params`` for likelihood evaluation but excluded from links,
        constraints, scores, Hessians, and initialization.
        """
        return self.family.estimable_parameters

    def parameters(self) -> tuple[Parameter, ...]:
        return parameters_from_names(self._estimable_parameter_names())

    def _require(self, attr: str) -> Any:
        fn = getattr(self.family, attr)
        if fn is None:
            raise NotImplementedError(
                f"Family {self.family.name!r} does not define {attr!r}"
            )
        return fn

    def _ordered_values(self, params: Mapping[str, Any]) -> tuple[Any, ...]:
        return tuple(params[name] for name in self.family.parameters if name in params)

    def _supports_keyword_params(self, fn: Any, params: Mapping[str, Any]) -> bool:
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            return False

        signature_parameters = signature.parameters
        if any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature_parameters.values()
        ):
            return True
        return all(name in signature_parameters for name in params)

    def _call_dpqr(
        self,
        fn: Any,
        leading_args: tuple[Any, ...],
        params: Mapping[str, Any],
        extra_kwargs: Mapping[str, Any] | None = None,
    ) -> Any:
        extra_kwargs = {} if extra_kwargs is None else dict(extra_kwargs)
        if self._supports_keyword_params(fn, params):
            return fn(*leading_args, **dict(params), **extra_kwargs)
        return fn(*leading_args, *self._ordered_values(params), **extra_kwargs)

    def logpdf(self, y: Any, params: Mapping[str, Any]) -> jnp.ndarray:
        result = self._call_dpqr(self._require("d"), (y,), params, {"log": True})
        return jnp.asarray(result)

    def cdf(self, y: Any, params: Mapping[str, Any]) -> jnp.ndarray:
        result = self._call_dpqr(self._require("p"), (y,), params)
        return jnp.asarray(result)

    def ppf(self, q: Any, params: Mapping[str, Any]) -> jnp.ndarray:
        result = self._call_dpqr(self._require("q"), (q,), params)
        return jnp.asarray(result)

    def sample(
        self, key: Any, params: Mapping[str, Any], shape: tuple[int, ...] = ()
    ) -> jnp.ndarray:
        n = int(np.prod(shape)) if shape else 1
        draws = jnp.asarray(self._call_dpqr(self._require("r"), (key, n), params))
        return jnp.reshape(draws, shape) if shape else draws

    def _row_inputs(
        self, y: Any, params: Mapping[str, Any], names: tuple[str, ...]
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Flatten ``y`` to one row per observation and broadcast the parameters.

        The canonical derivative contract is per observation (shape ``(n,)``);
        summing first would make the result a scalar whose shape depends on
        which branch of the family implementation ran.
        """
        rows = jnp.reshape(jnp.asarray(y), (-1,))
        count = int(rows.shape[0])
        row_params = {
            name: jnp.broadcast_to(
                jnp.asarray(params[name]),
                (count,) + jnp.asarray(params[name]).shape[1:],
            )
            for name in names
        }
        return rows, row_params

    def _row_loglik(
        self, y_row: Any, row_params: Mapping[str, Any], params: Mapping[str, Any]
    ) -> jnp.ndarray:
        likelihood_params = dict(params)
        likelihood_params.update(row_params)
        return self.logpdf(y_row, likelihood_params)

    def score(self, y: Any, params: Mapping[str, Any]) -> Mapping[str, jnp.ndarray]:
        """Per-observation d log-likelihood / d parameter (shape ``(n,)``)."""
        estimable = tuple(
            name for name in self._estimable_parameter_names() if name in params
        )
        if not estimable:
            return {}
        # Fixed data parameters (``bd``, ``nu`` flags, ...) stay in the row dict so
        # the density sees one value per row; they are simply dropped afterwards.
        rows, row_params = self._row_inputs(y, params, tuple(params))
        grads = jax.vmap(
            jax.grad(
                lambda y_row, row: self._row_loglik(y_row, row, params), argnums=1
            ),
            in_axes=(0, 0),
        )(rows, row_params)
        result = {name: jnp.asarray(grads[name]) for name in estimable}
        for name, fn in (self.family.score_functions or {}).items():
            if name in result:
                analytic = jnp.asarray(fn(y=y, **dict(params)))
                result[name] = jnp.broadcast_to(analytic, result[name].shape)
        return result

    def hessian(
        self, y: Any, params: Mapping[str, Any]
    ) -> Mapping[str, Mapping[str, jnp.ndarray]]:
        """Per-observation d^2 log-likelihood / d parameter^2, including cross terms.

        Every estimable parameter present in ``params`` appears as a key, so the
        ondil ``dl2_dpp`` cross-derivative can be mapped without special cases;
        analytic diagonal entries from ``hessian_functions`` override the
        autodiff value (they are the same quantity, checked in the contract
        tests).
        """
        estimable = tuple(
            name for name in self._estimable_parameter_names() if name in params
        )
        if not estimable:
            return {}
        rows, row_params = self._row_inputs(y, params, tuple(params))
        per_row = jax.vmap(
            jax.hessian(
                lambda y_row, row: self._row_loglik(y_row, row, params), argnums=1
            ),
            in_axes=(0, 0),
        )(rows, row_params)
        result = {
            first: {second: jnp.asarray(per_row[first][second]) for second in estimable}
            for first in estimable
        }
        for name, fn in (self.family.hessian_functions or {}).items():
            if name in result:
                analytic = jnp.asarray(fn(y=y, **dict(params)))
                result[name][name] = jnp.broadcast_to(
                    analytic, result[name][name].shape
                )
        return result

    def init_params(self, y: Any) -> Mapping[str, jnp.ndarray]:
        return {
            parameter.name: parameter.initialize(y) for parameter in self.parameters()
        }

    def parameter_constraints(self) -> Mapping[str, Constraint]:
        return {parameter.name: parameter.constraint for parameter in self.parameters()}

    def links(self) -> Mapping[str, Link]:
        return {parameter.name: parameter.link for parameter in self.parameters()}


def as_distribution_protocol(family: FamilyDefinition) -> FamilyDistributionAdapter:
    """Wrap a legacy family definition in the canonical distribution protocol."""
    return FamilyDistributionAdapter(family)


__all__ = ["FamilyDistributionAdapter", "as_distribution_protocol"]
