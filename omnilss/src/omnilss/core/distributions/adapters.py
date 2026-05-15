"""Adapters that let existing ``FamilyDefinition`` objects participate in the new protocol.

The adapter is a migration bridge only.  It prevents immediate large-scale file
moves while giving tests and new code a single target contract.
"""

from __future__ import annotations

from collections.abc import Mapping
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

    def logpdf(self, y: Any, params: Mapping[str, Any]) -> jnp.ndarray:
        fn = self._require("d")
        try:
            result = fn(y, log=True, **dict(params))
        except TypeError:
            result = fn(y, *self._ordered_values(params), log=True)
        return jnp.asarray(result)

    def cdf(self, y: Any, params: Mapping[str, Any]) -> jnp.ndarray:
        fn = self._require("p")
        try:
            result = fn(y, **dict(params))
        except TypeError:
            result = fn(y, *self._ordered_values(params))
        return jnp.asarray(result)

    def ppf(self, q: Any, params: Mapping[str, Any]) -> jnp.ndarray:
        fn = self._require("q")
        try:
            result = fn(q, **dict(params))
        except TypeError:
            result = fn(q, *self._ordered_values(params))
        return jnp.asarray(result)

    def sample(
        self, key: Any, params: Mapping[str, Any], shape: tuple[int, ...] = ()
    ) -> jnp.ndarray:
        n = int(np.prod(shape)) if shape else 1
        fn = self._require("r")
        try:
            draws = jnp.asarray(fn(key, n, **dict(params)))
        except TypeError:
            draws = jnp.asarray(fn(key, n, *self._ordered_values(params)))
        return jnp.reshape(draws, shape) if shape else draws

    def score(self, y: Any, params: Mapping[str, Any]) -> Mapping[str, jnp.ndarray]:
        estimable = set(self._estimable_parameter_names())
        if self.family.score_functions:
            return {
                name: jnp.asarray(fn(y=y, **dict(params)))
                for name, fn in self.family.score_functions.items()
                if name in estimable
            }

        def scalar_loglik(flat_params: Mapping[str, Any]) -> jnp.ndarray:
            likelihood_params = dict(params)
            likelihood_params.update(flat_params)
            return jnp.sum(self.logpdf(y, likelihood_params))

        differentiable_params = {
            name: params[name] for name in estimable if name in params
        }
        return jax.grad(scalar_loglik)(differentiable_params)

    def hessian(
        self, y: Any, params: Mapping[str, Any]
    ) -> Mapping[str, Mapping[str, jnp.ndarray]]:
        estimable = set(self._estimable_parameter_names())
        if self.family.hessian_functions:
            return {
                name: {name: jnp.asarray(fn(y=y, **dict(params)))}
                for name, fn in self.family.hessian_functions.items()
                if name in estimable
            }

        def scalar_loglik(flat_params: Mapping[str, Any]) -> jnp.ndarray:
            likelihood_params = dict(params)
            likelihood_params.update(flat_params)
            return jnp.sum(self.logpdf(y, likelihood_params))

        differentiable_params = {
            name: params[name] for name in estimable if name in params
        }
        return jax.hessian(scalar_loglik)(differentiable_params)

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
