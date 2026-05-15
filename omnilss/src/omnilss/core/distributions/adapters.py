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

    def parameters(self) -> tuple[Parameter, ...]:
        return parameters_from_names(self.family.parameters)

    def _require(self, attr: str) -> Any:
        fn = getattr(self.family, attr)
        if fn is None:
            raise NotImplementedError(f"Family {self.family.name!r} does not define {attr!r}")
        return fn

    def logpdf(self, y: Any, params: Mapping[str, Any]) -> jnp.ndarray:
        return jnp.asarray(self._require("d")(y, log=True, **dict(params)))

    def cdf(self, y: Any, params: Mapping[str, Any]) -> jnp.ndarray:
        return jnp.asarray(self._require("p")(y, **dict(params)))

    def ppf(self, q: Any, params: Mapping[str, Any]) -> jnp.ndarray:
        return jnp.asarray(self._require("q")(q, **dict(params)))

    def sample(self, key: Any, params: Mapping[str, Any], shape: tuple[int, ...] = ()) -> jnp.ndarray:
        n = int(np.prod(shape)) if shape else 1
        draws = jnp.asarray(self._require("r")(key, n, **dict(params)))
        return jnp.reshape(draws, shape) if shape else draws

    def score(self, y: Any, params: Mapping[str, Any]) -> Mapping[str, jnp.ndarray]:
        if self.family.score_functions:
            return {
                name: jnp.asarray(fn(y, **dict(params)))
                for name, fn in self.family.score_functions.items()
            }

        def scalar_loglik(flat_params: Mapping[str, Any]) -> jnp.ndarray:
            return jnp.sum(self.logpdf(y, flat_params))

        return jax.grad(scalar_loglik)(dict(params))

    def hessian(self, y: Any, params: Mapping[str, Any]) -> Mapping[str, Mapping[str, jnp.ndarray]]:
        if self.family.hessian_functions:
            return {
                name: {name: jnp.asarray(fn(y, **dict(params)))}
                for name, fn in self.family.hessian_functions.items()
            }

        def scalar_loglik(flat_params: Mapping[str, Any]) -> jnp.ndarray:
            return jnp.sum(self.logpdf(y, flat_params))

        return jax.hessian(scalar_loglik)(dict(params))

    def init_params(self, y: Any) -> Mapping[str, jnp.ndarray]:
        return {parameter.name: parameter.initialize(y) for parameter in self.parameters()}

    def parameter_constraints(self) -> Mapping[str, Constraint]:
        return {parameter.name: parameter.constraint for parameter in self.parameters()}

    def links(self) -> Mapping[str, Link]:
        return {parameter.name: parameter.link for parameter in self.parameters()}


def as_distribution_protocol(family: FamilyDefinition) -> FamilyDistributionAdapter:
    """Wrap a legacy family definition in the canonical distribution protocol."""
    return FamilyDistributionAdapter(family)


__all__ = ["FamilyDistributionAdapter", "as_distribution_protocol"]
