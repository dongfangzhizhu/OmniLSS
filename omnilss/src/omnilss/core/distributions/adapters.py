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
