"""Canonical stateless parameter system for OmniLSS.

The ``Parameter`` object captures name, link, constraint, and initialization in a
single PyTree-compatible frozen dataclass.  It is intentionally generic so
families no longer need distribution-specific parameter hacks.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from ..constraints import Constraint, Positive, Real, UnitInterval
from ..links import IdentityLink, Link, LogLink, LogitLink

Initializer = Callable[[Any], Any]


def _mean_initializer(data: Any) -> jnp.ndarray:
    return jnp.asarray(jnp.mean(jnp.asarray(data)))


def _ones_initializer(data: Any) -> jnp.ndarray:
    data = jnp.asarray(data)
    return jnp.ones(data.shape[:1] or (), dtype=data.dtype)


@dataclass(frozen=True)
class Parameter:
    """Unified parameter definition for ``mu``, ``sigma``, ``nu``, and ``tau``.

    Methods are pure and return new arrays, which keeps the object safe to reuse
    under ``jax.jit``, ``jax.grad``, and batching transforms.
    """

    name: str
    link: Link
    constraint: Constraint
    initializer: Initializer = _mean_initializer

    def transform(self, value: Any) -> jnp.ndarray:
        """Transform a constrained value to predictor scale."""
        return self.link.transform(self.constraint.project(value))

    def inverse_transform(self, eta: Any) -> jnp.ndarray:
        """Transform predictor values back to this parameter's valid domain."""
        return self.constraint.project(self.link.inverse(eta))

    def validate(self, value: Any) -> jnp.ndarray:
        """Return a boolean mask for values satisfying the parameter constraint."""
        return self.constraint.validate(value)

    def initialize(self, data: Any) -> jnp.ndarray:
        """Initialize this parameter from data and project into its domain."""
        return self.constraint.project(self.initializer(data))


# Register as a PyTree node so static metadata can travel through JAX transforms.
def _flatten_parameter(parameter: Parameter) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    return (), (parameter.name, parameter.link, parameter.constraint, parameter.initializer)


def _unflatten_parameter(aux_data: tuple[Any, ...], children: tuple[Any, ...]) -> Parameter:
    del children
    name, link, constraint, initializer = aux_data
    return Parameter(name=name, link=link, constraint=constraint, initializer=initializer)


jax.tree_util.register_pytree_node(Parameter, _flatten_parameter, _unflatten_parameter)


MU = Parameter("mu", IdentityLink(), Real(), _mean_initializer)
SIGMA = Parameter("sigma", LogLink(), Positive(), _ones_initializer)
NU = Parameter("nu", IdentityLink(), Real(), _mean_initializer)
TAU = Parameter("tau", LogLink(), Positive(), _ones_initializer)
PI = Parameter("pi", LogitLink(), UnitInterval(), lambda data: jnp.full_like(jnp.asarray(data), 0.5))

DEFAULT_PARAMETERS: Mapping[str, Parameter] = {
    "mu": MU,
    "sigma": SIGMA,
    "nu": NU,
    "tau": TAU,
    "pi": PI,
}


def parameters_from_names(names: Sequence[str]) -> tuple[Parameter, ...]:
    """Return canonical parameter definitions for known GAMLSS parameter names."""
    missing = tuple(name for name in names if name not in DEFAULT_PARAMETERS)
    if missing:
        raise KeyError(f"Unknown canonical parameters: {missing}")
    return tuple(DEFAULT_PARAMETERS[name] for name in names)


__all__ = [
    "DEFAULT_PARAMETERS",
    "Initializer",
    "MU",
    "NU",
    "PI",
    "SIGMA",
    "TAU",
    "Parameter",
    "parameters_from_names",
]
