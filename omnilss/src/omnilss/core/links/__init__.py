"""Stateless link objects for canonical parameters.

These wrappers deliberately mirror the existing functional link operations while
exposing a small object protocol suitable for ``Parameter`` definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import jax.numpy as jnp

ArrayLike = object


@runtime_checkable
class Link(Protocol):
    """Pure transform/inverse-transform interface."""

    def transform(self, value: ArrayLike) -> jnp.ndarray:
        """Map constrained values to the linear-predictor scale."""
        ...

    def inverse(self, eta: ArrayLike) -> jnp.ndarray:
        """Map linear-predictor values back to constrained values."""
        ...


@dataclass(frozen=True)
class IdentityLink:
    """Identity transform for unconstrained parameters."""

    def transform(self, value: ArrayLike) -> jnp.ndarray:
        return jnp.asarray(value)

    def inverse(self, eta: ArrayLike) -> jnp.ndarray:
        return jnp.asarray(eta)


@dataclass(frozen=True)
class LogLink:
    """Log transform for positive parameters."""

    eps: float = 1e-12

    def transform(self, value: ArrayLike) -> jnp.ndarray:
        return jnp.log(jnp.maximum(jnp.asarray(value), self.eps))

    def inverse(self, eta: ArrayLike) -> jnp.ndarray:
        return jnp.maximum(jnp.exp(jnp.asarray(eta)), self.eps)


@dataclass(frozen=True)
class LogitLink:
    """Logit transform for probabilities."""

    eps: float = 1e-12

    def transform(self, value: ArrayLike) -> jnp.ndarray:
        value = jnp.clip(jnp.asarray(value), self.eps, 1.0 - self.eps)
        return jnp.log(value / (1.0 - value))

    def inverse(self, eta: ArrayLike) -> jnp.ndarray:
        value = 1.0 / (1.0 + jnp.exp(-jnp.asarray(eta)))
        return jnp.clip(value, self.eps, 1.0 - self.eps)


__all__ = ["IdentityLink", "Link", "LogLink", "LogitLink"]
