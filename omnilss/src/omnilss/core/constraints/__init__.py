"""Constraint primitives for the canonical OmniLSS parameter system.

The classes in this module are intentionally stateless and JAX-friendly.  They
provide the minimum validation/initialization contract needed by the 30-day
architecture freeze without adding distribution-specific behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import jax.numpy as jnp

ArrayLike = object


@runtime_checkable
class Constraint(Protocol):
    """Pure validation/repair interface for model parameters."""

    def validate(self, value: ArrayLike) -> jnp.ndarray:
        """Return a boolean mask indicating whether values satisfy the constraint."""
        ...

    def project(self, value: ArrayLike) -> jnp.ndarray:
        """Project values into the valid domain without mutating inputs."""
        ...


@dataclass(frozen=True)
class Real:
    """Unconstrained real-valued parameter domain."""

    def validate(self, value: ArrayLike) -> jnp.ndarray:
        value = jnp.asarray(value)
        return jnp.isfinite(value)

    def project(self, value: ArrayLike) -> jnp.ndarray:
        return jnp.asarray(value)


@dataclass(frozen=True)
class Positive:
    """Strictly positive parameter domain with an epsilon floor."""

    eps: float = 1e-12

    def validate(self, value: ArrayLike) -> jnp.ndarray:
        value = jnp.asarray(value)
        return jnp.logical_and(jnp.isfinite(value), value > self.eps)

    def project(self, value: ArrayLike) -> jnp.ndarray:
        return jnp.maximum(jnp.asarray(value), self.eps)


@dataclass(frozen=True)
class UnitInterval:
    """Open unit interval parameter domain."""

    eps: float = 1e-12

    def validate(self, value: ArrayLike) -> jnp.ndarray:
        value = jnp.asarray(value)
        return jnp.logical_and(value > self.eps, value < 1.0 - self.eps)

    def project(self, value: ArrayLike) -> jnp.ndarray:
        return jnp.clip(jnp.asarray(value), self.eps, 1.0 - self.eps)


__all__ = ["ArrayLike", "Constraint", "Positive", "Real", "UnitInterval"]
