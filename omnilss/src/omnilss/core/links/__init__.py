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

    def derivative(self, eta: ArrayLike) -> jnp.ndarray:
        """Derivative of inverse link w.r.t. eta."""
        ...


@dataclass(frozen=True)
class IdentityLink:
    """Identity transform for unconstrained parameters."""

    def transform(self, value: ArrayLike) -> jnp.ndarray:
        return jnp.asarray(value)

    def inverse(self, eta: ArrayLike) -> jnp.ndarray:
        return jnp.asarray(eta)

    def derivative(self, eta: ArrayLike) -> jnp.ndarray:
        return jnp.ones_like(jnp.asarray(eta), dtype=jnp.asarray(eta).dtype)


@dataclass(frozen=True)
class LogLink:
    """Log transform for positive parameters."""

    eps: float = 1e-12

    def transform(self, value: ArrayLike) -> jnp.ndarray:
        return jnp.log(jnp.maximum(jnp.asarray(value), self.eps))

    def inverse(self, eta: ArrayLike) -> jnp.ndarray:
        return jnp.maximum(jnp.exp(jnp.asarray(eta)), self.eps)

    def derivative(self, eta: ArrayLike) -> jnp.ndarray:
        return self.inverse(eta)


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

    def derivative(self, eta: ArrayLike) -> jnp.ndarray:
        p = self.inverse(eta)
        return p * (1.0 - p)


@dataclass(frozen=True)
class SoftplusLink:
    """Softplus link for positive parameters with smoother gradients."""

    eps: float = 1e-12

    def transform(self, value: ArrayLike) -> jnp.ndarray:
        value = jnp.maximum(jnp.asarray(value), self.eps)
        return jnp.log(jnp.expm1(value))

    def inverse(self, eta: ArrayLike) -> jnp.ndarray:
        return jnp.logaddexp(jnp.asarray(eta), 0.0) + self.eps

    def derivative(self, eta: ArrayLike) -> jnp.ndarray:
        eta = jnp.asarray(eta)
        return 1.0 / (1.0 + jnp.exp(-eta))


__all__ = ["IdentityLink", "Link", "LogLink", "LogitLink", "SoftplusLink"]
