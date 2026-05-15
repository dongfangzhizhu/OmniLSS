"""Canonical optimizer interface for architecture-stable optimization code."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import jax
import optax

Params = Mapping[str, Any]
LossFn = Callable[[Params], Any]


@runtime_checkable
class OptimizerProtocol(Protocol):
    """Stateless optimizer boundary compatible with JAX transformations."""

    def init(self, params: Params) -> Any:
        """Create optimizer state from parameters without mutating model internals."""
        ...

    def step(self, loss_fn: LossFn, params: Params, state: Any) -> tuple[Params, Any]:
        """Return updated parameters and state for one optimization step."""
        ...


@dataclass(frozen=True)
class OptaxOptimizer:
    """Small adapter exposing Optax transformations via ``OptimizerProtocol``."""

    transform: optax.GradientTransformation

    def init(self, params: Params) -> Any:
        return self.transform.init(params)

    def step(self, loss_fn: LossFn, params: Params, state: Any) -> tuple[Params, Any]:
        loss, grads = jax.value_and_grad(loss_fn)(params)
        del loss
        updates, new_state = self.transform.update(grads, state, params)
        return optax.apply_updates(params, updates), new_state


__all__ = ["LossFn", "OptaxOptimizer", "OptimizerProtocol", "Params"]
