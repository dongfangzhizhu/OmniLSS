"""Phase 0 optimizer protocol freeze helpers.

Defines a minimal runtime-neutral interface used by RS and gradient-based
optimizers so convergence diagnostics can be consumed consistently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class OptimizerState:
    iteration: int
    converged: bool
    diagnostics: dict[str, Any]


@runtime_checkable
class Phase0OptimizerProtocol(Protocol):
    def initialize(self, context: dict[str, Any]) -> OptimizerState:
        ...

    def step(self, context: dict[str, Any], state: OptimizerState) -> OptimizerState:
        ...

    def converged(self, state: OptimizerState) -> bool:
        ...


class RSOptimizerAdapter:
    """Adapter exposing RS model outputs through Phase0 optimizer protocol."""

    def initialize(self, context: dict[str, Any]) -> OptimizerState:
        return OptimizerState(iteration=0, converged=False, diagnostics={})

    def step(self, context: dict[str, Any], state: OptimizerState) -> OptimizerState:
        model = context["model"]
        slots = model.additional_slots
        diagnostics = {
            "loss": float(model.g_dev),
            "gradient_norm": float(slots.get("gradient_norm", float("nan"))),
            "step_size": slots.get("step_size_by_param", {}),
            "condition_number": float(slots.get("condition_number", float("nan"))),
        }
        return OptimizerState(
            iteration=int(slots.get("cycles", model.iter)),
            converged=bool(slots.get("converged", False)),
            diagnostics=diagnostics,
        )

    def converged(self, state: OptimizerState) -> bool:
        return bool(state.converged)
