"""Runtime optimizer abstraction layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class OptimizerResult:
    params: dict[str, Any]
    converged: bool
    iterations: int
    diagnostics: dict[str, Any]


class Optimizer(Protocol):
    def optimize(self, context: dict[str, Any]) -> OptimizerResult:
        ...


class RSOptimizer:
    def optimize(self, context: dict[str, Any]) -> OptimizerResult:
        model = context["runner"](method="RS")
        return OptimizerResult(
            params=getattr(model, "coef", {}),
            converged=bool(model.additional_slots.get("converged", False)),
            iterations=int(model.additional_slots.get("cycles", model.iter)),
            diagnostics=dict(model.additional_slots),
        )


class CGOptimizer:
    def optimize(self, context: dict[str, Any]) -> OptimizerResult:
        model = context["runner"](method="CG")
        return OptimizerResult(
            params=getattr(model, "coef", {}),
            converged=bool(model.additional_slots.get("converged", False)),
            iterations=int(model.additional_slots.get("cycles", model.iter)),
            diagnostics=dict(model.additional_slots),
        )


class NewtonOptimizer:
    def optimize(self, context: dict[str, Any]) -> OptimizerResult:
        model = context["runner"](method="joint")
        return OptimizerResult(
            params=getattr(model, "coef", {}),
            converged=bool(model.additional_slots.get("converged", False)),
            iterations=int(model.additional_slots.get("cycles", model.iter)),
            diagnostics=dict(model.additional_slots),
        )


class TrustRegionOptimizer:
    def optimize(self, context: dict[str, Any]) -> OptimizerResult:
        model = context["runner"](method="lbfgs")
        return OptimizerResult(
            params=getattr(model, "coef", {}),
            converged=bool(model.additional_slots.get("converged", False)),
            iterations=int(model.additional_slots.get("cycles", model.iter)),
            diagnostics=dict(model.additional_slots),
        )
