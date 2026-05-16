"""Phase 1 automatic warning system for numerical runtime diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WarningEvent:
    code: str
    message: str
    level: str = "warning"


def evaluate_numerical_warnings(slots: dict[str, Any]) -> list[WarningEvent]:
    events: list[WarningEvent] = []

    grad = float(slots.get("gradient_norm", np.nan))
    cond = float(slots.get("condition_number", np.nan))
    step_map = slots.get("step_size_by_param", {})

    if np.isfinite(grad) and grad > 1e5:
        events.append(WarningEvent("EXPLODING_GRADIENT", f"gradient_norm too high: {grad:.3e}"))

    if np.isfinite(cond) and cond > 1e10:
        events.append(WarningEvent("BAD_CONDITIONING", f"condition_number too high: {cond:.3e}"))

    if isinstance(step_map, dict):
        tiny_steps = [k for k, v in step_map.items() if isinstance(v, (int, float)) and v < 1e-4]
        if tiny_steps:
            events.append(WarningEvent("TINY_STEPS", f"very small step sizes for params: {tiny_steps}"))

    if slots.get("lambda_update_failed_params"):
        events.append(WarningEvent("UNSTABLE_SPLINE", "lambda update failures detected"))

    return events
