"""Public API result schemas (immutable) for API-freeze stabilization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class FitResult:
    family: str
    method: str
    converged: bool
    iterations: int
    deviance: float
    coefficients: Mapping[str, Any]
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class PredictResult:
    family: str
    parameter_predictions: Mapping[str, Any]


@dataclass(frozen=True)
class ScoreResult:
    family: str
    metric: str
    value: float
