"""Typed public API contracts for fit/predict/score/family freeze boundary."""

from __future__ import annotations

from typing import Any, Protocol

from .results import FitResult, PredictResult, ScoreResult


class PublicGAMLSSAPI(Protocol):
    def fit(self, *args: Any, **kwargs: Any) -> FitResult:
        ...

    def predict(self, *args: Any, **kwargs: Any) -> PredictResult:
        ...

    def score(self, *args: Any, **kwargs: Any) -> ScoreResult:
        ...


def freeze_fit_result(
    *,
    family: str,
    method: str,
    converged: bool,
    iterations: int,
    deviance: float,
    coefficients: dict[str, Any],
    diagnostics: dict[str, Any],
) -> FitResult:
    return FitResult(
        family=family,
        method=method,
        converged=bool(converged),
        iterations=int(iterations),
        deviance=float(deviance),
        coefficients=dict(coefficients),
        diagnostics=dict(diagnostics),
    )


def freeze_predict_result(*, family: str, parameter_predictions: dict[str, Any]) -> PredictResult:
    return PredictResult(family=family, parameter_predictions=dict(parameter_predictions))


def freeze_score_result(*, family: str, metric: str, value: float) -> ScoreResult:
    return ScoreResult(family=family, metric=metric, value=float(value))
