"""Failure-mode hardening helpers for stress validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...runtime.errors import ConvergenceError, NumericalError


@dataclass(frozen=True)
class StressCheckResult:
    has_nan: bool
    has_inf: bool
    iterations_exhausted: bool


def check_numerical_state(values: dict[str, Any]) -> StressCheckResult:
    has_nan = False
    has_inf = False
    for value in values.values():
        arr = np.asarray(value)
        has_nan = has_nan or bool(np.isnan(arr).any())
        has_inf = has_inf or bool(np.isinf(arr).any())
    return StressCheckResult(has_nan=has_nan, has_inf=has_inf, iterations_exhausted=False)


def enforce_no_nan_propagation(values: dict[str, Any]) -> None:
    result = check_numerical_state(values)
    if result.has_nan or result.has_inf:
        raise NumericalError("Detected NaN/Inf propagation in stress validation")


def enforce_iteration_budget(iteration: int, max_iter: int) -> None:
    if int(iteration) >= int(max_iter):
        raise ConvergenceError(
            f"Iteration budget exhausted ({iteration} >= {max_iter}); potential infinite loop"
        )
