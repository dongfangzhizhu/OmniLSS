"""Convergence monitor for optimizer/runtime loops."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ConvergenceThresholds:
    gradient_norm_tol: float = 1e-6
    deviance_delta_tol: float = 1e-6
    parameter_delta_tol: float = 1e-6
    curvature_tol: float = 1e12


@dataclass(frozen=True)
class ConvergenceStatus:
    converged: bool
    reason: str


class ConvergenceMonitor:
    def __init__(self, thresholds: ConvergenceThresholds | None = None) -> None:
        self.thresholds = thresholds or ConvergenceThresholds()

    def evaluate(
        self,
        gradient_norm: float,
        deviance_delta: float,
        parameter_delta: float,
        condition_number: float,
    ) -> ConvergenceStatus:
        if not np.isfinite(gradient_norm):
            return ConvergenceStatus(False, "non_finite_gradient")
        if condition_number > self.thresholds.curvature_tol:
            return ConvergenceStatus(False, "unstable_curvature")
        if abs(deviance_delta) <= self.thresholds.deviance_delta_tol and abs(parameter_delta) <= self.thresholds.parameter_delta_tol:
            return ConvergenceStatus(True, "deviance_and_parameter_delta")
        if abs(gradient_norm) <= self.thresholds.gradient_norm_tol:
            return ConvergenceStatus(True, "gradient_norm")
        return ConvergenceStatus(False, "in_progress")
