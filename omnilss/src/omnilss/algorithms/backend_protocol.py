"""Backend protocol for RS/IWLS execution.

This module defines a lightweight interface used to decouple RS orchestration
from backend-specific compute (NumPy/JAX/TF).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class IWLSState:
    """Numerical state for one IWLS update."""

    design: np.ndarray
    eta: np.ndarray
    fitted: np.ndarray
    observation_weights: np.ndarray


class SolverBackend(Protocol):
    """Protocol for backend-specific score/Hessian and WLS operations."""

    name: str

    def compute_score_hessian(
        self,
        *,
        parameter: str,
        y: np.ndarray,
        fitted: np.ndarray,
        context: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (score, hessian) for the requested parameter."""

    def iwls_step(
        self,
        *,
        state: IWLSState,
        score: np.ndarray,
        hessian: np.ndarray,
        link_derivative: np.ndarray,
        offset: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (working_response, iterative_weights)."""

    def wls_solve(
        self,
        *,
        design: np.ndarray,
        working_response: np.ndarray,
        weights: np.ndarray,
        ridge: float | None = None,
        skip_stabilize: bool = False,
    ) -> np.ndarray:
        """Solve weighted least squares and return coefficients.

        When ``ridge`` is None and ``skip_stabilize`` is False, backends derive
        the regularization from their Hessian/Gram stabilization pass.  When
        ``skip_stabilize`` is True the caller has already stabilized the Gram
        and passes the resulting ``ridge`` directly.
        """
