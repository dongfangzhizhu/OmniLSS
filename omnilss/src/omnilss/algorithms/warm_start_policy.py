"""Warm-start policies for RS-style fitting.

Policies are intentionally separated from solver backends so the project can
measure warm-start cost independently from core iteration cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class WarmStartResult:
    """Warm-start output used by outer RS loop initialization."""

    initial_eta: dict[str, np.ndarray]
    initial_fitted: dict[str, np.ndarray]
    metadata: dict[str, float | str]


class WarmStartPolicy(Protocol):
    """Policy interface for initializing model parameters before RS iterations."""

    name: str

    def initialize(
        self,
        *,
        y: np.ndarray,
        parameter_names: tuple[str, ...],
    ) -> WarmStartResult:
        """Return initial eta/fitted vectors for each estimable parameter."""


class NoWarmStart:
    """Cheap constant initialization used as a baseline policy."""

    name = "none"

    def initialize(
        self,
        *,
        y: np.ndarray,
        parameter_names: tuple[str, ...],
    ) -> WarmStartResult:
        n = y.shape[0]
        eta = {p: np.zeros(n, dtype=np.float64) for p in parameter_names}
        fitted = {p: np.ones(n, dtype=np.float64) for p in parameter_names}
        return WarmStartResult(
            initial_eta=eta,
            initial_fitted=fitted,
            metadata={"policy": self.name, "n_obs": float(n)},
        )


class FamilyHeuristicInit:
    """Heuristic initialization with stable defaults for common parameters."""

    name = "family_heuristic"

    def initialize(
        self,
        *,
        y: np.ndarray,
        parameter_names: tuple[str, ...],
    ) -> WarmStartResult:
        n = y.shape[0]
        y_mean = float(np.mean(y)) if n else 0.0
        y_sd = float(np.std(y)) if n else 1.0
        y_sd = max(y_sd, 1e-6)

        eta: dict[str, np.ndarray] = {}
        fitted: dict[str, np.ndarray] = {}
        for p in parameter_names:
            if p == "mu":
                fitted[p] = np.full(n, y_mean, dtype=np.float64)
            elif p == "sigma":
                fitted[p] = np.full(n, y_sd, dtype=np.float64)
            else:
                fitted[p] = np.ones(n, dtype=np.float64)
            eta[p] = np.zeros(n, dtype=np.float64)

        return WarmStartResult(
            initial_eta=eta,
            initial_fitted=fitted,
            metadata={"policy": self.name, "y_mean": y_mean, "y_sd": y_sd},
        )
