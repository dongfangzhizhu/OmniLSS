"""Deterministic runtime policies and seed management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

DTypePolicy = Literal["float32", "float64"]


@dataclass(frozen=True)
class RuntimeTolerancePolicy:
    """Centralized convergence tolerances used by iterative solvers."""

    abs_tol: float = 1e-8
    rel_tol: float = 1e-6
    grad_tol: float = 1e-6
    max_iter: int = 200


@dataclass(frozen=True)
class DeterministicPolicy:
    """Global deterministic execution policy."""

    dtype: DTypePolicy = "float64"
    tolerance: RuntimeTolerancePolicy = RuntimeTolerancePolicy()
    deterministic_optimizer_order: bool = True


@dataclass
class SeedManager:
    """Simple deterministic random-state manager for runtime components."""

    seed: int = 0

    def rng(self) -> np.random.Generator:
        """Return a reproducible numpy generator."""
        return np.random.default_rng(self.seed)

    def with_seed(self, seed: int) -> "SeedManager":
        """Return a cloned seed manager using a new seed."""
        return SeedManager(seed=seed)
