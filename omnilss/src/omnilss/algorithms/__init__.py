"""GAMLSS fitting algorithms.

This module contains various algorithms for fitting GAMLSS models:
- RS (Rigby-Stasinopoulos): The original GAMLSS algorithm (fully implemented)
- CG (Cole-Green): Alternative algorithm with complete JAX Hessian cross-derivative adjustments
- Mixed: Intelligent algorithm selection combining RS and CG

The RS algorithm is the default and most commonly used.
"""

from .rs_algorithm import rs_fit, rs_step
from .cg_algorithm import cg_fit
from .mixed_algorithm import mixed_fit, compare_algorithms

__all__ = [
    "rs_fit",
    "rs_step",
    "cg_fit",
    "mixed_fit",
    "compare_algorithms",
]
