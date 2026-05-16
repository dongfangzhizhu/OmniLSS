"""GAMLSS fitting algorithms.

This module contains various algorithms for fitting GAMLSS models:
- RS (Rigby-Stasinopoulos): The original GAMLSS algorithm (fully implemented)
- CG (Cole-Green): Alternative algorithm with complete JAX Hessian cross-derivative adjustments
- Mixed: Intelligent algorithm selection combining RS and CG

The RS algorithm is the default and most commonly used.
"""

from .rs_algorithm import rs_fit, rs_step
from .cg_algorithm import joint_lbfgs_fit as cg_fit_lbfgs
from .cg_algorithm_v2 import cg_fit_v2

# New default CG implementation
cg_fit = cg_fit_v2
from .mixed_algorithm import mixed_fit, compare_algorithms

__all__ = [
    "rs_fit",
    "rs_step",
    "cg_fit",
    "cg_fit_lbfgs",
    "joint_lbfgs_fit",
    "cg_fit_v2",
    "mixed_fit",
    "compare_algorithms",
]
