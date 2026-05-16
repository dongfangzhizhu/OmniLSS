# SPDX-License-Identifier: GPL-3.0-or-later
"""GAMLSS fitting algorithms.

This module contains various algorithms for fitting GAMLSS models:
- RS (Rigby-Stasinopoulos): The original GAMLSS algorithm (fully implemented)
- CG (Cole-Green): Alternative algorithm with complete JAX Hessian cross-derivative adjustments
- Mixed: Intelligent algorithm selection combining RS and CG
- RS_JAX: JAX-native RS with jax.lax.while_loop + jnp.linalg.lstsq (GPU/TPU ready)

The RS algorithm is the default and most commonly used.
"""

from .rs_algorithm import rs_fit, rs_step
from .cg_algorithm import joint_lbfgs_fit as cg_fit_lbfgs
from .cg_algorithm_v2 import cg_fit_v2

# New default CG implementation
cg_fit = cg_fit_v2
from .mixed_algorithm import mixed_fit, compare_algorithms

# JAX-native RS core (GPU/TPU ready)
from .jax_family_specs import (
    FamilyJAXSpec,
    get_jax_spec,
    make_no_spec,
    make_ga_spec,
    make_po_spec,
    make_bi_spec,
    make_wei_spec,
    make_tf_spec,
    supported_families,
)
from .jax_rs_core import JaxRSResult, jax_rs_fit_core
from .jax_rs_integration import gamlss_rs_jax

__all__ = [
    # NumPy RS
    "rs_fit",
    "rs_step",
    # CG variants
    "cg_fit",
    "cg_fit_lbfgs",
    "cg_fit_v2",
    # Mixed
    "mixed_fit",
    "compare_algorithms",
    # JAX-native RS
    "FamilyJAXSpec",
    "JaxRSResult",
    "gamlss_rs_jax",
    "get_jax_spec",
    "jax_rs_fit_core",
    "make_no_spec",
    "make_ga_spec",
    "make_po_spec",
    "make_bi_spec",
    "make_wei_spec",
    "make_tf_spec",
    "supported_families",
]
