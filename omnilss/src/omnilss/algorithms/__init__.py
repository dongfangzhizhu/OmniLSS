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

def cg_fit(
    formula: str,
    sigma_formula: str = "~ 1",
    nu_formula: str | None = None,
    tau_formula: str | None = None,
    family: str = "NO",
    data: dict | None = None,
    mu_step: float = 1.0,
    sigma_step: float = 1.0,
    nu_step: float = 1.0,
    tau_step: float = 1.0,
    max_outer_iter: int = 50,
    outer_tol: float = 1e-4,
    verbose: bool = False,
):
    """Backward-compatible CG entrypoint.

    Accepts legacy keyword arguments used by existing tests/callers and routes
    them into ``cg_fit_v2``.
    """
    return cg_fit_lbfgs(
        formula=formula,
        sigma_formula=sigma_formula,
        nu_formula=nu_formula,
        tau_formula=tau_formula,
        family=family,
        data=data,
        mu_step=mu_step,
        sigma_step=sigma_step,
        nu_step=nu_step,
        tau_step=tau_step,
        max_outer_iter=max_outer_iter,
        outer_tol=outer_tol,
        verbose=verbose,
    )
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
    "joint_lbfgs_fit",
    "cg_fit_v2",
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
