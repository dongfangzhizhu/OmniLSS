# SPDX-License-Identifier: GPL-3.0-or-later
"""GAMLSS fitting algorithms.

This module contains various algorithms for fitting GAMLSS models:
- RS (Rigby-Stasinopoulos): The original GAMLSS algorithm (fully implemented)
- joint_lbfgs_fit/lbfgs_fit: L-BFGS joint optimizer
- cole_green_fit: True Cole-Green full-Hessian implementation from omnilss.fitting_cg.fit_cg
- Mixed: Intelligent algorithm selection combining RS and CG
- RS_JAX: JAX-native RS with jax.lax.while_loop + jnp.linalg.lstsq (GPU/TPU ready)

The RS algorithm is the default and most commonly used.
"""

from .rs_algorithm import rs_fit, rs_step
import warnings

from .lbfgs_algorithm import joint_lbfgs_fit, lbfgs_fit
from .cg_algorithm_v2 import cg_fit_v2
from ..fitting_cg import fit_cg

cole_green_fit = fit_cg


def cg_fit_lbfgs(*args, **kwargs):
    """Deprecated public alias for the historical L-BFGS optimizer."""
    warnings.warn(
        "cg_fit_lbfgs is deprecated; use joint_lbfgs_fit or lbfgs_fit.",
        DeprecationWarning,
        stacklevel=2,
    )
    return joint_lbfgs_fit(*args, **kwargs)


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
    """Deprecated compatibility entrypoint for historical ``cg_fit`` calls.

    The true Cole-Green implementation is ``omnilss.fitting_cg.fit_cg`` and is
    exported here as ``fit_cg``/``cole_green_fit``.  This wrapper keeps the
    legacy formula-based public API working while warning that the old name was
    historically associated with the L-BFGS backend.
    """
    warnings.warn(
        "omnilss.algorithms.cg_fit is deprecated as a formula-based L-BFGS "
        "compatibility alias; use joint_lbfgs_fit/lbfgs_fit for L-BFGS or "
        "fit_cg/cole_green_fit for the true Cole-Green algorithm.",
        DeprecationWarning,
        stacklevel=2,
    )
    return joint_lbfgs_fit(
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
    "lbfgs_fit",
    "cole_green_fit",
    "fit_cg",
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
