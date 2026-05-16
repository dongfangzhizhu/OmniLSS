"""Cole-Green algorithm v2 entrypoint.

Current implementation reuses the production CG backend exposed through
`joint_lbfgs_fit` while providing the stable public API for CG routing.
"""

from __future__ import annotations

from typing import Any

from .cg_algorithm import joint_lbfgs_fit


def cg_fit_v2(
    formula: str,
    family: Any,
    data: dict,
    sigma_formula: str = "~ 1",
    parameter_formulas: dict | None = None,
    weights=None,
    max_iter: int = 50,
    tol: float = 1e-4,
    step_sizes: dict | None = None,
    verbose: bool = False,
):
    """Fit GAMLSS model using the CG public API.

    Parameters mirror the planned v2 interface; this wrapper maps them to the
    maintained backend implementation.
    """
    nu_formula = None
    tau_formula = None
    if parameter_formulas:
        nu_formula = parameter_formulas.get("nu")
        tau_formula = parameter_formulas.get("tau")

    model = joint_lbfgs_fit(
        formula=formula,
        sigma_formula=sigma_formula,
        nu_formula=nu_formula,
        tau_formula=tau_formula,
        family=family,
        data=data,
        max_outer_iter=max_iter,
        outer_tol=tol,
        verbose=verbose,
    )
    model.additional_slots["method"] = "CG"
    return model
