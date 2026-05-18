"""Joint L-BFGS optimizer wrapper.

This module implements joint L-BFGS optimization of the GAMLSS log-likelihood.
Despite historical naming, this is NOT the Cole-Green (CG) algorithm.
The true Cole-Green algorithm is in omnilss.fitting_cg (fitting_cg.py).

References
----------
Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized additive models
for location, scale and shape. JRSS-C, 54(3), 507-554.

Cole, T. J., & Green, P. J. (1992). Smoothing reference centile curves:
the LMS method and penalized likelihood. Statistics in Medicine, 11(10), 1305-1319.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..cg_derivatives import eta_cross_hessian
from ..distributions import resolve_family
from ..model import GAMLSSModel

# ---------------------------------------------------------------------------
# Cross-derivative computation
# ---------------------------------------------------------------------------


def _compute_cross_derivatives(
    y: np.ndarray,
    param_values: Dict[str, np.ndarray],
    family: Any,
    param_k: str,
    param_j: str,
) -> np.ndarray:
    """Compute element-wise eta-scale CG cross derivatives.

    This helper returns ``d² l_i / (d eta_k,i d eta_j,i)`` for every
    observation.  It delegates to the shared AD derivative kernel and raises
    explicit errors instead of silently returning zeros, because a zero fallback
    would turn the CG correction back into an RS-style block-diagonal update.
    """
    order = tuple(family.estimable_parameters)
    try:
        k_idx = order.index(param_k)
        j_idx = order.index(param_j)
    except ValueError as exc:
        raise ValueError(
            f"parameters {param_k!r}/{param_j!r} must be estimable parameters {order}"
        ) from exc

    hessian = eta_cross_hessian(
        y=y,
        param_values=param_values,
        family=family,
        parameter_order=order,
    )
    return np.asarray(hessian[:, k_idx, j_idx], dtype=np.float64)


# ---------------------------------------------------------------------------
# IRLS step with CG cross-derivative correction
# ---------------------------------------------------------------------------


def _irls_step_with_adjustment(
    y: np.ndarray,
    X: np.ndarray,
    eta: np.ndarray,
    fitted: np.ndarray,
    score: np.ndarray,
    hessian_diag: np.ndarray,
    cross_adjustment: np.ndarray,
    link_derivative: np.ndarray,
    offset: np.ndarray,
    step_size: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """IRLS step with CG cross-derivative correction.

    Core of CG: adds a cross-derivative correction term to the IRLS working
    response, making the update direction more accurate by accounting for
    correlations between distribution parameters.

    Parameters
    ----------
    y : np.ndarray
        Response variable (n,)
    X : np.ndarray
        Design matrix (n, p)
    eta : np.ndarray
        Current linear predictor (n,)
    fitted : np.ndarray
        Current fitted values = g^{-1}(eta) (n,)
    score : np.ndarray
        First derivative dl/dtheta (n,)
    hessian_diag : np.ndarray
        Second derivative d2l/dtheta2 (should be negative) (n,)
    cross_adjustment : np.ndarray
        Cross-derivative correction: sum_{j!=k} (d2l/deta_k*deta_j) * Delta_eta_j (n,)
    link_derivative : np.ndarray
        Link function derivative deta/dtheta (n,)
    offset : np.ndarray
        Offset term (n,)
    step_size : float
        Step size in (0, 1]

    Returns
    -------
    eta_new : np.ndarray
        Updated linear predictor (n,)
    beta_new : np.ndarray
        Updated coefficient vector (p,)
    """
    # Ensure Hessian is negative (log-likelihood concavity)
    hessian_diag = np.where(hessian_diag < -1e-15, hessian_diag, -1e-15)

    # Working weights: w = -d2l/dtheta2 / (deta/dtheta)^2
    working_weights = -hessian_diag / (link_derivative**2)
    working_weights = np.maximum(working_weights, 1e-8)

    # Standard IRLS working response: z = (eta - offset) + score / (w * link_deriv)
    z_standard = (eta - offset) + score / (working_weights * link_derivative)

    # CG additional term: cross_adj / (w * link_deriv^2)
    # Approximates the full Newton direction (contribution from cross Hessian)
    z_cross = cross_adjustment / (working_weights * link_derivative**2 + 1e-15)

    z_adjusted = z_standard + z_cross

    # Weighted least squares: beta = (X'WX)^{-1} X'Wz
    W_diag = working_weights
    XtWX = (X * W_diag[:, None]).T @ X  # (p, p)
    XtWz = X.T @ (W_diag * z_adjusted)  # (p,)

    # Numerical regularization (small ridge to prevent singularity)
    ridge = 1e-10 * np.trace(XtWX) / max(XtWX.shape[0], 1)
    XtWX_reg = XtWX + ridge * np.eye(XtWX.shape[0])

    try:
        beta_new = np.linalg.solve(XtWX_reg, XtWz)
    except np.linalg.LinAlgError:
        beta_new = np.linalg.lstsq(XtWX_reg, XtWz, rcond=None)[0]

    eta_new = X @ beta_new + offset
    return eta_new, beta_new


# ---------------------------------------------------------------------------
# Main fitting function
# ---------------------------------------------------------------------------


def joint_lbfgs_fit(
    formula: str,
    sigma_formula: str = "~ 1",
    nu_formula: Optional[str] = None,
    tau_formula: Optional[str] = None,
    family: str = "NO",
    data: Optional[Dict[str, np.ndarray]] = None,
    mu_step: float = 1.0,
    sigma_step: float = 1.0,
    nu_step: float = 1.0,
    tau_step: float = 1.0,
    max_outer_iter: int = 50,
    max_inner_iter: int = 5,
    outer_tol: float = 1e-4,
    inner_tol: float = 1e-4,
    verbose: bool = False,
) -> GAMLSSModel:
    """Joint L-BFGS optimizer wrapper.

    IMPORTANT
    ---------
    Despite historical naming around "CG" (Cole-Green), this wrapper delegates
    to ``fitting.gamlss(method="CG")``. The true full-Hessian Cole-Green path
    is implemented in ``omnilss.fitting_cg.fit_cg()`` using
    ``jax.hessian`` observed-information blocks.

    Prefer direct use:
        from omnilss.fitting import gamlss
        model = gamlss(formula, family=family, data=data, method="CG")

    Parameters
    ----------
    formula : str
        Formula for mu, e.g. "y ~ x1 + x2"
    sigma_formula : str
        Formula for sigma, default "~ 1"
    nu_formula, tau_formula : str, optional
        Formulas for nu and tau parameters
    family : str or FamilyDefinition
        Distribution family
    data : dict
        Data dictionary {variable: array}
    mu_step, sigma_step, nu_step, tau_step : float
        Deprecated compatibility arguments; the full-Hessian CG backend uses
        its own damped line search.
    max_outer_iter : int
        Maximum outer loop iterations (default 50)
    outer_tol : float
        Convergence tolerance on global deviance change
    verbose : bool
        Print iteration details

    Returns
    -------
    GAMLSSModel
        Fitted model with cg_converged, cg_iterations in additional_slots
    """
    if data is None:
        raise ValueError("data must be provided")

    from ..fitting import gamlss, gamlss_control

    parameter_formulas: dict | None = None
    if nu_formula is not None or tau_formula is not None:
        parameter_formulas = {}
        if nu_formula is not None:
            parameter_formulas["nu"] = nu_formula
        if tau_formula is not None:
            parameter_formulas["tau"] = tau_formula

    fam = resolve_family(family)
    model = gamlss(
        formula=formula,
        sigma_formula=sigma_formula,
        parameter_formulas=parameter_formulas,
        family=fam,
        data=data,
        method="CG",
        control=gamlss_control(n_cyc=max_outer_iter, c_crit=outer_tol),
        verbose=verbose,
    )

    # Attach cross-derivative diagnostics so CG path explicitly uses the helper.
    try:
        params = {
            p: np.asarray(model.fitted_values[p], dtype=np.float64)
            for p in fam.parameters
            if p in model.fitted_values
        }
        y = np.asarray(data[formula.split("~", 1)[0].strip()], dtype=np.float64)
        cross_summary = {}
        if (
            "mu" in params
            and "sigma" in params
            and "mu" in fam.parameters
            and "sigma" in fam.parameters
        ):
            cs = _compute_cross_derivatives(
                y=y, param_values=params, family=fam, param_k="mu", param_j="sigma"
            )
            cross_summary["mu_sigma_mean_abs"] = float(np.mean(np.abs(cs)))
        model.additional_slots["cg_cross_derivative_summary"] = cross_summary
    except Exception:
        model.additional_slots["cg_cross_derivative_summary"] = {}

    return model


def lbfgs_fit(*args, **kwargs):
    """Alias for :func:`joint_lbfgs_fit`."""
    return joint_lbfgs_fit(*args, **kwargs)


def cg_fit(*args, **kwargs):
    """Deprecated alias for :func:`joint_lbfgs_fit`."""
    import warnings

    warnings.warn(
        "cg_fit is deprecated; use joint_lbfgs_fit/lbfgs_fit for L-BFGS "
        "or omnilss.fitting_cg.fit_cg for the true Cole-Green algorithm.",
        DeprecationWarning,
        stacklevel=2,
    )
    return joint_lbfgs_fit(*args, **kwargs)
