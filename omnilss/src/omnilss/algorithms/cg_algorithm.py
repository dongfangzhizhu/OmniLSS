"""Joint L-BFGS optimizer wrapper (historical CG module).

NAMING NOTE: Despite the "CG" name, this module wraps an L-BFGS joint optimizer.
The true Cole-Green algorithm (full Hessian) is in omnilss.fitting_cg.fit_cg().

References
----------
Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized additive models
for location, scale and shape. JRSS-C, 54(3), 507-554.

Cole, T. J., & Green, P. J. (1992). Smoothing reference centile curves:
the LMS method and penalized likelihood. Statistics in Medicine, 11(10), 1305-1319.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

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
    """Compute cross second-order derivatives d2l/deta_k*deta_j via JAX autodiff.

    Uses JAX automatic differentiation to compute cross-derivatives without
    requiring hand-written analytical formulas for each distribution family.

    Parameters
    ----------
    y : np.ndarray
        Response variable
    param_values : dict
        Current parameter values {param_name: current_fitted_values}
    family : FamilyDefinition
        Distribution family
    param_k : str
        First parameter name (the parameter being updated)
    param_j : str
        Second parameter name (the other parameter in the cross-derivative)

    Returns
    -------
    cross_deriv : np.ndarray
        Element-wise cross-derivative array (n,), i.e. d2l_i/(deta_k,i * deta_j,i)
        for each observation i
    """
    y_jax = jnp.asarray(y, dtype=jnp.float64)

    # Get current fitted values (fitted values = g^{-1}(eta))
    link_k = family.link_functions[param_k]
    link_j = family.link_functions[param_j]

    fv_k = jnp.asarray(param_values[param_k], dtype=jnp.float64)
    fv_j = jnp.asarray(param_values[param_j], dtype=jnp.float64)

    # Fix other parameters (not differentiated)
    fixed_params = {}
    for p, v in param_values.items():
        if p not in (param_k, param_j):
            fixed_params[p] = jnp.asarray(v, dtype=jnp.float64)

    # Convert fitted values to eta (linear predictor)
    eta_k_all = jnp.asarray(link_k(fv_k), dtype=jnp.float64)
    eta_j_all = jnp.asarray(link_j(fv_j), dtype=jnp.float64)

    try:
        # Use vmap to vectorize cross-derivative computation over all observations
        def cross_deriv_single(eta_k_i, eta_j_i, y_i):
            """Compute d2l/(deta_k * deta_j) for a single observation."""
            # Apply inverse link to get distribution parameters
            theta_k_i = family.link_inverses[param_k](eta_k_i)
            theta_j_i = family.link_inverses[param_j](eta_j_i)

            # Scalar fixed parameters (avoid shape issues in vmap)
            params_i = {
                p: v[0] if hasattr(v, "__len__") else v for p, v in fixed_params.items()
            }
            params_i[param_k] = theta_k_i
            params_i[param_j] = theta_j_i

            # Differentiate dl/deta_k with respect to eta_j:
            # d/deta_j [ d/deta_k log p(y | params) ]
            d_kj = jax.grad(
                lambda ej: jax.grad(
                    lambda ek: family.logpdf(
                        y=y_i,
                        **{
                            **params_i,
                            param_k: family.link_inverses[param_k](ek),
                            param_j: family.link_inverses[param_j](ej),
                        },
                    )
                )(eta_k_i)
            )(eta_j_i)

            return d_kj

        # Vectorize over all n observations
        cross_derivs = jax.vmap(cross_deriv_single)(eta_k_all, eta_j_all, y_jax)
        return np.asarray(cross_derivs, dtype=np.float64)

    except Exception:
        # Fall back to zero vector — degrades to RS update (no cross-derivative correction)
        return np.zeros(len(y), dtype=np.float64)


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


def cg_fit(*args, **kwargs):
    """Deprecated alias for joint_lbfgs_fit().

    Notes
    -----
    This function name is kept for backward compatibility.
    It currently delegates to the same joint optimization backend.
    """
    import warnings

    warnings.warn(
        "cg_fit is deprecated; use joint_lbfgs_fit for accurate naming.",
        DeprecationWarning,
        stacklevel=2,
    )
    return joint_lbfgs_fit(*args, **kwargs)
