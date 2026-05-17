"""Cole-Green IRLS cross-derivative backend.

This module keeps an eta-scale CG implementation alongside the coefficient-level
full-Hessian correctness backend in :mod:`omnilss.fitting_cg`.  The backend is
intended for experiments and medium/large designs where retaining an RS-style
weighted least-squares structure is useful, while still using Cole-Green
cross-derivative corrections.
"""

from __future__ import annotations

from typing import Any, Dict

import jax.numpy as jnp
import numpy as np

from .._fitting_init import _initial_mu_beta, _initial_parameter_value, _initial_sigma
from .._fitting_utils import _is_intercept_only_formula, _resolve_parameter_formulas
from ..cg_derivatives import EtaDerivativeBundle, eta_score_hessian
from ..distributions import resolve_family
from ..fitting import (
    _build_design_matrix_with_smooths,
    _build_rqres_callable,
    _compute_residuals,
    _fixed_parameter_term,
    _parse_formula,
    _resolve_fixed_parameter_values,
)
from ..model import GAMLSSModel
from ._model_metrics import df_fit_with_smooth_edf


def _safe_eta_weighted_least_squares(
    X: np.ndarray,
    z: np.ndarray,
    working_weights: np.ndarray,
    observation_weights: np.ndarray,
) -> np.ndarray:
    """Solve an eta-scale weighted least-squares system with ridge fallback."""
    w = np.asarray(working_weights * observation_weights, dtype=np.float64)
    w = np.where(np.isfinite(w), w, 0.0)
    w = np.maximum(w, 1e-10)
    XtWX = (X * w[:, None]).T @ X
    XtWz = X.T @ (w * z)
    ridge = 1e-10 * max(float(np.trace(XtWX)) / max(XtWX.shape[0], 1), 1.0)
    XtWX = XtWX + ridge * np.eye(XtWX.shape[0], dtype=np.float64)
    try:
        return np.linalg.solve(XtWX, XtWz)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(XtWX, XtWz, rcond=None)[0]


def _params_from_etas(
    family: Any,
    etas: dict[str, np.ndarray],
    fixed_parameter_values: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Convert eta dictionaries back to family parameter values."""
    params: dict[str, np.ndarray] = {}
    fixed_parameter_values = fixed_parameter_values or {}
    for parameter in family.parameters:
        if parameter in fixed_parameter_values:
            params[parameter] = np.asarray(
                fixed_parameter_values[parameter], dtype=np.float64
            )
        else:
            eta = jnp.asarray(etas[parameter], dtype=jnp.float64)
            params[parameter] = np.asarray(
                family.link_inverses[parameter](eta), dtype=np.float64
            )
    return params


def _global_deviance(
    family: Any,
    y: np.ndarray,
    weights: np.ndarray,
    params: dict[str, np.ndarray],
) -> float:
    """Return weighted global deviance for current family parameters."""
    return float(
        np.sum(weights * np.asarray(family.g_dev_inc(y=y, **params), dtype=np.float64))
    )


def cg_outer_step(
    y: np.ndarray,
    design_matrices: dict[str, np.ndarray],
    parameter_values: dict[str, np.ndarray],
    etas: dict[str, np.ndarray],
    coefficients: dict[str, np.ndarray],
    family: Any,
    weights: np.ndarray,
    step_sizes: dict[str, float],
    offsets: dict[str, np.ndarray],
    derivative_bundle: EtaDerivativeBundle | None = None,
) -> tuple[dict, dict, dict, EtaDerivativeBundle]:
    """Run one eta-scale CG outer step using a shared derivative bundle.

    The derivative bundle is computed once at the start of the outer iteration.
    Each parameter update then reuses the same per-observation Hessian tensor and
    accumulates ``sum_j H_kj * Delta eta_j`` as the Cole-Green cross correction.
    """
    params = list(family.estimable_parameters)
    bundle = derivative_bundle or eta_score_hessian(
        y=y,
        param_values=parameter_values,
        family=family,
        parameter_order=params,
    )
    order_index = {name: idx for idx, name in enumerate(bundle.parameter_order)}

    eta_old = {p: etas[p].copy() for p in params}
    eta_new = {p: etas[p].copy() for p in params}
    delta_eta = {p: np.zeros_like(etas[p]) for p in params}
    new_params = {
        p: np.asarray(parameter_values[p], dtype=np.float64).copy()
        for p in family.parameters
    }
    new_coefs = {p: coefficients[p].copy() for p in params}

    for param_k in params:
        k_idx = order_index[param_k]
        X_k = design_matrices[param_k]
        offset_k = offsets.get(param_k, np.zeros(len(y), dtype=np.float64))

        score_eta = np.asarray(bundle.score[:, k_idx], dtype=np.float64)
        hessian_eta = np.asarray(bundle.hessian[:, k_idx, k_idx], dtype=np.float64)
        working_weights = np.maximum(-hessian_eta, 1e-10)

        cross_adjustment = np.zeros(len(y), dtype=np.float64)
        for param_j in params:
            if param_j == param_k:
                continue
            j_idx = order_index[param_j]
            cross_adjustment += (
                np.asarray(bundle.hessian[:, k_idx, j_idx], dtype=np.float64)
                * delta_eta[param_j]
            )

        eta_base = eta_new[param_k] - offset_k
        z_adjusted = eta_base + (score_eta + cross_adjustment) / working_weights
        beta_proposed = _safe_eta_weighted_least_squares(
            X=X_k,
            z=z_adjusted,
            working_weights=working_weights,
            observation_weights=weights,
        )
        step = float(step_sizes.get(param_k, 1.0))
        beta_k_new = coefficients[param_k] + step * (
            beta_proposed - coefficients[param_k]
        )
        eta_k_new = X_k @ beta_k_new + offset_k

        delta_eta[param_k] = eta_k_new - eta_old[param_k]
        eta_new[param_k] = eta_k_new
        new_coefs[param_k] = beta_k_new
        new_params[param_k] = np.asarray(
            family.link_inverses[param_k](jnp.asarray(eta_k_new)), dtype=np.float64
        )

    return new_params, eta_new, new_coefs, bundle


def cg_fit_v2(
    formula: str,
    family: Any,
    data: dict,
    sigma_formula: str = "~ 1",
    parameter_formulas: dict | None = None,
    weights: np.ndarray | None = None,
    max_iter: int = 50,
    tol: float = 1e-4,
    step_sizes: dict | None = None,
    verbose: bool = False,
) -> GAMLSSModel:
    """Fit with the eta-level CG IRLS cross-derivative backend."""
    family = resolve_family(family)
    response_name, _ = _parse_formula(formula)
    resolved_formulas = _resolve_parameter_formulas(
        response_name, family, formula, sigma_formula, parameter_formulas
    )

    y = np.asarray(data[response_name], dtype=np.float64)
    n = len(y)
    w = (
        np.ones(n, dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )

    fixed_parameter_values = _resolve_fixed_parameter_values(family, data, n)
    design_matrices: Dict[str, np.ndarray] = {}
    predictor_labels: Dict[str, list[str]] = {}
    smooth_infos: Dict[str, Any] = {}

    for parameter in family.parameters:
        if parameter in fixed_parameter_values:
            continue
        _, X, labels, smooth = _build_design_matrix_with_smooths(
            resolved_formulas[parameter], data, weights=weights
        )
        design_matrices[parameter] = np.asarray(X, dtype=np.float64)
        predictor_labels[parameter] = labels
        smooth_infos[parameter] = smooth

    params: dict[str, np.ndarray] = {}
    coefs: dict[str, np.ndarray] = {}
    etas: dict[str, np.ndarray] = {}
    mu_X = design_matrices["mu"]
    beta_mu = _initial_mu_beta(
        family, mu_X, y, w, fixed_parameter_values=fixed_parameter_values
    )
    eta_mu = mu_X @ beta_mu
    mu = np.asarray(family.link_inverses["mu"](jnp.asarray(eta_mu)), dtype=np.float64)
    params["mu"], coefs["mu"], etas["mu"] = mu, beta_mu, eta_mu

    for p in family.parameters:
        if p == "mu" or p in fixed_parameter_values:
            continue
        X = design_matrices[p]
        init = (
            _initial_sigma(
                family, y, mu, w, fixed_parameter_values=fixed_parameter_values
            )
            if p == "sigma"
            else _initial_parameter_value(family, p, y, mu, w)
        )
        eta_init = np.full(
            n,
            float(
                np.asarray(
                    family.link_functions[p](jnp.asarray([init], dtype=jnp.float64))
                )[0]
            ),
        )
        beta, *_ = np.linalg.lstsq(X, eta_init, rcond=None)
        eta = X @ beta
        pv = np.asarray(family.link_inverses[p](jnp.asarray(eta)), dtype=np.float64)
        params[p], coefs[p], etas[p] = pv, beta, eta

    for p, v in fixed_parameter_values.items():
        params[p] = np.asarray(v, dtype=np.float64)

    g_dev = _global_deviance(family, y, w, params)
    it = 0
    history = [g_dev]
    line_search_steps: list[int] = []
    steps = {"mu": 1.0, "sigma": 1.0, "nu": 1.0, "tau": 1.0, **(step_sizes or {})}
    offsets = {p: np.zeros(n, dtype=np.float64) for p in family.estimable_parameters}
    last_bundle: EtaDerivativeBundle | None = None

    while it < max_iter:
        old_g_dev = g_dev
        proposed_params, proposed_etas, proposed_coefs, last_bundle = cg_outer_step(
            y=y,
            design_matrices=design_matrices,
            parameter_values=params,
            etas=etas,
            coefficients=coefs,
            family=family,
            weights=w,
            step_sizes=steps,
            offsets=offsets,
        )

        accepted = False
        for halving in range(31):
            alpha = 0.5**halving
            candidate_etas = {
                p: etas[p] + alpha * (proposed_etas[p] - etas[p])
                for p in family.estimable_parameters
            }
            candidate_coefs = {
                p: coefs[p] + alpha * (proposed_coefs[p] - coefs[p])
                for p in family.estimable_parameters
            }
            candidate_params = _params_from_etas(
                family, candidate_etas, fixed_parameter_values=fixed_parameter_values
            )
            candidate_g_dev = _global_deviance(family, y, w, candidate_params)
            if np.isfinite(candidate_g_dev) and candidate_g_dev <= old_g_dev + 1e-8:
                params, etas, coefs = candidate_params, candidate_etas, candidate_coefs
                g_dev = candidate_g_dev
                line_search_steps.append(halving)
                accepted = True
                break

        it += 1
        history.append(g_dev)
        if verbose:
            print(
                f"CG_IRLS_CROSS iter {it}: deviance={g_dev:.6f}, change={abs(old_g_dev - g_dev):.2e}"
            )
        if not accepted or abs(old_g_dev - g_dev) <= tol:
            break

    converged = len(history) > 1 and abs(history[-2] - history[-1]) <= tol
    if len(history) > 1 and not history[-1] < history[-2]:
        # Remove duplicate convergence probe points for monotonicity diagnostics.
        history = history[:-1]
    fitted_values_jax = {
        p: jnp.asarray(params[p], dtype=jnp.float64) for p in family.parameters
    }
    coefficients_jax = {
        p: jnp.asarray(coefs.get(p, np.array([params[p][0]])), dtype=jnp.float64)
        for p in family.parameters
    }
    linear_predictors_jax = {
        p: jnp.asarray(
            etas.get(p, np.asarray(family.link_functions[p](jnp.asarray(params[p])))),
            dtype=jnp.float64,
        )
        for p in family.parameters
    }
    design_matrices_jax = {
        p: jnp.asarray(design_matrices.get(p, np.zeros((n, 1))), dtype=jnp.float64)
        for p in family.parameters
    }
    terms = {
        p: (
            _fixed_parameter_term(response_name, p)
            if p in fixed_parameter_values
            else {
                "term_labels": predictor_labels.get(p, []),
                "response": response_name,
                "intercept": _is_intercept_only_formula(resolved_formulas[p]),
                "formula": resolved_formulas[p],
            }
        )
        for p in family.parameters
    }
    # Compute df_fit using effective degrees of freedom for smooth terms.
    df_fit, smooth_edf = df_fit_with_smooth_edf(
        coefficients=coefs,
        estimable_parameters=family.estimable_parameters,
        design_matrices=design_matrices,
        weights=w,
        smooth_infos=smooth_infos,
    )

    rqres_callable = _build_rqres_callable(family)
    mu_vals = params["mu"]
    sigma_vals = params.get("sigma")
    if rqres_callable is not None:
        residual_values = rqres_callable(y=y, mu=mu_vals, sigma=sigma_vals)
    else:
        residual_values = _compute_residuals(family, y, mu_vals, sigma_vals)
    hessian_shape = (
        None
        if last_bundle is None
        else tuple(int(v) for v in last_bundle.hessian.shape)
    )

    return GAMLSSModel(
        par=family.parameters,
        family=family,
        df_fit=df_fit,
        g_dev=g_dev,
        n=n,
        y=jnp.asarray(y, dtype=jnp.float64),
        fitted_values=fitted_values_jax,
        coefficients=coefficients_jax,
        linear_predictors=linear_predictors_jax,
        working_vectors={p: linear_predictors_jax[p] for p in family.parameters},
        iterative_weights={
            p: jnp.ones(n, dtype=jnp.float64) for p in family.parameters
        },
        offsets={p: jnp.zeros(n, dtype=jnp.float64) for p in family.parameters},
        formulas=resolved_formulas,
        terms=terms,
        design_matrices=design_matrices_jax,
        weights=jnp.asarray(w, dtype=jnp.float64),
        residuals=jnp.asarray(residual_values, dtype=jnp.float64),
        rqres=rqres_callable,
        iter=it,
        type=family.type,
        parameters=family.parameters,
        call={
            "data": data,
            "formula": resolved_formulas["mu"],
            "parameter_formulas": dict(resolved_formulas),
            "method": "CG_IRLS_CROSS",
        },
        control={"n.cyc": max_iter},
        additional_slots={
            "method": "CG",
            "cg_backend": "CG_IRLS_CROSS",
            "cg_cross_derivatives": "eta_correction",
            "cg_iterations": it,
            "cg_converged": converged,
            "cg_final_deviance": float(g_dev),
            "cg_line_search_steps": tuple(line_search_steps),
            "cg_eta_hessian_shape": hessian_shape,
            "smooth_edf": smooth_edf,
            "aic": float(g_dev + 2.0 * df_fit),
            "sbc": float(g_dev + np.log(max(n, 1)) * df_fit),
            "df.residual": float(n - df_fit),
            "df_residual": float(n - df_fit),
            "deviance_history": tuple(float(v) for v in history),
        },
    )
