"""Cole-Green algorithm v2 with explicit outer-loop updates."""

from __future__ import annotations

from typing import Any, Dict

import jax.numpy as jnp
import numpy as np

from ..distributions import resolve_family
from ..fitting import (
    _build_design_matrix_with_smooths,
    _fixed_parameter_term,
    _initial_mu_beta,
    _initial_parameter_value,
    _initial_sigma,
    _is_intercept_only_formula,
    _parse_formula,
    _resolve_fixed_parameter_values,
    _resolve_parameter_formulas,
)
from ..model import GAMLSSModel
from .cg_algorithm import _compute_cross_derivatives, _irls_step_with_adjustment


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
) -> tuple[dict, dict, dict]:
    params = list(family.estimable_parameters)
    eta_old = {p: etas[p].copy() for p in params}
    eta_new = {p: etas[p].copy() for p in params}
    delta_eta = {p: np.zeros_like(etas[p]) for p in params}

    new_params = {p: parameter_values[p].copy() for p in family.parameters}
    new_coefs = {p: coefficients[p].copy() for p in params}

    for param_k in params:
        X_k = design_matrices[param_k]
        offset_k = offsets.get(param_k, np.zeros(len(y)))

        score_fn = family.score_functions[param_k]
        hessian_fn = family.hessian_functions[param_k]
        link_deriv = family.link_derivatives[param_k]

        param_dict_k = {"y": y, **new_params}
        score_k = np.asarray(score_fn(**param_dict_k), dtype=np.float64)
        hessian_kk = np.asarray(hessian_fn(**param_dict_k), dtype=np.float64)
        hessian_kk = np.where(hessian_kk < -1e-15, hessian_kk, -1e-15)

        dtheta_deta = np.asarray(link_deriv(jnp.asarray(eta_new[param_k])), dtype=np.float64)
        dtheta_deta = np.where(np.abs(dtheta_deta) < 1e-12, 1e-12, dtheta_deta)

        cross_adjustment = np.zeros(len(y), dtype=np.float64)
        for param_j in params:
            if param_j == param_k:
                continue
            cross_kj = _compute_cross_derivatives(y, new_params, family, param_k, param_j)
            cross_adjustment += cross_kj * delta_eta[param_j]

        eta_k_new, beta_k_new = _irls_step_with_adjustment(
            y=y,
            X=X_k,
            eta=eta_new[param_k],
            fitted=new_params[param_k],
            score=score_k,
            hessian_diag=hessian_kk,
            cross_adjustment=cross_adjustment,
            link_derivative=dtheta_deta,
            offset=offset_k,
            step_size=step_sizes.get(param_k, 1.0),
        )

        delta_eta[param_k] = eta_k_new - eta_old[param_k]
        eta_new[param_k] = eta_k_new
        new_coefs[param_k] = beta_k_new
        new_params[param_k] = np.asarray(family.link_inverses[param_k](jnp.asarray(eta_k_new)), dtype=np.float64)

    return new_params, eta_new, new_coefs


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
    family = resolve_family(family)
    response_name, _ = _parse_formula(formula)
    resolved_formulas = _resolve_parameter_formulas(response_name, family, formula, sigma_formula, parameter_formulas)

    y = np.asarray(data[response_name], dtype=np.float64)
    n = len(y)
    w = np.ones(n, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)

    fixed_parameter_values = _resolve_fixed_parameter_values(family, data, n)
    design_matrices: Dict[str, np.ndarray] = {}
    predictor_labels: Dict[str, list[str]] = {}
    smooth_infos: Dict[str, Any] = {}

    for parameter in family.parameters:
        if parameter in fixed_parameter_values:
            continue
        _, X, labels, smooth = _build_design_matrix_with_smooths(resolved_formulas[parameter], data, weights=weights)
        design_matrices[parameter] = X
        predictor_labels[parameter] = labels
        smooth_infos[parameter] = smooth

    params, coefs, etas = {}, {}, {}
    mu_X = design_matrices["mu"]
    beta_mu = _initial_mu_beta(family, mu_X, y, w, fixed_parameter_values=fixed_parameter_values)
    eta_mu = mu_X @ beta_mu
    mu = np.asarray(family.link_inverses["mu"](jnp.asarray(eta_mu)), dtype=np.float64)
    params["mu"], coefs["mu"], etas["mu"] = mu, beta_mu, eta_mu

    for p in family.parameters:
        if p == "mu" or p in fixed_parameter_values:
            continue
        X = design_matrices[p]
        init = _initial_sigma(family, y, mu, w, fixed_parameter_values=fixed_parameter_values) if p == "sigma" else _initial_parameter_value(family, p, y, mu, w)
        eta_init = np.full(n, float(np.asarray(family.link_functions[p](jnp.asarray([init], dtype=jnp.float64)))[0]))
        beta, *_ = np.linalg.lstsq(X, eta_init, rcond=None)
        eta = X @ beta
        pv = np.asarray(family.link_inverses[p](jnp.asarray(eta)), dtype=np.float64)
        params[p], coefs[p], etas[p] = pv, beta, eta

    for p, v in fixed_parameter_values.items():
        params[p] = v

    g_dev = float(np.sum(w * np.asarray(family.g_dev_inc(y=y, **params), dtype=np.float64)))
    g_dev_old = g_dev + 1.0
    it = 0
    history = [g_dev]
    steps = {"mu": 1.0, "sigma": 1.0, "nu": 1.0, "tau": 1.0, **(step_sizes or {})}
    offsets = {p: np.zeros(n, dtype=np.float64) for p in family.estimable_parameters}

    while abs(g_dev_old - g_dev) > tol and it < max_iter:
        it += 1
        g_dev_old = g_dev
        params, etas, coefs = cg_outer_step(y, design_matrices, params, etas, coefs, family, w, steps, offsets)
        g_dev = float(np.sum(w * np.asarray(family.g_dev_inc(y=y, **params), dtype=np.float64)))
        history.append(g_dev)
        if verbose:
            print(f"CG iter {it}: deviance={g_dev:.6f}, change={abs(g_dev_old-g_dev):.2e}")

    converged = abs(g_dev_old - g_dev) < tol
    fitted_values_jax = {p: jnp.asarray(params[p], dtype=jnp.float64) for p in family.parameters}
    coefficients_jax = {p: jnp.asarray(coefs.get(p, np.array([params[p][0]])), dtype=jnp.float64) for p in family.parameters}
    linear_predictors_jax = {p: jnp.asarray(etas.get(p, np.asarray(family.link_functions[p](jnp.asarray(params[p])))), dtype=jnp.float64) for p in family.parameters}
    design_matrices_jax = {p: jnp.asarray(design_matrices.get(p, np.zeros((n, 1))), dtype=jnp.float64) for p in family.parameters}
    terms = {p: (_fixed_parameter_term(response_name, p) if p in fixed_parameter_values else {"term_labels": predictor_labels.get(p, []), "response": response_name, "intercept": _is_intercept_only_formula(resolved_formulas[p]), "formula": resolved_formulas[p]}) for p in family.parameters}

    return GAMLSSModel(
        par=family.parameters, family=family, df_fit=float(sum(len(np.asarray(coefs.get(p, [0]))) for p in family.estimable_parameters)),
        g_dev=g_dev, n=n, y=jnp.asarray(y, dtype=jnp.float64), fitted_values=fitted_values_jax, coefficients=coefficients_jax,
        linear_predictors=linear_predictors_jax, working_vectors={p: linear_predictors_jax[p] for p in family.parameters},
        iterative_weights={p: jnp.ones(n, dtype=jnp.float64) for p in family.parameters}, offsets={p: jnp.zeros(n, dtype=jnp.float64) for p in family.parameters},
        formulas=resolved_formulas, terms=terms, design_matrices=design_matrices_jax, weights=jnp.asarray(w, dtype=jnp.float64),
        residuals=jnp.asarray(y - params["mu"], dtype=jnp.float64), rqres=None, iter=it, type=family.type, parameters=family.parameters,
        call={"data": data, "formula": resolved_formulas["mu"], "parameter_formulas": dict(resolved_formulas), "method": "CG"}, control={"n.cyc": max_iter},
        additional_slots={"method": "CG", "cg_iterations": it, "cg_converged": converged, "deviance_history": tuple(float(v) for v in history)},
    )
