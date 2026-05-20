# SPDX-License-Identifier: GPL-3.0-or-later
"""Integration layer: bridge between gamlss() and the JAX RS core.

This module provides ``gamlss_rs_jax``, a drop-in replacement for the
NumPy-based RS path that uses ``jax_rs_fit_core`` under the hood.

Usage (via gamlss() entry point)::

    model = gamlss("y ~ x", family=NO(), data=data, method="RS_JAX")

The function:
1. Reuses all existing formula parsing and design-matrix helpers.
2. Reuses all existing initialisation helpers (_initial_mu_beta, etc.).
3. Calls ``jax_rs_fit_core`` for the actual fitting.
4. Wraps the result in a ``GAMLSSModel`` identical to the NumPy path.

Families supported: NO, GA, PO, BI, WEI, TF.
For other families, a ``ValueError`` is raised with a helpful message.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

import jax.numpy as jnp
import numpy as np

from ..controls import GAMLSSControl, GLIMControl, gamlss_control, glim_control
from ..distributions import resolve_family
from ..families import FamilyDefinition
from ..model import GAMLSSModel
from .jax_family_specs import get_jax_spec, supported_families
from .jax_rs_core import jax_rs_fit_core


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def gamlss_rs_jax(
    formula: str,
    family: Any,
    data: Mapping[str, Any],
    sigma_formula: str = "~1",
    parameter_formulas: Mapping[str, str] | None = None,
    weights: Any | None = None,
    control: GAMLSSControl | None = None,
    i_control: GLIMControl | None = None,
    max_inner: int = 1,
    verbose: bool = False,
    routing_decision: dict[str, Any] | None = None,
) -> GAMLSSModel:
    """Fit a GAMLSS model using the JAX-native RS algorithm.

    This is the ``method='RS_JAX'`` backend for :func:`omnilss.gamlss`.
    It compiles the entire RS fitting loop to XLA and runs it on the
    available device (CPU/GPU/TPU) without host-device round-trips.

    Parameters
    ----------
    formula : str
        Formula for the μ parameter, e.g. ``"y ~ x1 + x2"``.
    family : str or FamilyDefinition
        Distribution family.  Must be one of the JAX-supported families:
        ``NO``, ``GA``, ``PO``, ``BI``, ``WEI``, ``TF``.
    data : dict
        Data dictionary mapping variable names to arrays.
    sigma_formula : str, default ``"~1"``
        Formula for the σ parameter.
    parameter_formulas : dict, optional
        Formulas for ν and τ parameters (TF only uses ν).
    weights : array-like, optional
        Observation weights.
    control : GAMLSSControl, optional
        Fitting control parameters (``n_cyc``, ``c_crit``).
    i_control : GLIMControl, optional
        Inner GLIM control (currently unused in JAX path).
    max_inner : int, default 1
        Fixed number of inner IRLS iterations per parameter per outer step.
    verbose : bool, default False
        Print convergence information.

    Returns
    -------
    GAMLSSModel
        Fitted model, identical structure to the NumPy RS path.

    Raises
    ------
    ValueError
        If the family is not supported by the JAX RS core.

    Notes
    -----
    Smooth terms (``pb()``, ``ps()``, ``cs()``) are **not** supported in
    this path.  Use ``method='RS'`` for models with smooth terms.

    The first call triggers JIT compilation (cold time).  Subsequent calls
    with the same formula structure and family reuse the compiled XLA graph.
    """
    # ── Resolve family ──────────────────────────────────────────────────────
    family = resolve_family(family)

    # ── Check JAX support ───────────────────────────────────────────────────
    if family.name not in supported_families():
        raise ValueError(
            f"Family '{family.name}' is not supported by method='RS_JAX'. "
            f"Supported families: {supported_families()}. "
            f"Use method='RS' instead."
        )

    # ── Controls ────────────────────────────────────────────────────────────
    control   = gamlss_control() if control is None else control
    i_control = glim_control()   if i_control is None else i_control

    # ── Import fitting helpers (reuse existing infrastructure) ──────────────
    from ..fitting import (
        _build_design_matrix,
        _build_rqres_callable,
        _compute_residuals,
        _fixed_parameter_term,
        _initial_mu_beta,
        _initial_parameter_value,
        _initial_sigma,
        _normalize_parameter_formula,
        _parse_formula,
        _resolve_fixed_parameter_values,
        _resolve_parameter_formulas,
        _has_smooth_terms,
    )

    # ── Parse formula ────────────────────────────────────────────────────────
    response_name, _ = _parse_formula(formula)
    resolved_formulas = _resolve_parameter_formulas(
        response=response_name,
        family=family,
        mu_formula=formula,
        sigma_formula=sigma_formula,
        parameter_formulas=parameter_formulas,
    )

    # ── Reject smooth terms ──────────────────────────────────────────────────
    for param, fml in resolved_formulas.items():
        if _has_smooth_terms(fml):
            raise ValueError(
                f"method='RS_JAX' does not support smooth terms (found in "
                f"formula for '{param}': '{fml}'). Use method='RS' instead."
            )

    # ── Data preparation ─────────────────────────────────────────────────────
    y = np.asarray(data[response_name], dtype=np.float64)
    n = len(y)
    w = np.ones(n, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)

    fixed_parameter_values = _resolve_fixed_parameter_values(family, data, n)

    # ── Build design matrices ─────────────────────────────────────────────────
    design_matrices_np: dict[str, np.ndarray] = {}
    predictor_labels:   dict[str, list[str]]  = {}

    for param in family.estimable_parameters:
        _, X_param, labels_param = _build_design_matrix(resolved_formulas[param], data)
        design_matrices_np[param] = X_param
        predictor_labels[param]   = labels_param

    # ── Initialise parameters ─────────────────────────────────────────────────
    X_mu = design_matrices_np["mu"]
    eps = np.finfo(np.float64).eps

    if family.name == "BI":
        # Standard Bernoulli/logistic GLM IRLS cold-start.  This is deliberately
        # not a NumPy RS warm-start: it only initializes the single mu predictor
        # from the canonical GLM equations and avoids the unstable linear-y
        # initialization for binary responses.
        beta_mu = np.zeros(X_mu.shape[1], dtype=np.float64)
        if X_mu.shape[1] > 0:
            p0 = np.clip(np.average(y, weights=w), 1e-6, 1.0 - 1e-6)
            beta_mu[0] = float(np.log(p0 / (1.0 - p0)))
        for _ in range(8):
            eta_work = X_mu @ beta_mu
            p_work = np.clip(1.0 / (1.0 + np.exp(-eta_work)), 1e-6, 1.0 - 1e-6)
            ww = np.clip(p_work * (1.0 - p_work) * w, 1e-10, 1e10)
            z = eta_work + (y - p_work) / np.clip(
                p_work * (1.0 - p_work), 1e-10, None
            )
            sqrt_ww = np.sqrt(ww)
            beta_next, _, _, _ = np.linalg.lstsq(
                X_mu * sqrt_ww[:, None], z * sqrt_ww, rcond=None
            )
            if np.max(np.abs(beta_next - beta_mu)) < 1e-8:
                beta_mu = beta_next
                break
            beta_mu = beta_next
    elif family.name == "WEI":
        # Weibull uses a log link for mu; log-y least squares gives a stable
        # shape-independent slope cold-start and lets subsequent RS updates
        # refine the scale and shape jointly.
        beta_mu, _, _, _ = np.linalg.lstsq(
            X_mu, np.log(np.maximum(y, eps)), rcond=None
        )
    else:
        beta_mu = _initial_mu_beta(
            family, X_mu, y, w,
            fixed_parameter_values=fixed_parameter_values,
        )

    eta_mu = X_mu @ beta_mu
    mu     = np.asarray(family.link_inverses["mu"](jnp.asarray(eta_mu)), dtype=np.float64)

    init_params_list: list[np.ndarray] = [mu]
    init_etas_list:   list[np.ndarray] = [eta_mu]

    for param in family.estimable_parameters:
        if param == "mu":
            continue
        if param == "sigma":
            init_val = _initial_sigma(family, y, mu, w, fixed_parameter_values=fixed_parameter_values)
        else:
            init_val = _initial_parameter_value(family, param, y, mu, w)

        X_param  = design_matrices_np[param]
        eta_init = np.full(n, float(
            np.asarray(family.link_functions[param](jnp.asarray([init_val])))[0]
        ), dtype=np.float64)
        beta_init, _, _, _ = np.linalg.lstsq(X_param, eta_init, rcond=None)
        eta_param = X_param @ beta_init
        param_val = np.asarray(family.link_inverses[param](jnp.asarray(eta_param)), dtype=np.float64)

        init_params_list.append(param_val)
        init_etas_list.append(eta_param)

    # ── Get JAX spec ──────────────────────────────────────────────────────────
    # For BI, pass bd if it's a scalar (constant denominator)
    spec_kwargs: dict[str, Any] = {}
    if family.name == "BI" and "bd" in fixed_parameter_values:
        bd_arr = fixed_parameter_values["bd"]
        if np.all(bd_arr == bd_arr[0]):
            spec_kwargs["bd"] = float(bd_arr[0])
    spec = get_jax_spec(family.name, **spec_kwargs)

    # ── Cold-start only: do not run NumPy RS for initialization. ──────────────

    # ── Convert to JAX arrays ─────────────────────────────────────────────────
    y_jax   = jnp.asarray(y, dtype=jnp.float64)
    w_jax   = jnp.asarray(w, dtype=jnp.float64)
    Xs_jax  = tuple(jnp.asarray(design_matrices_np[p], dtype=jnp.float64)
                    for p in family.estimable_parameters)

    init_params_jax = jnp.stack(
        [jnp.asarray(p, dtype=jnp.float64) for p in init_params_list], axis=0
    )  # [n_params, n]
    init_etas_jax = jnp.stack(
        [jnp.asarray(e, dtype=jnp.float64) for e in init_etas_list], axis=0
    )  # [n_params, n]

    # ── Run JAX RS core ───────────────────────────────────────────────────────
    if verbose:
        print(f"[RS_JAX] Family: {family.name}, n={n}, "
              f"params={list(family.estimable_parameters)}")

    result = jax_rs_fit_core(
        y=y_jax,
        Xs=Xs_jax,
        init_params=init_params_jax,
        init_etas=init_etas_jax,
        obs_weights=w_jax,
        spec=spec,
        max_outer=control.n_cyc,
        max_inner=max_inner,   # default 1 IRLS step per outer iteration
        eta_clip_scale=3.0,
        tol=control.c_crit,
    )

    if verbose:
        print(f"[RS_JAX] Converged={result.converged}, "
              f"iterations={result.iterations}, deviance={result.g_dev:.6f}")

    # ── Unpack result ─────────────────────────────────────────────────────────
    param_names = list(family.estimable_parameters)
    fitted_vals_np: dict[str, np.ndarray] = {}
    etas_np:        dict[str, np.ndarray] = {}
    betas_np:       dict[str, np.ndarray] = {}

    for k, param in enumerate(param_names):
        fitted_vals_np[param] = np.asarray(result.params[k], dtype=np.float64)
        etas_np[param]        = np.asarray(result.etas[k],   dtype=np.float64)
        betas_np[param]       = np.asarray(result.betas[k],  dtype=np.float64)

    # Add fixed parameters
    for param, val in fixed_parameter_values.items():
        fitted_vals_np[param] = val

    # ── Build GAMLSSModel ─────────────────────────────────────────────────────
    fitted_values_jax = {
        p: jnp.asarray(fitted_vals_np[p], dtype=jnp.float64)
        for p in family.parameters
    }
    coefficients_jax = {
        p: jnp.asarray(betas_np[p], dtype=jnp.float64)
        for p in family.estimable_parameters
    }
    linear_predictors_jax = {
        p: jnp.asarray(etas_np[p], dtype=jnp.float64)
        for p in family.estimable_parameters
    }
    # Fixed parameters: store as-is
    for param, val in fixed_parameter_values.items():
        linear_predictors_jax[param] = jnp.asarray(val, dtype=jnp.float64)

    design_matrices_jax = {
        p: jnp.asarray(design_matrices_np[p], dtype=jnp.float64)
        for p in family.estimable_parameters
    }
    for param in fixed_parameter_values:
        design_matrices_jax[param] = jnp.zeros((n, 1), dtype=jnp.float64)

    # Working vectors (use linear predictors as proxy)
    working_vectors_jax = {
        p: linear_predictors_jax.get(p, jnp.zeros(n, dtype=jnp.float64))
        for p in family.parameters
    }
    iterative_weights_jax = {p: jnp.ones(n, dtype=jnp.float64) for p in family.parameters}
    offsets_jax           = {p: jnp.zeros(n, dtype=jnp.float64) for p in family.parameters}

    # Terms
    terms: dict[str, Any] = {}
    for param in family.parameters:
        if param in fixed_parameter_values:
            terms[param] = _fixed_parameter_term(response_name, param)
        else:
            terms[param] = {
                "term_labels": predictor_labels.get(param, []),
                "response":    response_name,
                "intercept":   True,
                "formula":     resolved_formulas[param],
            }

    # Residuals
    rqres_callable = _build_rqres_callable(family)
    mu_final    = fitted_vals_np.get("mu",    np.zeros(n))
    sigma_final = fitted_vals_np.get("sigma", None)
    if rqres_callable is not None:
        residual_values = rqres_callable(y=y, mu=mu_final, sigma=sigma_final)
    else:
        residual_values = _compute_residuals(family, y, mu_final, sigma_final)

    # df_fit
    df_fit = float(sum(betas_np[p].size for p in family.estimable_parameters))

    return GAMLSSModel(
        par=tuple(family.parameters),
        family=family,
        df_fit=df_fit,
        g_dev=result.g_dev,
        n=n,
        y=y_jax,
        fitted_values=fitted_values_jax,
        coefficients=coefficients_jax,
        linear_predictors=linear_predictors_jax,
        working_vectors=working_vectors_jax,
        iterative_weights=iterative_weights_jax,
        offsets=offsets_jax,
        formulas=dict(resolved_formulas),
        terms=terms,
        design_matrices=design_matrices_jax,
        weights=w_jax,
        residuals=residual_values,
        rqres=rqres_callable,
        iter=result.iterations,
        type=family.type,
        parameters=tuple(family.parameters),
        call={
            "data":              data,
            "formula":           resolved_formulas["mu"],
            "parameter_formulas": dict(resolved_formulas),
            "method":            "RS_JAX",
        },
        control={"n.cyc": control.n_cyc, **asdict(control)},
        additional_slots={
            "method":          "RS_JAX",
            "G.deviance":      result.g_dev,
            "P.deviance":      result.g_dev,
            "noObs":           int(n),
            "aic":             result.g_dev + 2.0 * df_fit,
            "sbc":             result.g_dev + np.log(max(n, 1)) * df_fit,
            "df.residual":     float(n - df_fit),
            "converged":       result.converged,
            "cycles":          result.iterations,
            "rs_jax_max_inner": max_inner,
            "method_routing": routing_decision,
        },
    )


def gamlss_rs_jax_batch(
    formula: str,
    families: Any,
    datasets: Mapping[str, Any] | list[Mapping[str, Any]],
    sigma_formula: str = "~1",
    parameter_formulas: Mapping[str, str] | None = None,
    weights: Any | list[Any] | None = None,
    control: GAMLSSControl | None = None,
    i_control: GLIMControl | None = None,
    max_inner: int = 1,
    verbose: bool = False,
) -> list[GAMLSSModel]:
    """Fit multiple JAX RS models through the formula integration layer.

    This is the formula-level companion to ``batch_jax_rs_fit``.  It preserves
    exactly the same cold-start-only semantics as ``gamlss_rs_jax`` and accepts
    either one dataset/family broadcast to all models or one dataset/family per
    model.

    Parameters
    ----------
    formula : str
        Formula for the mu parameter.
    families : family object, string, or list
        One family broadcast to all datasets, or one family per dataset.
    datasets : mapping or list[mapping]
        One dataset or a list of datasets.
    weights : array-like, list[array-like], optional
        One weight vector broadcast to all models, or one weight vector per
        dataset.

    Returns
    -------
    list[GAMLSSModel]
        Fitted models in input order.
    """
    dataset_list = list(datasets) if isinstance(datasets, list) else [datasets]
    k_models = len(dataset_list)

    if isinstance(families, list):
        if len(families) != k_models:
            raise ValueError("families must be one value or one family per dataset.")
        family_list = families
    else:
        family_list = [families for _ in range(k_models)]

    if isinstance(weights, list):
        if len(weights) != k_models:
            raise ValueError("weights must be one value or one weight vector per dataset.")
        weights_list = weights
    else:
        weights_list = [weights for _ in range(k_models)]

    return [
        gamlss_rs_jax(
            formula=formula,
            family=family_list[idx],
            data=dataset_list[idx],
            sigma_formula=sigma_formula,
            parameter_formulas=parameter_formulas,
            weights=weights_list[idx],
            control=control,
            i_control=i_control,
            max_inner=max_inner,
            verbose=verbose,
        )
        for idx in range(k_models)
    ]


__all__ = ["gamlss_rs_jax", "gamlss_rs_jax_batch"]
