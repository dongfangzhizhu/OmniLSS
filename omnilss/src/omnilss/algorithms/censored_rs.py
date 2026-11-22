"""Censored-response RS fitting path.

Dispatched from gamlss() when the response is a Surv(...) expression and the
family is a CensoredFamily.  It reuses the shared design-matrix builder, link
functions, working-weight computation and weighted least-squares solver, and
obtains the censored score/Hessian by automatic differentiation of the censored
log-likelihood.  Only method=RS is supported.
"""

from __future__ import annotations

from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from ..gamlss_cens import CensoredFamily, extract_surv_response


def _score_hessian(
    family: CensoredFamily, parameter: str, y_struct: Any, fitted: dict[str, np.ndarray]
):
    """Elementwise score and diagonal Hessian of the censored log-likelihood.

    The censored likelihood is a sum of per-observation terms, so grad(sum)
    is the elementwise score vector.  The Hessian is diagonal; we obtain it by
    a central difference of that score vector (the same strategy R's
    numeric.deriv uses), which stays O(n) instead of materialising an n x n
    matrix.
    """

    others = {
        name: jnp.asarray(values, dtype=jnp.float64)
        for name, values in fitted.items()
        if name != parameter
    }

    def total(value: Any) -> Any:
        kwargs = dict(others)
        kwargs[parameter] = value
        return jnp.sum(family.d(y_struct, log=True, **kwargs))

    start = jnp.asarray(fitted[parameter], dtype=jnp.float64)
    score = jax.grad(total)(start)
    step = 1e-5 * jnp.maximum(jnp.abs(start), 1.0)
    plus = jax.grad(total)(start + step)
    minus = jax.grad(total)(start - step)
    hessian = (plus - minus) / (2.0 * step)
    return np.asarray(score, dtype=np.float64), np.asarray(hessian, dtype=np.float64)


def censored_fit(
    formula: str,
    family: CensoredFamily,
    data: Mapping[str, Any],
    sigma_formula: str = "~1",
    parameter_formulas: Mapping[str, str] | None = None,
    weights: Any = None,
    max_iter: int = 20,
    tol: float = 1e-4,
    verbose: bool = False,
):
    """Fit a censored GAMLSS by RS/IRLS and return a GAMLSSModel."""

    from ..fitting import _build_design_matrix, _resolve_parameter_formulas
    from ..model import GAMLSSModel
    from ..fast_wls import solve_weighted_least_squares_cholesky
    from .rs_algorithm import compute_working_weights_and_response

    y_struct = extract_surv_response(formula, data)
    t1 = np.asarray(y_struct["time"], dtype=np.float64)
    n = int(t1.shape[0])
    if n == 0:
        raise ValueError("censored response is empty")

    rhs = formula.split("~", 1)[1].strip()
    synthetic = "__surv__"
    data2 = dict(data)
    data2[synthetic] = t1
    mu_formula = f"{synthetic} ~ {rhs}"
    resolved = _resolve_parameter_formulas(
        response=synthetic,
        family=family,
        mu_formula=mu_formula,
        sigma_formula=sigma_formula,
        parameter_formulas=parameter_formulas,
    )

    design: dict[str, np.ndarray] = {}
    labels: dict[str, list[str]] = {}
    for parameter in family.estimable_parameters:
        _, matrix, label = _build_design_matrix(resolved[parameter], data2)
        design[parameter] = np.asarray(matrix, dtype=np.float64)
        labels[parameter] = list(label)

    w = (
        np.ones(n, dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )

    fitted: dict[str, np.ndarray] = {}
    for parameter in family.estimable_parameters:
        start = max(float(np.mean(t1)), 1e-3) if parameter == "mu" else 1.0
        fitted[parameter] = np.full(n, start, dtype=np.float64)

    coefficients: dict[str, np.ndarray] = {}
    linear_predictors: dict[str, np.ndarray] = {}
    last_weights: dict[str, np.ndarray] = {}
    last_working_response: dict[str, np.ndarray] = {}
    deviance: float | None = None
    converged = False
    termination_reason = "max_iterations_reached"
    iteration = 0

    for iteration in range(1, max_iter + 1):
        previous = deviance
        for parameter in family.estimable_parameters:
            eta = np.asarray(
                family.link_functions[parameter](
                    jnp.asarray(fitted[parameter], dtype=jnp.float64)
                ),
                dtype=np.float64,
            )
            dmu_deta = np.asarray(
                family.link_derivatives[parameter](jnp.asarray(eta, dtype=jnp.float64)),
                dtype=np.float64,
            )
            link_deriv = 1.0 / (dmu_deta + 1e-15)
            score, hessian = _score_hessian(family, parameter, y_struct, fitted)
            working_weights, working_response = compute_working_weights_and_response(
                y=fitted[parameter],
                fitted_values=fitted[parameter],
                link_derivative=link_deriv,
                first_derivative=score,
                second_derivative=hessian,
                offset=np.zeros(n, dtype=np.float64),
                eta=eta,
            )
            last_weights[parameter] = np.asarray(working_weights, dtype=np.float64)
            last_working_response[parameter] = np.asarray(
                working_response, dtype=np.float64
            )
            beta = solve_weighted_least_squares_cholesky(
                X=design[parameter],
                z=working_response,
                w=working_weights * w,
            )
            eta_new = design[parameter] @ beta
            coefficients[parameter] = np.asarray(beta, dtype=np.float64)
            linear_predictors[parameter] = np.asarray(eta_new, dtype=np.float64)
            fitted[parameter] = np.asarray(
                family.link_inverses[parameter](
                    jnp.asarray(eta_new, dtype=jnp.float64)
                ),
                dtype=np.float64,
            )

        loglik = np.asarray(family.d(y_struct, log=True, **fitted), dtype=np.float64)
        deviance = float(-2.0 * np.sum(w * loglik))
        if verbose:
            print(f"censored RS cycle {iteration}: deviance = {deviance:.6f}")
        if previous is not None and abs(previous - deviance) < tol:
            converged = True
            termination_reason = "converged"
            break

    df_fit = float(sum(design[p].shape[1] for p in family.estimable_parameters))
    g_dev = float(deviance if deviance is not None else np.nan)

    # Parameter-block covariance from the final IRLS working weights
    # (cov = (X' W X)^-1).  Stored so se_fit / predict_all avoid the generic
    # vcov path, which assumes a numeric response.
    parameter_covariance: dict[str, np.ndarray] = {}
    for parameter in family.estimable_parameters:
        weights_vector = last_weights.get(parameter)
        if weights_vector is None:
            continue
        information = design[parameter].T @ (
            design[parameter] * (weights_vector * w)[:, None]
        )
        try:
            parameter_covariance[parameter] = np.linalg.inv(information)
        except np.linalg.LinAlgError:
            parameter_covariance[parameter] = np.linalg.pinv(information)

    terms_metadata = {
        parameter: {
            "term_labels": labels[parameter],
            "response": synthetic,
            "intercept": True,
            "formula": resolved[parameter],
        }
        for parameter in family.estimable_parameters
    }

    # Randomized quantile residuals via the censored PIT: rqres = Phi^-1(PIT).
    from .._fitting_residuals import _normal_quantile

    eps = float(np.finfo(np.float64).eps)
    pit = np.asarray(family.p(y_struct, **fitted), dtype=np.float64)
    pit = np.clip(pit, eps, 1.0 - eps)
    residual_values = _normal_quantile(pit)

    def censored_rqres(**kwargs: Any) -> Any:
        params = {name: kwargs[name] for name in family.parameters}
        values = np.asarray(family.p(kwargs["y"], **params), dtype=np.float64)
        values = np.clip(values, eps, 1.0 - eps)
        return jnp.asarray(_normal_quantile(values), dtype=jnp.float64)

    fitted_values = {
        parameter: jnp.asarray(fitted[parameter], dtype=jnp.float64)
        for parameter in family.parameters
    }
    return GAMLSSModel(
        par=family.parameters,
        family=family,
        df_fit=df_fit,
        g_dev=g_dev,
        n=n,
        y=jnp.asarray(t1, dtype=jnp.float64),
        fitted_values=fitted_values,
        coefficients={
            parameter: jnp.asarray(value, dtype=jnp.float64)
            for parameter, value in coefficients.items()
        },
        linear_predictors={
            parameter: jnp.asarray(value, dtype=jnp.float64)
            for parameter, value in linear_predictors.items()
        },
        formulas=resolved,
        terms=terms_metadata,
        design_matrices={
            parameter: jnp.asarray(matrix, dtype=jnp.float64)
            for parameter, matrix in design.items()
        },
        weights=jnp.asarray(w, dtype=jnp.float64),
        working_vectors={
            parameter: jnp.asarray(last_working_response[parameter], dtype=jnp.float64)
            for parameter in family.estimable_parameters
        },
        iterative_weights={
            parameter: jnp.asarray(last_weights[parameter], dtype=jnp.float64)
            for parameter in family.estimable_parameters
        },
        offsets={
            parameter: jnp.zeros(n, dtype=jnp.float64)
            for parameter in family.estimable_parameters
        },
        residuals=jnp.asarray(residual_values, dtype=jnp.float64),
        rqres=censored_rqres,
        iter=iteration,
        converged=converged,
        termination_reason=termination_reason,
        type=family.type,
        parameters=family.parameters,
        call={
            "data": dict(data),
            "formula": formula,
            "family": family.name,
            "method": "RS_censored",
        },
        control={"n.cyc": max_iter, "c.crit": tol},
        additional_slots={
            "method": "RS_censored",
            "censoring_type": family.censoring_type,
            "surv": {key: np.asarray(value) for key, value in y_struct.items()},
            "G.deviance": g_dev,
            "P.deviance": g_dev,
            "aic": g_dev + 2.0 * df_fit,
            "converged": converged,
            "cycles": iteration,
            "term_labels": labels,
            "parameter_covariance": parameter_covariance,
        },
    )


__all__ = ["censored_fit"]
