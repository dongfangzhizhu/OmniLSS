"""Initial fitting helpers for staged migration of `gamlssML` and `gamlss`.

R source references:
- file: `gamlss/R/gamlssML.R`
- function: `gamlssML`
- file: `gamlss/R/gamlss-5.R`
- function: `gamlss`
- file: `gamlss/R/add.r`
- function: `additive.fit`
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import warnings
import math
from statistics import NormalDist
from typing import Any

import jax.numpy as jnp
import numpy as np

from .controls import GAMLSSControl, GLIMControl, gamlss_control, glim_control
from .distributions import resolve_family
from .families import FamilyDefinition
from ._fitting_utils import (
    _apply_method_step as _apply_method_step_shared,
    _eval_linear_term as _eval_linear_term_shared,
    _is_intercept_only_formula as _is_intercept_only_formula_shared,
    _normalize_parameter_formula as _normalize_parameter_formula_shared,
    _resolve_parameter_formulas as _resolve_parameter_formulas_shared,
    _weighted_least_squares as _weighted_least_squares_shared,
)
from ._fitting_init import (
    _initial_mu_beta as _initial_mu_beta_shared,
    _initial_parameter_value as _initial_parameter_value_shared,
    _initial_sigma as _initial_sigma_shared,
)
from ._fitting_residuals import (
    _build_rqres_callable as _build_rqres_callable_shared,
    _compute_residuals as _compute_residuals_shared,
)
from .formula_parser import parse_formula as parse_full_formula
from .model import GAMLSSModel

_STANDARD_NORMAL = NormalDist()


def _has_smooth_terms(formula: str) -> bool:
    """Check if formula contains smooth terms.

    Parameters
    ----------
    formula : str
        Formula string

    Returns
    -------
    has_smooths : bool
        True if formula contains smooth terms like pb(), ps(), cs(), etc.

    Examples
    --------
    >>> _has_smooth_terms("y ~ x1 + x2")
    False
    >>> _has_smooth_terms("y ~ pb(x1) + x2")
    True
    >>> _has_smooth_terms("y ~ x1 + ps(x2, df=5)")
    True
    """
    import re

    # Pattern to match smooth terms: pb(...), ps(...), cs(...), random(...), re(...), lo(...)
    smooth_pattern = r"\b(pb|ps|cs|random|re|lo)\s*\("
    return bool(re.search(smooth_pattern, formula))


def _parse_formula(formula: str) -> tuple[str, list[str]]:
    """Deprecated shim: use omnilss.formula_parser.parse_formula internally."""
    warnings.warn(
        "fitting._parse_formula is deprecated; formula_parser.parse_formula is used internally.",
        DeprecationWarning,
        stacklevel=2,
    )
    parsed = parse_full_formula(formula)
    predictors = [term.variable for term in parsed.linear_terms]
    return parsed.response, predictors


def _parse_formula_core(formula: str):
    """Internal parser returning ParsedFormula without deprecation warnings."""
    return parse_full_formula(formula)


def _eval_linear_term(term: str, data: Mapping[str, Any], n: int) -> np.ndarray:
    return _eval_linear_term_shared(term, data, n)


def _build_design_matrix(
    formula: str,
    data: Mapping[str, Any],
) -> tuple[str, np.ndarray, list[str]]:
    parsed = _parse_formula_core(formula)
    response = parsed.response
    predictors = [term.variable for term in parsed.linear_terms]
    n = len(np.asarray(data[response]))
    columns = [np.ones(n, dtype=np.float64)] if parsed.has_intercept else []
    labels = []
    for predictor in predictors:
        columns.append(_eval_linear_term(predictor, data, n))
        labels.append(predictor)
    design = np.column_stack(columns)
    return response, design, labels


def _build_design_matrix_with_smooths(
    formula: str,
    data: Mapping[str, Any],
    weights: np.ndarray | None = None,
) -> tuple[str, np.ndarray, list[str], Any]:
    """Build design matrix, detecting and handling smooth terms.

    This function checks if the formula contains smooth terms and uses
    the appropriate method to build the design matrix.

    Parameters
    ----------
    formula : str
        Formula string
    data : dict
        Data dictionary
    weights : np.ndarray, optional
        Observation weights

    Returns
    -------
    response : str
        Response variable name
    design : np.ndarray
        Design matrix
    labels : list[str]
        Predictor labels
    smooth_info : SmoothDesignInfo or None
        Smooth term information if present, None otherwise

    Examples
    --------
    >>> response, X, labels, smooth_info = _build_design_matrix_with_smooths(
    ...     "y ~ x1 + pb(x2, df=5)", data
    ... )
    """
    if _has_smooth_terms(formula):
        # Use smooth design builder
        from .smooth_fitting import build_smooth_design

        smooth_info = build_smooth_design(formula, data, weights)
        response = formula.split("~")[0].strip()
        return response, smooth_info.X, [], smooth_info
    else:
        # Use simple design builder
        response, design, labels = _build_design_matrix(formula, data)
        return response, design, labels, None


def _normalize_parameter_formula(response: str, formula: str) -> str:
    return _normalize_parameter_formula_shared(response, formula)


def _resolve_parameter_formulas(
    response: str,
    family: FamilyDefinition,
    mu_formula: str,
    sigma_formula: str,
    parameter_formulas: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Resolve staged per-parameter formulas with early validation.

    R reference:
    - `gamlss/R/gamlss-5.R::gamlss`

    Current staged behavior:
    - Accepts explicit formulas for `mu` and `sigma`.
    - Accepts a staged `parameter_formulas` mapping for future `nu/tau` work.
    - Rejects parameters that are unknown or unsupported by the current family.
    """

    return _resolve_parameter_formulas_shared(
        response,
        family,
        mu_formula,
        sigma_formula,
        parameter_formulas,
    )


def _prepare_fixed_parameter_array(
    value: Any,
    *,
    name: str,
    n_obs: int,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return np.full(n_obs, float(array), dtype=np.float64)

    flat = np.ravel(array).astype(np.float64, copy=False)
    if flat.size != n_obs:
        raise ValueError(
            f"fixed parameter {name!r} must have length {n_obs} or be scalar; got length {flat.size}"
        )
    return flat


def _resolve_fixed_parameter_values(
    family: FamilyDefinition,
    data: Mapping[str, Any],
    n_obs: int,
) -> dict[str, np.ndarray]:
    resolved: dict[str, np.ndarray] = {}
    for parameter in family.fixed_parameters or ():
        if parameter not in data:
            # Special case: BI family's bd parameter defaults to 1 (Bernoulli)
            if family.name == "BI" and parameter == "bd":
                resolved[parameter] = np.ones(n_obs, dtype=np.float64)
                continue

            raise ValueError(
                f"family {family.name!r} requires fixed parameter {parameter!r} in data"
            )
        resolved[parameter] = _prepare_fixed_parameter_array(
            data[parameter],
            name=parameter,
            n_obs=n_obs,
        )
    return resolved


def _fixed_parameter_formula(parameter: str) -> str:
    return f"<fixed:{parameter}>"


def _fixed_parameter_term(response: str, parameter: str) -> dict[str, Any]:
    return {
        "term_labels": [parameter],
        "response": response,
        "intercept": False,
        "formula": _fixed_parameter_formula(parameter),
        "fixed": True,
    }


def _initial_parameter_value(
    family: FamilyDefinition,
    parameter: str,
    y: np.ndarray,
    mu: np.ndarray,
    w: np.ndarray,
) -> float:
    """Construct staged starting values for non-mu parameters."""
    return _initial_parameter_value_shared(family, parameter, y, mu, w)


def _is_intercept_only_formula(formula: str) -> bool:
    return _is_intercept_only_formula_shared(formula)


def _weighted_least_squares(
    x: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
    smooth_info: Any = None,
) -> np.ndarray:
    """Weighted least squares with optional penalty for smooth terms.

    Parameters
    ----------
    x : np.ndarray
        Design matrix
    z : np.ndarray
        Working response
    w : np.ndarray
        Weights
    smooth_info : SmoothDesignInfo, optional
        Smooth term information for penalized fitting

    Returns
    -------
    coef : np.ndarray
        Fitted coefficients
    """
    return _weighted_least_squares_shared(x, z, w, smooth_info)


def _require_method_family_capability(
    family_name: str, method_name: str, *, allow_experimental: bool = True
) -> None:
    """Validate method/family support before starting an expensive fit."""

    from .family_capabilities import (
        FamilyCapabilityError,
        method_route_capability_report,
    )

    report = method_route_capability_report(
        family_name, method_name, strict=not allow_experimental
    )
    if report["code"] == "unknown_method":
        return
    if report["ok"]:
        return

    if method_name == "RS_JAX":
        raise FamilyCapabilityError(
            f"Family '{family_name}' is not supported by method='RS_JAX'. "
            "Use method='RS' instead."
        )
    raise FamilyCapabilityError(
        f"Family '{family_name}' cannot use method='{method_name}' "
        f"because capability feature '{report['feature']}' is unavailable "
        "at the requested evidence tier."
    )


def _apply_method_step(
    previous_beta: np.ndarray,
    proposed_beta: np.ndarray,
    method_name: str,
) -> np.ndarray:
    """Apply method-specific coefficient damping.

    Notes
    -----
    - RS: full step (no damping)
    - MIXED: 75% step (damped RS, experimental)
    - CG path: handled separately via ``fitting_cg.fit_cg()``;
      this branch applies a 50% damped fallback for the legacy inline path only.
    """

    return _apply_method_step_shared(previous_beta, proposed_beta, method_name)


def _compute_residuals(
    family: FamilyDefinition,
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray | None,
) -> jnp.ndarray:
    return _compute_residuals_shared(family, y, mu, sigma)


def _normal_quantile(probabilities: np.ndarray) -> np.ndarray:
    eps = np.finfo(np.float64).eps
    probs = np.clip(np.asarray(probabilities, dtype=np.float64), eps, 1.0 - eps)
    flattened = probs.ravel()
    quantiles = np.array(
        [_STANDARD_NORMAL.inv_cdf(float(value)) for value in flattened],
        dtype=np.float64,
    )
    return quantiles.reshape(probs.shape)


def _poisson_cdf_scalar(k: int, mu: float) -> float:
    if k < 0:
        return 0.0
    mu = max(float(mu), np.finfo(np.float64).eps)
    pmf = math.exp(-mu)
    total = pmf
    for index in range(1, k + 1):
        pmf *= mu / index
        total += pmf
    return float(min(max(total, 0.0), 1.0))


def _negative_binomial_cdf_scalar(k: int, mu: float, sigma: float) -> float:
    if k < 0:
        return 0.0
    eps = np.finfo(np.float64).eps
    mu = max(float(mu), eps)
    sigma = max(float(sigma), eps)
    size = 1.0 / sigma
    pmf = math.exp(size * math.log(size / (size + mu)))
    total = pmf
    for index in range(1, k + 1):
        pmf *= ((index - 1 + size) / index) * (mu / (size + mu))
        total += pmf
    return float(min(max(total, 0.0), 1.0))


def _geometric_cdf_scalar(k: int, mu: float) -> float:
    if k < 0:
        return 0.0
    mu = max(float(mu), np.finfo(np.float64).eps)
    ratio = mu / (1.0 + mu)
    return float(min(max(1.0 - ratio ** (k + 1), 0.0), 1.0))


def _zip_cdf_scalar(k: int, mu: float, sigma: float) -> float:
    if k < 0:
        return 0.0
    eps = np.finfo(np.float64).eps
    sigma = min(max(float(sigma), eps), 1.0 - eps)
    poisson_cdf = _poisson_cdf_scalar(k, mu)
    return float(min(max(sigma + (1.0 - sigma) * poisson_cdf, 0.0), 1.0))


def _discrete_midpoint_rqres(
    family: FamilyDefinition,
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray | None = None,
) -> np.ndarray:
    y_arr = np.asarray(y, dtype=np.float64)
    mu_arr = np.asarray(mu, dtype=np.float64)
    sigma_arr = None if sigma is None else np.asarray(sigma, dtype=np.float64)
    midpoint = np.zeros_like(y_arr, dtype=np.float64)

    for index, value in enumerate(y_arr):
        observed = int(np.floor(value))
        mu_value = float(mu_arr[index])
        sigma_value = None if sigma_arr is None else float(sigma_arr[index])

        if family.name == "PO":
            lower = _poisson_cdf_scalar(observed - 1, mu_value)
            upper = _poisson_cdf_scalar(observed, mu_value)
        elif family.name == "BI":
            prob = min(
                max(mu_value, np.finfo(np.float64).eps), 1.0 - np.finfo(np.float64).eps
            )
            lower = 0.0 if observed <= 0 else 1.0 - prob
            upper = 1.0 - prob if observed <= 0 else 1.0
        elif family.name == "GEOM":
            lower = _geometric_cdf_scalar(observed - 1, mu_value)
            upper = _geometric_cdf_scalar(observed, mu_value)
        elif family.name == "NBI":
            lower = _negative_binomial_cdf_scalar(
                observed - 1, mu_value, float(sigma_value)
            )
            upper = _negative_binomial_cdf_scalar(
                observed, mu_value, float(sigma_value)
            )
        elif family.name == "ZIP":
            lower = _zip_cdf_scalar(observed - 1, mu_value, float(sigma_value))
            upper = _zip_cdf_scalar(observed, mu_value, float(sigma_value))
        else:
            raise NotImplementedError(
                f"rqres is not implemented for family {family.name!r}"
            )

        midpoint[index] = 0.5 * (lower + upper)

    return _normal_quantile(midpoint)


def _build_rqres_callable(family: FamilyDefinition) -> Any | None:
    return _build_rqres_callable_shared(family)


def _initial_mu_beta(
    family: FamilyDefinition,
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    fixed_parameter_values: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Construct a family-aware starting point for the mu predictor."""
    return _initial_mu_beta_shared(family, x, y, w, fixed_parameter_values)


def _initial_sigma(
    family: FamilyDefinition,
    y: np.ndarray,
    mu: np.ndarray,
    w: np.ndarray,
    fixed_parameter_values: Mapping[str, np.ndarray] | None = None,
) -> float:
    """Construct a family-aware starting point for the sigma parameter."""
    return _initial_sigma_shared(family, y, mu, w, fixed_parameter_values)


def gamlss_ml(
    formula: str,
    family: Any | None = None,
    data: Mapping[str, Any] | None = None,
    weights: Any | None = None,
    sigma_formula: str = "~1",
    parameter_formulas: Mapping[str, str] | None = None,
    control: GAMLSSControl | None = None,
    i_control: GLIMControl | None = None,
) -> GAMLSSModel:
    """Staged Python port of `gamlssML` for currently supported families."""

    if data is None:
        raise ValueError("data is required")
    family = resolve_family(family)
    rqres_callable = _build_rqres_callable(family)

    control = gamlss_control() if control is None else control
    i_control = glim_control() if i_control is None else i_control

    response_name, _ = _parse_formula(formula)
    resolved_formulas = _resolve_parameter_formulas(
        response=response_name,
        family=family,
        mu_formula=formula,
        sigma_formula=sigma_formula,
        parameter_formulas=parameter_formulas,
    )

    # Build design matrix with smooth support
    _, mu_x, predictor_labels, mu_smooth_info = _build_design_matrix_with_smooths(
        resolved_formulas["mu"], data, weights=weights
    )

    y = np.asarray(data[response_name], dtype=np.float64)
    fixed_parameter_values = _resolve_fixed_parameter_values(family, data, len(y))
    if weights is None:
        w = np.ones_like(y, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)

    # Use penalized WLS if there are smooth terms
    if mu_smooth_info is not None:
        beta_mu = _weighted_least_squares(mu_x, y, w, smooth_info=mu_smooth_info)
    else:
        beta_mu = _initial_mu_beta(
            family,
            mu_x,
            y,
            w,
            fixed_parameter_values=fixed_parameter_values,
        )

    par = tuple(family.parameters)
    mu_linear = mu_x @ beta_mu
    mu = np.asarray(family.link_inverses["mu"](mu_linear), dtype=np.float64)

    fitted_values = {"mu": jnp.asarray(mu, dtype=jnp.float64)}
    coefficients = {"mu": jnp.asarray(beta_mu, dtype=jnp.float64)}
    linear_predictors = {
        "mu": jnp.asarray(
            family.link_functions["mu"](jnp.asarray(mu, dtype=jnp.float64)),
            dtype=jnp.float64,
        )
    }
    formulas = {"mu": resolved_formulas["mu"]}
    terms = {
        "mu": {
            "term_labels": predictor_labels,
            "response": response_name,
            "intercept": True,
            "formula": resolved_formulas["mu"],
        }
    }
    design_matrices = {"mu": jnp.asarray(mu_x, dtype=jnp.float64)}
    parameter_values: dict[str, np.ndarray] = {"mu": mu, **fixed_parameter_values}

    # Store smooth information for all parameters
    smooth_infos = {"mu": mu_smooth_info}

    for parameter in family.estimable_parameters:
        if parameter == "mu":
            continue
        parameter_formula = resolved_formulas[parameter]

        # Build design matrix with smooth support
        _, parameter_design, parameter_labels, param_smooth_info = (
            _build_design_matrix_with_smooths(parameter_formula, data, weights=weights)
        )
        smooth_infos[parameter] = param_smooth_info

        initial_value = _initial_parameter_value(family, parameter, y, mu, w)
        if parameter == "sigma":
            initial_value = _initial_sigma(
                family,
                y,
                mu,
                w,
                fixed_parameter_values=fixed_parameter_values,
            )
        eta_init = np.full(
            y.shape[0],
            float(
                np.asarray(
                    family.link_functions[parameter](
                        jnp.asarray([initial_value], dtype=jnp.float64)
                    ),
                    dtype=np.float64,
                )[0]
            ),
            dtype=np.float64,
        )

        # Use penalized WLS if there are smooth terms
        if param_smooth_info is not None:
            beta_parameter = _weighted_least_squares(
                parameter_design, eta_init, w, smooth_info=param_smooth_info
            )
            _, _, _, _ = None, None, None, None  # Dummy for compatibility
        else:
            beta_parameter, _, _, _ = np.linalg.lstsq(
                parameter_design, eta_init, rcond=None
            )

        eta_parameter = parameter_design @ beta_parameter
        parameter_vector = np.asarray(
            family.link_inverses[parameter](
                jnp.asarray(eta_parameter, dtype=jnp.float64)
            ),
            dtype=np.float64,
        )
        parameter_values[parameter] = parameter_vector
        fitted_values[parameter] = jnp.asarray(parameter_vector, dtype=jnp.float64)
        if parameter == "sigma" and _is_intercept_only_formula(parameter_formula):
            coefficients[parameter] = jnp.asarray([initial_value], dtype=jnp.float64)
        else:
            coefficients[parameter] = jnp.asarray(beta_parameter, dtype=jnp.float64)
        linear_predictors[parameter] = jnp.asarray(eta_parameter, dtype=jnp.float64)
        formulas[parameter] = parameter_formula
        terms[parameter] = {
            "term_labels": parameter_labels,
            "response": response_name,
            "intercept": True,
            "formula": parameter_formula,
        }
        design_matrices[parameter] = jnp.asarray(parameter_design, dtype=jnp.float64)

    for parameter, value in fixed_parameter_values.items():
        fitted_values[parameter] = jnp.asarray(value, dtype=jnp.float64)
        # Fixed parameters don't have link functions - they are used as-is
        linear_predictors[parameter] = jnp.asarray(value, dtype=jnp.float64)
        formulas[parameter] = _fixed_parameter_formula(parameter)
        terms[parameter] = _fixed_parameter_term(response_name, parameter)

    dev_kwargs = {"y": y, **parameter_values, **fixed_parameter_values}
    g_dev = float(np.sum(np.asarray(family.g_dev_inc(**dev_kwargs)) * w))
    deviance_history = (g_dev,)

    residual_values = (
        rqres_callable(y=y, mu=mu, sigma=parameter_values.get("sigma"))
        if rqres_callable is not None
        else _compute_residuals(family, y, mu, parameter_values.get("sigma"))
    )

    # Compute df_fit including smooth terms
    df_fit = 0.0
    smooth_edf = {}

    for parameter in family.estimable_parameters:
        # Add linear parameters
        df_fit += float(np.asarray(coefficients[parameter], dtype=np.float64).size)

        # Adjust for smooth terms
        if (
            smooth_infos.get(parameter) is not None
            and len(smooth_infos[parameter].smooth_fits) > 0
        ):
            from .smooth_fitting import compute_smooth_edf

            # Subtract nominal basis columns
            for smooth in smooth_infos[parameter].smooth_fits:
                start, end = smooth.basis_columns
                df_fit -= float(end - start)
            # Add effective df
            param_edf = compute_smooth_edf(
                design_matrices[parameter], w, smooth_infos[parameter].smooth_fits
            )
            df_fit += param_edf
            smooth_edf[parameter] = param_edf
        else:
            smooth_edf[parameter] = 0.0
    for parameter in family.fixed_parameters or ():
        smooth_edf[parameter] = 0.0

    return GAMLSSModel(
        par=par,
        family=family,
        df_fit=df_fit,
        g_dev=g_dev,
        n=len(y),
        y=jnp.asarray(y, dtype=jnp.float64),
        fitted_values=fitted_values,
        coefficients=coefficients,
        linear_predictors=linear_predictors,
        formulas=formulas,
        terms=terms,
        design_matrices=design_matrices,
        additional_slots={
            "G.deviance": g_dev,
            "P.deviance": g_dev,
            "noObs": int(len(y)),
            "df.residual": float(len(y) - df_fit),
            "aic": float(g_dev + df_fit * 2.0),
            "sbc": float(g_dev + df_fit * np.log(max(len(y), 1))),
            "method": "ML",
            "converged": True,
            "cycles": int(control.iter),
            "deviance_history": deviance_history,
            "smooth_fits": {
                p: smooth_infos[p].smooth_fits if smooth_infos.get(p) else []
                for p in par
            },
            "smooth_edf": smooth_edf,
        },
        call={
            "data": data,
            "formula": resolved_formulas["mu"],
            "parameter_formulas": dict(resolved_formulas),
        },
        control={"n.cyc": control.n_cyc, **asdict(control)},
        iter=control.iter,
        weights=jnp.asarray(w, dtype=jnp.float64),
        residuals=residual_values,
        type=family.type,
        parameters=par,
        rqres=rqres_callable,
    )


def gamlss(
    formula: str,
    sigma_formula: str = "~1",
    family: FamilyDefinition | None = None,
    data: Mapping[str, Any] | None = None,
    weights: Any | None = None,
    parameter_formulas: Mapping[str, str] | None = None,
    method: str = "RS",
    control: GAMLSSControl | None = None,
    i_control: GLIMControl | None = None,
    verbose: bool = False,
    strict_capabilities: bool = False,
    # New optimizer parameters
    optimizer: str = "adam",
    learning_rate: float = 0.01,
    max_iter: int | None = None,
    history_size: int = 10,
    **optimizer_kwargs,
) -> GAMLSSModel:
    """Fit a GAMLSS model.

    Parameters
    ----------
    formula : str
        Formula for the mu parameter, e.g. ``"y ~ x1 + x2"``.
    sigma_formula : str, default ``"~1"``
        Formula for the sigma parameter.
    family : FamilyDefinition or str or None
        Distribution family.  Pass a family object (e.g. ``NO()``) or a
        string name (e.g. ``"NO"``).
    data : dict or None
        Data dictionary mapping variable names to arrays.
    weights : array-like or None
        Observation weights.
    parameter_formulas : dict or None
        Formulas for additional parameters (nu, tau).
    method : str, default ``"RS"``
        Fitting algorithm selection.  One of:

        ``"RS"``
            NumPy IRLS (default).  This is the fastest CPU path, supports all
            OmniLSS distribution families, and supports smooth terms such as
            ``pb()``, ``cs()``, and ``ps()``.

        ``"RS_JAX"``
            JAX JIT-compiled IRLS.  This path currently supports the six core
            JAX families (NO, GA, PO, BI, WEI, TF) and does not support smooth
            terms.  It may be faster for large GPU/TPU workloads after
            hardware-specific crossover thresholds are configured.

        ``"auto"``
            Device-aware routing.  CPU always resolves to ``"RS"``.  GPU/TPU
            resolves to ``"RS_JAX"`` only when ``n`` is greater than or equal
            to ``config.GPU_CROSSOVER_N[family]`` or
            ``config.TPU_CROSSOVER_N[family]``; otherwise it resolves to
            ``"RS"``.  Current placeholder thresholds are ``math.inf`` until
            benchmarks establish a crossover for target hardware.

        Manual routing examples::

            # Force JAX for supported families, even on CPU, for testing.
            import omnilss.config as cfg
            cfg.FORCE_JAX = True

            # Force NumPy, even on a GPU host.
            model = gamlss("y ~ x", family="NO", data=data, method="RS")

            # Configure the current session's GPU threshold.
            cfg.set_crossover("gpu", n=50_000, family="NO")
            model = gamlss("y ~ x", family="NO", data=data, method="auto")

        ``"CG"``
            Cole-Green algorithm with full JAX Hessian cross-derivatives.

        ``"MIXED"``
            Mixed RS/CG damped update path.

        ``"joint"``
            Joint optimization with Optax (Adam/SGD/RMSprop/Adagrad).

        ``"lbfgs"``
            L-BFGS quasi-Newton method.

    control : GAMLSSControl or None
        Control parameters for the outer loop (n_cyc, c_crit).
    i_control : GLIMControl or None
        Control parameters for the inner GLIM loop.
    verbose : bool, default False
        Print detailed fitting progress.
    strict_capabilities : bool, default False
        If True, reject experimental family/method routes and allow only
        validated capability features.
    optimizer : str, default ``"adam"``
        Optimizer type for ``method="joint"``.
    learning_rate : float, default 0.01
        Learning rate for gradient-based optimizers.
    max_iter : int or None
        Maximum iterations (overrides ``control.n_cyc`` if provided).
    history_size : int, default 10
        L-BFGS history size (for ``method="lbfgs"``).
    **optimizer_kwargs
        Additional optimizer arguments.

    Returns
    -------
    GAMLSSModel
        Fitted GAMLSS model.

    Notes
    -----
    **Method selection guide**:

    - For most use cases, ``method="RS"`` (default) is the best choice.
    - ``method="RS_JAX"`` is available for GPU workloads but currently
      shows no speedup over NumPy RS on tested hardware (RTX 3060, n ≤ 500k).
      See ``docs/benchmarks/`` for details.
    - ``method="auto"`` is a forward-compatible alias that will automatically
      use JAX when a crossover point is established for your hardware.
      Override thresholds via ``omnilss.config.GPU_CROSSOVER_N``.

    **Customising auto-selection**::

        import omnilss.config as cfg
        cfg.set_crossover("gpu", n=50_000, family="NO")
        cfg.set_crossover("gpu", n=100_000)
        cfg.crossover_summary()

    Examples
    --------
    >>> model = gamlss("y ~ x", family=NO(), data=data)
    >>> model = gamlss("y ~ x", family="GA", data=data, method="RS")
    >>> model = gamlss("y ~ pb(x)", family=NO(), data=data)  # P-spline
    """

    if data is None:
        raise ValueError("data is required")
    family = resolve_family(family)
    if "algorithm" in optimizer_kwargs:
        # Backward-compatible alias used by examples and benchmark scripts.
        method = optimizer_kwargs.pop("algorithm")
    method_name = str(method).upper()
    requested_method_name = method_name
    routing_decision: dict[str, Any] | None = None

    # ── Device-aware RS routing ──────────────────────────────────────────────
    # ``method="RS"`` is the public default and now participates in the same
    # device-aware route selection as the forward-compatible ``method="auto"``
    # alias.  CPU and placeholder infinite thresholds still resolve to NumPy RS,
    # while configured GPU/TPU crossovers route supported families to RS_JAX.
    if method_name in {"AUTO", "RS"}:
        from . import config as _cfg

        requested_method = method_name
        try:
            _resp, _ = _parse_formula(formula)
            _n_obs = len(data[_resp])
        except Exception:
            _n_obs = 0
        _backend, _ = _cfg._current_backend()
        if _backend == "gpu":
            _table = _cfg.GPU_CROSSOVER_N
        elif _backend == "tpu":
            _table = _cfg.TPU_CROSSOVER_N
        else:
            _table = None
        _threshold = (
            _cfg._get_crossover(_table, family.name) if _table is not None else math.inf
        )
        _decision = _cfg.auto_select_method_trace(family.name, _n_obs)
        method_name = _decision.method.upper()
        routing_decision = {
            "requested_method": requested_method,
            "selected_method": method_name,
            "reason": _decision.reason,
            "reason_detail": _cfg.describe_method_routing_reason(_decision.reason),
            "backend": _decision.backend,
            "threshold": _decision.threshold,
            "n_obs": int(_n_obs),
            "family": family.name,
        }
        if verbose:
            print("OmniLSS method routing")
            print(
                f"  Requested: {requested_method} · Device: {_backend.upper()} "
                f"· n={_n_obs} · Family: {family.name}"
            )
            if _backend in {"gpu", "tpu"}:
                _threshold_text = "inf" if _threshold == math.inf else f"{_threshold:g}"
                print(
                    f"  {_backend.upper()} crossover threshold ({family.name}): "
                    f"{_threshold_text} -> selected {method_name}"
                )
            else:
                print(f"  CPU/unknown backend selected {method_name} (NumPy RS)")
            if method_name == "RS" and _backend in {"gpu", "tpu"}:
                print(
                    f'  Tip: run benchmarks and call cfg.set_crossover("{_backend}", n=...) '
                    "to enable automatic JAX acceleration"
                )

    if requested_method_name == "RS_JAX":
        import warnings

        if routing_decision is None:
            from . import config as _cfg

            _backend, _ = _cfg._current_backend()
            routing_decision = {
                "requested_method": "RS_JAX",
                "selected_method": "RS_JAX",
                "reason": "explicit_method_requested",
                "reason_detail": _cfg.describe_method_routing_reason("explicit_method_requested"),
                "backend": _backend,
                "threshold": None,
                "n_obs": None,
                "family": family.name,
            }

        warnings.warn(
            "method='RS_JAX' is deprecated as a user-facing route; use "
            "method='RS' and configure device-aware crossover thresholds instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    supported_methods = {"RS", "RS_JAX", "CG", "MIXED", "JOINT", "LBFGS"}
    if method_name not in supported_methods:
        raise ValueError(
            "method must be one of 'RS', 'RS_JAX', 'auto', 'CG', 'MIXED', "
            f"'joint', or 'lbfgs', got {method!r}"
        )

    _require_method_family_capability(
        family.name, method_name, allow_experimental=not strict_capabilities
    )

    # Handle new optimizer methods
    if method_name in {"JOINT", "LBFGS"}:
        # Import integration functions
        from .core.gamlss_integration import fit_with_joint_optimizer, fit_with_lbfgs

        # Get initial model using RS/ML
        if verbose:
            print(f"\n{'='*70}")
            print("Step 1: Getting initial estimates with RS/ML")
            print(f"{'='*70}")

        initial_model = gamlss_ml(
            formula=formula,
            sigma_formula=sigma_formula,
            parameter_formulas=parameter_formulas,
            family=family,
            data=data,
            weights=weights,
            control=control,
            i_control=i_control,
        )

        if verbose:
            print(f"Initial deviance: {initial_model.g_dev:.6f}")
            print(f"\n{'='*70}")
            print(f"Step 2: Refining with {method_name} optimizer")
            print(f"{'='*70}")

        # Prepare design matrices
        response_name, _ = _parse_formula(formula)
        resolved_formulas = _resolve_parameter_formulas(
            response=response_name,
            family=family,
            mu_formula=formula,
            sigma_formula=sigma_formula,
            parameter_formulas=parameter_formulas,
        )

        design_matrices = {}
        for param in family.estimable_parameters:
            if param in initial_model.design_matrices:
                design_matrices[param] = np.asarray(
                    initial_model.design_matrices[param], dtype=np.float64
                )

        y = np.asarray(data[response_name], dtype=np.float64)
        n = len(y)
        fixed_parameter_values = _resolve_fixed_parameter_values(family, data, n)

        if weights is None:
            w = np.ones(n, dtype=np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64)

        # Determine max_iter
        if max_iter is None:
            max_iter = control.n_cyc if control is not None else 1000

        # Fit with appropriate optimizer
        if method_name == "JOINT":
            return fit_with_joint_optimizer(
                initial_model=initial_model,
                family=family,
                design_matrices=design_matrices,
                y=y,
                weights=w,
                fixed_parameters=fixed_parameter_values,
                optimizer=optimizer,
                learning_rate=learning_rate,
                max_iter=max_iter,
                verbose=verbose,
                **optimizer_kwargs,
            )
        else:  # LBFGS
            return fit_with_lbfgs(
                initial_model=initial_model,
                family=family,
                design_matrices=design_matrices,
                y=y,
                weights=w,
                fixed_parameters=fixed_parameter_values,
                max_iter=max_iter,
                history_size=history_size,
                learning_rate=learning_rate,
                verbose=verbose,
                **optimizer_kwargs,
            )

    # Traditional RS/CG/MIXED methods

    # RS_JAX: fully JAX-traced RS loop (GPU/TPU ready, no smooth terms)
    if method_name == "RS_JAX":
        from .algorithms.jax_rs_integration import gamlss_rs_jax

        return gamlss_rs_jax(
            formula=formula,
            family=family,
            data=data,
            sigma_formula=sigma_formula,
            parameter_formulas=parameter_formulas,
            weights=weights,
            control=control,
            i_control=i_control,
            verbose=verbose,
            routing_decision=routing_decision,
        )

    # For RS method, use the dedicated rs_fit function which implements the correct algorithm
    if method_name == "RS":
        from .algorithms.rs_algorithm import rs_fit

        return rs_fit(
            formula=formula,
            family=family,
            data=data,
            sigma_formula=sigma_formula,
            parameter_formulas=parameter_formulas,
            weights=weights,
            max_iter=control.n_cyc if control is not None else 20,
            tol=control.c_crit if control is not None else 1e-4,
            verbose=verbose,
            routing_decision=routing_decision,
        )

    if method_name == "CG":
        cg_backend_name = str(
            optimizer_kwargs.pop("cg_backend", "full_hessian")
        ).upper()
        if cg_backend_name in {
            "IRLS",
            "IRLS_CROSS",
            "CG_IRLS_CROSS",
            "ETA",
            "ETA_CROSS",
        }:
            from .algorithms.cg_algorithm_v2 import cg_fit_v2

            return cg_fit_v2(
                formula=formula,
                sigma_formula=sigma_formula,
                parameter_formulas=(
                    dict(parameter_formulas) if parameter_formulas is not None else None
                ),
                family=family,
                data=dict(data),
                weights=weights,
                max_iter=control.n_cyc if control is not None else 20,
                tol=control.c_crit if control is not None else 1e-4,
                verbose=verbose,
            )
        if cg_backend_name not in {
            "FULL_HESSIAN",
            "CG_FULL_HESSIAN",
            "FULL",
            "HESSIAN",
        }:
            raise ValueError(
                "cg_backend must be one of 'full_hessian' or 'irls_cross' when method='CG', "
                f"got {cg_backend_name!r}"
            )

    # CG and MIXED methods use the inline implementation below
    rqres_callable = _build_rqres_callable(family)

    control = gamlss_control() if control is None else control
    i_control = glim_control() if i_control is None else i_control

    # Initialize performance monitoring if verbose
    if verbose:
        from .performance import PerformanceMonitor

        monitor = PerformanceMonitor(
            use_jit=False,
            n_observations=len(data[list(data.keys())[0]]),
            family_name=family.name if hasattr(family, "name") else str(family),
        )
        monitor.start()
        print(f"\n{'='*70}")
        print(f"Fitting {monitor.family_name} model")
        print(f"{'='*70}")
        print(f"Observations: {monitor.n_observations:,}")
        print(f"Method: {method_name}")
        print(f"Max iterations: {control.n_cyc}")
        print(f"Convergence criterion: {control.c_crit}")
    else:
        monitor = None

    response_name, _ = _parse_formula(formula)
    resolved_formulas = _resolve_parameter_formulas(
        response=response_name,
        family=family,
        mu_formula=formula,
        sigma_formula=sigma_formula,
        parameter_formulas=parameter_formulas,
    )

    # Build design matrix with smooth support
    _, mu_x, predictor_labels, mu_smooth_info = _build_design_matrix_with_smooths(
        resolved_formulas["mu"], data, weights=weights
    )

    y = np.asarray(data[response_name], dtype=np.float64)
    n = len(y)
    fixed_parameter_values = _resolve_fixed_parameter_values(family, data, n)
    parameter_designs: dict[str, np.ndarray] = {}
    parameter_labels: dict[str, list[str]] = {}
    smooth_infos: dict[str, Any] = {"mu": mu_smooth_info}

    for parameter in family.estimable_parameters:
        if parameter == "mu":
            continue
        # Build design matrix with smooth support
        (
            _,
            parameter_designs[parameter],
            parameter_labels[parameter],
            param_smooth_info,
        ) = _build_design_matrix_with_smooths(
            resolved_formulas[parameter],
            data,
            weights=weights,
        )
        smooth_infos[parameter] = param_smooth_info
    if weights is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)

    initial = None
    if method_name != "CG":
        initial = gamlss_ml(
            formula=formula,
            sigma_formula=sigma_formula,
            parameter_formulas=parameter_formulas,
            family=family,
            data=data,
            weights=weights,
            control=control,
            i_control=i_control,
        )
        mu = np.asarray(initial.fitted_values["mu"], dtype=np.float64)
        eta_mu = np.asarray(initial.linear_predictors["mu"], dtype=np.float64)
        beta_mu = np.asarray(initial.coefficients["mu"], dtype=np.float64)
        parameter_values = {
            parameter: np.asarray(initial.fitted_values[parameter], dtype=np.float64)
            for parameter in family.parameters
            if parameter != "mu"
        }
        parameter_eta = {
            parameter: np.asarray(
                initial.linear_predictors[parameter], dtype=np.float64
            )
            for parameter in family.estimable_parameters
            if parameter != "mu"
        }
        parameter_beta = {
            parameter: np.asarray(initial.coefficients[parameter], dtype=np.float64)
            for parameter in family.estimable_parameters
            if parameter != "mu"
        }
        extra_parameter_values = {
            parameter: np.asarray(initial.fitted_values[parameter], dtype=np.float64)
            for parameter in family.parameters
            if parameter not in {"mu", "sigma"}
        }
        g_dev = float(initial.g_dev)
        deviance_history = [g_dev]
    else:
        beta_mu = np.zeros(mu_x.shape[1], dtype=np.float64)
        eta_mu = mu_x @ beta_mu
        mu = np.asarray(family.link_inverses["mu"](eta_mu), dtype=np.float64)
        parameter_values = {}
        parameter_eta = {}
        parameter_beta = {}
        extra_parameter_values = {}
        g_dev = float("nan")
        deviance_history = []
    old_g_dev = g_dev + 1.0 if np.isfinite(g_dev) else float("inf")
    iteration = 0  # Start from 0, not control.iter
    working_vectors_extra: dict[str, np.ndarray] = {}
    iterative_weights_extra: dict[str, np.ndarray] = {}

    # Initialize z_mu and w_mu in case loop doesn't execute
    z_mu = eta_mu.copy()
    w_mu = np.ones_like(eta_mu, dtype=np.float64)

    # Print initial status if verbose
    if verbose:
        print(f"\n{'='*70}")
        print("Starting iterative fitting")
        print(f"{'='*70}")
        if np.isfinite(g_dev):
            print(f"Initial deviance: {g_dev:.4f}")
        else:
            print("Initial deviance: computed inside CG backend")
        print("\nIteration progress:")

    if method_name == "CG":
        from .fitting_cg import fit_cg

        cg_start_params = None
        cg_designs: dict[str, Any] = {"X_sigma": None, "X_nu": None, "X_tau": None}
        for parameter in family.estimable_parameters:
            if parameter == "mu":
                continue
            design = parameter_designs.get(parameter)
            if design is not None:
                cg_designs[f"X_{parameter}"] = jnp.asarray(design, dtype=jnp.float64)

        cg_result = fit_cg(
            family=family,
            y=jnp.asarray(y, dtype=jnp.float64),
            X_mu=jnp.asarray(mu_x, dtype=jnp.float64),
            X_sigma=cg_designs["X_sigma"],
            X_nu=cg_designs["X_nu"],
            X_tau=cg_designs["X_tau"],
            weights=jnp.asarray(w, dtype=jnp.float64),
            start_params=cg_start_params,
            max_iter=control.n_cyc,
            tol=control.c_crit,
            verbose=verbose,
            return_fisher=True,
        )

        beta_mu = np.asarray(cg_result.params["beta_mu"], dtype=np.float64)
        eta_mu = mu_x @ beta_mu
        mu = np.asarray(cg_result.fitted_values["mu"], dtype=np.float64)
        z_mu = eta_mu.copy()
        w_mu = np.ones_like(eta_mu, dtype=np.float64)

        for parameter in family.estimable_parameters:
            if parameter == "mu" or f"beta_{parameter}" not in cg_result.params:
                continue
            beta_param = np.asarray(
                cg_result.params[f"beta_{parameter}"], dtype=np.float64
            )
            x_param = parameter_designs.get(parameter)
            if x_param is None:
                continue
            eta_param = x_param @ beta_param
            value_param = np.asarray(
                cg_result.fitted_values[parameter], dtype=np.float64
            )
            parameter_beta[parameter] = beta_param
            parameter_eta[parameter] = eta_param
            parameter_values[parameter] = value_param
            if parameter not in {"mu", "sigma"}:
                extra_parameter_values[parameter] = value_param
            working_vectors_extra[parameter] = eta_param.copy()
            iterative_weights_extra[parameter] = np.ones_like(
                eta_param, dtype=np.float64
            )

        g_dev = float(cg_result.final_deviance)
        old_g_dev = deviance_history[-1] if deviance_history else g_dev
        iteration = int(cg_result.n_iter)
        deviance_history = [float(value) for value in cg_result.deviance_history]
        converged = bool(cg_result.converged)

    while (
        method_name != "CG"
        and abs(old_g_dev - g_dev) > control.c_crit
        and iteration < control.n_cyc
    ):
        if verbose:
            monitor.start_iteration()
        old_g_dev = g_dev

        mu_kwargs: dict[str, Any] = {"y": y, "mu": mu, **extra_parameter_values}
        if "sigma" in parameter_values:
            mu_kwargs["sigma"] = parameter_values["sigma"]
        dldm = np.asarray(family.score_functions["mu"](**mu_kwargs), dtype=np.float64)
        d2ldm2 = np.asarray(
            family.hessian_functions["mu"](**mu_kwargs), dtype=np.float64
        )

        # Enhanced numerical stability for mixed distributions (zero-inflated/altered)
        # These distributions can have zero or near-zero hessians at y=0
        eps = np.finfo(np.float64).eps

        # Replace non-finite values
        dldm = np.where(np.isfinite(dldm), dldm, 0.0)
        d2ldm2 = np.where(np.isfinite(d2ldm2), d2ldm2, -1e-10)

        # Add floor to prevent division by zero in IWLS
        # For mixed distributions, use a more conservative floor
        is_mixed = getattr(family, "type_", None) == "Mixed"
        if is_mixed:
            d2ldm2 = np.where(np.abs(d2ldm2) < 1e-8, -1e-8, d2ldm2)
        else:
            d2ldm2 = np.where(np.abs(d2ldm2) < 1e-10, -1e-10, d2ldm2)

        # Compute link derivative with safety checks
        dr_mu = np.asarray(family.link_derivatives["mu"](eta_mu), dtype=np.float64)
        dr_mu = np.where(np.isfinite(dr_mu), dr_mu, eps)
        dr_mu = np.where(np.abs(dr_mu) < eps, eps, dr_mu)

        # Compute deta/dmu with safety
        deta_dmu = 1.0 / dr_mu
        deta_dmu = np.where(np.isfinite(deta_dmu), deta_dmu, 1.0)

        # Compute weights with enhanced safety for mixed distributions
        deta_dmu_sq = np.square(deta_dmu)
        deta_dmu_sq = np.where(deta_dmu_sq < eps, eps, deta_dmu_sq)

        w_mu = -(d2ldm2 / deta_dmu_sq)
        w_mu = np.where(np.isfinite(w_mu), w_mu, 1e-10)

        # More conservative clipping for mixed distributions
        if is_mixed:
            w_mu = np.clip(w_mu, 1e-8, 1e8)
        else:
            w_mu = np.clip(w_mu, 1e-10, 1e10)
            w_mu = np.clip(w_mu, 1e-10, 1e10)

        # Compute working response with safety
        denominator = deta_dmu * w_mu
        denominator = np.where(np.abs(denominator) < eps, eps, denominator)
        z_mu = eta_mu + dldm / denominator
        z_mu = np.where(np.isfinite(z_mu), z_mu, eta_mu)

        beta_mu_proposed = _weighted_least_squares(
            mu_x, z_mu, w_mu * w, smooth_info=mu_smooth_info
        )
        beta_mu = _apply_method_step(beta_mu, beta_mu_proposed, method_name)
        eta_mu = mu_x @ beta_mu
        mu = np.asarray(family.link_inverses["mu"](eta_mu), dtype=np.float64)

        if "sigma" in family.parameters and "sigma" in parameter_values:
            sigma = parameter_values["sigma"]
            sigma_x = parameter_designs.get("sigma")
            eta_sigma = parameter_eta.get("sigma")
            beta_sigma = parameter_beta.get("sigma")
            if (
                sigma is not None
                and sigma_x is not None
                and eta_sigma is not None
                and beta_sigma is not None
                and family.name == "NBI"
            ):
                eps = np.finfo(np.float64).eps
                mean_component = np.maximum(mu, eps)
                sigma_moment = np.maximum(
                    (np.square(y - mu) - mean_component)
                    / np.maximum(np.square(mean_component), eps),
                    eps,
                )
                sigma_value = np.maximum(np.sum(w * sigma_moment) / np.sum(w), eps)
                beta_sigma_proposed = np.array([np.log(sigma_value)], dtype=np.float64)
                beta_sigma = _apply_method_step(
                    beta_sigma, beta_sigma_proposed, method_name
                )
                eta_sigma = sigma_x @ beta_sigma
                parameter_values["sigma"] = np.asarray(
                    family.link_inverses["sigma"](eta_sigma), dtype=np.float64
                )
                parameter_eta["sigma"] = eta_sigma
                parameter_beta["sigma"] = beta_sigma
                working_vectors_extra["sigma"] = eta_sigma.copy()
                iterative_weights_extra["sigma"] = np.ones_like(
                    eta_sigma, dtype=np.float64
                )
            elif (
                sigma is not None
                and sigma_x is not None
                and eta_sigma is not None
                and beta_sigma is not None
            ):
                sigma_kwargs: dict[str, Any] = {
                    "y": y,
                    "mu": mu,
                    "sigma": sigma,
                    **extra_parameter_values,
                }
                dldsigma = np.asarray(
                    family.score_functions["sigma"](**sigma_kwargs), dtype=np.float64
                )
                d2ldsigma2 = np.asarray(
                    family.hessian_functions["sigma"](**sigma_kwargs), dtype=np.float64
                )
                # For mixed distributions, hessian can be zero at y=0
                d2ldsigma2 = np.where(np.abs(d2ldsigma2) < 1e-10, -1e-10, d2ldsigma2)
                dr_sigma = np.asarray(
                    family.link_derivatives["sigma"](eta_sigma), dtype=np.float64
                )
                dr_sigma = np.where(
                    np.isfinite(dr_sigma), dr_sigma, np.finfo(np.float64).eps
                )
                dr_sigma = np.where(
                    np.abs(dr_sigma) < np.finfo(np.float64).eps,
                    np.finfo(np.float64).eps,
                    dr_sigma,
                )
                deta_dsigma = 1.0 / dr_sigma
                w_sigma = -(d2ldsigma2 / np.square(deta_dsigma))
                w_sigma = np.clip(w_sigma, 1e-10, 1e10)
                z_sigma = eta_sigma + dldsigma / (deta_dsigma * w_sigma)
                z_sigma = np.where(np.isfinite(z_sigma), z_sigma, eta_sigma)
                beta_sigma_proposed = _weighted_least_squares(
                    sigma_x, z_sigma, w_sigma * w, smooth_info=smooth_infos.get("sigma")
                )
                beta_sigma = _apply_method_step(
                    beta_sigma, beta_sigma_proposed, method_name
                )
                eta_sigma = sigma_x @ beta_sigma
                sigma = np.asarray(
                    family.link_inverses["sigma"](eta_sigma), dtype=np.float64
                )
                sigma = np.where(np.isfinite(sigma), sigma, np.finfo(np.float64).eps)
                sigma = np.maximum(sigma, np.finfo(np.float64).eps)
                parameter_values["sigma"] = sigma
                parameter_eta["sigma"] = eta_sigma
                parameter_beta["sigma"] = beta_sigma
                working_vectors_extra["sigma"] = z_sigma
                iterative_weights_extra["sigma"] = w_sigma

        for parameter in family.estimable_parameters:
            if parameter in {"mu", "sigma"}:
                continue
            x_param = parameter_designs.get(parameter)
            eta_param = parameter_eta.get(parameter)
            beta_param = parameter_beta.get(parameter)
            value_param = parameter_values.get(parameter)
            if (
                x_param is None
                or eta_param is None
                or beta_param is None
                or value_param is None
            ):
                continue
            parameter_kwargs: dict[str, Any] = {
                "y": y,
                "mu": mu,
                **extra_parameter_values,
            }
            if "sigma" in parameter_values:
                parameter_kwargs["sigma"] = parameter_values["sigma"]
            parameter_kwargs[parameter] = value_param
            score = np.asarray(
                family.score_functions[parameter](**parameter_kwargs), dtype=np.float64
            )
            hessian = np.asarray(
                family.hessian_functions[parameter](**parameter_kwargs),
                dtype=np.float64,
            )
            # For mixed distributions, hessian can be zero at y=0
            hessian = np.where(np.abs(hessian) < 1e-10, -1e-10, hessian)
            dr_param = np.asarray(
                family.link_derivatives[parameter](eta_param), dtype=np.float64
            )
            dr_param = np.where(
                np.isfinite(dr_param), dr_param, np.finfo(np.float64).eps
            )
            dr_param = np.where(
                np.abs(dr_param) < np.finfo(np.float64).eps,
                np.finfo(np.float64).eps,
                dr_param,
            )
            deta_dparam = 1.0 / dr_param
            w_param = -(hessian / np.square(deta_dparam))
            w_param = np.clip(w_param, 1e-10, 1e10)
            z_param = eta_param + score / (deta_dparam * w_param)
            z_param = np.where(np.isfinite(z_param), z_param, eta_param)
            beta_proposed = _weighted_least_squares(
                x_param, z_param, w_param * w, smooth_info=smooth_infos.get(parameter)
            )
            beta_param = _apply_method_step(beta_param, beta_proposed, method_name)
            eta_param = x_param @ beta_param
            value_param = np.asarray(
                family.link_inverses[parameter](
                    jnp.asarray(eta_param, dtype=jnp.float64)
                ),
                dtype=np.float64,
            )
            parameter_values[parameter] = value_param
            parameter_eta[parameter] = eta_param
            parameter_beta[parameter] = beta_param
            extra_parameter_values[parameter] = value_param
            working_vectors_extra[parameter] = z_param
            iterative_weights_extra[parameter] = w_param

        dev_kwargs: dict[str, Any] = {
            "y": y,
            "mu": mu,
            **extra_parameter_values,
            **fixed_parameter_values,
        }
        if "sigma" in parameter_values:
            dev_kwargs["sigma"] = parameter_values["sigma"]
        g_dev = float(np.sum(np.asarray(family.g_dev_inc(**dev_kwargs)) * w))
        iteration += 1
        deviance_history.append(g_dev)

        # Print iteration progress if verbose
        if verbose:
            monitor.finish_iteration()
            monitor.print_iteration(iteration, g_dev, converged=False)

    if method_name != "CG":
        converged = abs(old_g_dev - g_dev) <= control.c_crit

    # Print final status if verbose
    if verbose:
        monitor.finish()
        print(f"\n{'='*70}")
        if converged:
            print(f"✓ Converged after {iteration} iterations")
        else:
            print(f"⚠ Did not converge after {iteration} iterations")
        print(f"Final deviance: {g_dev:.4f}")
        print(f"{'='*70}")
        monitor.print_summary(verbose=True)

    fitted_values = {"mu": jnp.asarray(mu, dtype=jnp.float64)}
    coefficients = {"mu": jnp.asarray(beta_mu, dtype=jnp.float64)}
    linear_predictors = {"mu": jnp.asarray(eta_mu, dtype=jnp.float64)}
    working_vectors = {"mu": jnp.asarray(z_mu, dtype=jnp.float64)}
    iterative_weights = {"mu": jnp.asarray(w_mu, dtype=jnp.float64)}
    offsets = {"mu": jnp.zeros(n, dtype=jnp.float64)}
    formulas = {"mu": resolved_formulas["mu"]}
    term_map = {
        "mu": {
            "term_labels": predictor_labels,
            "response": response_name,
            "intercept": True,
            "formula": resolved_formulas["mu"],
        }
    }
    design_matrices = {"mu": jnp.asarray(mu_x, dtype=jnp.float64)}

    if "sigma" in family.parameters and "sigma" in parameter_values:
        sigma = parameter_values["sigma"]
        beta_sigma = parameter_beta["sigma"]
        eta_sigma = parameter_eta["sigma"]
        fitted_values["sigma"] = jnp.asarray(sigma, dtype=jnp.float64)
        coefficients["sigma"] = jnp.asarray(beta_sigma, dtype=jnp.float64)
        linear_predictors["sigma"] = jnp.asarray(eta_sigma, dtype=jnp.float64)
        if "sigma" in working_vectors_extra and "sigma" in iterative_weights_extra:
            working_vectors["sigma"] = jnp.asarray(
                working_vectors_extra["sigma"], dtype=jnp.float64
            )
            iterative_weights["sigma"] = jnp.asarray(
                iterative_weights_extra["sigma"], dtype=jnp.float64
            )
        offsets["sigma"] = jnp.zeros(n, dtype=jnp.float64)
        formulas["sigma"] = resolved_formulas["sigma"]
        term_map["sigma"] = {
            "term_labels": parameter_labels.get("sigma", []),
            "response": response_name,
            "intercept": True,
            "formula": resolved_formulas["sigma"],
        }
        design_matrices["sigma"] = jnp.asarray(
            parameter_designs["sigma"], dtype=jnp.float64
        )

    for parameter in family.estimable_parameters:
        if parameter in {"mu", "sigma"}:
            continue
        fitted_values[parameter] = jnp.asarray(
            parameter_values[parameter], dtype=jnp.float64
        )
        coefficients[parameter] = jnp.asarray(
            parameter_beta[parameter], dtype=jnp.float64
        )
        linear_predictors[parameter] = jnp.asarray(
            parameter_eta[parameter], dtype=jnp.float64
        )
        if parameter in working_vectors_extra and parameter in iterative_weights_extra:
            working_vectors[parameter] = jnp.asarray(
                working_vectors_extra[parameter], dtype=jnp.float64
            )
            iterative_weights[parameter] = jnp.asarray(
                iterative_weights_extra[parameter], dtype=jnp.float64
            )
        offsets[parameter] = jnp.zeros(n, dtype=jnp.float64)
        formulas[parameter] = resolved_formulas[parameter]
        term_labels = parameter_labels.get(parameter, [])
        term_map[parameter] = {
            "term_labels": term_labels,
            "response": response_name,
            "intercept": True,
            "formula": resolved_formulas[parameter],
        }
        if parameter in parameter_designs:
            design_matrices[parameter] = jnp.asarray(
                parameter_designs[parameter], dtype=jnp.float64
            )

    for parameter, value in fixed_parameter_values.items():
        fitted_values[parameter] = jnp.asarray(value, dtype=jnp.float64)
        # Fixed parameters don't have link functions - they are used as-is
        linear_predictors[parameter] = jnp.asarray(value, dtype=jnp.float64)
        offsets[parameter] = jnp.zeros(n, dtype=jnp.float64)
        formulas[parameter] = _fixed_parameter_formula(parameter)
        term_map[parameter] = _fixed_parameter_term(response_name, parameter)

    method_slots: dict[str, Any] = {}
    if method_name == "CG":
        method_slots = {
            "cg_converged": bool(converged),
            "cg_iterations": int(iteration),
            "cg_final_deviance": float(g_dev),
            "cg_backend": getattr(cg_result, "cg_backend", "CG_FULL_HESSIAN"),
            "cg_cross_derivatives": getattr(
                cg_result, "cross_derivatives", "full_hessian"
            ),
            "cg_line_search_steps": tuple(getattr(cg_result, "line_search_steps", ())),
            "cg_condition_number": getattr(cg_result, "condition_number", None),
            "cg_param_slices": getattr(cg_result, "param_slices", None),
        }
    elif method_name == "MIXED":
        method_slots = {
            "mixed_converged": bool(converged),
            "mixed_iterations": int(iteration),
            "mixed_final_deviance": float(g_dev),
        }

    residual_values = (
        rqres_callable(y=y, mu=mu, sigma=parameter_values.get("sigma"))
        if rqres_callable is not None
        else _compute_residuals(family, y, mu, parameter_values.get("sigma"))
    )

    # Compute df_fit including smooth terms
    df_fit = 0.0
    smooth_edf = {}

    for parameter in family.estimable_parameters:
        # Add linear parameters
        df_fit += float(np.asarray(coefficients[parameter], dtype=np.float64).size)

        # Adjust for smooth terms
        if (
            smooth_infos.get(parameter) is not None
            and len(smooth_infos[parameter].smooth_fits) > 0
        ):
            from .smooth_fitting import compute_smooth_edf

            # Subtract nominal basis columns
            for smooth in smooth_infos[parameter].smooth_fits:
                start, end = smooth.basis_columns
                df_fit -= float(end - start)
            # Add effective df
            param_edf = compute_smooth_edf(
                design_matrices[parameter], w, smooth_infos[parameter].smooth_fits
            )
            df_fit += param_edf
            smooth_edf[parameter] = param_edf
        else:
            smooth_edf[parameter] = 0.0
    for parameter in family.fixed_parameters or ():
        smooth_edf[parameter] = 0.0

    return GAMLSSModel(
        par=tuple(family.parameters),
        family=family,
        df_fit=df_fit,
        g_dev=g_dev,
        n=n,
        y=jnp.asarray(y, dtype=jnp.float64),
        fitted_values=fitted_values,
        coefficients=coefficients,
        linear_predictors=linear_predictors,
        working_vectors=working_vectors,
        iterative_weights=iterative_weights,
        offsets=offsets,
        formulas=formulas,
        terms=term_map,
        design_matrices=design_matrices,
        additional_slots={
            "G.deviance": g_dev,
            "P.deviance": g_dev,
            "noObs": int(n),
            "df.residual": float(n - df_fit),
            "aic": float(g_dev + df_fit * 2.0),
            "sbc": float(g_dev + df_fit * np.log(max(n, 1))),
            "method": method_name,
            "converged": bool(converged),
            "cycles": int(iteration),
            "deviance_history": tuple(float(value) for value in deviance_history),
            "smooth_fits": {
                p: smooth_infos[p].smooth_fits if smooth_infos.get(p) else []
                for p in family.parameters
            },
            "smooth_edf": smooth_edf,
            "method_routing": routing_decision,
            **method_slots,
        },
        call={
            "data": data,
            "formula": resolved_formulas["mu"],
            "parameter_formulas": dict(resolved_formulas),
            "method": method_name,
        },
        control={
            "n.cyc": control.n_cyc,
            **asdict(control),
            **{"glim.cyc": i_control.cyc},
        },
        iter=iteration,
        weights=jnp.asarray(w, dtype=jnp.float64),
        residuals=residual_values,
        type=family.type,
        parameters=tuple(family.parameters),
        rqres=rqres_callable,
    )
