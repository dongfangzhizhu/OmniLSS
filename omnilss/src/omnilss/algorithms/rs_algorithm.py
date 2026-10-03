"""Rigby-Stasinopoulos (RS) Algorithm for GAMLSS.

This module implements the RS algorithm, which is the original and default
fitting algorithm for GAMLSS models. The algorithm alternates between
updating each distribution parameter (μ, σ, ν, τ) until convergence.

Reference:
    Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized additive models
    for location, scale and shape. Journal of the Royal Statistical Society:
    Series C (Applied Statistics), 54(3), 507-554.

The RS algorithm is based on the GLIM (Generalized Linear Interactive Modeling)
iterative procedure, extended to handle multiple distribution parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
import warnings
from typing import Any, Dict, Optional, Tuple

import jax.numpy as jnp
import jax
import numpy as np

from ..distributions import resolve_family
from ..families import FamilyDefinition
from ..model import GAMLSSModel
from ..numerical_stability import sanitize_gradient, step_halving
from ..tensor_protocol import validate_design_matrix, validate_vector
from ..diagnostic_warnings import evaluate_numerical_warnings
from ..family_validation import ensure_valid_likelihood_inputs
from .stabilized_hessian import stabilize_hessian
from ..fast_wls import solve_weighted_least_squares_cholesky
from ._model_metrics import df_fit_with_smooth_edf
from .backend_protocol import IWLSState, SolverBackend
from .warm_start_policy import FamilyHeuristicInit, WarmStartPolicy


class ConvergenceWarning(UserWarning):
    """Convergence warning for RS updates."""


_LINK_ETA_BOUNDS: dict[str, tuple[float, float]] = {
    "log": (-15.0, 15.0),
    "logit": (-15.0, 15.0),
    "identity": (-1e8, 1e8),
    "sqrt": (0.0, 1e4),
    "inverse": (1e-6, 1e6),
    "log_log": (-10.0, 10.0),
    "clog_log": (-10.0, 10.0),
    "probit": (-6.0, 6.0),
}


def _get_eta_bounds(link_name: str) -> tuple[float, float]:
    return _LINK_ETA_BOUNDS.get(link_name, (-1e8, 1e8))


class NumpySolverBackend(SolverBackend):
    """Default NumPy backend for RS/IWLS operations."""

    name = "numpy"

    def __init__(self, family: FamilyDefinition) -> None:
        self._family = family

    def compute_score_hessian(
        self,
        *,
        parameter: str,
        y: np.ndarray,
        fitted: np.ndarray,
        context: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        score_fn = self._family.score_functions[parameter]
        hess_fn = self._family.hessian_functions[parameter]
        param_dict = {"y": y, parameter: fitted}
        param_dict.update(context)
        score = np.asarray(
            sanitize_gradient(jnp.asarray(score_fn(**param_dict), dtype=jnp.float64)),
            dtype=np.float64,
        )
        hessian = np.asarray(hess_fn(**param_dict), dtype=np.float64)
        return score, hessian

    def iwls_step(
        self,
        *,
        state: IWLSState,
        score: np.ndarray,
        hessian: np.ndarray,
        link_derivative: np.ndarray,
        offset: np.ndarray,
        safeguard_events: list[dict[str, object]] | None = None,
        parameter: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        iter_w, z = compute_working_weights_and_response(
            y=state.fitted,
            fitted_values=state.fitted,
            link_derivative=link_derivative,
            first_derivative=score,
            second_derivative=hessian,
            offset=offset,
            eta=state.eta,
            safeguard_events=safeguard_events,
            parameter=parameter,
        )
        return z, iter_w

    def wls_solve(
        self,
        *,
        design: np.ndarray,
        working_response: np.ndarray,
        weights: np.ndarray,
        ridge: float | None = None,
        skip_stabilize: bool = False,
    ) -> np.ndarray:
        # Phase 7.3: the caller (rs_step) may already have run
        # stabilize_hessian on the Gram for condition-number diagnostics; in
        # that case skip the duplicate Gram + stabilization here.
        if not skip_stabilize:
            sqrt_w = np.sqrt(weights)
            weighted_x = design * sqrt_w[:, None]
            gram = weighted_x.T @ weighted_x
            if ridge is None:
                stabilized = stabilize_hessian(gram)
                # Only apply the ridge when stabilization actually kicked in
                # (lambda_value > 0); otherwise keep the pure lstsq path.
                ridge = (
                    stabilized.lambda_value
                    if stabilized.lambda_value > 0.0
                    else None
                )
        return solve_weighted_least_squares_cholesky(
            X=design,
            z=working_response,
            w=weights,
            ridge=ridge,
        )


class JaxSolverBackend(SolverBackend):
    """JAX-backed implementation for RS/IWLS operations."""

    name = "jax"

    def __init__(self, family: FamilyDefinition) -> None:
        self._family = family

    def compute_score_hessian(
        self,
        *,
        parameter: str,
        y: np.ndarray,
        fitted: np.ndarray,
        context: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        score_fn = self._family.score_functions[parameter]
        hess_fn = self._family.hessian_functions[parameter]
        param_dict = {"y": y, parameter: fitted}
        param_dict.update(context)
        score = np.asarray(
            sanitize_gradient(jnp.asarray(score_fn(**param_dict), dtype=jnp.float64)),
            dtype=np.float64,
        )
        hessian = np.asarray(hess_fn(**param_dict), dtype=np.float64)
        return score, hessian

    def iwls_step(
        self,
        *,
        state: IWLSState,
        score: np.ndarray,
        hessian: np.ndarray,
        link_derivative: np.ndarray,
        offset: np.ndarray,
        safeguard_events: list[dict[str, object]] | None = None,
        parameter: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        iter_w, z = compute_working_weights_and_response(
            y=state.fitted,
            fitted_values=state.fitted,
            link_derivative=link_derivative,
            first_derivative=score,
            second_derivative=hessian,
            offset=offset,
            eta=state.eta,
            safeguard_events=safeguard_events,
            parameter=parameter,
        )
        return z, iter_w

    def wls_solve(
        self,
        *,
        design: np.ndarray,
        working_response: np.ndarray,
        weights: np.ndarray,
        ridge: float | None = None,
        skip_stabilize: bool = False,
    ) -> np.ndarray:
        sqrt_w = jnp.sqrt(jnp.asarray(weights, dtype=jnp.float64))
        xw = jnp.asarray(design, dtype=jnp.float64) * sqrt_w[:, None]
        zw = jnp.asarray(working_response, dtype=jnp.float64) * sqrt_w
        if not skip_stabilize and ridge is None:
            # Mirror NumpySolverBackend: derive the stabilization ridge from
            # the weighted Gram unless the caller already stabilized it
            # (skip_stabilize=True) or supplied an explicit ridge.
            gram = np.asarray(xw.T @ xw, dtype=np.float64)
            stabilized = stabilize_hessian(gram)
            if stabilized.lambda_value > 0.0:
                ridge = float(stabilized.lambda_value)
        if ridge is not None:
            # Levenberg-style damping for the normal equations, matching the
            # NumPy backend's stabilized ridge.
            p = xw.shape[1]
            xw = jnp.concatenate([xw, jnp.sqrt(jnp.asarray(ridge, dtype=jnp.float64)) * jnp.eye(p)], axis=0)
            zw = jnp.concatenate([zw, jnp.zeros(p, dtype=jnp.float64)])
        beta, _, _, _ = jnp.linalg.lstsq(xw, zw, rcond=None)
        return np.asarray(beta, dtype=np.float64)


def _select_solver_backend(
    family: FamilyDefinition,
    n: int,
    routing_decision: Optional[Dict[str, Any]],
) -> SolverBackend:
    if isinstance(routing_decision, dict) and routing_decision.get("solver_backend"):
        backend_name = str(routing_decision["solver_backend"]).lower()
    else:
        # Phase 7.2: align the automatic backend choice with config.py's
        # device policy instead of a hard-coded 50k threshold — CPU (and
        # placeholder-infinite thresholds) always stay on NumPy RS; only
        # GPU/TPU with n above the configured crossover use the JAX backend.
        from .. import config as _cfg

        backend, _devices = _cfg._current_backend()
        if backend in ("gpu", "tpu"):
            table = _cfg.GPU_CROSSOVER_N if backend == "gpu" else _cfg.TPU_CROSSOVER_N
            threshold = _cfg._get_crossover(table, family.name)
            backend_name = "jax" if n >= threshold else "numpy"
        else:
            backend_name = "numpy"
    if backend_name == "jax":
        return JaxSolverBackend(family)
    return NumpySolverBackend(family)


@dataclass
class RSStepResult:
    """Result from a single RS algorithm step.

    Attributes
    ----------
    fitted_values : np.ndarray
        Updated fitted values for the parameter
    linear_predictor : np.ndarray
        Updated linear predictor (eta)
    coefficients : np.ndarray
        Updated regression coefficients
    working_weights : np.ndarray
        Working weights from the iteration
    working_response : np.ndarray
        Working response (adjusted dependent variable)
    deviance : float
        Deviance after this step
    converged : bool
        Whether the inner iteration converged
    iterations : int
        Number of inner iterations
    """

    fitted_values: np.ndarray
    linear_predictor: np.ndarray
    coefficients: np.ndarray
    working_weights: np.ndarray
    working_response: np.ndarray
    deviance: float
    converged: bool
    iterations: int
    step_halving_count: int
    last_condition_number: float
    last_gradient_norm: float


def compute_working_weights_and_response(
    y: np.ndarray,
    fitted_values: np.ndarray,
    link_derivative: np.ndarray,
    first_derivative: np.ndarray,
    second_derivative: np.ndarray,
    offset: np.ndarray,
    eta: np.ndarray,
    safeguard_events: list[dict[str, object]] | None = None,
    parameter: str | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute working weights and working response for IRLS.

    This implements the GLIM-style iterative reweighted least squares (IRLS)
    procedure for a single parameter.

    Parameters
    ----------
    y : np.ndarray
        Response variable
    fitted_values : np.ndarray
        Current fitted values for this parameter
    link_derivative : np.ndarray
        Derivative of link function (dη/dμ)
    first_derivative : np.ndarray
        First derivative of log-likelihood w.r.t. parameter
    second_derivative : np.ndarray
        Second derivative of log-likelihood w.r.t. parameter
    offset : np.ndarray
        Offset for this parameter
    eta : np.ndarray
        Current linear predictor

    Returns
    -------
    working_weights : np.ndarray
        Working weights for weighted least squares
    working_response : np.ndarray
        Working response (adjusted dependent variable)

    Notes
    -----
    The working weights are computed as:
        w = -(d²l/dθ²) * (dμ/dη)²

    The working response is computed as:
        z = (η - offset) + (dl/dθ) / (dη/dμ * w)

    This follows the GLIM algorithm as described in Rigby & Stasinopoulos (2005).
    """
    # Ensure second derivative is negative (for concave log-likelihood)
    floor_applied = np.sum(second_derivative >= -1e-15)
    second_derivative = np.where(second_derivative < -1e-15, second_derivative, -1e-15)
    if safeguard_events is not None and floor_applied > 0:
        safeguard_events.append({
            "type": "hessian_floor",
            "parameter": parameter,
            "n_floor": int(floor_applied),
        })

    # Compute working weights
    # w = -(d²l/dθ²) / (dη/dμ)²
    working_weights = -(second_derivative / (link_derivative**2))

    # Clip weights to avoid numerical issues
    working_weights = np.clip(working_weights, 1e-10, 1e10)

    # Compute working response
    # z = (η - offset) + (dl/dθ) / (dη/dμ * w)
    working_response = (eta - offset) + first_derivative / (
        link_derivative * working_weights
    )

    # Handle NaN values (can occur with mixed distributions)
    n_nan = int(np.sum(np.isnan(working_response)))
    working_response = np.where(np.isnan(working_response), 0.0, working_response)
    if safeguard_events is not None and n_nan > 0:
        safeguard_events.append({
            "type": "nan_replacement",
            "parameter": parameter,
            "n_replaced": n_nan,
        })

    # R's glim.fit does NOT clip the working response; a hard ±1e6 clip
    # corrupted IRLS for zero-inflated data (e.g. y=0 in Poisson made the
    # response explode and the clip then dominated the WLS solve).  Only
    # guard against non-finite values; extreme-but-finite responses are the
    # algorithm's signal, and step-halving / convergence checks handle them.
    if np.any(~np.isfinite(working_response)):
        warnings.warn(
            "non-finite working_response detected; replaced with 0.0. "
            "Model may not have converged. Consider checking data scale.",
            ConvergenceWarning,
            stacklevel=4,
        )
    working_response = np.where(np.isfinite(working_response), working_response, 0.0)

    return working_weights, working_response


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    """Compact numeric summary used in smoother update diagnostics (Phase 3.4)."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"min": float("nan"), "mean": float("nan"), "max": float("nan"), "n_nonfinite": 0}
    finite = arr[np.isfinite(arr)]
    return {
        "min": float(np.min(finite)) if finite.size else float("nan"),
        "mean": float(np.mean(finite)) if finite.size else float("nan"),
        "max": float(np.max(finite)) if finite.size else float("nan"),
        "n_nonfinite": int(arr.size - finite.size),
    }


def rs_step(
    y: np.ndarray,
    X: np.ndarray,
    fitted_values: np.ndarray,
    weights: np.ndarray,
    family: FamilyDefinition,
    parameter: str,
    other_parameters: Optional[Dict[str, np.ndarray]] = None,
    offset: Optional[np.ndarray] = None,
    max_iter: int = 1,
    tol: float = 1e-4,
    step_size: float = 1.0,
    auto_step: bool = True,
    verbose: bool = False,
    smooth_info=None,  # 新增：SmoothDesignInfo 或 None
    solver_backend: SolverBackend | None = None,
    safeguard_events: list[dict[str, object]] | None = None,
) -> RSStepResult:
    """Perform one RS algorithm step for a single parameter.

    This implements the inner loop of the RS algorithm, which updates
    one distribution parameter (μ, σ, ν, or τ) using iterative reweighted
    least squares (IRLS).

    Parameters
    ----------
    y : np.ndarray
        Response variable
    X : np.ndarray
        Design matrix for this parameter
    fitted_values : np.ndarray
        Current fitted values for this parameter
    weights : np.ndarray
        Observation weights
    family : FamilyDefinition
        Distribution family
    parameter : str
        Parameter name ("mu", "sigma", "nu", or "tau")
    other_parameters : dict, optional
        Dictionary of other parameter values (e.g., {"sigma": sigma_values})
    offset : np.ndarray, optional
        Offset for this parameter
    max_iter : int, default=1
        Maximum inner IRLS iterations. Set to 1 (matching R's gamlss::glim.fit)
        to prevent oscillation near warm-start values.
    tol : float, default=1e-4
        Convergence tolerance for deviance
    step_size : float, default=1.0
        Step size for parameter updates (0 < step_size <= 1)
    auto_step : bool, default=True
        Whether to automatically reduce step size if deviance increases
    verbose : bool, default=False
        Whether to print iteration details

    Returns
    -------
    result : RSStepResult
        Result containing updated values and convergence information

    Notes
    -----
    The RS algorithm uses a GLIM-style iterative procedure:

    1. Compute working weights and working response
    2. Fit weighted least squares
    3. Update fitted values
    4. Check convergence
    5. Repeat until convergence or max iterations

    The algorithm includes automatic step-halving if the deviance increases,
    which helps with convergence in difficult cases.

    Examples
    --------
    >>> result = rs_step(
    ...     y=y,
    ...     X=X_mu,
    ...     fitted_values=mu,
    ...     weights=w,
    ...     family=family,
    ...     parameter="mu",
    ...     other_parameters={"sigma": sigma}
    ... )
    >>> mu_new = result.fitted_values
    """
    if offset is None:
        offset = np.zeros_like(y)

    if other_parameters is None:
        other_parameters = {}

    # For families that require other parameters (like sigma for Normal),
    # provide default values if not specified
    if parameter == "mu" and "sigma" not in other_parameters:
        # Estimate sigma from residuals if not provided
        if "sigma" in family.parameters:
            residuals = y - fitted_values
            sigma_est = np.std(residuals) if np.std(residuals) > 0 else 1.0
            # Ensure sigma is not too small to avoid numerical issues
            sigma_est = max(sigma_est, 1e-6)
            other_parameters = {**other_parameters, "sigma": np.full_like(y, sigma_est)}
    elif parameter == "sigma" and "mu" not in other_parameters:
        # Use fitted_values as mu if not provided
        if "mu" in family.parameters:
            other_parameters = {**other_parameters, "mu": fitted_values.copy()}

    # Week 4 validation: enforce parameter-domain safety before updates
    _validation_inputs = {parameter: fitted_values}
    _validation_inputs.update(other_parameters)
    ensure_valid_likelihood_inputs(family, _validation_inputs)

    # Get link functions for this parameter
    link_fun = family.link_functions[parameter]
    link_inv = family.link_inverses[parameter]

    backend = solver_backend or NumpySolverBackend(family)

    # Initialize
    eta = np.asarray(
        link_fun(jnp.asarray(fitted_values, dtype=jnp.float64)), dtype=np.float64
    )
    old_deviance = np.inf
    converged = False
    step_halving_count = 0
    last_condition_number = float("nan")
    last_gradient_norm = float("nan")

    # Compute INITIAL derivatives (before loop, following R's glim.fit structure)
    first_deriv, second_deriv = backend.compute_score_hessian(
        parameter=parameter,
        y=y,
        fitted=fitted_values,
        context=other_parameters,
    )

    # Compute link derivative (dη/dμ) using analytical formula
    link_deriv_func = family.link_derivatives[parameter]
    # dμ/dη
    dmu_deta = np.asarray(
        link_deriv_func(jnp.asarray(eta, dtype=jnp.float64)), dtype=np.float64
    )
    # dη/dμ = 1 / (dμ/dη)
    link_deriv = 1.0 / (dmu_deta + 1e-15)

    # Compute INITIAL working weights and response
    working_response, working_weights = backend.iwls_step(
        state=IWLSState(
            design=X,
            eta=eta,
            fitted=fitted_values,
            observation_weights=weights,
        ),
        score=first_deriv,
        hessian=second_deriv,
        link_derivative=link_deriv,
        offset=offset,
        safeguard_events=safeguard_events,
        parameter=parameter,
    )

    for iteration in range(max_iter):
        # Check for invalid values
        if np.any(np.isnan(working_weights)) or np.any(np.isnan(working_response)):
            raise ValueError(
                f"NaN values in working weights or response for parameter {parameter}"
            )
        if np.any(~np.isfinite(working_weights)) or np.any(
            ~np.isfinite(working_response)
        ):
            raise ValueError(
                f"Inf values in working weights or response for parameter {parameter}"
            )

        # Fit weighted least squares with CURRENT working weights and response
        # (which were computed from the PREVIOUS iteration's fitted values)
        W = working_weights * weights
        if (
            smooth_info is not None
            and hasattr(smooth_info, "smooth_fits")
            and smooth_info.smooth_fits
        ):
            # ── 带惩罚的加权最小二乘（P-IRLS）──
            # 从 SmoothDesignInfo 提取每个平滑项的惩罚信息
            from ..smooth_fitting import penalized_wls_no_jit

            penalties = [
                (
                    sf.basis_columns[0],
                    sf.basis_columns[1],
                    jnp.asarray(sf.penalty, dtype=jnp.float64),
                    float(sf.lambda_),
                )
                for sf in smooth_info.smooth_fits
            ]
            coef = np.asarray(
                penalized_wls_no_jit(
                    jnp.asarray(X, dtype=jnp.float64),
                    jnp.asarray(working_response, dtype=jnp.float64),
                    jnp.asarray(W, dtype=jnp.float64),
                    penalties,
                ),
                dtype=np.float64,
            )
        else:
            # ── 标准加权最小二乘（无平滑项）──
            sqrt_W = np.sqrt(W)
            WX_np = X * sqrt_W[:, None]
            gram = WX_np.T @ WX_np
            stabilized = stabilize_hessian(gram)
            last_condition_number = stabilized.condition_number
            if (safeguard_events is not None) and stabilized.lambda_value > 0.0:
                safeguard_events.append({
                    "type": "hessian_ridge",
                    "parameter": parameter,
                    "iteration": iteration,
                    "lambda": float(stabilized.lambda_value),
                })
            coef = backend.wls_solve(
                design=X,
                working_response=working_response,
                weights=W,
                ridge=(
                    stabilized.lambda_value
                    if stabilized.lambda_value > 0.0
                    else None
                ),
                # Phase 7.3: Gram/stabilization already computed above.
                skip_stabilize=True,
            )

        # Update linear predictor with damped step size
        eta_old = eta.copy()
        eta_new = X @ coef + offset
        eta_candidate = step_size * eta_new + (1 - step_size) * eta_old
        link_name = family.links.get(parameter, "identity")
        eta_lo, eta_hi = _get_eta_bounds(link_name)
        eta_clipped = np.clip(eta_candidate, eta_lo, eta_hi)
        if np.any(eta_clipped != eta_candidate):
            n_clipped = int(np.sum(eta_clipped != eta_candidate))
            if safeguard_events is not None:
                safeguard_events.append({
                    "type": "eta_clip",
                    "parameter": parameter,
                    "iteration": iteration,
                    "n_clipped": n_clipped,
                    "total": int(len(eta_candidate)),
                })
            warnings.warn(
                f"eta clipped for parameter '{parameter}' (link='{link_name}'): {n_clipped}/{len(eta_candidate)} values out of [{eta_lo}, {eta_hi}]",
                ConvergenceWarning,
                stacklevel=3,
            )
        eta = eta_clipped

        # Update fitted values
        fitted_values = np.asarray(
            link_inv(jnp.asarray(eta, dtype=jnp.float64)), dtype=np.float64
        )

        # Compute deviance with all parameters
        dev_param_dict = {"y": y, parameter: fitted_values}
        dev_param_dict.update(other_parameters)
        dev_incr = np.asarray(family.g_dev_inc(**dev_param_dict), dtype=np.float64)
        deviance = np.sum(weights * dev_incr)

        # Auto step-halving if deviance increases
        if auto_step and deviance > old_deviance and iteration >= 1:
            for _ in range(5):
                eta = np.asarray(
                    step_halving(jnp.asarray(eta_old), jnp.asarray(eta), factor=0.5),
                    dtype=np.float64,
                )
                step_halving_count += 1
                fitted_values = np.asarray(
                    link_inv(jnp.asarray(eta, dtype=jnp.float64)), dtype=np.float64
                )
                dev_param_dict = {"y": y, parameter: fitted_values}
                dev_param_dict.update(other_parameters)
                dev_incr = np.asarray(
                    family.g_dev_inc(**dev_param_dict), dtype=np.float64
                )
                deviance = np.sum(weights * dev_incr)
                if deviance <= old_deviance:
                    break

        if verbose:
            print(f"  Inner iteration {iteration + 1}: deviance = {deviance:.6f}")

        # Check convergence
        if abs(old_deviance - deviance) < tol:
            converged = True
            break

        old_deviance = deviance

        # Recompute derivatives at NEW fitted values for NEXT iteration
        # This follows R's glim.fit structure (lines 262-271)
        first_deriv, second_deriv = backend.compute_score_hessian(
            parameter=parameter,
            y=y,
            fitted=fitted_values,
            context=other_parameters,
        )
        last_gradient_norm = float(np.linalg.norm(first_deriv))

        # Recompute link derivative
        dmu_deta = np.asarray(
            link_deriv_func(jnp.asarray(eta, dtype=jnp.float64)), dtype=np.float64
        )
        link_deriv = 1.0 / (dmu_deta + 1e-15)

        # Recompute working weights and response for NEXT iteration
        working_response, working_weights = backend.iwls_step(
            state=IWLSState(
                design=X,
                eta=eta,
                fitted=fitted_values,
                observation_weights=weights,
            ),
            score=first_deriv,
            hessian=second_deriv,
            link_derivative=link_deriv,
            offset=offset,
            safeguard_events=safeguard_events,
            parameter=parameter,
        )

    return RSStepResult(
        fitted_values=fitted_values,
        linear_predictor=eta,
        coefficients=coef,
        working_weights=working_weights,
        working_response=working_response,
        deviance=deviance,
        converged=converged,
        iterations=iteration + 1,
        step_halving_count=step_halving_count,
        last_condition_number=last_condition_number,
        last_gradient_norm=last_gradient_norm,
    )


def _has_converged(dev_old: float, dev_new: float, tol: float, rtol: float) -> bool:
    """Relative + absolute convergence criterion."""
    abs_change = abs(dev_old - dev_new)
    rel_change = abs_change / (abs(dev_old) + 1e-10)
    return abs_change < tol or rel_change < rtol


def rs_fit(
    formula: str,
    family: Any,
    data: Dict[str, Any],
    sigma_formula: str = "~1",
    parameter_formulas: Optional[Dict[str, str]] = None,
    weights: Optional[np.ndarray] = None,
    max_iter: int = 20,
    tol: float = 1e-4,
    rtol: float = 1e-6,
    mu_step: float = 1.0,
    sigma_step: float = 1.0,
    nu_step: float = 1.0,
    tau_step: float = 1.0,
    auto_step: bool = True,
    verbose: bool = False,
    raise_on_lambda_failure: bool = False,
    routing_decision: Optional[Dict[str, Any]] = None,
    warm_start_policy: WarmStartPolicy | None = None,
) -> GAMLSSModel:
    """Fit GAMLSS model using RS (Rigby-Stasinopoulos) algorithm.

    This is the main fitting function that implements the complete RS algorithm,
    which alternates between updating each distribution parameter until the
    global deviance converges.

    Parameters
    ----------
    formula : str
        Formula for μ parameter
    family : Any
        Distribution family (name or FamilyDefinition)
    data : dict
        Data dictionary
    sigma_formula : str, default="~1"
        Formula for σ parameter
    parameter_formulas : dict, optional
        Formulas for ν and τ parameters
    weights : np.ndarray, optional
        Observation weights
    max_iter : int, default=20
        Maximum number of outer iterations
    tol : float, default=1e-4
        Convergence tolerance for global deviance
    mu_step : float, default=1.0
        Step size for μ updates
    sigma_step : float, default=1.0
        Step size for σ updates
    nu_step : float, default=1.0
        Step size for ν updates
    tau_step : float, default=1.0
        Step size for τ updates
    auto_step : bool, default=True
        Whether to automatically reduce step size if deviance increases
    verbose : bool, default=False
        Whether to print iteration details

    Returns
    -------
    model : GAMLSSModel
        Fitted GAMLSS model

    Notes
    -----
    The RS algorithm proceeds as follows:

    **Outer Loop** (until convergence):
        1. Update μ (if not fixed)
        2. Update σ (if not fixed)
        3. Update ν (if not fixed)
        4. Update τ (if not fixed)
        5. Compute global deviance
        6. Check convergence

    **Inner Loop** (for each parameter):
        - Use IRLS to update the parameter
        - Iterate until parameter-specific convergence

    The algorithm is robust and typically converges in 5-10 outer iterations.

    Examples
    --------
    >>> from omnilss.algorithms import rs_fit
    >>>
    >>> # Basic usage
    >>> model = rs_fit(
    ...     formula="y ~ x1 + x2",
    ...     family="NO",
    ...     data=data
    ... )
    >>>
    >>> # With smoothing
    >>> model = rs_fit(
    ...     formula="y ~ pb(x1) + x2",
    ...     sigma_formula="~ pb(x3)",
    ...     family="NO",
    ...     data=data,
    ...     verbose=True
    ... )
    >>>
    >>> # Four-parameter distribution
    >>> model = rs_fit(
    ...     formula="y ~ x1",
    ...     sigma_formula="~ x2",
    ...     parameter_formulas={"nu": "~ x3", "tau": "~ x4"},
    ...     family="BCT",
    ...     data=data
    ... )

    References
    ----------
    Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized additive models
    for location, scale and shape. Journal of the Royal Statistical Society:
    Series C (Applied Statistics), 54(3), 507-554.
    """
    from ..distributions import resolve_family
    from ..fitting import (
        _build_design_matrix_with_smooths,
        _fixed_parameter_formula,
        _fixed_parameter_term,
        _initial_mu_beta,
        _initial_parameter_value,
        _initial_sigma,
        _is_intercept_only_formula,
        _parse_formula,
        _resolve_fixed_parameter_values,
        _resolve_parameter_formulas,
        _weighted_least_squares,
    )

    # Resolve family
    family = resolve_family(family)

    if verbose:
        print("=" * 70)
        print("RS Algorithm - Full Implementation")
        print("=" * 70)
        print(f"Family: {family.name}")
        print(f"Parameters: {family.parameters}")
        print(f"Max iterations: {max_iter}")
        print(f"Tolerance: {tol}")
        print("RS inner-loop policy: max_inner=1 (gamlss::glim.fit aligned)")
        print("=" * 70)

    # Parse formulas
    response_name, _ = _parse_formula(formula)
    resolved_formulas = _resolve_parameter_formulas(
        response=response_name,
        family=family,
        mu_formula=formula,
        sigma_formula=sigma_formula,
        parameter_formulas=parameter_formulas,
    )

    # Get response variable
    y = np.asarray(data[response_name], dtype=np.float64)
    n = len(y)

    # Get weights
    if weights is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)

    # Get fixed parameter values
    fixed_parameter_values = _resolve_fixed_parameter_values(family, data, n)

    # Build design matrices for all parameters
    design_matrices = {}
    smooth_infos = {}
    predictor_labels = {}

    for parameter in family.parameters:
        if parameter in fixed_parameter_values:
            continue

        param_formula = resolved_formulas[parameter]
        _, param_X, param_labels, param_smooth_info = _build_design_matrix_with_smooths(
            param_formula, data, weights=weights
        )

        design_matrices[parameter] = param_X
        smooth_infos[parameter] = param_smooth_info
        predictor_labels[parameter] = param_labels

    timing_split = {"warm_start": 0.0, "core_rs": 0.0, "host_device_sync": 0.0}
    warm_start_t0 = time.perf_counter()
    warm_start = warm_start_policy or FamilyHeuristicInit()
    warm_start_kwargs = {
        "y": y,
        "parameter_names": tuple(family.estimable_parameters),
    }
    if isinstance(warm_start, FamilyHeuristicInit):
        warm_start_kwargs["family_name"] = family.name
    warm_start_result = warm_start.initialize(**warm_start_kwargs)
    solver_backend = _select_solver_backend(family, n, routing_decision)

    # Initialize parameter values
    parameter_values = {}
    coefficients = {}
    linear_predictors = {}

    # Initialize mu
    if "mu" in design_matrices:
        mu_X = design_matrices["mu"]
        mu_smooth_info = smooth_infos["mu"]
        warm_mu = warm_start_result.initial_fitted.get("mu")

        if mu_smooth_info is not None:
            beta_mu = _weighted_least_squares(mu_X, y, w, smooth_info=mu_smooth_info)
        else:
            if warm_mu is not None:
                warm_mu = np.asarray(warm_mu, dtype=np.float64)
                if warm_mu.shape == (n,):
                    eta_warm_mu = np.asarray(
                        family.link_functions["mu"](
                            jnp.asarray(warm_mu, dtype=jnp.float64)
                        ),
                        dtype=np.float64,
                    )
                    beta_mu, _, _, _ = np.linalg.lstsq(mu_X, eta_warm_mu, rcond=None)
                else:
                    beta_mu = _initial_mu_beta(
                        family, mu_X, y, w, fixed_parameter_values=fixed_parameter_values
                    )
            else:
                beta_mu = _initial_mu_beta(
                    family, mu_X, y, w, fixed_parameter_values=fixed_parameter_values
                )

        mu_linear = mu_X @ beta_mu
        mu = np.asarray(
            family.link_inverses["mu"](jnp.asarray(mu_linear, dtype=jnp.float64)),
            dtype=np.float64,
        )

        parameter_values["mu"] = mu
        coefficients["mu"] = beta_mu
        linear_predictors["mu"] = mu_linear
    else:
        # Fixed mu
        parameter_values["mu"] = fixed_parameter_values["mu"]
        linear_predictors["mu"] = np.asarray(
            family.link_functions["mu"](
                jnp.asarray(parameter_values["mu"], dtype=jnp.float64)
            ),
            dtype=np.float64,
        )

    # Initialize other parameters
    for parameter in family.parameters:
        if parameter == "mu" or parameter in fixed_parameter_values:
            continue

        param_X = design_matrices[parameter]
        param_smooth_info = smooth_infos[parameter]
        param_formula = resolved_formulas[parameter]

        # Get initial value
        if parameter == "sigma":
            initial_value = _initial_sigma(
                family,
                y,
                parameter_values["mu"],
                w,
                fixed_parameter_values=fixed_parameter_values,
            )
            # Ensure sigma is not too small to avoid numerical issues
            initial_value = max(initial_value, 1e-6)
        else:
            initial_value = _initial_parameter_value(
                family, parameter, y, parameter_values["mu"], w
            )
        # NOTE (Phase 2.1): family-aware initialisation (_initial_sigma /
        # _initial_parameter_value) is the single source of truth for non-mu
        # parameters.  The generic warm-start heuristic (mu=y_mean, sigma=y_sd,
        # others=1.0) has no family context and its sigma/nu values can be far
        # off the R initialisation (e.g. ZAGA: nu=1.0 -> logit(1.0)~36, and
        # sigma=y_sd vs R's sigma=1), which corrupted the first RS step.
        # warm_start_result is intentionally not consulted here; see NOTE above.
        warm_param = None

        # Initialize linear predictor
        eta_init = np.full(
            n,
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

        # Fit initial coefficients
        if param_smooth_info is not None:
            beta_param = _weighted_least_squares(
                param_X, eta_init, w, smooth_info=param_smooth_info
            )
        else:
            beta_param, _, _, _ = np.linalg.lstsq(param_X, eta_init, rcond=None)

        eta_param = param_X @ beta_param
        param_value = np.asarray(
            family.link_inverses[parameter](jnp.asarray(eta_param, dtype=jnp.float64)),
            dtype=np.float64,
        )

        # Ensure sigma is not too small to avoid numerical issues
        if parameter == "sigma":
            param_value = np.maximum(param_value, 1e-6)

        parameter_values[parameter] = param_value
        coefficients[parameter] = beta_param
        linear_predictors[parameter] = eta_param

    # Add fixed parameters (e.g. BI's bd denominator).
    # Fixed parameters are data, not model parameters — they have no link
    # function and are stored directly without transformation.
    for parameter, value in fixed_parameter_values.items():
        parameter_values[parameter] = value
        # Only store a linear predictor if the family actually defines a link
        # for this parameter (most fixed params like bd do not).
        if parameter in family.link_functions:
            linear_predictors[parameter] = np.asarray(
                family.link_functions[parameter](jnp.asarray(value, dtype=jnp.float64)),
                dtype=np.float64,
            )
        else:
            linear_predictors[parameter] = np.asarray(value, dtype=np.float64)

    # Compute initial global deviance
    dev_kwargs = {"y": y, **parameter_values}
    g_dev_incr = np.asarray(family.g_dev_inc(**dev_kwargs), dtype=np.float64)
    g_dev = np.sum(w * g_dev_incr)
    g_dev_old = g_dev + 1
    timing_split["warm_start"] = float(time.perf_counter() - warm_start_t0)

    # 记录每个参数最新的 IRLS 工作向量（用于λ更新）
    last_working_weights: dict = {}
    last_working_response: dict = {}

    if verbose:
        print(f"Initial Global Deviance: {g_dev:.6f}")
        print("=" * 70)

    # Step sizes for each parameter
    step_sizes = {"mu": mu_step, "sigma": sigma_step, "nu": nu_step, "tau": tau_step}

    # Phase 2.3: structured safeguard diagnostics collected during fitting
    # (eta_clip / hessian_ridge / hessian_floor / nan_replacement / step_halving).
    safeguard_events: list[dict[str, object]] = []

    # Phase 3.4: smoother lambda-update diagnostics (design.md Property 9) —
    # old/new lambda, EDF, working-response/weight summary, solver status.
    smooth_update_diagnostics: list[dict[str, object]] = []

    # Outer loop: iterate until convergence
    iteration = 0
    converged = False
    deviance_history = [float(g_dev)]
    lambda_update_warnings: list[str] = []
    lambda_update_failed_params: set[str] = set()
    rs_step_halving_by_param: dict[str, int] = {
        p: 0 for p in family.estimable_parameters
    }
    rs_last_condition_number_by_param: dict[str, float] = {
        p: float("nan") for p in family.estimable_parameters
    }
    rs_last_gradient_norm_by_param: dict[str, float] = {
        p: float("nan") for p in family.estimable_parameters
    }

    core_rs_t0 = time.perf_counter()
    while not _has_converged(g_dev_old, g_dev, tol, rtol) and iteration < max_iter:
        iteration += 1
        g_dev_old = g_dev

        if verbose:
            print(f"\nOuter Iteration {iteration}:")

        # Update each parameter in sequence
        for parameter in family.estimable_parameters:
            if parameter in fixed_parameter_values:
                continue

            # Get current values for other parameters
            other_params = {
                p: parameter_values[p] for p in family.parameters if p != parameter
            }

            # Perform RS step for this parameter
            param_smooth_info = smooth_infos.get(parameter)
            result = rs_step(
                y=y,
                X=design_matrices[parameter],
                fitted_values=parameter_values[parameter],
                weights=w,
                family=family,
                parameter=parameter,
                other_parameters=other_params,
                offset=None,
                max_iter=1,
                tol=1e-4,
                step_size=step_sizes.get(parameter, 1.0),
                auto_step=auto_step,
                verbose=False,
                smooth_info=param_smooth_info,  # 新增
                solver_backend=solver_backend,
                safeguard_events=safeguard_events,
            )

            # 记录 IRLS 工作向量，用于λ更新
            last_working_weights[parameter] = result.working_weights
            last_working_response[parameter] = result.working_response

            rs_step_halving_by_param[parameter] = rs_step_halving_by_param.get(
                parameter, 0
            ) + int(result.step_halving_count)
            rs_last_condition_number_by_param[parameter] = float(
                result.last_condition_number
            )
            rs_last_gradient_norm_by_param[parameter] = float(result.last_gradient_norm)

            # Update parameter values
            parameter_values[parameter] = result.fitted_values
            coefficients[parameter] = result.coefficients
            linear_predictors[parameter] = result.linear_predictor

            # Ensure sigma is not too small to avoid numerical issues
            if parameter == "sigma":
                parameter_values[parameter] = np.maximum(
                    parameter_values[parameter], 1e-6
                )

            if verbose:
                print(
                    f"  {parameter}: converged={result.converged}, "
                    f"iterations={result.iterations}, deviance={result.deviance:.6f}"
                )

        # Compute global deviance
        dev_kwargs = {"y": y, **parameter_values}
        g_dev_incr = np.asarray(family.g_dev_inc(**dev_kwargs), dtype=np.float64)
        g_dev = np.sum(w * g_dev_incr)
        deviance_history.append(float(g_dev))

        if verbose:
            print(
                f"  Global Deviance: {g_dev:.6f}, Change: {abs(g_dev_old - g_dev):.6e}"
            )

        # ── 每轮外循环后更新平滑参数 λ（Back-fitting 外层）──
        if iteration >= 1:  # 第一轮先建立合理的系数估计
            from ..smooth_fitting import SmoothDesignInfo, update_smooth_lambdas

            for param_k in family.parameters:
                if param_k in fixed_parameter_values:
                    continue
                smooth_design = smooth_infos.get(param_k)
                if (
                    smooth_design is None
                    or not hasattr(smooth_design, "smooth_fits")
                    or not smooth_design.smooth_fits
                ):
                    continue

                X_k = design_matrices[param_k]
                beta_k = np.asarray(coefficients.get(param_k, np.zeros(X_k.shape[1])))
                # 用 IRLS 工作响应（z）和工作权重（w）作为 λ 更新的输入
                w_k = last_working_weights.get(param_k, w)
                z_k = last_working_response.get(param_k, np.zeros(n))

                try:
                    old_lambdas = [
                        float(sf.lambda_) for sf in smooth_design.smooth_fits
                    ]
                    updated_fits = update_smooth_lambdas(
                        X=X_k,
                        y=z_k,  # IRLS 工作响应
                        beta=beta_k,
                        w=w_k,  # IRLS 工作权重
                        smooth_fits=smooth_design.smooth_fits,
                        method="GCV",  # 默认用 GCV
                    )
                    # Phase 3.4: record per-term smoother update diagnostics.
                    for idx, (old_sf, new_sf) in enumerate(
                        zip(smooth_design.smooth_fits, updated_fits)
                    ):
                        smooth_update_diagnostics.append(
                            {
                                "parameter": param_k,
                                "iteration": iteration,
                                "smoother": new_sf.smoother,
                                "old_lambda": old_sf.lambda_,
                                "new_lambda": new_sf.lambda_,
                                "edf": new_sf.edf,
                                "lambda_origin": new_sf.lambda_origin,
                                "working_response": _summary_stats(z_k),
                                "weights": _summary_stats(w_k),
                                "solver_status": (
                                    "preserved"
                                    if new_sf.lambda_origin in {"user_fixed", "df_target"}
                                    else "reestimated"
                                ),
                            }
                        )
                    # 更新 smooth_infos 以便下轮使用新 λ
                    smooth_infos[param_k] = SmoothDesignInfo(
                        X=smooth_design.X,
                        smooth_fits=updated_fits,
                        linear_columns=smooth_design.linear_columns,
                        has_intercept=smooth_design.has_intercept,
                    )
                    if verbose:
                        new_lambdas = [float(sf.lambda_) for sf in updated_fits]
                        print(
                            f"  lambda update ({param_k}): {old_lambdas} -> {new_lambdas}"
                        )
                except Exception as e:
                    msg = (
                        f"Lambda update failed for parameter '{param_k}': {e}. "
                        "Keeping previous lambda value."
                    )
                    lambda_update_failed_params.add(param_k)
                    lambda_update_warnings.append(msg)
                    for sf in smooth_design.smooth_fits:
                        smooth_update_diagnostics.append(
                            {
                                "parameter": param_k,
                                "iteration": iteration,
                                "smoother": sf.smoother,
                                "old_lambda": sf.lambda_,
                                "new_lambda": sf.lambda_,
                                "edf": sf.edf,
                                "lambda_origin": sf.lambda_origin,
                                "working_response": _summary_stats(z_k),
                                "weights": _summary_stats(w_k),
                                "solver_status": "failed",
                            }
                        )
                    if raise_on_lambda_failure:
                        raise ConvergenceWarning(msg) from e
                    warnings.warn(msg, UserWarning, stacklevel=2)

        # Check for increasing deviance
        if g_dev > g_dev_old + 1e-6 and iteration > 1:
            if verbose:
                print(f"  WARNING: Global deviance increased!")

    # Check convergence
    if _has_converged(g_dev_old, g_dev, tol, rtol):
        converged = True
    # Classify the outcome so callers can tell "converged" from "diverged"
    # without re-deriving it from the deviance history.  A non-finite deviance
    # is fail-closed: it can never be reported as a converged fit even if the
    # change criterion happened to pass.
    if not np.isfinite(g_dev):
        converged = False
        termination_reason = "non_finite_deviance"
    elif converged:
        termination_reason = "relative_deviance_converged"
    else:
        termination_reason = "max_iterations_reached"
    timing_split["core_rs"] = float(time.perf_counter() - core_rs_t0)
    if solver_backend.name == "jax":
        sync_t0 = time.perf_counter()
        try:
            for _param in family.estimable_parameters:
                jax.block_until_ready(jnp.asarray(parameter_values[_param]))
        except Exception:
            pass
        timing_split["host_device_sync"] = float(time.perf_counter() - sync_t0)

    if verbose:
        print("=" * 70)
        if converged:
            print(f"Converged in {iteration} iterations")
        else:
            print(f"Did not converge in {max_iter} iterations")
        print(f"Final Global Deviance: {g_dev:.6f}")
        print("=" * 70)

    # Build GAMLSSModel object
    fitted_values_jax = {
        p: jnp.asarray(parameter_values[p], dtype=jnp.float64)
        for p in family.parameters
    }
    coefficients_jax = {
        p: jnp.asarray(
            coefficients.get(p, np.array([parameter_values[p][0]])), dtype=jnp.float64
        )
        for p in family.estimable_parameters
    }
    linear_predictors_jax = {
        p: jnp.asarray(linear_predictors[p], dtype=jnp.float64)
        for p in family.parameters
    }
    design_matrices_jax = {
        p: jnp.asarray(design_matrices.get(p, np.zeros((n, 1))), dtype=jnp.float64)
        for p in family.estimable_parameters
    }

    # Build terms dictionary
    terms = {}
    for parameter in family.parameters:
        if parameter in fixed_parameter_values:
            terms[parameter] = _fixed_parameter_term(response_name, parameter)
        else:
            terms[parameter] = {
                "term_labels": predictor_labels.get(parameter, []),
                "response": response_name,
                "intercept": True,
                "formula": resolved_formulas[parameter],
            }

    # Compute residuals for diagnostics
    from ..fitting import (
        _build_rqres_callable,
        _compute_residuals,
    )

    rqres_callable = _build_rqres_callable(family)
    mu_vals = parameter_values["mu"]
    sigma_vals = parameter_values.get("sigma", None)
    if rqres_callable is not None:
        residual_values = rqres_callable(
            y=y,
            mu=mu_vals,
            sigma=sigma_vals,
        )
    else:
        residual_values = _compute_residuals(family, y, mu_vals, sigma_vals)

    # Build working vectors / iterative weights for residuals compatibility
    # (simple approximation: use raw residuals and unit weights)
    working_vectors_jax: dict = {}
    iterative_weights_jax: dict = {}
    offsets_jax: dict = {}
    for parameter in family.parameters:
        working_vectors_jax[parameter] = jnp.asarray(
            linear_predictors[parameter], dtype=jnp.float64
        )
        iterative_weights_jax[parameter] = jnp.ones(n, dtype=jnp.float64)
        offsets_jax[parameter] = jnp.zeros(n, dtype=jnp.float64)

    # Compute df_fit using effective degrees of freedom for smooth terms.
    # Counting all smooth basis coefficients as ordinary parameters overstates
    # model complexity after penalization and corrupts AIC/SBC.
    df_fit_val, smooth_edf = df_fit_with_smooth_edf(
        coefficients=coefficients,
        estimable_parameters=family.estimable_parameters,
        design_matrices=design_matrices,
        weights=w,
        smooth_infos=smooth_infos,
    )

    _cond_vals = np.asarray(
        list(rs_last_condition_number_by_param.values()), dtype=np.float64
    )
    _grad_vals = np.asarray(
        list(rs_last_gradient_norm_by_param.values()), dtype=np.float64
    )

    _phase1_slots_seed = {
        "gradient_norm": (
            float(np.nanmax(_grad_vals))
            if (_grad_vals.size and np.isfinite(_grad_vals).any())
            else float("nan")
        ),
        "condition_number": (
            float(np.nanmax(_cond_vals))
            if (_cond_vals.size and np.isfinite(_cond_vals).any())
            else float("nan")
        ),
        "step_size_by_param": dict(step_sizes),
        "lambda_update_failed_params": tuple(sorted(lambda_update_failed_params)),
    }
    _warning_events = evaluate_numerical_warnings(_phase1_slots_seed)
    _delta_deviance = float("nan")
    if len(deviance_history) >= 2:
        _delta_deviance = float(deviance_history[-1] - deviance_history[-2])
    _beta_l2_delta = float("nan")
    if family.estimable_parameters:
        _norms = []
        for _p in family.estimable_parameters:
            _coef = np.asarray(coefficients.get(_p, np.array([])), dtype=np.float64)
            if _coef.size:
                _norms.append(float(np.linalg.norm(_coef)))
        if _norms:
            _beta_l2_delta = float(max(_norms))
    _max_abs_score = _phase1_slots_seed["gradient_norm"]

    model = GAMLSSModel(
        par=family.parameters,
        family=family,
        df_fit=df_fit_val,
        g_dev=g_dev,
        n=n,
        y=jnp.asarray(y, dtype=jnp.float64),
        fitted_values=fitted_values_jax,
        coefficients=coefficients_jax,
        linear_predictors=linear_predictors_jax,
        working_vectors=working_vectors_jax,
        iterative_weights=iterative_weights_jax,
        offsets=offsets_jax,
        formulas=resolved_formulas,
        terms=terms,
        design_matrices=design_matrices_jax,
        weights=jnp.asarray(w, dtype=jnp.float64),
        residuals=jnp.asarray(residual_values, dtype=jnp.float64),
        rqres=rqres_callable,
        iter=iteration,
        converged=converged,
        termination_reason=termination_reason,
        type=family.type,
        parameters=family.parameters,
        call={
            "data": data,
            "formula": resolved_formulas["mu"],
            "parameter_formulas": dict(resolved_formulas),
            "method": "RS",
        },
        control={"n.cyc": max_iter},
        additional_slots={
            "method": "RS",
            # Store smooth-term metadata (knots/degree, etc.) for prediction-time basis reconstruction.
            "smooth_infos": smooth_infos,
            "smooth_edf": smooth_edf,
            "rs_iterations": iteration,
            "rs_converged": converged,
            "step_sizes": step_sizes,
            "auto_step": auto_step,
            "rs_policy_max_inner": 1,
            "method_routing": routing_decision,
            "noObs": int(n),
            "G.deviance": g_dev,
            "P.deviance": g_dev,
            "aic": g_dev + 2.0 * df_fit_val,
            "sbc": g_dev + np.log(max(n, 1)) * df_fit_val,
            "df.residual": float(n - df_fit_val),
            "df_residual": float(n - df_fit_val),
            "converged": converged,
            "cycles": int(iteration),
            "deviance_history": tuple(float(v) for v in deviance_history),
            "lambda_update_warnings": tuple(lambda_update_warnings),
            "lambda_update_failed_params": tuple(sorted(lambda_update_failed_params)),
            "rs_step_halving_by_param": dict(rs_step_halving_by_param),
            "rs_last_condition_number_by_param": dict(
                rs_last_condition_number_by_param
            ),
            "rs_last_gradient_norm_by_param": dict(rs_last_gradient_norm_by_param),
            "step_size_by_param": dict(step_sizes),
            "condition_number": _phase1_slots_seed["condition_number"],
            "gradient_norm": _phase1_slots_seed["gradient_norm"],
            "phase1_warning_events": tuple(
                (e.code, e.level, e.message) for e in _warning_events
            ),
            "delta_deviance": _delta_deviance,
            "beta_l2_delta": _beta_l2_delta,
            "max_abs_score": _max_abs_score,
            "timing_split": timing_split,
            "warm_start_policy": warm_start.name,
            "warm_start_metadata": dict(warm_start_result.metadata),
            "solver_backend": solver_backend.name,
            "safeguard_events": tuple(safeguard_events),
            "smooth_lambda_diagnostics": tuple(smooth_update_diagnostics),
        },
    )

    return model
