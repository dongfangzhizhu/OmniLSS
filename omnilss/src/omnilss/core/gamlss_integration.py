"""Integration of modern optimizers with GAMLSS fitting.

This module provides the bridge between the new Joint/L-BFGS optimizers
and the traditional GAMLSS fitting framework. It allows users to choose
between traditional RS/CG algorithms and modern gradient-based optimizers.

Examples:
    Using Joint Optimizer with Adam:
    
    >>> from omnilss import gamlss
    >>> model = gamlss(
    ...     "y ~ x1 + x2",
    ...     family="NO",
    ...     data=data,
    ...     method="joint",
    ...     optimizer="adam",
    ...     learning_rate=0.01
    ... )
    
    Using L-BFGS:
    
    >>> model = gamlss(
    ...     "y ~ x1 + x2",
    ...     family="NO",
    ...     data=data,
    ...     method="lbfgs",
    ...     max_iter=100
    ... )
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .optimizer import JointOptimizer, OptimizationResult
from .lbfgs_optimizer import LBFGSOptimizer
from ..distributions import resolve_family
from ..families import FamilyDefinition
from ..model import GAMLSSModel


def _build_loss_function(
    family: FamilyDefinition,
    y: np.ndarray,
    design_matrices: dict[str, np.ndarray],
    weights: np.ndarray,
    fixed_parameters: dict[str, np.ndarray],
) -> callable:
    """Build a loss function for optimizer.
    
    The loss function takes a parameter dictionary and returns the
    negative log-likelihood (global deviance).
    
    Parameters
    ----------
    family : FamilyDefinition
        Distribution family
    y : np.ndarray
        Response variable
    design_matrices : dict[str, np.ndarray]
        Design matrices for each parameter
    weights : np.ndarray
        Observation weights
    fixed_parameters : dict[str, np.ndarray]
        Fixed parameter values
    
    Returns
    -------
    loss_fn : callable
        Loss function with signature loss_fn(params, data) -> loss
    """
    
    def loss_fn(params: dict[str, jnp.ndarray], data: Any) -> float:
        """Compute negative log-likelihood (global deviance).
        
        Parameters
        ----------
        params : dict
            Dictionary with keys like "beta_mu", "beta_sigma", etc.
            Each value is a coefficient vector.
        data : Any
            Unused, kept for API compatibility
        
        Returns
        -------
        loss : float
            Global deviance (negative log-likelihood)
        """
        # Reconstruct fitted values from coefficients
        fitted_params = {}
        
        for param_name in family.estimable_parameters:
            beta_key = f"beta_{param_name}"
            if beta_key not in params:
                continue
            
            # Linear predictor: eta = X @ beta
            X = design_matrices[param_name]
            beta = params[beta_key]
            eta = X @ beta
            
            # Apply inverse link: param = g^{-1}(eta)
            param_values = family.link_inverses[param_name](eta)
            fitted_params[param_name] = param_values
        
        # Add fixed parameters
        for param_name, param_value in fixed_parameters.items():
            fitted_params[param_name] = jnp.asarray(param_value, dtype=jnp.float64)
        
        # Compute deviance increments
        dev_kwargs = {"y": jnp.asarray(y, dtype=jnp.float64), **fitted_params}
        dev_increments = family.g_dev_inc(**dev_kwargs)
        
        # Weighted sum
        w_jax = jnp.asarray(weights, dtype=jnp.float64)
        global_deviance = jnp.sum(dev_increments * w_jax)
        
        return global_deviance
    
    return loss_fn


def _extract_initial_params(
    initial_model: GAMLSSModel,
    family: FamilyDefinition,
) -> dict[str, jnp.ndarray]:
    """Extract initial parameters from a fitted model.
    
    Parameters
    ----------
    initial_model : GAMLSSModel
        Initial model fitted with RS/ML
    family : FamilyDefinition
        Distribution family
    
    Returns
    -------
    init_params : dict
        Dictionary with keys like "beta_mu", "beta_sigma", etc.
    """
    init_params = {}
    
    for param_name in family.estimable_parameters:
        if param_name in initial_model.coefficients:
            beta = np.asarray(initial_model.coefficients[param_name], dtype=np.float64)
            init_params[f"beta_{param_name}"] = jnp.asarray(beta, dtype=jnp.float64)
    
    return init_params


def _reconstruct_model(
    optimized_params: dict[str, jnp.ndarray],
    initial_model: GAMLSSModel,
    family: FamilyDefinition,
    design_matrices: dict[str, np.ndarray],
    y: np.ndarray,
    weights: np.ndarray,
    fixed_parameters: dict[str, np.ndarray],
    optimization_result: OptimizationResult,
    method_name: str,
) -> GAMLSSModel:
    """Reconstruct a GAMLSSModel from optimized parameters.
    
    Parameters
    ----------
    optimized_params : dict
        Optimized coefficient vectors
    initial_model : GAMLSSModel
        Initial model (for structure)
    family : FamilyDefinition
        Distribution family
    design_matrices : dict
        Design matrices
    y : np.ndarray
        Response variable
    weights : np.ndarray
        Observation weights
    fixed_parameters : dict
        Fixed parameter values
    optimization_result : OptimizationResult
        Optimization result object
    method_name : str
        Method name ("joint" or "lbfgs")
    
    Returns
    -------
    model : GAMLSSModel
        Reconstructed model
    """
    n = len(y)
    
    # Reconstruct fitted values and linear predictors
    fitted_values = {}
    coefficients = {}
    linear_predictors = {}
    
    for param_name in family.estimable_parameters:
        beta_key = f"beta_{param_name}"
        if beta_key not in optimized_params:
            continue
        
        # Get coefficients
        beta = np.asarray(optimized_params[beta_key], dtype=np.float64)
        coefficients[param_name] = jnp.asarray(beta, dtype=jnp.float64)
        
        # Compute linear predictor
        X = design_matrices[param_name]
        eta = X @ beta
        linear_predictors[param_name] = jnp.asarray(eta, dtype=jnp.float64)
        
        # Apply inverse link
        param_values = np.asarray(
            family.link_inverses[param_name](jnp.asarray(eta, dtype=jnp.float64)),
            dtype=np.float64
        )
        fitted_values[param_name] = jnp.asarray(param_values, dtype=jnp.float64)
    
    # Add fixed parameters
    for param_name, param_value in fixed_parameters.items():
        fitted_values[param_name] = jnp.asarray(param_value, dtype=jnp.float64)
        linear_predictors[param_name] = jnp.asarray(
            family.link_functions[param_name](jnp.asarray(param_value, dtype=jnp.float64)),
            dtype=jnp.float64
        )
    
    # Compute final deviance
    dev_kwargs = {"y": y, **{k: np.asarray(v) for k, v in fitted_values.items()}}
    g_dev = float(np.sum(np.asarray(family.g_dev_inc(**dev_kwargs)) * weights))
    
    # Compute degrees of freedom
    df_fit = sum(
        float(np.asarray(coefficients[p], dtype=np.float64).size)
        for p in family.estimable_parameters
        if p in coefficients
    )
    
    # Compute residuals
    mu = np.asarray(fitted_values["mu"], dtype=np.float64)
    sigma = np.asarray(fitted_values.get("sigma"), dtype=np.float64) if "sigma" in fitted_values else None
    
    if initial_model.rqres is not None:
        residuals = initial_model.rqres(y=y, mu=mu, sigma=sigma)
    else:
        # Simple residuals
        if sigma is not None:
            residuals = jnp.asarray((y - mu) / sigma, dtype=jnp.float64)
        else:
            residuals = jnp.asarray(y - mu, dtype=jnp.float64)
    
    # Create new model
    return GAMLSSModel(
        par=initial_model.par,
        family=family,
        df_fit=df_fit,
        g_dev=g_dev,
        n=n,
        y=jnp.asarray(y, dtype=jnp.float64),
        fitted_values=fitted_values,
        coefficients=coefficients,
        linear_predictors=linear_predictors,
        working_vectors=initial_model.working_vectors,  # Keep from initial
        iterative_weights=initial_model.iterative_weights,  # Keep from initial
        offsets=initial_model.offsets,
        formulas=initial_model.formulas,
        terms=initial_model.terms,
        design_matrices={k: jnp.asarray(v, dtype=jnp.float64) for k, v in design_matrices.items()},
        xlevels=initial_model.xlevels,
        additional_slots={
            "G.deviance": g_dev,
            "P.deviance": g_dev,
            "noObs": int(n),
            "df.residual": float(n - df_fit),
            "aic": float(g_dev + df_fit * 2.0),
            "sbc": float(g_dev + df_fit * np.log(max(n, 1))),
            "method": method_name.upper(),
            "converged": bool(optimization_result.converged),
            "cycles": int(optimization_result.n_iter),
            "deviance_history": tuple(float(v) for v in optimization_result.loss_history),
            "loss_history": optimization_result.loss_history,
            "grad_norms": optimization_result.grad_norms,
            "optimizer_result": optimization_result,
            "gradient_norm": (
                float(optimization_result.grad_norms[-1])
                if optimization_result.grad_norms
                else float("nan")
            ),
            "step_size": float("nan"),
            "condition_number": float("nan"),
        },
        call=initial_model.call,
        control=initial_model.control,
        iter=optimization_result.n_iter,
        weights=jnp.asarray(weights, dtype=jnp.float64),
        residuals=residuals,
        type=family.type,
        parameters=initial_model.parameters,
        rqres=initial_model.rqres,
    )


def fit_with_joint_optimizer(
    initial_model: GAMLSSModel,
    family: FamilyDefinition,
    design_matrices: dict[str, np.ndarray],
    y: np.ndarray,
    weights: np.ndarray,
    fixed_parameters: dict[str, np.ndarray],
    optimizer: str = "adam",
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-6,
    verbose: bool = False,
    **optimizer_kwargs
) -> GAMLSSModel:
    """Fit GAMLSS model using Joint Optimizer.
    
    This function takes an initial model (fitted with RS/ML) and refines
    it using a modern gradient-based optimizer from Optax.
    
    Parameters
    ----------
    initial_model : GAMLSSModel
        Initial model fitted with RS/ML
    family : FamilyDefinition
        Distribution family
    design_matrices : dict
        Design matrices for each parameter
    y : np.ndarray
        Response variable
    weights : np.ndarray
        Observation weights
    fixed_parameters : dict
        Fixed parameter values
    optimizer : str, default="adam"
        Optimizer type: "adam", "sgd", "rmsprop", "adagrad"
    learning_rate : float, default=0.01
        Learning rate
    max_iter : int, default=1000
        Maximum iterations
    tol : float, default=1e-6
        Convergence tolerance
    verbose : bool, default=False
        Print progress
    **optimizer_kwargs
        Additional optimizer arguments
    
    Returns
    -------
    model : GAMLSSModel
        Fitted model
    
    Examples
    --------
    >>> # Get initial model
    >>> initial = gamlss_ml("y ~ x1 + x2", family="NO", data=data)
    >>> 
    >>> # Refine with Adam
    >>> model = fit_with_joint_optimizer(
    ...     initial, family, design_matrices, y, weights, {},
    ...     optimizer="adam", learning_rate=0.01, verbose=True
    ... )
    """
    # Build loss function
    loss_fn = _build_loss_function(
        family, y, design_matrices, weights, fixed_parameters
    )
    
    # Extract initial parameters
    init_params = _extract_initial_params(initial_model, family)
    
    # Create optimizer
    opt = JointOptimizer(
        method=optimizer,
        learning_rate=learning_rate,
        max_iter=max_iter,
        tol=tol,
        track_grad_norms=True,
        **optimizer_kwargs
    )
    
    # Optimize
    if verbose:
        print(f"\nRefining with {optimizer.upper()} optimizer")
        print(f"Initial deviance: {initial_model.g_dev:.6f}")
    
    result = opt.optimize(
        loss_fn=loss_fn,
        init_params=init_params,
        data=None,  # Data is captured in loss_fn
        verbose=verbose
    )
    
    if verbose:
        print(f"Final deviance: {result.loss:.6f}")
        print(f"Converged: {result.converged}")
        print(f"Iterations: {result.n_iter}")
    
    # Reconstruct model
    return _reconstruct_model(
        optimized_params=result.params,
        initial_model=initial_model,
        family=family,
        design_matrices=design_matrices,
        y=y,
        weights=weights,
        fixed_parameters=fixed_parameters,
        optimization_result=result,
        method_name="joint"
    )


def fit_with_lbfgs(
    initial_model: GAMLSSModel,
    family: FamilyDefinition,
    design_matrices: dict[str, np.ndarray],
    y: np.ndarray,
    weights: np.ndarray,
    fixed_parameters: dict[str, np.ndarray],
    max_iter: int = 100,
    history_size: int = 10,
    learning_rate: float = 1.0,
    tol: float = 1e-6,
    verbose: bool = False,
    **optimizer_kwargs
) -> GAMLSSModel:
    """Fit GAMLSS model using L-BFGS optimizer.
    
    This function takes an initial model (fitted with RS/ML) and refines
    it using the L-BFGS quasi-Newton method.
    
    Parameters
    ----------
    initial_model : GAMLSSModel
        Initial model fitted with RS/ML
    family : FamilyDefinition
        Distribution family
    design_matrices : dict
        Design matrices for each parameter
    y : np.ndarray
        Response variable
    weights : np.ndarray
        Observation weights
    fixed_parameters : dict
        Fixed parameter values
    max_iter : int, default=100
        Maximum iterations
    history_size : int, default=10
        L-BFGS history size
    learning_rate : float, default=1.0
        Initial step size
    tol : float, default=1e-6
        Convergence tolerance
    verbose : bool, default=False
        Print progress
    **optimizer_kwargs
        Additional optimizer arguments
    
    Returns
    -------
    model : GAMLSSModel
        Fitted model
    
    Examples
    --------
    >>> # Get initial model
    >>> initial = gamlss_ml("y ~ x1 + x2", family="NO", data=data)
    >>> 
    >>> # Refine with L-BFGS
    >>> model = fit_with_lbfgs(
    ...     initial, family, design_matrices, y, weights, {},
    ...     max_iter=100, verbose=True
    ... )
    """
    # Build loss function
    loss_fn = _build_loss_function(
        family, y, design_matrices, weights, fixed_parameters
    )
    
    # Extract initial parameters
    init_params = _extract_initial_params(initial_model, family)
    
    # Create optimizer
    opt = LBFGSOptimizer(
        max_iter=max_iter,
        history_size=history_size,
        learning_rate=learning_rate,
        tol=tol,
        **optimizer_kwargs
    )
    
    # Optimize
    if verbose:
        print(f"\nRefining with L-BFGS optimizer")
        print(f"Initial deviance: {initial_model.g_dev:.6f}")
    
    result = opt.optimize(
        loss_fn=loss_fn,
        init_params=init_params,
        data=None,  # Data is captured in loss_fn
        verbose=verbose
    )
    
    if verbose:
        print(f"Final deviance: {result.loss:.6f}")
        print(f"Converged: {result.converged}")
        print(f"Iterations: {result.n_iter}")
    
    # Reconstruct model
    return _reconstruct_model(
        optimized_params=result.params,
        initial_model=initial_model,
        family=family,
        design_matrices=design_matrices,
        y=y,
        weights=weights,
        fixed_parameters=fixed_parameters,
        optimization_result=result,
        method_name="lbfgs"
    )
