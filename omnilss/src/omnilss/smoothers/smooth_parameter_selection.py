"""Automatic smoothing parameter selection with outer loop.

This module integrates GCV/REML smoothing parameter selection into the
GAMLSS fitting process using an outer loop strategy.

The outer loop alternates between:
1. Inner loop: Fix λ, optimize model parameters (μ, σ, ν, τ)
2. Outer loop: Fix model parameters, optimize λ

This continues until convergence.

NOTE: This is currently a PROTOTYPE implementation. The outer loop structure
is in place, but full λ optimization between iterations requires deeper
integration with the smooth fitting code.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Callable, Any
import warnings

from .gcv import optimize_lambda_gcv
from .reml import optimize_lambda_reml


def optimize_smoothing_parameters(
    y: np.ndarray,
    design_matrices: Dict[str, np.ndarray],
    penalty_matrices: Dict[str, List[np.ndarray]],
    weights: np.ndarray,
    method: str = "REML",
    lambda_init: Optional[Dict[str, List[float]]] = None,
    max_iter: int = 10,
    tol: float = 1e-4,
    verbose: bool = False
) -> Dict[str, List[float]]:
    """Optimize smoothing parameters for all smooth terms.
    
    This function optimizes λ values for all smooth terms across all
    distribution parameters using coordinate descent.
    
    Parameters
    ----------
    y : np.ndarray
        Response vector
    design_matrices : dict
        Design matrices for each parameter (μ, σ, ν, τ)
    penalty_matrices : dict
        Penalty matrices for each parameter's smooth terms
        Format: {parameter: [S1, S2, ...]} where Si is penalty for smooth i
    weights : np.ndarray
        Observation weights
    method : str, default="REML"
        Method for λ selection: "GCV" or "REML"
    lambda_init : dict, optional
        Initial λ values. If None, uses 1.0 for all
    max_iter : int, default=10
        Maximum outer loop iterations
    tol : float, default=1e-4
        Convergence tolerance
    verbose : bool, default=False
        Print progress
        
    Returns
    -------
    lambdas : dict
        Optimized λ values for each parameter
        Format: {parameter: [lambda1, lambda2, ...]}
        
    Notes
    -----
    This uses coordinate descent: optimize λ for each smooth term
    one at a time, holding others fixed.
    
    For production use, this should be integrated into the main
    GAMLSS fitting loop (outer loop strategy).
    """
    # Initialize λ values
    if lambda_init is None:
        lambdas = {}
        for param, penalties in penalty_matrices.items():
            lambdas[param] = [1.0] * len(penalties)
    else:
        lambdas = {k: list(v) for k, v in lambda_init.items()}
    
    # Select optimization function
    if method.upper() == "GCV":
        optimize_func = optimize_lambda_gcv
    elif method.upper() == "REML":
        optimize_func = optimize_lambda_reml
    else:
        raise ValueError(f"method must be 'GCV' or 'REML', got {method}")
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Optimizing smoothing parameters using {method}")
        print(f"{'='*70}")
        print(f"Parameters with smooths: {list(penalty_matrices.keys())}")
        total_smooths = sum(len(v) for v in penalty_matrices.values())
        print(f"Total smooth terms: {total_smooths}")
    
    # Outer loop: iterate until convergence
    for iteration in range(max_iter):
        lambdas_old = {k: list(v) for k, v in lambdas.items()}
        
        # Optimize λ for each parameter
        for param in penalty_matrices.keys():
            if param not in design_matrices:
                continue
            
            X = design_matrices[param]
            penalties = penalty_matrices[param]
            
            # Optimize each smooth term's λ
            for i, S in enumerate(penalties):
                if verbose:
                    print(f"\nIteration {iteration+1}, {param}, smooth {i+1}/{len(penalties)}")
                
                # Optimize this λ
                lambda_opt, score_opt = optimize_func(y, X, S, lambda_range=(1e-6, 1e6))
                
                if verbose:
                    print(f"  λ: {lambdas[param][i]:.4e} → {lambda_opt:.4e}")
                    print(f"  {method} score: {score_opt:.4f}")
                
                lambdas[param][i] = lambda_opt
        
        # Check convergence
        max_change = 0.0
        for param in lambdas.keys():
            for i in range(len(lambdas[param])):
                old_val = lambdas_old[param][i]
                new_val = lambdas[param][i]
                change = abs(new_val - old_val) / (old_val + 1e-8)
                max_change = max(max_change, change)
        
        if verbose:
            print(f"\nMax relative change: {max_change:.6f}")
        
        if max_change < tol:
            if verbose:
                print(f"✓ Converged after {iteration+1} iterations")
            break
    else:
        if verbose:
            print(f"⚠ Did not converge after {max_iter} iterations")
    
    return lambdas


def gamlss_with_automatic_smoothing(
    fit_function: Callable,
    formula: str,
    family: Any,
    data: Dict[str, Any],
    sigma_formula: str = "~1",
    parameter_formulas: Optional[Dict[str, str]] = None,
    lambda_method: str = "REML",
    max_outer_iter: int = 5,
    outer_tol: float = 1e-3,
    verbose: bool = False,
    **kwargs
) -> Any:
    """Fit GAMLSS with automatic smoothing parameter selection.
    
    This implements the outer loop strategy:
    1. Fit model with current λ values (inner loop)
    2. Optimize λ values given current fit (outer loop)
    3. Repeat until convergence
    
    Parameters
    ----------
    fit_function : callable
        The base GAMLSS fitting function (e.g., gamlss_ml or gamlss)
    formula : str
        Formula for μ parameter
    family : Any
        Distribution family
    data : dict
        Data dictionary
    sigma_formula : str, default="~1"
        Formula for σ parameter
    parameter_formulas : dict, optional
        Formulas for other parameters
    lambda_method : str, default="REML"
        Method for λ selection: "GCV" or "REML"
    max_outer_iter : int, default=5
        Maximum outer loop iterations
    outer_tol : float, default=1e-3
        Convergence tolerance for deviance
    verbose : bool, default=False
        Print progress
    **kwargs
        Additional arguments passed to fit_function
        
    Returns
    -------
    model : GAMLSSModel
        Fitted model with optimized λ values
        
    Notes
    -----
    **PROTOTYPE IMPLEMENTATION**
    
    This is a prototype implementation that demonstrates the outer loop
    structure for automatic smoothing parameter selection. The algorithm
    alternates between:
    
    1. Inner loop: Fix λ, optimize distribution parameters (μ, σ, ν, τ)
    2. Outer loop: Fix parameters, optimize λ using GCV or REML
    
    Currently, the outer loop is a placeholder that returns the initial fit.
    Full implementation requires:
    - Extracting penalty matrices from smooth_fits
    - Optimizing λ using GCV/REML
    - Passing new λ values back to the fitting function
    - Modifying smooth fitting code to accept external λ values
    
    The outer loop can be expensive, so we limit iterations to 5 by default.
    In practice, convergence usually happens in 2-3 iterations.
    
    Examples
    --------
    >>> from omnilss.fitting import gamlss_ml
    >>> from omnilss.smoothers.smooth_parameter_selection import gamlss_with_automatic_smoothing
    >>> 
    >>> model = gamlss_with_automatic_smoothing(
    ...     fit_function=gamlss_ml,
    ...     formula="y ~ s(x1) + s(x2)",
    ...     sigma_formula="~ s(x3)",
    ...     family="NO",
    ...     data=data,
    ...     lambda_method="REML",
    ...     verbose=True
    ... )
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"GAMLSS with Automatic Smoothing Parameter Selection")
        print(f"{'='*70}")
        print(f"Method: {lambda_method}")
        print(f"Max outer iterations: {max_outer_iter}")
        print(f"Convergence tolerance: {outer_tol}")
        print(f"\n⚠ NOTE: This is a PROTOTYPE implementation")
        print(f"The outer loop structure is in place but not fully active.")
    
    # Initial fit with default λ values
    if verbose:
        print(f"\n{'='*70}")
        print(f"Initial fit (iteration 0)")
        print(f"{'='*70}")
    
    model = fit_function(
        formula=formula,
        sigma_formula=sigma_formula,
        parameter_formulas=parameter_formulas,
        family=family,
        data=data,
        **kwargs
    )
    
    old_deviance = float(model.g_dev)
    
    if verbose:
        print(f"Initial deviance: {old_deviance:.6f}")
    
    # Check if model has smooth terms
    smooth_fits = model.additional_slots.get("smooth_fits", {})
    has_smooths = any(len(fits) > 0 for fits in smooth_fits.values())
    
    if not has_smooths:
        if verbose:
            print("\n⚠ No smooth terms found, returning initial fit")
        return model
    
    # Display smooth term information
    if verbose:
        print(f"\nSmooth terms in model:")
        for param, smooth_list in smooth_fits.items():
            if len(smooth_list) > 0:
                print(f"  {param}: {len(smooth_list)} smooth term(s)")
                for i, smooth_fit in enumerate(smooth_list):
                    print(f"    - {smooth_fit.variable}: λ={smooth_fit.lambda_:.4e}, edf={smooth_fit.edf:.2f}")
    
    # NOTE: Full outer loop implementation would go here
    # For now, we return the initial fit
    # Future implementation will:
    # 1. Extract penalty matrices from smooth_fits
    # 2. Optimize λ values using optimize_smoothing_parameters()
    # 3. Refit model with new λ values
    # 4. Repeat until convergence
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Final model (prototype - no outer loop iterations)")
        print(f"{'='*70}")
        print(f"Deviance: {model.g_dev:.6f}")
        print(f"AIC: {model.additional_slots.get('aic', np.nan):.6f}")
        print(f"Effective df: {model.df_fit:.2f}")
        print(f"\nTo enable full outer loop optimization, the smooth fitting")
        print(f"code needs to be modified to accept external λ values.")
    
    return model


# Convenience function for users
def auto_smooth(
    formula: str,
    family: Any,
    data: Dict[str, Any],
    sigma_formula: str = "~1",
    parameter_formulas: Optional[Dict[str, str]] = None,
    method: str = "REML",
    verbose: bool = False,
    **kwargs
) -> Any:
    """Fit GAMLSS with automatic smoothing parameter selection.
    
    This is a convenience wrapper around gamlss_with_automatic_smoothing
    that uses gamlss_ml as the base fitting function.
    
    Parameters
    ----------
    formula : str
        Formula for μ parameter (can include s() terms)
    family : Any
        Distribution family
    data : dict
        Data dictionary
    sigma_formula : str, default="~1"
        Formula for σ parameter (can include s() terms)
    parameter_formulas : dict, optional
        Formulas for other parameters (can include s() terms)
    method : str, default="REML"
        Method for λ selection: "GCV" or "REML"
    verbose : bool, default=False
        Print progress
    **kwargs
        Additional arguments
        
    Returns
    -------
    model : GAMLSSModel
        Fitted model with optimized λ values
        
    Notes
    -----
    **PROTOTYPE IMPLEMENTATION**
    
    This is currently a prototype that demonstrates the API for automatic
    smoothing parameter selection. The model is fitted with default λ values.
    
    Full outer loop optimization will be enabled in future versions once the
    smooth fitting code is modified to accept external λ values.
        
    Examples
    --------
    >>> from omnilss.smoothers.smooth_parameter_selection import auto_smooth
    >>> 
    >>> # Automatic λ selection with REML (recommended)
    >>> model = auto_smooth(
    ...     "y ~ s(x1) + s(x2)",
    ...     family="NO",
    ...     data=data,
    ...     method="REML",
    ...     verbose=True
    ... )
    >>> 
    >>> # With GCV
    >>> model = auto_smooth(
    ...     "y ~ s(x1) + s(x2)",
    ...     sigma_formula="~ s(x3)",
    ...     family="NO",
    ...     data=data,
    ...     method="GCV",
    ...     verbose=True
    ... )
    """
    from ..fitting import gamlss_ml
    
    return gamlss_with_automatic_smoothing(
        fit_function=gamlss_ml,
        formula=formula,
        sigma_formula=sigma_formula,
        parameter_formulas=parameter_formulas,
        family=family,
        data=data,
        lambda_method=method,
        verbose=verbose,
        **kwargs
    )
