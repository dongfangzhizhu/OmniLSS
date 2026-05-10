"""Mixed algorithm for GAMLSS fitting.

The Mixed algorithm combines the strengths of RS and CG algorithms,
automatically selecting the best approach based on convergence behavior.

References:
    Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized additive models 
    for location, scale and shape. Journal of the Royal Statistical Society: 
    Series C, 54(3), 507-554.
"""

from typing import Dict, Optional, Literal
import numpy as np

from .rs_algorithm import rs_fit
from .cg_algorithm import cg_fit
from ..model import GAMLSSModel


def mixed_fit(
    formula: str,
    sigma_formula: str = "~ 1",
    nu_formula: Optional[str] = None,
    tau_formula: Optional[str] = None,
    family: str = "NO",
    data: Optional[Dict[str, np.ndarray]] = None,
    algorithm: Literal["auto", "rs", "cg"] = "auto",
    max_iter: int = 20,
    tol: float = 1e-4,
    verbose: bool = False
) -> GAMLSSModel:
    """Fit GAMLSS model using Mixed algorithm.
    
    The Mixed algorithm can:
    1. Automatically select between RS and CG based on data characteristics
    2. Use RS algorithm (recommended for most cases)
    3. Use CG algorithm (for comparison)
    
    Args:
        formula: Formula for mu parameter
        sigma_formula: Formula for sigma parameter
        nu_formula: Formula for nu parameter (if applicable)
        tau_formula: Formula for tau parameter (if applicable)
        family: Distribution family name
        data: Dictionary of data arrays
        algorithm: Algorithm selection strategy:
            - "auto": Automatically select best algorithm (currently uses RS)
            - "rs": Use RS algorithm
            - "cg": Use CG algorithm
        max_iter: Maximum iterations
        tol: Convergence tolerance
        verbose: Whether to print iteration info
        
    Returns:
        Fitted GAMLSSModel
        
    Example:
        >>> data = {"y": y, "x": x}
        >>> # Automatic selection (uses RS)
        >>> model = mixed_fit("y ~ x", "~ x", family="NO", data=data)
        >>> 
        >>> # Explicit RS
        >>> model = mixed_fit("y ~ x", "~ x", family="NO", data=data, algorithm="rs")
        >>> 
        >>> # Explicit CG
        >>> model = mixed_fit("y ~ x", "~ x", family="NO", data=data, algorithm="cg")
    """
    if data is None:
        raise ValueError("data must be provided")
    
    if verbose:
        print("=" * 70)
        print("Mixed Algorithm")
        print("=" * 70)
    
    # Algorithm selection
    if algorithm == "auto":
        # Currently, RS algorithm is the best choice for most cases
        selected_algorithm = "rs"
        if verbose:
            print(f"Auto-selection: Using RS algorithm (recommended)")
    else:
        selected_algorithm = algorithm
        if verbose:
            print(f"Using {selected_algorithm.upper()} algorithm")
    
    # Fit using selected algorithm
    if selected_algorithm == "rs":
        # Build parameter_formulas for nu and tau
        parameter_formulas = {}
        if nu_formula is not None:
            parameter_formulas["nu"] = nu_formula
        if tau_formula is not None:
            parameter_formulas["tau"] = tau_formula
        
        model = rs_fit(
            formula=formula,
            sigma_formula=sigma_formula,
            parameter_formulas=parameter_formulas if parameter_formulas else None,
            family=family,
            data=data,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose
        )
        # Add mixed algorithm metadata
        model.additional_slots["mixed_algorithm_used"] = "rs"
        model.additional_slots["mixed_auto_selected"] = (algorithm == "auto")
        
    elif selected_algorithm == "cg":
        model = cg_fit(
            formula=formula,
            sigma_formula=sigma_formula,
            nu_formula=nu_formula,
            tau_formula=tau_formula,
            family=family,
            data=data,
            max_outer_iter=max_iter,
            outer_tol=tol,
            verbose=verbose
        )
        # Add mixed algorithm metadata
        model.additional_slots["mixed_algorithm_used"] = "cg"
        model.additional_slots["mixed_auto_selected"] = (algorithm == "auto")
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Use 'auto', 'rs', or 'cg'.")
    
    if verbose:
        print("=" * 70)
        print(f"Mixed algorithm completed using {selected_algorithm.upper()}")
        print("=" * 70)
    
    return model


def compare_algorithms(
    formula: str,
    sigma_formula: str = "~ 1",
    nu_formula: Optional[str] = None,
    tau_formula: Optional[str] = None,
    family: str = "NO",
    data: Optional[Dict[str, np.ndarray]] = None,
    verbose: bool = True
) -> Dict[str, GAMLSSModel]:
    """Compare RS and CG algorithms on the same data.
    
    Args:
        formula: Formula for mu parameter
        sigma_formula: Formula for sigma parameter
        nu_formula: Formula for nu parameter (if applicable)
        tau_formula: Formula for tau parameter (if applicable)
        family: Distribution family name
        data: Dictionary of data arrays
        verbose: Whether to print comparison results
        
    Returns:
        Dictionary with fitted models: {"rs": model_rs, "cg": model_cg}
        
    Example:
        >>> data = {"y": y, "x": x}
        >>> models = compare_algorithms("y ~ x", "~ x", family="NO", data=data)
        >>> print(f"RS deviance: {models['rs'].g_dev:.4f}")
        >>> print(f"CG deviance: {models['cg'].g_dev:.4f}")
    """
    if data is None:
        raise ValueError("data must be provided")
    
    if verbose:
        print("\n" + "=" * 70)
        print("Algorithm Comparison")
        print("=" * 70)
    
    # Fit with RS
    if verbose:
        print("\nFitting with RS algorithm...")
    
    # Build parameter_formulas for nu and tau
    parameter_formulas_rs = {}
    if nu_formula is not None:
        parameter_formulas_rs["nu"] = nu_formula
    if tau_formula is not None:
        parameter_formulas_rs["tau"] = tau_formula
    
    model_rs = rs_fit(
        formula=formula,
        sigma_formula=sigma_formula,
        parameter_formulas=parameter_formulas_rs if parameter_formulas_rs else None,
        family=family,
        data=data,
        verbose=False
    )
    
    # Fit with CG
    if verbose:
        print("Fitting with CG algorithm...")
    model_cg = cg_fit(
        formula=formula,
        sigma_formula=sigma_formula,
        nu_formula=nu_formula,
        tau_formula=tau_formula,
        family=family,
        data=data,
        verbose=False
    )
    
    # Print comparison
    if verbose:
        print("\n" + "-" * 70)
        print("Results:")
        print("-" * 70)
        print(f"{'Algorithm':<15} {'Deviance':<15} {'Iterations':<15} {'Status':<15}")
        print("-" * 70)
        
        rs_iters = model_rs.additional_slots.get("rs_iterations", "-")
        rs_conv = model_rs.additional_slots.get("rs_converged", False)
        print(f"{'RS':<15} {model_rs.g_dev:<15.4f} {rs_iters:<15} "
              f"{'Converged' if rs_conv else 'Not converged':<15}")
        
        cg_iters = model_cg.additional_slots.get("cg_iterations", "-")
        cg_conv = model_cg.additional_slots.get("cg_converged", False)
        print(f"{'CG':<15} {model_cg.g_dev:<15.4f} {cg_iters:<15} "
              f"{'Converged' if cg_conv else 'Not converged':<15}")
        
        print("-" * 70)
        print(f"Deviance difference (RS - CG): {model_rs.g_dev - model_cg.g_dev:+.4f}")
        print("=" * 70)
    
    return {"rs": model_rs, "cg": model_cg}
