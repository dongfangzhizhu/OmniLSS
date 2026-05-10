"""
UBRE (Unbiased Risk Estimator) for smoothing parameter selection.

UBRE is an alternative to GCV for selecting smoothing parameters when the
error variance is known or can be estimated reliably.

References
----------
Craven, P., & Wahba, G. (1978). Smoothing noisy data with spline functions.
Numerische Mathematik, 31(4), 377-403.

Wahba, G. (1990). Spline models for observational data. SIAM.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import jit
from typing import Optional, Tuple


@jit
def ubre_score(
    y: jnp.ndarray,
    fitted: jnp.ndarray,
    hat_matrix: jnp.ndarray,
    sigma2: float,
) -> float:
    """Calculate UBRE score for a given fit.
    
    The UBRE (Unbiased Risk Estimator) is defined as:
    
        UBRE = (1/n) * RSS + 2 * σ² * (tr(H) / n) - σ²
    
    where:
    - RSS is the residual sum of squares
    - H is the hat (smoother) matrix
    - σ² is the error variance (assumed known)
    - n is the number of observations
    
    Lower UBRE scores indicate better fits.
    
    Parameters
    ----------
    y : jnp.ndarray
        Response variable (n,)
    fitted : jnp.ndarray
        Fitted values (n,)
    hat_matrix : jnp.ndarray
        Hat matrix (smoother matrix) (n, n)
    sigma2 : float
        Error variance (assumed known)
        
    Returns
    -------
    float
        UBRE score (lower is better)
        
    Notes
    -----
    UBRE is equivalent to GCV when σ² is known. The main advantage of UBRE
    is that it can be more stable than GCV in some situations.
    
    When σ² is unknown, it can be estimated from a preliminary fit or
    using robust methods.
    
    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from omnilss.smoothers.ubre import ubre_score
    >>> 
    >>> # Generate data
    >>> n = 100
    >>> y = jnp.array([...])  # Response
    >>> fitted = jnp.array([...])  # Fitted values
    >>> H = jnp.array([...])  # Hat matrix
    >>> sigma2 = 1.0  # Known variance
    >>> 
    >>> # Calculate UBRE
    >>> score = ubre_score(y, fitted, H, sigma2)
    >>> print(f"UBRE score: {score:.4f}")
    """
    n = len(y)
    
    # Residual sum of squares
    residuals = y - fitted
    rss = jnp.sum(jnp.square(residuals))
    
    # Trace of hat matrix (effective degrees of freedom)
    trace_h = jnp.trace(hat_matrix)
    
    # UBRE formula
    ubre = (rss / n) + 2.0 * sigma2 * (trace_h / n) - sigma2
    
    return ubre


def select_lambda_ubre(
    X: jnp.ndarray,
    y: jnp.ndarray,
    P: jnp.ndarray,
    sigma2: float,
    lambda_grid: Optional[jnp.ndarray] = None,
    return_scores: bool = False,
) -> Tuple[float, Optional[jnp.ndarray]]:
    """Select smoothing parameter using UBRE.
    
    This function searches over a grid of lambda values and selects the one
    that minimizes the UBRE score.
    
    Parameters
    ----------
    X : jnp.ndarray
        Design matrix (n, p)
    y : jnp.ndarray
        Response variable (n,)
    P : jnp.ndarray
        Penalty matrix (p, p)
    sigma2 : float
        Error variance (assumed known)
    lambda_grid : jnp.ndarray, optional
        Grid of lambda values to search. If None, uses a default grid
        from 10^-6 to 10^6 with 100 points.
    return_scores : bool, default=False
        If True, also return the UBRE scores for all lambda values
        
    Returns
    -------
    best_lambda : float
        Optimal smoothing parameter
    ubre_scores : jnp.ndarray, optional
        UBRE scores for all lambda values (only if return_scores=True)
        
    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from omnilss.smoothers.ubre import select_lambda_ubre
    >>> from omnilss.smoothers.penalties import difference_penalty
    >>> 
    >>> # Generate data
    >>> n, p = 100, 20
    >>> X = jnp.array([...])  # Design matrix
    >>> y = jnp.array([...])  # Response
    >>> P = difference_penalty(p, order=2)  # Penalty matrix
    >>> sigma2 = 1.0  # Known variance
    >>> 
    >>> # Select lambda
    >>> best_lambda, scores = select_lambda_ubre(
    ...     X, y, P, sigma2, return_scores=True
    ... )
    >>> print(f"Optimal lambda: {best_lambda:.4e}")
    """
    if lambda_grid is None:
        lambda_grid = jnp.logspace(-6, 6, 100)
    
    n, p = X.shape
    XtX = X.T @ X
    Xty = X.T @ y
    
    ubre_scores = []
    
    for lam in lambda_grid:
        # Penalized least squares
        XtX_pen = XtX + lam * P
        
        # Solve for coefficients
        try:
            coef = jnp.linalg.solve(XtX_pen, Xty)
        except:
            # If singular, use pseudoinverse
            coef = jnp.linalg.lstsq(XtX_pen, Xty, rcond=None)[0]
        
        # Fitted values
        fitted = X @ coef
        
        # Hat matrix: H = X (X'X + λP)^(-1) X'
        try:
            XtX_pen_inv = jnp.linalg.inv(XtX_pen)
        except:
            XtX_pen_inv = jnp.linalg.pinv(XtX_pen)
        
        hat_matrix = X @ XtX_pen_inv @ X.T
        
        # Calculate UBRE
        ubre = ubre_score(y, fitted, hat_matrix, sigma2)
        ubre_scores.append(ubre)
    
    ubre_scores = jnp.array(ubre_scores)
    
    # Find minimum
    best_idx = jnp.argmin(ubre_scores)
    best_lambda = lambda_grid[best_idx]
    
    if return_scores:
        return best_lambda, ubre_scores
    else:
        return best_lambda, None


def estimate_sigma2(
    y: jnp.ndarray,
    fitted: jnp.ndarray,
    edf: float,
) -> float:
    """Estimate error variance from residuals.
    
    This function estimates σ² using:
    
        σ² = RSS / (n - edf)
    
    where edf is the effective degrees of freedom.
    
    Parameters
    ----------
    y : jnp.ndarray
        Response variable (n,)
    fitted : jnp.ndarray
        Fitted values (n,)
    edf : float
        Effective degrees of freedom
        
    Returns
    -------
    float
        Estimated error variance
        
    Notes
    -----
    This is a simple estimate that assumes the model is correctly specified.
    For more robust estimates, consider using methods like MAD (Median
    Absolute Deviation) or other robust scale estimators.
    
    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from omnilss.smoothers.ubre import estimate_sigma2
    >>> 
    >>> y = jnp.array([...])
    >>> fitted = jnp.array([...])
    >>> edf = 10.0  # From hat matrix trace
    >>> 
    >>> sigma2 = estimate_sigma2(y, fitted, edf)
    >>> print(f"Estimated σ²: {sigma2:.4f}")
    """
    n = len(y)
    residuals = y - fitted
    rss = jnp.sum(jnp.square(residuals))
    
    # Degrees of freedom for error
    df_error = n - edf
    
    # Ensure positive df
    df_error = jnp.maximum(df_error, 1.0)
    
    sigma2 = rss / df_error
    
    return sigma2


def ubre_vs_gcv(
    X: jnp.ndarray,
    y: jnp.ndarray,
    P: jnp.ndarray,
    sigma2: float,
    lambda_grid: Optional[jnp.ndarray] = None,
) -> dict:
    """Compare UBRE and GCV for smoothing parameter selection.
    
    This function computes both UBRE and GCV scores over a grid of lambda
    values and returns the results for comparison.
    
    Parameters
    ----------
    X : jnp.ndarray
        Design matrix (n, p)
    y : jnp.ndarray
        Response variable (n,)
    P : jnp.ndarray
        Penalty matrix (p, p)
    sigma2 : float
        Error variance for UBRE
    lambda_grid : jnp.ndarray, optional
        Grid of lambda values to search
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'lambda_grid': Grid of lambda values
        - 'ubre_scores': UBRE scores
        - 'gcv_scores': GCV scores
        - 'best_lambda_ubre': Optimal lambda by UBRE
        - 'best_lambda_gcv': Optimal lambda by GCV
        - 'sigma2': Error variance used
        
    Examples
    --------
    >>> from omnilss.smoothers.ubre import ubre_vs_gcv
    >>> 
    >>> results = ubre_vs_gcv(X, y, P, sigma2=1.0)
    >>> print(f"UBRE optimal λ: {results['best_lambda_ubre']:.4e}")
    >>> print(f"GCV optimal λ: {results['best_lambda_gcv']:.4e}")
    """
    from .gcv import compute_gcv_score
    
    if lambda_grid is None:
        lambda_grid = jnp.logspace(-6, 6, 100)
    
    n, p = X.shape
    XtX = X.T @ X
    Xty = X.T @ y
    
    ubre_scores = []
    gcv_scores = []
    
    for lam in lambda_grid:
        # Penalized least squares
        XtX_pen = XtX + lam * P
        
        try:
            coef = jnp.linalg.solve(XtX_pen, Xty)
            XtX_pen_inv = jnp.linalg.inv(XtX_pen)
        except:
            coef = jnp.linalg.lstsq(XtX_pen, Xty, rcond=None)[0]
            XtX_pen_inv = jnp.linalg.pinv(XtX_pen)
        
        fitted = X @ coef
        hat_matrix = X @ XtX_pen_inv @ X.T
        
        # UBRE
        ubre = ubre_score(y, fitted, hat_matrix, sigma2)
        ubre_scores.append(ubre)
        
        # GCV - convert to numpy for compute_gcv_score
        import numpy as np
        y_np = np.asarray(y)
        fitted_np = np.asarray(fitted)
        hat_matrix_np = np.asarray(hat_matrix)
        
        # Compute GCV manually since compute_gcv_score needs different inputs
        residuals = y_np - fitted_np
        rss = np.sum(residuals ** 2)
        edf = np.trace(hat_matrix_np)
        denominator = n - edf
        if denominator <= 0:
            gcv = np.inf
        else:
            gcv = n * rss / (denominator ** 2)
        gcv_scores.append(gcv)
    
    ubre_scores = jnp.array(ubre_scores)
    gcv_scores = jnp.array(gcv_scores)
    
    best_lambda_ubre = lambda_grid[jnp.argmin(ubre_scores)]
    best_lambda_gcv = lambda_grid[jnp.argmin(gcv_scores)]
    
    return {
        'lambda_grid': lambda_grid,
        'ubre_scores': ubre_scores,
        'gcv_scores': gcv_scores,
        'best_lambda_ubre': best_lambda_ubre,
        'best_lambda_gcv': best_lambda_gcv,
        'sigma2': sigma2,
    }


__all__ = [
    'ubre_score',
    'select_lambda_ubre',
    'estimate_sigma2',
    'ubre_vs_gcv',
]
