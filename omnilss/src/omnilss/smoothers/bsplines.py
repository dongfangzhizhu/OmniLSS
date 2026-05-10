"""B-splines basis functions for GAMLSS smoothers.

This module implements B-splines basis functions using the Cox-de Boor recursion formula.
B-splines are the foundation for P-splines and other smoothing methods.

R source: gamlss/R/pb.R, mgcv package
References:
- de Boor, C. (1978). A Practical Guide to Splines.
- Eilers, P. H. C., & Marx, B. D. (1996). Flexible smoothing with B-splines and penalties.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax.numpy as jnp
import numpy as np


def bspline_basis(
    x: jnp.ndarray,
    knots: jnp.ndarray,
    degree: int = 3,
    derivative: int = 0,
) -> jnp.ndarray:
    """Compute B-spline basis functions using Cox-de Boor recursion.
    
    Parameters
    ----------
    x : jnp.ndarray
        Evaluation points, shape (n,)
    knots : jnp.ndarray
        Knot sequence (including boundary knots), shape (m,)
        Must be non-decreasing
    degree : int, default=3
        Degree of the B-spline (3 for cubic splines)
    derivative : int, default=0
        Order of derivative (0 for basis functions, 1 for first derivative, etc.)
    
    Returns
    -------
    basis : jnp.ndarray
        B-spline basis matrix, shape (n, m - degree - 1)
        Each column is a basis function evaluated at x
    
    Notes
    -----
    The Cox-de Boor recursion formula:
    B_{i,0}(x) = 1 if knots[i] <= x < knots[i+1], else 0
    B_{i,p}(x) = (x - knots[i]) / (knots[i+p] - knots[i]) * B_{i,p-1}(x)
                 + (knots[i+p+1] - x) / (knots[i+p+1] - knots[i+1]) * B_{i+1,p-1}(x)
    
    R equivalent: splines::bs() or mgcv::smoothCon()
    """
    x = jnp.atleast_1d(x)
    knots = jnp.atleast_1d(knots)
    
    n = len(x)
    m = len(knots)
    n_basis = m - degree - 1
    
    if n_basis <= 0:
        raise ValueError(f"Not enough knots: need at least {degree + 2}, got {m}")
    
    # Initialize basis matrix
    basis = jnp.zeros((n, n_basis))
    
    # Compute basis functions using Cox-de Boor recursion
    if derivative == 0:
        basis = _bspline_basis_recursive(x, knots, degree, n_basis)
    else:
        # For derivatives, use the derivative formula
        basis = _bspline_derivative(x, knots, degree, derivative, n_basis)
    
    return basis


def _bspline_basis_recursive(
    x: jnp.ndarray,
    knots: jnp.ndarray,
    degree: int,
    n_basis: int,
) -> jnp.ndarray:
    """Compute B-spline basis using Cox-de Boor recursion.
    
    This implementation follows the standard B-spline convention:
    - Intervals are [knots[i], knots[i+1]) (left-closed, right-open)
    - At the right boundary knots[-1], we include the point in the last interval
    
    This is a simplified implementation that works for JAX.
    For production use, consider using scipy.interpolate.BSpline or similar.
    """
    n = len(x)
    eps = 1e-10
    
    # Start with degree 0 (piecewise constant)
    basis_prev = jnp.zeros((n, n_basis + degree))
    
    # B_{i,0}(x) = 1 if knots[i] <= x < knots[i+1], else 0
    # Special case: at right boundary, include points in intervals ending at the boundary
    for i in range(n_basis + degree):
        left_cond = x >= knots[i]
        right_cond = x < knots[i + 1]
        
        # Check if x is at the right boundary and this interval ends at the boundary
        at_right_boundary = jnp.abs(x - knots[-1]) < eps
        interval_ends_at_boundary = jnp.abs(knots[i + 1] - knots[-1]) < eps
        
        # Include point if: (left_cond AND right_cond) OR (at_right_boundary AND interval_ends_at_boundary)
        mask = (left_cond & right_cond) | (at_right_boundary & interval_ends_at_boundary)
        basis_prev = basis_prev.at[:, i].set(jnp.where(mask, 1.0, 0.0))
    
    # Recursively compute higher degrees
    for p in range(1, degree + 1):
        basis_curr = jnp.zeros((n, n_basis + degree - p))
        
        for i in range(n_basis + degree - p):
            # Left term: (x - knots[i]) / (knots[i+p] - knots[i]) * B_{i,p-1}(x)
            denom_left = knots[i + p] - knots[i]
            if denom_left > eps:
                left_term = (x - knots[i]) / denom_left * basis_prev[:, i]
            else:
                left_term = jnp.zeros(n)
            
            # Right term: (knots[i+p+1] - x) / (knots[i+p+1] - knots[i+1]) * B_{i+1,p-1}(x)
            denom_right = knots[i + p + 1] - knots[i + 1]
            if denom_right > eps:
                right_term = (knots[i + p + 1] - x) / denom_right * basis_prev[:, i + 1]
            else:
                right_term = jnp.zeros(n)
            
            basis_curr = basis_curr.at[:, i].set(left_term + right_term)
        
        basis_prev = basis_curr
    
    return basis_prev


def _bspline_derivative(
    x: jnp.ndarray,
    knots: jnp.ndarray,
    degree: int,
    derivative: int,
    n_basis: int,
) -> jnp.ndarray:
    """Compute derivatives of B-spline basis functions.
    
    The derivative of a B-spline is a linear combination of lower-degree B-splines:
    B'_{i,p}(x) = p / (knots[i+p] - knots[i]) * B_{i,p-1}(x)
                  - p / (knots[i+p+1] - knots[i+1]) * B_{i+1,p-1}(x)
    """
    if derivative == 0:
        return _bspline_basis_recursive(x, knots, degree, n_basis)
    
    # Compute lower-degree basis
    basis_lower = _bspline_basis_recursive(x, knots, degree - 1, n_basis + 1)
    
    # Compute derivative
    n = len(x)
    basis_deriv = jnp.zeros((n, n_basis))
    
    for i in range(n_basis):
        # Left term
        denom_left = knots[i + degree] - knots[i]
        if denom_left > 1e-10:
            left_term = degree / denom_left * basis_lower[:, i]
        else:
            left_term = jnp.zeros(n)
        
        # Right term
        denom_right = knots[i + degree + 1] - knots[i + 1]
        if denom_right > 1e-10:
            right_term = degree / denom_right * basis_lower[:, i + 1]
        else:
            right_term = jnp.zeros(n)
        
        basis_deriv = basis_deriv.at[:, i].set(left_term - right_term)
    
    # Recursively compute higher-order derivatives
    if derivative > 1:
        # This is a simplified version; full implementation would be more complex
        raise NotImplementedError("Higher-order derivatives not yet implemented")
    
    return basis_deriv


def create_knots(
    x: np.ndarray,
    n_knots: Optional[int] = None,
    degree: int = 3,
    boundary: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Create knot sequence for B-splines.
    
    Parameters
    ----------
    x : np.ndarray
        Data points
    n_knots : int, optional
        Number of interior knots. If None, uses a default based on sample size.
    degree : int, default=3
        Degree of the B-spline
    boundary : tuple, optional
        (lower, upper) boundary knots. If None, uses min and max of x.
    
    Returns
    -------
    knots : np.ndarray
        Full knot sequence including boundary knots and multiplicities
    
    Notes
    -----
    The knot sequence includes:
    - degree + 1 copies of the lower boundary knot
    - n_knots interior knots (evenly spaced quantiles)
    - degree + 1 copies of the upper boundary knot
    
    R equivalent: Default knot placement in mgcv::s() or gamlss::pb()
    """
    if boundary is None:
        x_min, x_max = np.min(x), np.max(x)
    else:
        x_min, x_max = boundary
    
    if n_knots is None:
        # Default: use a rule of thumb based on sample size
        n = len(x)
        n_knots = min(max(5, n // 10), 40)
    
    # Create interior knots at evenly spaced quantiles
    if n_knots > 0:
        quantiles = np.linspace(0, 1, n_knots + 2)[1:-1]
        interior_knots = np.quantile(x, quantiles)
    else:
        interior_knots = np.array([])
    
    # Create full knot sequence with boundary multiplicities
    knots = np.concatenate([
        np.repeat(x_min, degree + 1),
        interior_knots,
        np.repeat(x_max, degree + 1),
    ])
    
    return knots


def bspline_design_matrix(
    x: np.ndarray,
    n_knots: Optional[int] = None,
    degree: int = 3,
    boundary: Optional[Tuple[float, float]] = None,
    derivative: int = 0,
) -> Tuple[jnp.ndarray, np.ndarray]:
    """Create B-spline design matrix for regression.
    
    Parameters
    ----------
    x : np.ndarray
        Predictor variable
    n_knots : int, optional
        Number of interior knots
    degree : int, default=3
        Degree of the B-spline
    boundary : tuple, optional
        (lower, upper) boundary knots
    derivative : int, default=0
        Order of derivative
    
    Returns
    -------
    design_matrix : jnp.ndarray
        B-spline design matrix, shape (n, n_basis)
    knots : np.ndarray
        Knot sequence used
    
    Examples
    --------
    >>> x = np.linspace(0, 1, 100)
    >>> X, knots = bspline_design_matrix(x, n_knots=10, degree=3)
    >>> X.shape
    (100, 14)  # 10 interior knots + 3 + 1 = 14 basis functions
    """
    knots = create_knots(x, n_knots=n_knots, degree=degree, boundary=boundary)
    design_matrix = bspline_basis(jnp.array(x), jnp.array(knots), degree=degree, derivative=derivative)
    
    return design_matrix, knots
