"""Thin Plate Spline (TPS) smoother for GAMLSS.

Thin plate splines are a generalization of cubic splines to multiple dimensions.
They minimize a measure of surface roughness subject to interpolation constraints.

For 2D data, the TPS minimizes:
    RSS + λ * ∫∫ [(∂²f/∂x²)² + 2(∂²f/∂x∂y)² + (∂²f/∂y²)²] dx dy

References
----------
- Wood, S. N. (2003). Thin plate regression splines. Journal of the Royal 
  Statistical Society: Series B, 65(1), 95-114.
- Duchon, J. (1977). Splines minimizing rotation-invariant semi-norms in 
  Sobolev spaces. In Constructive theory of functions of several variables.
- Green, P. J., & Silverman, B. W. (1993). Nonparametric regression and 
  generalized linear models: a roughness penalty approach. CRC Press.

Implementation Notes
--------------------
This implementation follows the approach in Wood (2003) and mgcv:
1. Construct radial basis functions φ(r) where r is Euclidean distance
2. Add polynomial terms for unpenalized space
3. Apply penalty to radial basis coefficients only
4. Use eigendecomposition for efficient computation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import jax.numpy as jnp
import numpy as np
from scipy import linalg
from scipy.optimize import minimize_scalar

from omnilss.smoothers.penalties import effective_df


@dataclass
class TPSResult:
    """Result from thin plate spline fitting.
    
    Attributes
    ----------
    coefficients : np.ndarray
        Spline coefficients (includes both radial and polynomial terms)
    fitted_values : np.ndarray
        Fitted values at training points
    lambda_ : float
        Smoothing parameter used
    edf : float
        Effective degrees of freedom
    knots : np.ndarray
        Knot locations (subset of data points or all points)
    X_train : np.ndarray
        Training predictor matrix (n_obs, n_dims)
    n_dims : int
        Number of dimensions (2 for 2D TPS, 3 for 3D, etc.)
    m : int
        Order of derivative in penalty (typically 2)
    selection_method : str, optional
        Method used for lambda selection
    criterion_value : float, optional
        Value of the optimization criterion
    basis_matrix : np.ndarray
        Full basis matrix (radial + polynomial)
    penalty_matrix : np.ndarray
        Penalty matrix (only penalizes radial terms)
    """
    coefficients: np.ndarray
    fitted_values: np.ndarray
    lambda_: float
    edf: float
    knots: np.ndarray
    X_train: np.ndarray
    n_dims: int
    m: int = 2
    selection_method: Optional[str] = None
    criterion_value: Optional[float] = None
    basis_matrix: Optional[np.ndarray] = None
    penalty_matrix: Optional[np.ndarray] = None
    
    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """Predict at new locations.
        
        Parameters
        ----------
        X_new : np.ndarray
            New predictor locations, shape (n_new, n_dims)
            
        Returns
        -------
        np.ndarray
            Predicted values at new locations
        """
        X_new = np.asarray(X_new)
        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)
        
        if X_new.shape[1] != self.n_dims:
            raise ValueError(
                f"X_new must have {self.n_dims} columns, got {X_new.shape[1]}"
            )
        
        # Build basis matrix at new locations
        basis_new = _tps_basis_matrix(X_new, self.knots, self.n_dims, self.m)
        
        # Predict
        return basis_new @ self.coefficients


def _tps_radial_basis(r: np.ndarray, n_dims: int, m: int = 2) -> np.ndarray:
    """Thin plate spline radial basis function.
    
    For 2D (d=2), m=2: φ(r) = r² log(r)
    For 3D (d=3), m=2: φ(r) = r
    General form depends on dimension d and derivative order m.
    
    Parameters
    ----------
    r : np.ndarray
        Euclidean distances
    n_dims : int
        Number of dimensions
    m : int
        Order of derivative in penalty (typically 2)
        
    Returns
    -------
    np.ndarray
        Radial basis function values
        
    Notes
    -----
    The radial basis function is chosen such that the penalty
    corresponds to the desired derivative order m.
    
    For even dimensions d and m=2:
        φ(r) = r^(2m-d) log(r)  if 2m > d
        φ(r) = log(r)           if 2m = d
        
    For odd dimensions d and m=2:
        φ(r) = r^(2m-d)         if 2m > d
    """
    r = np.asarray(r)
    eps = 1e-10  # Avoid log(0)
    r = np.maximum(r, eps)
    
    if n_dims == 1:
        # 1D: cubic spline
        return r ** 3
    elif n_dims == 2:
        # 2D: r² log(r)
        return r ** 2 * np.log(r)
    elif n_dims == 3:
        # 3D: r (for m=2)
        return r
    else:
        # General case: r^(2m-d) for odd d, r^(2m-d) log(r) for even d
        power = 2 * m - n_dims
        if n_dims % 2 == 0:
            # Even dimension
            if power > 0:
                return r ** power * np.log(r)
            else:
                return np.log(r)
        else:
            # Odd dimension
            return r ** power


def _euclidean_distance(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances.
    
    Parameters
    ----------
    X1 : np.ndarray
        First set of points, shape (n1, d)
    X2 : np.ndarray
        Second set of points, shape (n2, d)
        
    Returns
    -------
    np.ndarray
        Distance matrix, shape (n1, n2)
    """
    # Efficient computation: ||x - y||² = ||x||² + ||y||² - 2x·y
    X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)  # (n1, 1)
    X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True).T  # (1, n2)
    cross = X1 @ X2.T  # (n1, n2)
    
    dist_sq = X1_sq + X2_sq - 2 * cross
    dist_sq = np.maximum(dist_sq, 0.0)  # Numerical safety
    
    return np.sqrt(dist_sq)


def _polynomial_basis(X: np.ndarray, m: int = 2) -> np.ndarray:
    """Construct polynomial basis for unpenalized space.
    
    For m=2 (second derivative penalty):
    - 1D: [1, x]
    - 2D: [1, x, y]
    - 3D: [1, x, y, z]
    
    Parameters
    ----------
    X : np.ndarray
        Predictor matrix, shape (n, d)
    m : int
        Order of derivative (determines polynomial degree = m-1)
        
    Returns
    -------
    np.ndarray
        Polynomial basis matrix, shape (n, M)
        where M = (d + m - 1)! / (d! * (m-1)!)
    """
    n, d = X.shape
    
    if m == 2:
        # Linear polynomial: [1, x1, x2, ..., xd]
        return np.column_stack([np.ones(n), X])
    elif m == 3:
        # Quadratic polynomial: [1, x, x², xy, y, y², ...]
        # For simplicity, we implement only m=2 case
        # Full implementation would include all terms up to degree m-1
        raise NotImplementedError("Only m=2 (linear polynomial) is currently supported")
    else:
        raise ValueError(f"m={m} not supported, use m=2")


def _tps_basis_matrix(
    X: np.ndarray,
    knots: np.ndarray,
    n_dims: int,
    m: int = 2,
) -> np.ndarray:
    """Construct full TPS basis matrix.
    
    The basis consists of:
    1. Radial basis functions φ(||x - knot_i||) for each knot
    2. Polynomial terms for the null space
    
    Parameters
    ----------
    X : np.ndarray
        Evaluation points, shape (n, d)
    knots : np.ndarray
        Knot locations, shape (k, d)
    n_dims : int
        Number of dimensions
    m : int
        Derivative order
        
    Returns
    -------
    np.ndarray
        Basis matrix, shape (n, k + M)
        where M is the number of polynomial terms
    """
    n = X.shape[0]
    k = knots.shape[0]
    
    # Radial basis part
    distances = _euclidean_distance(X, knots)  # (n, k)
    radial = _tps_radial_basis(distances, n_dims, m)  # (n, k)
    
    # Polynomial part
    poly = _polynomial_basis(X, m)  # (n, M)
    
    # Combine: [radial | polynomial]
    return np.column_stack([radial, poly])


def _tps_penalty_matrix(n_knots: int, n_poly: int) -> np.ndarray:
    """Construct TPS penalty matrix.
    
    The penalty only applies to the radial basis coefficients,
    not to the polynomial coefficients (null space).
    
    Parameters
    ----------
    n_knots : int
        Number of knots (radial basis functions)
    n_poly : int
        Number of polynomial terms
        
    Returns
    -------
    np.ndarray
        Penalty matrix, shape (n_knots + n_poly, n_knots + n_poly)
        Block diagonal: [I_k | 0; 0 | 0]
    """
    total = n_knots + n_poly
    P = np.zeros((total, total))
    P[:n_knots, :n_knots] = np.eye(n_knots)
    return P


def _select_knots(
    X: np.ndarray,
    k: Optional[int] = None,
    method: Literal["uniform", "kmeans", "all"] = "uniform",
) -> np.ndarray:
    """Select knot locations for TPS.
    
    Parameters
    ----------
    X : np.ndarray
        Data points, shape (n, d)
    k : int, optional
        Number of knots. If None, uses min(n, 100)
    method : str
        Knot selection method:
        - "uniform": Uniform random sampling
        - "kmeans": K-means clustering
        - "all": Use all data points as knots
        
    Returns
    -------
    np.ndarray
        Selected knot locations, shape (k, d)
    """
    n, d = X.shape
    
    if method == "all" or k is None or k >= n:
        return X.copy()
    
    if method == "uniform":
        # Random uniform sampling
        rng = np.random.RandomState(42)
        indices = rng.choice(n, size=k, replace=False)
        return X[indices]
    
    elif method == "kmeans":
        # K-means clustering
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            return kmeans.cluster_centers_
        except ImportError:
            # Fallback to uniform if sklearn not available
            rng = np.random.RandomState(42)
            indices = rng.choice(n, size=k, replace=False)
            return X[indices]
    
    else:
        raise ValueError(f"Unknown knot selection method: {method}")


def _penalized_ls_tps(
    basis: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    penalty: np.ndarray,
    lambda_: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Penalized weighted least squares for TPS.
    
    Solves: (B^T W B + λ P) β = B^T W y
    
    Parameters
    ----------
    basis : np.ndarray
        Basis matrix, shape (n, p)
    y : np.ndarray
        Response vector
    w : np.ndarray
        Weights
    penalty : np.ndarray
        Penalty matrix
    lambda_ : float
        Smoothing parameter
        
    Returns
    -------
    coefficients : np.ndarray
        Fitted coefficients
    fitted : np.ndarray
        Fitted values
    edf : float
        Effective degrees of freedom
    """
    # Weighted basis
    W_sqrt = np.sqrt(w)
    B_w = basis * W_sqrt[:, np.newaxis]
    y_w = y * W_sqrt
    
    # Penalized normal equations
    BtWB = B_w.T @ B_w
    BtWy = B_w.T @ y_w
    
    # Add penalty
    A = BtWB + lambda_ * penalty
    
    # Solve
    try:
        coefficients = linalg.solve(A, BtWy, assume_a='pos')
    except linalg.LinAlgError:
        # Fallback to lstsq if not positive definite
        coefficients = linalg.lstsq(A, BtWy)[0]
    
    # Fitted values
    fitted = basis @ coefficients
    
    # Effective degrees of freedom
    # edf = tr(B (B^T W B + λ P)^-1 B^T W)
    try:
        A_inv = linalg.inv(A)
        hat_matrix = basis @ A_inv @ B_w.T @ np.diag(W_sqrt)
        edf = np.trace(hat_matrix)
    except linalg.LinAlgError:
        # Fallback: use simple approximation
        edf = np.sum(np.diag(BtWB) / (np.diag(BtWB) + lambda_ * np.diag(penalty)))
    
    return coefficients, fitted, float(edf)


def _gcv_score_tps(
    basis: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    penalty: np.ndarray,
    lambda_: float,
) -> float:
    """Compute GCV score for TPS.
    
    GCV = (n * RSS) / (n - edf)²
    
    Parameters
    ----------
    basis : np.ndarray
        Basis matrix
    y : np.ndarray
        Response
    w : np.ndarray
        Weights
    penalty : np.ndarray
        Penalty matrix
    lambda_ : float
        Smoothing parameter
        
    Returns
    -------
    float
        GCV score (lower is better)
    """
    if lambda_ < 0:
        return np.inf
    
    coefficients, fitted, edf = _penalized_ls_tps(basis, y, w, penalty, lambda_)
    
    residuals = y - fitted
    rss = np.sum(w * residuals ** 2)
    n = len(y)
    
    if edf >= n:
        return np.inf
    
    gcv = (n * rss) / (n - edf) ** 2
    return gcv


def fit_tps(
    X: np.ndarray,
    y: np.ndarray,
    w: Optional[np.ndarray] = None,
    lambda_: Optional[float] = None,
    k: Optional[int] = None,
    m: int = 2,
    method: Literal["GCV", "REML", "ML", "fixed"] = "GCV",
    knot_method: Literal["uniform", "kmeans", "all"] = "uniform",
) -> TPSResult:
    """Fit thin plate spline smoother.
    
    Parameters
    ----------
    X : np.ndarray
        Predictor matrix, shape (n, d) where d is number of dimensions
    y : np.ndarray
        Response vector, shape (n,)
    w : np.ndarray, optional
        Weights, shape (n,). Default is uniform weights.
    lambda_ : float, optional
        Smoothing parameter. If None, selected automatically using `method`.
    k : int, optional
        Number of knots. If None, uses min(n, 100).
        Using fewer knots speeds up computation.
    m : int
        Order of derivative in penalty (default: 2)
    method : str
        Method for selecting lambda if not provided:
        - "GCV": Generalized Cross-Validation
        - "REML": Restricted Maximum Likelihood (not yet implemented)
        - "ML": Maximum Likelihood (not yet implemented)
        - "fixed": Use provided lambda (must provide lambda_)
    knot_method : str
        Method for selecting knots:
        - "uniform": Random uniform sampling
        - "kmeans": K-means clustering
        - "all": Use all data points (can be slow for large n)
        
    Returns
    -------
    TPSResult
        Fitted TPS model
        
    Examples
    --------
    >>> import numpy as np
    >>> from omnilss.smoothers.tps import fit_tps
    >>> 
    >>> # 2D example
    >>> n = 100
    >>> x1 = np.random.uniform(0, 1, n)
    >>> x2 = np.random.uniform(0, 1, n)
    >>> X = np.column_stack([x1, x2])
    >>> y = np.sin(2 * np.pi * x1) * np.cos(2 * np.pi * x2) + np.random.normal(0, 0.1, n)
    >>> 
    >>> # Fit TPS
    >>> result = fit_tps(X, y, k=20, method="GCV")
    >>> print(f"Lambda: {result.lambda_:.4e}")
    >>> print(f"EDF: {result.edf:.2f}")
    >>> 
    >>> # Predict at new locations
    >>> X_new = np.column_stack([
    ...     np.linspace(0, 1, 50),
    ...     np.linspace(0, 1, 50)
    ... ])
    >>> y_pred = result.predict(X_new)
    
    Notes
    -----
    Thin plate splines are particularly useful for:
    - Spatial smoothing (2D geographic data)
    - Image smoothing
    - Multi-dimensional response surfaces
    
    For large datasets (n > 1000), consider using fewer knots (k < 100)
    to speed up computation. The knot selection method can affect results:
    - "uniform": Fast, random
    - "kmeans": Better coverage, slower
    - "all": Most accurate, slowest
    
    References
    ----------
    Wood, S. N. (2003). Thin plate regression splines. JRSS-B, 65(1), 95-114.
    """
    # Input validation
    X = np.asarray(X)
    y = np.asarray(y)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n, n_dims = X.shape
    
    if len(y) != n:
        raise ValueError(f"X and y must have same length, got {n} and {len(y)}")
    
    if w is None:
        w = np.ones(n)
    else:
        w = np.asarray(w)
        if len(w) != n:
            raise ValueError(f"w must have length {n}, got {len(w)}")
    
    # Select knots
    if k is None:
        k = min(n, 100)
    
    knots = _select_knots(X, k, knot_method)
    n_knots = knots.shape[0]
    
    # Build basis matrix
    basis = _tps_basis_matrix(X, knots, n_dims, m)
    
    # Build penalty matrix
    n_poly = n_dims + 1  # For m=2: 1 + d terms
    penalty = _tps_penalty_matrix(n_knots, n_poly)
    
    # Select or use provided lambda
    if method == "fixed":
        if lambda_ is None:
            raise ValueError("Must provide lambda_ when method='fixed'")
        selected_lambda = lambda_
        criterion_value = None
    elif method == "GCV":
        # Optimize lambda using GCV
        def objective(log_lambda):
            return _gcv_score_tps(basis, y, w, penalty, np.exp(log_lambda))
        
        result_opt = minimize_scalar(
            objective,
            bounds=(-10, 10),
            method='bounded',
        )
        selected_lambda = np.exp(result_opt.x)
        criterion_value = result_opt.fun
    elif method in ("REML", "ML"):
        raise NotImplementedError(f"Method {method} not yet implemented for TPS")
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Fit with selected lambda
    coefficients, fitted, edf = _penalized_ls_tps(
        basis, y, w, penalty, selected_lambda
    )
    
    return TPSResult(
        coefficients=coefficients,
        fitted_values=fitted,
        lambda_=selected_lambda,
        edf=edf,
        knots=knots,
        X_train=X,
        n_dims=n_dims,
        m=m,
        selection_method=method,
        criterion_value=criterion_value,
        basis_matrix=basis,
        penalty_matrix=penalty,
    )


def tps(
    *variables: str,
    k: Optional[int] = None,
    m: int = 2,
    method: Literal["GCV", "REML", "ML"] = "GCV",
    knot_method: Literal["uniform", "kmeans", "all"] = "uniform",
) -> dict:
    """Thin plate spline smooth term for use in GAMLSS formulas.
    
    This function creates a specification for a TPS smooth term
    that can be used in GAMLSS model formulas.
    
    Parameters
    ----------
    *variables : str
        Variable names to smooth (2 or more for multidimensional TPS)
    k : int, optional
        Number of knots (basis functions)
    m : int
        Order of penalty (default: 2)
    method : str
        Smoothing parameter selection method
    knot_method : str
        Knot selection method
        
    Returns
    -------
    dict
        TPS specification dictionary
        
    Examples
    --------
    >>> # In a GAMLSS formula (future integration)
    >>> # model = gamlss("y ~ tps(x1, x2)", family=NO(), data=data)
    >>> 
    >>> # Current usage: direct fitting
    >>> from omnilss.smoothers.tps import fit_tps
    >>> result = fit_tps(X, y, k=20)
    
    Notes
    -----
    This function is a placeholder for future integration with
    the GAMLSS formula interface. Currently, use `fit_tps()` directly.
    """
    if len(variables) < 2:
        raise ValueError("TPS requires at least 2 variables")
    
    return {
        "type": "tps",
        "variables": variables,
        "k": k,
        "m": m,
        "method": method,
        "knot_method": knot_method,
    }
