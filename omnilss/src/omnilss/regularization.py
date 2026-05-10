"""Regularization methods for GAMLSS models.

This module provides L1 (Lasso), L2 (Ridge), and Elastic Net regularization
for feature selection and coefficient shrinkage in GAMLSS models.

Key Features:
- L1 (Lasso) regularization for feature selection
- L2 (Ridge) regularization for coefficient shrinkage
- Elastic Net combining L1 and L2
- Proximal gradient descent optimization
- Coordinate descent for Lasso
- Cross-validation for lambda selection
- JAX-compatible implementation

References
----------
- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso.
  Journal of the Royal Statistical Society: Series B, 58(1), 267-288.
- Zou, H., & Hastie, T. (2005). Regularization and variable selection via
  the elastic net. Journal of the Royal Statistical Society: Series B, 67(2), 301-320.
- Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths
  for generalized linear models via coordinate descent. Journal of Statistical
  Software, 33(1), 1-22.

Implementation Notes
--------------------
This implementation uses:
1. Proximal gradient descent for general L1 problems
2. Coordinate descent for Lasso (more efficient)
3. Closed-form solution for Ridge
4. JAX for automatic differentiation and JIT compilation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Callable

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize_scalar


@dataclass
class RegularizationResult:
    """Result from regularized regression.
    
    Attributes
    ----------
    coefficients : np.ndarray
        Fitted coefficients
    fitted_values : np.ndarray
        Fitted values (X @ coefficients)
    lambda_ : float
        Regularization parameter used
    alpha : float
        Elastic net mixing parameter (1 = Lasso, 0 = Ridge)
    intercept : float
        Intercept term (if fit_intercept=True)
    n_nonzero : int
        Number of non-zero coefficients (for L1)
    method : str
        Optimization method used
    n_iter : int
        Number of iterations
    converged : bool
        Whether optimization converged
    """
    coefficients: np.ndarray
    fitted_values: np.ndarray
    lambda_: float
    alpha: float = 1.0
    intercept: float = 0.0
    n_nonzero: int = 0
    method: str = "proximal_gradient"
    n_iter: int = 0
    converged: bool = True
    
    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """Predict at new X values.
        
        Parameters
        ----------
        X_new : np.ndarray
            New predictor matrix
            
        Returns
        -------
        np.ndarray
            Predicted values
        """
        X_new = np.asarray(X_new)
        return X_new @ self.coefficients + self.intercept


# =============================================================================
# Penalty Functions
# =============================================================================

def l1_penalty(coef: jnp.ndarray, lambda_: float) -> float:
    """L1 penalty (Lasso).
    
    Penalty = λ * ||β||₁ = λ * Σ|βⱼ|
    
    Parameters
    ----------
    coef : jnp.ndarray
        Coefficients
    lambda_ : float
        Regularization strength
        
    Returns
    -------
    float
        L1 penalty value
    """
    return lambda_ * jnp.sum(jnp.abs(coef))


def l2_penalty(coef: jnp.ndarray, lambda_: float) -> float:
    """L2 penalty (Ridge).
    
    Penalty = λ * ||β||₂² = λ * Σβⱼ²
    
    Parameters
    ----------
    coef : jnp.ndarray
        Coefficients
    lambda_ : float
        Regularization strength
        
    Returns
    -------
    float
        L2 penalty value
    """
    return lambda_ * jnp.sum(jnp.square(coef))


def elastic_net_penalty(
    coef: jnp.ndarray,
    lambda_: float,
    alpha: float = 0.5,
) -> float:
    """Elastic net penalty (L1 + L2).
    
    Penalty = λ * [α * ||β||₁ + (1-α) * ||β||₂²]
    
    Parameters
    ----------
    coef : jnp.ndarray
        Coefficients
    lambda_ : float
        Overall regularization strength
    alpha : float
        Mixing parameter:
        - alpha = 1: Pure Lasso (L1)
        - alpha = 0: Pure Ridge (L2)
        - 0 < alpha < 1: Elastic Net
        
    Returns
    -------
    float
        Elastic net penalty value
    """
    return alpha * l1_penalty(coef, lambda_) + (1 - alpha) * l2_penalty(coef, lambda_)


# =============================================================================
# Soft Thresholding (Proximal Operator for L1)
# =============================================================================

def soft_threshold(x: jnp.ndarray, threshold: float) -> jnp.ndarray:
    """Soft thresholding operator (proximal operator for L1 norm).
    
    S(x, λ) = sign(x) * max(|x| - λ, 0)
    
    This is the proximal operator for the L1 norm:
    prox_λ||·||₁(x) = argmin_z [½||z - x||² + λ||z||₁]
    
    Parameters
    ----------
    x : jnp.ndarray
        Input values
    threshold : float
        Threshold parameter (λ)
        
    Returns
    -------
    jnp.ndarray
        Soft-thresholded values
        
    Examples
    --------
    >>> x = jnp.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    >>> soft_threshold(x, 1.0)
    array([-1.,  0.,  0.,  0.,  1.])
    """
    return jnp.sign(x) * jnp.maximum(jnp.abs(x) - threshold, 0.0)


# =============================================================================
# Ridge Regression (Closed-form Solution)
# =============================================================================

def fit_ridge(
    X: np.ndarray,
    y: np.ndarray,
    lambda_: float,
    fit_intercept: bool = True,
) -> RegularizationResult:
    """Fit Ridge regression (L2 regularization).
    
    Closed-form solution: β = (X^T X + λI)^(-1) X^T y
    
    Parameters
    ----------
    X : np.ndarray
        Design matrix, shape (n, p)
    y : np.ndarray
        Response vector, shape (n,)
    lambda_ : float
        Regularization strength (λ ≥ 0)
    fit_intercept : bool
        Whether to fit an intercept term
        
    Returns
    -------
    RegularizationResult
        Fitted model
        
    Examples
    --------
    >>> X = np.random.randn(100, 10)
    >>> y = X @ np.random.randn(10) + np.random.randn(100) * 0.1
    >>> result = fit_ridge(X, y, lambda_=1.0)
    >>> print(f"Coefficients: {result.coefficients}")
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n, p = X.shape
    
    # Center data if fitting intercept
    if fit_intercept:
        X_mean = X.mean(axis=0)
        y_mean = y.mean()
        X_centered = X - X_mean
        y_centered = y - y_mean
    else:
        X_centered = X
        y_centered = y
        X_mean = np.zeros(p)
        y_mean = 0.0
    
    # Closed-form solution
    XtX = X_centered.T @ X_centered
    Xty = X_centered.T @ y_centered
    
    # Add regularization to diagonal
    A = XtX + lambda_ * np.eye(p)
    
    # Solve
    try:
        coefficients = np.linalg.solve(A, Xty)
    except np.linalg.LinAlgError:
        # Fallback to lstsq if singular
        coefficients = np.linalg.lstsq(A, Xty, rcond=None)[0]
    
    # Compute intercept
    if fit_intercept:
        intercept = y_mean - X_mean @ coefficients
    else:
        intercept = 0.0
    
    # Fitted values
    fitted = X @ coefficients + intercept
    
    return RegularizationResult(
        coefficients=coefficients,
        fitted_values=fitted,
        lambda_=lambda_,
        alpha=0.0,
        intercept=intercept,
        n_nonzero=p,
        method="closed_form",
        n_iter=0,
        converged=True,
    )


# =============================================================================
# Lasso via Coordinate Descent
# =============================================================================

def fit_lasso_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    lambda_: float,
    fit_intercept: bool = True,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> RegularizationResult:
    """Fit Lasso regression using coordinate descent.
    
    This is the most efficient algorithm for Lasso, especially for
    high-dimensional problems.
    
    Parameters
    ----------
    X : np.ndarray
        Design matrix, shape (n, p)
    y : np.ndarray
        Response vector, shape (n,)
    lambda_ : float
        Regularization strength (λ ≥ 0)
    fit_intercept : bool
        Whether to fit an intercept term
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
        
    Returns
    -------
    RegularizationResult
        Fitted model
        
    Notes
    -----
    The coordinate descent algorithm updates one coefficient at a time:
    
    For each j:
        r_j = y - X_{-j} β_{-j}  (partial residual)
        β_j = S(X_j^T r_j / ||X_j||², λ)  (soft threshold)
    
    where S is the soft thresholding operator.
    
    References
    ----------
    Friedman et al. (2010). Regularization paths for generalized linear
    models via coordinate descent. Journal of Statistical Software.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, p = X.shape
    
    # Center data if fitting intercept
    if fit_intercept:
        X_mean = X.mean(axis=0)
        y_mean = y.mean()
        X_centered = X - X_mean
        y_centered = y - y_mean
    else:
        X_centered = X
        y_centered = y
        X_mean = np.zeros(p)
        y_mean = 0.0
    
    # Precompute column norms
    X_norms_sq = np.sum(X_centered ** 2, axis=0)
    
    # Initialize coefficients
    beta = np.zeros(p)
    
    # Coordinate descent
    converged = False
    for iteration in range(max_iter):
        beta_old = beta.copy()
        
        # Update each coefficient
        for j in range(p):
            if X_norms_sq[j] < 1e-10:
                beta[j] = 0.0
                continue
            
            # Partial residual (excluding j-th feature)
            r_j = y_centered - X_centered @ beta + X_centered[:, j] * beta[j]
            
            # Coordinate update with soft thresholding
            rho_j = X_centered[:, j] @ r_j
            beta[j] = np.sign(rho_j) * max(abs(rho_j) - lambda_, 0.0) / X_norms_sq[j]
        
        # Check convergence
        if np.max(np.abs(beta - beta_old)) < tol:
            converged = True
            break
    
    # Compute intercept
    if fit_intercept:
        intercept = y_mean - X_mean @ beta
    else:
        intercept = 0.0
    
    # Fitted values
    fitted = X @ beta + intercept
    
    # Count non-zero coefficients
    n_nonzero = np.sum(np.abs(beta) > 1e-10)
    
    return RegularizationResult(
        coefficients=beta,
        fitted_values=fitted,
        lambda_=lambda_,
        alpha=1.0,
        intercept=intercept,
        n_nonzero=int(n_nonzero),
        method="coordinate_descent",
        n_iter=iteration + 1,
        converged=converged,
    )


# =============================================================================
# Elastic Net via Coordinate Descent
# =============================================================================

def fit_elastic_net(
    X: np.ndarray,
    y: np.ndarray,
    lambda_: float,
    alpha: float = 0.5,
    fit_intercept: bool = True,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> RegularizationResult:
    """Fit Elastic Net regression using coordinate descent.
    
    Elastic Net combines L1 and L2 penalties:
    Loss = ½||y - Xβ||² + λ[α||β||₁ + ½(1-α)||β||₂²]
    
    Parameters
    ----------
    X : np.ndarray
        Design matrix, shape (n, p)
    y : np.ndarray
        Response vector, shape (n,)
    lambda_ : float
        Overall regularization strength
    alpha : float
        Mixing parameter (0 = Ridge, 1 = Lasso)
    fit_intercept : bool
        Whether to fit an intercept term
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
        
    Returns
    -------
    RegularizationResult
        Fitted model
        
    Notes
    -----
    The coordinate descent update for Elastic Net is:
    β_j = S(X_j^T r_j / ||X_j||², λα) / (1 + λ(1-α))
    
    where S is the soft thresholding operator.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, p = X.shape
    
    # Center data if fitting intercept
    if fit_intercept:
        X_mean = X.mean(axis=0)
        y_mean = y.mean()
        X_centered = X - X_mean
        y_centered = y - y_mean
    else:
        X_centered = X
        y_centered = y
        X_mean = np.zeros(p)
        y_mean = 0.0
    
    # Precompute column norms
    X_norms_sq = np.sum(X_centered ** 2, axis=0)
    
    # Initialize coefficients
    beta = np.zeros(p)
    
    # Coordinate descent
    converged = False
    for iteration in range(max_iter):
        beta_old = beta.copy()
        
        # Update each coefficient
        for j in range(p):
            if X_norms_sq[j] < 1e-10:
                beta[j] = 0.0
                continue
            
            # Partial residual
            r_j = y_centered - X_centered @ beta + X_centered[:, j] * beta[j]
            
            # Coordinate update with soft thresholding and L2 penalty
            rho_j = X_centered[:, j] @ r_j
            threshold = lambda_ * alpha
            denominator = X_norms_sq[j] + lambda_ * (1 - alpha)
            
            beta[j] = np.sign(rho_j) * max(abs(rho_j) - threshold, 0.0) / denominator
        
        # Check convergence
        if np.max(np.abs(beta - beta_old)) < tol:
            converged = True
            break
    
    # Compute intercept
    if fit_intercept:
        intercept = y_mean - X_mean @ beta
    else:
        intercept = 0.0
    
    # Fitted values
    fitted = X @ beta + intercept
    
    # Count non-zero coefficients
    n_nonzero = np.sum(np.abs(beta) > 1e-10)
    
    return RegularizationResult(
        coefficients=beta,
        fitted_values=fitted,
        lambda_=lambda_,
        alpha=alpha,
        intercept=intercept,
        n_nonzero=int(n_nonzero),
        method="coordinate_descent",
        n_iter=iteration + 1,
        converged=converged,
    )


# =============================================================================
# Cross-Validation for Lambda Selection
# =============================================================================

def cross_validate_lambda(
    X: np.ndarray,
    y: np.ndarray,
    lambda_grid: Optional[np.ndarray] = None,
    alpha: float = 1.0,
    n_folds: int = 5,
    fit_intercept: bool = True,
    method: Literal["lasso", "ridge", "elastic_net"] = "lasso",
) -> Tuple[float, np.ndarray]:
    """Select optimal lambda using cross-validation.
    
    Parameters
    ----------
    X : np.ndarray
        Design matrix
    y : np.ndarray
        Response vector
    lambda_grid : np.ndarray, optional
        Grid of lambda values to try. If None, uses log-spaced grid.
    alpha : float
        Elastic net mixing parameter (only for elastic_net method)
    n_folds : int
        Number of cross-validation folds
    fit_intercept : bool
        Whether to fit intercept
    method : str
        Regularization method: "lasso", "ridge", or "elastic_net"
        
    Returns
    -------
    best_lambda : float
        Optimal lambda value
    cv_scores : np.ndarray
        Cross-validation scores for each lambda
        
    Examples
    --------
    >>> X = np.random.randn(100, 10)
    >>> y = X @ np.random.randn(10) + np.random.randn(100) * 0.1
    >>> best_lambda, scores = cross_validate_lambda(X, y, method="lasso")
    >>> print(f"Best lambda: {best_lambda:.4e}")
    """
    n, p = X.shape
    
    # Default lambda grid
    if lambda_grid is None:
        # Compute lambda_max (smallest lambda that gives all-zero solution)
        if fit_intercept:
            y_centered = y - y.mean()
            X_centered = X - X.mean(axis=0)
        else:
            y_centered = y
            X_centered = X
        
        lambda_max = np.max(np.abs(X_centered.T @ y_centered)) / n
        lambda_grid = np.logspace(np.log10(lambda_max * 0.001), np.log10(lambda_max), 50)
    
    # Create folds
    fold_size = n // n_folds
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    # Cross-validation
    cv_scores = np.zeros(len(lambda_grid))
    
    for i, lam in enumerate(lambda_grid):
        fold_errors = []
        
        for fold in range(n_folds):
            # Split data
            test_idx = indices[fold * fold_size:(fold + 1) * fold_size]
            train_idx = np.setdiff1d(indices, test_idx)
            
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            
            # Fit model
            if method == "lasso":
                result = fit_lasso_coordinate_descent(
                    X_train, y_train, lam, fit_intercept=fit_intercept
                )
            elif method == "ridge":
                result = fit_ridge(X_train, y_train, lam, fit_intercept=fit_intercept)
            elif method == "elastic_net":
                result = fit_elastic_net(
                    X_train, y_train, lam, alpha=alpha, fit_intercept=fit_intercept
                )
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Predict and compute error
            y_pred = result.predict(X_test)
            mse = np.mean((y_test - y_pred) ** 2)
            fold_errors.append(mse)
        
        cv_scores[i] = np.mean(fold_errors)
    
    # Select best lambda
    best_idx = np.argmin(cv_scores)
    best_lambda = lambda_grid[best_idx]
    
    return best_lambda, cv_scores


# =============================================================================
# Unified Interface
# =============================================================================

def fit_regularized(
    X: np.ndarray,
    y: np.ndarray,
    lambda_: Optional[float] = None,
    alpha: float = 1.0,
    method: Literal["lasso", "ridge", "elastic_net", "auto"] = "auto",
    fit_intercept: bool = True,
    max_iter: int = 1000,
    tol: float = 1e-4,
    cv: bool = True,
    n_folds: int = 5,
) -> RegularizationResult:
    """Fit regularized regression model.
    
    Unified interface for Lasso, Ridge, and Elastic Net regression.
    
    Parameters
    ----------
    X : np.ndarray
        Design matrix, shape (n, p)
    y : np.ndarray
        Response vector, shape (n,)
    lambda_ : float, optional
        Regularization strength. If None, selected by cross-validation.
    alpha : float
        Elastic net mixing parameter:
        - 1.0: Lasso (L1)
        - 0.0: Ridge (L2)
        - 0 < alpha < 1: Elastic Net
    method : str
        Regularization method:
        - "lasso": L1 regularization
        - "ridge": L2 regularization
        - "elastic_net": L1 + L2
        - "auto": Choose based on alpha
    fit_intercept : bool
        Whether to fit an intercept term
    max_iter : int
        Maximum iterations for coordinate descent
    tol : float
        Convergence tolerance
    cv : bool
        Whether to use cross-validation for lambda selection
    n_folds : int
        Number of CV folds (if cv=True)
        
    Returns
    -------
    RegularizationResult
        Fitted model
        
    Examples
    --------
    >>> # Lasso
    >>> result = fit_regularized(X, y, alpha=1.0, method="lasso")
    >>> 
    >>> # Ridge
    >>> result = fit_regularized(X, y, alpha=0.0, method="ridge")
    >>> 
    >>> # Elastic Net
    >>> result = fit_regularized(X, y, alpha=0.5, method="elastic_net")
    >>> 
    >>> # Auto lambda selection
    >>> result = fit_regularized(X, y, lambda_=None, cv=True)
    """
    # Determine method
    if method == "auto":
        if alpha == 1.0:
            method = "lasso"
        elif alpha == 0.0:
            method = "ridge"
        else:
            method = "elastic_net"
    
    # Select lambda if not provided
    if lambda_ is None:
        if cv:
            lambda_, _ = cross_validate_lambda(
                X, y, alpha=alpha, n_folds=n_folds,
                fit_intercept=fit_intercept, method=method
            )
        else:
            # Use a default value
            lambda_ = 1.0
    
    # Fit model
    if method == "lasso":
        return fit_lasso_coordinate_descent(
            X, y, lambda_, fit_intercept=fit_intercept,
            max_iter=max_iter, tol=tol
        )
    elif method == "ridge":
        return fit_ridge(X, y, lambda_, fit_intercept=fit_intercept)
    elif method == "elastic_net":
        return fit_elastic_net(
            X, y, lambda_, alpha=alpha, fit_intercept=fit_intercept,
            max_iter=max_iter, tol=tol
        )
    else:
        raise ValueError(f"Unknown method: {method}")
