"""Fisher Information Matrix computation for GAMLSS models.

This module provides functions to compute Fisher information matrices
and score vectors for GAMLSS models, which are essential for the
Cole-Green (CG) fitting algorithm.

The Fisher information matrix measures the amount of information that
an observable random variable carries about an unknown parameter. In
GAMLSS, we compute the Fisher information for all distribution parameters
(μ, σ, ν, τ) simultaneously.

Key Functions:
- compute_fisher_matrix: Compute full Fisher information matrix
- compute_score_vector: Compute score (gradient) vector
- compute_observed_information: Compute observed information matrix

References:
    Cole, T. J., & Green, P. J. (1992). Smoothing reference centile 
    curves: the LMS method and penalized likelihood. Statistics in 
    medicine, 11(10), 1305-1319.

Examples:
    >>> from omnilss.fisher_information import compute_fisher_matrix
    >>> import jax.numpy as jnp
    >>> 
    >>> # Define log-likelihood function
    >>> def log_lik(params, data):
    ...     mu = params["mu"]
    ...     sigma = jnp.exp(params["log_sigma"])
    ...     y = data["y"]
    ...     # Normal log-likelihood
    ...     return -0.5 * jnp.sum(((y - mu) / sigma)**2 + jnp.log(2 * jnp.pi * sigma**2))
    >>> 
    >>> # Compute Fisher matrix
    >>> params = {"mu": jnp.array(0.0), "log_sigma": jnp.array(0.0)}
    >>> data = {"y": jnp.array([1.0, 2.0, 3.0])}
    >>> fisher = compute_fisher_matrix(log_lik, params, data)
"""

from __future__ import annotations

from typing import Callable, Dict, Any, Tuple, Optional
import warnings

import jax
import jax.numpy as jnp
from jax import grad, hessian, jacfwd, jacrev


# Type aliases
Params = Dict[str, jnp.ndarray]
Data = Dict[str, Any]
LogLikFn = Callable[[Params, Data], float]


def compute_fisher_matrix(
    log_likelihood: LogLikFn,
    params: Params,
    data: Data,
    method: str = "observed"
) -> jnp.ndarray:
    """Compute Fisher information matrix.
    
    The Fisher information matrix measures the curvature of the log-likelihood
    surface. For GAMLSS, we typically use the observed Fisher information
    (negative Hessian of log-likelihood).
    
    Parameters
    ----------
    log_likelihood : Callable
        Log-likelihood function with signature log_lik(params, data) -> scalar
    params : Dict[str, jnp.ndarray]
        Current parameter values, e.g., {"mu": ..., "sigma": ..., "nu": ..., "tau": ...}
    data : Dict[str, Any]
        Data dictionary containing observations and design matrices
    method : str, default="observed"
        Method to compute Fisher information:
        - "observed": Use observed information (negative Hessian)
        - "expected": Use expected information (not yet implemented)
        
    Returns
    -------
    fisher_matrix : jnp.ndarray
        Fisher information matrix of shape (n_params, n_params)
        where n_params is the total number of parameters
        
    Notes
    -----
    The observed Fisher information is:
        I_obs = -H(θ)
    where H is the Hessian matrix of the log-likelihood.
    
    For numerical stability, we add a small regularization term to
    ensure the matrix is positive definite.
    
    Examples
    --------
    >>> def log_lik(params, data):
    ...     mu = params["mu"]
    ...     y = data["y"]
    ...     return -0.5 * jnp.sum((y - mu)**2)
    >>> 
    >>> params = {"mu": jnp.array(0.0)}
    >>> data = {"y": jnp.array([1.0, 2.0, 3.0])}
    >>> fisher = compute_fisher_matrix(log_lik, params, data)
    >>> print(fisher.shape)
    (1, 1)
    """
    if method == "observed":
        return compute_observed_information(log_likelihood, params, data)
    elif method == "expected":
        raise NotImplementedError("Expected Fisher information not yet implemented")
    else:
        raise ValueError(f"Unknown method: {method}. Use 'observed' or 'expected'")


def compute_observed_information(
    log_likelihood: LogLikFn,
    params: Params,
    data: Data,
    regularization: float = 1e-8
) -> jnp.ndarray:
    """Compute observed Fisher information (negative Hessian).
    
    Parameters
    ----------
    log_likelihood : Callable
        Log-likelihood function
    params : Dict[str, jnp.ndarray]
        Current parameters
    data : Dict[str, Any]
        Data
    regularization : float, default=1e-8
        Small value added to diagonal for numerical stability
        
    Returns
    -------
    observed_info : jnp.ndarray
        Observed information matrix
        
    Notes
    -----
    The observed information is the negative Hessian:
        I_obs = -∂²ℓ/∂θ∂θ'
    
    We add a small regularization term to ensure positive definiteness:
        I_reg = I_obs + ε * I
    """
    # Flatten parameters to vector
    param_vec, unflatten_fn = flatten_params(params)
    
    # Define log-likelihood as function of parameter vector
    def log_lik_vec(theta_vec):
        theta_dict = unflatten_fn(theta_vec)
        return log_likelihood(theta_dict, data)
    
    # Compute Hessian
    hess_fn = hessian(log_lik_vec)
    H = hess_fn(param_vec)
    
    # Observed information = -Hessian
    observed_info = -H
    
    # Add regularization for numerical stability
    n_params = observed_info.shape[0]
    observed_info = observed_info + regularization * jnp.eye(n_params)
    
    return observed_info


def compute_score_vector(
    log_likelihood: LogLikFn,
    params: Params,
    data: Data
) -> jnp.ndarray:
    """Compute score vector (gradient of log-likelihood).
    
    The score vector is the first derivative of the log-likelihood
    with respect to parameters. It indicates the direction of steepest
    ascent for the log-likelihood.
    
    Parameters
    ----------
    log_likelihood : Callable
        Log-likelihood function
    params : Dict[str, jnp.ndarray]
        Current parameters
    data : Dict[str, Any]
        Data
        
    Returns
    -------
    score : jnp.ndarray
        Score vector of shape (n_params,)
        
    Notes
    -----
    The score vector is:
        u(θ) = ∂ℓ/∂θ
    
    At the maximum likelihood estimate, the score is zero:
        u(θ̂) = 0
    
    Examples
    --------
    >>> def log_lik(params, data):
    ...     mu = params["mu"]
    ...     y = data["y"]
    ...     return -0.5 * jnp.sum((y - mu)**2)
    >>> 
    >>> params = {"mu": jnp.array(0.0)}
    >>> data = {"y": jnp.array([1.0, 2.0, 3.0])}
    >>> score = compute_score_vector(log_lik, params, data)
    >>> print(score)  # Should be [6.0] (sum of residuals)
    """
    # Flatten parameters
    param_vec, unflatten_fn = flatten_params(params)
    
    # Define log-likelihood as function of parameter vector
    def log_lik_vec(theta_vec):
        theta_dict = unflatten_fn(theta_vec)
        return log_likelihood(theta_dict, data)
    
    # Compute gradient
    grad_fn = grad(log_lik_vec)
    score = grad_fn(param_vec)
    
    return score


def flatten_params(params: Params) -> Tuple[jnp.ndarray, Callable]:
    """Flatten parameter dictionary to vector.
    
    Parameters
    ----------
    params : Dict[str, jnp.ndarray]
        Parameter dictionary
        
    Returns
    -------
    param_vec : jnp.ndarray
        Flattened parameter vector
    unflatten_fn : Callable
        Function to reconstruct dictionary from vector
        
    Examples
    --------
    >>> params = {"mu": jnp.array([1.0, 2.0]), "sigma": jnp.array([0.5])}
    >>> vec, unflatten = flatten_params(params)
    >>> print(vec)
    [1.0, 2.0, 0.5]
    >>> reconstructed = unflatten(vec)
    >>> print(reconstructed)
    {"mu": [1.0, 2.0], "sigma": [0.5]}
    """
    # Get parameter names in consistent order
    param_names = sorted(params.keys())
    
    # Store original shapes (including scalars)
    param_shapes = {name: jnp.asarray(params[name]).shape for name in param_names}
    
    # Flatten to vector
    param_list = [jnp.atleast_1d(params[name]).ravel() for name in param_names]
    param_vec = jnp.concatenate(param_list)
    
    # Create unflatten function
    param_sizes = {name: jnp.atleast_1d(params[name]).size for name in param_names}
    
    def unflatten_fn(vec: jnp.ndarray) -> Params:
        """Reconstruct parameter dictionary from vector."""
        result = {}
        start_idx = 0
        for name in param_names:
            size = param_sizes[name]
            shape = param_shapes[name]
            values = vec[start_idx:start_idx + size]
            # Preserve original shape (including scalars)
            if shape == ():
                result[name] = values[0]
            else:
                result[name] = values.reshape(shape)
            start_idx += size
        return result
    
    return param_vec, unflatten_fn


def check_fisher_matrix(
    fisher: jnp.ndarray,
    tol: float = 1e-10
) -> Dict[str, bool]:
    """Check properties of Fisher information matrix.
    
    Parameters
    ----------
    fisher : jnp.ndarray
        Fisher information matrix
    tol : float, default=1e-10
        Tolerance for numerical checks
        
    Returns
    -------
    checks : Dict[str, bool]
        Dictionary of check results:
        - "symmetric": Is matrix symmetric?
        - "positive_definite": Is matrix positive definite?
        - "well_conditioned": Is condition number reasonable?
        
    Examples
    --------
    >>> fisher = jnp.array([[2.0, 0.1], [0.1, 1.0]])
    >>> checks = check_fisher_matrix(fisher)
    >>> print(checks)
    {"symmetric": True, "positive_definite": True, "well_conditioned": True}
    """
    checks = {}
    
    # Check symmetry
    checks["symmetric"] = jnp.allclose(fisher, fisher.T, atol=tol)
    
    # Check positive definiteness (all eigenvalues > 0)
    eigenvalues = jnp.linalg.eigvalsh(fisher)
    checks["positive_definite"] = jnp.all(eigenvalues > tol)
    
    # Check condition number
    cond_number = jnp.linalg.cond(fisher)
    checks["well_conditioned"] = cond_number < 1e10
    checks["condition_number"] = float(cond_number)
    
    return checks


def compute_parameter_covariance(
    fisher: jnp.ndarray,
    regularization: float = 1e-8
) -> jnp.ndarray:
    """Compute parameter covariance matrix from Fisher information.
    
    The covariance matrix is the inverse of the Fisher information matrix.
    
    Parameters
    ----------
    fisher : jnp.ndarray
        Fisher information matrix
    regularization : float, default=1e-8
        Regularization for numerical stability
        
    Returns
    -------
    covariance : jnp.ndarray
        Parameter covariance matrix
        
    Notes
    -----
    The asymptotic covariance of the MLE is:
        Cov(θ̂) = I(θ)^(-1)
    
    Standard errors are the square roots of the diagonal elements.
    
    Examples
    --------
    >>> fisher = jnp.array([[4.0, 0.0], [0.0, 2.0]])
    >>> cov = compute_parameter_covariance(fisher)
    >>> se = jnp.sqrt(jnp.diag(cov))
    >>> print(se)  # [0.5, 0.707...]
    """
    # Add regularization
    n_params = fisher.shape[0]
    fisher_reg = fisher + regularization * jnp.eye(n_params)
    
    # Compute inverse (covariance)
    try:
        covariance = jnp.linalg.inv(fisher_reg)
    except:
        # If inversion fails, use pseudo-inverse
        warnings.warn(
            "Fisher matrix is singular, using pseudo-inverse",
            RuntimeWarning
        )
        covariance = jnp.linalg.pinv(fisher_reg)
    
    return covariance


def compute_standard_errors(
    fisher: jnp.ndarray,
    regularization: float = 1e-8
) -> jnp.ndarray:
    """Compute standard errors from Fisher information.
    
    Parameters
    ----------
    fisher : jnp.ndarray
        Fisher information matrix
    regularization : float, default=1e-8
        Regularization for numerical stability
        
    Returns
    -------
    standard_errors : jnp.ndarray
        Standard errors for each parameter
        
    Examples
    --------
    >>> fisher = jnp.array([[4.0, 0.0], [0.0, 2.0]])
    >>> se = compute_standard_errors(fisher)
    >>> print(se)  # [0.5, 0.707...]
    """
    covariance = compute_parameter_covariance(fisher, regularization)
    standard_errors = jnp.sqrt(jnp.diag(covariance))
    return standard_errors


__all__ = [
    "compute_fisher_matrix",
    "compute_observed_information",
    "compute_score_vector",
    "flatten_params",
    "check_fisher_matrix",
    "compute_parameter_covariance",
    "compute_standard_errors",
]
