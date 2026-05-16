"""JIT-compiled helper functions for GAMLSS fitting.

This module provides JIT-compiled versions of computationally intensive
operations to significantly improve performance, especially for distributions
like ZAGA and BE that have complex score/hessian calculations.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np


def create_jit_iwls_step(
    score_func: Callable,
    hessian_func: Callable,
    link_derivative: Callable,
    is_mixed: bool = False,
) -> Callable:
    """Create a JIT-compiled IWLS step function for a parameter.
    
    Parameters
    ----------
    score_func : Callable
        Score function (first derivative of log-likelihood)
    hessian_func : Callable
        Hessian function (second derivative of log-likelihood)
    link_derivative : Callable
        Derivative of the link function
    is_mixed : bool
        Whether this is a mixed distribution (zero-inflated/altered)
        
    Returns
    -------
    Callable
        JIT-compiled function that computes IWLS step
    """
    
    @jax.jit
    def iwls_step(y, eta, params_dict, w):
        """Compute one IWLS step.
        
        Parameters
        ----------
        y : jnp.ndarray
            Response variable
        eta : jnp.ndarray
            Current linear predictor
        params_dict : dict
            Dictionary of all distribution parameters
        w : jnp.ndarray
            Observation weights
            
        Returns
        -------
        tuple
            (working_response, iterative_weights)
        """
        eps = jnp.finfo(jnp.float64).eps
        
        # Compute score and hessian
        score = score_func(**params_dict)
        hess = hessian_func(**params_dict)
        
        # Enhanced numerical stability for mixed distributions
        score = jnp.where(jnp.isfinite(score), score, 0.0)
        hess = jnp.where(jnp.isfinite(hess), hess, -1e-10)
        
        # Add floor to prevent division by zero
        if is_mixed:
            hess = jnp.where(jnp.abs(hess) < 1e-8, -1e-8, hess)
        else:
            hess = jnp.where(jnp.abs(hess) < 1e-10, -1e-10, hess)
        
        # Compute link derivative with safety checks
        dr = link_derivative(eta)
        dr = jnp.where(jnp.isfinite(dr), dr, eps)
        dr = jnp.where(jnp.abs(dr) < eps, eps, dr)
        
        # Compute deta/dparam
        deta_dparam = 1.0 / dr
        deta_dparam = jnp.where(jnp.isfinite(deta_dparam), deta_dparam, 1.0)
        
        # Compute iterative weights
        deta_dparam_sq = jnp.square(deta_dparam)
        deta_dparam_sq = jnp.where(deta_dparam_sq < eps, eps, deta_dparam_sq)
        
        iter_w = -(hess / deta_dparam_sq)
        iter_w = jnp.where(jnp.isfinite(iter_w), iter_w, 1e-10)
        
        # More conservative clipping for mixed distributions
        if is_mixed:
            iter_w = jnp.clip(iter_w, 1e-8, 1e8)
        else:
            iter_w = jnp.clip(iter_w, 1e-10, 1e10)
        
        # Compute working response
        denominator = deta_dparam * iter_w
        denominator = jnp.where(jnp.abs(denominator) < eps, eps, denominator)
        z = eta + score / denominator
        z = jnp.where(jnp.isfinite(z), z, eta)
        
        return z, iter_w
    
    return iwls_step


def create_jit_wls_solver() -> Callable:
    """Create a JIT-compiled weighted least squares solver.
    
    Returns
    -------
    Callable
        JIT-compiled WLS solver
    """
    
    @jax.jit
    def wls_solve(X, z, w):
        """Solve weighted least squares: argmin_beta ||sqrt(w) * (z - X*beta)||^2
        
        Parameters
        ----------
        X : jnp.ndarray
            Design matrix (n x p)
        z : jnp.ndarray
            Working response (n,)
        w : jnp.ndarray
            Weights (n,)
            
        Returns
        -------
        jnp.ndarray
            Coefficient vector (p,)
        """
        # Compute sqrt(w) once
        sqrt_w = jnp.sqrt(w)
        
        # Weight the design matrix and response
        X_weighted = X * sqrt_w[:, None]
        z_weighted = z * sqrt_w
        
        # Solve using QR decomposition (more stable than normal equations)
        beta, _, _, _ = jnp.linalg.lstsq(X_weighted, z_weighted, rcond=None)
        
        return beta
    
    return wls_solve


def create_jit_deviance_computer(g_dev_inc_func: Callable) -> Callable:
    """Create a JIT-compiled deviance computation function.
    
    Parameters
    ----------
    g_dev_inc_func : Callable
        Incremental deviance function from family
        
    Returns
    -------
    Callable
        JIT-compiled deviance computer
    """
    
    @jax.jit
    def compute_deviance(y, params_dict, w):
        """Compute weighted deviance.
        
        Parameters
        ----------
        y : jnp.ndarray
            Response variable
        params_dict : dict
            Dictionary of all distribution parameters
        w : jnp.ndarray
            Observation weights
            
        Returns
        -------
        float
            Total deviance
        """
        dev_inc = g_dev_inc_func(**params_dict)
        return jnp.sum(dev_inc * w)
    
    return compute_deviance


def create_jit_parameter_updater(
    link_inverse: Callable,
) -> Callable:
    """Create a JIT-compiled parameter updater.
    
    Parameters
    ----------
    link_inverse : Callable
        Inverse link function
        
    Returns
    -------
    Callable
        JIT-compiled parameter updater
    """
    
    @jax.jit
    def update_parameter(X, beta):
        """Update parameter from coefficients.
        
        Parameters
        ----------
        X : jnp.ndarray
            Design matrix
        beta : jnp.ndarray
            Coefficient vector
            
        Returns
        -------
        tuple
            (eta, parameter_value)
        """
        eta = X @ beta
        param = link_inverse(eta)
        return eta, param
    
    return update_parameter


def create_jit_full_iteration(
    family: Any,
    parameter_names: tuple[str, ...],
    is_mixed: bool = False,
) -> Callable:
    """Create a JIT-compiled full GAMLSS iteration.
    
    This is the most aggressive optimization - compiling the entire
    iteration loop. Use with caution as it requires static shapes.
    
    Parameters
    ----------
    family : FamilyDefinition
        Distribution family
    parameter_names : tuple
        Names of parameters to estimate
    is_mixed : bool
        Whether this is a mixed distribution
        
    Returns
    -------
    Callable
        JIT-compiled iteration function
    """
    
    # Create JIT-compiled helpers for each parameter
    iwls_steps = {}
    param_updaters = {}
    
    for param in parameter_names:
        iwls_steps[param] = create_jit_iwls_step(
            family.score_functions[param],
            family.hessian_functions[param],
            family.link_derivatives[param],
            is_mixed=is_mixed,
        )
        param_updaters[param] = create_jit_parameter_updater(
            family.link_inverses[param],
        )
    
    wls_solver = create_jit_wls_solver()
    deviance_computer = create_jit_deviance_computer(family.g_dev_inc)
    
    @jax.jit
    def full_iteration(y, design_matrices, betas, etas, params, w):
        """Perform one full GAMLSS iteration.
        
        Parameters
        ----------
        y : jnp.ndarray
            Response variable
        design_matrices : dict
            Design matrices for each parameter
        betas : dict
            Current coefficient vectors
        etas : dict
            Current linear predictors
        params : dict
            Current parameter values
        w : jnp.ndarray
            Observation weights
            
        Returns
        -------
        tuple
            (new_betas, new_etas, new_params, deviance)
        """
        new_betas = {}
        new_etas = {}
        new_params = {}
        
        # Update each parameter
        for param in parameter_names:
            # Prepare parameter dictionary for this parameter's update
            params_dict = {"y": y, **params}
            
            # Compute IWLS step
            z, iter_w = iwls_steps[param](y, etas[param], params_dict, w)
            
            # Solve WLS
            X = design_matrices[param]
            beta = wls_solver(X, z, iter_w * w)
            
            # Update parameter
            eta, param_value = param_updaters[param](X, beta)
            
            new_betas[param] = beta
            new_etas[param] = eta
            new_params[param] = param_value
        
        # Compute deviance
        params_dict = {"y": y, **new_params}
        dev = deviance_computer(y, params_dict, w)
        
        return new_betas, new_etas, new_params, dev
    
    return full_iteration


# Convenience function to check if JIT optimization is beneficial
def should_use_jit(family_name: str, n_obs: int) -> bool:
    """Determine if JIT optimization should be used.
    
    Parameters
    ----------
    family_name : str
        Name of the distribution family
    n_obs : int
        Number of observations
        
    Returns
    -------
    bool
        True if JIT optimization is recommended
    
    Notes
    -----
    JIT compilation is beneficial for:
    1. Complex distributions with expensive score/hessian calculations
    2. Larger datasets (n > 100) where compilation overhead is amortized
    3. Mixed distributions (zero-inflated/altered) with conditional logic
    
    For simple distributions (NO, PO, BI) or small datasets (n < 100),
    the JIT compilation overhead may outweigh the benefits.
    """
    # Mixed distributions (zero-inflated/altered)
    mixed_families = {
        "ZAGA", "ZAIG",  # Zero-adjusted
        "BEINF", "BEOI", "BEZI",  # Beta inflated
        "ZIP", "ZIP2", "ZINBI", "ZAP",  # Zero-inflated Poisson
        "ZALG",  # Zero-adjusted logarithmic
    }
    
    # Complex continuous distributions
    complex_continuous = {
        "BE",  # Beta (complex score/hessian)
        "GB2", "GG",  # Generalized distributions
        "SHASH", "SHASHo", "SHASHo2",  # Sinh-arcsinh
        "NET",  # Normal-exponential-t
        "GT",  # Generalized t
        "JSU", "JSUo",  # Johnson's SU
        "BCT", "BCPE", "BCCGo",  # Box-Cox transformations
        "SEP1", "SEP2", "SEP3", "SEP4",  # Skew exponential power
        "ST1", "ST2", "ST3", "ST4", "ST5",  # Skew t
    }
    
    # Complex discrete distributions
    complex_discrete = {
        "BB", "BNB",  # Beta-binomial, Beta-negative binomial
        "PIG", "SICHEL",  # Poisson-inverse Gaussian, Sichel
        "DPO", "DEL",  # Double Poisson, Delaporte
        "WARING", "YULE",  # Waring, Yule
        "GPO",  # Generalized Poisson
    }
    
    # Combine all complex families
    complex_families = mixed_families | complex_continuous | complex_discrete
    
    # Use JIT for complex families with sufficient data
    # Threshold of 100 observations ensures compilation overhead is worthwhile
    return family_name in complex_families and n_obs > 100


def create_jit_rs_no_core(max_iter: int = 30, tol: float = 1e-6) -> Callable:
    """Experimental fully-jitted RS core loop for NO family (mu/sigma intercept).

    This is an isolated technical-debt sandbox for TASK-01-C.
    """

    def _core(y: jnp.ndarray):
        n = y.shape[0]
        X = jnp.ones((n, 1), dtype=jnp.float64)
        w = jnp.ones(n, dtype=jnp.float64)

        mu0 = jnp.mean(y)
        sigma0 = jnp.maximum(jnp.std(y), 1e-6)
        eta_mu0 = jnp.full((n,), mu0)
        eta_sigma0 = jnp.full((n,), jnp.log(sigma0))

        def irls_param(eta, other, is_mu: bool):
            def body(i, cur_eta):
                mu = jnp.where(is_mu, cur_eta, other)
                sigma = jnp.where(is_mu, jnp.exp(other), jnp.exp(cur_eta))
                if is_mu:
                    score = (y - mu) / (sigma**2)
                    hess = -jnp.ones_like(y) / (sigma**2)
                    link_deriv = jnp.ones_like(y)
                else:
                    score = -1.0 + ((y - mu) ** 2) / (sigma**2)
                    hess = -2.0 * ((y - mu) ** 2) / (sigma**2)
                    link_deriv = jnp.ones_like(y)
                hess = jnp.where(hess < -1e-12, hess, -1e-12)
                ww = -(hess / (link_deriv**2))
                ww = jnp.clip(ww, 1e-8, 1e8)
                z = cur_eta + score / (link_deriv * ww + 1e-12)
                beta = create_jit_wls_solver()(X, z, ww * w)
                return X @ beta

            return jax.lax.fori_loop(0, 5, body, eta)

        def cond_fn(state):
            gdev, gdev_old, it, eta_mu, eta_sigma = state
            return jnp.logical_and(jnp.abs(gdev_old - gdev) > tol, it < max_iter)

        def body_fn(state):
            gdev, _gdev_old, it, eta_mu, eta_sigma = state
            eta_mu_new = irls_param(eta_mu, eta_sigma, True)
            eta_sigma_new = irls_param(eta_sigma, eta_mu_new, False)
            mu = eta_mu_new
            sigma = jnp.exp(eta_sigma_new)
            dev = jnp.sum(-2.0 * (-jnp.log(sigma) - 0.5 * jnp.log(2 * jnp.pi) - 0.5 * ((y - mu) / sigma) ** 2))
            return (dev, gdev, it + 1, eta_mu_new, eta_sigma_new)

        init_mu = eta_mu0
        init_sigma = eta_sigma0
        mu = init_mu
        sigma = jnp.exp(init_sigma)
        gdev0 = jnp.sum(-2.0 * (-jnp.log(sigma) - 0.5 * jnp.log(2 * jnp.pi) - 0.5 * ((y - mu) / sigma) ** 2))
        init = (gdev0, gdev0 + 1.0, jnp.array(0), init_mu, init_sigma)
        gdev, _gdev_old, it, eta_mu, eta_sigma = jax.lax.while_loop(cond_fn, body_fn, init)
        return {"mu": eta_mu, "sigma": jnp.exp(eta_sigma), "g_dev": gdev, "iterations": it}

    return jax.jit(_core)
