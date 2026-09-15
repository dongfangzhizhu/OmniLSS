"""
Bootstrap methods for GAMLSS models using JAX vectorization.

This module implements efficient bootstrap methods that leverage JAX's vmap
and parallel computation capabilities, avoiding the inefficient loop-based
approaches common in R implementations.

References
----------
Efron, B., & Tibshirani, R. J. (1994). An introduction to the bootstrap.
CRC press.

Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized additive models
for location, scale and shape. Journal of the Royal Statistical Society:
Series C (Applied Statistics), 54(3), 507-554.
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional, Callable, Any
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap, jit


@dataclass
class BootstrapResult:
    """Results from bootstrap analysis.
    
    Attributes
    ----------
    coefficients : jnp.ndarray
        Bootstrap coefficient estimates (n_boots, n_params)
    fitted_values : jnp.ndarray
        Bootstrap fitted values (n_boots, n_obs)
    predictions : jnp.ndarray, optional
        Bootstrap predictions for new data (n_boots, n_new)
    lower_ci : jnp.ndarray
        Lower confidence interval bounds
    upper_ci : jnp.ndarray
        Upper confidence interval bounds
    mean : jnp.ndarray
        Mean of bootstrap estimates
    std : jnp.ndarray
        Standard deviation of bootstrap estimates
    n_boots : int
        Number of bootstrap iterations
    n_successful : int
        Number of successful fits (excluding NaN)
    alpha : float
        Significance level (e.g., 0.05 for 95% CI)
    method : str
        Bootstrap method used ('nonparametric' or 'parametric')
    """
    coefficients: jnp.ndarray
    fitted_values: Optional[jnp.ndarray] = None
    predictions: Optional[jnp.ndarray] = None
    lower_ci: Optional[jnp.ndarray] = None
    upper_ci: Optional[jnp.ndarray] = None
    mean: Optional[jnp.ndarray] = None
    std: Optional[jnp.ndarray] = None
    n_boots: int = 0
    n_successful: int = 0
    alpha: float = 0.05
    method: str = 'nonparametric'
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "Bootstrap Results",
            "=" * 60,
            f"Method: {self.method}",
            f"Number of bootstrap samples: {self.n_boots}",
            f"Successful fits: {self.n_successful} ({self.n_successful/self.n_boots*100 if self.n_boots else 0:.1f}%)",
            f"Confidence level: {(1-self.alpha)*100:.0f}%",
            "",
            "Coefficient Statistics:",
            "-" * 60,
        ]
        
        if self.mean is not None and self.std is not None:
            for i, (m, s) in enumerate(zip(self.mean, self.std)):
                ci_str = ""
                if self.lower_ci is not None and self.upper_ci is not None:
                    ci_str = f"  [{self.lower_ci[i]:.4f}, {self.upper_ci[i]:.4f}]"
                lines.append(f"Param {i:2d}: {m:8.4f} ± {s:8.4f}{ci_str}")
        
        return "\n".join(lines)


def generate_bootstrap_indices(
    key: jax.random.PRNGKey,
    n_obs: int,
    n_boots: int
) -> jnp.ndarray:
    """Generate bootstrap sample indices using JAX vectorization.
    
    This function generates all bootstrap indices at once, enabling
    efficient parallel computation.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random number generator key
    n_obs : int
        Number of observations in original data
    n_boots : int
        Number of bootstrap samples
        
    Returns
    -------
    jnp.ndarray
        Bootstrap indices of shape (n_boots, n_obs)
        
    Examples
    --------
    >>> key = jax.random.PRNGKey(42)
    >>> indices = generate_bootstrap_indices(key, n_obs=100, n_boots=1000)
    >>> indices.shape
    (1000, 100)
    """
    # Split key for each bootstrap sample
    keys = jax.random.split(key, n_boots)
    
    # Vectorized sampling: generate all indices at once
    def sample_indices(k):
        return jax.random.choice(k, n_obs, shape=(n_obs,), replace=True)
    
    # Use vmap for parallel sampling
    boot_indices = vmap(sample_indices)(keys)
    
    return boot_indices


def resample_data(
    data: Dict[str, jnp.ndarray],
    indices: jnp.ndarray
) -> Dict[str, jnp.ndarray]:
    """Resample data according to bootstrap indices.
    
    Parameters
    ----------
    data : dict
        Dictionary containing data arrays
    indices : jnp.ndarray
        Bootstrap indices (n_obs,)
        
    Returns
    -------
    dict
        Resampled data dictionary
    """
    return {k: v[indices] if isinstance(v, jnp.ndarray) else v 
            for k, v in data.items()}


def compute_confidence_intervals(
    boot_estimates: jnp.ndarray,
    alpha: float = 0.05,
    method: str = 'percentile',
    original_estimate: Optional[jnp.ndarray] = None,
    jackknife_estimates: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute bootstrap confidence intervals.
    
    Parameters
    ----------
    boot_estimates : jnp.ndarray
        Bootstrap estimates (n_boots, n_params)
    alpha : float, default=0.05
        Significance level (e.g., 0.05 for 95% CI)
    method : str, default='percentile'
        Method for computing CI:
        - 'percentile': Simple percentile method
        - 'bca': Bias-corrected and accelerated. Requires ``original_estimate``.
          Without jackknife estimates the acceleration is zero, giving a
          bias-corrected (BC), not fully accelerated BCa, interval.
    original_estimate : array_like, optional
        Finite statistic from the original data, shape (n_params,).
    jackknife_estimates : array_like, optional
        Complete finite leave-one-out statistics, shape (n_jackknife, n_params),
        with at least two rows. Failed jackknife fits cannot be discarded.
        
    Returns
    -------
    lower_ci : jnp.ndarray
        Lower confidence bounds
    upper_ci : jnp.ndarray
        Upper confidence bounds
        
    Notes
    -----
    The percentile method computes CI as the alpha/2 and 1-alpha/2
    quantiles of the bootstrap distribution.
    Bootstrap rows containing any non-finite value are excluded. If no rows
    remain, both bounds are NaN. Bias correction uses midranks for ties and
    clips extreme proportions to half a bootstrap observation. Constant
    jackknife columns use zero acceleration. This post-processing function
    uses data-dependent filtering and is not JIT-compatible.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    boot_estimates = jnp.asarray(boot_estimates)
    if boot_estimates.ndim != 2:
        raise ValueError("boot_estimates must be a two-dimensional array")
    if method not in ('percentile', 'bca'):
        raise ValueError(f"Unknown method: {method}")
    if method == 'bca':
        if original_estimate is None:
            raise ValueError("original_estimate is required for bca intervals")
        theta_hat = jnp.asarray(original_estimate)
        if theta_hat.shape != (boot_estimates.shape[1],):
            raise ValueError("original_estimate must have shape (n_params,)")
        if not bool(jnp.all(jnp.isfinite(theta_hat))):
            raise ValueError("original_estimate must be finite")
        if jackknife_estimates is not None:
            jack = jnp.asarray(jackknife_estimates)
            if jack.ndim != 2 or jack.shape[1] != boot_estimates.shape[1] or jack.shape[0] < 2:
                raise ValueError("jackknife_estimates must have shape (n_jackknife, n_params) with at least two rows")
            if not bool(jnp.all(jnp.isfinite(jack))):
                raise ValueError("jackknife_estimates must be finite")

    # Exclude failed fits as complete rows, preserving joint parameter samples.
    valid_mask = jnp.isfinite(boot_estimates).all(axis=1)
    valid_estimates = boot_estimates[valid_mask]
    if valid_estimates.shape[0] == 0:
        missing = jnp.full((boot_estimates.shape[1],), jnp.nan)
        return missing, missing
    
    if method == 'percentile':
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_ci = jnp.percentile(valid_estimates, lower_percentile, axis=0)
        upper_ci = jnp.percentile(valid_estimates, upper_percentile, axis=0)
        
    elif method == 'bca':
        proportion = (jnp.mean(valid_estimates < theta_hat, axis=0)
                      + 0.5 * jnp.mean(valid_estimates == theta_hat, axis=0))
        eps = 0.5 / valid_estimates.shape[0]
        z0 = jax.scipy.special.ndtri(jnp.clip(proportion, eps, 1 - eps))
        if jackknife_estimates is None:
            acceleration = jnp.zeros_like(z0)
        else:
            center = jnp.mean(jack, axis=0)
            delta = center - jack
            denominator = 6 * jnp.sum(delta ** 2, axis=0) ** 1.5
            acceleration = jnp.where(
                denominator > 0, jnp.sum(delta ** 3, axis=0) / denominator, 0.0
            )
        z_alpha = jax.scipy.special.ndtri(jnp.array([alpha / 2, 1 - alpha / 2]))
        adjusted = jax.scipy.stats.norm.cdf(
            z0[:, None] + (z0[:, None] + z_alpha[None, :]) /
            (1 - acceleration[:, None] * (z0[:, None] + z_alpha[None, :]))
        )
        lower_ci = vmap(lambda values, q: jnp.percentile(values, q))(valid_estimates.T, adjusted[:, 0] * 100)
        upper_ci = vmap(lambda values, q: jnp.percentile(values, q))(valid_estimates.T, adjusted[:, 1] * 100)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return lower_ci, upper_ci


def nonparametric_bootstrap(
    fit_function: Callable,
    data: Dict[str, jnp.ndarray],
    key: jax.random.PRNGKey,
    n_boots: int = 1000,
    alpha: float = 0.05,
    parallel: bool = True,
    handle_failures: bool = True,
    min_success_rate: float = 0.0,
) -> BootstrapResult:
    """Perform non-parametric bootstrap for GAMLSS models.
    
    This is the most robust bootstrap method as it doesn't rely on
    distributional assumptions. It resamples the original data with
    replacement and refits the model.
    
    Parameters
    ----------
    fit_function : callable
        Function that takes data dict and returns fitted parameters.
        Should have signature: fit_function(data) -> jnp.ndarray
    data : dict
        Dictionary containing original data (y, X, etc.)
    key : jax.random.PRNGKey
        Random number generator key
    n_boots : int, default=1000
        Number of bootstrap samples
    alpha : float, default=0.05
        Significance level for confidence intervals
    parallel : bool, default=True
        Whether to use vmap for parallel computation
    handle_failures : bool, default=True
        Whether to handle failed fits by marking them as NaN
        
    Returns
    -------
    BootstrapResult
        Bootstrap results including coefficients and confidence intervals
        
    Examples
    --------
    >>> def my_fit(data):
    ...     # Your GAMLSS fitting logic
    ...     return fitted_params
    >>> 
    >>> key = jax.random.PRNGKey(42)
    >>> result = nonparametric_bootstrap(my_fit, data, key, n_boots=1000)
    >>> print(result.summary())
    
    Notes
    -----
    This function uses JAX's vmap to parallelize bootstrap iterations,
    which is much faster than sequential loops. However, it requires
    more memory as all bootstrap samples are processed simultaneously.
    
    If memory is limited, consider using smaller n_boots or implementing
    a batched version using jax.lax.map.
    """
    if not 0 <= min_success_rate <= 1:
        raise ValueError("min_success_rate must be between 0 and 1")
    n_obs = len(data['y'])
    
    # Generate all bootstrap indices at once
    boot_indices = generate_bootstrap_indices(key, n_obs, n_boots)
    
    # Define single bootstrap iteration
    def single_bootstrap_fit(indices):
        """Fit model on resampled data."""
        try:
            resampled_data = resample_data(data, indices)
            params = fit_function(resampled_data)
            return params
        except Exception as e:
            if handle_failures:
                # Return NaN for failed fits
                return jnp.full_like(fit_function(data), jnp.nan)
            else:
                raise e
    
    # Parallel bootstrap using vmap
    if parallel:
        # Use vmap for maximum parallelization
        boot_coefficients = vmap(single_bootstrap_fit)(boot_indices)
    else:
        # Sequential fallback (slower but uses less memory)
        boot_coefficients = jnp.array([
            single_bootstrap_fit(indices) for indices in boot_indices
        ])
    
    # Compute statistics
    valid_mask = ~jnp.isnan(boot_coefficients).any(axis=1)
    n_successful = jnp.sum(valid_mask)
    if int(n_successful) / n_boots < min_success_rate:
        raise RuntimeError(
            f"Bootstrap success rate {int(n_successful) / n_boots:.1%} is below "
            f"min_success_rate {min_success_rate:.1%}"
        )
    
    mean_estimate = jnp.nanmean(boot_coefficients, axis=0)
    std_estimate = jnp.nanstd(boot_coefficients, axis=0)
    
    # Compute confidence intervals
    lower_ci, upper_ci = compute_confidence_intervals(
        boot_coefficients, alpha=alpha, method='percentile'
    )
    
    return BootstrapResult(
        coefficients=boot_coefficients,
        lower_ci=lower_ci,
        upper_ci=upper_ci,
        mean=mean_estimate,
        std=std_estimate,
        n_boots=n_boots,
        n_successful=int(n_successful),
        alpha=alpha,
        method='nonparametric'
    )


def parametric_bootstrap(
    fit_function: Callable,
    sample_function: Callable,
    data: Dict[str, jnp.ndarray],
    fitted_params: jnp.ndarray,
    key: jax.random.PRNGKey,
    n_boots: int = 1000,
    alpha: float = 0.05,
    parallel: bool = True,
    min_success_rate: float = 0.0
) -> BootstrapResult:
    """Perform parametric bootstrap for GAMLSS models.
    
    This method assumes the fitted distribution is correct and generates
    new data from it. It's more efficient when the distributional
    assumption is valid.
    
    Parameters
    ----------
    fit_function : callable
        Function that takes data dict and returns fitted parameters
    sample_function : callable
        Function that generates samples from fitted distribution.
        Should have signature: sample_function(params, key) -> jnp.ndarray
    data : dict
        Dictionary containing original data
    fitted_params : jnp.ndarray
        Parameters from original fit
    key : jax.random.PRNGKey
        Random number generator key
    n_boots : int, default=1000
        Number of bootstrap samples
    alpha : float, default=0.05
        Significance level for confidence intervals
    parallel : bool, default=True
        Whether to use vmap for parallel computation
        
    Returns
    -------
    BootstrapResult
        Bootstrap results including coefficients and confidence intervals
        
    Examples
    --------
    >>> def my_sample(params, key):
    ...     # Generate samples from fitted distribution
    ...     return jax.random.normal(key, shape=(n,)) * params[1] + params[0]
    >>> 
    >>> key = jax.random.PRNGKey(42)
    >>> result = parametric_bootstrap(
    ...     my_fit, my_sample, data, fitted_params, key, n_boots=1000
    ... )
    
    Notes
    -----
    Parametric bootstrap is more powerful when the distributional
    assumption is correct, but can be misleading if the model is
    misspecified. Use diagnostic plots to verify model fit before
    relying on parametric bootstrap results.
    """
    if not 0 <= min_success_rate <= 1:
        raise ValueError("min_success_rate must be between 0 and 1")
    # Split keys for each bootstrap sample
    keys = jax.random.split(key, n_boots)
    
    # Define single bootstrap iteration
    def single_parametric_fit(k):
        """Generate data and refit model."""
        # Generate new response from fitted distribution
        y_boot = sample_function(fitted_params, k)
        
        # Create new data dict with bootstrapped response
        boot_data = {**data, 'y': y_boot}
        
        # Refit model
        params = fit_function(boot_data)
        return params
    
    # Parallel bootstrap using vmap
    if parallel:
        boot_coefficients = vmap(single_parametric_fit)(keys)
    else:
        boot_coefficients = jnp.array([
            single_parametric_fit(k) for k in keys
        ])
    
    # Compute statistics
    valid_mask = ~jnp.isnan(boot_coefficients).any(axis=1)
    n_successful = jnp.sum(valid_mask)
    if int(n_successful) / n_boots < min_success_rate:
        raise RuntimeError(f"Bootstrap success rate {int(n_successful) / n_boots:.1%} is below min_success_rate {min_success_rate:.1%}")
    
    mean_estimate = jnp.nanmean(boot_coefficients, axis=0)
    std_estimate = jnp.nanstd(boot_coefficients, axis=0)
    
    # Compute confidence intervals
    lower_ci, upper_ci = compute_confidence_intervals(
        boot_coefficients, alpha=alpha, method='percentile'
    )
    
    return BootstrapResult(
        coefficients=boot_coefficients,
        lower_ci=lower_ci,
        upper_ci=upper_ci,
        mean=mean_estimate,
        std=std_estimate,
        n_boots=n_boots,
        n_successful=int(n_successful),
        alpha=alpha,
        method='parametric'
    )


def residual_bootstrap(
    fit_function: Callable,
    data: Dict[str, jnp.ndarray],
    fitted_values: jnp.ndarray,
    residuals: jnp.ndarray,
    key: jax.random.PRNGKey,
    n_boots: int = 1000,
    alpha: float = 0.05,
    parallel: bool = True,
    min_success_rate: float = 0.0
) -> BootstrapResult:
    """Perform residual bootstrap for GAMLSS models.
    
    This method resamples residuals and adds them to fitted values,
    which is appropriate when the model structure is correct but
    you're uncertain about the error distribution.
    
    Parameters
    ----------
    fit_function : callable
        Function that takes data dict and returns fitted parameters
    data : dict
        Dictionary containing original data
    fitted_values : jnp.ndarray
        Fitted values from original model
    residuals : jnp.ndarray
        Residuals from original model
    key : jax.random.PRNGKey
        Random number generator key
    n_boots : int, default=1000
        Number of bootstrap samples
    alpha : float, default=0.05
        Significance level for confidence intervals
    parallel : bool, default=True
        Whether to use vmap for parallel computation
        
    Returns
    -------
    BootstrapResult
        Bootstrap results including coefficients and confidence intervals
        
    Notes
    -----
    Residual bootstrap assumes:
    1. The model structure (covariates, link functions) is correct
    2. Residuals are exchangeable
    3. Heteroscedasticity is captured by the model
    """
    if not 0 <= min_success_rate <= 1:
        raise ValueError("min_success_rate must be between 0 and 1")
    n_obs = len(residuals)
    
    # Generate bootstrap indices for residuals
    boot_indices = generate_bootstrap_indices(key, n_obs, n_boots)
    
    # Define single bootstrap iteration
    def single_residual_fit(indices):
        """Resample residuals and refit."""
        # Resample residuals
        boot_residuals = residuals[indices]
        
        # Create new response
        y_boot = fitted_values + boot_residuals
        
        # Create new data dict
        boot_data = {**data, 'y': y_boot}
        
        # Refit model
        params = fit_function(boot_data)
        return params
    
    # Parallel bootstrap using vmap
    if parallel:
        boot_coefficients = vmap(single_residual_fit)(boot_indices)
    else:
        boot_coefficients = jnp.array([
            single_residual_fit(indices) for indices in boot_indices
        ])
    
    # Compute statistics
    valid_mask = ~jnp.isnan(boot_coefficients).any(axis=1)
    n_successful = jnp.sum(valid_mask)
    if int(n_successful) / n_boots < min_success_rate:
        raise RuntimeError(f"Bootstrap success rate {int(n_successful) / n_boots:.1%} is below min_success_rate {min_success_rate:.1%}")
    
    mean_estimate = jnp.nanmean(boot_coefficients, axis=0)
    std_estimate = jnp.nanstd(boot_coefficients, axis=0)
    
    # Compute confidence intervals
    lower_ci, upper_ci = compute_confidence_intervals(
        boot_coefficients, alpha=alpha, method='percentile'
    )
    
    return BootstrapResult(
        coefficients=boot_coefficients,
        lower_ci=lower_ci,
        upper_ci=upper_ci,
        mean=mean_estimate,
        std=std_estimate,
        n_boots=n_boots,
        n_successful=int(n_successful),
        alpha=alpha,
        method='residual'
    )


__all__ = [
    'BootstrapResult',
    'generate_bootstrap_indices',
    'resample_data',
    'compute_confidence_intervals',
    'nonparametric_bootstrap',
    'parametric_bootstrap',
    'residual_bootstrap',
]
