"""Data generators for performance testing."""

from __future__ import annotations

import numpy as np
from typing import Any


def generate_predictors(n: int, n_predictors: int, seed: int = 42) -> dict[str, np.ndarray]:
    """Generate predictor variables.
    
    Parameters
    ----------
    n : int
        Number of observations
    n_predictors : int
        Number of predictors
    seed : int
        Random seed
    
    Returns
    -------
    data : dict
        Dictionary with predictor arrays x1, x2, ...
    """
    rng = np.random.RandomState(seed)
    data = {}
    
    for i in range(1, n_predictors + 1):
        # Mix of continuous and categorical-like predictors
        if i % 3 == 0:
            # Categorical-like (0, 1, 2)
            data[f"x{i}"] = rng.choice([0, 1, 2], size=n)
        else:
            # Continuous
            data[f"x{i}"] = rng.normal(0, 1, size=n)
    
    return data


def generate_normal_data(n: int, n_predictors: int = 1, seed: int = 42) -> dict[str, Any]:
    """Generate data from normal distribution."""
    # Use different seeds for predictors and response to avoid numerical issues
    # but keep them deterministic
    rng_pred = np.random.RandomState(seed + 1000)
    rng_resp = np.random.RandomState(seed)
    
    # Generate predictors with separate RNG
    data = {}
    for i in range(1, n_predictors + 1):
        if i % 3 == 0:
            data[f"x{i}"] = rng_pred.choice([0, 1, 2], size=n)
        else:
            data[f"x{i}"] = rng_pred.normal(0, 1, size=n)
    
    # Generate response
    mu = np.full(n, 10.0)  # Start with array of constant values
    for i in range(1, n_predictors + 1):
        mu = mu + 2.0 * data[f"x{i}"]
    
    sigma = 2.5  # Increased from 2.0 for more noise
    y = rng_resp.normal(mu, sigma)
    data["y"] = y
    
    return data


def generate_gamma_data(n: int, n_predictors: int = 1, seed: int = 42) -> dict[str, Any]:
    """Generate data from gamma distribution."""
    rng = np.random.RandomState(seed)
    
    data = generate_predictors(n, n_predictors, seed)
    
    # Generate response
    log_mu = 2.0
    for i in range(1, n_predictors + 1):
        log_mu = log_mu + 0.5 * data[f"x{i}"]
    
    mu = np.exp(log_mu)
    sigma = 0.5  # CV
    shape = 1.0 / (sigma ** 2)
    scale = mu / shape
    
    y = rng.gamma(shape, scale)
    data["y"] = y
    
    return data


def generate_poisson_data(n: int, n_predictors: int = 1, seed: int = 42) -> dict[str, Any]:
    """Generate data from Poisson distribution."""
    rng = np.random.RandomState(seed)
    
    data = generate_predictors(n, n_predictors, seed)
    
    # Generate response
    log_mu = 1.0
    for i in range(1, n_predictors + 1):
        log_mu = log_mu + 0.3 * data[f"x{i}"]
    
    mu = np.exp(log_mu)
    y = rng.poisson(mu)
    data["y"] = y
    
    return data


def generate_binomial_data(n: int, n_predictors: int = 1, bd: int = 10, seed: int = 42) -> dict[str, Any]:
    """Generate data from binomial distribution."""
    rng = np.random.RandomState(seed)
    
    data = generate_predictors(n, n_predictors, seed)
    
    # Generate response
    logit_p = np.zeros(n)  # Initialize as array
    for i in range(1, n_predictors + 1):
        logit_p = logit_p + 0.5 * data[f"x{i}"]
    
    p = 1.0 / (1.0 + np.exp(-logit_p))
    y_counts = rng.binomial(bd, p)
    
    # Store as proportions for R compatibility
    data["y"] = y_counts / bd  # Convert to proportions [0, 1]
    data["bd"] = np.full(n, bd, dtype=float)
    
    return data


def generate_beta_data(n: int, n_predictors: int = 1, seed: int = 42) -> dict[str, Any]:
    """Generate data from beta distribution."""
    rng = np.random.RandomState(seed)
    
    data = generate_predictors(n, n_predictors, seed)
    
    # Generate response
    logit_mu = 0.0
    for i in range(1, n_predictors + 1):
        logit_mu = logit_mu + 0.5 * data[f"x{i}"]
    
    mu = 1.0 / (1.0 + np.exp(-logit_mu))
    sigma = 0.5  # Dispersion
    
    # Convert to alpha, beta parameters
    phi = 1.0 / sigma
    alpha = mu * phi
    beta_param = (1.0 - mu) * phi
    
    y = rng.beta(alpha, beta_param)
    data["y"] = y
    
    return data


def generate_zip_data(n: int, n_predictors: int = 1, seed: int = 42) -> dict[str, Any]:
    """Generate data from zero-inflated Poisson distribution."""
    rng = np.random.RandomState(seed)
    
    data = generate_predictors(n, n_predictors, seed)
    
    # Generate response
    log_mu = 1.0
    for i in range(1, n_predictors + 1):
        log_mu = log_mu + 0.3 * data[f"x{i}"]
    
    mu = np.exp(log_mu)
    
    # Zero-inflation probability
    pi = 0.2
    
    # Generate zero-inflated Poisson
    is_zero = rng.binomial(1, pi, size=n)
    poisson_part = rng.poisson(mu)
    y = np.where(is_zero, 0, poisson_part)
    
    data["y"] = y
    
    return data


def generate_nbi_data(n: int, n_predictors: int = 1, seed: int = 42) -> dict[str, Any]:
    """Generate data from negative binomial distribution."""
    rng = np.random.RandomState(seed)
    
    data = generate_predictors(n, n_predictors, seed)
    
    # Generate response
    log_mu = 1.5
    for i in range(1, n_predictors + 1):
        log_mu = log_mu + 0.3 * data[f"x{i}"]
    
    mu = np.exp(log_mu)
    sigma = 0.5  # Overdispersion
    
    # Convert to n, p parameters
    size = 1.0 / sigma
    prob = size / (size + mu)
    
    y = rng.negative_binomial(size, prob)
    data["y"] = y
    
    return data


def generate_lognormal_data(n: int, n_predictors: int = 1, seed: int = 42) -> dict[str, Any]:
    """Generate data from lognormal distribution."""
    # Use different seeds for predictors and response
    rng_pred = np.random.RandomState(seed + 2000)
    rng_resp = np.random.RandomState(seed)
    
    # Generate predictors with separate RNG
    data = {}
    for i in range(1, n_predictors + 1):
        if i % 3 == 0:
            data[f"x{i}"] = rng_pred.choice([0, 1, 2], size=n)
        else:
            data[f"x{i}"] = rng_pred.normal(0, 1, size=n)
    
    # Generate response
    log_mu = np.full(n, 2.0)  # Start with array
    for i in range(1, n_predictors + 1):
        log_mu = log_mu + 0.3 * data[f"x{i}"]
    
    mu = log_mu  # mu is the mean of log(Y)
    sigma = 0.6  # Increased from 0.5
    
    y = np.exp(rng_resp.normal(mu, sigma))
    data["y"] = y
    
    return data


def generate_weibull_data(n: int, n_predictors: int = 1, seed: int = 42) -> dict[str, Any]:
    """Generate data from Weibull distribution."""
    rng = np.random.RandomState(seed)
    
    data = generate_predictors(n, n_predictors, seed)
    
    # Generate response
    log_mu = 2.0
    for i in range(1, n_predictors + 1):
        log_mu = log_mu + 0.3 * data[f"x{i}"]
    
    mu = np.exp(log_mu)
    sigma = 0.5
    
    # Weibull parameters
    shape = 1.0 / sigma
    scale = mu / np.exp(np.log(np.gamma(1.0 + sigma)))
    
    y = scale * rng.weibull(shape)
    data["y"] = y
    
    return data


def generate_zaga_data(n: int, n_predictors: int = 1, seed: int = 42) -> dict[str, Any]:
    """Generate data from zero-adjusted gamma distribution."""
    # Use different seeds for predictors and response
    rng_pred = np.random.RandomState(seed + 3000)
    rng_resp = np.random.RandomState(seed)
    
    # Generate predictors with separate RNG
    data = {}
    for i in range(1, n_predictors + 1):
        if i % 3 == 0:
            data[f"x{i}"] = rng_pred.choice([0, 1, 2], size=n)
        else:
            data[f"x{i}"] = rng_pred.normal(0, 1, size=n)
    
    # Generate response with zero-inflation
    log_mu = np.full(n, 1.5)  # Start with array
    for i in range(1, n_predictors + 1):
        log_mu = log_mu + 0.2 * data[f"x{i}"]  # Reduced coefficient
    
    mu = np.exp(log_mu)
    sigma = 0.6  # Increased from 0.5 for more variability
    
    # Zero probability - make it depend on predictors for more realistic data
    logit_pi = np.full(n, -1.0)  # Base zero probability ~0.27
    for i in range(1, min(n_predictors + 1, 2)):  # Use first predictor if available
        if f"x{i}" in data:
            logit_pi = logit_pi + 0.3 * data[f"x{i}"]
    pi = 1.0 / (1.0 + np.exp(-logit_pi))
    pi = np.clip(pi, 0.1, 0.4)  # Keep zero proportion reasonable
    
    # Generate zero-adjusted gamma
    is_zero = rng_resp.binomial(1, pi)
    
    # Gamma part with more stable parameters
    shape = 1.0 / (sigma ** 2)
    shape = np.maximum(shape, 0.5)  # Ensure shape is not too small
    scale = mu / shape
    gamma_part = rng_resp.gamma(shape, scale)
    
    y = np.where(is_zero, 0, gamma_part)
    data["y"] = y
    
    return data


def generate_zaig_data(n: int, n_predictors: int = 1, seed: int = 42) -> dict[str, Any]:
    """Generate data from zero-adjusted inverse Gaussian distribution."""
    # Use different seeds for predictors and response
    rng_pred = np.random.RandomState(seed + 4000)
    rng_resp = np.random.RandomState(seed)
    
    # Generate predictors with separate RNG
    data = {}
    for i in range(1, n_predictors + 1):
        if i % 3 == 0:
            data[f"x{i}"] = rng_pred.choice([0, 1, 2], size=n)
        else:
            data[f"x{i}"] = rng_pred.normal(0, 1, size=n)
    
    # Generate response with zero-inflation
    log_mu = np.full(n, 1.5)  # Start with array
    for i in range(1, n_predictors + 1):
        log_mu = log_mu + 0.2 * data[f"x{i}"]
    
    mu = np.exp(log_mu)
    sigma = 0.6  # Increased for more variability
    
    # Zero probability - make it depend on predictors
    logit_pi = np.full(n, -1.0)  # Base zero probability ~0.27
    for i in range(1, min(n_predictors + 1, 2)):
        if f"x{i}" in data:
            logit_pi = logit_pi + 0.3 * data[f"x{i}"]
    pi = 1.0 / (1.0 + np.exp(-logit_pi))
    pi = np.clip(pi, 0.1, 0.4)
    
    # Generate zero-adjusted inverse Gaussian
    is_zero = rng_resp.binomial(1, pi)
    
    # Inverse Gaussian part (using Wald distribution)
    try:
        from scipy.stats import invgauss
        # invgauss parameterization: mu (mean), scale
        ig_part = invgauss.rvs(mu / sigma, scale=mu, size=n, random_state=rng_resp)
    except ImportError:
        # Fallback: use gamma as approximation
        shape = 1.0 / (sigma ** 2)
        shape = np.maximum(shape, 0.5)
        scale = mu / shape
        ig_part = rng_resp.gamma(shape, scale)
    
    y = np.where(is_zero, 0, ig_part)
    data["y"] = y
    
    return data


# Distribution to generator mapping
DISTRIBUTION_GENERATORS = {
    "NO": generate_normal_data,
    "GA": generate_gamma_data,
    "PO": generate_poisson_data,
    "BI": generate_binomial_data,
    "BE": generate_beta_data,
    "ZIP": generate_zip_data,
    "NBI": generate_nbi_data,
    "LOGNO": generate_lognormal_data,
    "WEI": generate_weibull_data,
    "ZAGA": generate_zaga_data,
    "ZAIG": generate_zaig_data,
    # Add more as needed
}


def generate_data_for_distribution(
    distribution: str,
    n: int,
    n_predictors: int = 1,
    seed: int = 42,
    **kwargs: Any,
) -> dict[str, Any]:
    """Generate data for a specific distribution.
    
    Parameters
    ----------
    distribution : str
        Distribution name (e.g., "NO", "GA", "PO")
    n : int
        Number of observations
    n_predictors : int
        Number of predictors
    seed : int
        Random seed
    **kwargs
        Additional distribution-specific parameters
    
    Returns
    -------
    data : dict
        Dictionary with response and predictors
    """
    generator = DISTRIBUTION_GENERATORS.get(distribution)
    
    if generator is None:
        # Default to normal data
        return generate_normal_data(n, n_predictors, seed)
    
    return generator(n, n_predictors, seed, **kwargs)
