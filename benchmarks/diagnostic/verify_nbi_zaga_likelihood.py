"""Verify NBI and ZAGA log-likelihood calculations against R.

This script tests the log-likelihood functions with fixed parameters
to isolate formula errors from optimization issues.
"""

import numpy as np
import jax.numpy as jnp
import subprocess
import json
import tempfile
from pathlib import Path

# Enable float64
import jax
jax.config.update("jax_enable_x64", True)

from omnilss.distributions import NBI
from omnilss.distributions_b5 import ZAGA


def test_nbi_likelihood():
    """Test NBI log-likelihood with fixed parameters."""
    print("=" * 80)
    print("Testing NBI Log-Likelihood")
    print("=" * 80)
    
    # Create test data
    np.random.seed(42)
    n = 10
    y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float64)
    
    # Fixed parameters (from a typical fit)
    mu = np.full(n, 3.0, dtype=np.float64)
    sigma = np.full(n, 0.5, dtype=np.float64)
    
    # Calculate Python log-likelihood
    nbi_family = NBI()
    python_deviance = nbi_family.g_dev_inc(y, mu, sigma)
    python_ll = -0.5 * np.sum(python_deviance)
    
    print(f"\nPython calculation:")
    print(f"  mu = {mu[0]}, sigma = {sigma[0]}")
    print(f"  y = {y}")
    print(f"  Individual deviances: {python_deviance}")
    print(f"  Total deviance: {np.sum(python_deviance):.6f}")
    print(f"  Log-likelihood: {python_ll:.6f}")
    
    # Calculate R log-likelihood
    r_code = f"""
    library(gamlss.dist)
    
    y <- c({','.join(map(str, y))})
    mu <- {mu[0]}
    sigma <- {sigma[0]}
    
    # Calculate log-likelihood for each observation
    ll_individual <- dNBI(y, mu=mu, sigma=sigma, log=TRUE)
    
    # Calculate deviance for each observation
    dev_individual <- -2 * ll_individual
    
    # Total
    total_ll <- sum(ll_individual)
    total_dev <- sum(dev_individual)
    
    cat("R calculation:\\n")
    cat("  mu =", mu, ", sigma =", sigma, "\\n")
    cat("  y =", y, "\\n")
    cat("  Individual log-likelihoods:", ll_individual, "\\n")
    cat("  Individual deviances:", dev_individual, "\\n")
    cat("  Total log-likelihood:", total_ll, "\\n")
    cat("  Total deviance:", total_dev, "\\n")
    
    # Also check the parameterization
    size <- 1/sigma
    cat("\\nParameterization check:\\n")
    cat("  size (1/sigma) =", size, "\\n")
    cat("  Expected mean =", mu, "\\n")
    cat("  Expected variance =", mu + sigma * mu^2, "\\n")
    """
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
        f.write(r_code)
        r_file = f.name
    
    try:
        result = subprocess.run(
            ['Rscript', r_file],
            capture_output=True,
            text=True,
            timeout=30
        )
        print(f"\n{result.stdout}")
        if result.stderr:
            print(f"R stderr: {result.stderr}")
    finally:
        Path(r_file).unlink()
    
    print("\n" + "=" * 80)


def test_zaga_likelihood():
    """Test ZAGA log-likelihood with fixed parameters."""
    print("\n" + "=" * 80)
    print("Testing ZAGA Log-Likelihood")
    print("=" * 80)
    
    # Create test data with zeros
    np.random.seed(42)
    n = 10
    y = np.array([0.0, 0.0, 1.5, 2.3, 3.1, 4.2, 5.5, 6.8, 7.2, 8.9], dtype=np.float64)
    
    # Fixed parameters
    mu = np.full(n, 4.0, dtype=np.float64)
    sigma = np.full(n, 0.6, dtype=np.float64)
    nu = np.full(n, 0.2, dtype=np.float64)  # 20% zeros
    
    # Calculate Python log-likelihood
    zaga_family = ZAGA()
    python_deviance = zaga_family.g_dev_inc(y, mu, sigma, nu)
    python_ll = -0.5 * np.sum(python_deviance)
    
    print(f"\nPython calculation:")
    print(f"  mu = {mu[0]}, sigma = {sigma[0]}, nu = {nu[0]}")
    print(f"  y = {y}")
    print(f"  Individual deviances: {python_deviance}")
    print(f"  Total deviance: {np.sum(python_deviance):.6f}")
    print(f"  Log-likelihood: {python_ll:.6f}")
    
    # Check Gamma parameterization
    print(f"\nPython Gamma parameterization:")
    sigma_sq = sigma[0] ** 2
    alpha = 1.0 / sigma_sq
    beta = mu[0] * sigma_sq
    print(f"  alpha (shape) = 1/sigma^2 = {alpha:.6f}")
    print(f"  beta (scale) = mu*sigma^2 = {beta:.6f}")
    print(f"  Expected mean = alpha*beta = {alpha*beta:.6f} (should be {mu[0]})")
    print(f"  Expected variance = alpha*beta^2 = {alpha*beta**2:.6f}")
    
    # Calculate R log-likelihood
    r_code = f"""
    library(gamlss.dist)
    
    y <- c({','.join(map(str, y))})
    mu <- {mu[0]}
    sigma <- {sigma[0]}
    nu <- {nu[0]}
    
    # Calculate log-likelihood for each observation
    ll_individual <- dZAGA(y, mu=mu, sigma=sigma, nu=nu, log=TRUE)
    
    # Calculate deviance for each observation
    dev_individual <- -2 * ll_individual
    
    # Total
    total_ll <- sum(ll_individual)
    total_dev <- sum(dev_individual)
    
    cat("R calculation:\\n")
    cat("  mu =", mu, ", sigma =", sigma, ", nu =", nu, "\\n")
    cat("  y =", y, "\\n")
    cat("  Individual log-likelihoods:", ll_individual, "\\n")
    cat("  Individual deviances:", dev_individual, "\\n")
    cat("  Total log-likelihood:", total_ll, "\\n")
    cat("  Total deviance:", total_dev, "\\n")
    
    # Check Gamma parameterization
    cat("\\nR Gamma parameterization:\\n")
    shape <- 1/sigma^2
    scale <- mu * sigma^2
    cat("  shape (1/sigma^2) =", shape, "\\n")
    cat("  scale (mu*sigma^2) =", scale, "\\n")
    cat("  Expected mean = shape*scale =", shape*scale, "\\n")
    cat("  Expected variance = shape*scale^2 =", shape*scale^2, "\\n")
    
    # Manual calculation for one positive value
    cat("\\nManual calculation for y[3] =", y[3], ":\\n")
    y_test <- y[3]
    log_gamma <- dgamma(y_test, shape=shape, scale=scale, log=TRUE)
    log_zaga <- log(1-nu) + log_gamma
    cat("  log(1-nu) =", log(1-nu), "\\n")
    cat("  log Gamma(y) =", log_gamma, "\\n")
    cat("  log ZAGA(y) =", log_zaga, "\\n")
    cat("  dZAGA result =", dZAGA(y_test, mu=mu, sigma=sigma, nu=nu, log=TRUE), "\\n")
    """
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
        f.write(r_code)
        r_file = f.name
    
    try:
        result = subprocess.run(
            ['Rscript', r_file],
            capture_output=True,
            text=True,
            timeout=30
        )
        print(f"\n{result.stdout}")
        if result.stderr:
            print(f"R stderr: {result.stderr}")
    finally:
        Path(r_file).unlink()
    
    print("\n" + "=" * 80)


def test_single_observation():
    """Test with a single observation to isolate per-sample errors."""
    print("\n" + "=" * 80)
    print("Testing Single Observation (NBI)")
    print("=" * 80)
    
    y = np.array([5.0], dtype=np.float64)
    mu = np.array([3.0], dtype=np.float64)
    sigma = np.array([0.5], dtype=np.float64)
    
    nbi_family = NBI()
    python_dev = nbi_family.g_dev_inc(y, mu, sigma)[0]
    python_ll = -0.5 * python_dev
    
    print(f"\nPython: y={y[0]}, mu={mu[0]}, sigma={sigma[0]}")
    print(f"  Deviance: {python_dev:.10f}")
    print(f"  Log-likelihood: {python_ll:.10f}")
    
    # R calculation
    r_code = f"""
    library(gamlss.dist)
    y <- {y[0]}
    mu <- {mu[0]}
    sigma <- {sigma[0]}
    
    ll <- dNBI(y, mu=mu, sigma=sigma, log=TRUE)
    dev <- -2 * ll
    
    cat("R: y =", y, ", mu =", mu, ", sigma =", sigma, "\\n")
    cat("  Log-likelihood:", ll, "\\n")
    cat("  Deviance:", dev, "\\n")
    cat("  Difference (Python - R):", {python_dev}, "- dev =", {python_dev} - dev, "\\n")
    """
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
        f.write(r_code)
        r_file = f.name
    
    try:
        result = subprocess.run(
            ['Rscript', r_file],
            capture_output=True,
            text=True,
            timeout=30
        )
        print(f"\n{result.stdout}")
    finally:
        Path(r_file).unlink()


if __name__ == "__main__":
    test_single_observation()
    test_nbi_likelihood()
    test_zaga_likelihood()
