"""Diagnose ZAGA log PDF differences."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

# Add omnilss to path
sys.path.insert(0, str(Path(__file__).parent.parent / "omnilss" / "src"))

import jax.numpy as jnp
from jax.scipy.special import gammaln


def r_zaga_log_pdf(y, mu, sigma, nu):
    """R ZAGA log PDF implementation.
    
    From R code:
    log.lik <- ifelse(x==0, log(nu), 
                      log(1-nu)+(1/sigma^2)*log(x/(mu*sigma^2))-x/(mu*sigma^2)-log(x)-lgamma(1/sigma^2))
    """
    eps = 1e-16
    
    # For y == 0
    log_lik_zero = np.log(nu)
    
    # For y > 0
    sigma_sq = sigma ** 2
    alpha = 1.0 / sigma_sq
    
    # R formula: log(1-nu) + alpha*log(y/(mu*sigma_sq)) - y/(mu*sigma_sq) - log(y) - lgamma(alpha)
    # Simplify: log(1-nu) + alpha*log(y) - alpha*log(mu*sigma_sq) - y/(mu*sigma_sq) - log(y) - lgamma(alpha)
    # = log(1-nu) + (alpha-1)*log(y) - alpha*log(mu*sigma_sq) - y/(mu*sigma_sq) - lgamma(alpha)
    
    log_lik_pos = (
        np.log(1.0 - nu)
        + alpha * np.log(np.maximum(y, eps) / (mu * sigma_sq))
        - y / (mu * sigma_sq)
        - np.log(np.maximum(y, eps))
        - np.vectorize(lambda a: float(gammaln(a)))(alpha)
    )
    
    return np.where(y == 0, log_lik_zero, log_lik_pos)


def python_zaga_log_pdf(y, mu, sigma, nu):
    """Python ZAGA log PDF implementation."""
    from omnilss.distributions_b5_optimized import _zaga_log_pdf_opt
    
    y_jax = jnp.array(y, dtype=jnp.float64)
    mu_jax = jnp.array(mu, dtype=jnp.float64)
    sigma_jax = jnp.array(sigma, dtype=jnp.float64)
    nu_jax = jnp.array(nu, dtype=jnp.float64)
    
    result = _zaga_log_pdf_opt(y_jax, mu_jax, sigma_jax, nu_jax)
    return np.array(result)


def test_zaga_log_pdf():
    """Test ZAGA log PDF against R formula."""
    print("=" * 80)
    print("Testing ZAGA Log PDF")
    print("=" * 80)
    print()
    
    # Test parameters
    mu = 2.0
    sigma = 0.5
    nu = 0.3
    
    # Test values
    y_values = np.array([0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    
    print(f"Parameters: mu={mu}, sigma={sigma}, nu={nu}")
    print()
    print(f"{'y':<10} {'R log PDF':<15} {'Python log PDF':<15} {'Difference':<15}")
    print("-" * 60)
    
    for y in y_values:
        r_logpdf = r_zaga_log_pdf(y, mu, sigma, nu)
        py_logpdf = python_zaga_log_pdf(y, mu, sigma, nu)
        diff = py_logpdf - r_logpdf
        
        print(f"{y:<10.1f} {r_logpdf:<15.6f} {py_logpdf:<15.6f} {diff:<15.6e}")
    
    print()
    
    # Test with arrays
    print("Testing with array of values...")
    y_array = np.array([0.0, 0.0, 1.5, 2.3, 0.0, 3.1, 0.8])
    mu_array = np.full_like(y_array, mu)
    sigma_array = np.full_like(y_array, sigma)
    nu_array = np.full_like(y_array, nu)
    
    r_logpdf_array = r_zaga_log_pdf(y_array, mu_array, sigma_array, nu_array)
    py_logpdf_array = python_zaga_log_pdf(y_array, mu_array, sigma_array, nu_array)
    
    diff_array = py_logpdf_array - r_logpdf_array
    
    print(f"Max difference: {np.abs(diff_array).max():.6e}")
    print(f"Mean difference: {np.mean(diff_array):.6e}")
    print()
    
    # Compute deviance
    r_deviance = -2.0 * np.sum(r_logpdf_array)
    py_deviance = -2.0 * np.sum(py_logpdf_array)
    
    print(f"R deviance: {r_deviance:.6f}")
    print(f"Python deviance: {py_deviance:.6f}")
    print(f"Deviance difference: {py_deviance - r_deviance:.6f}")
    print()


def test_gamma_part():
    """Test just the Gamma part of ZAGA."""
    print("=" * 80)
    print("Testing Gamma Part of ZAGA")
    print("=" * 80)
    print()
    
    mu = 2.0
    sigma = 0.5
    y = 1.5
    
    sigma_sq = sigma ** 2
    alpha = 1.0 / sigma_sq
    beta = mu * sigma_sq
    
    print(f"Parameters: mu={mu}, sigma={sigma}, y={y}")
    print(f"Gamma parameters: alpha={alpha}, beta={beta}")
    print()
    
    # R formula (for y > 0 part only, without the (1-nu) term)
    r_log_gamma = (
        alpha * np.log(y / (mu * sigma_sq))
        - y / (mu * sigma_sq)
        - np.log(y)
        - float(gammaln(alpha))
    )
    
    # Standard Gamma log PDF: (alpha-1)*log(y) - y/beta - alpha*log(beta) - lgamma(alpha)
    std_log_gamma = (
        (alpha - 1.0) * np.log(y)
        - y / beta
        - alpha * np.log(beta)
        - float(gammaln(alpha))
    )
    
    # Python implementation (from optimized version)
    eps = jnp.finfo(jnp.float64).eps
    y_safe = jnp.maximum(y, eps)
    py_log_gamma = (
        (alpha - 1.0) * jnp.log(y_safe)
        - y_safe / beta
        - gammaln(alpha)
        - alpha * jnp.log(beta)
    )
    
    print(f"R formula result: {r_log_gamma:.10f}")
    print(f"Standard Gamma formula: {std_log_gamma:.10f}")
    print(f"Python formula: {float(py_log_gamma):.10f}")
    print()
    
    # Check if R formula equals standard formula
    print("Verifying R formula transformation:")
    print(f"  alpha * log(y/(mu*sigma^2)) = alpha*log(y) - alpha*log(mu*sigma^2)")
    print(f"  = {alpha * np.log(y):.10f} - {alpha * np.log(mu * sigma_sq):.10f}")
    print(f"  = {alpha * np.log(y) - alpha * np.log(mu * sigma_sq):.10f}")
    print()
    print(f"  R: alpha*log(y/(mu*sigma^2)) - log(y) = {alpha * np.log(y / (mu * sigma_sq)) - np.log(y):.10f}")
    print(f"  Std: (alpha-1)*log(y) - alpha*log(beta) = {(alpha - 1.0) * np.log(y) - alpha * np.log(beta):.10f}")
    print()
    
    # They should be equal
    diff = r_log_gamma - std_log_gamma
    print(f"Difference: {diff:.6e}")
    print()


if __name__ == "__main__":
    test_gamma_part()
    print("\n")
    test_zaga_log_pdf()
