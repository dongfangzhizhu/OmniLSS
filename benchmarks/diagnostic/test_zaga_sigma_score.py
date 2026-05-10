"""Test ZAGA sigma score function."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

# Add omnilss to path
sys.path.insert(0, str(Path(__file__).parent.parent / "omnilss" / "src"))

import jax.numpy as jnp
from jax.scipy.special import polygamma


def r_zaga_sigma_score(y, mu, sigma):
    """R ZAGA sigma score (dldd).
    
    From R code:
    dldd = function(y,mu,sigma) ifelse(y==0,0,(2/sigma^3)*((y/mu)-log(y)+log(mu)+log(sigma^2)-1+digamma(1/(sigma^2))))
    """
    eps = 1e-16
    
    # For y == 0
    score_zero = 0.0
    
    # For y > 0
    sigma_sq = sigma ** 2
    alpha = 1.0 / sigma_sq
    
    # R formula: (2/sigma^3) * ((y/mu) - log(y) + log(mu) + log(sigma^2) - 1 + digamma(alpha))
    # = (2/sigma^3) * ((y/mu) - log(y/mu) + log(sigma^2) - 1 + digamma(alpha))
    
    y_safe = np.maximum(y, eps)
    score_pos = (2.0 / (sigma ** 3)) * (
        (y_safe / mu)
        - np.log(y_safe)
        + np.log(mu)
        + np.log(sigma_sq)
        - 1.0
        + np.vectorize(lambda a: float(polygamma(0, a)))(alpha)
    )
    
    return np.where(y == 0, score_zero, score_pos)


def python_zaga_sigma_score(y, mu, sigma):
    """Python ZAGA sigma score."""
    from omnilss.distributions_b5_optimized import _zaga_score_sigma_opt
    
    y_jax = jnp.array(y, dtype=jnp.float64)
    mu_jax = jnp.array(mu, dtype=jnp.float64)
    sigma_jax = jnp.array(sigma, dtype=jnp.float64)
    nu_jax = jnp.array([0.3] * len(y), dtype=jnp.float64)  # nu doesn't affect sigma score
    
    result = _zaga_score_sigma_opt(y_jax, mu_jax, sigma_jax, nu_jax)
    return np.array(result)


def test_sigma_score():
    """Test sigma score function."""
    print("=" * 80)
    print("Testing ZAGA Sigma Score Function")
    print("=" * 80)
    print()
    
    # Test parameters
    mu = 2.0
    sigma = 0.5
    
    # Test values
    y_values = np.array([0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    mu_array = np.full_like(y_values, mu)
    sigma_array = np.full_like(y_values, sigma)
    
    print(f"Parameters: mu={mu}, sigma={sigma}")
    print()
    print(f"{'y':<10} {'R score':<15} {'Python score':<15} {'Difference':<15}")
    print("-" * 60)
    
    r_scores = r_zaga_sigma_score(y_values, mu_array, sigma_array)
    py_scores = python_zaga_sigma_score(y_values, mu_array, sigma_array)
    
    for i, y in enumerate(y_values):
        diff = py_scores[i] - r_scores[i]
        print(f"{y:<10.1f} {r_scores[i]:<15.6f} {py_scores[i]:<15.6f} {diff:<15.6e}")
    
    print()
    print(f"Max difference: {np.abs(py_scores - r_scores).max():.6e}")
    print(f"Mean difference: {np.mean(py_scores - r_scores):.6e}")
    print()
    
    # Test the formula transformation
    print("=" * 80)
    print("Verifying Formula Transformation")
    print("=" * 80)
    print()
    
    y = 1.5
    sigma_sq = sigma ** 2
    alpha = 1.0 / sigma_sq
    
    # R formula
    r_formula = (2.0 / (sigma ** 3)) * (
        (y / mu)
        - np.log(y)
        + np.log(mu)
        + np.log(sigma_sq)
        - 1.0
        + float(polygamma(0, alpha))
    )
    
    # Simplified formula (what we use in Python)
    # (2/sigma^3) * ((y/mu) - log(y) + log(mu) + log(sigma^2) - 1 + digamma(alpha))
    # = (2/sigma^3) * ((y/mu) - log(y/mu) + log(sigma^2) - 1 + digamma(alpha))
    # = (2/sigma^3) * ((y/mu) - log(y/mu) + 2*log(sigma) - 1 + digamma(alpha))
    
    simplified = (2.0 / (sigma ** 3)) * (
        (y / mu)
        - np.log(y / mu)
        + 2.0 * np.log(sigma)
        - 1.0
        + float(polygamma(0, alpha))
    )
    
    # Our Python formula
    # (2/sigma^3) * (y/mu - 1 - log(y/mu) + 2 + digamma(alpha))
    # Wait, this doesn't match! Let me recalculate...
    
    # R: (2/sigma^3) * ((y/mu) - log(y) + log(mu) + log(sigma^2) - 1 + digamma(alpha))
    # = (2/sigma^3) * ((y/mu) - log(y) + log(mu) + 2*log(sigma) - 1 + digamma(alpha))
    
    # Let's expand log(y) - log(mu) = log(y/mu)
    # So: (y/mu) - log(y) + log(mu) = (y/mu) - log(y/mu)
    
    # Therefore: (2/sigma^3) * ((y/mu) - log(y/mu) + 2*log(sigma) - 1 + digamma(alpha))
    
    # Our Python: (2/sigma^3) * (y/mu - 1 - log(y/mu) + 2 + digamma(alpha))
    # = (2/sigma^3) * (y/mu - log(y/mu) + 1 + digamma(alpha))
    
    # These don't match! The constant term is different:
    # R: 2*log(sigma) - 1
    # Python: 1
    
    print(f"y = {y}, mu = {mu}, sigma = {sigma}")
    print()
    print(f"R formula result: {r_formula:.10f}")
    print(f"Simplified (should match R): {simplified:.10f}")
    print()
    
    # Check the constant term
    print(f"R constant term: 2*log(sigma) - 1 = 2*log({sigma}) - 1 = {2.0 * np.log(sigma) - 1.0:.10f}")
    print(f"Python constant term: 2 = 2.000000")
    print()
    print("ERROR: The formulas don't match!")
    print()


if __name__ == "__main__":
    test_sigma_score()
