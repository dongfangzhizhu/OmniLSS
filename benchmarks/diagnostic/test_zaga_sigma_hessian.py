"""Test ZAGA sigma hessian function."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

# Add omnilss to path
sys.path.insert(0, str(Path(__file__).parent.parent / "omnilss" / "src"))

import jax.numpy as jnp
from jax.scipy.special import polygamma


def r_zaga_sigma_hessian(y, sigma):
    """R ZAGA sigma hessian (d2ldd2).
    
    From R code:
    d2ldd2 = function(y,sigma) ifelse(y==0,0,(4/sigma^4)-(4/sigma^6)*trigamma((1/sigma^2)))
    """
    eps = 1e-16
    
    # For y == 0
    hess_zero = 0.0
    
    # For y > 0
    sigma_sq = sigma ** 2
    alpha = 1.0 / sigma_sq
    
    # R formula: (4/sigma^4) - (4/sigma^6)*trigamma(alpha)
    hess_pos = (4.0 / (sigma ** 4)) - (4.0 / (sigma ** 6)) * np.vectorize(lambda a: float(polygamma(1, a)))(alpha)
    
    return np.where(y == 0, hess_zero, hess_pos)


def python_zaga_sigma_hessian(y, sigma):
    """Python ZAGA sigma hessian."""
    from omnilss.distributions_b5_optimized import _zaga_hessian_sigma_opt
    
    y_jax = jnp.array(y, dtype=jnp.float64)
    mu_jax = jnp.array([2.0] * len(y), dtype=jnp.float64)  # mu doesn't affect hessian
    sigma_jax = jnp.array(sigma, dtype=jnp.float64)
    nu_jax = jnp.array([0.3] * len(y), dtype=jnp.float64)  # nu doesn't affect hessian
    
    result = _zaga_hessian_sigma_opt(y_jax, mu_jax, sigma_jax, nu_jax)
    return np.array(result)


def test_sigma_hessian():
    """Test sigma hessian function."""
    print("=" * 80)
    print("Testing ZAGA Sigma Hessian Function")
    print("=" * 80)
    print()
    
    # Test parameters
    sigma = 0.5
    
    # Test values
    y_values = np.array([0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    sigma_array = np.full_like(y_values, sigma)
    
    print(f"Parameters: sigma={sigma}")
    print()
    print(f"{'y':<10} {'R hessian':<15} {'Python hessian':<15} {'Difference':<15}")
    print("-" * 60)
    
    r_hess = r_zaga_sigma_hessian(y_values, sigma_array)
    py_hess = python_zaga_sigma_hessian(y_values, sigma_array)
    
    for i, y in enumerate(y_values):
        diff = py_hess[i] - r_hess[i]
        print(f"{y:<10.1f} {r_hess[i]:<15.6f} {py_hess[i]:<15.6f} {diff:<15.6e}")
    
    print()
    print(f"Max difference: {np.abs(py_hess - r_hess).max():.6e}")
    print(f"Mean difference: {np.mean(py_hess - r_hess):.6e}")
    print()


if __name__ == "__main__":
    test_sigma_hessian()
