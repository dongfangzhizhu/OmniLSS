"""Diagnose BI deviance calculation."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "omnilss" / "src"))

import jax.numpy as jnp
from jax.scipy.special import gammaln


def python_bi_deviance(y, mu, bd):
    """Python BI deviance calculation."""
    y = jnp.asarray(y, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    bd = jnp.asarray(bd, dtype=jnp.float64)
    
    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.clip(mu, eps, 1.0 - eps)
    
    # Convert proportions to counts
    y_counts = jnp.round(y * bd)
    
    # Binomial log-likelihood
    log_comb = gammaln(bd + 1.0) - gammaln(y_counts + 1.0) - gammaln(bd - y_counts + 1.0)
    log_density = log_comb + y_counts * jnp.log(mu) + (bd - y_counts) * jnp.log(1.0 - mu)
    
    deviance = -2.0 * log_density
    return float(jnp.sum(deviance))


def r_bi_deviance_formula(y, mu, bd):
    """R BI deviance calculation (what R actually computes).
    
    R uses dbinom(y_counts, size=bd, prob=mu, log=TRUE)
    where y_counts = round(y * bd)
    """
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    bd = np.asarray(bd, dtype=np.float64)
    
    # Convert proportions to counts
    y_counts = np.round(y * bd).astype(int)
    bd_int = bd.astype(int)
    
    # Use scipy's binom.logpmf
    from scipy.stats import binom
    log_density = binom.logpmf(y_counts, bd_int, mu)
    
    deviance = -2.0 * log_density
    return float(np.sum(deviance))


def test_deviance_calculation():
    """Test deviance calculation."""
    print("="*80)
    print("BI Deviance Calculation Test")
    print("="*80)
    print()
    
    # Simple test case
    n = 100
    bd = 10
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] * 12 + [0.5] * 4)  # 100 values
    mu = np.full(n, 0.481)  # Constant mu
    bd_array = np.full(n, bd, dtype=float)
    
    print(f"Test data:")
    print(f"  n = {n}")
    print(f"  bd = {bd}")
    print(f"  y: min={y.min():.3f}, max={y.max():.3f}, mean={y.mean():.3f}")
    print(f"  mu: {mu[0]:.3f} (constant)")
    print()
    
    # Python calculation
    py_dev = python_bi_deviance(y, mu, bd_array)
    print(f"Python deviance: {py_dev:.6f}")
    
    # R-style calculation
    r_dev = r_bi_deviance_formula(y, mu, bd_array)
    print(f"R-style deviance: {r_dev:.6f}")
    
    # Difference
    diff = abs(py_dev - r_dev)
    print(f"Difference: {diff:.6f}")
    print()
    
    # Check individual terms
    print("Checking first 5 observations:")
    y_counts = np.round(y * bd).astype(int)
    for i in range(5):
        print(f"  y[{i}]={y[i]:.1f}, y_counts={y_counts[i]}, mu={mu[i]:.3f}, bd={bd}")
        
        # Python
        y_i = jnp.array([y[i]], dtype=jnp.float64)
        mu_i = jnp.array([mu[i]], dtype=jnp.float64)
        bd_i = jnp.array([bd], dtype=jnp.float64)
        py_dev_i = python_bi_deviance(y_i, mu_i, bd_i)
        
        # R-style
        from scipy.stats import binom
        log_lik_r = binom.logpmf(y_counts[i], bd, mu[i])
        r_dev_i = -2.0 * log_lik_r
        
        print(f"    Python deviance: {py_dev_i:.6f}")
        print(f"    R-style deviance: {r_dev_i:.6f}")
        print(f"    Difference: {abs(py_dev_i - r_dev_i):.6e}")
        print()


if __name__ == "__main__":
    test_deviance_calculation()
