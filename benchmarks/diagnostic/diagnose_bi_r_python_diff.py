"""Diagnose the exact difference between Python and R BI deviance calculations."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import json
import tempfile
import subprocess

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "omnilss" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "omnilss" / "tests"))
sys.path.insert(0, str(Path(__file__).parent.parent / "performance"))

from omnilss.distributions import resolve_family
from omnilss.fitting import gamlss
from benchmarks.data_generators import generate_binomial_data


def test_simple_case():
    """Test a simple case to understand the deviance difference."""
    print("="*80)
    print("Simple BI Deviance Test: Python vs R")
    print("="*80)
    print()
    
    # Generate simple data
    n = 100
    bd = 10
    data = generate_binomial_data(n, 0, bd=bd, seed=42)
    
    # Ensure y is an array
    if not isinstance(data['y'], np.ndarray):
        y_val = data['y']
        data['y'] = np.full(n, y_val, dtype=float)
    
    print(f"Data info:")
    print(f"  n = {n}")
    print(f"  bd = {bd}")
    print(f"  y (proportions): min={data['y'].min():.3f}, max={data['y'].max():.3f}, mean={data['y'].mean():.3f}")
    print(f"  y_counts: min={np.round(data['y']*bd).min():.0f}, max={np.round(data['y']*bd).max():.0f}, mean={np.round(data['y']*bd).mean():.1f}")
    print()
    
    # Fit Python model
    print("Fitting Python model...")
    family = resolve_family("BI")
    py_model = gamlss(
        formula="y ~ 1",
        sigma_formula="~1",
        family=family,
        data=data,
    )
    
    py_deviance = py_model.deviance
    py_mu = py_model.fitted_values['mu'][0]
    
    print(f"  Python deviance: {py_deviance:.6f}")
    print(f"  Python mu: {py_mu:.6f}")
    print()
    
    # Manually calculate Python deviance using the distribution function
    print("Manual Python deviance calculation:")
    import jax.numpy as jnp
    from jax.scipy.special import gammaln
    
    y = jnp.array(data['y'], dtype=jnp.float64)
    mu = jnp.full(n, py_mu, dtype=jnp.float64)
    bd_arr = jnp.full(n, bd, dtype=jnp.float64)
    
    # Convert proportions to counts
    y_counts = jnp.round(y * bd_arr)
    
    # Binomial log-likelihood
    log_comb = gammaln(bd_arr + 1.0) - gammaln(y_counts + 1.0) - gammaln(bd_arr - y_counts + 1.0)
    log_density = log_comb + y_counts * jnp.log(mu) + (bd_arr - y_counts) * jnp.log(1.0 - mu)
    
    manual_deviance = float(jnp.sum(-2.0 * log_density))
    print(f"  Manual deviance: {manual_deviance:.6f}")
    print(f"  Difference from fitted: {abs(manual_deviance - py_deviance):.6e}")
    print()
    
    # Now call R directly to see what it calculates
    print("Calling R GAMLSS...")
    
    # Prepare data for R
    y_counts_r = np.round(data['y'] * bd).astype(int)
    
    # Create R script
    r_script = f"""
    library(gamlss)
    
    # Data
    y_counts <- c({','.join(map(str, y_counts_r))})
    bd <- {bd}
    n <- {n}
    
    # Create two-column response
    y_cbind <- cbind(y_counts, bd - y_counts)
    
    # Fit model
    model <- gamlss(y_cbind ~ 1, family=BI(), trace=FALSE)
    
    # Extract results
    cat("R deviance:", deviance(model), "\\n")
    cat("R mu:", fitted(model, "mu")[1], "\\n")
    cat("R mu logit:", coef(model, "mu"), "\\n")
    
    # Manual deviance calculation
    mu_fitted <- fitted(model, "mu")[1]
    log_lik <- dbinom(y_counts, size=bd, prob=mu_fitted, log=TRUE)
    manual_dev <- -2 * sum(log_lik)
    cat("R manual deviance:", manual_dev, "\\n")
    
    # Check individual terms
    cat("\\nFirst 5 deviance increments:\\n")
    for (i in 1:5) {{
        dev_inc <- -2 * dbinom(y_counts[i], size=bd, prob=mu_fitted, log=TRUE)
        cat("  i=", i, "y_counts=", y_counts[i], "dev_inc=", dev_inc, "\\n")
    }}
    """
    
    # Write R script to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
        f.write(r_script)
        r_script_path = f.name
    
    try:
        # Run R script
        result = subprocess.run(
            ['Rscript', r_script_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        print(result.stdout)
        if result.stderr:
            print("R stderr:", result.stderr)
        
    finally:
        # Clean up
        Path(r_script_path).unlink()
    
    print()
    print("="*80)
    print("Analysis")
    print("="*80)
    print()
    print("If Python and R deviances differ significantly, the issue is likely:")
    print("1. Different data being used (proportions vs counts)")
    print("2. Different bd values being used")
    print("3. Different fitted mu values")
    print("4. Numerical precision issues")


if __name__ == "__main__":
    test_simple_case()
