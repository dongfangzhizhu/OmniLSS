"""Compare fitted parameters between Python and R to identify optimization issues."""

import numpy as np
import subprocess
import tempfile
from pathlib import Path
import json

# Enable float64
import jax
jax.config.update("jax_enable_x64", True)

from omnilss import gamlss
from omnilss.distributions import NBI
from omnilss.distributions_b5 import ZAGA


def compare_nbi_fit():
    """Compare NBI fitted parameters between Python and R."""
    print("=" * 80)
    print("Comparing NBI Fitted Parameters")
    print("=" * 80)
    
    # Generate test data
    np.random.seed(42)
    n = 100
    x1 = np.random.randn(n)
    
    # Generate NBI data from R
    r_code = f"""
    library(gamlss)
    set.seed(42)
    n <- {n}
    x1 <- rnorm(n)
    
    # Generate data
    mu_true <- exp(1 + 0.5*x1)
    sigma_true <- 0.5
    y <- rNBI(n, mu=mu_true, sigma=sigma_true)
    
    # Fit model
    model <- gamlss(y ~ x1, family=NBI(), trace=FALSE)
    
    # Extract parameters
    mu_fitted <- fitted(model, "mu")
    sigma_fitted <- fitted(model, "sigma")
    
    # Calculate deviance
    dev <- deviance(model)
    
    # Save results
    results <- list(
        y = y,
        mu_fitted = mu_fitted,
        sigma_fitted = sigma_fitted,
        deviance = dev,
        mu_coef = coef(model, "mu"),
        sigma_coef = coef(model, "sigma")
    )
    
    cat("R NBI Fit:\\n")
    cat("  Deviance:", dev, "\\n")
    cat("  mu coefficients:", coef(model, "mu"), "\\n")
    cat("  sigma coefficients:", coef(model, "sigma"), "\\n")
    cat("  mu range:", range(mu_fitted), "\\n")
    cat("  sigma range:", range(sigma_fitted), "\\n")
    
    # Save to JSON
    library(jsonlite)
    write_json(results, "nbi_r_fit.json")
    """
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
        f.write(r_code)
        r_file = f.name
    
    try:
        result = subprocess.run(
            ['Rscript', r_file],
            capture_output=True,
            text=True,
            timeout=60,
            cwd='.'
        )
        print(f"\n{result.stdout}")
        if result.stderr and 'Warning' not in result.stderr:
            print(f"R stderr: {result.stderr}")
        
        # Load R results
        with open('nbi_r_fit.json', 'r') as f:
            r_results = json.load(f)
        
        y = np.array(r_results['y'])
        
        # Fit with Python
        print("\nFitting with Python...")
        model = gamlss(
            formula="y ~ x1",
            data={"y": y, "x1": x1},
            family=NBI(),
            trace=False
        )
        
        python_dev = model.deviance
        python_mu = model.fitted_values['mu']
        python_sigma = model.fitted_values['sigma']
        
        print(f"\nPython NBI Fit:")
        print(f"  Deviance: {python_dev:.6f}")
        print(f"  mu range: [{python_mu.min():.6f}, {python_mu.max():.6f}]")
        print(f"  sigma range: [{python_sigma.min():.6f}, {python_sigma.max():.6f}]")
        
        # Compare
        r_dev = r_results['deviance']
        r_mu = np.array(r_results['mu_fitted'])
        r_sigma = np.array(r_results['sigma_fitted'])
        
        print(f"\nComparison:")
        print(f"  Deviance difference: {python_dev - r_dev:.6f}")
        print(f"  Deviance relative diff: {100*(python_dev - r_dev)/r_dev:.4f}%")
        print(f"  mu RMSE: {np.sqrt(np.mean((python_mu - r_mu)**2)):.6f}")
        print(f"  sigma RMSE: {np.sqrt(np.mean((python_sigma - r_sigma)**2)):.6f}")
        
        # Calculate deviance with R parameters in Python
        nbi_family = NBI()
        python_dev_with_r_params = np.sum(nbi_family.g_dev_inc(y, r_mu, r_sigma))
        
        print(f"\nDiagnostic:")
        print(f"  Python deviance with R parameters: {python_dev_with_r_params:.6f}")
        print(f"  R deviance: {r_dev:.6f}")
        print(f"  Difference: {python_dev_with_r_params - r_dev:.10f}")
        print(f"  (Should be ~0 if likelihood formula is correct)")
        
    finally:
        Path(r_file).unlink()
        if Path('nbi_r_fit.json').exists():
            Path('nbi_r_fit.json').unlink()
    
    print("\n" + "=" * 80)


def compare_zaga_fit():
    """Compare ZAGA fitted parameters between Python and R."""
    print("\n" + "=" * 80)
    print("Comparing ZAGA Fitted Parameters")
    print("=" * 80)
    
    # Generate test data
    np.random.seed(42)
    n = 100
    x1 = np.random.randn(n)
    
    # Generate ZAGA data from R
    r_code = f"""
    library(gamlss)
    set.seed(42)
    n <- {n}
    x1 <- rnorm(n)
    
    # Generate data
    mu_true <- exp(1 + 0.3*x1)
    sigma_true <- 0.6
    nu_true <- 0.2
    y <- rZAGA(n, mu=mu_true, sigma=sigma_true, nu=nu_true)
    
    # Fit model
    model <- gamlss(y ~ x1, family=ZAGA(), trace=FALSE)
    
    # Extract parameters
    mu_fitted <- fitted(model, "mu")
    sigma_fitted <- fitted(model, "sigma")
    nu_fitted <- fitted(model, "nu")
    
    # Calculate deviance
    dev <- deviance(model)
    
    # Save results
    results <- list(
        y = y,
        mu_fitted = mu_fitted,
        sigma_fitted = sigma_fitted,
        nu_fitted = nu_fitted,
        deviance = dev
    )
    
    cat("R ZAGA Fit:\\n")
    cat("  Deviance:", dev, "\\n")
    cat("  mu range:", range(mu_fitted), "\\n")
    cat("  sigma range:", range(sigma_fitted), "\\n")
    cat("  nu range:", range(nu_fitted), "\\n")
    cat("  Number of zeros:", sum(y == 0), "\\n")
    
    # Save to JSON
    library(jsonlite)
    write_json(results, "zaga_r_fit.json")
    """
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
        f.write(r_code)
        r_file = f.name
    
    try:
        result = subprocess.run(
            ['Rscript', r_file],
            capture_output=True,
            text=True,
            timeout=60,
            cwd='.'
        )
        print(f"\n{result.stdout}")
        if result.stderr and 'Warning' not in result.stderr:
            print(f"R stderr: {result.stderr}")
        
        # Load R results
        with open('zaga_r_fit.json', 'r') as f:
            r_results = json.load(f)
        
        y = np.array(r_results['y'])
        
        # Fit with Python
        print("\nFitting with Python...")
        model = gamlss(
            formula="y ~ x1",
            data={"y": y, "x1": x1},
            family=ZAGA(),
            trace=False
        )
        
        python_dev = model.deviance
        python_mu = model.fitted_values['mu']
        python_sigma = model.fitted_values['sigma']
        python_nu = model.fitted_values['nu']
        
        print(f"\nPython ZAGA Fit:")
        print(f"  Deviance: {python_dev:.6f}")
        print(f"  mu range: [{python_mu.min():.6f}, {python_mu.max():.6f}]")
        print(f"  sigma range: [{python_sigma.min():.6f}, {python_sigma.max():.6f}]")
        print(f"  nu range: [{python_nu.min():.6f}, {python_nu.max():.6f}]")
        
        # Compare
        r_dev = r_results['deviance']
        r_mu = np.array(r_results['mu_fitted'])
        r_sigma = np.array(r_results['sigma_fitted'])
        r_nu = np.array(r_results['nu_fitted'])
        
        print(f"\nComparison:")
        print(f"  Deviance difference: {python_dev - r_dev:.6f}")
        print(f"  Deviance relative diff: {100*(python_dev - r_dev)/r_dev:.4f}%")
        print(f"  mu RMSE: {np.sqrt(np.mean((python_mu - r_mu)**2)):.6f}")
        print(f"  sigma RMSE: {np.sqrt(np.mean((python_sigma - r_sigma)**2)):.6f}")
        print(f"  nu RMSE: {np.sqrt(np.mean((python_nu - r_nu)**2)):.6f}")
        
        # Calculate deviance with R parameters in Python
        zaga_family = ZAGA()
        python_dev_with_r_params = np.sum(zaga_family.g_dev_inc(y, r_mu, r_sigma, r_nu))
        
        print(f"\nDiagnostic:")
        print(f"  Python deviance with R parameters: {python_dev_with_r_params:.6f}")
        print(f"  R deviance: {r_dev:.6f}")
        print(f"  Difference: {python_dev_with_r_params - r_dev:.10f}")
        print(f"  (Should be ~0 if likelihood formula is correct)")
        
    finally:
        Path(r_file).unlink()
        if Path('zaga_r_fit.json').exists():
            Path('zaga_r_fit.json').unlink()
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    compare_nbi_fit()
    compare_zaga_fit()
