"""Investigate ps smoother lambda selection differences between Python and R.

This script helps understand why Python's ps() smoother produces different
results than R's ps() even with aligned parameters.
"""

import os
os.environ["JAX_ENABLE_X64"] = "true"
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import subprocess
import tempfile
import json
from pathlib import Path

def write_csv(data: dict) -> str:
    """Write data to CSV file."""
    import csv
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        writer = csv.writer(f)
        keys = list(data.keys())
        writer.writerow(keys)
        for row in zip(*[data[k] for k in keys]):
            writer.writerow(row)
        return f.name


def test_ps_with_r():
    """Test ps smoother and compare with R."""
    # Generate test data
    rng = np.random.RandomState(42)
    n = 300
    x = np.linspace(0, 10, n)
    y = np.sin(x) + rng.normal(0, 0.3, n)
    
    # Test with Python
    print("=" * 70)
    print("Testing Python ps() with different configurations")
    print("=" * 70)
    
    from omnilss.smoothers.ps import fit_pspline_smooth
    
    # Test 1: Default (should use df=3, which becomes df=5)
    print("\n1. Python ps() with defaults:")
    result1 = fit_pspline_smooth(x, y)
    print(f"   Lambda: {result1.lambda_:.6f}")
    print(f"   EDF: {result1.edf:.4f}")
    print(f"   N basis: {result1.design_matrix.shape[1]}")
    print(f"   Deviance (RSS): {result1.rss:.6f}")
    print(f"   ps_intervals: {result1.ps_intervals}")
    print(f"   degree: {result1.degree}")
    print(f"   order: {result1.order}")
    
    # Test 2: Explicit df=5 (matching R's internal df after adding 2)
    print("\n2. Python ps() with explicit df=5:")
    result2 = fit_pspline_smooth(x, y, df=5)
    print(f"   Lambda: {result2.lambda_:.6f}")
    print(f"   EDF: {result2.edf:.4f}")
    print(f"   Deviance (RSS): {result2.rss:.6f}")
    
    # Test 3: Explicit df=3 (user's input before adding 2)
    print("\n3. Python ps() with explicit df=3:")
    result3 = fit_pspline_smooth(x, y, df=3)
    print(f"   Lambda: {result3.lambda_:.6f}")
    print(f"   EDF: {result3.edf:.4f}")
    print(f"   Deviance (RSS): {result3.rss:.6f}")
    
    # Test with R via gamlss
    print("\n" + "=" * 70)
    print("Testing R ps() via gamlss")
    print("=" * 70)
    
    r_script = r"""
    suppressMessages(library(gamlss))
    suppressMessages(library(jsonlite))
    args <- commandArgs(trailingOnly=TRUE)
    data_file <- args[1]
    
    df <- read.csv(data_file)
    
    # Fit with ps() using default parameters
    fit <- gamlss(y ~ ps(x), family="NO", data=df, trace=FALSE)
    
    # Get ps smoother details
    mu_coef <- fit$mu.coefficients
    mu_fv <- fit$mu.fv
    global_dev <- fit$G.deviance
    
    # Try to extract ps-specific information
    mu_terms <- fit$mu.terms
    mu_s <- fit$mu.s
    
    # Get lambda and edf if available
    lambda_val <- NA
    edf_val <- fit$mu.df
    
    if (!is.null(mu_s) && length(mu_s) > 0) {
        ps_info <- mu_s[[1]]
        if (!is.null(ps_info$lambda)) {
            lambda_val <- ps_info$lambda
        }
        if (!is.null(ps_info$edf)) {
            edf_val <- ps_info$edf
        }
    }
    
    cat(toJSON(list(
        success = TRUE,
        global_deviance = global_dev,
        mu_fitted = as.numeric(mu_fv),
        n_coef = length(mu_coef),
        lambda = lambda_val,
        edf = edf_val,
        mu_coef = as.numeric(mu_coef)
    ), auto_unbox=TRUE), "\n")
    """
    
    # Write data and run R
    data = {"y": y, "x": x}
    csv_path = write_csv(data)
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as f:
        f.write(r_script)
        r_path = f.name
    
    try:
        result = subprocess.run(
            ["Rscript", r_path, csv_path],
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        if result.returncode != 0:
            print(f"\nR script failed:")
            print(result.stderr)
            return
        
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if lines:
            r_result = json.loads(lines[-1])
            print(f"\nR ps() results:")
            print(f"   Global deviance: {r_result['global_deviance']:.6f}")
            print(f"   N coefficients: {r_result['n_coef']}")
            print(f"   Lambda: {r_result['lambda']}")
            print(f"   EDF: {r_result['edf']}")
            print(f"   Coefficients: {r_result['mu_coef']}")
            
            # Compare with Python
            print("\n" + "=" * 70)
            print("Comparison")
            print("=" * 70)
            print(f"Python (default) EDF: {result1.edf:.4f}")
            print(f"R EDF: {r_result['edf']}")
            print(f"EDF difference: {abs(result1.edf - r_result['edf']):.4f}")
            print(f"\nPython (default) Lambda: {result1.lambda_:.6f}")
            print(f"R Lambda: {r_result['lambda']}")
            if not np.isnan(r_result['lambda']):
                print(f"Lambda ratio (Python/R): {result1.lambda_ / r_result['lambda']:.6f}")
            
    finally:
        Path(csv_path).unlink(missing_ok=True)
        Path(r_path).unlink(missing_ok=True)


def test_ps_internals():
    """Test ps smoother internal calculations."""
    print("\n" + "=" * 70)
    print("Testing ps() internal calculations")
    print("=" * 70)
    
    from omnilss.smoothers.ps import fit_pspline_smooth, _create_ps_knots
    from omnilss.smoothers.penalties import penalty_matrix, find_lambda_for_df
    import jax.numpy as jnp
    
    # Generate test data
    rng = np.random.RandomState(42)
    n = 300
    x = np.linspace(0, 10, n)
    y = np.sin(x) + rng.normal(0, 0.3, n)
    
    # Create knots
    knots = _create_ps_knots(x, ps_intervals=20, degree=3)
    print(f"\nKnots: {len(knots)} total")
    print(f"Interior knots: {len(knots) - 2 * (3 + 1)}")
    
    # Create design matrix
    from omnilss.smoothers.bsplines import bspline_basis
    X = np.asarray(bspline_basis(
        jnp.array(x, dtype=jnp.float64),
        jnp.array(knots, dtype=jnp.float64),
        degree=3
    ))
    print(f"Design matrix shape: {X.shape}")
    print(f"Number of basis functions: {X.shape[1]}")
    
    # Create penalty matrix with order=3
    P = penalty_matrix(X.shape[1], order=3)
    print(f"Penalty matrix shape: {P.shape}")
    print(f"Penalty order: 3")
    
    # Test lambda for different target df
    print("\nLambda for different target df:")
    for target_df in [3, 5, 7, 10]:
        lambda_opt = find_lambda_for_df(
            jnp.array(X), jnp.array(P), float(target_df), jnp.ones(n)
        )
        print(f"   df={target_df}: lambda={lambda_opt:.6f}")


if __name__ == "__main__":
    # Run tests
    test_ps_with_r()
    test_ps_internals()
