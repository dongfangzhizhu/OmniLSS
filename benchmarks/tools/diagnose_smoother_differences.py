"""Diagnose differences between Python and R smoother implementations.

This script compares the fitted values from Python smoothers (pb, ps, cs)
with R gamlss smoothers to identify the source of discrepancies.
"""

# ── JAX configuration must happen before any jax import ──────────────────────
import os
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import subprocess
import tempfile
import json
from pathlib import Path
import sys

# Add omnilss to path
sys.path.insert(0, str(Path(__file__).parent / "omnilss" / "src"))

# Verify JAX 64-bit mode is enabled
import jax
print(f"JAX 64-bit mode: {jax.config.jax_enable_x64}")
print(f"JAX default dtype: {jax.numpy.zeros(1).dtype}")
print()

from omnilss import gamlss
from omnilss.distributions import resolve_family


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


def run_r_smoother(data: dict, formula: str) -> dict:
    """Run R gamlss with smoother and return results."""
    r_script = r"""
suppressMessages(library(gamlss))
suppressMessages(library(jsonlite))
args      <- commandArgs(trailingOnly=TRUE)
data_file <- args[1]
formula   <- args[2]

df <- read.csv(data_file)
mu_formula <- as.formula(formula)

fit <- tryCatch(
  gamlss(mu_formula, family="NO", data=df, trace=FALSE),
  error=function(e) NULL
)

if (is.null(fit)) {
  cat(toJSON(list(success=FALSE, error="gamlss failed"), auto_unbox=TRUE), "\n")
} else {
  # Get smoother details
  mu_terms <- fit$mu.terms
  mu_x <- fit$mu.x
  
  cat(toJSON(list(
    success   = TRUE,
    deviance  = fit$G.deviance,
    mu_fitted = as.numeric(fit$mu.fv),
    mu_coef   = as.numeric(fit$mu.coefficients),
    n_coef    = length(fit$mu.coefficients),
    edf       = fit$mu.df
  ), auto_unbox=TRUE), "\n")
}
"""
    
    csv_path = write_csv(data)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as f:
        f.write(r_script)
        r_path = f.name
    
    try:
        result = subprocess.run(
            ["Rscript", r_path, csv_path, formula],
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        if result.returncode != 0:
            return {"success": False, "error": result.stderr}
        
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            return {"success": False, "error": "no output"}
        
        return json.loads(lines[-1])
    finally:
        Path(csv_path).unlink(missing_ok=True)
        Path(r_path).unlink(missing_ok=True)


def diagnose_smoother(smoother: str):
    """Diagnose differences for a specific smoother."""
    print(f"\n{'='*70}")
    print(f"Diagnosing {smoother} smoother")
    print(f"{'='*70}\n")
    
    # Generate test data
    rng = np.random.RandomState(42)
    n = 300
    x = np.linspace(0, 10, n)
    y = np.sin(x) + rng.normal(0, 0.3, n)
    data = {"y": y, "x": x}
    formula = f"y ~ {smoother}(x)"
    
    # Fit with Python
    print("Fitting with Python...")
    try:
        family = resolve_family("NO")
        model = gamlss(formula, family=family, data=data)
        py_fitted = np.asarray(model.fitted_values["mu"])
        py_dev = float(model.deviance)
        py_coef = np.asarray(model.coefficients["mu"])
        print(f"  ✓ Success")
        print(f"    Deviance: {py_dev:.6f}")
        print(f"    N coefficients: {len(py_coef)}")
        print(f"    Fitted range: [{py_fitted.min():.4f}, {py_fitted.max():.4f}]")
        print(f"    Fitted mean: {py_fitted.mean():.4f}")
        print(f"    Fitted std: {py_fitted.std():.4f}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return
    
    # Fit with R
    print("\nFitting with R...")
    r_result = run_r_smoother(data, formula)
    
    if not r_result.get("success", False):
        print(f"  ✗ Failed: {r_result.get('error', 'unknown')}")
        return
    
    r_fitted = np.array(r_result["mu_fitted"])
    r_dev = float(r_result["deviance"])
    r_coef = np.array(r_result["mu_coef"])
    r_edf = r_result.get("edf", "unknown")
    
    print(f"  ✓ Success")
    print(f"    Deviance: {r_dev:.6f}")
    print(f"    N coefficients: {len(r_coef)}")
    print(f"    EDF: {r_edf}")
    print(f"    Fitted range: [{r_fitted.min():.4f}, {r_fitted.max():.4f}]")
    print(f"    Fitted mean: {r_fitted.mean():.4f}")
    print(f"    Fitted std: {r_fitted.std():.4f}")
    
    # Compare
    print("\nComparison:")
    print(f"  Deviance diff: {abs(py_dev - r_dev):.6e}")
    print(f"  Coefficient count diff: {len(py_coef) - len(r_coef)}")
    
    # Fitted values comparison
    abs_err = np.abs(py_fitted - r_fitted)
    rel_err = abs_err / (np.abs(r_fitted) + 1e-10)
    
    print(f"\n  Fitted values errors:")
    print(f"    Max absolute: {abs_err.max():.6e}")
    print(f"    Mean absolute: {abs_err.mean():.6e}")
    print(f"    Max relative: {rel_err.max():.6e}")
    print(f"    Mean relative: {rel_err.mean():.6e}")
    print(f"    RMSE: {np.sqrt(np.mean((py_fitted - r_fitted)**2)):.6e}")
    print(f"    Correlation: {np.corrcoef(py_fitted, r_fitted)[0,1]:.6f}")
    
    # Check if the shapes are similar
    print(f"\n  Shape comparison:")
    print(f"    Python range: {py_fitted.max() - py_fitted.min():.4f}")
    print(f"    R range: {r_fitted.max() - r_fitted.min():.4f}")
    print(f"    Range ratio: {(py_fitted.max() - py_fitted.min()) / (r_fitted.max() - r_fitted.min()):.4f}")
    
    # Sample some points
    print(f"\n  Sample fitted values (first 10):")
    print(f"    Python: {py_fitted[:10]}")
    print(f"    R:      {r_fitted[:10]}")
    print(f"    Diff:   {(py_fitted - r_fitted)[:10]}")


def main():
    """Main diagnostic routine."""
    print("OmniLSS Smoother Diagnostic Tool")
    print("=" * 70)
    
    # Check if R is available
    try:
        result = subprocess.run(
            ["Rscript", "-e", "suppressMessages(library(gamlss)); cat('ok')"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0 or "ok" not in result.stdout:
            print("ERROR: R or gamlss package not available")
            return
    except Exception as e:
        print(f"ERROR: Cannot run R: {e}")
        return
    
    print("✓ R and gamlss package available\n")
    
    # Diagnose each smoother
    for smoother in ["pb", "ps", "cs"]:
        try:
            diagnose_smoother(smoother)
        except Exception as e:
            print(f"\nERROR diagnosing {smoother}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
