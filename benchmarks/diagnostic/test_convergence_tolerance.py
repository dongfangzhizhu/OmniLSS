"""Test effect of convergence tolerance on NBI fitting."""

import numpy as np
import subprocess
import tempfile
from pathlib import Path

# Enable float64
import jax
jax.config.update("jax_enable_x64", True)

from omnilss import gamlss
from omnilss.distributions import NBI

# Generate data in R first
r_code = """
library(gamlss)
set.seed(42)
n <- 100
x1 <- rnorm(n)
mu_true <- exp(1 + 0.5*x1)
sigma_true <- 0.5
y <- rNBI(n, mu=mu_true, sigma=sigma_true)
model <- gamlss(y ~ x1, family=NBI(), trace=FALSE)
mu_fitted <- fitted(model, "mu")
sigma_fitted <- fitted(model, "sigma")
dev <- deviance(model)
write.csv(data.frame(y=y, x1=x1, mu=mu_fitted, sigma=sigma_fitted), 
          "nbi_data.csv", row.names=FALSE)
cat("R deviance:", dev, "\\n")
cat("R sigma:", unique(sigma_fitted), "\\n")
"""

with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
    f.write(r_code)
    r_file = f.name

try:
    result = subprocess.run(['Rscript', r_file], capture_output=True, text=True, timeout=60, cwd='.')
    print(result.stdout)
finally:
    Path(r_file).unlink()

# Load the data from R
import pandas as pd
data = pd.read_csv('nbi_data.csv')
y = data['y'].values
x1 = data['x1'].values
r_mu = data['mu'].values
r_sigma = data['sigma'].values

print("=" * 80)
print("Testing NBI with different convergence tolerances")
print("=" * 80)

# R deviance (from previous run)
r_dev = 405.8704

# Calculate deviance with R parameters
nbi_family = NBI()
python_dev_with_r_params = np.sum(nbi_family.g_dev_inc(y, r_mu, r_sigma))
print(f"\nR deviance: {r_dev:.6f}")
print(f"Python deviance with R parameters: {python_dev_with_r_params:.6f}")
print(f"Difference: {python_dev_with_r_params - r_dev:.10f} (should be ~0)")

# Test different tolerances
tolerances = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

print(f"\n{'Tolerance':<12} {'Deviance':<15} {'Diff from R':<15} {'sigma':<12}")
print("-" * 60)

for tol in tolerances:
    model = gamlss(
        formula="y ~ x1",
        data={"y": y, "x1": x1},
        family=NBI(),
        trace=False,
        n_cycles=100,  # More iterations
        tolerance=tol
    )
    
    python_dev = model.deviance
    python_sigma = model.fitted_values['sigma'][0]
    diff = python_dev - r_dev
    
    print(f"{tol:<12.0e} {python_dev:<15.6f} {diff:<15.6f} {python_sigma:<12.6f}")

print("\nR sigma: 0.6296997")
print("\n" + "=" * 80)

# Clean up
if Path('nbi_data.csv').exists():
    Path('nbi_data.csv').unlink()
