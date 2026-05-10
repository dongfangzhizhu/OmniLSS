"""Trace Python fitting process to see what's happening."""

import numpy as np
import subprocess
import tempfile
from pathlib import Path

# Enable float64
import jax
jax.config.update("jax_enable_x64", True)

from omnilss import gamlss
from omnilss.distributions import NBI

# Generate data
r_code = """
library(gamlss)
set.seed(42)
n <- 100
x1 <- rnorm(n)
mu_true <- exp(1 + 0.5*x1)
sigma_true <- 0.5
y <- rNBI(n, mu=mu_true, sigma=sigma_true)
write.csv(data.frame(y=y, x1=x1), "nbi_data.csv", row.names=FALSE)
"""

with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
    f.write(r_code)
    r_file = f.name

try:
    subprocess.run(['Rscript', r_file], capture_output=True, text=True, timeout=60, cwd='.')
finally:
    Path(r_file).unlink()

# Load data
import pandas as pd
data = pd.read_csv('nbi_data.csv')
y = data['y'].values
x1 = data['x1'].values

print("=" * 80)
print("Tracing Python NBI Fitting Process")
print("=" * 80)

# Fit with trace
model = gamlss(
    formula="y ~ x1",
    data={"y": y, "x1": x1},
    family=NBI(),
    trace=True,  # Enable tracing
    n_cycles=20
)

print(f"\nFinal Results:")
print(f"  Deviance: {model.deviance:.6f}")
print(f"  Sigma: {model.fitted_values['sigma'][0]:.6f}")
print(f"  Number of cycles: {model.iter if hasattr(model, 'iter') else 'N/A'}")

print(f"\nR Results for comparison:")
print(f"  Deviance: 405.8704")
print(f"  Sigma: 0.6296997")
print(f"  Iterations: 2")

print("\n" + "=" * 80)

# Clean up
if Path('nbi_data.csv').exists():
    Path('nbi_data.csv').unlink()
