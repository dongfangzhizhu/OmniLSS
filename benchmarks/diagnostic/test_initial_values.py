"""Test effect of initial values on NBI fitting."""

import numpy as np
import subprocess
import tempfile
from pathlib import Path

# Enable float64
import jax
jax.config.update("jax_enable_x64", True)

from omnilss import gamlss
from omnilss.distributions import NBI

# Generate data in R
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

# Check R's initial sigma
mu_init <- (y + mean(y))/2
sigma_init_formula <- max( ((var(y)-mean(y))/(mean(y)^2)), 0.1)
cat("R initial sigma:", sigma_init_formula, "\\n")
"""

with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
    f.write(r_code)
    r_file = f.name

try:
    result = subprocess.run(['Rscript', r_file], capture_output=True, text=True, timeout=60, cwd='.')
    print(result.stdout)
finally:
    Path(r_file).unlink()

# Load data
import pandas as pd
data = pd.read_csv('nbi_data.csv')
y = data['y'].values
x1 = data['x1'].values

print("=" * 80)
print("Testing Python initial sigma calculation")
print("=" * 80)

# Calculate Python's initial sigma (without the 0.1 floor)
mean_y = np.mean(y)
var_y = np.var(y)
python_sigma_init = max((var_y - mean_y) / (mean_y ** 2), np.finfo(np.float64).eps)
r_sigma_init = max((var_y - mean_y) / (mean_y ** 2), 0.1)

print(f"\nInitial sigma calculations:")
print(f"  Python (with eps floor): {python_sigma_init:.6f}")
print(f"  R (with 0.1 floor): {r_sigma_init:.6f}")
print(f"  Difference: {r_sigma_init - python_sigma_init:.6f}")

# Test with different initial values
print(f"\n{'Initial sigma':<15} {'Final Deviance':<20} {'Final sigma':<15}")
print("-" * 50)

for init_sigma in [python_sigma_init, 0.1, 0.5, 1.0, r_sigma_init]:
    # Manually set initial sigma by modifying the fitting process
    # This is a hack - we'll need to modify the actual code
    model = gamlss(
        formula="y ~ x1",
        data={"y": y, "x1": x1},
        family=NBI(),
        trace=False
    )
    
    python_dev = model.deviance
    python_sigma = model.fitted_values['sigma'][0]
    
    print(f"{init_sigma:<15.6f} {python_dev:<20.6f} {python_sigma:<15.6f}")

print("\nR result: deviance = 405.8704, sigma = 0.6296997")
print("\n" + "=" * 80)

# Clean up
if Path('nbi_data.csv').exists():
    Path('nbi_data.csv').unlink()
