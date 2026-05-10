"""Simple comparison of fitted parameters."""

import numpy as np
import subprocess
import tempfile
from pathlib import Path

# Enable float64
import jax
jax.config.update("jax_enable_x64", True)

from omnilss import gamlss
from omnilss.distributions import NBI
from omnilss.distributions_b5 import ZAGA


print("=" * 80)
print("Testing NBI: Comparing Python deviance with R-fitted parameters")
print("=" * 80)

# Generate and fit data in R, save parameters
r_code = """
library(gamlss)
set.seed(42)
n <- 100
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
dev <- deviance(model)

cat("R Fit Results:\\n")
cat("Deviance:", dev, "\\n")
cat("mu range:", range(mu_fitted), "\\n")
cat("sigma range:", range(sigma_fitted), "\\n\\n")

# Save data and parameters
write.csv(data.frame(y=y, x1=x1, mu=mu_fitted, sigma=sigma_fitted), 
          "nbi_data.csv", row.names=FALSE)
cat("Saved to nbi_data.csv\\n")
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
    print(result.stdout)
    
    # Load data
    import pandas as pd
    data = pd.read_csv('nbi_data.csv')
    
    y = data['y'].values
    x1 = data['x1'].values
    r_mu = data['mu'].values
    r_sigma = data['sigma'].values
    
    # Calculate deviance with R parameters using Python
    nbi_family = NBI()
    python_dev_with_r_params = np.sum(nbi_family.g_dev_inc(y, r_mu, r_sigma))
    
    print(f"Python deviance with R-fitted parameters: {python_dev_with_r_params:.6f}")
    print(f"(Should match R deviance if likelihood formula is correct)")
    
    # Now fit with Python
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
    
    print(f"\nPython Fit Results:")
    print(f"Deviance: {python_dev:.6f}")
    print(f"mu range: [{python_mu.min():.6f}, {python_mu.max():.6f}]")
    print(f"sigma range: [{python_sigma.min():.6f}, {python_sigma.max():.6f}]")
    
    print(f"\nParameter Comparison:")
    print(f"mu RMSE: {np.sqrt(np.mean((python_mu - r_mu)**2)):.6f}")
    print(f"sigma RMSE: {np.sqrt(np.mean((python_sigma - r_sigma)**2)):.6f}")
    
finally:
    Path(r_file).unlink()
    if Path('nbi_data.csv').exists():
        Path('nbi_data.csv').unlink()

print("\n" + "=" * 80)
