"""Test PARETO2 fitting in Python."""

import numpy as np
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omnilss.distributions_b1 import PARETO2
from omnilss.fitting import gamlss
import subprocess

# Generate data using R
r_script = """
library(gamlss.dist)
set.seed(42)
y <- rPARETO2(100, mu=2.0, sigma=0.5)
cat(paste(y, collapse=" "))
"""

result = subprocess.run(
    ["Rscript", "-e", r_script],
    capture_output=True,
    text=True
)

y = np.array([float(v) for v in result.stdout.strip().split()])

print("Data summary:")
print(f"  n = {len(y)}")
print(f"  mean = {np.mean(y):.4f}")
print(f"  min = {np.min(y):.4f}")
print(f"  max = {np.max(y):.4f}")

print("\nFitting with Python...")
family = PARETO2()
data = {"y": y}

try:
    # Import control
    from omnilss.controls import gamlss_control
    
    # Test with default iterations (20)
    print("\n--- Default (20 iterations) ---")
    model1 = gamlss(formula="y ~ 1", family=family, data=data)
    print(f"  Iterations: {model1.iter}")
    print(f"  Mu coefficient: {model1.coefficients['mu'][0]:.6f}")
    print(f"  Sigma coefficient: {model1.coefficients['sigma'][0]:.6f}")
    print(f"  Deviance: {model1.deviance:.6f}")
    
    # Test with more iterations (100)
    print("\n--- More iterations (100) ---")
    control = gamlss_control(n_cyc=100)
    model2 = gamlss(formula="y ~ 1", family=family, data=data, control=control)
    print(f"  Iterations: {model2.iter}")
    print(f"  Mu coefficient: {model2.coefficients['mu'][0]:.6f}")
    print(f"  Sigma coefficient: {model2.coefficients['sigma'][0]:.6f}")
    print(f"  Deviance: {model2.deviance:.6f}")
    
    print("\nExpected from R (33 iterations):")
    print("  Mu coefficient: 0.847590")
    print("  Sigma coefficient: -0.705677")
    print("  Deviance: 327.136900")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
