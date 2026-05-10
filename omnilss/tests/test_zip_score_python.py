"""Test ZIP score functions in Python."""

import numpy as np
import jax.numpy as jnp
import sys
sys.path.insert(0, 'src')

from omnilss.distributions import ZIP

# Test values
y_vals = [0, 1, 2, 3]
mu = 2.0
sigma = 0.3

zip_family = ZIP()

print("Testing ZIP score functions in Python")
print("=" * 70)
print(f"\nParameters: mu = {mu}, sigma = {sigma}\n")

for y in y_vals:
    y_arr = jnp.array([y])
    mu_arr = jnp.array([mu])
    sigma_arr = jnp.array([sigma])
    
    # Score for mu
    dldm_val = float(zip_family.score_functions["mu"](y_arr, mu_arr, sigma_arr)[0])
    
    # Score for sigma
    dlds_val = float(zip_family.score_functions["sigma"](y_arr, mu_arr, sigma_arr)[0])
    
    # Hessian for mu
    d2ldm2_val = float(zip_family.hessian_functions["mu"](y_arr, mu_arr, sigma_arr)[0])
    
    # Hessian for sigma
    d2lds2_val = float(zip_family.hessian_functions["sigma"](y_arr, mu_arr, sigma_arr)[0])
    
    print(f"y = {y}:")
    print(f"  dldm = {dldm_val:.6f}")
    print(f"  dlds = {dlds_val:.6f}")
    print(f"  d2ldm2 = {d2ldm2_val:.6f}")
    print(f"  d2lds2 = {d2lds2_val:.6f}\n")

print("\nExpected from R:")
print("-" * 70)
print("y = 0: dldm = -0.240, dldd = 2.190, d2ldm2 = -0.058, d2ldd2 = -4.798")
print("y = 1: dldm = -0.500, dldd = -1.429, d2ldm2 = -0.250, d2ldd2 = -2.041")
print("y = 2: dldm = 0.000, dldd = -1.429, d2ldm2 = -0.000, d2ldd2 = -2.041")
print("y = 3: dldm = 0.500, dldd = -1.429, d2ldm2 = -0.250, d2ldd2 = -2.041")
