"""Detailed trace of RS algorithm to compare with R."""

import numpy as np
import jax.numpy as jnp
import sys
sys.path.insert(0, 'omnilss/src')

from omnilss.distributions import resolve_family

# Set random seed for reproducibility
np.random.seed(42)

# Generate test data (same as in R)
n = 100
x1 = np.random.normal(0, 1, n)
mu_true = np.exp(1 + 0.5 * x1)
sigma_true = 0.5
y = np.random.negative_binomial(n=1/sigma_true**2, p=1/(1 + mu_true * sigma_true**2), size=n)

# Create design matrix
X = np.column_stack([np.ones(n), x1])

# Get family
family = resolve_family("NBI")

# Initial values
mu = np.mean(y) * np.ones(n)
sigma = 0.5 * np.ones(n)

print("=" * 70)
print("Detailed RS Algorithm Trace for NBI")
print("=" * 70)
print(f"Initial mu (mean): {np.mean(mu):.6f}")
print(f"Initial sigma: {sigma[0]:.6f}")

# Compute initial deviance
dev_incr = np.asarray(family.g_dev_inc(y=y, mu=mu, sigma=sigma), dtype=np.float64)
initial_dev = np.sum(dev_incr)
print(f"Initial deviance: {initial_dev:.6f}")
print("=" * 70)

# Manually trace first iteration for sigma
print("\nTracing SIGMA update (first iteration):")
print("-" * 70)

# Get link functions
link_fun = family.link_functions["sigma"]
link_inv = family.link_inverses["sigma"]
link_deriv_func = family.link_derivatives["sigma"]

# Get derivative functions
dldp = family.score_functions["sigma"]
d2ldp2 = family.hessian_functions["sigma"]

# Step 1: Compute eta
eta = np.asarray(link_fun(jnp.asarray(sigma, dtype=jnp.float64)), dtype=np.float64)
print(f"1. Initial eta: mean={np.mean(eta):.6f}, std={np.std(eta):.6f}")

# Step 2: Compute derivatives at initial sigma
first_deriv = np.asarray(dldp(y=y, mu=mu, sigma=sigma), dtype=np.float64)
second_deriv = np.asarray(d2ldp2(y=y, mu=mu, sigma=sigma), dtype=np.float64)
print(f"2. First derivative: mean={np.mean(first_deriv):.6f}, std={np.std(first_deriv):.6f}")
print(f"   Second derivative: mean={np.mean(second_deriv):.6f}, std={np.std(second_deriv):.6f}")

# Step 3: Compute link derivative
dmu_deta = np.asarray(link_deriv_func(jnp.asarray(eta, dtype=jnp.float64)), dtype=np.float64)
link_deriv = 1.0 / (dmu_deta + 1e-15)
print(f"3. Link derivative (deta/dsigma): mean={np.mean(link_deriv):.6f}")

# Step 4: Ensure second derivative is negative
second_deriv = np.where(second_deriv < -1e-15, second_deriv, -1e-15)

# Step 5: Compute working weights
working_weights = -(second_deriv / (link_deriv ** 2))
working_weights = np.clip(working_weights, 1e-10, 1e10)
print(f"4. Working weights: mean={np.mean(working_weights):.6f}, min={np.min(working_weights):.6f}, max={np.max(working_weights):.6f}")

# Step 6: Compute working response
offset = np.zeros(n)
working_response = (eta - offset) + first_deriv / (link_deriv * working_weights)
print(f"5. Working response: mean={np.mean(working_response):.6f}, std={np.std(working_response):.6f}")

# Step 7: Fit weighted least squares
W = working_weights
sqrt_W = np.sqrt(W)
WX = X[:, 0:1] * sqrt_W[:, None]  # Only intercept for sigma
Wy = working_response * sqrt_W

coef, _, _, _ = np.linalg.lstsq(WX, Wy, rcond=None)
print(f"6. Fitted coefficient: {coef[0]:.6f}")

# Step 8: Update eta and sigma
eta_new = X[:, 0:1] @ coef + offset
sigma_new = np.asarray(link_inv(jnp.asarray(eta_new, dtype=jnp.float64)), dtype=np.float64)
print(f"7. New eta: mean={np.mean(eta_new):.6f}")
print(f"   New sigma: mean={np.mean(sigma_new):.6f}")

# Step 9: Compute new deviance
dev_incr_new = np.asarray(family.g_dev_inc(y=y, mu=mu, sigma=sigma_new), dtype=np.float64)
new_dev = np.sum(dev_incr_new)
print(f"8. New deviance: {new_dev:.6f}")
print(f"   Change: {initial_dev - new_dev:.6f}")

print("=" * 70)
print("\nNow let's check what R would compute...")
print("R uses log link for sigma, so:")
print(f"  Initial sigma = {sigma[0]:.6f}")
print(f"  Initial eta = log(sigma) = {np.log(sigma[0]):.6f}")
print(f"  After fitting, new eta = {eta_new[0]:.6f}")
print(f"  New sigma = exp(eta) = {np.exp(eta_new[0]):.6f}")
print("=" * 70)
