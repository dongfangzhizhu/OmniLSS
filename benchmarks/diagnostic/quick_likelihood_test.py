"""Quick test of NBI and ZAGA likelihood formulas."""

import numpy as np
import jax.numpy as jnp
from jax.scipy.special import gammaln

# Enable float64
import jax
jax.config.update("jax_enable_x64", True)

print("Testing NBI log-likelihood formula")
print("=" * 60)

# Single observation test
y = 5.0
mu = 3.0
sigma = 0.5

# Python calculation (our implementation)
size = 1.0 / sigma
python_ll = (
    gammaln(y + size)
    - gammaln(size)
    - gammaln(y + 1.0)
    + size * jnp.log(size / (size + mu))
    + y * jnp.log(mu / (size + mu))
)

print(f"y = {y}, mu = {mu}, sigma = {sigma}")
print(f"size (1/sigma) = {size}")
print(f"\nPython log-likelihood: {python_ll:.10f}")
print(f"Python deviance: {-2.0 * python_ll:.10f}")

# Expected R formula (from R/gamlss.dist/R/NBI.R):
# R uses dnbinom(x, size=1/sigma, mu=mu, log=TRUE)
# which is: lgamma(y+size) - lgamma(size) - lgamma(y+1) 
#           + size*log(size/(size+mu)) + y*log(mu/(size+mu))

print(f"\nFormula breakdown:")
print(f"  gammaln(y + size) = gammaln({y + size}) = {float(gammaln(y + size)):.10f}")
print(f"  gammaln(size) = gammaln({size}) = {float(gammaln(size)):.10f}")
print(f"  gammaln(y + 1) = gammaln({y + 1}) = {float(gammaln(y + 1)):.10f}")
print(f"  size * log(size/(size+mu)) = {size} * log({size}/{size+mu}) = {float(size * jnp.log(size / (size + mu))):.10f}")
print(f"  y * log(mu/(size+mu)) = {y} * log({mu}/{size+mu}) = {float(y * jnp.log(mu / (size + mu))):.10f}")

print("\n" + "=" * 60)
print("Testing ZAGA log-likelihood formula")
print("=" * 60)

# Test with positive value
y_pos = 2.5
mu = 4.0
sigma = 0.6
nu = 0.2

# Gamma parameters
sigma_sq = sigma ** 2
alpha = 1.0 / sigma_sq
beta = mu * sigma_sq

print(f"\ny = {y_pos}, mu = {mu}, sigma = {sigma}, nu = {nu}")
print(f"\nGamma parameterization:")
print(f"  alpha (shape) = 1/sigma^2 = {alpha:.6f}")
print(f"  beta (scale) = mu*sigma^2 = {beta:.6f}")
print(f"  Expected mean = alpha*beta = {alpha*beta:.6f} (should be {mu})")

# Python calculation
log_gamma = (
    (alpha - 1.0) * jnp.log(y_pos)
    - y_pos / beta
    - gammaln(alpha)
    - alpha * jnp.log(beta)
)
log_zaga = jnp.log(1.0 - nu) + log_gamma

print(f"\nPython calculation:")
print(f"  log(1-nu) = {float(jnp.log(1.0 - nu)):.10f}")
print(f"  log Gamma = {float(log_gamma):.10f}")
print(f"  log ZAGA = {float(log_zaga):.10f}")
print(f"  Deviance = {float(-2.0 * log_zaga):.10f}")

# Test with zero
y_zero = 0.0
log_zaga_zero = jnp.log(nu)
print(f"\nFor y = 0:")
print(f"  log ZAGA = log(nu) = {float(log_zaga_zero):.10f}")
print(f"  Deviance = {float(-2.0 * log_zaga_zero):.10f}")

print("\n" + "=" * 60)
print("Now run this R code to compare:")
print("=" * 60)
print("""
library(gamlss.dist)

# NBI test
y <- 5
mu <- 3
sigma <- 0.5
ll_nbi <- dNBI(y, mu=mu, sigma=sigma, log=TRUE)
dev_nbi <- -2 * ll_nbi
cat("NBI: log-likelihood =", ll_nbi, ", deviance =", dev_nbi, "\\n")

# ZAGA test (positive value)
y <- 2.5
mu <- 4
sigma <- 0.6
nu <- 0.2
ll_zaga <- dZAGA(y, mu=mu, sigma=sigma, nu=nu, log=TRUE)
dev_zaga <- -2 * ll_zaga
cat("ZAGA (y>0): log-likelihood =", ll_zaga, ", deviance =", dev_zaga, "\\n")

# ZAGA test (zero)
y <- 0
ll_zaga_zero <- dZAGA(y, mu=mu, sigma=sigma, nu=nu, log=TRUE)
dev_zaga_zero <- -2 * ll_zaga_zero
cat("ZAGA (y=0): log-likelihood =", ll_zaga_zero, ", deviance =", dev_zaga_zero, "\\n")
""")
