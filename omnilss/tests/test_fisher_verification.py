"""Verify Fisher information formulas numerically.

This script numerically computes the expected Hessian (Fisher information)
and compares it with the analytical formulas we're using.
"""

import numpy as np
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omnilss.distributions_b1 import PARETO2, _pareto2_log_pdf
from omnilss.distributions_b2 import LOGNO2, _logno2_log_pdf
import jax
import jax.numpy as jnp


def numerical_fisher_info(log_pdf_func, param_name, mu, sigma, n_samples=10000, seed=42):
    """Numerically compute Fisher information by Monte Carlo.
    
    I(θ) = -E[∂²log L/∂θ²]
    
    We approximate this by:
    1. Generate many samples from the distribution
    2. Compute ∂²log L/∂θ² for each sample
    3. Take the average
    4. Negate
    """
    # Generate samples from the distribution
    np.random.seed(seed)
    
    # For PARETO2/Lomax
    from scipy.stats import lomax
    shape = 1.0 / sigma
    y_samples = lomax.rvs(shape, scale=mu, size=n_samples)
    
    # Compute second derivative for each sample
    def hessian_func(y_val, mu_val, sigma_val):
        if param_name == "mu":
            return jax.grad(jax.grad(lambda m: log_pdf_func(y_val, m, sigma_val)))(mu)
        elif param_name == "sigma":
            return jax.grad(jax.grad(lambda s: log_pdf_func(y_val, mu_val, s)))(sigma)
    
    hessians = []
    for y in y_samples:
        h = hessian_func(jnp.array(y), jnp.array(mu), jnp.array(sigma))
        hessians.append(float(h))
    
    # Fisher information is -E[Hessian]
    fisher_info = -np.mean(hessians)
    fisher_std = np.std(hessians) / np.sqrt(n_samples)
    
    return fisher_info, fisher_std


def test_pareto2_fisher():
    """Test PARETO2 Fisher information."""
    print("=" * 70)
    print("PARETO2 Fisher Information Verification")
    print("=" * 70)
    
    mu = 2.0
    sigma = 0.8  # < 1 for finite moments
    
    print(f"\nParameters:")
    print(f"  mu = {mu}")
    print(f"  sigma = {sigma}")
    print(f"  shape = 1/sigma = {1.0/sigma:.4f}")
    
    # Get analytical Fisher information (current implementation)
    family = PARETO2()
    d2ldm2 = family.hessian_functions["mu"]
    d2lds2 = family.hessian_functions["sigma"]
    
    # Dummy y value (shouldn't matter for expected Hessian)
    y_dummy = jnp.array([1.0])
    
    analytical_mu = float(np.asarray(d2ldm2(y_dummy, jnp.array(mu), jnp.array(sigma))).reshape(-1)[0])
    analytical_sigma = float(np.asarray(d2lds2(y_dummy, jnp.array(mu), jnp.array(sigma))).reshape(-1)[0])
    
    print(f"\n" + "-" * 70)
    print("Fisher Information for MU")
    print("-" * 70)
    
    # Numerical Fisher information
    numerical_mu, std_mu = numerical_fisher_info(_pareto2_log_pdf, "mu", mu, sigma)
    
    print(f"\nCurrent analytical formula: -(1/σ + 1)/μ²")
    print(f"  Value: {analytical_mu:.6f}")
    
    print(f"\nNumerical (Monte Carlo, n=10000):")
    print(f"  Value: {numerical_mu:.6f} ± {std_mu:.6f}")
    
    # Alternative formula from our derivation
    alternative_mu = 2.0 / (mu**2 * (1.0 - sigma))
    print(f"\nAlternative formula: 2/(μ²(1-σ))")
    print(f"  Value: {alternative_mu:.6f}")
    
    print(f"\nComparison:")
    print(f"  Current vs Numerical: {abs(analytical_mu - numerical_mu):.6f}")
    print(f"  Alternative vs Numerical: {abs(alternative_mu - numerical_mu):.6f}")
    
    if abs(analytical_mu - numerical_mu) < abs(alternative_mu - numerical_mu):
        print(f"  ✓ Current formula is closer to numerical!")
    else:
        print(f"  ✗ Alternative formula is closer to numerical!")
    
    print(f"\n" + "-" * 70)
    print("Fisher Information for SIGMA")
    print("-" * 70)
    
    # Numerical Fisher information
    numerical_sigma, std_sigma = numerical_fisher_info(_pareto2_log_pdf, "sigma", mu, sigma)
    
    print(f"\nCurrent analytical formula: -2/σ²")
    print(f"  Value: {analytical_sigma:.6f}")
    
    print(f"\nNumerical (Monte Carlo, n=10000):")
    print(f"  Value: {numerical_sigma:.6f} ± {std_sigma:.6f}")
    
    print(f"\nComparison:")
    print(f"  Difference: {abs(analytical_sigma - numerical_sigma):.6f}")
    
    if abs(analytical_sigma - numerical_sigma) < 3 * std_sigma:
        print(f"  ✓ Analytical formula matches numerical (within 3σ)!")
    else:
        print(f"  ✗ Analytical formula differs from numerical!")


def test_logno2_fisher():
    """Test LOGNO2 Fisher information (should be correct)."""
    print("\n\n" + "=" * 70)
    print("LOGNO2 Fisher Information Verification (Control)")
    print("=" * 70)
    
    mu = 2.0
    sigma = 0.5
    
    print(f"\nParameters:")
    print(f"  mu = {mu}")
    print(f"  sigma = {sigma}")
    
    # Generate samples
    np.random.seed(42)
    n_samples = 10000
    # log(Y) ~ N(log(mu), sigma^2)
    log_y = np.random.normal(np.log(mu), sigma, n_samples)
    y_samples = np.exp(log_y)
    
    # Compute Hessian for mu
    def hessian_mu(y_val):
        return jax.grad(jax.grad(lambda m: _logno2_log_pdf(y_val, m, jnp.array(sigma))))(jnp.array(mu))
    
    hessians_mu = [float(hessian_mu(jnp.array(y))) for y in y_samples]
    numerical_mu = -np.mean(hessians_mu)
    std_mu = np.std(hessians_mu) / np.sqrt(n_samples)
    
    # Analytical
    analytical_mu = -1.0 / (mu**2 * sigma**2)
    
    print(f"\n" + "-" * 70)
    print("Fisher Information for MU")
    print("-" * 70)
    
    print(f"\nAnalytical formula: -1/(μ²σ²)")
    print(f"  Value: {analytical_mu:.6f}")
    
    print(f"\nNumerical (Monte Carlo, n=10000):")
    print(f"  Value: {numerical_mu:.6f} ± {std_mu:.6f}")
    
    print(f"\nComparison:")
    print(f"  Difference: {abs(analytical_mu - numerical_mu):.6f}")
    
    if abs(analytical_mu - numerical_mu) < 3 * std_mu:
        print(f"  ✓ Analytical formula matches numerical (within 3σ)!")
    else:
        print(f"  ✗ Analytical formula differs from numerical!")
    
    # Compute Hessian for sigma
    def hessian_sigma(y_val):
        return jax.grad(jax.grad(lambda s: _logno2_log_pdf(y_val, jnp.array(mu), s)))(jnp.array(sigma))
    
    hessians_sigma = [float(hessian_sigma(jnp.array(y))) for y in y_samples]
    numerical_sigma = -np.mean(hessians_sigma)
    std_sigma = np.std(hessians_sigma) / np.sqrt(n_samples)
    
    # Analytical
    analytical_sigma = -2.0 / sigma**2
    
    print(f"\n" + "-" * 70)
    print("Fisher Information for SIGMA")
    print("-" * 70)
    
    print(f"\nAnalytical formula: -2/σ²")
    print(f"  Value: {analytical_sigma:.6f}")
    
    print(f"\nNumerical (Monte Carlo, n=10000):")
    print(f"  Value: {numerical_sigma:.6f} ± {std_sigma:.6f}")
    
    print(f"\nComparison:")
    print(f"  Difference: {abs(analytical_sigma - numerical_sigma):.6f}")
    
    if abs(analytical_sigma - numerical_sigma) < 3 * std_sigma:
        print(f"  ✓ Analytical formula matches numerical (within 3σ)!")
    else:
        print(f"  ✗ Analytical formula differs from numerical!")


if __name__ == "__main__":
    # Test LOGNO2 first (should be correct, as a control)
    test_logno2_fisher()
    
    # Test PARETO2 (this is what we're investigating)
    test_pareto2_fisher()
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\nThis numerical verification helps us determine which Fisher")
    print("information formula is correct by comparing analytical formulas")
    print("with Monte Carlo estimates from actual samples.")
