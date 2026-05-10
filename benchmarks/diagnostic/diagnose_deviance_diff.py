"""Diagnose deviance differences between Python and R."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import json

# Add omnilss to path
sys.path.insert(0, str(Path(__file__).parent.parent / "omnilss" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "omnilss" / "tests"))

from omnilss.distributions import resolve_family
from omnilss.fitting import gamlss
from rbus.r_bridge import RBridge


def diagnose_bi_deviance():
    """Diagnose BI deviance differences."""
    print("=" * 80)
    print("Diagnosing BI Deviance Differences")
    print("=" * 80)
    print()
    
    # Generate simple test data
    np.random.seed(42)
    n = 100
    x1 = np.random.randn(n)
    
    # Generate binary response
    true_mu = 1 / (1 + np.exp(-(0.5 + 0.8 * x1)))
    y = (np.random.rand(n) < true_mu).astype(float)
    
    data = {"y": y, "x1": x1}
    
    print(f"Data: n={n}, y mean={y.mean():.3f}, y sum={y.sum():.0f}")
    print()
    
    # Fit Python model
    print("Fitting Python model...")
    family = resolve_family("BI")
    py_model = gamlss(
        formula="y ~ x1",
        sigma_formula="~1",
        family=family,
        data=data,
    )
    
    print(f"Python deviance: {py_model.deviance:.6f}")
    print(f"Python mu coefficients: {py_model.coefficients['mu']}")
    print(f"Python fitted mu (first 5): {py_model.fitted_values['mu'][:5]}")
    print()
    
    # Fit R model
    print("Fitting R model...")
    bridge = RBridge()
    r_result = bridge.call_r_gamlss(
        data=data,
        formula="y ~ x1",
        family="BI",
        sigma_formula="~1",
    )
    
    print(f"R deviance: {r_result['deviance']:.6f}")
    print(f"R AIC: {r_result['aic']:.6f}")
    print(f"R mu coefficients: {r_result['coefficients']['mu']}")
    print(f"R fitted mu (first 5): {r_result['fitted_values']['mu'][:5]}")
    print()
    
    # Compare
    dev_diff = abs(py_model.deviance - r_result['deviance'])
    dev_rel_diff = dev_diff / r_result['deviance'] * 100
    
    print("=" * 80)
    print("Comparison")
    print("=" * 80)
    print(f"Deviance difference: {dev_diff:.6f} ({dev_rel_diff:.2f}%)")
    print()
    
    # Check if coefficients match
    py_coef = np.array(py_model.coefficients['mu'])
    r_coef = np.array(r_result['coefficients']['mu'])
    coef_diff = np.abs(py_coef - r_coef)
    
    print(f"Coefficient differences: {coef_diff}")
    print(f"Max coefficient difference: {coef_diff.max():.6e}")
    print()
    
    # Check if fitted values match
    py_fitted = np.array(py_model.fitted_values['mu'])
    r_fitted = np.array(r_result['fitted_values']['mu'])
    fitted_diff = np.abs(py_fitted - r_fitted)
    
    print(f"Fitted value differences (first 5): {fitted_diff[:5]}")
    print(f"Max fitted value difference: {fitted_diff.max():.6e}")
    print(f"Mean fitted value difference: {fitted_diff.mean():.6e}")
    print()
    
    # Manually compute deviance for both
    print("=" * 80)
    print("Manual Deviance Computation")
    print("=" * 80)
    
    # Python deviance computation
    import jax.numpy as jnp
    y_jax = jnp.array(y, dtype=jnp.float64)
    mu_jax = jnp.array(py_fitted, dtype=jnp.float64)
    eps = jnp.finfo(jnp.float64).eps
    mu_jax = jnp.clip(mu_jax, eps, 1.0 - eps)
    
    log_lik_py = y_jax * jnp.log(mu_jax) + (1.0 - y_jax) * jnp.log(1.0 - mu_jax)
    dev_inc_py = -2.0 * log_lik_py
    manual_dev_py = float(jnp.sum(dev_inc_py))
    
    print(f"Python manual deviance: {manual_dev_py:.6f}")
    print(f"Python model deviance: {py_model.deviance:.6f}")
    print(f"Difference: {abs(manual_dev_py - py_model.deviance):.6e}")
    print()
    
    # R deviance computation (using R fitted values)
    mu_r = np.clip(r_fitted, eps, 1.0 - eps)
    log_lik_r = y * np.log(mu_r) + (1.0 - y) * np.log(1.0 - mu_r)
    dev_inc_r = -2.0 * log_lik_r
    manual_dev_r = np.sum(dev_inc_r)
    
    print(f"R manual deviance (using R fitted): {manual_dev_r:.6f}")
    print(f"R model deviance: {r_result['deviance']:.6f}")
    print(f"Difference: {abs(manual_dev_r - r_result['deviance']):.6e}")
    print()
    
    # Cross-check: Python deviance using R fitted values
    mu_r_jax = jnp.array(r_fitted, dtype=jnp.float64)
    mu_r_jax = jnp.clip(mu_r_jax, eps, 1.0 - eps)
    log_lik_cross = y_jax * jnp.log(mu_r_jax) + (1.0 - y_jax) * jnp.log(1.0 - mu_r_jax)
    dev_inc_cross = -2.0 * log_lik_cross
    cross_dev = float(jnp.sum(dev_inc_cross))
    
    print(f"Python deviance using R fitted values: {cross_dev:.6f}")
    print(f"Should match R deviance: {r_result['deviance']:.6f}")
    print(f"Difference: {abs(cross_dev - r_result['deviance']):.6e}")
    print()


def diagnose_zaga_deviance():
    """Diagnose ZAGA deviance differences."""
    print("=" * 80)
    print("Diagnosing ZAGA Deviance Differences")
    print("=" * 80)
    print()
    
    # Generate test data with zeros
    np.random.seed(42)
    n = 100
    x1 = np.random.randn(n)
    
    # Generate zero-inflated gamma response
    true_nu = 0.3  # 30% zeros
    true_mu = np.exp(0.5 + 0.5 * x1)
    true_sigma = 0.5
    
    # Generate zeros
    is_zero = np.random.rand(n) < true_nu
    
    # Generate positive values from gamma
    shape = 1 / (true_sigma ** 2)
    scale = true_mu * (true_sigma ** 2)
    y = np.where(is_zero, 0, np.random.gamma(shape, scale, n))
    
    data = {"y": y, "x1": x1}
    
    print(f"Data: n={n}, zeros={is_zero.sum()}, mean={y.mean():.3f}")
    print()
    
    # Fit Python model
    print("Fitting Python model...")
    family = resolve_family("ZAGA")
    py_model = gamlss(
        formula="y ~ x1",
        sigma_formula="~1",
        family=family,
        data=data,
    )
    
    print(f"Python deviance: {py_model.deviance:.6f}")
    print(f"Python mu coefficients: {py_model.coefficients['mu']}")
    print(f"Python sigma coefficients: {py_model.coefficients['sigma']}")
    print(f"Python nu coefficients: {py_model.coefficients['nu']}")
    print(f"Python fitted mu (first 5): {py_model.fitted_values['mu'][:5]}")
    print(f"Python fitted sigma (first 5): {py_model.fitted_values['sigma'][:5]}")
    print(f"Python fitted nu (first 5): {py_model.fitted_values['nu'][:5]}")
    print()
    
    # Fit R model
    print("Fitting R model...")
    bridge = RBridge()
    r_result = bridge.call_r_gamlss(
        data=data,
        formula="y ~ x1",
        family="ZAGA",
        sigma_formula="~1",
    )
    
    print(f"R deviance: {r_result['deviance']:.6f}")
    print(f"R AIC: {r_result['aic']:.6f}")
    print(f"R mu coefficients: {r_result['coefficients']['mu']}")
    print(f"R sigma coefficients: {r_result['coefficients']['sigma']}")
    print(f"R nu coefficients: {r_result['coefficients']['nu']}")
    print(f"R fitted mu (first 5): {r_result['fitted_values']['mu'][:5]}")
    print(f"R fitted sigma (first 5): {r_result['fitted_values']['sigma'][:5]}")
    print(f"R fitted nu (first 5): {r_result['fitted_values']['nu'][:5]}")
    print()
    
    # Compare
    dev_diff = abs(py_model.deviance - r_result['deviance'])
    dev_rel_diff = dev_diff / r_result['deviance'] * 100
    
    print("=" * 80)
    print("Comparison")
    print("=" * 80)
    print(f"Deviance difference: {dev_diff:.6f} ({dev_rel_diff:.2f}%)")
    print()


if __name__ == "__main__":
    diagnose_bi_deviance()
    print("\n\n")
    diagnose_zaga_deviance()
