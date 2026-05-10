"""Test BI with bd parameter."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "omnilss" / "src"))

from omnilss.distributions import resolve_family
from omnilss.fitting import gamlss


def test_bi_bernoulli():
    """Test BI with Bernoulli data (no bd)."""
    print("="*80)
    print("Test 1: Bernoulli (bd=1, implicit)")
    print("="*80)
    
    np.random.seed(42)
    n = 100
    x1 = np.random.randn(n)
    true_mu = 1 / (1 + np.exp(-(0.5 + 0.8 * x1)))
    y = (np.random.rand(n) < true_mu).astype(float)
    
    data = {"y": y, "x1": x1}
    
    family = resolve_family("BI")
    model = gamlss(
        formula="y ~ x1",
        sigma_formula="~1",
        family=family,
        data=data,
    )
    
    print(f"Deviance: {model.deviance:.6f}")
    print(f"Coefficients: {model.coefficients['mu']}")
    print()


def test_bi_with_bd_counts():
    """Test BI with binomial data as counts."""
    print("="*80)
    print("Test 2: Binomial with bd=10 (counts)")
    print("="*80)
    
    np.random.seed(42)
    n = 100
    x1 = np.random.randn(n)
    bd = 10
    
    true_mu = 1 / (1 + np.exp(-(0.5 + 0.5 * x1)))
    y_counts = np.random.binomial(bd, true_mu)
    
    data = {"y": y_counts, "x1": x1, "bd": np.full(n, bd, dtype=float)}
    
    print(f"y range: [{y_counts.min()}, {y_counts.max()}]")
    print(f"y mean: {y_counts.mean():.2f}")
    print()
    
    family = resolve_family("BI")
    try:
        model = gamlss(
            formula="y ~ x1",
            sigma_formula="~1",
            family=family,
            data=data,
        )
        
        print(f"Deviance: {model.deviance:.6f}")
        print(f"Coefficients: {model.coefficients['mu']}")
        print(f"Fitted mu (first 5): {model.fitted_values['mu'][:5]}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    print()


def test_bi_with_bd_proportions():
    """Test BI with binomial data as proportions."""
    print("="*80)
    print("Test 3: Binomial with bd=10 (proportions)")
    print("="*80)
    
    np.random.seed(42)
    n = 100
    x1 = np.random.randn(n)
    bd = 10
    
    true_mu = 1 / (1 + np.exp(-(0.5 + 0.5 * x1)))
    y_counts = np.random.binomial(bd, true_mu)
    y_proportions = y_counts / bd
    
    data = {"y": y_proportions, "x1": x1, "bd": np.full(n, bd, dtype=float)}
    
    print(f"y range: [{y_proportions.min():.2f}, {y_proportions.max():.2f}]")
    print(f"y mean: {y_proportions.mean():.2f}")
    print()
    
    family = resolve_family("BI")
    try:
        model = gamlss(
            formula="y ~ x1",
            sigma_formula="~1",
            family=family,
            data=data,
        )
        
        print(f"Deviance: {model.deviance:.6f}")
        print(f"Coefficients: {model.coefficients['mu']}")
        print(f"Fitted mu (first 5): {model.fitted_values['mu'][:5]}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    print()


if __name__ == "__main__":
    test_bi_bernoulli()
    test_bi_with_bd_counts()
    test_bi_with_bd_proportions()
