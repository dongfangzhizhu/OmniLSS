"""Profile BE distribution performance to identify bottlenecks."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import time

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "omnilss" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "omnilss" / "tests"))
sys.path.insert(0, str(Path(__file__).parent.parent / "performance"))

from omnilss.distributions import resolve_family
from omnilss.fitting import gamlss
from rbus.r_bridge import RBridge


def profile_be_case(n_obs, n_predictors, formula):
    """Profile a specific BE test case."""
    print(f"\n{'='*80}")
    print(f"Profiling BE: n={n_obs}, predictors={n_predictors}, formula={formula}")
    print(f"{'='*80}")
    
    # Generate data
    np.random.seed(42)
    data = {"y": np.random.beta(2, 5, n_obs)}
    
    for i in range(1, n_predictors + 1):
        data[f"x{i}"] = np.random.randn(n_obs)
    
    print(f"Data generated: y range [{data['y'].min():.3f}, {data['y'].max():.3f}]")
    
    # Python timing breakdown
    print("\nPython timing breakdown:")
    
    family = resolve_family("BE")
    
    # Time 1: Family creation
    start = time.time()
    family = resolve_family("BE")
    t_family = time.time() - start
    print(f"  Family creation: {t_family:.4f}s")
    
    # Time 2: Full fitting
    start = time.time()
    py_model = gamlss(
        formula=formula,
        sigma_formula="~1",
        family=family,
        data=data,
    )
    t_total = time.time() - start
    print(f"  Total fitting: {t_total:.4f}s")
    print(f"  Iterations: {py_model.iter}")
    print(f"  Deviance: {py_model.deviance:.6f}")
    
    # R timing
    print("\nR timing:")
    bridge = RBridge()
    
    start = time.time()
    r_result = bridge.call_r_gamlss(
        data=data,
        formula=formula,
        family="BE",
        sigma_formula="~1",
    )
    t_r = time.time() - start
    print(f"  Total fitting: {t_r:.4f}s")
    print(f"  Deviance: {r_result['deviance']:.6f}")
    
    # Comparison
    print("\nComparison:")
    speedup = t_r / t_total
    dev_diff = abs(py_model.deviance - r_result['deviance'])
    dev_rel = dev_diff / r_result['deviance'] * 100
    
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Deviance diff: {dev_diff:.6f} ({dev_rel:.4f}%)")
    
    if speedup < 1.0:
        print(f"  WARNING: Python is SLOWER than R!")
    else:
        print(f"  OK: Python is faster")
    
    return {
        "n": n_obs,
        "formula": formula,
        "py_time": t_total,
        "r_time": t_r,
        "speedup": speedup,
        "iterations": py_model.iter,
    }


def main():
    """Profile BE distribution performance."""
    print("="*80)
    print("BE Distribution Performance Profiling")
    print("="*80)
    
    # Test cases from the report that are slow
    slow_cases = [
        (100, 1, "y ~ x1"),      # 0.71x
        (500, 0, "y ~ 1"),       # 0.63x
        (500, 1, "y ~ x1"),      # 0.62x
        (500, 2, "y ~ x1 + x2"), # 0.95x
    ]
    
    results = []
    for n_obs, n_predictors, formula in slow_cases:
        result = profile_be_case(n_obs, n_predictors, formula)
        results.append(result)
    
    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    
    print("\nSlow cases (speedup < 1.0):")
    for r in results:
        if r['speedup'] < 1.0:
            print(f"  n={r['n']:5d}, {r['formula']:20s}: {r['speedup']:.2f}x, {r['iterations']:2d} iters")
    
    print("\nPotential issues:")
    print("  1. Too many iterations?")
    print("  2. Slow convergence?")
    print("  3. Initialization issues?")
    print("  4. JIT compilation overhead?")


if __name__ == "__main__":
    main()
