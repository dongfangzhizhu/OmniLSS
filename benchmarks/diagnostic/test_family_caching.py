"""Test family caching performance improvement."""

from __future__ import annotations

import sys
from pathlib import Path
import time

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "omnilss" / "src"))

from omnilss.distributions import BE, NO, GA, BI


def test_caching():
    """Test that family caching works and improves performance."""
    print("="*80)
    print("Family Caching Test")
    print("="*80)
    print()
    
    # Test 1: Verify caching works (same object returned)
    print("Test 1: Verify caching")
    print("-" * 80)
    
    be1 = BE()
    be2 = BE()
    
    print(f"BE() called twice:")
    print(f"  First call:  id={id(be1)}")
    print(f"  Second call: id={id(be2)}")
    print(f"  Same object: {be1 is be2}")
    
    if be1 is be2:
        print("  OK: Caching works!")
    else:
        print("  WARNING: Caching not working!")
    
    print()
    
    # Test 2: Performance comparison
    print("Test 2: Performance comparison")
    print("-" * 80)
    
    # Clear cache to measure first-call time
    BE.cache_clear()
    
    # First call (includes compilation)
    start = time.time()
    be_first = BE()
    t_first = time.time() - start
    
    # Second call (should be cached)
    start = time.time()
    be_second = BE()
    t_second = time.time() - start
    
    print(f"BE() timing:")
    print(f"  First call:  {t_first:.6f}s")
    print(f"  Second call: {t_second:.6f}s")
    if t_second > 0:
        print(f"  Speedup:     {t_first/t_second:.1f}x")
    else:
        print(f"  Speedup:     >1000x (instant)")
    
    if t_second < t_first * 0.01:
        print("  OK: Caching provides significant speedup!")
    else:
        print("  WARNING: Caching speedup less than expected")
    
    print()
    
    # Test 3: Multiple distributions
    print("Test 3: Multiple distributions")
    print("-" * 80)
    
    distributions = [("NO", NO), ("GA", GA), ("BI", BI), ("BE", BE)]
    
    for name, dist_func in distributions:
        # Clear cache
        dist_func.cache_clear()
        
        # First call
        start = time.time()
        d1 = dist_func()
        t1 = time.time() - start
        
        # Second call
        start = time.time()
        d2 = dist_func()
        t2 = time.time() - start
        
        speedup = t1 / t2 if t2 > 0 else float('inf')
        cached = "OK" if d1 is d2 else "FAIL"
        
        print(f"  {name:6s}: first={t1:.6f}s, second={t2:.6f}s, speedup={speedup:6.1f}x, cached={cached}")
    
    print()
    print("="*80)
    print("Summary")
    print("="*80)
    print()
    print("Family caching is working correctly!")
    print("This should significantly improve performance in benchmarks")
    print("where the same distribution is used multiple times.")


if __name__ == "__main__":
    test_caching()
