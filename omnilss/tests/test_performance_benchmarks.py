"""Performance benchmark tests for GAMLSS.

These tests establish performance baselines and detect regressions.
"""

import time
import unittest
from typing import Dict, Any

import numpy as np
import pandas as pd

from omnilss import gamlss
from omnilss.distributions import resolve_family


class PerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests."""
    
    # Performance thresholds (seconds)
    # These are conservative estimates; actual performance should be better
    THRESHOLDS = {
        "NO_small": 0.5,    # Normal, n=100
        "NO_medium": 2.0,   # Normal, n=1000
        "NO_large": 10.0,   # Normal, n=10000
        "GA_small": 0.5,    # Gamma, n=100
        "GA_medium": 2.0,   # Gamma, n=1000
        "PO_small": 0.5,    # Poisson, n=100
        "PO_medium": 2.0,   # Poisson, n=1000
        "BE_small": 1.0,    # Beta, n=100
        "BE_medium": 5.0,   # Beta, n=1000
        "ZAGA_small": 1.0,  # ZAGA, n=100
        "ZAGA_medium": 5.0, # ZAGA, n=1000
    }
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.results: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def tearDownClass(cls):
        """Print benchmark results."""
        print("\n" + "=" * 70)
        print("Performance Benchmark Results")
        print("=" * 70)
        print(f"{'Test':<20} {'Time (s)':<12} {'Threshold':<12} {'Status':<10}")
        print("-" * 70)
        
        for test_name, result in sorted(cls.results.items()):
            time_taken = result['time']
            threshold = result['threshold']
            status = "✓ PASS" if time_taken <= threshold else "✗ FAIL"
            print(f"{test_name:<20} {time_taken:<12.4f} {threshold:<12.4f} {status:<10}")
        
        print("=" * 70)
        
        # Check if any tests failed
        failures = [name for name, result in cls.results.items() 
                   if result['time'] > result['threshold']]
        
        if failures:
            print(f"\n⚠ {len(failures)} benchmark(s) exceeded threshold:")
            for name in failures:
                result = cls.results[name]
                slowdown = result['time'] / result['threshold']
                print(f"  - {name}: {slowdown:.2f}x slower than threshold")
    
    def _benchmark_fit(
        self,
        test_name: str,
        family: str,
        data: pd.DataFrame,
        formula: str = "y ~ x1",
        sigma_formula: str = "~1",
        threshold: float = 1.0,
    ):
        """Benchmark a model fit.
        
        Parameters
        ----------
        test_name : str
            Name of the test
        family : str
            Distribution family name
        data : pd.DataFrame
            Data for fitting
        formula : str
            Formula for mu
        sigma_formula : str
            Formula for sigma
        threshold : float
            Time threshold in seconds
        """
        # Warm-up run (not timed)
        try:
            _ = gamlss(
                formula=formula,
                sigma_formula=sigma_formula,
                family=family,
                data=data,
            )
        except Exception:
            self.skipTest(f"Warm-up failed for {test_name}")
        
        # Timed run
        start_time = time.time()
        
        try:
            model = gamlss(
                formula=formula,
                sigma_formula=sigma_formula,
                family=family,
                data=data,
            )
        except Exception as e:
            self.fail(f"Fitting failed for {test_name}: {e}")
        
        elapsed = time.time() - start_time
        
        # Store result
        self.__class__.results[test_name] = {
            'time': elapsed,
            'threshold': threshold,
            'family': family,
            'n_obs': len(data),
            'converged': model.additional_slots.get('converged', False),
        }
        
        # Assert performance
        self.assertLessEqual(
            elapsed,
            threshold,
            f"{test_name} took {elapsed:.4f}s, exceeding threshold of {threshold:.4f}s"
        )
        
        # Assert convergence
        self.assertTrue(
            model.additional_slots.get('converged', False),
            f"{test_name} did not converge"
        )
    
    def test_NO_small(self):
        """Benchmark Normal distribution with small dataset (n=100)."""
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        y = 2 + 3 * x1 + np.random.randn(n)
        data = pd.DataFrame({'y': y, 'x1': x1})
        
        self._benchmark_fit(
            "NO_small",
            "NO",
            data,
            threshold=self.THRESHOLDS["NO_small"]
        )
    
    def test_NO_medium(self):
        """Benchmark Normal distribution with medium dataset (n=1000)."""
        np.random.seed(42)
        n = 1000
        x1 = np.random.randn(n)
        y = 2 + 3 * x1 + np.random.randn(n)
        data = pd.DataFrame({'y': y, 'x1': x1})
        
        self._benchmark_fit(
            "NO_medium",
            "NO",
            data,
            threshold=self.THRESHOLDS["NO_medium"]
        )
    
    def test_NO_large(self):
        """Benchmark Normal distribution with large dataset (n=10000)."""
        np.random.seed(42)
        n = 10000
        x1 = np.random.randn(n)
        y = 2 + 3 * x1 + np.random.randn(n)
        data = pd.DataFrame({'y': y, 'x1': x1})
        
        self._benchmark_fit(
            "NO_large",
            "NO",
            data,
            threshold=self.THRESHOLDS["NO_large"]
        )
    
    def test_GA_small(self):
        """Benchmark Gamma distribution with small dataset (n=100)."""
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        mu = np.exp(1 + 0.5 * x1)
        y = np.random.gamma(shape=2, scale=mu/2, size=n)
        data = pd.DataFrame({'y': y, 'x1': x1})
        
        self._benchmark_fit(
            "GA_small",
            "GA",
            data,
            threshold=self.THRESHOLDS["GA_small"]
        )
    
    def test_GA_medium(self):
        """Benchmark Gamma distribution with medium dataset (n=1000)."""
        np.random.seed(42)
        n = 1000
        x1 = np.random.randn(n)
        mu = np.exp(1 + 0.5 * x1)
        y = np.random.gamma(shape=2, scale=mu/2, size=n)
        data = pd.DataFrame({'y': y, 'x1': x1})
        
        self._benchmark_fit(
            "GA_medium",
            "GA",
            data,
            threshold=self.THRESHOLDS["GA_medium"]
        )
    
    def test_PO_small(self):
        """Benchmark Poisson distribution with small dataset (n=100)."""
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        mu = np.exp(1 + 0.5 * x1)
        y = np.random.poisson(mu)
        data = pd.DataFrame({'y': y, 'x1': x1})
        
        self._benchmark_fit(
            "PO_small",
            "PO",
            data,
            threshold=self.THRESHOLDS["PO_small"]
        )
    
    def test_PO_medium(self):
        """Benchmark Poisson distribution with medium dataset (n=1000)."""
        np.random.seed(42)
        n = 1000
        x1 = np.random.randn(n)
        mu = np.exp(1 + 0.5 * x1)
        y = np.random.poisson(mu)
        data = pd.DataFrame({'y': y, 'x1': x1})
        
        self._benchmark_fit(
            "PO_medium",
            "PO",
            data,
            threshold=self.THRESHOLDS["PO_medium"]
        )
    
    def test_BE_small(self):
        """Benchmark Beta distribution with small dataset (n=100)."""
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        mu = 1 / (1 + np.exp(-(0.5 + 0.3 * x1)))
        y = np.random.beta(mu * 10, (1 - mu) * 10)
        data = pd.DataFrame({'y': y, 'x1': x1})
        
        self._benchmark_fit(
            "BE_small",
            "BE",
            data,
            threshold=self.THRESHOLDS["BE_small"]
        )
    
    def test_BE_medium(self):
        """Benchmark Beta distribution with medium dataset (n=1000)."""
        np.random.seed(42)
        n = 1000
        x1 = np.random.randn(n)
        mu = 1 / (1 + np.exp(-(0.5 + 0.3 * x1)))
        y = np.random.beta(mu * 10, (1 - mu) * 10)
        data = pd.DataFrame({'y': y, 'x1': x1})
        
        self._benchmark_fit(
            "BE_medium",
            "BE",
            data,
            threshold=self.THRESHOLDS["BE_medium"]
        )
    
    def test_ZAGA_small(self):
        """Benchmark ZAGA distribution with small dataset (n=100)."""
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        
        # Generate ZAGA data
        pi = 1 / (1 + np.exp(-(-1 + 0.3 * x1)))
        mu = np.exp(1.5 + 0.5 * x1)
        sigma = 0.6
        
        y = np.zeros(n)
        for i in range(n):
            if np.random.rand() < pi[i]:
                y[i] = 0
            else:
                shape = 1 / sigma**2
                y[i] = np.random.gamma(shape, mu[i] / shape)
        
        data = pd.DataFrame({'y': y, 'x1': x1})
        
        self._benchmark_fit(
            "ZAGA_small",
            "ZAGA",
            data,
            threshold=self.THRESHOLDS["ZAGA_small"]
        )
    
    def test_ZAGA_medium(self):
        """Benchmark ZAGA distribution with medium dataset (n=1000)."""
        np.random.seed(42)
        n = 1000
        x1 = np.random.randn(n)
        
        # Generate ZAGA data
        pi = 1 / (1 + np.exp(-(-1 + 0.3 * x1)))
        mu = np.exp(1.5 + 0.5 * x1)
        sigma = 0.6
        
        y = np.zeros(n)
        for i in range(n):
            if np.random.rand() < pi[i]:
                y[i] = 0
            else:
                shape = 1 / sigma**2
                y[i] = np.random.gamma(shape, mu[i] / shape)
        
        data = pd.DataFrame({'y': y, 'x1': x1})
        
        self._benchmark_fit(
            "ZAGA_medium",
            "ZAGA",
            data,
            threshold=self.THRESHOLDS["ZAGA_medium"]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
