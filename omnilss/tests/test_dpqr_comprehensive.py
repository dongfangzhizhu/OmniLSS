"""Comprehensive d/p/q/r function tests for all distributions.

This test ensures that all distributions have working d/p/q/r functions
that return actual values (not NaN).
"""

import unittest
from pathlib import Path

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from omnilss.distributions import resolve_family


class TestDPQRFunctions(unittest.TestCase):
    """Test d/p/q/r functions for all distributions."""
    
    @classmethod
    def setUpClass(cls):
        cls.key = jrandom.PRNGKey(42)
        
        # Distribution parameter configurations
        cls.dist_configs = {
            # Continuous - Basic
            "NO": {"params": (0.0, 1.0), "x": 0.0, "type": "continuous"},
            "GA": {"params": (1.0, 1.0), "x": 1.0, "type": "continuous"},
            "LOGNO": {"params": (1.0, 1.0), "x": 1.0, "type": "continuous"},
            "WEI": {"params": (1.0, 1.0), "x": 1.0, "type": "continuous"},
            "EXP": {"params": (1.0,), "x": 1.0, "type": "continuous"},
            "IG": {"params": (1.0, 1.0), "x": 1.0, "type": "continuous"},
            "LO": {"params": (0.0, 1.0), "x": 0.0, "type": "continuous"},
            "TF": {"params": (0.0, 1.0, 10.0), "x": 0.0, "type": "continuous"},
            
            # Continuous - Extended
            "NO2": {"params": (0.0, 1.0), "x": 0.0, "type": "continuous"},
            "LOGNO2": {"params": (1.0, 1.0), "x": 1.0, "type": "continuous"},
            "PE": {"params": (1.0, 1.0, 2.0), "x": 1.0, "type": "continuous"},
            "SIMPLEX": {"params": (0.5, 1.0), "x": 0.5, "type": "continuous"},
            "EXGAUS": {"params": (0.0, 1.0, 1.0), "x": 0.0, "type": "continuous"},
            
            # Continuous - Skewed
            "SHASH": {"params": (0.0, 1.0, 0.0, 1.0), "x": 0.0, "type": "continuous"},
            "SN1": {"params": (0.0, 1.0, 0.0), "x": 0.0, "type": "continuous"},
            "SN2": {"params": (0.0, 1.0, 0.0), "x": 0.0, "type": "continuous"},
            "GT": {"params": (0.0, 1.0, 2.0, 10.0), "x": 0.0, "type": "continuous"},
            
            # Continuous - Advanced
            "GG": {"params": (1.0, 1.0, 1.0), "x": 1.0, "type": "continuous"},
            "GB2": {"params": (1.0, 1.0, 1.0, 1.0), "x": 1.0, "type": "continuous"},
            "NET": {"params": (0.0, 1.0, 1.0, 10.0), "x": 0.0, "type": "continuous"},
            
            # Discrete - Basic
            "PO": {"params": (1.0,), "x": 5.0, "type": "discrete"},
            "BI": {"params": (0.5,), "x": 1.0, "type": "discrete"},
            "GEOM": {"params": (1.0,), "x": 5.0, "type": "discrete"},
            "NBI": {"params": (1.0, 1.0), "x": 5.0, "type": "discrete"},
            "NBII": {"params": (1.0, 1.0), "x": 5.0, "type": "discrete"},
            
            # Discrete - Zero-Inflated
            "ZIP": {"params": (1.0, 0.1), "x": 5.0, "type": "discrete"},
            "ZIP2": {"params": (1.0, 0.1), "x": 5.0, "type": "discrete"},
            "ZINBI": {"params": (1.0, 1.0, 0.1), "x": 5.0, "type": "discrete"},
            "ZAP": {"params": (1.0, 0.1), "x": 5.0, "type": "discrete"},
            
            # Discrete - Advanced
            "BB": {"params": (0.5, 1.0, 10.0), "x": 5.0, "type": "discrete"},
            "BNB": {"params": (1.0, 1.0, 1.0), "x": 5.0, "type": "discrete"},
            "PIG": {"params": (1.0, 1.0), "x": 5.0, "type": "discrete"},
            "SICHEL": {"params": (1.0, 1.0, -0.5), "x": 5.0, "type": "discrete"},
            "DPO": {"params": (1.0, 1.0), "x": 5.0, "type": "discrete"},
            "DEL": {"params": (1.0, 1.0, 1.0), "x": 5.0, "type": "discrete"},
            "YULE": {"params": (1.0,), "x": 5.0, "type": "discrete"},
            "WARING": {"params": (1.0, 1.0), "x": 5.0, "type": "discrete"},
            
            # Beta and Zero-Altered
            "BE": {"params": (0.5, 0.1), "x": 0.5, "type": "continuous"},
            "BEINF": {"params": (0.5, 0.1, 0.1, 0.1), "x": 0.5, "type": "mixed"},
            "ZAGA": {"params": (1.0, 1.0, 0.1), "x": 1.0, "type": "mixed"},
            "ZAIG": {"params": (1.0, 1.0, 0.1), "x": 1.0, "type": "mixed"},
            
            # Batch 1 remaining
            "GU": {"params": (0.0, 1.0), "x": 0.0, "type": "continuous"},
            "RG": {"params": (0.0, 1.0), "x": 0.0, "type": "continuous"},
            "IGAMMA": {"params": (1.0, 1.0), "x": 1.0, "type": "continuous"},
            "PARETO2": {"params": (1.0, 1.0), "x": 1.0, "type": "continuous"},
        }
    
    def _test_d_function(self, dist_name: str):
        """Test d (density/PMF) function."""
        config = self.dist_configs[dist_name]
        family = resolve_family(dist_name)
        
        x_test = config["x"]
        params = config["params"]
        
        # Test d function
        d_result = family.d(x_test, *params)
        
        # Check that result is not NaN and is finite
        self.assertFalse(jnp.isnan(d_result).any(), 
                        f"{dist_name}: d function returned NaN")
        self.assertTrue(jnp.isfinite(d_result).any(), 
                       f"{dist_name}: d function returned non-finite value")
        
        # Check that result is positive (density/PMF should be >= 0)
        self.assertTrue((d_result >= 0).all() or jnp.isclose(d_result, 0.0).any(),
                       f"{dist_name}: d function returned negative value")
    
    def _test_p_function(self, dist_name: str):
        """Test p (CDF) function."""
        config = self.dist_configs[dist_name]
        family = resolve_family(dist_name)
        
        x_test = config["x"]
        params = config["params"]
        
        # Test p function
        p_result = family.p(x_test, *params)
        
        # Check that result is not NaN and is finite
        self.assertFalse(jnp.isnan(p_result).any(), 
                        f"{dist_name}: p function returned NaN")
        self.assertTrue(jnp.isfinite(p_result).any(), 
                       f"{dist_name}: p function returned non-finite value")
        
        # Check that result is in [0, 1] (CDF should be a probability)
        self.assertTrue((p_result >= 0).all() and (p_result <= 1).all(),
                       f"{dist_name}: p function returned value outside [0, 1]")
    
    def _test_q_function(self, dist_name: str):
        """Test q (quantile) function."""
        config = self.dist_configs[dist_name]
        family = resolve_family(dist_name)
        
        params = config["params"]
        
        # Test q function at median
        q_result = family.q(0.5, *params)
        
        # Check that result is not NaN and is finite
        self.assertFalse(jnp.isnan(q_result).any(), 
                        f"{dist_name}: q function returned NaN")
        self.assertTrue(jnp.isfinite(q_result).any(), 
                       f"{dist_name}: q function returned non-finite value")
    
    def _test_r_function(self, dist_name: str):
        """Test r (random generation) function."""
        config = self.dist_configs[dist_name]
        family = resolve_family(dist_name)
        
        params = config["params"]
        
        # Test r function
        r_result = family.r(self.key, 10, *params)
        
        # Check that result is not NaN and is finite
        self.assertFalse(jnp.isnan(r_result).any(), 
                        f"{dist_name}: r function returned NaN")
        self.assertTrue(jnp.isfinite(r_result).any(), 
                       f"{dist_name}: r function returned non-finite value")
        
        # Check that we got the right number of samples
        self.assertEqual(len(r_result), 10,
                        f"{dist_name}: r function returned wrong number of samples")
    
    def _test_dpqr_consistency(self, dist_name: str):
        """Test consistency between d/p/q/r functions."""
        config = self.dist_configs[dist_name]
        family = resolve_family(dist_name)
        
        params = config["params"]
        
        # Generate random samples
        samples = family.r(self.key, 100, *params)
        
        # Skip if r returns NaN
        if jnp.isnan(samples).any():
            self.skipTest(f"{dist_name}: r function returns NaN, skipping consistency test")
        
        # Test that p(q(0.5)) ≈ 0.5
        q_median = family.q(0.5, *params)
        if not jnp.isnan(q_median).any():
            p_of_q = family.p(q_median, *params)
            if not jnp.isnan(p_of_q).any():
                self.assertTrue(jnp.isclose(p_of_q, 0.5, atol=0.01).any(),
                              f"{dist_name}: p(q(0.5)) != 0.5 (got {p_of_q})")


# Dynamically generate test methods for each distribution
def _generate_test_methods():
    """Generate test methods for each distribution."""
    # Hard-code the distribution list since we can't access class variables before instantiation
    for dist_name in [
        "NO", "GA", "LOGNO", "WEI", "EXP", "IG", "LO", "TF",
        "NO2", "LOGNO2", "PE", "SIMPLEX", "EXGAUS",
        "SHASH", "SN1", "SN2", "GT",
        "GG", "GB2", "NET",
        "PO", "BI", "GEOM", "NBI", "NBII",
        "ZIP", "ZIP2", "ZINBI", "ZAP",
        "BB", "BNB", "PIG", "SICHEL", "DPO", "DEL", "YULE", "WARING",
        "BE", "BEINF", "ZAGA", "ZAIG",
        "GU", "RG", "IGAMMA", "PARETO2"
    ]:
        # Test d function
        def make_d_test(d):
            def test(self):
                self._test_d_function(d)
            return test
        setattr(TestDPQRFunctions, f"test_{dist_name}_d", make_d_test(dist_name))
        
        # Test p function
        def make_p_test(d):
            def test(self):
                self._test_p_function(d)
            return test
        setattr(TestDPQRFunctions, f"test_{dist_name}_p", make_p_test(dist_name))
        
        # Test q function
        def make_q_test(d):
            def test(self):
                self._test_q_function(d)
            return test
        setattr(TestDPQRFunctions, f"test_{dist_name}_q", make_q_test(dist_name))
        
        # Test r function
        def make_r_test(d):
            def test(self):
                self._test_r_function(d)
            return test
        setattr(TestDPQRFunctions, f"test_{dist_name}_r", make_r_test(dist_name))
        
        # Test consistency
        def make_consistency_test(d):
            def test(self):
                self._test_dpqr_consistency(d)
            return test
        setattr(TestDPQRFunctions, f"test_{dist_name}_consistency", make_consistency_test(dist_name))


# Generate test methods
_generate_test_methods()
    



class TestDPQRCoverage(unittest.TestCase):
    """Test that all distributions have d/p/q/r functions defined."""
    
    def test_all_distributions_have_dpqr(self):
        """Test that all distributions have d/p/q/r attributes."""
        from performance.config import DISTRIBUTIONS
        
        missing = []
        
        for dist_config in DISTRIBUTIONS:
            dist_name = dist_config.name
            
            try:
                family = resolve_family(dist_name)
                
                # Check for d/p/q/r attributes
                if not hasattr(family, "d") or family.d is None:
                    missing.append(f"{dist_name}: missing d")
                if not hasattr(family, "p") or family.p is None:
                    missing.append(f"{dist_name}: missing p")
                if not hasattr(family, "q") or family.q is None:
                    missing.append(f"{dist_name}: missing q")
                if not hasattr(family, "r") or family.r is None:
                    missing.append(f"{dist_name}: missing r")
                    
            except Exception as e:
                missing.append(f"{dist_name}: error resolving family - {e}")
        
        if missing:
            self.fail(f"Distributions with missing d/p/q/r:\n" + "\n".join(missing))


if __name__ == "__main__":
    unittest.main()
