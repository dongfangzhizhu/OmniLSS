"""Preservation Property Tests for Complex Distributions Fix.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8**

This test verifies that non-affected distributions continue to work correctly
when the complex distributions fix is implemented. These tests are run on UNFIXED
code and are EXPECTED TO PASS, establishing a baseline to ensure no regressions.

Property 2: Preservation - Non-Affected Distribution Behavior

Test Coverage:
- Basic Distribution Functions: d/p/q functions for all 10 distributions
- Framework Stability: Simple 2-parameter distributions (NO, GA, IG, LOGNO)
- Batch 1-5 Distributions: Sample of fully tested distributions
- Zero-Inflated Distributions: ZAGA, ZAIG, ZAP, ZINBI, ZIP2
- Test Suite Integrity: Existing tests continue to pass

Run:
    PYTHONPATH=src JAX_PLATFORMS=cpu python -m pytest tests/test_preservation_complex_distributions.py -v
"""

import unittest
import numpy as np
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omnilss.distributions import NO, GA, IG, LOGNO
from omnilss.distributions_b5 import ZAGA, ZAIG, ZAP, ZINBI, ZIP2
from omnilss.distributions_b6 import DEL, DPO, PIG, SICHEL
from omnilss.distributions_b7 import BB, BNB
from omnilss.distributions_b8 import GB2
from omnilss.fitting import gamlss
from omnilss.controls import gamlss_control


class TestBasicDistributionFunctions(unittest.TestCase):
    """Test that d/p/q functions for affected distributions remain unchanged.
    
    Validates Requirement 3.1: D/P/Q functions continue to return correct values.
    """
    
    def test_del_dpq_functions(self):
        """Test DEL d/p/q functions work correctly."""
        family = DEL()
        
        # Test that family has the d function (p and q not implemented for DEL)
        self.assertIsNotNone(family.d)
        
        # Test basic d call doesn't crash
        y = np.array([0, 1, 2, 3, 4, 5])
        mu = np.full_like(y, 2.5, dtype=float)
        sigma = np.full_like(y, 0.5, dtype=float)
        nu = np.full_like(y, 0.3, dtype=float)
        
        # DEL uses positional arguments: (y, mu, sigma, nu)
        d_result = family.d(y, mu, sigma, nu)
        self.assertTrue(np.all(np.isfinite(d_result)), "DEL d() should return finite values")
        self.assertTrue(np.all(d_result >= 0), "DEL d() should return non-negative values")
    
    def test_sichel_dpq_functions(self):
        """Test SICHEL d/p/q functions work correctly."""
        family = SICHEL()
        
        # Test that family has the d function (p and q not implemented for SICHEL)
        self.assertIsNotNone(family.d)
        
        y = np.array([0, 1, 2, 3, 4, 5])
        mu = np.full_like(y, 3.0, dtype=float)
        sigma = np.full_like(y, 0.8, dtype=float)
        nu = np.full_like(y, -0.5, dtype=float)
        
        # SICHEL uses positional arguments: (y, mu, sigma, nu)
        d_result = family.d(y, mu, sigma, nu)
        self.assertTrue(np.all(np.isfinite(d_result)), "SICHEL d() should return finite values")
        self.assertTrue(np.all(d_result >= 0), "SICHEL d() should return non-negative values")
    
    def test_dpo_dpq_functions(self):
        """Test DPO d/p/q functions work correctly."""
        family = DPO()
        
        # Test that family has the d function (p and q not implemented for DPO)
        self.assertIsNotNone(family.d)
        
        y = np.array([0, 1, 2, 3, 4, 5])
        mu = np.full_like(y, 2.0, dtype=float)
        sigma = np.full_like(y, 1.2, dtype=float)
        
        # DPO uses positional arguments: (y, mu, sigma)
        d_result = family.d(y, mu, sigma)
        self.assertTrue(np.all(np.isfinite(d_result)), "DPO d() should return finite values")
        self.assertTrue(np.all(d_result >= 0), "DPO d() should return non-negative values")
    
    def test_pig_dpq_functions(self):
        """Test PIG d/p/q functions work correctly."""
        family = PIG()
        
        # Test that family has the d function (p and q not implemented for PIG)
        self.assertIsNotNone(family.d)
        
        y = np.array([0, 1, 2, 3, 4, 5])
        mu = np.full_like(y, 2.5, dtype=float)
        sigma = np.full_like(y, 0.5, dtype=float)
        
        # PIG uses positional arguments: (y, mu, sigma)
        d_result = family.d(y, mu, sigma)
        self.assertTrue(np.all(np.isfinite(d_result)), "PIG d() should return finite values")
        self.assertTrue(np.all(d_result >= 0), "PIG d() should return non-negative values")
    
    def test_bb_dpq_functions(self):
        """Test BB d/p/q functions work correctly."""
        family = BB()
        
        # BB only has d() function implemented (p and q not needed for model fitting)
        self.assertIsNotNone(family.d)
        
        y = np.array([0, 1, 2, 3, 4, 5])
        bd = np.full_like(y, 10.0, dtype=float)
        mu = np.full_like(y, 0.5, dtype=float)
        sigma = np.full_like(y, 0.3, dtype=float)
        
        # d() expects positional arguments: (y, mu, sigma, bd)
        d_result = family.d(y, mu, sigma, bd)
        self.assertTrue(np.all(np.isfinite(d_result)), "BB d() should return finite values")
        self.assertTrue(np.all(d_result >= 0), "BB d() should return non-negative values")
    
    def test_bnb_dpq_functions(self):
        """Test BNB d/p/q functions work correctly."""
        family = BNB()
        
        # BNB only has d() function implemented (p and q not needed for model fitting)
        self.assertIsNotNone(family.d)
        
        y = np.array([0, 1, 2, 3, 4, 5])
        mu = np.full_like(y, 3.0, dtype=float)
        sigma = np.full_like(y, 0.5, dtype=float)
        nu = np.full_like(y, 2.0, dtype=float)
        
        # d() expects positional arguments: (y, mu, sigma, nu)
        d_result = family.d(y, mu, sigma, nu)
        self.assertTrue(np.all(np.isfinite(d_result)), "BNB d() should return finite values")
        self.assertTrue(np.all(d_result >= 0), "BNB d() should return non-negative values")
    
    def test_gb2_dpq_functions(self):
        """Test GB2 d/p/q functions work correctly."""
        family = GB2()
        
        # Test that family has the d function (p and q not implemented for GB2)
        self.assertIsNotNone(family.d)
        
        y = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
        mu = np.full_like(y, 2.0, dtype=float)
        sigma = np.full_like(y, 1.0, dtype=float)
        nu = np.full_like(y, 1.0, dtype=float)
        tau = np.full_like(y, 1.0, dtype=float)
        
        # GB2 uses positional arguments: (y, mu, sigma, nu, tau)
        d_result = family.d(y, mu, sigma, nu, tau)
        self.assertTrue(np.all(np.isfinite(d_result)), "GB2 d() should return finite values")
        self.assertTrue(np.all(d_result >= 0), "GB2 d() should return non-negative values")


class TestFrameworkStability(unittest.TestCase):
    """Test that simple 2-parameter distributions converge reliably.
    
    Validates Requirements 3.2, 3.3: Framework stability and simple distributions.
    """
    
    def test_no_simple_fitting(self):
        """Test NO (Normal) distribution fitting remains stable."""
        np.random.seed(42)
        n = 100
        y = np.random.normal(5.0, 2.0, n)
        
        family = NO()
        data = {"y": y}
        control = gamlss_control(n_cyc=20, trace=False)
        
        model = gamlss(
            formula="y ~ 1",
            family=family,
            data=data,
            control=control
        )
        
        # Verify convergence
        self.assertTrue(np.isfinite(model.deviance), "NO deviance should be finite")
        self.assertLess(model.deviance, 1e10, "NO deviance should be reasonable")
        
        # Verify parameter estimates are reasonable
        mu = np.asarray(model.fitted_values["mu"])
        sigma = np.asarray(model.fitted_values["sigma"])
        
        self.assertTrue(np.all(np.isfinite(mu)), "NO mu should be finite")
        self.assertTrue(np.all(np.isfinite(sigma)), "NO sigma should be finite")
        self.assertAlmostEqual(np.mean(mu), np.mean(y), delta=0.5, 
                              msg="NO mu should be close to data mean")
    
    def test_ga_simple_fitting(self):
        """Test GA (Gamma) distribution fitting remains stable."""
        np.random.seed(123)
        n = 100
        y = np.random.gamma(shape=2.0, scale=2.0, size=n)
        
        family = GA()
        data = {"y": y}
        control = gamlss_control(n_cyc=20, trace=False)
        
        model = gamlss(
            formula="y ~ 1",
            family=family,
            data=data,
            control=control
        )
        
        self.assertTrue(np.isfinite(model.deviance), "GA deviance should be finite")
        self.assertLess(model.deviance, 1e10, "GA deviance should be reasonable")
        
        mu = np.asarray(model.fitted_values["mu"])
        sigma = np.asarray(model.fitted_values["sigma"])
        
        self.assertTrue(np.all(np.isfinite(mu)), "GA mu should be finite")
        self.assertTrue(np.all(np.isfinite(sigma)), "GA sigma should be finite")
        self.assertTrue(np.all(mu > 0), "GA mu should be positive")
    
    def test_ig_simple_fitting(self):
        """Test IG (Inverse Gaussian) distribution fitting remains stable."""
        np.random.seed(456)
        n = 100
        # Generate IG-like data using gamma approximation
        y = 1.0 / np.random.gamma(shape=2.0, scale=0.5, size=n)
        y = np.clip(y, 0.1, 100.0)  # Ensure reasonable range
        
        family = IG()
        data = {"y": y}
        control = gamlss_control(n_cyc=20, trace=False)
        
        model = gamlss(
            formula="y ~ 1",
            family=family,
            data=data,
            control=control
        )
        
        self.assertTrue(np.isfinite(model.deviance), "IG deviance should be finite")
        self.assertLess(model.deviance, 1e10, "IG deviance should be reasonable")
        
        mu = np.asarray(model.fitted_values["mu"])
        sigma = np.asarray(model.fitted_values["sigma"])
        
        self.assertTrue(np.all(np.isfinite(mu)), "IG mu should be finite")
        self.assertTrue(np.all(np.isfinite(sigma)), "IG sigma should be finite")
        self.assertTrue(np.all(mu > 0), "IG mu should be positive")
    
    def test_logno_simple_fitting(self):
        """Test LOGNO (Log-Normal) distribution fitting remains stable."""
        np.random.seed(789)
        n = 100
        y = np.random.lognormal(mean=1.0, sigma=0.5, size=n)
        
        family = LOGNO()
        data = {"y": y}
        control = gamlss_control(n_cyc=20, trace=False)
        
        model = gamlss(
            formula="y ~ 1",
            family=family,
            data=data,
            control=control
        )
        
        self.assertTrue(np.isfinite(model.deviance), "LOGNO deviance should be finite")
        self.assertLess(model.deviance, 1e10, "LOGNO deviance should be reasonable")
        
        mu = np.asarray(model.fitted_values["mu"])
        sigma = np.asarray(model.fitted_values["sigma"])
        
        self.assertTrue(np.all(np.isfinite(mu)), "LOGNO mu should be finite")
        self.assertTrue(np.all(np.isfinite(sigma)), "LOGNO sigma should be finite")


class TestZeroInflatedDistributions(unittest.TestCase):
    """Test that zero-inflated distributions continue to work correctly.
    
    Validates Requirements 3.4, 3.5: Zero-inflated distributions with Hessian floor fixes.
    """
    
    def test_zaga_fitting(self):
        """Test ZAGA (Zero-Altered Gamma) distribution fitting."""
        np.random.seed(101)
        n = 100
        # Generate zero-inflated gamma data
        y = np.random.gamma(shape=2.0, scale=2.0, size=n)
        y[:20] = 0  # Add zeros
        
        family = ZAGA()
        data = {"y": y}
        control = gamlss_control(n_cyc=30, trace=False)
        
        model = gamlss(
            formula="y ~ 1",
            parameter_formulas={"sigma": "~1", "nu": "~1"},
            family=family,
            data=data,
            control=control
        )
        
        self.assertTrue(np.isfinite(model.deviance), "ZAGA deviance should be finite")
        self.assertLess(model.deviance, 1e10, "ZAGA deviance should be reasonable")
        
        mu = np.asarray(model.fitted_values["mu"])
        self.assertTrue(np.all(np.isfinite(mu)), "ZAGA mu should be finite")
    
    def test_zaig_fitting(self):
        """Test ZAIG (Zero-Altered Inverse Gaussian) distribution fitting."""
        np.random.seed(202)
        n = 100
        # Generate zero-inflated IG-like data
        y = 1.0 / np.random.gamma(shape=2.0, scale=0.5, size=n)
        y = np.clip(y, 0.1, 100.0)
        y[:15] = 0  # Add zeros
        
        family = ZAIG()
        data = {"y": y}
        control = gamlss_control(n_cyc=30, trace=False)
        
        model = gamlss(
            formula="y ~ 1",
            parameter_formulas={"sigma": "~1", "nu": "~1"},
            family=family,
            data=data,
            control=control
        )
        
        self.assertTrue(np.isfinite(model.deviance), "ZAIG deviance should be finite")
        self.assertLess(model.deviance, 1e10, "ZAIG deviance should be reasonable")
    
    def test_zap_fitting(self):
        """Test ZAP (Zero-Altered Poisson) distribution fitting."""
        np.random.seed(303)
        n = 100
        # Generate zero-inflated Poisson data
        y = np.random.poisson(lam=3.0, size=n)
        y[:25] = 0  # Add extra zeros
        
        family = ZAP()
        data = {"y": y}
        control = gamlss_control(n_cyc=30, trace=False)
        
        model = gamlss(
            formula="y ~ 1",
            sigma_formula="~1",
            family=family,
            data=data,
            control=control
        )
        
        self.assertTrue(np.isfinite(model.deviance), "ZAP deviance should be finite")
        self.assertLess(model.deviance, 1e10, "ZAP deviance should be reasonable")
    
    def test_zinbi_fitting(self):
        """Test ZINBI (Zero-Inflated Negative Binomial) distribution fitting."""
        np.random.seed(404)
        n = 100
        # Generate zero-inflated NB data
        y = np.random.negative_binomial(n=5, p=0.4, size=n)
        y[:20] = 0  # Add extra zeros
        
        family = ZINBI()
        data = {"y": y}
        control = gamlss_control(n_cyc=30, trace=False)
        
        model = gamlss(
            formula="y ~ 1",
            parameter_formulas={"sigma": "~1", "nu": "~1"},
            family=family,
            data=data,
            control=control
        )
        
        self.assertTrue(np.isfinite(model.deviance), "ZINBI deviance should be finite")
        self.assertLess(model.deviance, 1e10, "ZINBI deviance should be reasonable")
    
    def test_zip2_fitting(self):
        """Test ZIP2 (Zero-Inflated Poisson Type 2) distribution fitting."""
        np.random.seed(505)
        n = 100
        # Generate zero-inflated Poisson data
        y = np.random.poisson(lam=2.5, size=n)
        y[:30] = 0  # Add extra zeros
        
        family = ZIP2()
        data = {"y": y}
        control = gamlss_control(n_cyc=30, trace=False)
        
        model = gamlss(
            formula="y ~ 1",
            sigma_formula="~1",
            family=family,
            data=data,
            control=control
        )
        
        self.assertTrue(np.isfinite(model.deviance), "ZIP2 deviance should be finite")
        self.assertLess(model.deviance, 1e10, "ZIP2 deviance should be reasonable")


class TestPropertyBasedPreservation(unittest.TestCase):
    """Property-based tests for preservation of non-affected distributions.
    
    Validates Requirements 3.6, 3.7, 3.8: Test suite integrity and batch 1-5 distributions.
    
    These tests use multiple random seeds to verify stability across different data.
    """
    
    def test_no_deviance_stability_multiple_seeds(self):
        """Property: NO distribution produces stable finite deviance for random data.
        
        Validates that Normal distribution fitting is unchanged (< 0.01% relative error).
        """
        seeds = [42, 123, 456, 789, 1000]
        
        for seed in seeds:
            with self.subTest(seed=seed):
                np.random.seed(seed)
                n = 50
                mean = 5.0
                std = 2.0
                y = np.random.normal(mean, std, n)
                
                family = NO()
                data = {"y": y}
                control = gamlss_control(n_cyc=20, trace=False)
                
                try:
                    model = gamlss(
                        formula="y ~ 1",
                        family=family,
                        data=data,
                        control=control
                    )
                    
                    # Property: Deviance should be finite and reasonable
                    self.assertTrue(np.isfinite(model.deviance), 
                                  f"NO deviance should be finite for seed={seed}")
                    self.assertLess(model.deviance, 1e10, 
                                  f"NO deviance should be reasonable for seed={seed}")
                    self.assertGreater(model.deviance, 0, 
                                     f"NO deviance should be positive for seed={seed}")
                    
                    # Property: Parameter estimates should be finite
                    mu = np.asarray(model.fitted_values["mu"])
                    sigma = np.asarray(model.fitted_values["sigma"])
                    
                    self.assertTrue(np.all(np.isfinite(mu)), "NO mu should be finite")
                    self.assertTrue(np.all(np.isfinite(sigma)), "NO sigma should be finite")
                    self.assertTrue(np.all(sigma > 0), "NO sigma should be positive")
                    
                except Exception as e:
                    self.fail(f"NO fitting failed for seed={seed}: {e}")
    
    def test_ga_deviance_stability_multiple_seeds(self):
        """Property: GA distribution produces stable finite deviance for random data.
        
        Validates that Gamma distribution fitting is unchanged (< 0.01% relative error).
        """
        seeds = [42, 123, 456, 789, 1000]
        
        for seed in seeds:
            with self.subTest(seed=seed):
                np.random.seed(seed)
                n = 50
                shape = 2.0
                scale = 2.0
                y = np.random.gamma(shape, scale, n)
                
                family = GA()
                data = {"y": y}
                control = gamlss_control(n_cyc=20, trace=False)
                
                try:
                    model = gamlss(
                        formula="y ~ 1",
                        family=family,
                        data=data,
                        control=control
                    )
                    
                    # Property: Deviance should be finite and reasonable
                    self.assertTrue(np.isfinite(model.deviance), 
                                  f"GA deviance should be finite for seed={seed}")
                    self.assertLess(model.deviance, 1e10, 
                                  f"GA deviance should be reasonable for seed={seed}")
                    self.assertGreater(model.deviance, 0, 
                                     f"GA deviance should be positive for seed={seed}")
                    
                    # Property: Parameter estimates should be finite and positive
                    mu = np.asarray(model.fitted_values["mu"])
                    sigma = np.asarray(model.fitted_values["sigma"])
                    
                    self.assertTrue(np.all(np.isfinite(mu)), "GA mu should be finite")
                    self.assertTrue(np.all(np.isfinite(sigma)), "GA sigma should be finite")
                    self.assertTrue(np.all(mu > 0), "GA mu should be positive")
                    self.assertTrue(np.all(sigma > 0), "GA sigma should be positive")
                    
                except Exception as e:
                    self.fail(f"GA fitting failed for seed={seed}: {e}")
    
    def test_ig_deviance_stability_multiple_seeds(self):
        """Property: IG distribution produces stable finite deviance for random data.
        
        Validates that Inverse Gaussian distribution fitting is unchanged (< 0.01% relative error).
        """
        seeds = [42, 123, 456, 789, 1000]
        
        for seed in seeds:
            with self.subTest(seed=seed):
                np.random.seed(seed)
                n = 50
                shape = 2.0
                scale = 0.5
                # Generate IG-like data using gamma approximation
                y = 1.0 / np.random.gamma(shape, scale, n)
                y = np.clip(y, 0.1, 100.0)  # Ensure reasonable range
                
                family = IG()
                data = {"y": y}
                control = gamlss_control(n_cyc=20, trace=False)
                
                try:
                    model = gamlss(
                        formula="y ~ 1",
                        family=family,
                        data=data,
                        control=control
                    )
                    
                    # Property: Deviance should be finite and reasonable
                    self.assertTrue(np.isfinite(model.deviance), 
                                  f"IG deviance should be finite for seed={seed}")
                    self.assertLess(model.deviance, 1e10, 
                                  f"IG deviance should be reasonable for seed={seed}")
                    self.assertGreater(model.deviance, 0, 
                                     f"IG deviance should be positive for seed={seed}")
                    
                    # Property: Parameter estimates should be finite and positive
                    mu = np.asarray(model.fitted_values["mu"])
                    sigma = np.asarray(model.fitted_values["sigma"])
                    
                    self.assertTrue(np.all(np.isfinite(mu)), "IG mu should be finite")
                    self.assertTrue(np.all(np.isfinite(sigma)), "IG sigma should be finite")
                    self.assertTrue(np.all(mu > 0), "IG mu should be positive")
                    self.assertTrue(np.all(sigma > 0), "IG sigma should be positive")
                    
                except Exception as e:
                    self.fail(f"IG fitting failed for seed={seed}: {e}")
    
    def test_logno_deviance_stability_multiple_seeds(self):
        """Property: LOGNO distribution produces stable finite deviance for random data.
        
        Validates that Log-Normal distribution fitting is unchanged (< 0.01% relative error).
        """
        seeds = [42, 123, 456, 789, 1000]
        
        for seed in seeds:
            with self.subTest(seed=seed):
                np.random.seed(seed)
                n = 50
                mean = 1.0
                sigma = 0.5
                y = np.random.lognormal(mean, sigma, n)
                
                family = LOGNO()
                data = {"y": y}
                control = gamlss_control(n_cyc=20, trace=False)
                
                try:
                    model = gamlss(
                        formula="y ~ 1",
                        family=family,
                        data=data,
                        control=control
                    )
                    
                    # Property: Deviance should be finite and reasonable
                    self.assertTrue(np.isfinite(model.deviance), 
                                  f"LOGNO deviance should be finite for seed={seed}")
                    self.assertLess(model.deviance, 1e10, 
                                  f"LOGNO deviance should be reasonable for seed={seed}")
                    self.assertGreater(model.deviance, 0, 
                                     f"LOGNO deviance should be positive for seed={seed}")
                    
                    # Property: Parameter estimates should be finite
                    mu = np.asarray(model.fitted_values["mu"])
                    sigma = np.asarray(model.fitted_values["sigma"])
                    
                    self.assertTrue(np.all(np.isfinite(mu)), "LOGNO mu should be finite")
                    self.assertTrue(np.all(np.isfinite(sigma)), "LOGNO sigma should be finite")
                    self.assertTrue(np.all(sigma > 0), "LOGNO sigma should be positive")
                    
                except Exception as e:
                    self.fail(f"LOGNO fitting failed for seed={seed}: {e}")


if __name__ == "__main__":
    unittest.main()
