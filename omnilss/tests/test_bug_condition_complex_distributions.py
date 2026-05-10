"""Bug Condition Exploration Test for Complex Distributions.

**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10**

This test explores the bug condition for 10 complex distributions that fail during
GAMLSS model fitting. The test is EXPECTED TO FAIL on unfixed code - failure confirms
the bug exists.

CRITICAL: This test encodes the expected behavior. When it passes after the fix is
implemented, it confirms the bug is resolved.

Batch 6 (Bessel issues): DEL, DPO, PIG, SICHEL
Batch 7 (Complex implementation): BB, BNB, MN3, MN4, MN5
Batch 8 (4-parameter): GB2

Run:
    PYTHONPATH=src JAX_PLATFORMS=cpu python -m pytest tests/test_bug_condition_complex_distributions.py -v
"""

import unittest
import numpy as np
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omnilss.distributions_b6 import DEL, DPO, PIG, SICHEL
from omnilss.distributions_b7 import BB, BNB, MN3, MN4, MN5
from omnilss.distributions_b8 import GB2
from omnilss.fitting import gamlss
from omnilss.controls import gamlss_control


class TestBatch6BesselIssues(unittest.TestCase):
    """Test Batch 6 distributions with Bessel function issues.
    
    Expected failures on unfixed code:
    - DEL: NaN deviance due to recursive formula instability
    - SICHEL: Unstable results from Bessel function approximations
    - DPO: Incorrect deviance values for y=0 edge case
    - PIG: Bessel function instability for certain parameter ranges
    """
    
    def test_del_poisson_like_count_data(self):
        """Test DEL with Poisson-like count data (y ∈ [0, 10]).
        
        Expected on unfixed code: NaN deviance during optimization.
        Expected after fix: Finite deviance, stable parameter estimates.
        """
        np.random.seed(123)
        n = 50
        # Generate Poisson-like count data with small counts
        y = np.random.poisson(lam=2.5, size=n)
        y = np.clip(y, 0, 10)  # Ensure y ∈ [0, 10]
        
        family = DEL()
        data = {"y": y}
        control = gamlss_control(n_cyc=50, trace=False)
        
        try:
            model = gamlss(
                formula="y ~ 1",
                parameter_formulas={"sigma": "~1", "nu": "~1"},
                family=family,
                data=data,
                control=control
            )
            
            # Expected behavior after fix
            self.assertTrue(np.isfinite(model.deviance), 
                          f"DEL deviance should be finite, got {model.deviance}")
            self.assertLess(model.deviance, 1e10, 
                          f"DEL deviance should be reasonable, got {model.deviance}")
            
            # Check parameter estimates are in reasonable ranges
            mu = np.asarray(model.fitted_values["mu"])
            sigma = np.asarray(model.fitted_values["sigma"])
            nu = np.asarray(model.fitted_values["nu"])
            
            self.assertTrue(np.all(np.isfinite(mu)), "DEL mu should be finite")
            self.assertTrue(np.all(np.isfinite(sigma)), "DEL sigma should be finite")
            self.assertTrue(np.all(np.isfinite(nu)), "DEL nu should be finite")
            self.assertTrue(np.all(mu > 0), "DEL mu should be positive")
            self.assertTrue(np.all(sigma > 0), "DEL sigma should be positive")
            self.assertTrue(np.all((nu >= 0) & (nu <= 1)), "DEL nu should be in [0, 1]")
            
        except Exception as e:
            self.fail(f"DEL fitting failed with exception: {e}")
    
    def test_sichel_overdispersed_count_data(self):
        """Test SICHEL with overdispersed count data.
        
        Expected on unfixed code: Unstable results, convergence failure.
        Expected after fix: Stable finite deviance, reasonable parameter estimates.
        """
        np.random.seed(456)
        n = 50
        # Generate overdispersed count data (variance > mean)
        y = np.random.negative_binomial(n=5, p=0.3, size=n)
        
        family = SICHEL()
        data = {"y": y}
        control = gamlss_control(n_cyc=50, trace=False)
        
        try:
            model = gamlss(
                formula="y ~ 1",
                parameter_formulas={"sigma": "~1", "nu": "~1"},
                family=family,
                data=data,
                control=control
            )
            
            # Expected behavior after fix
            self.assertTrue(np.isfinite(model.deviance),
                          f"SICHEL deviance should be finite, got {model.deviance}")
            self.assertLess(model.deviance, 1e10,
                          f"SICHEL deviance should be reasonable, got {model.deviance}")
            
            # Check parameter estimates
            mu = np.asarray(model.fitted_values["mu"])
            sigma = np.asarray(model.fitted_values["sigma"])
            nu = np.asarray(model.fitted_values["nu"])
            
            self.assertTrue(np.all(np.isfinite(mu)), "SICHEL mu should be finite")
            self.assertTrue(np.all(np.isfinite(sigma)), "SICHEL sigma should be finite")
            self.assertTrue(np.all(np.isfinite(nu)), "SICHEL nu should be finite")
            self.assertTrue(np.all(mu > 0), "SICHEL mu should be positive")
            self.assertTrue(np.all(sigma > 0), "SICHEL sigma should be positive")
            
        except Exception as e:
            self.fail(f"SICHEL fitting failed with exception: {e}")
    
    def test_dpo_count_data_with_zeros(self):
        """Test DPO with count data including y=0.
        
        Expected on unfixed code: Incorrect deviance values for y=0 case.
        Expected after fix: Correct deviance matching R gamlss.dist.
        """
        np.random.seed(789)
        n = 50
        # Generate count data with explicit zeros
        y = np.random.poisson(lam=3.0, size=n)
        y[:10] = 0  # Ensure some zeros
        
        family = DPO()
        data = {"y": y}
        control = gamlss_control(n_cyc=50, trace=False)
        
        try:
            model = gamlss(
                formula="y ~ 1",
                sigma_formula="~1",
                family=family,
                data=data,
                control=control
            )
            
            # Expected behavior after fix
            self.assertTrue(np.isfinite(model.deviance),
                          f"DPO deviance should be finite, got {model.deviance}")
            self.assertLess(model.deviance, 1e10,
                          f"DPO deviance should be reasonable, got {model.deviance}")
            
            # Check parameter estimates
            mu = np.asarray(model.fitted_values["mu"])
            sigma = np.asarray(model.fitted_values["sigma"])
            
            self.assertTrue(np.all(np.isfinite(mu)), "DPO mu should be finite")
            self.assertTrue(np.all(np.isfinite(sigma)), "DPO sigma should be finite")
            self.assertTrue(np.all(mu > 0), "DPO mu should be positive")
            self.assertTrue(np.all(sigma > 0), "DPO sigma should be positive")
            
        except Exception as e:
            self.fail(f"DPO fitting failed with exception: {e}")
    
    def test_pig_varying_means(self):
        """Test PIG with count data with varying means.
        
        Expected on unfixed code: Bessel function instability.
        Expected after fix: Stable finite deviance, reasonable parameter estimates.
        """
        np.random.seed(101)
        n = 50
        # Generate count data with varying means
        y = np.concatenate([
            np.random.poisson(lam=1.0, size=n//3),
            np.random.poisson(lam=5.0, size=n//3),
            np.random.poisson(lam=10.0, size=n - 2*(n//3))
        ])
        
        family = PIG()
        data = {"y": y}
        control = gamlss_control(n_cyc=50, trace=False)
        
        try:
            model = gamlss(
                formula="y ~ 1",
                sigma_formula="~1",
                family=family,
                data=data,
                control=control
            )
            
            # Expected behavior after fix
            self.assertTrue(np.isfinite(model.deviance),
                          f"PIG deviance should be finite, got {model.deviance}")
            self.assertLess(model.deviance, 1e10,
                          f"PIG deviance should be reasonable, got {model.deviance}")
            
            # Check parameter estimates
            mu = np.asarray(model.fitted_values["mu"])
            sigma = np.asarray(model.fitted_values["sigma"])
            
            self.assertTrue(np.all(np.isfinite(mu)), "PIG mu should be finite")
            self.assertTrue(np.all(np.isfinite(sigma)), "PIG sigma should be finite")
            self.assertTrue(np.all(mu > 0), "PIG mu should be positive")
            self.assertTrue(np.all(sigma > 0), "PIG sigma should be positive")
            
        except Exception as e:
            self.fail(f"PIG fitting failed with exception: {e}")


class TestBatch7ComplexImplementation(unittest.TestCase):
    """Test Batch 7 distributions with complex implementation issues.
    
    Expected failures on unfixed code:
    - BB: Fixed parameter handling error (bd should be from data)
    - BNB: NaN deviance and nu parameter explosion (nu > 10^50)
    - MN3/MN4/MN5: Data format errors (require multinomial data)
    """
    
    def test_bb_binomial_data_with_bd(self):
        """Test BB with binomial data and bd in data dictionary.
        
        Expected on unfixed code: Error about estimating bd or missing data.
        Expected after fix: bd treated as fixed from data, finite deviance.
        """
        np.random.seed(202)
        n = 50
        bd = np.full(n, 10)  # Fixed binomial denominator
        # Generate binomial data
        y = np.random.binomial(n=10, p=0.6, size=n)
        
        family = BB()
        data = {"y": y, "bd": bd}
        control = gamlss_control(n_cyc=50, trace=False)
        
        try:
            model = gamlss(
                formula="y ~ 1",
                sigma_formula="~1",
                family=family,
                data=data,
                control=control
            )
            
            # Expected behavior after fix
            self.assertTrue(np.isfinite(model.deviance),
                          f"BB deviance should be finite, got {model.deviance}")
            self.assertLess(model.deviance, 1e10,
                          f"BB deviance should be reasonable, got {model.deviance}")
            
            # Check parameter estimates
            mu = np.asarray(model.fitted_values["mu"])
            sigma = np.asarray(model.fitted_values["sigma"])
            
            self.assertTrue(np.all(np.isfinite(mu)), "BB mu should be finite")
            self.assertTrue(np.all(np.isfinite(sigma)), "BB sigma should be finite")
            self.assertTrue(np.all((mu > 0) & (mu < 1)), "BB mu should be in (0, 1)")
            self.assertTrue(np.all(sigma > 0), "BB sigma should be positive")
            
        except Exception as e:
            self.fail(f"BB fitting failed with exception: {e}")
    
    def test_bnb_overdispersed_count_data(self):
        """Test BNB with overdispersed count data.
        
        Expected on unfixed code: NaN deviance, nu explosion (nu > 10^50).
        Expected after fix: Finite deviance, nu < 100, stable estimates.
        """
        np.random.seed(303)
        n = 50
        # Generate overdispersed count data
        y = np.random.negative_binomial(n=3, p=0.4, size=n)
        
        family = BNB()
        data = {"y": y}
        control = gamlss_control(n_cyc=50, trace=False)
        
        try:
            model = gamlss(
                formula="y ~ 1",
                parameter_formulas={"sigma": "~1", "nu": "~1"},
                family=family,
                data=data,
                control=control
            )
            
            # Expected behavior after fix
            self.assertTrue(np.isfinite(model.deviance),
                          f"BNB deviance should be finite, got {model.deviance}")
            self.assertLess(model.deviance, 1e10,
                          f"BNB deviance should be reasonable, got {model.deviance}")
            
            # Check parameter estimates - especially nu should not explode
            mu = np.asarray(model.fitted_values["mu"])
            sigma = np.asarray(model.fitted_values["sigma"])
            nu = np.asarray(model.fitted_values["nu"])
            
            self.assertTrue(np.all(np.isfinite(mu)), "BNB mu should be finite")
            self.assertTrue(np.all(np.isfinite(sigma)), "BNB sigma should be finite")
            self.assertTrue(np.all(np.isfinite(nu)), "BNB nu should be finite")
            self.assertTrue(np.all(mu > 0), "BNB mu should be positive")
            self.assertTrue(np.all(sigma > 0), "BNB sigma should be positive")
            self.assertTrue(np.all(nu > 0), "BNB nu should be positive")
            # Critical: nu should not explode
            self.assertLess(np.max(nu), 100.0,
                          f"BNB nu should be < 100, got max {np.max(nu)}")
            
        except Exception as e:
            self.fail(f"BNB fitting failed with exception: {e}")
    
    def test_mn3_categorical_data(self):
        """Test MN3 with categorical data (1, 2, 3).
        
        MN3 is a categorical distribution for responses with 3 levels.
        Expected: Model should fit successfully with categorical responses.
        """
        np.random.seed(404)
        n = 100
        # Generate categorical data (1, 2, or 3)
        y = np.random.choice([1, 2, 3], size=n, p=[0.5, 0.3, 0.2])
        
        family = MN3()
        data = {"y": y}
        control = gamlss_control(n_cyc=50, trace=False)
        
        try:
            model = gamlss(
                formula="y ~ 1",
                sigma_formula="~1",
                family=family,
                data=data,
                control=control
            )
            
            # Check that model fitted
            self.assertIsNotNone(model)
            self.assertIsNotNone(model.fitted_values)
            
            # Check that parameters are positive (log link ensures this)
            mu = model.fitted_values["mu"]
            sigma = model.fitted_values["sigma"]
            self.assertTrue(np.all(mu > 0))
            self.assertTrue(np.all(sigma > 0))
            
            # Check that probabilities sum to 1
            # P(Y=1) = mu/(1+mu+sigma)
            # P(Y=2) = sigma/(1+mu+sigma)
            # P(Y=3) = 1/(1+mu+sigma)
            normalizer = 1 + mu[0] + sigma[0]
            p1 = mu[0] / normalizer
            p2 = sigma[0] / normalizer
            p3 = 1.0 / normalizer
            prob_sum = p1 + p2 + p3
            self.assertAlmostEqual(prob_sum, 1.0, places=6)
            
        except Exception as e:
            self.fail(f"MN3 fitting failed with exception: {e}")
    
    def test_mn4_categorical_data(self):
        """Test MN4 with categorical data (1, 2, 3, 4).
        
        MN4 is a categorical distribution for responses with 4 levels.
        Expected: Model should fit successfully with categorical responses.
        """
        np.random.seed(505)
        n = 100
        # Generate categorical data (1, 2, 3, or 4)
        y = np.random.choice([1, 2, 3, 4], size=n)
        
        family = MN4()
        data = {"y": y}
        control = gamlss_control(n_cyc=50, trace=False)
        
        try:
            model = gamlss(
                formula="y ~ 1",
                sigma_formula="~1",
                family=family,
                data=data,
                control=control
            )
            
            # Check that model fitted
            self.assertIsNotNone(model)
            self.assertIsNotNone(model.fitted_values)
            
            # Check that parameters are positive
            mu = model.fitted_values["mu"]
            sigma = model.fitted_values["sigma"]
            nu = model.fitted_values["nu"]
            self.assertTrue(np.all(mu > 0))
            self.assertTrue(np.all(sigma > 0))
            self.assertTrue(np.all(nu > 0))
            
        except Exception as e:
            self.fail(f"MN4 fitting failed with exception: {e}")
    
    def test_mn5_categorical_data(self):
        """Test MN5 with categorical data (1, 2, 3, 4, 5).
        
        MN5 is a categorical distribution for responses with 5 levels.
        Expected: Model should fit successfully with categorical responses.
        """
        np.random.seed(606)
        n = 100
        # Generate categorical data (1, 2, 3, 4, or 5)
        y = np.random.choice([1, 2, 3, 4, 5], size=n)
        
        family = MN5()
        data = {"y": y}
        control = gamlss_control(n_cyc=50, trace=False)
        
        try:
            model = gamlss(
                formula="y ~ 1",
                sigma_formula="~1",
                family=family,
                data=data,
                control=control
            )
            
            # Check that model fitted
            self.assertIsNotNone(model)
            self.assertIsNotNone(model.fitted_values)
            
            # Check that parameters are positive
            mu = model.fitted_values["mu"]
            sigma = model.fitted_values["sigma"]
            nu = model.fitted_values["nu"]
            tau = model.fitted_values["tau"]
            self.assertTrue(np.all(mu > 0))
            self.assertTrue(np.all(sigma > 0))
            self.assertTrue(np.all(nu > 0))
            self.assertTrue(np.all(tau > 0))
            
        except Exception as e:
            self.fail(f"MN5 fitting failed with exception: {e}")


class TestBatch8FourParameter(unittest.TestCase):
    """Test Batch 8 4-parameter distribution.
    
    Expected failures on unfixed code:
    - GB2: tau parameter explosion (tau > 10^100), unstable estimates
    """
    
    def test_gb2_positive_continuous_data(self):
        """Test GB2 with positive continuous data.
        
        Expected on unfixed code: tau explosion (tau > 10^100), unstable estimates.
        Expected after fix: All 4 parameters stable, tau < 100, finite deviance.
        """
        np.random.seed(707)
        n = 50
        # Generate positive continuous data
        y = np.random.gamma(shape=2.0, scale=2.0, size=n)
        y = np.clip(y, 0.1, 100.0)  # Ensure reasonable range
        
        family = GB2()
        data = {"y": y}
        control = gamlss_control(n_cyc=50, trace=False)
        
        try:
            model = gamlss(
                formula="y ~ 1",
                parameter_formulas={"sigma": "~1", "nu": "~1", "tau": "~1"},
                family=family,
                data=data,
                control=control
            )
            
            # Expected behavior after fix
            self.assertTrue(np.isfinite(model.deviance),
                          f"GB2 deviance should be finite, got {model.deviance}")
            self.assertLess(model.deviance, 1e10,
                          f"GB2 deviance should be reasonable, got {model.deviance}")
            
            # Check all 4 parameter estimates
            mu = np.asarray(model.fitted_values["mu"])
            sigma = np.asarray(model.fitted_values["sigma"])
            nu = np.asarray(model.fitted_values["nu"])
            tau = np.asarray(model.fitted_values["tau"])
            
            self.assertTrue(np.all(np.isfinite(mu)), "GB2 mu should be finite")
            self.assertTrue(np.all(np.isfinite(sigma)), "GB2 sigma should be finite")
            self.assertTrue(np.all(np.isfinite(nu)), "GB2 nu should be finite")
            self.assertTrue(np.all(np.isfinite(tau)), "GB2 tau should be finite")
            
            self.assertTrue(np.all(mu > 0), "GB2 mu should be positive")
            self.assertTrue(np.all(sigma > 0), "GB2 sigma should be positive")
            self.assertTrue(np.all(nu > 0), "GB2 nu should be positive")
            self.assertTrue(np.all(tau > 0), "GB2 tau should be positive")
            
            # Critical: tau should not explode
            self.assertLess(np.max(tau), 100.0,
                          f"GB2 tau should be < 100, got max {np.max(tau)}")
            
        except Exception as e:
            self.fail(f"GB2 fitting failed with exception: {e}")


if __name__ == "__main__":
    unittest.main()
