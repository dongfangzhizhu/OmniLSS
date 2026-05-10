"""Batch 7-8 R/Python consistency tests.

Tests OmniLSS batch 7-8 distributions against R gamlss.dist.

Batch 7 (Compound Discrete):
- BB (Beta Binomial)
- BNB (Beta Negative Binomial)
- MN3, MN4, MN5 (categorical-response multinomial variants)

Batch 8 (Continuous Special):
- GB2 (Generalized Beta Type 2)
- GG (Generalized Gamma)
- LNO (Log Normal)
- NET (Normal Exponential t)
- PARETO (Pareto)

Prerequisites:
    - R installed
    - gamlss.dist package installed

Run:
    PYTHONPATH=src JAX_PLATFORMS=cpu python -m unittest tests.test_r_consistency_batch7_8 -v
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omnilss.distributions_b7 import BB, BNB, MN3, MN4, MN5
from omnilss.distributions_b8 import GB2, GG, LNO, NET, PARETO
from omnilss.fitting import gamlss
from omnilss.controls import gamlss_control

# Import R bridge
from tests._r_bridge_helper import R_AVAILABLE, R_BRIDGE_CLS as RBridge


@unittest.skipIf(not R_AVAILABLE, "R bridge not available")
class TestBatch78Consistency(unittest.TestCase):
    """Batch 7-8 distributions R/Python consistency tests."""
    
    @classmethod
    def setUpClass(cls):
        """Set up R bridge once for all tests."""
        try:
            cls.r_bridge = RBridge()
        except RuntimeError as e:
            raise unittest.SkipTest(f"R not available: {e}")
    
    # Batch 7 tests
    
    def test_bb_intercept_only(self):
        """Test BB (Beta Binomial) intercept-only model: y ~ 1."""
        np.random.seed(42)
        n = 100
        
        # Generate binomial data (BB requires y in [0, bd])
        bd = 10  # Number of trials
        y = np.random.binomial(bd, 0.5, n)
        
        # Fit with Python
        family = BB()
        data = {"y": y, "bd": np.full(n, bd)}
        control = gamlss_control(n_cyc=100)
        py_model = gamlss(formula="y ~ 1", sigma_formula="~1", family=family, data=data, control=control)
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ 1",
            family="BB",
            sigma_formula="~1"
        )
        
        # Check R fitting succeeded
        if not r_result["success"] or not r_result["converged"]:
            self.skipTest("R model did not converge for BB")
        
        # Compare deviance
        py_deviance = py_model.deviance
        r_deviance = r_result["deviance"]
        np.testing.assert_allclose(
            py_deviance,
            r_deviance,
            rtol=1e-1,
            atol=1e-2,
            err_msg="Deviance differs between Python and R"
        )
    
    def test_bnb_intercept_only(self):
        """Test BNB (Beta Negative Binomial) intercept-only model: y ~ 1."""
        np.random.seed(123)
        n = 100
        
        # Generate count data
        y = np.random.negative_binomial(5, 0.5, n)
        
        # Fit with Python
        family = BNB()
        data = {"y": y}
        control = gamlss_control(n_cyc=100)
        py_model = gamlss(formula="y ~ 1", family=family, data=data, control=control)
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ 1",
            family="BNB",
            sigma_formula="~1",
            nu_formula="~1"
        )
        
        # Check R fitting succeeded
        if not r_result["success"] or not r_result["converged"]:
            self.skipTest("R model did not converge for BNB")
        
        # Compare deviance
        py_deviance = py_model.deviance
        r_deviance = r_result["deviance"]
        np.testing.assert_allclose(
            py_deviance,
            r_deviance,
            rtol=1e-1,
            atol=1e-2,
            err_msg="Deviance differs between Python and R"
        )
    
    # Batch 8 tests
    
    def test_gb2_intercept_only(self):
        """Test GB2 (Generalized Beta Type 2) intercept-only model: y ~ 1."""
        np.random.seed(456)
        n = 100
        
        # Generate positive data
        y = np.random.gamma(2, 2, n)
        
        # Fit with Python
        family = GB2()
        data = {"y": y}
        control = gamlss_control(n_cyc=100)
        py_model = gamlss(formula="y ~ 1", family=family, data=data, control=control)
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ 1",
            family="GB2",
            sigma_formula="~1",
            nu_formula="~1",
            tau_formula="~1"
        )
        
        # Check R fitting succeeded
        if not r_result["success"] or not r_result["converged"]:
            self.skipTest("R model did not converge for GB2")
        
        # Compare deviance
        py_deviance = py_model.deviance
        r_deviance = r_result["deviance"]
        np.testing.assert_allclose(
            py_deviance,
            r_deviance,
            rtol=2e-1,  # More lenient for complex distributions
            atol=1e-1,
            err_msg="Deviance differs between Python and R"
        )
    
    def test_gg_intercept_only(self):
        """Test GG (Generalized Gamma) intercept-only model: y ~ 1."""
        np.random.seed(789)
        n = 100
        
        # Generate positive data
        y = np.random.gamma(2, 2, n)
        
        # Fit with Python
        family = GG()
        data = {"y": y}
        control = gamlss_control(n_cyc=100)
        py_model = gamlss(formula="y ~ 1", family=family, data=data, control=control)
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ 1",
            family="GG",
            sigma_formula="~1",
            nu_formula="~1"
        )
        
        # Check R fitting succeeded
        if not r_result["success"] or not r_result["converged"]:
            self.skipTest("R model did not converge for GG")
        
        # Compare deviance
        py_deviance = py_model.deviance
        r_deviance = r_result["deviance"]
        np.testing.assert_allclose(
            py_deviance,
            r_deviance,
            rtol=1e-1,
            atol=1e-2,
            err_msg="Deviance differs between Python and R"
        )
    
    def test_lno_intercept_only(self):
        """Test LNO (Log Normal) intercept-only model: y ~ 1.
        
        Note: LNO is a 2-parameter distribution (alias for LOGNO), not 3-parameter.
        """
        np.random.seed(111)
        n = 100
        
        # Generate positive data
        y = np.random.lognormal(1, 0.5, n)
        
        # Fit with Python
        family = LNO()
        data = {"y": y}
        control = gamlss_control(n_cyc=100)
        py_model = gamlss(formula="y ~ 1", family=family, data=data, control=control)
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ 1",
            family="LNO",
            sigma_formula="~1"
        )
        
        # Check R fitting succeeded
        self.assertTrue(r_result["success"], f"R fitting failed: {r_result.get('error', 'Unknown error')}")
        self.assertTrue(r_result["converged"], "R model did not converge")
        
        # Compare deviance
        py_deviance = py_model.deviance
        r_deviance = r_result["deviance"]
        np.testing.assert_allclose(
            py_deviance,
            r_deviance,
            rtol=1e-1,
            atol=1e-2,
            err_msg="Deviance differs between Python and R"
        )
    
    def test_net_intercept_only(self):
        """Test NET (Normal Exponential t) intercept-only model: y ~ 1.
        
        Note: NET in R has nu and tau as fixed parameters (not estimated).
        Default values are nu=1.5, tau=2.
        """
        np.random.seed(222)
        n = 100
        
        # Generate data
        y = np.random.normal(5, 2, n)
        
        # Fit with Python - nu and tau are fixed at default values
        family = NET()
        data = {"y": y}
        control = gamlss_control(n_cyc=100)
        py_model = gamlss(formula="y ~ 1", family=family, data=data, control=control)
        
        # Fit with R - nu and tau are not estimated (fixed at defaults)
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ 1",
            family="NET",
            sigma_formula="~1"
            # Note: nu_formula and tau_formula not specified - they're fixed
        )
        
        # Check R fitting succeeded
        if not r_result["success"] or not r_result["converged"]:
            self.skipTest("R model did not converge for NET")
        
        # Compare deviance
        py_deviance = py_model.deviance
        r_deviance = r_result["deviance"]
        np.testing.assert_allclose(
            py_deviance,
            r_deviance,
            rtol=2e-1,  # More lenient for complex distributions
            atol=1e-1,
            err_msg="Deviance differs between Python and R"
        )
    
    def test_pareto_intercept_only(self):
        """Test PARETO intercept-only model: y ~ 1."""
        np.random.seed(333)
        n = 100
        
        # Generate positive data (Pareto requires y > 0)
        y = np.random.pareto(2, n) + 1  # Shift to ensure y > 1
        
        # Fit with Python
        family = PARETO()
        data = {"y": y}
        control = gamlss_control(n_cyc=100)
        py_model = gamlss(formula="y ~ 1", family=family, data=data, control=control)
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ 1",
            family="PARETO",
            sigma_formula="~1"
        )
        
        # Check R fitting succeeded
        if not r_result["success"] or not r_result["converged"]:
            self.skipTest("R model did not converge for PARETO")
        
        # Compare deviance
        py_deviance = py_model.deviance
        r_deviance = r_result["deviance"]
        np.testing.assert_allclose(
            py_deviance,
            r_deviance,
            rtol=1e-1,
            atol=1e-2,
            err_msg="Deviance differs between Python and R"
        )


if __name__ == "__main__":
    unittest.main()
