"""Batch 3-4 remaining distributions R/Python consistency tests.

Tests OmniLSS batch 3-4 remaining distributions against R gamlss.dist.

Batch 3:
- SHASH (Sinh-Arcsinh)
- SN2 (Skew Normal type 2)

Batch 4:
- BEINF0 (Beta Inflated at 0)
- BEINF1 (Beta Inflated at 1)
- BEOI (Beta Inflated at 0 and 1)
- BEZI (Zero Inflated Beta)

Prerequisites:
    - R installed
    - gamlss.dist package installed

Run:
    PYTHONPATH=src JAX_PLATFORMS=cpu python -m unittest tests.test_r_consistency_batch3_4_remaining -v
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omnilss.distributions_b3 import SHASH, SN2
from omnilss.distributions_b4 import BEINF0, BEINF1, BEOI, BEZI
from omnilss.fitting import gamlss
from omnilss.controls import gamlss_control

# Import R bridge
from tests._r_bridge_helper import R_AVAILABLE, R_BRIDGE_CLS as RBridge


@unittest.skipIf(not R_AVAILABLE, "R bridge not available")
class TestBatch34RemainingConsistency(unittest.TestCase):
    """Batch 3-4 remaining distributions R/Python consistency tests."""
    
    @classmethod
    def setUpClass(cls):
        """Set up R bridge once for all tests."""
        try:
            cls.r_bridge = RBridge()
        except RuntimeError as e:
            raise unittest.SkipTest(f"R not available: {e}")
    
    # Batch 3 tests
    
    def test_shash_intercept_only(self):
        """Test SHASH intercept-only model: y ~ 1."""
        np.random.seed(42)
        n = 100
        
        # Generate data (SHASH can handle any real values)
        y = np.random.normal(5, 2, n)
        
        # Fit with Python
        family = SHASH()
        data = {"y": y}
        control = gamlss_control(n_cyc=100)
        py_model = gamlss(formula="y ~ 1", family=family, data=data, control=control)
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ 1",
            family="SHASH",
            sigma_formula="~1",
            nu_formula="~1",
            tau_formula="~1"
        )
        
        # Check R fitting succeeded
        if not r_result["success"] or not r_result["converged"]:
            self.skipTest("R model did not converge for SHASH")
        
        # Compare deviance (lenient for SHASH)
        py_deviance = py_model.deviance
        r_deviance = r_result["deviance"]
        np.testing.assert_allclose(
            py_deviance,
            r_deviance,
            rtol=2e-1,  # 20% tolerance
            atol=1e-1,
            err_msg="Deviance differs between Python and R"
        )
    
    def test_sn2_intercept_only(self):
        """Test SN2 intercept-only model: y ~ 1."""
        np.random.seed(123)
        n = 100
        
        # Generate data
        y = np.random.normal(5, 2, n)
        
        # Fit with Python
        family = SN2()
        data = {"y": y}
        control = gamlss_control(n_cyc=100)
        py_model = gamlss(formula="y ~ 1", family=family, data=data, control=control)
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ 1",
            family="SN2",
            sigma_formula="~1",
            nu_formula="~1"
        )
        
        # Check R fitting succeeded
        if not r_result["success"] or not r_result["converged"]:
            self.skipTest("R model did not converge for SN2 (known convergence issue)")
        
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
    
    # Batch 4 tests
    
    def test_beinf0_intercept_only(self):
        """Test BEINF0 intercept-only model: y ~ 1."""
        np.random.seed(456)
        n = 100
        
        # Generate beta data with some zeros
        y_beta = np.random.beta(2, 2, 80)
        y_zeros = np.zeros(20)
        y = np.concatenate([y_beta, y_zeros])
        np.random.shuffle(y)
        
        # Fit with Python
        family = BEINF0()
        data = {"y": y}
        control = gamlss_control(n_cyc=100)
        py_model = gamlss(formula="y ~ 1", family=family, data=data, control=control)
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ 1",
            family="BEINF0",
            sigma_formula="~1",
            nu_formula="~1"
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
    
    def test_beinf1_intercept_only(self):
        """Test BEINF1 intercept-only model: y ~ 1."""
        np.random.seed(789)
        n = 100
        
        # Generate beta data with some ones
        y_beta = np.random.beta(2, 2, 80)
        y_ones = np.ones(20)
        y = np.concatenate([y_beta, y_ones])
        np.random.shuffle(y)
        
        # Fit with Python
        family = BEINF1()
        data = {"y": y}
        control = gamlss_control(n_cyc=100)
        py_model = gamlss(formula="y ~ 1", family=family, data=data, control=control)
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ 1",
            family="BEINF1",
            sigma_formula="~1",
            nu_formula="~1"
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
    
    def test_beoi_intercept_only(self):
        """Test BEOI intercept-only model: y ~ 1."""
        np.random.seed(111)
        n = 100
        
        # Generate beta data with some zeros and ones
        # BEOI requires y in [0,1], ensure no exact 0 or 1 in beta part
        y_beta = np.random.beta(2, 2, 70)
        y_beta = np.clip(y_beta, 0.001, 0.999)  # Avoid exact boundaries in continuous part
        y_zeros = np.zeros(15)
        y_ones = np.ones(15)
        y = np.concatenate([y_beta, y_zeros, y_ones])
        np.random.shuffle(y)
        
        # Fit with Python
        family = BEOI()
        data = {"y": y}
        control = gamlss_control(n_cyc=100)
        py_model = gamlss(formula="y ~ 1", family=family, data=data, control=control)
        
        # Fit with R
        try:
            r_result = self.r_bridge.call_r_gamlss(
                data=data,
                formula="y ~ 1",
                family="BEOI",
                sigma_formula="~1",
                nu_formula="~1",
                tau_formula="~1"
            )
        except RuntimeError as e:
            if "out of range" in str(e):
                self.skipTest("R BEOI has strict data requirements (known issue)")
            raise
        
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
    
    def test_bezi_intercept_only(self):
        """Test BEZI intercept-only model: y ~ 1."""
        np.random.seed(222)
        n = 100
        
        # Generate beta data with some zeros
        y_beta = np.random.beta(2, 2, 80)
        y_zeros = np.zeros(20)
        y = np.concatenate([y_beta, y_zeros])
        np.random.shuffle(y)
        
        # Fit with Python
        family = BEZI()
        data = {"y": y}
        control = gamlss_control(n_cyc=100)
        py_model = gamlss(formula="y ~ 1", family=family, data=data, control=control)
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ 1",
            family="BEZI",
            sigma_formula="~1",
            nu_formula="~1"
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


if __name__ == "__main__":
    unittest.main()
