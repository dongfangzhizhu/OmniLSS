"""ZAGA (Zero-Altered Gamma) R/Python consistency tests.

Tests OmniLSS ZAGA distribution against R gamlss.dist::ZAGA.

Prerequisites:
    - R installed
    - gamlss.dist package installed

Run:
    PYTHONPATH=src JAX_PLATFORMS=cpu python -m unittest tests.test_r_consistency_zaga -v
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omnilss.distributions_b5 import ZAGA
from omnilss.fitting import gamlss

# Import R bridge
from tests._r_bridge_helper import R_AVAILABLE, R_BRIDGE_CLS as RBridge


@unittest.skipIf(not R_AVAILABLE, "R bridge not available")
class TestZAGAConsistency(unittest.TestCase):
    """ZAGA distribution R/Python consistency tests."""
    
    @classmethod
    def setUpClass(cls):
        """Set up R bridge once for all tests."""
        try:
            cls.r_bridge = RBridge()
        except RuntimeError as e:
            raise unittest.SkipTest(f"R not available: {e}")
    
    def test_intercept_only_zaga(self):
        """Test intercept-only ZAGA model: y ~ 1."""
        np.random.seed(42)
        n = 100
        
        # Generate gamma data with some zeros
        y_gamma = np.random.gamma(2, 2, 80)
        y_zeros = np.zeros(20)
        y = np.concatenate([y_gamma, y_zeros])
        np.random.shuffle(y)
        
        # Fit with Python (use more cycles for zero-altered distributions)
        family = ZAGA()
        data = {"y": y}
        from omnilss.controls import gamlss_control
        control = gamlss_control(n_cyc=100)
        py_model = gamlss(formula="y ~ 1", family=family, data=data, control=control)
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ 1",
            family="ZAGA",
            sigma_formula="~1",
            nu_formula="~1"
        )
        
        # Check R fitting succeeded
        self.assertTrue(r_result["success"], f"R fitting failed: {r_result.get('error', 'Unknown error')}")
        self.assertTrue(r_result["converged"], "R model did not converge")
        
        # Extract coefficients
        py_mu_coef = py_model.coefficients["mu"]
        r_mu_coef = np.array(r_result["coefficients"]["mu"])
        
        # Compare mu coefficients (lenient for mixed distributions)
        np.testing.assert_allclose(
            py_mu_coef,
            r_mu_coef,
            rtol=1e-1,
            atol=1e-2,
            err_msg="Mu coefficients differ between Python and R"
        )
        
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
    
    def test_simple_zaga_model(self):
        """Test simple ZAGA model: y ~ x."""
        np.random.seed(123)
        n = 100
        x = np.random.uniform(-1, 1, n)
        
        # Generate data with linear relationship
        # Most data positive, some zeros
        y_gamma = np.random.gamma(2, 2, 80)
        y_zeros = np.zeros(20)
        y = np.concatenate([y_gamma, y_zeros])
        x_full = np.concatenate([x[:80], x[80:]])
        
        # Fit with Python (use more cycles for zero-altered distributions)
        family = ZAGA()
        data = {"y": y, "x": x_full}
        from omnilss.controls import gamlss_control
        control = gamlss_control(n_cyc=100)
        py_model = gamlss(formula="y ~ x", data=data, family=family, control=control)
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ x",
            family="ZAGA",
            sigma_formula="~1",
            nu_formula="~1"
        )
        
        # Check R fitting succeeded
        self.assertTrue(r_result["success"], f"R fitting failed: {r_result.get('error', 'Unknown error')}")
        self.assertTrue(r_result["converged"], "R model did not converge")
        
        # Extract coefficients
        py_mu_coef = py_model.coefficients["mu"]
        r_mu_coef = np.array(r_result["coefficients"]["mu"])
        
        # Compare mu coefficients
        np.testing.assert_allclose(
            py_mu_coef,
            r_mu_coef,
            rtol=1e-1,
            atol=1e-2,
            err_msg="Mu coefficients differ between Python and R"
        )
        
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
