"""BEINF (Beta Inflated at 0 and 1) R/Python consistency tests.

Tests OmniLSS BEINF distribution against R gamlss.dist::BEINF.

Prerequisites:
    - R installed
    - gamlss.dist package installed

Run:
    PYTHONPATH=src JAX_PLATFORMS=cpu python -m unittest tests.test_r_consistency_beinf -v
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omnilss.distributions_b4 import BEINF
from omnilss.fitting import gamlss

# Import R bridge
from tests._r_bridge_helper import R_AVAILABLE, R_BRIDGE_CLS as RBridge


@unittest.skipIf(not R_AVAILABLE, "R bridge not available")
class TestBEINFConsistency(unittest.TestCase):
    """BEINF distribution R/Python consistency tests."""
    
    @classmethod
    def setUpClass(cls):
        """Set up R bridge once for all tests."""
        try:
            cls.r_bridge = RBridge()
        except RuntimeError as e:
            raise unittest.SkipTest(f"R not available: {e}")
    
    def test_intercept_only_beinf(self):
        """Test intercept-only BEINF model: y ~ 1."""
        np.random.seed(42)
        n = 100
        
        # Generate beta data with some 0s and 1s
        y_beta = np.random.beta(2, 2, 80)
        y_zeros = np.zeros(10)
        y_ones = np.ones(10)
        y = np.concatenate([y_beta, y_zeros, y_ones])
        np.random.shuffle(y)
        
        # Fit with Python
        family = BEINF()
        data = {"y": y}
        py_model = gamlss(formula="y ~ 1", family=family, data=data)
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ 1",
            family="BEINF",
            sigma_formula="~1",
            nu_formula="~1",
            tau_formula="~1"
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
    
    def test_simple_beinf_model(self):
        """Test simple BEINF model: y ~ x."""
        np.random.seed(123)
        n = 100
        x = np.random.uniform(-1, 1, n)
        
        # Generate data with linear relationship
        # Most data in (0,1), some at boundaries
        y_beta = np.random.beta(2, 2, 80)
        y_zeros = np.zeros(10)
        y_ones = np.ones(10)
        y = np.concatenate([y_beta, y_zeros, y_ones])
        x_full = np.concatenate([x[:80], x[80:90], x[90:]])
        
        # Fit with Python
        family = BEINF()
        data = {"y": y, "x": x_full}
        py_model = gamlss(formula="y ~ x", data=data, family=family)
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ x",
            family="BEINF",
            sigma_formula="~1",
            nu_formula="~1",
            tau_formula="~1"
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
