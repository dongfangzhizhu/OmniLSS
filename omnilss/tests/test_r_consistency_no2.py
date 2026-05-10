"""NO2 (Normal with variance parameterization) R/Python consistency tests.

Tests OmniLSS NO2 distribution against R gamlss.dist::NO2.

Prerequisites:
    - R installed
    - gamlss.dist package installed

Run:
    PYTHONPATH=src JAX_PLATFORMS=cpu python -m unittest tests.test_r_consistency_no2 -v
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omnilss.distributions_b2 import NO2
from omnilss.fitting import gamlss

# Import R bridge
from tests._r_bridge_helper import R_AVAILABLE, R_BRIDGE_CLS as RBridge


@unittest.skipIf(not R_AVAILABLE, "R bridge not available")
class TestNO2Consistency(unittest.TestCase):
    """NO2 distribution R/Python consistency tests."""
    
    @classmethod
    def setUpClass(cls):
        """Set up R bridge once for all tests."""
        try:
            cls.r_bridge = RBridge()
        except RuntimeError as e:
            raise unittest.SkipTest(f"R not available: {e}")
    
    def test_intercept_only_no2(self):
        """Test intercept-only NO2 model: y ~ 1."""
        np.random.seed(42)
        n = 100
        
        # Generate data from NO2 (variance parameterization)
        mu_true = 5.0
        sigma2_true = 4.0  # variance
        y = np.random.normal(mu_true, np.sqrt(sigma2_true), n)
        
        # Fit with Python
        family = NO2()
        data = {"y": y}
        py_model = gamlss(formula="y ~ 1", family=family, data=data)
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ 1",
            family="NO2",
            sigma_formula="~1"
        )
        
        # Check R fitting succeeded
        self.assertTrue(r_result["success"], f"R fitting failed: {r_result.get('error', 'Unknown error')}")
        self.assertTrue(r_result["converged"], "R model did not converge")
        
        # Extract coefficients
        py_mu_coef = py_model.coefficients["mu"]
        py_sigma_coef = py_model.coefficients["sigma"]
        r_mu_coef = np.array(r_result["coefficients"]["mu"])
        r_sigma_coef = np.array(r_result["coefficients"]["sigma"])
        
        # Compare mu coefficients
        np.testing.assert_allclose(
            py_mu_coef,
            r_mu_coef,
            rtol=5e-3,
            atol=1e-5,
            err_msg="Mu coefficients differ between Python and R"
        )
        
        # Compare sigma coefficients
        np.testing.assert_allclose(
            py_sigma_coef,
            r_sigma_coef,
            rtol=5e-3,
            atol=1e-5,
            err_msg="Sigma coefficients differ between Python and R"
        )
        
        # Compare deviance
        py_deviance = py_model.deviance
        r_deviance = r_result["deviance"]
        np.testing.assert_allclose(
            py_deviance,
            r_deviance,
            rtol=1e-3,
            atol=1e-5,
            err_msg="Deviance differs between Python and R"
        )
    
    def test_simple_no2_model(self):
        """Test simple NO2 model: y ~ x."""
        np.random.seed(123)
        n = 100
        x = np.random.uniform(0, 10, n)
        
        # Generate data with linear relationship
        mu_true = 2.0 + 0.5 * x
        sigma2_true = 2.0  # constant variance
        y = np.random.normal(mu_true, np.sqrt(sigma2_true))
        
        # Fit with Python
        family = NO2()
        data = {"y": y, "x": x}
        py_model = gamlss(formula="y ~ x", data=data, family=family)
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ x",
            family="NO2",
            sigma_formula="~1"
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
            rtol=1e-3,
            atol=1e-5,
            err_msg="Mu coefficients differ between Python and R"
        )
        
        # Compare deviance
        py_deviance = py_model.deviance
        r_deviance = r_result["deviance"]
        np.testing.assert_allclose(
            py_deviance,
            r_deviance,
            rtol=1e-3,
            atol=1e-5,
            err_msg="Deviance differs between Python and R"
        )
    
    def test_multiple_predictors_no2(self):
        """Test NO2 model with multiple predictors: y ~ x1 + x2."""
        np.random.seed(456)
        n = 100
        x1 = np.random.uniform(0, 5, n)
        x2 = np.random.uniform(-2, 2, n)
        
        # Generate data
        mu_true = 1.0 + 0.8 * x1 - 0.3 * x2
        sigma2_true = 3.0
        y = np.random.normal(mu_true, np.sqrt(sigma2_true))
        
        # Fit with Python
        family = NO2()
        data = {"y": y, "x1": x1, "x2": x2}
        py_model = gamlss(
            formula="y ~ x1 + x2",
            data=data,
            family=family
        )
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ x1 + x2",
            family="NO2",
            sigma_formula="~1"
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
            rtol=1e-3,
            atol=1e-5,
            err_msg="Mu coefficients differ between Python and R"
        )
        
        # Compare deviance
        py_deviance = py_model.deviance
        r_deviance = r_result["deviance"]
        np.testing.assert_allclose(
            py_deviance,
            r_deviance,
            rtol=1e-3,
            atol=1e-5,
            err_msg="Deviance differs between Python and R"
        )


if __name__ == "__main__":
    unittest.main()
