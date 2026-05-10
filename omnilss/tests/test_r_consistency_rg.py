"""RG (Reverse Gumbel) R/Python consistency tests.

Tests OmniLSS RG distribution against R gamlss.dist::RG.

Prerequisites:
    - R installed
    - gamlss.dist package installed

Run:
    PYTHONPATH=src JAX_PLATFORMS=cpu python -m unittest tests.test_r_consistency_rg -v
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omnilss.distributions_b1 import RG
from omnilss.fitting import gamlss

# Import R bridge
from tests._r_bridge_helper import R_AVAILABLE, R_BRIDGE_CLS as RBridge


@unittest.skipIf(not R_AVAILABLE, "R bridge not available")
class TestRGConsistency(unittest.TestCase):
    """RG distribution R/Python consistency tests."""
    
    @classmethod
    def setUpClass(cls):
        """Set up R bridge once for all tests."""
        try:
            cls.r_bridge = RBridge()
        except RuntimeError as e:
            raise unittest.SkipTest(f"R not available: {e}")
    
    def test_intercept_only_rg(self):
        """Test intercept-only RG model: y ~ 1."""
        np.random.seed(42)
        n = 100
        
        # Generate data from Reverse Gumbel distribution
        # Using scipy's gumbel_l (left-skewed Gumbel)
        from scipy.stats import gumbel_l
        mu_true = 5.0
        sigma_true = 2.0
        y = gumbel_l.rvs(loc=mu_true, scale=sigma_true, size=n)
        
        # Fit with Python
        family = RG()
        data = {"y": y}
        py_model = gamlss(formula="y ~ 1", family=family, data=data)
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ 1",
            family="RG",
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
            rtol=1.5e-2,
            atol=1e-5,
            err_msg="Mu coefficients differ between Python and R"
        )
        
        # Compare sigma coefficients
        np.testing.assert_allclose(
            py_sigma_coef,
            r_sigma_coef,
            rtol=1.5e-2,
            atol=1e-5,
            err_msg="Sigma coefficients differ between Python and R"
        )
        
        # Compare deviance
        py_deviance = py_model.deviance
        r_deviance = r_result["deviance"]
        np.testing.assert_allclose(
            py_deviance,
            r_deviance,
            rtol=1.5e-2,
            atol=1e-5,
            err_msg="Deviance differs between Python and R"
        )
    
    def test_simple_rg_model(self):
        """Test simple RG model: y ~ x."""
        np.random.seed(123)
        n = 100
        x = np.random.uniform(0, 5, n)
        
        # Generate data with linear relationship
        from scipy.stats import gumbel_l
        mu_true = 2.0 + 0.5 * x
        sigma_true = 1.5
        y = gumbel_l.rvs(loc=mu_true, scale=sigma_true)
        
        # Fit with Python
        family = RG()
        data = {"y": y, "x": x}
        py_model = gamlss(formula="y ~ x", data=data, family=family)
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ x",
            family="RG",
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
            rtol=1.5e-2,
            atol=1e-5,
            err_msg="Mu coefficients differ between Python and R"
        )
        
        # Compare deviance
        py_deviance = py_model.deviance
        r_deviance = r_result["deviance"]
        np.testing.assert_allclose(
            py_deviance,
            r_deviance,
            rtol=1.5e-2,
            atol=1e-5,
            err_msg="Deviance differs between Python and R"
        )
    
    def test_multiple_predictors_rg(self):
        """Test RG model with multiple predictors: y ~ x1 + x2."""
        np.random.seed(456)
        n = 100
        x1 = np.random.uniform(0, 3, n)
        x2 = np.random.uniform(-1, 1, n)
        
        # Generate data
        from scipy.stats import gumbel_l
        mu_true = 3.0 + 0.6 * x1 - 0.4 * x2
        sigma_true = 1.0
        y = gumbel_l.rvs(loc=mu_true, scale=sigma_true)
        
        # Fit with Python
        family = RG()
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
            family="RG",
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
            rtol=1.5e-2,
            atol=1e-5,
            err_msg="Mu coefficients differ between Python and R"
        )
        
        # Compare deviance
        py_deviance = py_model.deviance
        r_deviance = r_result["deviance"]
        np.testing.assert_allclose(
            py_deviance,
            r_deviance,
            rtol=1.5e-2,
            atol=1e-5,
            err_msg="Deviance differs between Python and R"
        )


if __name__ == "__main__":
    unittest.main()
