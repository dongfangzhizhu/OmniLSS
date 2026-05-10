"""R/Python consistency tests for LOGNO (Log-Normal) distribution.

Tests that OmniLSS produces results consistent with R GAMLSS
for the Log-Normal distribution family.
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omnilss.distributions import LOGNO
from omnilss.fitting import gamlss
from omnilss.operations import deviance, ic

# Import R bridge (兼容 discover 和直接运行两种模式)
from tests._r_bridge_helper import R_AVAILABLE, R_BRIDGE_CLS as RBridge


@unittest.skipIf(not R_AVAILABLE, "R bridge not available")
class TestLOGNOConsistency(unittest.TestCase):
    """Test consistency between Python and R for LOGNO distribution."""
    
    @classmethod
    def setUpClass(cls):
        """Set up R bridge once for all tests."""
        try:
            cls.r_bridge = RBridge()
        except RuntimeError as e:
            raise unittest.SkipTest(f"R not available: {e}")
    
    def test_simple_lognormal_model(self):
        """Test simple log-normal model: y ~ x."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate test data
        n = 100
        x = np.random.normal(0, 1, n)
        
        # Generate log-normal data
        # log(y) ~ N(mu, sigma^2)
        # mu = beta0 + beta1 * x
        log_mu = 1.0 + 0.5 * x
        sigma = 0.3
        y = np.exp(log_mu + np.random.normal(0, sigma, n))
        
        # Prepare data dictionary
        data = {
            "y": y,
            "x": x
        }
        
        # Fit with Python
        py_model = gamlss(
            formula="y ~ x",
            family=LOGNO(),
            data=data
        )
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ x",
            family="LOGNO",
            sigma_formula="~1"
        )
        
        # Check that R fitting succeeded
        self.assertTrue(r_result["success"], f"R fitting failed: {r_result.get('error', 'Unknown error')}")
        self.assertTrue(r_result["converged"], "R model did not converge")
        
        # Compare coefficients for mu
        py_mu_coef = py_model.coefficients["mu"]
        r_mu_coef = np.array(r_result["coefficients"]["mu"])
        
        np.testing.assert_allclose(
            py_mu_coef,
            r_mu_coef,
            rtol=5e-5,
            atol=1e-7,
            err_msg="Mu coefficients differ between Python and R"
        )
        
        # Compare coefficients for sigma
        py_sigma_coef = py_model.coefficients["sigma"]
        r_sigma_coef = np.array(r_result["coefficients"]["sigma"])
        
        np.testing.assert_allclose(
            py_sigma_coef,
            r_sigma_coef,
            rtol=1e-1,
            atol=1e-2,
            err_msg="Sigma coefficients differ between Python and R"
        )
        
        # Compare fitted values for mu
        py_mu_fitted = py_model.fitted_values["mu"]
        r_mu_fitted = np.array(r_result["fitted_values"]["mu"])
        
        np.testing.assert_allclose(
            py_mu_fitted,
            r_mu_fitted,
            rtol=5e-5,
            atol=1e-7,
            err_msg="Mu fitted values differ between Python and R"
        )
        
        # Compare deviance
        py_deviance = deviance(py_model)
        r_deviance = r_result["deviance"]
        
        np.testing.assert_allclose(
            py_deviance,
            r_deviance,
            rtol=5e-3,
            atol=1e-3,
            err_msg="Deviance differs between Python and R"
        )
        
        # Compare AIC
        py_aic = ic(py_model, k=2.0)
        r_aic = r_result["aic"]
        
        np.testing.assert_allclose(
            py_aic,
            r_aic,
            rtol=5e-3,
            atol=1e-3,
            err_msg="AIC differs between Python and R"
        )
    
    def test_intercept_only_lognormal(self):
        """Test intercept-only log-normal model: y ~ 1."""
        # Set random seed
        np.random.seed(123)
        
        # Generate test data
        n = 50
        log_mu = 2.0
        sigma = 0.4
        y = np.exp(log_mu + np.random.normal(0, sigma, n))
        
        # Prepare data
        data = {"y": y}
        
        # Fit with Python
        py_model = gamlss(
            formula="y ~ 1",
            family=LOGNO(),
            data=data
        )
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ 1",
            family="LOGNO",
            sigma_formula="~1"
        )
        
        # Check R success
        self.assertTrue(r_result["success"])
        
        # Compare mu coefficient
        py_mu_coef = py_model.coefficients["mu"]
        r_mu_coef = np.array(r_result["coefficients"]["mu"])
        
        np.testing.assert_allclose(
            py_mu_coef,
            r_mu_coef,
            rtol=5e-5,
            atol=1e-7
        )
        
        # Compare sigma coefficient
        py_sigma_coef = py_model.coefficients["sigma"]
        r_sigma_coef = np.array(r_result["coefficients"]["sigma"])
        
        np.testing.assert_allclose(
            py_sigma_coef,
            r_sigma_coef,
            rtol=1e-1,
            atol=1e-2
        )
    
    def test_multiple_predictors_lognormal(self):
        """Test log-normal model with multiple predictors: y ~ x1 + x2."""
        # Set random seed
        np.random.seed(456)
        
        # Generate test data
        n = 80
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        
        # Generate log-normal data
        log_mu = 1.5 + 0.4 * x1 - 0.3 * x2
        sigma = 0.35
        y = np.exp(log_mu + np.random.normal(0, sigma, n))
        
        # Prepare data
        data = {
            "y": y,
            "x1": x1,
            "x2": x2
        }
        
        # Fit with Python
        py_model = gamlss(
            formula="y ~ x1 + x2",
            family=LOGNO(),
            data=data
        )
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ x1 + x2",
            family="LOGNO",
            sigma_formula="~1"
        )
        
        # Check R success
        self.assertTrue(r_result["success"])
        
        # Compare mu coefficients
        py_mu_coef = py_model.coefficients["mu"]
        r_mu_coef = np.array(r_result["coefficients"]["mu"])
        
        np.testing.assert_allclose(
            py_mu_coef,
            r_mu_coef,
            rtol=5e-5,
            atol=1e-7,
            err_msg="Mu coefficients differ for multiple predictors"
        )
        
        # Compare fitted values
        py_mu_fitted = py_model.fitted_values["mu"]
        r_mu_fitted = np.array(r_result["fitted_values"]["mu"])
        
        np.testing.assert_allclose(
            py_mu_fitted,
            r_mu_fitted,
            rtol=5e-5,
            atol=1e-7,
            err_msg="Fitted values differ for multiple predictors"
        )
        
        # Compare deviance
        np.testing.assert_allclose(
            deviance(py_model),
            r_result["deviance"],
            rtol=5e-3,
            atol=1e-3
        )


if __name__ == "__main__":
    unittest.main()
