"""R/Python consistency tests for IG (Inverse Gaussian) distribution.

Tests that OmniLSS produces results consistent with R GAMLSS
for the Inverse Gaussian distribution family.

运行方式:
    python -m unittest tests.test_r_consistency_ig
    或
    PYTHONPATH=src JAX_PLATFORMS=cpu python -m unittest tests.test_r_consistency_ig -v
"""

import unittest
import numpy as np
import jax.numpy as jnp
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omnilss.distributions import IG
from omnilss.fitting import gamlss
from omnilss.operations import deviance, ic

# Import R bridge
from tests._r_bridge_helper import R_AVAILABLE, R_BRIDGE_CLS as RBridge


@unittest.skipIf(not R_AVAILABLE, "R bridge not available")
class TestIGConsistency(unittest.TestCase):
    """Test consistency between Python and R for IG distribution."""
    
    @classmethod
    def setUpClass(cls):
        """Set up R bridge once for all tests."""
        try:
            cls.r_bridge = RBridge()
        except RuntimeError as e:
            raise unittest.SkipTest(f"R not available: {e}")
    
    def test_intercept_only(self):
        """截距模型 y ~ 1。"""
        # Set random seed
        np.random.seed(42)
        
        # Generate IG data
        # Using scipy's invgauss
        from scipy.stats import invgauss
        n = 100
        mu = 2.0
        scale = 0.5  # This is 1/lambda in scipy parameterization
        y = invgauss.rvs(mu/scale, scale=scale, size=n)
        
        # Prepare data
        data = {"y": y}
        
        # Fit with Python
        py_model = gamlss(
            formula="y ~ 1",
            family=IG(),
            data=data
        )
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ 1",
            family="IG",
            sigma_formula="~1"
        )
        
        # Check R success
        self.assertTrue(r_result["success"], f"R fitting failed: {r_result.get('error', 'Unknown error')}")
        self.assertTrue(r_result["converged"], "R model did not converge")
        
        # Compare mu coefficient
        py_mu_coef = py_model.coefficients["mu"]
        r_mu_coef = np.array(r_result["coefficients"]["mu"])
        
        np.testing.assert_allclose(
            py_mu_coef,
            r_mu_coef,
            rtol=1e-3,
            atol=1e-4,
            err_msg="Mu coefficients differ between Python and R"
        )
        
        # Compare sigma coefficient
        py_sigma_coef = py_model.coefficients["sigma"]
        r_sigma_coef = np.array(r_result["coefficients"]["sigma"])
        
        np.testing.assert_allclose(
            py_sigma_coef,
            r_sigma_coef,
            rtol=0.05,
            atol=0.05,
            err_msg="Sigma coefficients differ between Python and R"
        )
        
        # Compare deviance
        py_deviance = deviance(py_model)
        r_deviance = r_result["deviance"]
        
        np.testing.assert_allclose(
            py_deviance,
            r_deviance,
            rtol=0.01,
            atol=1.0,
            err_msg="Deviance differs between Python and R"
        )
    
    def test_simple_covariate(self):
        """单协变量 y ~ x。"""
        # Set random seed
        np.random.seed(123)
        
        # Generate data with covariate effect
        from scipy.stats import invgauss
        n = 100
        x = np.random.normal(0, 1, n)
        # log(mu) = 0.5 + 0.3 * x
        log_mu = 0.5 + 0.3 * x
        mu = np.exp(log_mu)
        
        # Generate IG data
        scale = 0.4
        y = np.array([invgauss.rvs(mu_i/scale, scale=scale) for mu_i in mu])
        
        # Prepare data
        data = {
            "y": y,
            "x": x
        }
        
        # Fit with Python
        py_model = gamlss(
            formula="y ~ x",
            family=IG(),
            data=data
        )
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ x",
            family="IG",
            sigma_formula="~1"
        )
        
        # Check R success
        self.assertTrue(r_result["success"])
        self.assertTrue(r_result["converged"])
        
        # Compare mu coefficients
        py_mu_coef = py_model.coefficients["mu"]
        r_mu_coef = np.array(r_result["coefficients"]["mu"])
        
        np.testing.assert_allclose(
            py_mu_coef,
            r_mu_coef,
            rtol=0.02,
            atol=0.01,
            err_msg="Mu coefficients differ between Python and R"
        )
        
        # Compare fitted values
        py_mu_fitted = py_model.fitted_values["mu"]
        r_mu_fitted = np.array(r_result["fitted_values"]["mu"])
        
        np.testing.assert_allclose(
            py_mu_fitted,
            r_mu_fitted,
            rtol=0.02,
            atol=0.01,
            err_msg="Fitted values differ between Python and R"
        )
        
        # Compare deviance
        np.testing.assert_allclose(
            deviance(py_model),
            r_result["deviance"],
            rtol=0.01,
            atol=2.0
        )
    
    def test_multiple_covariates(self):
        """多协变量 y ~ x1 + x2。"""
        # Set random seed
        np.random.seed(456)
        
        # Generate data
        from scipy.stats import invgauss
        n = 120
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        # log(mu) = 0.3 + 0.4 * x1 - 0.2 * x2
        log_mu = 0.3 + 0.4 * x1 - 0.2 * x2
        mu = np.exp(log_mu)
        
        # Generate IG data
        scale = 0.5
        y = np.array([invgauss.rvs(mu_i/scale, scale=scale) for mu_i in mu])
        
        # Prepare data
        data = {
            "y": y,
            "x1": x1,
            "x2": x2
        }
        
        # Fit with Python
        py_model = gamlss(
            formula="y ~ x1 + x2",
            family=IG(),
            data=data
        )
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ x1 + x2",
            family="IG",
            sigma_formula="~1"
        )
        
        # Check R success
        self.assertTrue(r_result["success"])
        self.assertTrue(r_result["converged"])
        
        # Compare mu coefficients
        py_mu_coef = py_model.coefficients["mu"]
        r_mu_coef = np.array(r_result["coefficients"]["mu"])
        
        np.testing.assert_allclose(
            py_mu_coef,
            r_mu_coef,
            rtol=0.02,
            atol=0.01,
            err_msg="Mu coefficients differ for multiple predictors"
        )
        
        # Compare fitted values
        py_mu_fitted = py_model.fitted_values["mu"]
        r_mu_fitted = np.array(r_result["fitted_values"]["mu"])
        
        np.testing.assert_allclose(
            py_mu_fitted,
            r_mu_fitted,
            rtol=0.02,
            atol=0.01,
            err_msg="Fitted values differ for multiple predictors"
        )
        
        # Compare deviance
        np.testing.assert_allclose(
            deviance(py_model),
            r_result["deviance"],
            rtol=0.01,
            atol=2.0
        )
        
        # Compare AIC
        np.testing.assert_allclose(
            ic(py_model, k=2.0),
            r_result["aic"],
            rtol=0.01,
            atol=2.0
        )


if __name__ == "__main__":
    unittest.main()
