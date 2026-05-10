"""R/Python consistency tests for EXP (Exponential) distribution.

Tests that OmniLSS produces results consistent with R GAMLSS
for the Exponential distribution family.

运行方式:
    python -m unittest tests.test_r_consistency_exp
    或
    PYTHONPATH=src JAX_PLATFORMS=cpu python -m unittest tests.test_r_consistency_exp -v
"""

import unittest
import numpy as np
import jax.numpy as jnp
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omnilss.distributions import EXP
from omnilss.fitting import gamlss
from omnilss.operations import deviance, ic

# Import R bridge
from tests._r_bridge_helper import R_AVAILABLE, R_BRIDGE_CLS as RBridge


@unittest.skipIf(not R_AVAILABLE, "R bridge not available")
class TestEXPConsistency(unittest.TestCase):
    """Test consistency between Python and R for EXP distribution."""
    
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
        
        # Generate exponential data
        n = 100
        rate = 0.5  # rate = 1/mu, so mu = 2.0
        y = np.random.exponential(scale=1.0/rate, size=n)
        
        # Prepare data
        data = {"y": y}
        
        # Fit with Python
        py_model = gamlss(
            formula="y ~ 1",
            family=EXP(),
            data=data
        )
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ 1",
            family="EXP",
            sigma_formula=None  # EXP is single-parameter
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
            rtol=1e-4,
            atol=1e-5,
            err_msg="Mu coefficients differ between Python and R"
        )
        
        # Compare fitted values
        py_mu_fitted = py_model.fitted_values["mu"]
        r_mu_fitted = np.array(r_result["fitted_values"]["mu"])
        
        np.testing.assert_allclose(
            py_mu_fitted,
            r_mu_fitted,
            rtol=1e-4,
            atol=1e-5,
            err_msg="Fitted values differ between Python and R"
        )
        
        # Compare deviance
        py_deviance = deviance(py_model)
        r_deviance = r_result["deviance"]
        
        np.testing.assert_allclose(
            py_deviance,
            r_deviance,
            rtol=1e-3,
            atol=1e-4,
            err_msg="Deviance differs between Python and R"
        )
    
    def test_simple_covariate(self):
        """单协变量 y ~ x。"""
        # Set random seed
        np.random.seed(123)
        
        # Generate data with covariate effect
        n = 100
        x = np.random.normal(0, 1, n)
        # log(mu) = 0.5 + 0.3 * x
        log_mu = 0.5 + 0.3 * x
        mu = np.exp(log_mu)
        
        # Generate exponential data
        y = np.array([np.random.exponential(scale=mu_i) for mu_i in mu])
        
        # Prepare data
        data = {
            "y": y,
            "x": x
        }
        
        # Fit with Python
        py_model = gamlss(
            formula="y ~ x",
            family=EXP(),
            data=data
        )
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ x",
            family="EXP",
            sigma_formula=None
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
            rtol=5e-4,
            atol=1e-4,
            err_msg="Mu coefficients differ between Python and R"
        )
        
        # Compare fitted values
        py_mu_fitted = py_model.fitted_values["mu"]
        r_mu_fitted = np.array(r_result["fitted_values"]["mu"])
        
        np.testing.assert_allclose(
            py_mu_fitted,
            r_mu_fitted,
            rtol=5e-4,
            atol=1e-4,
            err_msg="Fitted values differ between Python and R"
        )
        
        # Compare deviance
        np.testing.assert_allclose(
            deviance(py_model),
            r_result["deviance"],
            rtol=1e-3,
            atol=1e-4
        )
    
    def test_multiple_covariates(self):
        """多协变量 y ~ x1 + x2。"""
        # Set random seed
        np.random.seed(456)
        
        # Generate data
        n = 120
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        # log(mu) = 0.3 + 0.4 * x1 - 0.2 * x2
        log_mu = 0.3 + 0.4 * x1 - 0.2 * x2
        mu = np.exp(log_mu)
        
        # Generate exponential data
        y = np.array([np.random.exponential(scale=mu_i) for mu_i in mu])
        
        # Prepare data
        data = {
            "y": y,
            "x1": x1,
            "x2": x2
        }
        
        # Fit with Python
        py_model = gamlss(
            formula="y ~ x1 + x2",
            family=EXP(),
            data=data
        )
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ x1 + x2",
            family="EXP",
            sigma_formula=None
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
            rtol=5e-4,
            atol=1e-4,
            err_msg="Mu coefficients differ for multiple predictors"
        )
        
        # Compare fitted values
        py_mu_fitted = py_model.fitted_values["mu"]
        r_mu_fitted = np.array(r_result["fitted_values"]["mu"])
        
        np.testing.assert_allclose(
            py_mu_fitted,
            r_mu_fitted,
            rtol=1e-4,
            atol=1e-5,
            err_msg="Fitted values differ for multiple predictors"
        )
        
        # Compare deviance
        np.testing.assert_allclose(
            deviance(py_model),
            r_result["deviance"],
            rtol=1e-3,
            atol=1e-4
        )
        
        # Compare AIC
        np.testing.assert_allclose(
            ic(py_model, k=2.0),
            r_result["aic"],
            rtol=1e-3,
            atol=1e-4
        )


if __name__ == "__main__":
    unittest.main()
