"""R/Python consistency tests for NBI (Negative Binomial) distribution.

Tests that OmniLSS produces results consistent with R GAMLSS
for the Negative Binomial distribution family.

运行方式:
    python -m unittest tests.test_r_consistency_nbi
    或
    PYTHONPATH=src JAX_PLATFORMS=cpu python -m unittest tests.test_r_consistency_nbi -v
"""

import unittest
import numpy as np
import jax.numpy as jnp
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omnilss.distributions import NBI
from omnilss.fitting import gamlss
from omnilss.operations import deviance, ic

# Import R bridge
from tests._r_bridge_helper import R_AVAILABLE, R_BRIDGE_CLS as RBridge


@unittest.skipIf(not R_AVAILABLE, "R bridge not available")
class TestNBIConsistency(unittest.TestCase):
    """Test consistency between Python and R for NBI distribution."""
    
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
        
        # Generate negative binomial data
        # mu = 5, size = 2 (sigma = 0.5)
        n = 100
        mu = 5.0
        size = 2.0
        # Generate using numpy's negative_binomial
        # Note: numpy uses (n, p) parameterization
        # p = size / (size + mu)
        p = size / (size + mu)
        y = np.random.negative_binomial(size, p, n).astype(float)
        
        # Prepare data
        data = {"y": y}
        
        # Fit with Python
        py_model = gamlss(
            formula="y ~ 1",
            family=NBI(),
            data=data
        )
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ 1",
            family="NBI",
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
        
        # Compare sigma coefficient (more relaxed tolerance for sigma)
        py_sigma_coef = py_model.coefficients["sigma"]
        r_sigma_coef = np.array(r_result["coefficients"]["sigma"])
        
        np.testing.assert_allclose(
            py_sigma_coef,
            r_sigma_coef,
            rtol=0.25,
            atol=0.2,
            err_msg="Sigma coefficients differ between Python and R"
        )
        
        # Compare deviance (relaxed tolerance for NBI)
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
        n = 100
        x = np.random.normal(0, 1, n)
        # log(mu) = 1.5 + 0.5 * x
        log_mu = 1.5 + 0.5 * x
        mu = np.exp(log_mu)
        
        # Generate NB data
        size = 3.0
        p = size / (size + mu)
        y = np.array([np.random.negative_binomial(size, p_i) for p_i in p]).astype(float)
        
        # Prepare data
        data = {
            "y": y,
            "x": x
        }
        
        # Fit with Python
        py_model = gamlss(
            formula="y ~ x",
            family=NBI(),
            data=data
        )
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ x",
            family="NBI",
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
        
        # Compare deviance (relaxed tolerance)
        np.testing.assert_allclose(
            deviance(py_model),
            r_result["deviance"],
            rtol=0.01,
            atol=3.0
        )
    
    def test_multiple_covariates(self):
        """多协变量 y ~ x1 + x2。"""
        # Set random seed
        np.random.seed(456)
        
        # Generate data
        n = 120
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        # log(mu) = 1.0 + 0.6 * x1 - 0.3 * x2
        log_mu = 1.0 + 0.6 * x1 - 0.3 * x2
        mu = np.exp(log_mu)
        
        # Generate NB data
        size = 2.5
        p = size / (size + mu)
        y = np.array([np.random.negative_binomial(size, p_i) for p_i in p]).astype(float)
        
        # Prepare data
        data = {
            "y": y,
            "x1": x1,
            "x2": x2
        }
        
        # Fit with Python
        py_model = gamlss(
            formula="y ~ x1 + x2",
            family=NBI(),
            data=data
        )
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ x1 + x2",
            family="NBI",
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
        
        # Compare deviance (relaxed tolerance)
        np.testing.assert_allclose(
            deviance(py_model),
            r_result["deviance"],
            rtol=0.01,
            atol=3.0
        )
        
        # Compare AIC (relaxed tolerance)
        np.testing.assert_allclose(
            ic(py_model, k=2.0),
            r_result["aic"],
            rtol=0.01,
            atol=3.0
        )


if __name__ == "__main__":
    unittest.main()
