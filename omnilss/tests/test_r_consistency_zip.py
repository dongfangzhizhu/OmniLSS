"""R/Python consistency tests for ZIP (Zero-Inflated Poisson) distribution.

Tests that OmniLSS produces results consistent with R GAMLSS
for the Zero-Inflated Poisson distribution family (2 parameters: mu, sigma).
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omnilss.distributions import ZIP
from omnilss.fitting import gamlss
from omnilss.operations import deviance, ic

# Import R bridge
from tests._r_bridge_helper import R_AVAILABLE, R_BRIDGE_CLS as RBridge


@unittest.skipIf(not R_AVAILABLE, "R bridge not available")
class TestZIPConsistency(unittest.TestCase):
    """Test consistency between Python and R for ZIP distribution."""
    
    @classmethod
    def setUpClass(cls):
        """Set up R bridge once for all tests."""
        try:
            cls.r_bridge = RBridge()
        except RuntimeError as e:
            raise unittest.SkipTest(f"R not available: {e}")
    
    def test_intercept_only_zip(self):
        """Test intercept-only ZIP model: y ~ 1."""
        # Set random seed
        np.random.seed(42)
        
        # Generate test data
        # ZIP: P(Y=0) = sigma + (1-sigma)*exp(-mu)
        #      P(Y=k) = (1-sigma)*exp(-mu)*mu^k/k! for k > 0
        n = 100
        mu = 2.0
        sigma = 0.3  # Extra-zero probability
        
        # Generate ZIP data
        y = []
        for _ in range(n):
            if np.random.rand() < sigma:
                y.append(0)  # Extra zero
            else:
                y.append(np.random.poisson(mu))  # Poisson part
        y = np.array(y, dtype=float)
        
        # Prepare data
        data = {"y": y}
        
        # Fit with Python
        py_model = gamlss(
            formula="y ~ 1",
            family=ZIP(),
            data=data,
            sigma_formula="~1"
        )
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ 1",
            family="ZIP",
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
            rtol=0.15,
            atol=0.01,
            err_msg="Mu coefficients differ between Python and R"
        )
        
        # Compare sigma coefficient
        py_sigma_coef = py_model.coefficients["sigma"]
        r_sigma_coef = np.array(r_result["coefficients"]["sigma"])
        
        np.testing.assert_allclose(
            py_sigma_coef,
            r_sigma_coef,
            rtol=0.15,
            atol=0.01,
            err_msg="Sigma coefficients differ between Python and R"
        )
        
        # Compare deviance
        py_deviance = deviance(py_model)
        r_deviance = r_result["deviance"]
        
        np.testing.assert_allclose(
            py_deviance,
            r_deviance,
            rtol=0.15,
            atol=1e-3,
            err_msg="Deviance differs between Python and R"
        )
    
    def test_simple_zip_model(self):
        """Test simple ZIP model: y ~ x."""
        # Set random seed for reproducibility
        np.random.seed(123)
        
        # Generate test data
        n = 100
        x = np.random.normal(0, 1, n)
        
        # Generate ZIP data
        # log(mu) = 0.5 + 0.3 * x
        # logit(sigma) = -1.0 (constant)
        mu = np.exp(0.5 + 0.3 * x)
        sigma = 1 / (1 + np.exp(1.0))  # logit^{-1}(-1)
        
        y = []
        for i in range(n):
            if np.random.rand() < sigma:
                y.append(0)
            else:
                y.append(np.random.poisson(mu[i]))
        y = np.array(y, dtype=float)
        
        # Prepare data dictionary
        data = {
            "y": y,
            "x": x
        }
        
        # Fit with Python
        py_model = gamlss(
            formula="y ~ x",
            family=ZIP(),
            data=data,
            sigma_formula="~1"
        )
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ x",
            family="ZIP",
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
            rtol=0.15,
            atol=0.01,
            err_msg="Mu coefficients differ between Python and R"
        )
        
        # Compare coefficients for sigma
        py_sigma_coef = py_model.coefficients["sigma"]
        r_sigma_coef = np.array(r_result["coefficients"]["sigma"])
        
        np.testing.assert_allclose(
            py_sigma_coef,
            r_sigma_coef,
            rtol=0.15,
            atol=0.01,
            err_msg="Sigma coefficients differ between Python and R"
        )
        
        # Compare deviance
        py_deviance = deviance(py_model)
        r_deviance = r_result["deviance"]
        
        np.testing.assert_allclose(
            py_deviance,
            r_deviance,
            rtol=0.15,
            atol=1e-3,
            err_msg="Deviance differs between Python and R"
        )
    
    def test_multiple_predictors_zip(self):
        """Test ZIP model with multiple predictors: y ~ x1 + x2."""
        # Set random seed
        np.random.seed(456)
        
        # Generate test data
        # Use a moderately larger sample so that the ZIP likelihood surface is
        # sufficiently sharp and both optimizers settle on the same solution.
        n = 150
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        
        # Generate ZIP data
        # log(mu) = 0.3 + 0.4 * x1 - 0.2 * x2
        # logit(sigma) = -1.5 (constant)
        mu = np.exp(0.3 + 0.4 * x1 - 0.2 * x2)
        sigma = 1 / (1 + np.exp(1.5))
        
        y = []
        for i in range(n):
            if np.random.rand() < sigma:
                y.append(0)
            else:
                y.append(np.random.poisson(mu[i]))
        y = np.array(y, dtype=float)
        
        # Prepare data
        data = {
            "y": y,
            "x1": x1,
            "x2": x2
        }
        
        # Fit with Python
        py_model = gamlss(
            formula="y ~ x1 + x2",
            family=ZIP(),
            data=data,
            sigma_formula="~1"
        )
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ x1 + x2",
            family="ZIP",
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
            rtol=0.15,
            atol=0.01,
            err_msg="Mu coefficients differ for multiple predictors"
        )
        
        # Compare sigma coefficients
        py_sigma_coef = py_model.coefficients["sigma"]
        r_sigma_coef = np.array(r_result["coefficients"]["sigma"])
        
        np.testing.assert_allclose(
            py_sigma_coef,
            r_sigma_coef,
            rtol=0.15,
            atol=0.01,
            err_msg="Sigma coefficients differ for multiple predictors"
        )
        
        # Compare deviance
        np.testing.assert_allclose(
            deviance(py_model),
            r_result["deviance"],
            rtol=0.15,
            atol=1e-3
        )


if __name__ == "__main__":
    unittest.main()
