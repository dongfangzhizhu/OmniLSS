"""SIMPLEX R/Python 一致性测试。

测试 OmniLSS 的 SIMPLEX 分布与 R gamlss 的数值一致性。

前提条件:
    - 已安装 R
    - 已安装 gamlss.dist 包

运行:
    PYTHONPATH=src JAX_PLATFORMS=cpu python -m unittest tests.test_r_consistency_simplex -v
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omnilss.distributions_b2 import SIMPLEX
from omnilss.fitting import gamlss

# Import R bridge
from tests._r_bridge_helper import R_AVAILABLE, R_BRIDGE_CLS as RBridge


@unittest.skipIf(not R_AVAILABLE, "R bridge not available")
class TestSIMPLEXConsistency(unittest.TestCase):
    """SIMPLEX 一致性测试。"""

    @classmethod
    def setUpClass(cls):
        """设置 R bridge。"""
        try:
            cls.r_bridge = RBridge()
        except RuntimeError as e:
            raise unittest.SkipTest(f"R not available: {e}")

    def test_intercept_only_simplex(self):
        """Test intercept-only SIMPLEX model: y ~ 1."""
        np.random.seed(42)
        n = 100
        
        # Generate SIMPLEX data
        # Use beta distribution as approximation (SIMPLEX is similar to beta)
        # For SIMPLEX: E[Y] = mu, Var[Y] ≈ mu(1-mu)*sigma^2
        mu = 0.5
        sigma = 0.8
        # Generate from beta with appropriate parameters
        alpha = mu / (sigma**2)
        beta_param = (1 - mu) / (sigma**2)
        y = np.random.beta(alpha, beta_param, n)
        # Clip to avoid boundary issues
        eps = 1e-6
        y = np.clip(y, eps, 1 - eps)
        
        # Prepare data
        data = {"y": y}
        
        # Fit with Python
        py_model = gamlss(
            formula="y ~ 1",
            family=SIMPLEX(),
            data=data
        )
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ 1",
            family="SIMPLEX",
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
            rtol=5e-3,
            atol=1e-3,
            err_msg="Mu coefficients differ between Python and R"
        )
        
        # Compare coefficients for sigma
        py_sigma_coef = py_model.coefficients["sigma"]
        r_sigma_coef = np.array(r_result["coefficients"]["sigma"])
        
        np.testing.assert_allclose(
            py_sigma_coef,
            r_sigma_coef,
            rtol=5e-3,
            atol=1e-3,
            err_msg="Sigma coefficients differ between Python and R"
        )
        
        # Compare deviance
        py_deviance = py_model.deviance
        r_deviance = r_result["deviance"]
        
        np.testing.assert_allclose(
            py_deviance,
            r_deviance,
            rtol=5e-3,
            atol=1e-2,
            err_msg="Deviance differs between Python and R"
        )

    def test_simple_simplex_model(self):
        """Test simple SIMPLEX model: y ~ x."""
        np.random.seed(123)
        n = 100
        
        # Generate data
        x = np.random.uniform(-1, 1, n)
        # Use a milder linear predictor and smaller sigma so both
        # Python and R converge reliably on generated test data.
        from scipy.special import expit
        mu = expit(0.0 + 0.4 * x)
        sigma = 0.15
        
        # Generate from beta distribution
        alpha = mu / (sigma**2)
        beta_param = (1 - mu) / (sigma**2)
        y = np.array([np.random.beta(a, b) for a, b in zip(alpha, beta_param)])
        
        # Clip to avoid boundary issues
        eps = 1e-6
        y = np.clip(y, eps, 1 - eps)
        
        # Prepare data
        data = {"y": y, "x": x}
        
        # Fit with Python
        py_model = gamlss(
            formula="y ~ x",
            family=SIMPLEX(),
            data=data
        )
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ x",
            family="SIMPLEX",
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
            rtol=5e-3,
            atol=1e-3,
            err_msg="Mu coefficients differ between Python and R"
        )
        
        # Compare coefficients for sigma
        py_sigma_coef = py_model.coefficients["sigma"]
        r_sigma_coef = np.array(r_result["coefficients"]["sigma"])
        
        np.testing.assert_allclose(
            py_sigma_coef,
            r_sigma_coef,
            rtol=5e-3,
            atol=1e-3,
            err_msg="Sigma coefficients differ between Python and R"
        )
        
        # Compare deviance
        py_deviance = py_model.deviance
        r_deviance = r_result["deviance"]
        
        np.testing.assert_allclose(
            py_deviance,
            r_deviance,
            rtol=5e-3,
            atol=1e-2,
            err_msg="Deviance differs between Python and R"
        )

    def test_multiple_predictors_simplex(self):
        """Test SIMPLEX model with multiple predictors: y ~ x1 + x2."""
        np.random.seed(456)
        n = 100
        
        # Generate data
        x1 = np.random.uniform(-1, 1, n)
        x2 = np.random.uniform(-1, 1, n)
        # Choose a moderate signal and smaller sigma so R converges
        # on the synthetic data as well.
        from scipy.special import expit
        mu = expit(0.0 + 0.3 * x1 - 0.2 * x2)
        sigma = 0.12
        
        # Generate from beta distribution
        alpha = mu / (sigma**2)
        beta_param = (1 - mu) / (sigma**2)
        y = np.array([np.random.beta(a, b) for a, b in zip(alpha, beta_param)])
        
        # Clip to avoid boundary issues
        eps = 1e-6
        y = np.clip(y, eps, 1 - eps)
        
        # Prepare data
        data = {"y": y, "x1": x1, "x2": x2}
        
        # Fit with Python
        py_model = gamlss(
            formula="y ~ x1 + x2",
            family=SIMPLEX(),
            data=data
        )
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ x1 + x2",
            family="SIMPLEX",
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
            rtol=5e-3,
            atol=1e-3,
            err_msg="Mu coefficients differ between Python and R"
        )
        
        # Compare coefficients for sigma
        py_sigma_coef = py_model.coefficients["sigma"]
        r_sigma_coef = np.array(r_result["coefficients"]["sigma"])
        
        np.testing.assert_allclose(
            py_sigma_coef,
            r_sigma_coef,
            rtol=5e-3,
            atol=1e-3,
            err_msg="Sigma coefficients differ between Python and R"
        )
        
        # Compare deviance
        py_deviance = py_model.deviance
        r_deviance = r_result["deviance"]
        
        np.testing.assert_allclose(
            py_deviance,
            r_deviance,
            rtol=5e-3,
            atol=1e-2,
            err_msg="Deviance differs between Python and R"
        )


if __name__ == "__main__":
    unittest.main()
