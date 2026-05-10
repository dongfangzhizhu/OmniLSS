"""IGAMMA R/Python 一致性测试。

测试 OmniLSS 的 IGAMMA 分布与 R gamlss 的数值一致性。

前提条件:
    - 已安装 R
    - 已安装 gamlss.dist 包

运行:
    PYTHONPATH=src JAX_PLATFORMS=cpu python -m unittest tests.test_r_consistency_igamma -v
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omnilss.distributions_b1 import IGAMMA
from omnilss.fitting import gamlss

# Import R bridge
from tests._r_bridge_helper import R_AVAILABLE, R_BRIDGE_CLS as RBridge


@unittest.skipIf(not R_AVAILABLE, "R bridge not available")
class TestIGAMMAConsistency(unittest.TestCase):
    """IGAMMA 一致性测试。"""

    @classmethod
    def setUpClass(cls):
        """设置 R bridge。"""
        try:
            cls.r_bridge = RBridge()
        except RuntimeError as e:
            raise unittest.SkipTest(f"R not available: {e}")

    def test_intercept_only_igamma(self):
        """Test intercept-only IGAMMA model: y ~ 1."""
        np.random.seed(42)
        n = 100
        
        # Generate IGAMMA data using scipy
        # R parameterization: alpha = 1/sigma^2
        # scipy.stats.invgamma uses (a, scale) where a=alpha, scale=mu*(alpha+1)
        from scipy.stats import invgamma
        mu = 2.0
        sigma = 0.5
        alpha = 1.0 / (sigma ** 2)
        scale = mu * (alpha + 1.0)
        y = invgamma.rvs(a=alpha, scale=scale, size=n, random_state=42)
        
        # Prepare data
        data = {"y": y}
        
        # Fit with Python
        py_model = gamlss(
            formula="y ~ 1",
            family=IGAMMA(),
            data=data
        )
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ 1",
            family="IGAMMA",
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
            rtol=1e-3,
            atol=1e-4,
            err_msg="Mu coefficients differ between Python and R"
        )
        
        # Compare coefficients for sigma
        py_sigma_coef = py_model.coefficients["sigma"]
        r_sigma_coef = np.array(r_result["coefficients"]["sigma"])
        
        np.testing.assert_allclose(
            py_sigma_coef,
            r_sigma_coef,
            rtol=1e-3,
            atol=1e-4,
            err_msg="Sigma coefficients differ between Python and R"
        )
        
        # Compare deviance
        py_deviance = py_model.deviance
        r_deviance = r_result["deviance"]
        
        np.testing.assert_allclose(
            py_deviance,
            r_deviance,
            rtol=1e-3,
            atol=1e-2,
            err_msg="Deviance differs between Python and R"
        )

    def test_simple_igamma_model(self):
        """Test simple IGAMMA model: y ~ x."""
        np.random.seed(123)
        n = 100
        
        # Generate data
        from scipy.stats import invgamma
        x = np.random.uniform(0, 1, n)
        mu = np.exp(0.5 + 0.8 * x)
        sigma = 0.5
        alpha = 1.0 / (sigma ** 2)
        scale = mu * (alpha + 1.0)
        y = invgamma.rvs(a=alpha, scale=scale, random_state=123)
        
        # Prepare data
        data = {"y": y, "x": x}
        
        # Fit with Python
        py_model = gamlss(
            formula="y ~ x",
            family=IGAMMA(),
            data=data
        )
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ x",
            family="IGAMMA",
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
            rtol=1e-3,
            atol=1e-4,
            err_msg="Mu coefficients differ between Python and R"
        )
        
        # Compare coefficients for sigma
        py_sigma_coef = py_model.coefficients["sigma"]
        r_sigma_coef = np.array(r_result["coefficients"]["sigma"])
        
        np.testing.assert_allclose(
            py_sigma_coef,
            r_sigma_coef,
            rtol=1e-3,
            atol=1e-4,
            err_msg="Sigma coefficients differ between Python and R"
        )
        
        # Compare deviance
        py_deviance = py_model.deviance
        r_deviance = r_result["deviance"]
        
        np.testing.assert_allclose(
            py_deviance,
            r_deviance,
            rtol=1e-3,
            atol=1e-2,
            err_msg="Deviance differs between Python and R"
        )

    def test_multiple_predictors_igamma(self):
        """Test IGAMMA model with multiple predictors: y ~ x1 + x2."""
        np.random.seed(456)
        n = 100
        
        # Generate data
        from scipy.stats import invgamma
        x1 = np.random.uniform(0, 1, n)
        x2 = np.random.uniform(0, 1, n)
        mu = np.exp(0.5 + 0.6 * x1 - 0.3 * x2)
        sigma = 0.5
        alpha = 1.0 / (sigma ** 2)
        scale = mu * (alpha + 1.0)
        y = invgamma.rvs(a=alpha, scale=scale, random_state=456)
        
        # Prepare data
        data = {"y": y, "x1": x1, "x2": x2}
        
        # Fit with Python
        py_model = gamlss(
            formula="y ~ x1 + x2",
            family=IGAMMA(),
            data=data
        )
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ x1 + x2",
            family="IGAMMA",
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
            rtol=1e-3,
            atol=1e-4,
            err_msg="Mu coefficients differ between Python and R"
        )
        
        # Compare coefficients for sigma
        py_sigma_coef = py_model.coefficients["sigma"]
        r_sigma_coef = np.array(r_result["coefficients"]["sigma"])
        
        np.testing.assert_allclose(
            py_sigma_coef,
            r_sigma_coef,
            rtol=1e-3,
            atol=1e-4,
            err_msg="Sigma coefficients differ between Python and R"
        )
        
        # Compare deviance
        py_deviance = py_model.deviance
        r_deviance = r_result["deviance"]
        
        np.testing.assert_allclose(
            py_deviance,
            r_deviance,
            rtol=1e-3,
            atol=1e-2,
            err_msg="Deviance differs between Python and R"
        )


if __name__ == "__main__":
    unittest.main()
