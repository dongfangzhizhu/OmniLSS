"""
R/Python consistency tests for TF (Student-t) distribution.

Tests that OmniLSS produces results consistent with R gamlss package
for the TF family (3 parameters: mu, sigma, nu).
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omnilss.distributions import TF
from omnilss.fitting import gamlss
from omnilss.operations import deviance, ic

# Import R bridge (兼容 discover 和直接运行两种模式)
from tests._r_bridge_helper import R_AVAILABLE, R_BRIDGE_CLS as RBridge


@unittest.skipIf(not R_AVAILABLE, "R bridge not available")
class TestTFConsistency(unittest.TestCase):
    """Test consistency between Python and R for TF distribution."""
    
    @classmethod
    def setUpClass(cls):
        """Set up R bridge once for all tests."""
        try:
            cls.r_bridge = RBridge()
        except RuntimeError as e:
            raise unittest.SkipTest(f"R not available: {e}")

    
    def test_tf_simple_intercept(self):
        """Test TF with intercept-only model for all parameters."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate data from t-distribution
        n = 100
        y = np.random.standard_t(df=5, size=n) * 2.0 + 10.0
        
        # Prepare data dictionary
        data = {"y": y}
        
        # Fit with Python
        py_model = gamlss(
            formula="y ~ 1",
            sigma_formula="~ 1",
            parameter_formulas={"nu": "~ 1"},
            family=TF(),
            data=data,
        )
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ 1",
            family="TF",
            sigma_formula="~1",
            nu_formula="~1",
        )
        
        # Check that R fitting succeeded
        self.assertTrue(r_result["success"], f"R fitting failed: {r_result.get('error', 'Unknown error')}")
        self.assertTrue(r_result["converged"], "R model did not converge")
        
        # Compare mu coefficients
        py_mu_coef = py_model.coefficients["mu"]
        r_mu_coef = np.array(r_result["coefficients"]["mu"])
        
        np.testing.assert_allclose(
            py_mu_coef,
            r_mu_coef,
            rtol=5e-5,
            atol=1e-7,
            err_msg="Mu coefficients differ between Python and R",
        )
        
        # Compare sigma coefficients (may need wider tolerance)
        py_sigma_coef = py_model.coefficients["sigma"]
        r_sigma_coef = np.array(r_result["coefficients"]["sigma"])
        
        np.testing.assert_allclose(
            py_sigma_coef,
            r_sigma_coef,
            rtol=1e-3,
            atol=1e-3,
            err_msg="Sigma coefficients differ between Python and R",
        )
        
        # Compare nu coefficients (degrees of freedom, may need wider tolerance)
        py_nu_coef = py_model.coefficients["nu"]
        r_nu_coef = np.array(r_result["coefficients"]["nu"])
        
        np.testing.assert_allclose(
            py_nu_coef,
            r_nu_coef,
            rtol=1e-2,
            atol=1e-1,
            err_msg="Nu coefficients differ between Python and R",
        )
        
        # Compare deviance
        py_deviance = deviance(py_model)
        r_deviance = r_result["deviance"]
        
        np.testing.assert_allclose(
            py_deviance,
            r_deviance,
            rtol=5e-3,
            atol=1e-3,
            err_msg="Deviance differs between Python and R",
        )
        
        # Compare AIC
        py_aic = ic(py_model, k=2.0)
        r_aic = r_result["aic"]
        
        np.testing.assert_allclose(
            py_aic,
            r_aic,
            rtol=5e-3,
            atol=1e-3,
            err_msg="AIC differs between Python and R",
        )


    def test_tf_with_covariate_mu(self):
        """Test TF with covariate in mu."""
        # Set random seed
        np.random.seed(123)
        
        n = 100
        x = np.linspace(0, 10, n)
        y = 2.0 * x + 5.0 + np.random.standard_t(df=5, size=n) * 2.0
        
        # Prepare data dictionary
        data = {"y": y, "x": x}
        
        # Fit with Python
        py_model = gamlss(
            formula="y ~ x",
            sigma_formula="~ 1",
            parameter_formulas={"nu": "~ 1"},
            family=TF(),
            data=data,
        )
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ x",
            family="TF",
            sigma_formula="~1",
            nu_formula="~1",
        )
        
        # Check R success
        self.assertTrue(r_result["success"])
        
        # Compare mu coefficients
        py_mu_coef = py_model.coefficients["mu"]
        r_mu_coef = np.array(r_result["coefficients"]["mu"])
        
        np.testing.assert_allclose(
            py_mu_coef,
            r_mu_coef,
            rtol=5e-4,
            atol=1e-4,
            err_msg="Mu coefficients differ between Python and R",
        )
        
        # Compare sigma coefficients
        py_sigma_coef = py_model.coefficients["sigma"]
        r_sigma_coef = np.array(r_result["coefficients"]["sigma"])
        
        np.testing.assert_allclose(
            py_sigma_coef,
            r_sigma_coef,
            rtol=1e-3,
            atol=1e-3,
            err_msg="Sigma coefficients differ between Python and R",
        )
        
        # Compare nu coefficients
        py_nu_coef = py_model.coefficients["nu"]
        r_nu_coef = np.array(r_result["coefficients"]["nu"])
        
        np.testing.assert_allclose(
            py_nu_coef,
            r_nu_coef,
            rtol=1e-2,
            atol=1e-1,
            err_msg="Nu coefficients differ between Python and R",
        )
        
        # Compare deviance
        np.testing.assert_allclose(
            deviance(py_model),
            r_result["deviance"],
            rtol=5e-3,
            atol=1e-3,
            err_msg="Deviance differs between Python and R",
        )


    def test_tf_with_covariate_sigma(self):
        """Test TF with covariate in sigma (heteroscedastic)."""
        # Set random seed
        np.random.seed(456)
        
        n = 100
        x = np.linspace(0, 5, n)
        # Heteroscedastic: variance increases with x
        sigma_true = 1.0 + 0.3 * x
        y = 10.0 + np.random.standard_t(df=5, size=n) * sigma_true
        
        # Prepare data dictionary
        data = {"y": y, "x": x}
        
        # Fit with Python
        py_model = gamlss(
            formula="y ~ 1",
            sigma_formula="~ x",
            parameter_formulas={"nu": "~ 1"},
            family=TF(),
            data=data,
        )
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ 1",
            family="TF",
            sigma_formula="~x",
            nu_formula="~1",
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
            err_msg="Mu coefficients differ between Python and R",
        )
        
        # Compare sigma coefficients (may need wider tolerance)
        py_sigma_coef = py_model.coefficients["sigma"]
        r_sigma_coef = np.array(r_result["coefficients"]["sigma"])
        
        np.testing.assert_allclose(
            py_sigma_coef,
            r_sigma_coef,
            rtol=1e-2,
            atol=1e-2,
            err_msg="Sigma coefficients differ between Python and R",
        )
        
        # Compare nu coefficients
        py_nu_coef = py_model.coefficients["nu"]
        r_nu_coef = np.array(r_result["coefficients"]["nu"])
        
        np.testing.assert_allclose(
            py_nu_coef,
            r_nu_coef,
            rtol=1e-2,
            atol=1e-1,
            err_msg="Nu coefficients differ between Python and R",
        )
        
        # Compare deviance
        np.testing.assert_allclose(
            deviance(py_model),
            r_result["deviance"],
            rtol=5e-3,
            atol=1e-3,
            err_msg="Deviance differs between Python and R",
        )


if __name__ == "__main__":
    unittest.main()
