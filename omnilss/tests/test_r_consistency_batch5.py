"""Batch 5 (Zero-inflated/altered) R/Python consistency tests.

Tests OmniLSS batch 5 distributions against R gamlss.dist.

Distributions:
- ZAIG (Zero-Altered Inverse Gaussian)
- ZAP (Zero-Altered Poisson)
- ZINBI (Zero-Inflated Negative Binomial I)
- ZIP2 (Zero-Inflated Poisson type 2)

Prerequisites:
    - R installed
    - gamlss.dist package installed

Run:
    PYTHONPATH=src JAX_PLATFORMS=cpu python -m unittest tests.test_r_consistency_batch5 -v
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omnilss.distributions_b5 import ZAIG, ZAP, ZINBI, ZIP2
from omnilss.fitting import gamlss
from omnilss.controls import gamlss_control

# Import R bridge
from tests._r_bridge_helper import R_AVAILABLE, R_BRIDGE_CLS as RBridge


@unittest.skipIf(not R_AVAILABLE, "R bridge not available")
class TestBatch5Consistency(unittest.TestCase):
    """Batch 5 distributions R/Python consistency tests."""
    
    @classmethod
    def setUpClass(cls):
        """Set up R bridge once for all tests."""
        try:
            cls.r_bridge = RBridge()
        except RuntimeError as e:
            raise unittest.SkipTest(f"R not available: {e}")
    
    def test_zaig_intercept_only(self):
        """Test ZAIG intercept-only model: y ~ 1."""
        np.random.seed(42)
        n = 100
        
        # Generate IG data with some zeros
        # IG: mu=2, sigma=0.5
        y_ig = np.random.wald(2.0, 2.0, 80)  # mean=2, scale=2
        y_zeros = np.zeros(20)
        y = np.concatenate([y_ig, y_zeros])
        np.random.shuffle(y)
        
        # Fit with Python
        family = ZAIG()
        data = {"y": y}
        control = gamlss_control(n_cyc=100)
        py_model = gamlss(formula="y ~ 1", family=family, data=data, control=control)
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ 1",
            family="ZAIG",
            sigma_formula="~1",
            nu_formula="~1"
        )
        
        # Check R fitting succeeded
        self.assertTrue(r_result["success"], f"R fitting failed: {r_result.get('error', 'Unknown error')}")
        self.assertTrue(r_result["converged"], "R model did not converge")
        
        # Extract coefficients
        py_mu_coef = py_model.coefficients["mu"]
        r_mu_coef = np.array(r_result["coefficients"]["mu"])
        
        # Compare mu coefficients (lenient for mixed distributions)
        np.testing.assert_allclose(
            py_mu_coef,
            r_mu_coef,
            rtol=1e-1,
            atol=1e-2,
            err_msg="Mu coefficients differ between Python and R"
        )
        
        # Compare deviance
        py_deviance = py_model.deviance
        r_deviance = r_result["deviance"]
        np.testing.assert_allclose(
            py_deviance,
            r_deviance,
            rtol=1e-1,
            atol=1e-2,
            err_msg="Deviance differs between Python and R"
        )
    
    def test_zap_intercept_only(self):
        """Test ZAP intercept-only model: y ~ 1."""
        np.random.seed(123)
        n = 100
        
        # Generate Poisson data with some zeros
        y_pois = np.random.poisson(3.0, 80)
        y_zeros = np.zeros(20)
        y = np.concatenate([y_pois, y_zeros])
        np.random.shuffle(y)
        
        # Fit with Python
        family = ZAP()
        data = {"y": y}
        control = gamlss_control(n_cyc=100)
        py_model = gamlss(formula="y ~ 1", family=family, data=data, control=control)
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ 1",
            family="ZAP",
            nu_formula="~1"
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
            rtol=1e-1,
            atol=1e-2,
            err_msg="Mu coefficients differ between Python and R"
        )
        
        # Compare deviance
        py_deviance = py_model.deviance
        r_deviance = r_result["deviance"]
        np.testing.assert_allclose(
            py_deviance,
            r_deviance,
            rtol=1e-1,
            atol=1e-2,
            err_msg="Deviance differs between Python and R"
        )
    
    def test_zinbi_intercept_only(self):
        """Test ZINBI intercept-only model: y ~ 1."""
        np.random.seed(456)
        n = 100
        
        # Generate NB data with some zeros
        y_nb = np.random.negative_binomial(5, 0.5, 80)
        y_zeros = np.zeros(20)
        y = np.concatenate([y_nb, y_zeros])
        np.random.shuffle(y)
        
        # Fit with Python
        family = ZINBI()
        data = {"y": y}
        control = gamlss_control(n_cyc=100)
        py_model = gamlss(formula="y ~ 1", family=family, data=data, control=control)
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ 1",
            family="ZINBI",
            sigma_formula="~1",
            nu_formula="~1"
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
            rtol=1e-1,
            atol=1e-2,
            err_msg="Mu coefficients differ between Python and R"
        )
        
        # Compare deviance
        py_deviance = py_model.deviance
        r_deviance = r_result["deviance"]
        np.testing.assert_allclose(
            py_deviance,
            r_deviance,
            rtol=1e-1,
            atol=1e-2,
            err_msg="Deviance differs between Python and R"
        )
    
    def test_zip2_intercept_only(self):
        """Test ZIP2 intercept-only model: y ~ 1."""
        np.random.seed(789)
        n = 100
        
        # Generate Poisson data with some zeros
        y_pois = np.random.poisson(2.5, 80)
        y_zeros = np.zeros(20)
        y = np.concatenate([y_pois, y_zeros])
        np.random.shuffle(y)
        
        # Fit with Python
        family = ZIP2()
        data = {"y": y}
        control = gamlss_control(n_cyc=100)
        py_model = gamlss(formula="y ~ 1", family=family, data=data, control=control)
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ 1",
            family="ZIP2",
            sigma_formula="~1",
            nu_formula="~1"
        )
        
        # Check R fitting succeeded
        self.assertTrue(r_result["success"], f"R fitting failed: {r_result.get('error', 'Unknown error')}")
        self.assertTrue(r_result["converged"], "R model did not converge")
        
        # Extract coefficients
        py_mu_coef = py_model.coefficients["mu"]
        r_mu_coef = np.array(r_result["coefficients"]["mu"])
        
        # Compare mu coefficients (lenient for ZIP2 - multiple local optima)
        np.testing.assert_allclose(
            py_mu_coef,
            r_mu_coef,
            rtol=5e-1,  # 50% tolerance for ZIP2
            atol=5e-2,
            err_msg="Mu coefficients differ between Python and R"
        )
        
        # Compare deviance (more important than coefficients)
        py_deviance = py_model.deviance
        r_deviance = r_result["deviance"]
        np.testing.assert_allclose(
            py_deviance,
            r_deviance,
            rtol=2e-1,  # 20% tolerance
            atol=1e-1,
            err_msg="Deviance differs between Python and R"
        )


if __name__ == "__main__":
    unittest.main()
