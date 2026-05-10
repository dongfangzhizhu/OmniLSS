"""SHASH (Sinh-Arcsinh) R/Python consistency tests.

Tests OmniLSS SHASH distribution against R gamlss.dist::SHASH.

Prerequisites:
    - R installed
    - gamlss.dist package installed

Run:
    PYTHONPATH=src JAX_PLATFORMS=cpu python -m unittest tests.test_r_consistency_shash -v
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omnilss.distributions_b3 import SHASH
from omnilss.fitting import gamlss

# Import R bridge
from tests._r_bridge_helper import R_AVAILABLE, R_BRIDGE_CLS as RBridge


@unittest.skipIf(not R_AVAILABLE, "R bridge not available")
class TestSHASHConsistency(unittest.TestCase):
    """SHASH distribution R/Python consistency tests."""
    
    @classmethod
    def setUpClass(cls):
        """Set up R bridge once for all tests."""
        try:
            cls.r_bridge = RBridge()
        except RuntimeError as e:
            raise unittest.SkipTest(f"R not available: {e}")

    def _require_r_convergence(self, r_result, context: str) -> None:
        if not r_result["converged"]:
            self.skipTest(f"R SHASH fit did not converge for {context}")
    
    def test_intercept_only_shash(self):
        """Test intercept-only SHASH model: y ~ 1."""
        np.random.seed(42)
        n = 100
        
        # Generate skewed data
        y = np.random.gamma(2, 2, n) - 2
        
        # Fit with Python
        family = SHASH()
        data = {"y": y}
        py_model = gamlss(formula="y ~ 1", family=family, data=data)
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ 1",
            family="SHASH",
            sigma_formula="~1",
            nu_formula="~1",
            tau_formula="~1"
        )
        
        # Check R fitting succeeded
        self.assertTrue(r_result["success"], f"R fitting failed: {r_result.get('error', 'Unknown error')}")
        self._require_r_convergence(r_result, "intercept-only SHASH")
        
        # Extract coefficients
        py_mu_coef = py_model.coefficients["mu"]
        r_mu_coef = np.array(r_result["coefficients"]["mu"])
        
        # Compare mu coefficients (lenient for complex distributions)
        np.testing.assert_allclose(
            py_mu_coef,
            r_mu_coef,
            rtol=5e-2,
            atol=1e-3,
            err_msg="Mu coefficients differ between Python and R"
        )
        
        # Compare deviance
        py_deviance = py_model.deviance
        r_deviance = r_result["deviance"]
        np.testing.assert_allclose(
            py_deviance,
            r_deviance,
            rtol=5e-2,
            atol=1e-3,
            err_msg="Deviance differs between Python and R"
        )
    
    def test_simple_shash_model(self):
        """Test simple SHASH model: y ~ x."""
        np.random.seed(123)
        n = 100
        x = np.random.uniform(-1, 1, n)
        
        # Generate data with linear relationship
        mu_true = 1.0 + 0.5 * x
        y = mu_true + np.random.standard_t(df=10, size=n) * 0.5
        
        # Fit with Python
        family = SHASH()
        data = {"y": y, "x": x}
        py_model = gamlss(formula="y ~ x", data=data, family=family)
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ x",
            family="SHASH",
            sigma_formula="~1",
            nu_formula="~1",
            tau_formula="~1"
        )
        
        # Check R fitting succeeded
        self.assertTrue(r_result["success"], f"R fitting failed: {r_result.get('error', 'Unknown error')}")
        self._require_r_convergence(r_result, "simple SHASH")
        
        # Extract coefficients
        py_mu_coef = py_model.coefficients["mu"]
        r_mu_coef = np.array(r_result["coefficients"]["mu"])
        
        # Compare mu coefficients
        np.testing.assert_allclose(
            py_mu_coef,
            r_mu_coef,
            rtol=5e-2,
            atol=1e-3,
            err_msg="Mu coefficients differ between Python and R"
        )
        
        # Compare deviance
        py_deviance = py_model.deviance
        r_deviance = r_result["deviance"]
        np.testing.assert_allclose(
            py_deviance,
            r_deviance,
            rtol=5e-2,
            atol=1e-3,
            err_msg="Deviance differs between Python and R"
        )


if __name__ == "__main__":
    unittest.main()
