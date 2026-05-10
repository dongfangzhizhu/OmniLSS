"""GT (Generalised t) R/Python consistency tests.

Tests OmniLSS GT distribution against R gamlss.dist::GT.

Prerequisites:
    - R installed
    - gamlss.dist package installed

Run:
    PYTHONPATH=src JAX_PLATFORMS=cpu python -m unittest tests.test_r_consistency_gt -v
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omnilss.distributions_b3 import GT
from omnilss.fitting import gamlss

# Import R bridge
from tests._r_bridge_helper import R_AVAILABLE, R_BRIDGE_CLS as RBridge


@unittest.skipIf(not R_AVAILABLE, "R bridge not available")
class TestGTConsistency(unittest.TestCase):
    """GT distribution R/Python consistency tests."""
    
    @classmethod
    def setUpClass(cls):
        """Set up R bridge once for all tests."""
        try:
            cls.r_bridge = RBridge()
        except RuntimeError as e:
            raise unittest.SkipTest(f"R not available: {e}")

    def _require_r_convergence(self, r_result, context: str) -> None:
        if not r_result["converged"]:
            self.skipTest(f"R GT fit did not converge for {context}")
    
    def test_intercept_only_gt(self):
        """Test intercept-only GT model: y ~ 1."""
        np.random.seed(42)
        n = 100
        
        # Generate data from t-distribution (GT with nu=df, tau=2 is close to t)
        # For simplicity, use normal data with some outliers
        y = np.concatenate([
            np.random.normal(0, 1, 90),
            np.random.normal(0, 3, 10)  # Add some outliers
        ])
        np.random.shuffle(y)
        
        # Fit with Python
        family = GT()
        data = {"y": y}
        py_model = gamlss(formula="y ~ 1", family=family, data=data)
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ 1",
            family="GT",
            sigma_formula="~1",
            nu_formula="~1",
            tau_formula="~1"
        )
        
        # Check R fitting succeeded
        self.assertTrue(r_result["success"], f"R fitting failed: {r_result.get('error', 'Unknown error')}")
        self._require_r_convergence(r_result, "intercept-only GT")
        
        # Extract coefficients
        py_mu_coef = py_model.coefficients["mu"]
        r_mu_coef = np.array(r_result["coefficients"]["mu"])
        
        # Compare mu coefficients (more lenient for complex distributions)
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
    
    def test_simple_gt_model(self):
        """Test simple GT model: y ~ x."""
        np.random.seed(123)
        n = 100
        x = np.random.uniform(-1, 1, n)
        
        # Generate data with linear relationship and heavy tails
        mu_true = 2.0 + 0.5 * x
        y = mu_true + np.random.standard_t(df=5, size=n)
        
        # Fit with Python
        family = GT()
        data = {"y": y, "x": x}
        py_model = gamlss(formula="y ~ x", data=data, family=family)
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ x",
            family="GT",
            sigma_formula="~1",
            nu_formula="~1",
            tau_formula="~1"
        )
        
        # Check R fitting succeeded
        self.assertTrue(r_result["success"], f"R fitting failed: {r_result.get('error', 'Unknown error')}")
        self._require_r_convergence(r_result, "simple GT")
        
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
