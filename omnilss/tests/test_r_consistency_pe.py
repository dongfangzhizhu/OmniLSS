"""PE R/Python consistency tests.

Tests the OmniLSS Power Exponential distribution against R gamlss fits.
"""

import unittest
import numpy as np
from pathlib import Path
import sys


src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omnilss.distributions_b2 import PE
from omnilss.fitting import gamlss
from tests._r_bridge_helper import R_AVAILABLE, R_BRIDGE_CLS as RBridge


@unittest.skipIf(not R_AVAILABLE, "R bridge not available")
class TestPEConsistency(unittest.TestCase):
    """PE consistency coverage against R."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.r_bridge = RBridge()
        except RuntimeError as e:
            raise unittest.SkipTest(f"R not available: {e}")

    def test_intercept_only_pe(self):
        """Test intercept-only PE model: y ~ 1."""
        np.random.seed(42)
        n = 120
        y = np.random.normal(loc=0.5, scale=1.0, size=n)
        data = {"y": y}

        py_model = gamlss(formula="y ~ 1", family=PE(), data=data)
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ 1",
            family="PE",
            sigma_formula="~1",
            nu_formula="~1",
        )

        self.assertTrue(r_result["success"], f"R fitting failed: {r_result.get('error', 'Unknown error')}")
        self.assertTrue(r_result["converged"], "R model did not converge")

        np.testing.assert_allclose(py_model.coefficients["mu"], np.array(r_result["coefficients"]["mu"]), rtol=5e-3, atol=1e-3)
        np.testing.assert_allclose(py_model.coefficients["sigma"], np.array(r_result["coefficients"]["sigma"]), rtol=5e-3, atol=1e-3)
        np.testing.assert_allclose(py_model.coefficients["nu"], np.array(r_result["coefficients"]["nu"]), rtol=5e-3, atol=1e-3)
        np.testing.assert_allclose(py_model.deviance, r_result["deviance"], rtol=5e-3, atol=1e-2)

    def test_simple_pe_model(self):
        """Test simple PE model: y ~ x."""
        np.random.seed(123)
        n = 120
        x = np.random.uniform(-1, 1, n)
        y = np.random.normal(loc=0.2 + 0.4 * x, scale=0.6, size=n)
        data = {"y": y, "x": x}

        py_model = gamlss(formula="y ~ x", family=PE(), data=data)
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ x",
            family="PE",
            sigma_formula="~1",
            nu_formula="~1",
        )

        self.assertTrue(r_result["success"], f"R fitting failed: {r_result.get('error', 'Unknown error')}")
        self.assertTrue(r_result["converged"], "R model did not converge")

        np.testing.assert_allclose(py_model.coefficients["mu"], np.array(r_result["coefficients"]["mu"]), rtol=8e-3, atol=2e-3)
        np.testing.assert_allclose(py_model.coefficients["sigma"], np.array(r_result["coefficients"]["sigma"]), rtol=8e-3, atol=2e-3)
        np.testing.assert_allclose(py_model.coefficients["nu"], np.array(r_result["coefficients"]["nu"]), rtol=8e-3, atol=2e-3)
        np.testing.assert_allclose(py_model.deviance, r_result["deviance"], rtol=5e-3, atol=1e-2)

    def test_multiple_predictors_pe(self):
        """Test PE model with multiple predictors: y ~ x1 + x2."""
        np.random.seed(456)
        n = 120
        x1 = np.random.uniform(-1, 1, n)
        x2 = np.random.uniform(-1, 1, n)
        y = np.random.normal(loc=0.1 + 0.3 * x1 - 0.2 * x2, scale=0.55, size=n)
        data = {"y": y, "x1": x1, "x2": x2}

        py_model = gamlss(formula="y ~ x1 + x2", family=PE(), data=data)
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ x1 + x2",
            family="PE",
            sigma_formula="~1",
            nu_formula="~1",
        )

        self.assertTrue(r_result["success"], f"R fitting failed: {r_result.get('error', 'Unknown error')}")
        self.assertTrue(r_result["converged"], "R model did not converge")

        np.testing.assert_allclose(py_model.coefficients["mu"], np.array(r_result["coefficients"]["mu"]), rtol=8e-3, atol=2e-3)
        np.testing.assert_allclose(py_model.coefficients["sigma"], np.array(r_result["coefficients"]["sigma"]), rtol=8e-3, atol=2e-3)
        np.testing.assert_allclose(py_model.coefficients["nu"], np.array(r_result["coefficients"]["nu"]), rtol=8e-3, atol=2e-3)
        np.testing.assert_allclose(py_model.deviance, r_result["deviance"], rtol=5e-3, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
