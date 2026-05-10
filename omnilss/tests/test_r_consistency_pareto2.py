"""PARETO2 (Pareto Type 2 / Lomax) R/Python consistency tests.

Tests OmniLSS PARETO2 distribution against R gamlss.dist::PARETO2.

Prerequisites:
    - R installed
    - gamlss.dist package installed

Run:
    PYTHONPATH=src JAX_PLATFORMS=cpu python -m unittest tests.test_r_consistency_pareto2 -v
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omnilss.distributions_b1 import PARETO2
from omnilss.fitting import gamlss

# Import R bridge
from tests._r_bridge_helper import R_AVAILABLE, R_BRIDGE_CLS as RBridge


@unittest.skipIf(not R_AVAILABLE, "R bridge not available")
class TestPARETO2Consistency(unittest.TestCase):
    """PARETO2 distribution R/Python consistency tests."""
    
    @classmethod
    def setUpClass(cls):
        """Set up R bridge once for all tests."""
        try:
            cls.r_bridge = RBridge()
        except RuntimeError as e:
            raise unittest.SkipTest(f"R not available: {e}")

    def _require_r_convergence(self, r_result, context: str) -> None:
        if not r_result["converged"]:
            self.skipTest(f"R PARETO2 fit did not converge for {context}")
    
    def _generate_pareto2_data_in_r(self, n, mu, sigma, seed=42):
        """Generate PARETO2 data using R to ensure correct parameterization.
        
        Args:
            n: Sample size
            mu: Can be scalar or array
            sigma: Scalar sigma parameter (must be < 1 for finite mean)
            seed: Random seed
            
        Returns:
            numpy array of generated data
        """
        import subprocess
        
        if isinstance(mu, (list, np.ndarray)):
            mu_str = "c(" + ",".join([str(m) for m in mu]) + ")"
        else:
            mu_str = str(mu)
        
        r_gen_script = f"""
        library(gamlss.dist)
        set.seed({seed})
        y <- rPARETO2({n}, mu={mu_str}, sigma={sigma})
        cat(paste(y, collapse=" "))
        """
        
        result = subprocess.run(
            ["Rscript", "-e", r_gen_script],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to generate data in R: {result.stderr}")
        
        # Parse output
        y_values = [float(v) for v in result.stdout.strip().split()]
        return np.array(y_values)
    
    def test_intercept_only_pareto2(self):
        """Test intercept-only PARETO2 model: y ~ 1."""
        n = 100
        mu_true = 2.0
        sigma_true = 0.5  # Must be < 1 for finite mean
        
        # Generate data using R
        y = self._generate_pareto2_data_in_r(n, mu_true, sigma_true, seed=42)
        
        # Fit with Python
        family = PARETO2()
        data = {"y": y}
        py_model = gamlss(formula="y ~ 1", family=family, data=data)
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ 1",
            family="PARETO2",
            sigma_formula="~1"
        )
        
        # Check R fitting succeeded
        self.assertTrue(r_result["success"], f"R fitting failed: {r_result.get('error', 'Unknown error')}")
        self._require_r_convergence(r_result, "intercept-only PARETO2")
        
        # Extract coefficients
        py_mu_coef = py_model.coefficients["mu"]
        py_sigma_coef = py_model.coefficients["sigma"]
        r_mu_coef = np.array(r_result["coefficients"]["mu"])
        r_sigma_coef = np.array(r_result["coefficients"]["sigma"])
        
        # Compare mu coefficients
        np.testing.assert_allclose(
            py_mu_coef,
            r_mu_coef,
            rtol=1e-2,
            atol=1e-5,
            err_msg="Mu coefficients differ between Python and R"
        )
        
        # Compare sigma coefficients
        np.testing.assert_allclose(
            py_sigma_coef,
            r_sigma_coef,
            rtol=1e-2,
            atol=1e-5,
            err_msg="Sigma coefficients differ between Python and R"
        )
        
        # Compare deviance
        py_deviance = py_model.deviance
        r_deviance = r_result["deviance"]
        np.testing.assert_allclose(
            py_deviance,
            r_deviance,
            rtol=1e-2,
            atol=1e-5,
            err_msg="Deviance differs between Python and R"
        )
    
    def test_simple_pareto2_model(self):
        """Test simple PARETO2 model: y ~ x."""
        np.random.seed(123)
        n = 100
        x = np.random.uniform(0, 2, n)
        
        # Generate data with linear relationship
        mu_true = np.exp(0.5 + 0.3 * x)
        sigma_true = 0.6  # Must be < 1 for finite mean
        
        # Generate data using R
        y = self._generate_pareto2_data_in_r(n, mu_true.tolist(), sigma_true, seed=123)
        
        # Fit with Python
        family = PARETO2()
        data = {"y": y, "x": x}
        py_model = gamlss(formula="y ~ x", data=data, family=family)
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ x",
            family="PARETO2",
            sigma_formula="~1"
        )
        
        # Check R fitting succeeded
        self.assertTrue(r_result["success"], f"R fitting failed: {r_result.get('error', 'Unknown error')}")
        self._require_r_convergence(r_result, "simple PARETO2")
        
        # Extract coefficients
        py_mu_coef = py_model.coefficients["mu"]
        r_mu_coef = np.array(r_result["coefficients"]["mu"])
        
        # Compare mu coefficients
        np.testing.assert_allclose(
            py_mu_coef,
            r_mu_coef,
            rtol=1e-2,
            atol=1e-5,
            err_msg="Mu coefficients differ between Python and R"
        )
        
        # Compare deviance
        py_deviance = py_model.deviance
        r_deviance = r_result["deviance"]
        np.testing.assert_allclose(
            py_deviance,
            r_deviance,
            rtol=1e-2,
            atol=1e-5,
            err_msg="Deviance differs between Python and R"
        )
    
    def test_multiple_predictors_pareto2(self):
        """Test PARETO2 model with multiple predictors: y ~ x1 + x2."""
        np.random.seed(456)
        n = 100
        x1 = np.random.uniform(0, 1, n)
        x2 = np.random.uniform(-0.5, 0.5, n)
        
        # Generate data
        mu_true = np.exp(0.8 + 0.4 * x1 - 0.2 * x2)
        sigma_true = 0.7  # Must be < 1 for finite mean
        
        # Generate data using R
        y = self._generate_pareto2_data_in_r(n, mu_true.tolist(), sigma_true, seed=456)
        
        # Fit with Python
        family = PARETO2()
        data = {"y": y, "x1": x1, "x2": x2}
        py_model = gamlss(
            formula="y ~ x1 + x2",
            data=data,
            family=family
        )
        
        # Fit with R
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ x1 + x2",
            family="PARETO2",
            sigma_formula="~1"
        )
        
        # Check R fitting succeeded
        self.assertTrue(r_result["success"], f"R fitting failed: {r_result.get('error', 'Unknown error')}")
        self._require_r_convergence(r_result, "multi-predictor PARETO2")
        
        # Extract coefficients
        py_mu_coef = py_model.coefficients["mu"]
        r_mu_coef = np.array(r_result["coefficients"]["mu"])
        
        # Compare deviance
        py_deviance = py_model.deviance
        r_deviance = r_result["deviance"]
        np.testing.assert_allclose(
            py_deviance,
            r_deviance,
            rtol=5e-2,
            atol=1e-4,
            err_msg="Deviance differs between Python and R"
        )


if __name__ == "__main__":
    unittest.main()
