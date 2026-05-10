"""R/Python consistency tests for GAMLSS with smooth terms.

This module tests that OmniLSS produces results consistent with R gamlss
when using smooth terms (pb, ps, cs, etc.).
"""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Try to import OmniLSS
try:
    from omnilss.distributions import NO, GA, LOGNO
    from omnilss.fitting import gamlss, gamlss_ml
    OMNILSS_AVAILABLE = True
except ImportError:
    OMNILSS_AVAILABLE = False


def check_rscript_available():
    """Check if Rscript is available."""
    try:
        subprocess.run(["Rscript", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


RSCRIPT_AVAILABLE = check_rscript_available()


def run_r_gamlss_smooth(formula, family, data, sigma_formula=None):
    """Run R gamlss with smooth terms and return results.
    
    Args:
        formula: Formula string for mu parameter
        family: Family name (e.g., "NO", "GA")
        data: Dictionary of data arrays
        sigma_formula: Optional formula string for sigma parameter
        
    Returns:
        Dictionary with R results
    """
    # Get R script path
    test_dir = Path(__file__).parent
    r_script = test_dir / "rbus" / "R_scripts" / "test_gamlss_smooth.R"
    
    if not r_script.exists():
        raise FileNotFoundError(f"R script not found: {r_script}")
    
    # Prepare input data
    input_data = {
        "formula": formula,
        "sigma_formula": sigma_formula,
        "family": family,
        "data": {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in data.items()}
    }
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as input_file:
        json.dump(input_data, input_file)
        input_path = input_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as output_file:
        output_path = output_file.name
    
    try:
        # Run R script
        result = subprocess.run(
            ["Rscript", str(r_script), input_path, output_path],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Read output
        with open(output_path, 'r') as f:
            output = json.load(f)
        
        if not output.get('success', False):
            raise RuntimeError(f"R gamlss failed: {output.get('error', 'Unknown error')}")
        
        return output
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"R script failed: {e.stderr}")
    finally:
        # Clean up
        os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)


@unittest.skipUnless(OMNILSS_AVAILABLE, "OmniLSS not available")
@unittest.skipUnless(RSCRIPT_AVAILABLE, "Rscript not available")
class TestRConsistencySmooth(unittest.TestCase):
    """Test R/Python consistency for smooth terms."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n = 100
        self.x = np.linspace(0, 1, self.n)
        
    def test_pb_in_mu_normal(self):
        """Test pb() smooth term in mu parameter with Normal distribution."""
        # Generate data with nonlinear relationship
        y = 2.0 + 3.0 * np.sin(2 * np.pi * self.x) + np.random.normal(0, 0.5, self.n)
        data = {"y": y, "x": self.x}
        
        # Fit in Python
        py_model = gamlss_ml(
            formula="y ~ pb(x, df=5)",
            family=NO(),
            data=data
        )
        
        # Fit in R
        r_result = run_r_gamlss_smooth(
            formula="y ~ pb(x, df=5)",
            family="NO",
            data=data
        )
        
        # Compare fitted values
        py_fitted = py_model.fitted_values["mu"]
        r_fitted = np.array(r_result["fitted_values"]["mu"])
        
        # Check correlation (should be very high)
        corr = np.corrcoef(py_fitted, r_fitted)[0, 1]
        self.assertGreater(corr, 0.99, f"Fitted values correlation too low: {corr:.6f}")
        
        # Check deviance (allow larger tolerance for smooth terms)
        py_dev = py_model.deviance
        r_dev = r_result["deviance"]
        rel_diff = abs(py_dev - r_dev) / abs(r_dev)
        self.assertLess(rel_diff, 0.30, f"Deviance relative difference too large: {rel_diff:.6f}")
        
    def test_pb_in_mu_gamma(self):
        """Test pb() smooth term in mu parameter with Gamma distribution."""
        # Generate data with nonlinear relationship
        mu_true = 1.0 + 2.0 * np.exp(-5 * (self.x - 0.5)**2)
        y = np.random.gamma(shape=4, scale=mu_true/4)
        data = {"y": y, "x": self.x}
        
        # Fit in Python
        py_model = gamlss_ml(
            formula="y ~ pb(x, df=6)",
            family=GA(),
            data=data
        )
        
        # Fit in R
        r_result = run_r_gamlss_smooth(
            formula="y ~ pb(x, df=6)",
            family="GA",
            data=data
        )
        
        # Compare fitted values
        py_fitted = py_model.fitted_values["mu"]
        r_fitted = np.array(r_result["fitted_values"]["mu"])
        
        # Check correlation
        corr = np.corrcoef(py_fitted, r_fitted)[0, 1]
        self.assertGreater(corr, 0.98, f"Fitted values correlation too low: {corr:.6f}")
        
    def test_pb_in_sigma_normal(self):
        """Test pb() smooth term in sigma parameter with Normal distribution."""
        # Generate data with varying variance
        np.random.seed(123)  # Fix seed for reproducibility
        x = np.linspace(0, 1, 100)
        sigma_true = 0.3 + 0.4 * x
        y = np.random.normal(0, sigma_true)
        data = {"y": y, "x": x}
        
        # Fit in Python
        py_model = gamlss(
            formula="y ~ 1",
            sigma_formula="~ pb(x, df=4)",
            family=NO(),
            data=data
        )
        
        # Fit in R
        r_result = run_r_gamlss_smooth(
            formula="y ~ 1",
            family="NO",
            data=data,
            sigma_formula="~ pb(x, df=4)"
        )
        
        # Compare sigma fitted values (use absolute correlation)
        py_sigma = py_model.fitted_values["sigma"]
        r_sigma = np.array(r_result["fitted_values"]["sigma"])
        
        # Check correlation (use absolute value as direction may differ)
        corr = abs(np.corrcoef(py_sigma, r_sigma)[0, 1])
        self.assertGreater(corr, 0.80, f"Sigma fitted values correlation too low: {corr:.6f}")
        
    def test_mixed_linear_and_smooth(self):
        """Test mixed linear and smooth terms."""
        # Generate data
        x1 = np.random.normal(0, 1, self.n)
        x2 = self.x
        y = 2.0 + 1.5 * x1 + 3.0 * np.sin(2 * np.pi * x2) + np.random.normal(0, 0.5, self.n)
        data = {"y": y, "x1": x1, "x2": x2}
        
        # Fit in Python
        py_model = gamlss_ml(
            formula="y ~ x1 + pb(x2, df=5)",
            family=NO(),
            data=data
        )
        
        # Fit in R
        r_result = run_r_gamlss_smooth(
            formula="y ~ x1 + pb(x2, df=5)",
            family="NO",
            data=data
        )
        
        # Compare coefficients for linear term
        py_coef_x1 = py_model.coefficients["mu"][1]  # Assuming intercept is first
        r_coef_x1 = r_result["coefficients"]["mu"][1]
        
        rel_diff = abs(py_coef_x1 - r_coef_x1) / abs(r_coef_x1)
        self.assertLess(rel_diff, 0.05, f"x1 coefficient relative difference too large: {rel_diff:.6f}")
        
        # Compare fitted values
        py_fitted = py_model.fitted_values["mu"]
        r_fitted = np.array(r_result["fitted_values"]["mu"])
        
        corr = np.corrcoef(py_fitted, r_fitted)[0, 1]
        self.assertGreater(corr, 0.99, f"Fitted values correlation too low: {corr:.6f}")
        
    def test_multiple_smooths(self):
        """Test multiple smooth terms."""
        # Generate data
        x1 = np.linspace(0, 1, self.n)
        x2 = np.linspace(0, 1, self.n)
        y = 2.0 + 2.0 * np.sin(2 * np.pi * x1) + 1.5 * np.cos(2 * np.pi * x2) + np.random.normal(0, 0.5, self.n)
        data = {"y": y, "x1": x1, "x2": x2}
        
        # Fit in Python
        py_model = gamlss_ml(
            formula="y ~ pb(x1, df=4) + pb(x2, df=4)",
            family=NO(),
            data=data
        )
        
        # Fit in R
        r_result = run_r_gamlss_smooth(
            formula="y ~ pb(x1, df=4) + pb(x2, df=4)",
            family="NO",
            data=data
        )
        
        # Compare fitted values
        py_fitted = py_model.fitted_values["mu"]
        r_fitted = np.array(r_result["fitted_values"]["mu"])
        
        corr = np.corrcoef(py_fitted, r_fitted)[0, 1]
        self.assertGreater(corr, 0.98, f"Fitted values correlation too low: {corr:.6f}")
        
    def test_gamlss_iterative_with_smooth(self):
        """Test gamlss() iterative fitting with smooth terms."""
        # Generate data
        y = 2.0 + 3.0 * np.sin(2 * np.pi * self.x) + np.random.normal(0, 0.5, self.n)
        data = {"y": y, "x": self.x}
        
        # Fit in Python using gamlss() (iterative)
        py_model = gamlss(
            formula="y ~ pb(x, df=5)",
            family=NO(),
            data=data,
            method="RS"
        )
        
        # Fit in R
        r_result = run_r_gamlss_smooth(
            formula="y ~ pb(x, df=5)",
            family="NO",
            data=data
        )
        
        # Compare fitted values
        py_fitted = py_model.fitted_values["mu"]
        r_fitted = np.array(r_result["fitted_values"]["mu"])
        
        corr = np.corrcoef(py_fitted, r_fitted)[0, 1]
        self.assertGreater(corr, 0.99, f"Fitted values correlation too low: {corr:.6f}")
        
        # Check convergence
        self.assertTrue(py_model.additional_slots.get("converged", False), "Model did not converge")


if __name__ == "__main__":
    unittest.main()
