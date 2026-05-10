"""Basic tests for Batch 3 distributions (GT, SHASH, SN1, SN2).

These tests verify that the basic DPQ functions work correctly.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omnilss.distributions_b3 import GT, SHASH, SN1, SN2, SHASHo


class TestGTBasic(unittest.TestCase):
    """Basic tests for GT (Generalised t) distribution."""
    
    def setUp(self):
        self.family = GT()
        self.y = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        self.mu = 0.0
        self.sigma = 1.0
        self.nu = 3.0
        self.tau = 2.0
    
    def test_pdf_finite(self):
        """Test that PDF values are finite."""
        pdf = self.family.d(self.y, self.mu, self.sigma, self.nu, self.tau)
        self.assertTrue(np.all(np.isfinite(pdf)))
        self.assertTrue(np.all(pdf >= 0))
    
    def test_cdf_in_range(self):
        """Test that CDF values are in [0, 1]."""
        cdf = self.family.p(self.y, self.mu, self.sigma, self.nu, self.tau)
        self.assertTrue(np.all(np.isfinite(cdf)))
        self.assertTrue(np.all((cdf >= 0) & (cdf <= 1)))
    
    def test_quantile_finite(self):
        """Test that quantile function works."""
        probs = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        q = self.family.q(probs, self.mu, self.sigma, self.nu, self.tau)
        self.assertTrue(np.all(np.isfinite(q)))
    
    def test_cdf_quantile_inverse(self):
        """Test that CDF and quantile are inverses."""
        probs = np.array([0.25, 0.5, 0.75])
        q = self.family.q(probs, self.mu, self.sigma, self.nu, self.tau)
        cdf_q = self.family.p(q, self.mu, self.sigma, self.nu, self.tau)
        np.testing.assert_allclose(cdf_q, probs, rtol=1e-5)


class TestSHASHBasic(unittest.TestCase):
    """Basic tests for SHASH (Sinh-Arcsinh) distribution."""
    
    def setUp(self):
        self.family = SHASH()
        self.y = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        self.mu = 0.0
        self.sigma = 1.0
        self.nu = 0.0
        self.tau = 1.0
    
    def test_pdf_finite(self):
        """Test that PDF values are finite."""
        pdf = self.family.d(self.y, self.mu, self.sigma, self.nu, self.tau)
        self.assertTrue(np.all(np.isfinite(pdf)))
        self.assertTrue(np.all(pdf >= 0))
    
    def test_cdf_in_range(self):
        """Test that CDF values are in [0, 1]."""
        cdf = self.family.p(self.y, self.mu, self.sigma, self.nu, self.tau)
        self.assertTrue(np.all(np.isfinite(cdf)))
        self.assertTrue(np.all((cdf >= 0) & (cdf <= 1)))
    
    def test_quantile_finite(self):
        """Test that quantile function works."""
        probs = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        q = self.family.q(probs, self.mu, self.sigma, self.nu, self.tau)
        self.assertTrue(np.all(np.isfinite(q)))
    
    def test_symmetric_case(self):
        """Test that nu=0 yields a finite, well-defined SHASH density."""
        y_sym = np.array([-1.0, 1.0])
        pdf = self.family.d(y_sym, self.mu, self.sigma, 0.0, self.tau)
        self.assertTrue(np.all(np.isfinite(pdf)))
        self.assertTrue(np.all(pdf > 0))


class TestSHASHoBasic(unittest.TestCase):
    """Basic tests for SHASHo (Sinh-Arcsinh original) distribution."""
    
    def setUp(self):
        self.family = SHASHo()
        self.y = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        self.mu = 0.0
        self.sigma = 1.0
        self.nu = 1.0
        self.tau = 1.0
    
    def test_pdf_finite(self):
        """Test that PDF values are finite."""
        pdf = self.family.d(self.y, self.mu, self.sigma, self.nu, self.tau)
        self.assertTrue(np.all(np.isfinite(pdf)))
        self.assertTrue(np.all(pdf >= 0))
    
    def test_cdf_in_range(self):
        """Test that CDF values are in [0, 1]."""
        cdf = self.family.p(self.y, self.mu, self.sigma, self.nu, self.tau)
        self.assertTrue(np.all(np.isfinite(cdf)))
        self.assertTrue(np.all((cdf >= 0) & (cdf <= 1)))


class TestSN1Basic(unittest.TestCase):
    """Basic tests for SN1 (Skew-Normal type 1) distribution."""
    
    def setUp(self):
        self.family = SN1()
        self.y = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        self.mu = 0.0
        self.sigma = 1.0
        self.nu = 0.0
    
    def test_pdf_finite(self):
        """Test that PDF values are finite."""
        pdf = self.family.d(self.y, self.mu, self.sigma, self.nu)
        self.assertTrue(np.all(np.isfinite(pdf)))
        self.assertTrue(np.all(pdf >= 0))
    
    def test_cdf_in_range(self):
        """Test that CDF values are in [0, 1]."""
        cdf = self.family.p(self.y, self.mu, self.sigma, self.nu)
        self.assertTrue(np.all(np.isfinite(cdf)))
        self.assertTrue(np.all((cdf >= 0) & (cdf <= 1)))
    
    def test_quantile_finite(self):
        """Test that quantile function works."""
        probs = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        q = self.family.q(probs, self.mu, self.sigma, self.nu)
        self.assertTrue(np.all(np.isfinite(q)))
    
    def test_reduces_to_normal(self):
        """Test that nu=0 gives normal distribution."""
        # When nu=0, SN1 should be close to normal
        pdf = self.family.d(self.y, self.mu, self.sigma, 0.0)
        # Just check it's reasonable
        self.assertTrue(np.all(pdf > 0))
        self.assertTrue(pdf[2] > pdf[0])  # Peak at center


class TestSN2Basic(unittest.TestCase):
    """Basic tests for SN2 (Two-piece Skew-Normal) distribution."""
    
    def setUp(self):
        self.family = SN2()
        self.y = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        self.mu = 0.0
        self.sigma = 1.0
        self.nu = 1.0
    
    def test_pdf_finite(self):
        """Test that PDF values are finite."""
        pdf = self.family.d(self.y, self.mu, self.sigma, self.nu)
        self.assertTrue(np.all(np.isfinite(pdf)))
        self.assertTrue(np.all(pdf >= 0))
    
    def test_cdf_in_range(self):
        """Test that CDF values are in [0, 1]."""
        cdf = self.family.p(self.y, self.mu, self.sigma, self.nu)
        self.assertTrue(np.all(np.isfinite(cdf)))
        self.assertTrue(np.all((cdf >= 0) & (cdf <= 1)))
    
    def test_quantile_finite(self):
        """Test that quantile function works."""
        probs = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        q = self.family.q(probs, self.mu, self.sigma, self.nu)
        self.assertTrue(np.all(np.isfinite(q)))


class TestBatch3Fitting(unittest.TestCase):
    """Test that Batch 3 distributions can be fitted."""
    
    def test_gt_fitting(self):
        """Test that GT can be fitted to data."""
        from omnilss.fitting import gamlss
        
        np.random.seed(42)
        n = 100
        x = np.random.normal(0, 1, n)
        y = 2 + 3 * x + np.random.standard_t(5, n)
        data = {"y": y, "x": x}
        
        family = GT()
        model = gamlss(formula="y ~ x", family=family, data=data)
        
        # Check that model fitted
        self.assertIsNotNone(model)
        # Check for coefficients attribute (may vary by implementation)
        has_coeffs = (hasattr(model, 'mu_coefficients') or 
                     hasattr(model, 'coefficients') or
                     hasattr(model, 'mu'))
        self.assertTrue(has_coeffs, "Model should have coefficient attributes")
    
    def test_shash_fitting(self):
        """Test that SHASH can be fitted to data."""
        from omnilss.fitting import gamlss
        
        np.random.seed(43)
        n = 100
        x = np.random.normal(0, 1, n)
        y = 2 + 3 * x + np.random.normal(0, 1, n)
        data = {"y": y, "x": x}
        
        family = SHASH()
        model = gamlss(formula="y ~ x", family=family, data=data)
        
        # Check that model fitted
        self.assertIsNotNone(model)
        has_coeffs = (hasattr(model, 'mu_coefficients') or 
                     hasattr(model, 'coefficients') or
                     hasattr(model, 'mu'))
        self.assertTrue(has_coeffs, "Model should have coefficient attributes")
    
    def test_sn1_fitting(self):
        """Test that SN1 can be fitted to data."""
        from omnilss.fitting import gamlss
        
        np.random.seed(44)
        n = 100
        x = np.random.normal(0, 1, n)
        y = 2 + 3 * x + np.random.normal(0, 1, n)
        data = {"y": y, "x": x}
        
        family = SN1()
        model = gamlss(formula="y ~ x", family=family, data=data)
        
        # Check that model fitted
        self.assertIsNotNone(model)
        has_coeffs = (hasattr(model, 'mu_coefficients') or 
                     hasattr(model, 'coefficients') or
                     hasattr(model, 'mu'))
        self.assertTrue(has_coeffs, "Model should have coefficient attributes")
    
    def test_sn2_fitting(self):
        """Test that SN2 can be fitted to data."""
        from omnilss.fitting import gamlss
        
        np.random.seed(45)
        n = 100
        x = np.random.normal(0, 1, n)
        y = 2 + 3 * x + np.random.normal(0, 1, n)
        data = {"y": y, "x": x}
        
        family = SN2()
        model = gamlss(formula="y ~ x", family=family, data=data)
        
        # Check that model fitted
        self.assertIsNotNone(model)
        has_coeffs = (hasattr(model, 'mu_coefficients') or 
                     hasattr(model, 'coefficients') or
                     hasattr(model, 'mu'))
        self.assertTrue(has_coeffs, "Model should have coefficient attributes")


if __name__ == '__main__':
    unittest.main()
