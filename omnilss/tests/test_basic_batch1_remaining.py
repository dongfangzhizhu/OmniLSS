"""Basic tests for remaining Batch 1 distributions (BCCG, BCPE, BCT, JSU)."""

import unittest
import numpy as np
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omnilss.distributions import BCCG, BCPE, BCT, JSU


class TestBatch1RemainingInstantiation(unittest.TestCase):
    """Test that remaining batch 1 distributions can be instantiated."""
    
    def test_bccg_instantiation(self):
        family = BCCG()
        self.assertIsNotNone(family)
        self.assertEqual(family.name, "BCCG")
        self.assertEqual(family.parameters, ("mu", "sigma", "nu"))
    
    def test_bcpe_instantiation(self):
        family = BCPE()
        self.assertIsNotNone(family)
        self.assertEqual(family.name, "BCPE")
        self.assertEqual(family.parameters, ("mu", "sigma", "nu", "tau"))
    
    def test_bct_instantiation(self):
        family = BCT()
        self.assertIsNotNone(family)
        self.assertEqual(family.name, "BCT")
        self.assertEqual(family.parameters, ("mu", "sigma", "nu", "tau"))
    
    def test_jsu_instantiation(self):
        family = JSU()
        self.assertIsNotNone(family)
        self.assertEqual(family.name, "JSU")
        self.assertEqual(family.parameters, ("mu", "sigma", "nu", "tau"))


class TestBatch1RemainingDeviance(unittest.TestCase):
    """Test g_dev_inc (deviance increment) for remaining batch 1 distributions."""
    
    def test_bccg_deviance(self):
        family = BCCG()
        y = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        mu = np.full(5, 1.5)
        sigma = np.full(5, 0.5)
        nu = np.full(5, 0.5)
        
        dev = family.g_dev_inc(y, mu, sigma, nu)
        self.assertTrue(np.all(np.isfinite(dev)))
        self.assertTrue(np.all(dev >= 0))
    
    def test_bcpe_deviance(self):
        family = BCPE()
        y = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        mu = np.full(5, 1.5)
        sigma = np.full(5, 0.5)
        nu = np.full(5, 0.5)
        tau = np.full(5, 2.0)
        
        dev = family.g_dev_inc(y, mu, sigma, nu, tau)
        self.assertTrue(np.all(np.isfinite(dev)))
        self.assertTrue(np.all(dev >= 0))
    
    def test_bct_deviance(self):
        family = BCT()
        y = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        mu = np.full(5, 1.5)
        sigma = np.full(5, 0.5)
        nu = np.full(5, 0.5)
        tau = np.full(5, 5.0)  # tau > 2 for BCT
        
        dev = family.g_dev_inc(y, mu, sigma, nu, tau)
        self.assertTrue(np.all(np.isfinite(dev)))
        self.assertTrue(np.all(dev >= 0))
    
    def test_jsu_deviance(self):
        family = JSU()
        y = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        mu = np.full(5, 0.0)
        sigma = np.full(5, 1.0)
        nu = np.full(5, 0.0)
        tau = np.full(5, 1.0)
        
        dev = family.g_dev_inc(y, mu, sigma, nu, tau)
        self.assertTrue(np.all(np.isfinite(dev)))
        self.assertTrue(np.all(dev >= 0))


class TestBatch1RemainingScoreFunctions(unittest.TestCase):
    """Test that score functions exist and return finite values."""
    
    def test_bccg_has_score(self):
        family = BCCG()
        y = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        mu = np.full(5, 1.5)
        sigma = np.full(5, 0.5)
        nu = np.full(5, 0.5)
        
        self.assertIn("mu", family.score_functions)
        self.assertIn("sigma", family.score_functions)
        self.assertIn("nu", family.score_functions)
        
        score_mu = family.score_functions["mu"](y, mu, sigma, nu)
        self.assertTrue(np.all(np.isfinite(score_mu)))
    
    def test_bcpe_has_score(self):
        family = BCPE()
        y = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        mu = np.full(5, 1.5)
        sigma = np.full(5, 0.5)
        nu = np.full(5, 0.5)
        tau = np.full(5, 2.0)
        
        self.assertIn("mu", family.score_functions)
        self.assertIn("sigma", family.score_functions)
        self.assertIn("nu", family.score_functions)
        self.assertIn("tau", family.score_functions)
        
        score_mu = family.score_functions["mu"](y, mu, sigma, nu, tau)
        self.assertTrue(np.all(np.isfinite(score_mu)))
    
    def test_bct_has_score(self):
        family = BCT()
        y = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        mu = np.full(5, 1.5)
        sigma = np.full(5, 0.5)
        nu = np.full(5, 0.5)
        tau = np.full(5, 5.0)
        
        self.assertIn("mu", family.score_functions)
        self.assertIn("sigma", family.score_functions)
        self.assertIn("nu", family.score_functions)
        self.assertIn("tau", family.score_functions)
        
        score_mu = family.score_functions["mu"](y, mu, sigma, nu, tau)
        self.assertTrue(np.all(np.isfinite(score_mu)))
    
    def test_jsu_has_score(self):
        family = JSU()
        y = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        mu = np.full(5, 0.0)
        sigma = np.full(5, 1.0)
        nu = np.full(5, 0.0)
        tau = np.full(5, 1.0)
        
        self.assertIn("mu", family.score_functions)
        self.assertIn("sigma", family.score_functions)
        self.assertIn("nu", family.score_functions)
        self.assertIn("tau", family.score_functions)
        
        score_mu = family.score_functions["mu"](y, mu, sigma, nu, tau)
        self.assertTrue(np.all(np.isfinite(score_mu)))


if __name__ == '__main__':
    unittest.main()
