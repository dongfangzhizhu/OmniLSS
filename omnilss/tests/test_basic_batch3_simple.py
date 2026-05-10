"""Simplified basic tests for Batch 3 distributions.

These tests verify that the distributions can be instantiated and used.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omnilss.distributions_b3 import GT, SHASH, SN1, SN2, SHASHo


class TestBatch3Instantiation(unittest.TestCase):
    """Test that Batch 3 distributions can be instantiated."""
    
    def test_gt_instantiation(self):
        """Test that GT can be instantiated."""
        family = GT()
        self.assertIsNotNone(family)
        self.assertEqual(family.name, "GT")
        self.assertEqual(family.nopar, 4)
    
    def test_shash_instantiation(self):
        """Test that SHASH can be instantiated."""
        family = SHASH()
        self.assertIsNotNone(family)
        self.assertEqual(family.name, "SHASH")
        self.assertEqual(family.nopar, 4)
    
    def test_shasho_instantiation(self):
        """Test that SHASHo can be instantiated."""
        family = SHASHo()
        self.assertIsNotNone(family)
        self.assertEqual(family.name, "SHASHo")
        self.assertEqual(family.nopar, 4)
    
    def test_sn1_instantiation(self):
        """Test that SN1 can be instantiated."""
        family = SN1()
        self.assertIsNotNone(family)
        self.assertEqual(family.name, "SN1")
        self.assertEqual(family.nopar, 3)
    
    def test_sn2_instantiation(self):
        """Test that SN2 can be instantiated."""
        family = SN2()
        self.assertIsNotNone(family)
        self.assertEqual(family.name, "SN2")
        self.assertEqual(family.nopar, 3)


class TestBatch3PDF(unittest.TestCase):
    """Test that Batch 3 distributions have working PDF functions."""
    
    def test_gt_pdf(self):
        """Test that GT PDF works."""
        family = GT()
        y = np.array([0.0, 1.0, 2.0])
        mu = np.array([0.0, 0.0, 0.0])
        sigma = np.array([1.0, 1.0, 1.0])
        nu = np.array([3.0, 3.0, 3.0])
        tau = np.array([2.0, 2.0, 2.0])
        pdf = family.pdf(y, mu, sigma, nu, tau)
        self.assertTrue(np.all(np.isfinite(pdf)))
        self.assertTrue(np.all(pdf >= 0))
    
    def test_shash_pdf(self):
        """Test that SHASH PDF works."""
        family = SHASH()
        y = np.array([0.0, 1.0, 2.0])
        mu = np.array([0.0, 0.0, 0.0])
        sigma = np.array([1.0, 1.0, 1.0])
        nu = np.array([0.5, 0.5, 0.5])
        tau = np.array([1.0, 1.0, 1.0])
        pdf = family.pdf(y, mu, sigma, nu, tau)
        self.assertTrue(np.all(np.isfinite(pdf)))
        self.assertTrue(np.all(pdf >= 0))
    
    def test_sn1_pdf(self):
        """Test that SN1 PDF works."""
        family = SN1()
        y = np.array([0.0, 1.0, 2.0])
        mu = np.array([0.0, 0.0, 0.0])
        sigma = np.array([1.0, 1.0, 1.0])
        nu = np.array([0.0, 0.0, 0.0])
        pdf = family.pdf(y, mu, sigma, nu)
        self.assertTrue(np.all(np.isfinite(pdf)))
        self.assertTrue(np.all(pdf >= 0))
    
    def test_sn2_pdf(self):
        """Test that SN2 PDF works."""
        family = SN2()
        y = np.array([0.0, 1.0, 2.0])
        mu = np.array([0.0, 0.0, 0.0])
        sigma = np.array([1.0, 1.0, 1.0])
        nu = np.array([1.0, 1.0, 1.0])
        pdf = family.pdf(y, mu, sigma, nu)
        self.assertTrue(np.all(np.isfinite(pdf)))
        self.assertTrue(np.all(pdf >= 0))


class TestBatch3ScoreFunctions(unittest.TestCase):
    """Test that Batch 3 distributions have score functions."""
    
    def test_gt_has_score(self):
        """Test that GT has score functions."""
        family = GT()
        self.assertTrue(hasattr(family, 'score_functions'))
        self.assertIsNotNone(family.score_functions)
    
    def test_shash_has_score(self):
        """Test that SHASH has score functions."""
        family = SHASH()
        self.assertTrue(hasattr(family, 'score_functions'))
        self.assertIsNotNone(family.score_functions)
    
    def test_sn1_has_score(self):
        """Test that SN1 has score functions."""
        family = SN1()
        self.assertTrue(hasattr(family, 'score_functions'))
        self.assertIsNotNone(family.score_functions)
    
    def test_sn2_has_score(self):
        """Test that SN2 has score functions."""
        family = SN2()
        self.assertTrue(hasattr(family, 'score_functions'))
        self.assertIsNotNone(family.score_functions)


if __name__ == '__main__':
    unittest.main()
