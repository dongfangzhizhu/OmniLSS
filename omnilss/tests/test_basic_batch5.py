"""Basic tests for Batch 5 distributions (Zero-inflated/altered families)."""

import unittest
import numpy as np
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omnilss.distributions_b5 import ZAGA, ZAIG, ZAP, ZINBI, ZIP2


class TestBatch5Instantiation(unittest.TestCase):
    """Test that Batch 5 distributions can be instantiated."""
    
    def test_zaga_instantiation(self):
        family = ZAGA()
        self.assertIsNotNone(family)
        self.assertEqual(family.name, "ZAGA")
    
    def test_zaig_instantiation(self):
        family = ZAIG()
        self.assertIsNotNone(family)
        self.assertEqual(family.name, "ZAIG")
    
    def test_zap_instantiation(self):
        family = ZAP()
        self.assertIsNotNone(family)
        self.assertEqual(family.name, "ZAP")
    
    def test_zinbi_instantiation(self):
        family = ZINBI()
        self.assertIsNotNone(family)
        self.assertEqual(family.name, "ZINBI")
    
    def test_zip2_instantiation(self):
        family = ZIP2()
        self.assertIsNotNone(family)
        self.assertEqual(family.name, "ZIP2")


class TestBatch5PDF(unittest.TestCase):
    """Test that Batch 5 distributions have working PDF functions."""
    
    def test_zaga_pdf(self):
        family = ZAGA()
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        mu = np.full(5, 2.0)
        sigma = np.full(5, 1.0)
        nu = np.full(5, 0.1)
        pdf = family.pdf(y, mu, sigma, nu)
        self.assertTrue(np.all(np.isfinite(pdf)))
        self.assertTrue(np.all(pdf >= 0))
    
    def test_zaig_pdf(self):
        family = ZAIG()
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        mu = np.full(5, 2.0)
        sigma = np.full(5, 1.0)
        nu = np.full(5, 0.1)
        pdf = family.pdf(y, mu, sigma, nu)
        self.assertTrue(np.all(np.isfinite(pdf)))
        self.assertTrue(np.all(pdf >= 0))
    
    def test_zap_pdf(self):
        family = ZAP()
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        mu = np.full(5, 2.0)
        sigma = np.full(5, 0.1)
        pdf = family.pdf(y, mu, sigma)
        self.assertTrue(np.all(np.isfinite(pdf)))
        self.assertTrue(np.all(pdf >= 0))
    
    def test_zinbi_pdf(self):
        family = ZINBI()
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        mu = np.full(5, 2.0)
        sigma = np.full(5, 1.0)
        nu = np.full(5, 0.1)
        pdf = family.pdf(y, mu, sigma, nu)
        self.assertTrue(np.all(np.isfinite(pdf)))
        self.assertTrue(np.all(pdf >= 0))
    
    def test_zip2_pdf(self):
        family = ZIP2()
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        mu = np.full(5, 2.0)
        sigma = np.full(5, 0.1)
        pdf = family.pdf(y, mu, sigma)
        self.assertTrue(np.all(np.isfinite(pdf)))
        self.assertTrue(np.all(pdf >= 0))


if __name__ == '__main__':
    unittest.main()
