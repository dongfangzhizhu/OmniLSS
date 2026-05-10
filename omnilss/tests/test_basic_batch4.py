"""Basic tests for Batch 4 distributions (Beta inflated families)."""

import unittest
import numpy as np
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omnilss.distributions_b4 import BEINF, BEINF0, BEINF1, BEOI, BEZI


class TestBatch4Instantiation(unittest.TestCase):
    """Test that Batch 4 distributions can be instantiated."""
    
    def test_beinf_instantiation(self):
        family = BEINF()
        self.assertIsNotNone(family)
        self.assertEqual(family.name, "BEINF")
    
    def test_beinf0_instantiation(self):
        family = BEINF0()
        self.assertIsNotNone(family)
        self.assertEqual(family.name, "BEINF0")
    
    def test_beinf1_instantiation(self):
        family = BEINF1()
        self.assertIsNotNone(family)
        self.assertEqual(family.name, "BEINF1")
    
    def test_beoi_instantiation(self):
        family = BEOI()
        self.assertIsNotNone(family)
        self.assertEqual(family.name, "BEOI")
    
    def test_bezi_instantiation(self):
        family = BEZI()
        self.assertIsNotNone(family)
        self.assertEqual(family.name, "BEZI")


class TestBatch4PDF(unittest.TestCase):
    """Test that Batch 4 distributions have working PDF functions."""
    
    def test_beinf_pdf(self):
        family = BEINF()
        y = np.array([0.0, 0.3, 0.5, 0.7, 1.0])
        mu = np.full(5, 0.5)
        sigma = np.full(5, 0.2)
        nu = np.full(5, 0.1)
        tau = np.full(5, 0.1)
        pdf = family.pdf(y, mu, sigma, nu, tau)
        self.assertTrue(np.all(np.isfinite(pdf)))
        self.assertTrue(np.all(pdf >= 0))
    
    def test_beinf0_pdf(self):
        family = BEINF0()
        y = np.array([0.0, 0.3, 0.5, 0.7, 1.0])
        mu = np.full(5, 0.5)
        sigma = np.full(5, 0.2)
        nu = np.full(5, 0.1)
        pdf = family.pdf(y, mu, sigma, nu)
        self.assertTrue(np.all(np.isfinite(pdf)))
        self.assertTrue(np.all(pdf >= 0))
    
    def test_beinf1_pdf(self):
        family = BEINF1()
        y = np.array([0.0, 0.3, 0.5, 0.7, 1.0])
        mu = np.full(5, 0.5)
        sigma = np.full(5, 0.2)
        nu = np.full(5, 0.1)
        pdf = family.pdf(y, mu, sigma, nu)
        self.assertTrue(np.all(np.isfinite(pdf)))
        self.assertTrue(np.all(pdf >= 0))
    
    def test_beoi_pdf(self):
        family = BEOI()
        y = np.array([0.0, 0.3, 0.5, 0.7, 1.0])
        mu = np.full(5, 0.5)
        sigma = np.full(5, 0.2)
        nu = np.full(5, 0.1)
        pdf = family.pdf(y, mu, sigma, nu)
        self.assertTrue(np.all(np.isfinite(pdf)))
        self.assertTrue(np.all(pdf >= 0))
    
    def test_bezi_pdf(self):
        family = BEZI()
        y = np.array([0.0, 0.3, 0.5, 0.7, 1.0])
        mu = np.full(5, 0.5)
        sigma = np.full(5, 0.2)
        nu = np.full(5, 0.1)
        pdf = family.pdf(y, mu, sigma, nu)
        self.assertTrue(np.all(np.isfinite(pdf)))
        self.assertTrue(np.all(pdf >= 0))


if __name__ == '__main__':
    unittest.main()
