"""Tests for B-splines basis functions."""

import unittest

import jax.numpy as jnp
import numpy as np

from omnilss.smoothers.bsplines import (
    bspline_basis,
    bspline_design_matrix,
    create_knots,
)


class TestBSplines(unittest.TestCase):
    def test_create_knots_basic(self):
        """Test basic knot creation."""
        x = np.linspace(0, 1, 100)
        knots = create_knots(x, n_knots=5, degree=3)
        
        # Should have 5 interior + 2*4 boundary = 13 knots
        self.assertEqual(len(knots), 13)
        
        # First and last knots should be repeated degree+1 times
        self.assertTrue(np.all(knots[:4] == knots[0]))
        self.assertTrue(np.all(knots[-4:] == knots[-1]))
    
    def test_bspline_basis_shape(self):
        """Test that basis matrix has correct shape."""
        x = jnp.linspace(0, 1, 100)
        knots = jnp.array([0, 0, 0, 0, 0.5, 1, 1, 1, 1])  # 5 basis functions
        
        basis = bspline_basis(x, knots, degree=3)
        
        # Should have n_knots - degree - 1 = 9 - 3 - 1 = 5 basis functions
        self.assertEqual(basis.shape, (100, 5))
    
    def test_bspline_basis_partition_of_unity(self):
        """Test that basis functions sum to 1 (partition of unity)."""
        x = jnp.linspace(0, 1, 100)
        knots = create_knots(np.array(x), n_knots=5, degree=3)
        
        basis = bspline_basis(x, jnp.array(knots), degree=3)
        
        # Sum across basis functions should be close to 1
        row_sums = jnp.sum(basis, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-10)
    
    def test_bspline_basis_non_negative(self):
        """Test that basis functions are non-negative."""
        x = jnp.linspace(0, 1, 100)
        knots = create_knots(np.array(x), n_knots=5, degree=3)
        
        basis = bspline_basis(x, jnp.array(knots), degree=3)
        
        # All values should be >= 0
        self.assertTrue(jnp.all(basis >= 0))
    
    def test_bspline_design_matrix(self):
        """Test design matrix creation."""
        x = np.linspace(0, 1, 100)
        X, knots = bspline_design_matrix(x, n_knots=10, degree=3)
        
        # Should have 10 interior knots + 3 + 1 = 14 basis functions
        self.assertEqual(X.shape[0], 100)
        self.assertEqual(X.shape[1], 14)
        
        # Check partition of unity
        row_sums = jnp.sum(X, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-10)
    
    def test_bspline_boundary_behavior(self):
        """Test behavior at boundaries."""
        x = jnp.array([0.0, 0.5, 1.0])
        knots = jnp.array([0, 0, 0, 0, 0.5, 1, 1, 1, 1])
        
        basis = bspline_basis(x, knots, degree=3)
        
        # At boundaries, at least one basis function should be non-zero
        self.assertTrue(jnp.any(basis[0, :] > 0))
        self.assertTrue(jnp.any(basis[-1, :] > 0))
    
    def test_bspline_different_degrees(self):
        """Test B-splines with different degrees."""
        x = jnp.linspace(0, 1, 100)
        
        for degree in [1, 2, 3]:
            knots = create_knots(np.array(x), n_knots=5, degree=degree)
            basis = bspline_basis(x, jnp.array(knots), degree=degree)
            
            # Check shape
            n_basis = len(knots) - degree - 1
            self.assertEqual(basis.shape, (100, n_basis))
            
            # Check partition of unity
            row_sums = jnp.sum(basis, axis=1)
            np.testing.assert_allclose(row_sums, 1.0, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
