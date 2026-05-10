"""Tests for penalty matrices."""

import unittest

import jax.numpy as jnp
import numpy as np

from omnilss.smoothers.penalties import (
    difference_penalty,
    effective_df,
    find_lambda_for_df,
    penalty_matrix,
)


class TestPenalties(unittest.TestCase):
    def test_difference_penalty_order0(self):
        """Test ridge penalty (order=0)."""
        D = difference_penalty(10, order=0)
        
        # Should be identity matrix
        np.testing.assert_allclose(D, jnp.eye(10))
    
    def test_difference_penalty_order1(self):
        """Test first-order difference penalty."""
        D = difference_penalty(5, order=1)
        
        # Shape should be (n-1, n)
        self.assertEqual(D.shape, (4, 5))
        
        # First row should be [-1, 1, 0, 0, 0]
        expected_first_row = jnp.array([-1, 1, 0, 0, 0])
        np.testing.assert_allclose(D[0, :], expected_first_row)
    
    def test_difference_penalty_order2(self):
        """Test second-order difference penalty."""
        D = difference_penalty(5, order=2)
        
        # Shape should be (n-2, n)
        self.assertEqual(D.shape, (3, 5))
        
        # First row should be [1, -2, 1, 0, 0]
        expected_first_row = jnp.array([1, -2, 1, 0, 0])
        np.testing.assert_allclose(D[0, :], expected_first_row)
    
    def test_difference_penalty_order3(self):
        """Test third-order difference penalty."""
        D = difference_penalty(6, order=3)
        
        # Shape should be (n-3, n)
        self.assertEqual(D.shape, (3, 6))
        
        # First row should be [-1, 3, -3, 1, 0, 0]
        expected_first_row = jnp.array([-1, 3, -3, 1, 0, 0])
        np.testing.assert_allclose(D[0, :], expected_first_row)
    
    def test_penalty_matrix_shape(self):
        """Test that penalty matrix has correct shape."""
        P = penalty_matrix(10, order=2)
        
        # Should be square matrix of size n x n
        self.assertEqual(P.shape, (10, 10))
    
    def test_penalty_matrix_symmetric(self):
        """Test that penalty matrix is symmetric."""
        P = penalty_matrix(10, order=2)
        
        # D^T D should be symmetric
        np.testing.assert_allclose(P, P.T, rtol=1e-10)
    
    def test_penalty_matrix_positive_semidefinite(self):
        """Test that penalty matrix is positive semi-definite."""
        P = penalty_matrix(10, order=2)
        
        # Eigenvalues should be non-negative
        eigenvalues = jnp.linalg.eigvalsh(P)
        self.assertTrue(jnp.all(eigenvalues >= -1e-10))
    
    def test_effective_df_no_penalty(self):
        """Test effective df with small penalty."""
        X = jnp.eye(10)  # Use identity matrix for full rank
        P = penalty_matrix(10, order=2)
        
        # With very small lambda, edf should be close to number of parameters
        edf = effective_df(X, P, lambda_=1e-8)
        
        # For identity design matrix, edf should be close to 10
        self.assertGreater(edf, 5)
        self.assertLess(edf, 10)
    
    def test_effective_df_large_penalty(self):
        """Test effective df with large penalty."""
        X = jnp.eye(10)
        P = penalty_matrix(10, order=2)
        
        # With large lambda, edf should be small
        edf = effective_df(X, P, lambda_=1e6)
        
        self.assertLess(edf, 5)
    
    def test_effective_df_decreases_with_lambda(self):
        """Test that edf decreases as lambda increases."""
        X = jnp.eye(10)
        P = penalty_matrix(10, order=2)
        
        edf1 = effective_df(X, P, lambda_=1.0)
        edf2 = effective_df(X, P, lambda_=10.0)
        edf3 = effective_df(X, P, lambda_=100.0)
        
        self.assertGreater(edf1, edf2)
        self.assertGreater(edf2, edf3)
    
    def test_effective_df_with_weights(self):
        """Test effective df with observation weights."""
        X = jnp.eye(10)
        P = penalty_matrix(10, order=2)
        weights = jnp.ones(10)
        
        edf = effective_df(X, P, lambda_=1.0, weights=weights)
        
        self.assertGreater(edf, 0)
        self.assertLess(edf, 10)
    
    def test_find_lambda_for_df(self):
        """Test finding lambda for target df."""
        X = jnp.eye(10)
        P = penalty_matrix(10, order=2)
        
        target_df = 5.0
        lambda_ = find_lambda_for_df(X, P, target_df=target_df)
        
        # Check that we achieved target df
        actual_df = effective_df(X, P, lambda_)
        np.testing.assert_allclose(actual_df, target_df, rtol=1e-3)
    
    def test_find_lambda_for_df_different_targets(self):
        """Test finding lambda for different target df values."""
        X = jnp.eye(10)
        P = penalty_matrix(10, order=2)
        
        for target_df in [3.0, 5.0, 7.0]:
            lambda_ = find_lambda_for_df(X, P, target_df=target_df)
            actual_df = effective_df(X, P, lambda_)
            np.testing.assert_allclose(actual_df, target_df, rtol=1e-3)
    
    def test_find_lambda_for_df_out_of_range(self):
        """Test that out-of-range target df raises error."""
        X = jnp.eye(10)
        P = penalty_matrix(10, order=2)
        
        # Target df too large
        with self.assertRaises(ValueError):
            find_lambda_for_df(X, P, target_df=20.0)
        
        # Target df too small (negative)
        with self.assertRaises(ValueError):
            find_lambda_for_df(X, P, target_df=-1.0)


if __name__ == "__main__":
    unittest.main()
