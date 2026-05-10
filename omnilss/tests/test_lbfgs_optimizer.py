"""Tests for L-BFGS optimizer module.

This module tests the L-BFGS optimization functionality, including:
- Basic optimization
- Convergence detection
- Comparison with other optimizers
- Real-world scenarios
"""

import pytest
import jax.numpy as jnp
import numpy as np
from omnilss.core.lbfgs_optimizer import (
    LBFGSOptimizer,
    lbfgs_optimize,
)
from omnilss.core.optimizer import JointOptimizer, OptimizationResult


class TestLBFGSOptimizer:
    """Test suite for LBFGSOptimizer class."""
    
    def test_basic_optimization(self):
        """Test basic L-BFGS optimization."""
        # Simple quadratic loss
        def loss_fn(params, data):
            x = params["x"]
            return jnp.sum((x - data["target"]) ** 2)
        
        init_params = {"x": jnp.array([0.0, 0.0])}
        data = {"target": jnp.array([1.0, 2.0])}
        
        optimizer = LBFGSOptimizer(max_iter=50, history_size=10)
        result = optimizer.optimize(loss_fn, init_params, data)
        
        assert result.converged, "L-BFGS should converge"
        assert result.loss < 1e-3, f"Loss should be near zero, got {result.loss}"
        assert jnp.allclose(result.params["x"], data["target"], atol=0.05)
    
    def test_convergence_speed(self):
        """Test that L-BFGS converges quickly."""
        def loss_fn(params, data):
            return jnp.sum(params["x"] ** 2)
        
        init_params = {"x": jnp.array([1.0, 1.0, 1.0])}
        data = {}
        
        optimizer = LBFGSOptimizer(max_iter=100, history_size=10)
        result = optimizer.optimize(loss_fn, init_params, data)
        
        assert result.converged
        # L-BFGS should converge very quickly for quadratic problems
        assert result.n_iter < 20, f"Should converge in < 20 iterations, got {result.n_iter}"
        assert result.loss < 1e-6
    
    def test_gradient_norm_convergence(self):
        """Test convergence based on gradient norm."""
        def loss_fn(params, data):
            return jnp.sum(params["x"] ** 2)
        
        init_params = {"x": jnp.array([0.5, 0.5])}
        data = {}
        
        optimizer = LBFGSOptimizer(
            max_iter=100,
            history_size=10,
            grad_tol=1e-6
        )
        result = optimizer.optimize(loss_fn, init_params, data)
        
        assert result.converged
        assert result.grad_norms is not None
        assert result.grad_norms[-1] < 1e-5
    
    def test_history_size_effect(self):
        """Test effect of different history sizes."""
        def loss_fn(params, data):
            return jnp.sum(params["x"] ** 2)
        
        init_params = {"x": jnp.array([1.0, 1.0, 1.0, 1.0])}
        data = {}
        
        history_sizes = [5, 10, 20]
        results = []
        
        for h_size in history_sizes:
            optimizer = LBFGSOptimizer(max_iter=50, history_size=h_size)
            result = optimizer.optimize(loss_fn, init_params, data)
            results.append(result)
        
        # All should converge
        for result in results:
            assert result.converged or result.loss < 1e-4
    
    def test_multiple_parameters(self):
        """Test L-BFGS with multiple parameter groups."""
        def loss_fn(params, data):
            mu_loss = (params["mu"] - 2.0) ** 2
            sigma_loss = (params["sigma"] - 3.0) ** 2
            return mu_loss + sigma_loss
        
        init_params = {
            "mu": jnp.array(0.0),
            "sigma": jnp.array(0.0)
        }
        data = {}
        
        optimizer = LBFGSOptimizer(max_iter=50)
        result = optimizer.optimize(loss_fn, init_params, data)
        
        assert result.converged or result.loss < 1e-3
        assert jnp.allclose(result.params["mu"], 2.0, atol=0.1)
        assert jnp.allclose(result.params["sigma"], 3.0, atol=0.1)
    
    def test_non_convergence_warning(self):
        """Test warning when L-BFGS doesn't converge."""
        def loss_fn(params, data):
            # Difficult loss landscape
            return jnp.sum(jnp.sin(10 * params["x"]) ** 2)
        
        init_params = {"x": jnp.array([5.0])}
        data = {}
        
        optimizer = LBFGSOptimizer(
            max_iter=5,  # Too few iterations
            history_size=5,
            tol=1e-10  # Very strict
        )
        
        with pytest.warns(RuntimeWarning, match="did not converge"):
            result = optimizer.optimize(loss_fn, init_params, data)
        
        assert not result.converged
        assert result.n_iter == 5
    
    def test_lbfgs_optimize_convenience(self):
        """Test convenience function."""
        def loss_fn(params, data):
            return jnp.sum((params["x"] - data["target"]) ** 2)
        
        init_params = {"x": jnp.array([0.0, 0.0])}
        data = {"target": jnp.array([1.0, 2.0])}
        
        result = lbfgs_optimize(
            loss_fn=loss_fn,
            init_params=init_params,
            data=data,
            max_iter=50
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.loss < 1e-3
    
    def test_repr(self):
        """Test string representation."""
        optimizer = LBFGSOptimizer(max_iter=100, history_size=10)
        repr_str = repr(optimizer)
        
        assert "LBFGSOptimizer" in repr_str
        assert "100" in repr_str
        assert "10" in repr_str


class TestLBFGSvsAdam:
    """Compare L-BFGS with Adam optimizer."""
    
    def test_convergence_speed_comparison(self):
        """Compare convergence speed of L-BFGS vs Adam."""
        def loss_fn(params, data):
            return jnp.sum(params["x"] ** 2)
        
        init_params = {"x": jnp.array([1.0, 1.0, 1.0])}
        data = {}
        
        # L-BFGS
        lbfgs_opt = LBFGSOptimizer(max_iter=100, history_size=10)
        lbfgs_result = lbfgs_opt.optimize(loss_fn, init_params, data)
        
        # Adam
        adam_opt = JointOptimizer(method="adam", learning_rate=0.1, max_iter=100)
        adam_result = adam_opt.optimize(loss_fn, init_params, data)
        
        # L-BFGS should converge faster for smooth problems
        assert lbfgs_result.converged
        assert adam_result.converged or adam_result.loss < 0.1
        
        # L-BFGS typically needs fewer iterations
        if lbfgs_result.converged and adam_result.converged:
            assert lbfgs_result.n_iter < adam_result.n_iter
    
    def test_final_loss_comparison(self):
        """Compare final loss values."""
        def loss_fn(params, data):
            x = params["x"]
            return jnp.sum((x - jnp.array([1.0, 2.0, 3.0])) ** 2)
        
        init_params = {"x": jnp.array([0.0, 0.0, 0.0])}
        data = {}
        
        # L-BFGS
        lbfgs_result = lbfgs_optimize(
            loss_fn, init_params, data, max_iter=50
        )
        
        # Adam
        adam_opt = JointOptimizer(method="adam", learning_rate=0.1, max_iter=50)
        adam_result = adam_opt.optimize(loss_fn, init_params, data)
        
        # Both should achieve low loss
        assert lbfgs_result.loss < 0.1
        assert adam_result.loss < 0.1


class TestLBFGSRealWorld:
    """Test L-BFGS on realistic problems."""
    
    def test_linear_regression(self):
        """Test on linear regression problem."""
        # Generate data
        np.random.seed(42)
        n = 100
        x = np.random.normal(size=n)
        y = 2.0 + 3.0 * x + np.random.normal(scale=0.5, size=n)
        
        def loss_fn(params, data):
            pred = params["intercept"] + params["slope"] * data["x"]
            return jnp.mean((data["y"] - pred) ** 2)
        
        init_params = {
            "intercept": jnp.array(0.0),
            "slope": jnp.array(0.0)
        }
        data = {"x": jnp.array(x), "y": jnp.array(y)}
        
        optimizer = LBFGSOptimizer(max_iter=100)
        result = optimizer.optimize(loss_fn, init_params, data)
        
        # Check convergence and parameter estimates
        assert result.converged or result.loss < 0.5
        assert jnp.allclose(result.params["intercept"], 2.0, atol=0.3)
        assert jnp.allclose(result.params["slope"], 3.0, atol=0.3)
    
    def test_gamlss_style_problem(self):
        """Test on GAMLSS-style negative log-likelihood."""
        # Generate data
        np.random.seed(42)
        n = 100
        x = np.random.normal(size=n)
        y = 2.0 + 3.0 * x + np.random.normal(scale=1.0, size=n)
        
        def loss_fn(params, data):
            # Normal distribution NLL
            mu = params["mu_intercept"] + params["mu_slope"] * data["x"]
            log_sigma = params["log_sigma"]
            sigma = jnp.exp(log_sigma)
            
            residuals = (data["y"] - mu) / sigma
            nll = 0.5 * jnp.log(2 * jnp.pi) + log_sigma + 0.5 * residuals ** 2
            return jnp.mean(nll)
        
        init_params = {
            "mu_intercept": jnp.array(0.0),
            "mu_slope": jnp.array(0.0),
            "log_sigma": jnp.array(0.0)
        }
        data = {"x": jnp.array(x), "y": jnp.array(y)}
        
        optimizer = LBFGSOptimizer(max_iter=200, history_size=15)
        result = optimizer.optimize(loss_fn, init_params, data)
        
        # Should converge
        assert result.converged or result.loss < 2.0
        
        # Parameters should be reasonable
        assert jnp.abs(result.params["mu_intercept"] - 2.0) < 1.0
        assert jnp.abs(result.params["mu_slope"] - 3.0) < 1.0
    
    def test_rosenbrock_function(self):
        """Test on Rosenbrock function (classic optimization benchmark)."""
        def loss_fn(params, data):
            # Rosenbrock function: (1-x)^2 + 100(y-x^2)^2
            x = params["x"]
            y = params["y"]
            return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
        
        init_params = {
            "x": jnp.array(-1.0),
            "y": jnp.array(1.0)
        }
        data = {}
        
        optimizer = LBFGSOptimizer(max_iter=200, history_size=20)
        result = optimizer.optimize(loss_fn, init_params, data)
        
        # Rosenbrock minimum is at (1, 1)
        assert result.loss < 0.1
        assert jnp.allclose(result.params["x"], 1.0, atol=0.1)
        assert jnp.allclose(result.params["y"], 1.0, atol=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
