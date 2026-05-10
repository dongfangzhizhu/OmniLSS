"""Tests for joint optimizer module.

This module tests the joint optimization functionality, including:
- Basic optimization with different methods
- Convergence detection
- Parameter updates
- Loss history tracking
"""

import pytest
import jax.numpy as jnp
import numpy as np
from omnilss.core.optimizer import (
    JointOptimizer,
    create_optimizer,
    joint_optimize,
    OptimizationResult,
)


class TestJointOptimizer:
    """Test suite for JointOptimizer class."""
    
    def test_basic_optimization(self):
        """Test basic optimization with Adam."""
        # Simple quadratic loss: (x - target)^2
        def loss_fn(params, data):
            x = params["x"]
            return jnp.sum((x - data["target"]) ** 2)
        
        # Initial parameters
        init_params = {"x": jnp.array([0.0, 0.0])}
        data = {"target": jnp.array([1.0, 2.0])}
        
        # Optimize
        optimizer = JointOptimizer(
            method="adam",
            learning_rate=0.1,
            max_iter=100,
            tol=1e-6
        )
        result = optimizer.optimize(loss_fn, init_params, data)
        
        # Check convergence
        assert result.converged, "Optimization should converge"
        assert result.loss < 1e-3, f"Loss should be near zero, got {result.loss}"
        assert jnp.allclose(
            result.params["x"],
            data["target"],
            atol=0.05
        ), "Parameters should match target"
    
    def test_different_optimizers(self):
        """Test different optimization methods."""
        methods = ["adam", "sgd", "rmsprop", "adagrad"]
        
        def loss_fn(params, data):
            return jnp.sum(params["x"] ** 2)
        
        init_params = {"x": jnp.array([1.0, 2.0])}
        data = {}
        
        for method in methods:
            optimizer = create_optimizer(
                method=method,
                learning_rate=0.1,
                max_iter=200
            )
            result = optimizer.optimize(loss_fn, init_params, data)
            
            # All methods should reduce loss significantly
            assert result.loss < 0.1, f"{method} should reduce loss"
            assert len(result.loss_history) > 0, "Should track loss history"
    
    def test_convergence_detection(self):
        """Test convergence detection based on loss change."""
        def loss_fn(params, data):
            return jnp.sum(params["x"] ** 2)
        
        init_params = {"x": jnp.array([0.1, 0.1])}
        data = {}
        
        optimizer = JointOptimizer(
            method="adam",
            learning_rate=0.5,
            max_iter=1000,
            tol=1e-6
        )
        result = optimizer.optimize(loss_fn, init_params, data)
        
        assert result.converged, "Should converge"
        assert result.n_iter < 100, "Should converge quickly"
        assert result.loss < 1e-4, "Should reach near-zero loss"
    
    def test_gradient_norm_tracking(self):
        """Test gradient norm tracking."""
        def loss_fn(params, data):
            return jnp.sum(params["x"] ** 2)
        
        init_params = {"x": jnp.array([1.0, 1.0])}
        data = {}
        
        optimizer = JointOptimizer(
            method="adam",
            learning_rate=0.1,
            max_iter=50,
            track_grad_norms=True
        )
        result = optimizer.optimize(loss_fn, init_params, data)
        
        assert result.grad_norms is not None, "Should track gradient norms"
        assert len(result.grad_norms) == len(result.loss_history)
        # Gradient norms should decrease
        assert result.grad_norms[-1] < result.grad_norms[0]
    
    def test_multiple_parameters(self):
        """Test optimization with multiple parameter groups."""
        def loss_fn(params, data):
            # Loss: (mu - 1)^2 + (sigma - 2)^2
            mu_loss = (params["mu"] - 1.0) ** 2
            sigma_loss = (params["sigma"] - 2.0) ** 2
            return mu_loss + sigma_loss
        
        init_params = {
            "mu": jnp.array(0.0),
            "sigma": jnp.array(0.0)
        }
        data = {}
        
        optimizer = JointOptimizer(
            method="adam",
            learning_rate=0.1,
            max_iter=200
        )
        result = optimizer.optimize(loss_fn, init_params, data)
        
        assert result.converged or result.loss < 1e-3
        assert jnp.allclose(result.params["mu"], 1.0, atol=1e-1)
        assert jnp.allclose(result.params["sigma"], 2.0, atol=1e-1)
    
    def test_non_convergence_warning(self):
        """Test warning when optimization doesn't converge."""
        def loss_fn(params, data):
            # Difficult loss landscape
            return jnp.sum(jnp.sin(params["x"]) ** 2)
        
        init_params = {"x": jnp.array([10.0])}
        data = {}
        
        optimizer = JointOptimizer(
            method="sgd",
            learning_rate=0.001,
            max_iter=10,  # Too few iterations
            tol=1e-10  # Very strict tolerance
        )
        
        with pytest.warns(RuntimeWarning, match="did not converge"):
            result = optimizer.optimize(loss_fn, init_params, data)
        
        assert not result.converged
        assert result.n_iter == 10
    
    def test_loss_history(self):
        """Test that loss history is properly tracked."""
        def loss_fn(params, data):
            return jnp.sum(params["x"] ** 2)
        
        init_params = {"x": jnp.array([1.0, 1.0])}
        data = {}
        
        optimizer = JointOptimizer(
            method="adam",
            learning_rate=0.1,
            max_iter=50
        )
        result = optimizer.optimize(loss_fn, init_params, data)
        
        assert len(result.loss_history) > 0
        # Loss should be monotonically decreasing (mostly)
        assert result.loss_history[-1] < result.loss_history[0]
    
    def test_invalid_method(self):
        """Test error handling for invalid optimization method."""
        with pytest.raises(ValueError, match="Unknown optimization method"):
            JointOptimizer(method="invalid_method")
    
    def test_create_optimizer_factory(self):
        """Test optimizer factory function."""
        # Test different methods
        opt_adam = create_optimizer("adam", learning_rate=0.01)
        assert isinstance(opt_adam, JointOptimizer)
        assert opt_adam.method == "adam"
        
        opt_sgd = create_optimizer("sgd", learning_rate=0.1)
        assert isinstance(opt_sgd, JointOptimizer)
        assert opt_sgd.method == "sgd"
    
    def test_joint_optimize_convenience(self):
        """Test convenience function for joint optimization."""
        def loss_fn(params, data):
            return jnp.sum((params["x"] - data["target"]) ** 2)
        
        init_params = {"x": jnp.array([0.0, 0.0])}
        data = {"target": jnp.array([1.0, 2.0])}
        
        result = joint_optimize(
            loss_fn=loss_fn,
            init_params=init_params,
            data=data,
            method="adam",
            learning_rate=0.1,
            max_iter=100
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.loss < 1e-3
    
    def test_optimization_result_repr(self):
        """Test OptimizationResult string representation."""
        result = OptimizationResult(
            params={"x": jnp.array([1.0])},
            loss=0.123456,
            n_iter=50,
            converged=True,
            loss_history=[1.0, 0.5, 0.123456]
        )
        
        repr_str = repr(result)
        assert "0.123456" in repr_str
        assert "50" in repr_str
        assert "converged" in repr_str


class TestOptimizationPerformance:
    """Test optimization performance and convergence speed."""
    
    def test_convergence_speed_comparison(self):
        """Compare convergence speed of different optimizers."""
        def loss_fn(params, data):
            return jnp.sum(params["x"] ** 2)
        
        init_params = {"x": jnp.array([1.0, 1.0, 1.0])}
        data = {}
        
        methods = ["adam", "sgd", "rmsprop"]
        results = {}
        
        for method in methods:
            optimizer = JointOptimizer(
                method=method,
                learning_rate=0.1,
                max_iter=200,
                tol=1e-6
            )
            result = optimizer.optimize(loss_fn, init_params, data)
            results[method] = result
        
        # Adam should typically converge fastest
        assert results["adam"].converged
        # All should achieve low loss
        for method, result in results.items():
            assert result.loss < 0.1, f"{method} should achieve low loss"
    
    def test_learning_rate_sensitivity(self):
        """Test sensitivity to learning rate."""
        def loss_fn(params, data):
            return jnp.sum(params["x"] ** 2)
        
        init_params = {"x": jnp.array([1.0])}
        data = {}
        
        learning_rates = [0.01, 0.1, 0.5]  # 移除太小的学习率
        
        for lr in learning_rates:
            optimizer = JointOptimizer(
                method="adam",
                learning_rate=lr,
                max_iter=100
            )
            result = optimizer.optimize(loss_fn, init_params, data)
            
            # All learning rates should work for this simple problem
            assert result.loss < 0.5, f"lr={lr} should reduce loss, got {result.loss}"


class TestRealWorldScenarios:
    """Test optimizer on realistic GAMLSS-like problems."""
    
    def test_linear_regression(self):
        """Test on simple linear regression problem."""
        # Generate data: y = 2 + 3*x + noise
        np.random.seed(42)
        n = 100
        x = np.random.normal(size=n)
        y = 2.0 + 3.0 * x + np.random.normal(scale=0.5, size=n)
        
        def loss_fn(params, data):
            # MSE loss
            pred = params["intercept"] + params["slope"] * data["x"]
            return jnp.mean((data["y"] - pred) ** 2)
        
        init_params = {
            "intercept": jnp.array(0.0),
            "slope": jnp.array(0.0)
        }
        data = {"x": jnp.array(x), "y": jnp.array(y)}
        
        optimizer = JointOptimizer(
            method="adam",
            learning_rate=0.1,
            max_iter=500
        )
        result = optimizer.optimize(loss_fn, init_params, data)
        
        # Check parameter estimates
        assert jnp.allclose(result.params["intercept"], 2.0, atol=0.2)
        assert jnp.allclose(result.params["slope"], 3.0, atol=0.2)
    
    def test_heteroscedastic_regression(self):
        """Test on heteroscedastic regression (varying variance)."""
        # Generate data with varying variance
        np.random.seed(42)
        n = 100
        x = np.random.normal(size=n)
        # Variance increases with x
        sigma = 0.5 + 0.5 * np.abs(x)
        y = 2.0 + 3.0 * x + np.random.normal(scale=sigma, size=n)
        
        def loss_fn(params, data):
            # Negative log-likelihood for normal distribution
            mu = params["mu_intercept"] + params["mu_slope"] * data["x"]
            log_sigma = params["log_sigma_intercept"] + params["log_sigma_slope"] * jnp.abs(data["x"])
            sigma = jnp.exp(log_sigma)
            
            # NLL
            nll = 0.5 * jnp.log(2 * jnp.pi) + log_sigma + 0.5 * ((data["y"] - mu) / sigma) ** 2
            return jnp.mean(nll)
        
        init_params = {
            "mu_intercept": jnp.array(0.0),
            "mu_slope": jnp.array(0.0),
            "log_sigma_intercept": jnp.array(0.0),
            "log_sigma_slope": jnp.array(0.0)
        }
        data = {"x": jnp.array(x), "y": jnp.array(y)}
        
        optimizer = JointOptimizer(
            method="adam",
            learning_rate=0.05,
            max_iter=1000
        )
        result = optimizer.optimize(loss_fn, init_params, data)
        
        # Should converge
        assert result.converged or result.loss < 2.0
        # Mean parameters should be reasonable
        assert jnp.abs(result.params["mu_intercept"] - 2.0) < 1.0
        assert jnp.abs(result.params["mu_slope"] - 3.0) < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
