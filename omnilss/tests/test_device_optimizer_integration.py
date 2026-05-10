"""Integration tests for device management with optimizers.

This module tests the integration of device management with
the joint optimizer and L-BFGS optimizer.
"""

import pytest
import jax.numpy as jnp
import numpy as np

from omnilss.core import (
    JointOptimizer,
    LBFGSOptimizer,
    DeviceManager,
    set_device,
    get_device,
    to_device,
    to_host,
)


# =============================================================================
# Joint Optimizer + Device Tests
# =============================================================================

def test_joint_optimizer_with_device_manager():
    """Test JointOptimizer with explicit device management."""
    # Set up device
    dm = DeviceManager("cpu")
    
    # Define loss function
    def loss_fn(params, data):
        x = params["x"]
        target = data["target"]
        return jnp.sum((x - target) ** 2)
    
    # Initialize parameters on device
    init_params = {"x": dm.to_device(jnp.array([0.0, 0.0]))}
    data = {"target": dm.to_device(jnp.array([1.0, 2.0]))}
    
    # Optimize
    optimizer = JointOptimizer(method="adam", learning_rate=0.1, max_iter=100)
    result = optimizer.optimize(loss_fn, init_params, data)
    
    # Check convergence (relaxed tolerance for integration test)
    assert result.loss < 1e-3
    
    # Move result to host
    x_host = dm.to_host(result.params["x"])
    assert np.allclose(x_host, np.array([1.0, 2.0]), atol=0.02)


def test_joint_optimizer_with_global_device():
    """Test JointOptimizer with global device management."""
    # Set global device
    set_device("cpu")
    assert get_device() == "cpu"
    
    # Define loss function
    def loss_fn(params, data):
        return jnp.sum(params["x"] ** 2)
    
    # Initialize parameters using global functions
    init_params = {"x": to_device(jnp.array([1.0, 2.0, 3.0]))}
    data = {}
    
    # Optimize
    optimizer = JointOptimizer(method="adam", learning_rate=0.5, max_iter=200)
    result = optimizer.optimize(loss_fn, init_params, data)
    
    # Check convergence (relaxed for integration test)
    assert result.loss < 1e-3
    
    # Move result to host
    x_host = to_host(result.params["x"])
    assert np.allclose(x_host, np.zeros(3), atol=1e-2)


def test_joint_optimizer_multiple_params_with_device():
    """Test JointOptimizer with multiple parameters on device."""
    dm = DeviceManager("cpu")
    
    # Define loss function (simple linear regression)
    def loss_fn(params, data):
        mu = params["mu"]
        log_sigma = params["log_sigma"]
        sigma = jnp.exp(log_sigma)
        
        x = data["x"]
        y = data["y"]
        
        pred = mu * x
        residuals = (y - pred) / sigma
        nll = 0.5 * jnp.log(2 * jnp.pi) + log_sigma + 0.5 * residuals ** 2
        return jnp.mean(nll)
    
    # Generate synthetic data
    np.random.seed(42)
    n = 100
    x_data = np.random.randn(n)
    y_data = 2.0 * x_data + np.random.randn(n) * 0.1
    
    # Move to device
    init_params = {
        "mu": dm.to_device(jnp.array(0.0)),
        "log_sigma": dm.to_device(jnp.array(0.0))
    }
    data = {
        "x": dm.to_device(jnp.array(x_data)),
        "y": dm.to_device(jnp.array(y_data))
    }
    
    # Optimize
    optimizer = JointOptimizer(method="adam", learning_rate=0.1, max_iter=200)
    result = optimizer.optimize(loss_fn, init_params, data)
    
    # Check convergence
    assert result.converged
    
    # Check parameter estimates
    mu_est = float(dm.to_host(result.params["mu"]))
    assert abs(mu_est - 2.0) < 0.1  # Should be close to true value


# =============================================================================
# L-BFGS Optimizer + Device Tests
# =============================================================================

def test_lbfgs_optimizer_with_device_manager():
    """Test LBFGSOptimizer with explicit device management."""
    dm = DeviceManager("cpu")
    
    # Define loss function (quadratic)
    def loss_fn(params, data):
        x = params["x"]
        return jnp.sum(x ** 2)
    
    # Initialize parameters on device
    init_params = {"x": dm.to_device(jnp.array([1.0, 2.0, 3.0]))}
    data = {}
    
    # Optimize
    optimizer = LBFGSOptimizer(max_iter=50, history_size=10)
    result = optimizer.optimize(loss_fn, init_params, data)
    
    # Check convergence
    assert result.converged
    assert result.loss < 1e-6
    
    # Move result to host
    x_host = dm.to_host(result.params["x"])
    assert np.allclose(x_host, np.zeros(3), atol=1e-3)


def test_lbfgs_optimizer_with_global_device():
    """Test LBFGSOptimizer with global device management."""
    set_device("cpu")
    
    # Define loss function (Rosenbrock)
    def loss_fn(params, data):
        x = params["x"]
        return jnp.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    
    # Initialize parameters
    init_params = {"x": to_device(jnp.array([0.0, 0.0]))}
    data = {}
    
    # Optimize
    optimizer = LBFGSOptimizer(max_iter=100, history_size=10)
    result = optimizer.optimize(loss_fn, init_params, data)
    
    # Check convergence (Rosenbrock is harder, so more lenient)
    assert result.loss < 1e-2
    
    # Move result to host
    x_host = to_host(result.params["x"])
    # Rosenbrock minimum is at (1, 1)
    assert np.allclose(x_host, np.ones(2), atol=0.1)


# =============================================================================
# Performance Comparison Tests
# =============================================================================

def test_device_performance_comparison():
    """Compare performance with and without explicit device management."""
    # Define loss function
    def loss_fn(params, data):
        return jnp.sum((params["x"] - data["target"]) ** 2)
    
    # Test data
    n = 1000
    init_params = {"x": jnp.zeros(n)}
    data = {"target": jnp.ones(n)}
    
    # Test 1: Without explicit device management
    optimizer1 = JointOptimizer(method="adam", learning_rate=0.1, max_iter=50)
    result1 = optimizer1.optimize(loss_fn, init_params, data)
    
    # Test 2: With explicit device management
    dm = DeviceManager("cpu")
    init_params_device = {"x": dm.to_device(jnp.zeros(n))}
    data_device = {"target": dm.to_device(jnp.ones(n))}
    
    optimizer2 = JointOptimizer(method="adam", learning_rate=0.1, max_iter=50)
    result2 = optimizer2.optimize(loss_fn, init_params_device, data_device)
    
    # Both should converge to similar results
    assert abs(result1.loss - result2.loss) < 1e-4


# =============================================================================
# Edge Cases
# =============================================================================

def test_optimizer_with_complex_data_structure():
    """Test optimizer with nested data structure on device."""
    dm = DeviceManager("cpu")
    
    # Define loss function with nested structure
    def loss_fn(params, data):
        loss = 0.0
        for key in params:
            loss += jnp.sum((params[key] - data[key]) ** 2)
        return loss
    
    # Initialize nested parameters
    init_params = {
        "a": dm.to_device(jnp.array([0.0, 0.0])),
        "b": dm.to_device(jnp.array([0.0, 0.0, 0.0])),
    }
    data = {
        "a": dm.to_device(jnp.array([1.0, 2.0])),
        "b": dm.to_device(jnp.array([3.0, 4.0, 5.0])),
    }
    
    # Optimize
    optimizer = JointOptimizer(method="adam", learning_rate=0.1, max_iter=200)
    result = optimizer.optimize(loss_fn, init_params, data)
    
    # Check convergence (relaxed for integration test)
    assert result.loss < 1e-2


def test_optimizer_with_scalar_params():
    """Test optimizer with scalar parameters on device."""
    dm = DeviceManager("cpu")
    
    # Define loss function
    def loss_fn(params, data):
        return (params["a"] - 5.0) ** 2 + (params["b"] - 3.0) ** 2
    
    # Initialize scalar parameters
    init_params = {
        "a": dm.to_device(jnp.array(0.0)),
        "b": dm.to_device(jnp.array(0.0)),
    }
    data = {}
    
    # Optimize
    optimizer = JointOptimizer(method="adam", learning_rate=0.5, max_iter=200)
    result = optimizer.optimize(loss_fn, init_params, data)
    
    # Check convergence (relaxed for integration test)
    assert result.loss < 1e-2
    
    # Check values
    a_val = float(dm.to_host(result.params["a"]))
    b_val = float(dm.to_host(result.params["b"]))
    assert abs(a_val - 5.0) < 0.2
    assert abs(b_val - 3.0) < 0.2


# =============================================================================
# GPU Tests (if available)
# =============================================================================

@pytest.mark.skipif(
    not any(d.platform == "gpu" for d in __import__("jax").devices()),
    reason="GPU not available"
)
def test_joint_optimizer_on_gpu():
    """Test JointOptimizer on GPU (if available)."""
    dm = DeviceManager("gpu")
    
    # Define loss function
    def loss_fn(params, data):
        return jnp.sum(params["x"] ** 2)
    
    # Large array to benefit from GPU
    n = 10000
    init_params = {"x": dm.to_device(jnp.ones(n))}
    data = {}
    
    # Optimize
    optimizer = JointOptimizer(method="adam", learning_rate=0.1, max_iter=50)
    result = optimizer.optimize(loss_fn, init_params, data)
    
    # Check convergence
    assert result.converged
    assert result.loss < 1e-2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
