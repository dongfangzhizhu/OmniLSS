"""Tests for device management module.

This module tests the device management functionality including:
- Device selection (auto, CPU, GPU, TPU)
- Device information queries
- Data movement between devices
- Global device management
- Multi-device support
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from omnilss.core.device import (
    DeviceManager,
    DeviceInfo,
    set_device,
    get_device_manager,
    get_device,
    get_device_info,
    to_device,
    to_host,
    list_devices,
    print_device_info,
)


# =============================================================================
# Device Manager Tests
# =============================================================================

def test_device_manager_auto():
    """Test automatic device selection (defaults to CPU)."""
    dm = DeviceManager("auto")
    
    # Should default to CPU
    assert dm.device_type == "cpu"
    
    # Should have at least one device
    assert dm.get_device_count() > 0
    
    # Info should be populated
    info = dm.info
    assert isinstance(info, DeviceInfo)
    assert info.device_type == "cpu"
    assert info.device_count > 0


def test_device_manager_cpu():
    """Test CPU device selection."""
    dm = DeviceManager("cpu")
    
    assert dm.device_type == "cpu"
    assert dm.get_device_count() > 0
    
    # CPU should always be available
    info = dm.info
    assert info.device_type == "cpu"
    assert "cpu" in info.available_devices


@pytest.mark.skipif(
    not any(d.platform == "gpu" for d in jax.devices()),
    reason="GPU not available"
)
def test_device_manager_gpu():
    """Test GPU device selection (if available)."""
    dm = DeviceManager("gpu")
    
    assert dm.device_type == "gpu"
    assert dm.get_device_count() > 0
    
    info = dm.info
    assert info.device_type == "gpu"
    assert "gpu" in info.available_devices


def test_device_manager_invalid():
    """Test invalid device selection."""
    # Invalid device type should raise ValueError
    with pytest.raises(ValueError, match="Invalid device type"):
        DeviceManager("invalid_device")


def test_device_manager_unavailable():
    """Test requesting unavailable device."""
    # Try to request GPU if not available
    available_devices = [d.platform for d in jax.devices()]
    
    if "gpu" not in available_devices:
        # GPU not available, should raise error
        with pytest.raises((ValueError, RuntimeError)):
            DeviceManager("gpu")


# =============================================================================
# Data Movement Tests
# =============================================================================

def test_to_device_numpy():
    """Test moving numpy array to device."""
    dm = DeviceManager("cpu")
    
    # Create numpy array
    x_np = np.array([1.0, 2.0, 3.0])
    
    # Move to device
    x_device = dm.to_device(x_np)
    
    # Should be JAX array
    assert isinstance(x_device, jnp.ndarray)
    assert jnp.allclose(x_device, x_np)


def test_to_device_jax():
    """Test moving JAX array to device."""
    dm = DeviceManager("cpu")
    
    # Create JAX array
    x_jax = jnp.array([1.0, 2.0, 3.0])
    
    # Move to device
    x_device = dm.to_device(x_jax)
    
    # Should be JAX array
    assert isinstance(x_device, jnp.ndarray)
    assert jnp.allclose(x_device, x_jax)


def test_to_device_list():
    """Test moving Python list to device."""
    dm = DeviceManager("cpu")
    
    # Create list
    x_list = [1.0, 2.0, 3.0]
    
    # Move to device
    x_device = dm.to_device(x_list)
    
    # Should be JAX array
    assert isinstance(x_device, jnp.ndarray)
    assert jnp.allclose(x_device, jnp.array(x_list))


def test_to_host():
    """Test moving array from device to host."""
    dm = DeviceManager("cpu")
    
    # Create array on device
    x_device = dm.to_device(jnp.array([1.0, 2.0, 3.0]))
    
    # Move to host
    x_host = dm.to_host(x_device)
    
    # Should be numpy array
    assert isinstance(x_host, np.ndarray)
    assert np.allclose(x_host, np.array([1.0, 2.0, 3.0]))


def test_roundtrip():
    """Test roundtrip: numpy -> device -> host."""
    dm = DeviceManager("cpu")
    
    # Original numpy array
    x_original = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Roundtrip
    x_device = dm.to_device(x_original)
    x_back = dm.to_host(x_device)
    
    # Should be identical
    assert isinstance(x_back, np.ndarray)
    assert np.allclose(x_back, x_original)


# =============================================================================
# Device Info Tests
# =============================================================================

def test_device_info_structure():
    """Test DeviceInfo structure."""
    dm = DeviceManager("auto")
    info = dm.info
    
    # Check all fields are present
    assert hasattr(info, "device_type")
    assert hasattr(info, "device_count")
    assert hasattr(info, "available_devices")
    assert hasattr(info, "current_device")
    assert hasattr(info, "platform")
    
    # Check types
    assert isinstance(info.device_type, str)
    assert isinstance(info.device_count, int)
    assert isinstance(info.available_devices, list)
    assert info.device_count > 0


def test_device_info_str():
    """Test DeviceInfo string representation."""
    dm = DeviceManager("cpu")
    info = dm.info
    
    # Should have readable string representation
    info_str = str(info)
    assert "Device Type" in info_str
    assert "cpu" in info_str.lower()
    assert "Device Count" in info_str


def test_get_device_count():
    """Test device count query."""
    dm = DeviceManager("cpu")
    count = dm.get_device_count()
    
    assert isinstance(count, int)
    assert count > 0


def test_get_all_devices():
    """Test getting all devices of a type."""
    dm = DeviceManager("cpu")
    devices = dm.get_all_devices()
    
    assert isinstance(devices, list)
    assert len(devices) > 0
    assert len(devices) == dm.get_device_count()


# =============================================================================
# Global Device Management Tests
# =============================================================================

def test_set_device():
    """Test setting global device."""
    # Set to CPU (default)
    set_device("cpu")
    assert get_device() == "cpu"
    
    # Set to auto (should be CPU)
    set_device("auto")
    assert get_device() == "cpu"


def test_get_device_manager():
    """Test getting global device manager."""
    set_device("cpu")
    dm = get_device_manager()
    
    assert isinstance(dm, DeviceManager)
    assert dm.device_type == "cpu"


def test_get_device():
    """Test getting current device type."""
    set_device("cpu")
    device = get_device()
    
    assert isinstance(device, str)
    assert device == "cpu"


def test_get_device_info_global():
    """Test getting device info from global manager."""
    set_device("cpu")
    info = get_device_info()
    
    assert isinstance(info, DeviceInfo)
    assert info.device_type == "cpu"


def test_to_device_global():
    """Test global to_device function."""
    set_device("cpu")
    
    x = to_device([1.0, 2.0, 3.0])
    
    assert isinstance(x, jnp.ndarray)
    assert jnp.allclose(x, jnp.array([1.0, 2.0, 3.0]))


def test_to_host_global():
    """Test global to_host function."""
    set_device("cpu")
    
    x_device = to_device([1.0, 2.0, 3.0])
    x_host = to_host(x_device)
    
    assert isinstance(x_host, np.ndarray)
    assert np.allclose(x_host, np.array([1.0, 2.0, 3.0]))


# =============================================================================
# Utility Function Tests
# =============================================================================

def test_list_devices():
    """Test listing all devices."""
    devices = list_devices()
    
    assert isinstance(devices, dict)
    assert len(devices) > 0
    
    # CPU should always be available
    assert "cpu" in devices
    
    # Each entry should be a list
    for device_type, device_list in devices.items():
        assert isinstance(device_list, list)
        assert len(device_list) > 0


def test_print_device_info(capsys):
    """Test printing device information."""
    print_device_info()
    
    # Capture output
    captured = capsys.readouterr()
    
    # Should contain key information
    assert "OmniLSS Device Information" in captured.out
    assert "Current Device" in captured.out
    assert "Available Devices" in captured.out


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

def test_device_manager_repr():
    """Test DeviceManager string representation."""
    dm = DeviceManager("cpu")
    repr_str = repr(dm)
    
    assert "DeviceManager" in repr_str
    assert "cpu" in repr_str


def test_multiple_device_managers():
    """Test creating multiple device managers."""
    dm1 = DeviceManager("cpu")
    dm2 = DeviceManager("auto")
    
    # Should be independent
    assert dm1.device_type == "cpu"
    assert dm2.device_type in ["cpu", "gpu", "tpu"]


def test_device_manager_with_2d_array():
    """Test device operations with 2D arrays."""
    dm = DeviceManager("cpu")
    
    # 2D array
    x = np.array([[1, 2], [3, 4], [5, 6]])
    
    # Move to device and back
    x_device = dm.to_device(x)
    x_host = dm.to_host(x_device)
    
    assert x_host.shape == (3, 2)
    assert np.allclose(x_host, x)


def test_device_manager_with_scalar():
    """Test device operations with scalars."""
    dm = DeviceManager("cpu")
    
    # Scalar
    x = 42.0
    
    # Move to device
    x_device = dm.to_device(x)
    
    assert isinstance(x_device, jnp.ndarray)
    assert float(x_device) == 42.0


def test_device_manager_with_complex_structure():
    """Test device operations with nested structures."""
    dm = DeviceManager("cpu")
    
    # Create nested structure
    x = {
        "a": [1.0, 2.0, 3.0],
        "b": [[4.0, 5.0], [6.0, 7.0]]
    }
    
    # Move individual arrays
    x_device = {
        "a": dm.to_device(x["a"]),
        "b": dm.to_device(x["b"])
    }
    
    # Check types
    assert isinstance(x_device["a"], jnp.ndarray)
    assert isinstance(x_device["b"], jnp.ndarray)


# =============================================================================
# Integration Tests
# =============================================================================

def test_device_with_computation():
    """Test device management with actual computation."""
    dm = DeviceManager("cpu")
    
    # Create data on device
    x = dm.to_device(jnp.array([1.0, 2.0, 3.0]))
    y = dm.to_device(jnp.array([4.0, 5.0, 6.0]))
    
    # Perform computation
    z = x + y
    
    # Move result to host
    z_host = dm.to_host(z)
    
    assert np.allclose(z_host, np.array([5.0, 7.0, 9.0]))


def test_device_with_jit():
    """Test device management with JIT compilation."""
    dm = DeviceManager("cpu")
    
    # Define JIT-compiled function
    @jax.jit
    def compute(x, y):
        return x ** 2 + y ** 2
    
    # Create data on device
    x = dm.to_device(jnp.array([1.0, 2.0, 3.0]))
    y = dm.to_device(jnp.array([4.0, 5.0, 6.0]))
    
    # Compute
    result = compute(x, y)
    
    # Move to host
    result_host = dm.to_host(result)
    
    expected = np.array([17.0, 29.0, 45.0])  # 1^2+4^2, 2^2+5^2, 3^2+6^2
    assert np.allclose(result_host, expected)


@pytest.mark.skipif(
    not any(d.platform == "gpu" for d in jax.devices()),
    reason="GPU not available"
)
def test_gpu_acceleration():
    """Test GPU acceleration (if available)."""
    # Create GPU device manager
    dm_gpu = DeviceManager("gpu")
    
    # Large array for meaningful GPU test
    n = 10000
    x = dm_gpu.to_device(jnp.ones(n))
    y = dm_gpu.to_device(jnp.ones(n))
    
    # Computation should run on GPU
    z = x + y
    
    # Verify result
    z_host = dm_gpu.to_host(z)
    assert np.allclose(z_host, 2.0 * np.ones(n))


# =============================================================================
# Performance Tests (Optional)
# =============================================================================

@pytest.mark.slow
def test_device_performance():
    """Test device performance characteristics."""
    dm = DeviceManager("cpu")
    
    # Create large array
    n = 100000
    x = np.random.randn(n)
    
    # Time data movement
    import time
    
    start = time.time()
    x_device = dm.to_device(x)
    to_device_time = time.time() - start
    
    start = time.time()
    x_host = dm.to_host(x_device)
    to_host_time = time.time() - start
    
    # Should be reasonably fast (< 1 second for 100k elements)
    assert to_device_time < 1.0
    assert to_host_time < 1.0
    
    # Data should be preserved
    assert np.allclose(x_host, x)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
