"""Device management for OmniLSS.

This module provides automatic device selection and management for
CPU, GPU, and TPU acceleration. It leverages JAX's device abstraction
to provide seamless hardware acceleration.

Key Features:
- Automatic device detection and selection
- Manual device override
- Device information and diagnostics
- Memory management utilities
- Multi-device support

Examples
--------
Automatic device selection:

>>> from omnilss.core import DeviceManager
>>> 
>>> # Auto-select best available device
>>> dm = DeviceManager("auto")
>>> print(dm.device_type)  # "gpu", "tpu", or "cpu"
>>> 
>>> # Get device info
>>> info = dm.info
>>> print(f"Device: {info.device_type}")
>>> print(f"Count: {info.device_count}")

Manual device selection:

>>> # Force CPU
>>> dm = DeviceManager("cpu")
>>> 
>>> # Force GPU
>>> dm = DeviceManager("gpu")

Move data to device:

>>> import jax.numpy as jnp
>>> 
>>> # Create array on device
>>> x = dm.to_device(jnp.array([1, 2, 3]))
>>> print(x.device())

Global device management:

>>> from omnilss.core.device import set_device, get_device_info
>>> 
>>> # Set global device
>>> set_device("gpu")
>>> 
>>> # Get info
>>> info = get_device_info()
>>> print(info)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Any, List
import warnings

import jax
import jax.numpy as jnp
import numpy as np


# Type alias for device types
DeviceType = Literal["cpu", "gpu", "tpu", "auto"]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(frozen=True)
class DeviceInfo:
    """Information about available devices.
    
    Attributes
    ----------
    device_type : str
        Current device type ("cpu", "gpu", or "tpu")
    device_count : int
        Number of devices of this type
    available_devices : list[str]
        List of all available device types
    current_device : Any
        JAX device object
    platform : str
        Platform name (e.g., "NVIDIA GPU", "CPU")
    memory_info : dict, optional
        Memory information if available
    """
    device_type: str
    device_count: int
    available_devices: List[str]
    current_device: Any
    platform: str
    memory_info: Optional[dict] = None
    
    def __str__(self) -> str:
        lines = [
            f"Device Type: {self.device_type}",
            f"Device Count: {self.device_count}",
            f"Platform: {self.platform}",
            f"Available: {', '.join(self.available_devices)}",
        ]
        if self.memory_info:
            lines.append(f"Memory: {self.memory_info}")
        return "\n".join(lines)


# =============================================================================
# Device Manager
# =============================================================================

class DeviceManager:
    """Device manager for automatic hardware acceleration.
    
    This class handles device selection, data placement, and device
    information queries. It provides a simple interface for leveraging
    GPU acceleration in OmniLSS when beneficial.
    
    **Important**: By default, CPU is used because:
    - For small datasets (< 10k samples), CPU is often faster
    - GPU has data transfer overhead
    - CPU is universally available
    
    Users should explicitly request GPU for large-scale problems where
    the computational benefits outweigh transfer costs.
    
    Parameters
    ----------
    device : DeviceType, default="auto"
        Device type to use:
        - "auto": Use CPU (default, best for most cases)
        - "cpu": Force CPU
        - "gpu": Force GPU (recommended for large datasets > 100k samples)
        - "tpu": Force TPU (not commonly used)
    
    Attributes
    ----------
    device_type : str
        Selected device type
    
    Examples
    --------
    Default usage (CPU):
    
    >>> dm = DeviceManager()  # or DeviceManager("auto")
    >>> print(dm.device_type)
    'cpu'
    
    Explicit GPU for large datasets:
    
    >>> dm = DeviceManager("gpu")
    >>> print(dm.device_type)
    'gpu'
    
    >>> # Move array to device
    >>> x = dm.to_device(jnp.array([1, 2, 3]))
    >>> print(x.device())
    """
    
    def __init__(self, device: DeviceType = "auto"):
        self.device_type = self._select_device(device)
        self._current_device = self._get_device()
        
        # Log device selection only if not default CPU
        if device != "auto" or self.device_type != "cpu":
            print(f"Using device: {self.device_type}")
    
    def _select_device(self, device: DeviceType) -> str:
        """Select device based on availability and preference.
        
        Parameters
        ----------
        device : DeviceType
            Requested device type
            
        Returns
        -------
        str
            Selected device type
            
        Raises
        ------
        ValueError
            If requested device is not available
            
        Notes
        -----
        When device="auto", CPU is selected by default. This is because:
        - For small datasets (< 10k samples), CPU is often faster due to 
          GPU transfer overhead
        - CPU is universally available
        - Users can explicitly request GPU when beneficial
        """
        if device == "auto":
            # Default to CPU for better performance on small datasets
            # Users should explicitly request GPU for large-scale problems
            return "cpu"
        
        # Manual selection
        if device in ["cpu", "gpu", "tpu"]:
            # Check if device is available
            try:
                devices = jax.devices(device)
                if len(devices) == 0:
                    raise ValueError(
                        f"Device '{device}' is not available. "
                        f"Available devices: {self._get_available_devices()}"
                    )
            except RuntimeError as e:
                # On some platforms (e.g., Windows), JAX raises RuntimeError
                # for unavailable backends
                raise ValueError(
                    f"Device '{device}' is not available: {str(e)}. "
                    f"Available devices: {self._get_available_devices()}"
                )
            return device
        
        raise ValueError(
            f"Invalid device type: {device}. "
            f"Must be one of: 'auto', 'cpu', 'gpu', 'tpu'"
        )
    
    def _get_device(self) -> Any:
        """Get JAX device object.
        
        Returns
        -------
        device
            JAX device object
        """
        devices = jax.devices(self.device_type)
        if len(devices) == 0:
            warnings.warn(
                f"No {self.device_type} devices found, falling back to default",
                RuntimeWarning
            )
            return jax.devices()[0]
        return devices[0]
    
    def _get_available_devices(self) -> List[str]:
        """Get list of available device types.
        
        Returns
        -------
        list[str]
            List of available device types
        """
        all_devices = jax.devices()
        return list(set(d.platform for d in all_devices))
    
    def to_device(self, array: Any) -> Any:
        """Move array to the selected device.
        
        Parameters
        ----------
        array : array-like
            Array to move (numpy, jax, or python list)
            
        Returns
        -------
        jax.Array
            Array on the selected device
            
        Examples
        --------
        >>> dm = DeviceManager("gpu")
        >>> x = dm.to_device([1, 2, 3])
        >>> print(x.device())
        """
        # Convert to JAX array if needed
        if not isinstance(array, jnp.ndarray):
            array = jnp.asarray(array)
        
        # Move to device
        if self._current_device is not None:
            return jax.device_put(array, self._current_device)
        return array
    
    def to_host(self, array: Any) -> np.ndarray:
        """Move array from device to host (CPU) as numpy array.
        
        Parameters
        ----------
        array : jax.Array
            Array on device
            
        Returns
        -------
        np.ndarray
            Numpy array on CPU
            
        Examples
        --------
        >>> dm = DeviceManager("gpu")
        >>> x = dm.to_device([1, 2, 3])
        >>> y = dm.to_host(x)
        >>> type(y)
        <class 'numpy.ndarray'>
        """
        return np.asarray(array)
    
    @property
    def info(self) -> DeviceInfo:
        """Get device information.
        
        Returns
        -------
        DeviceInfo
            Device information
            
        Examples
        --------
        >>> dm = DeviceManager("auto")
        >>> info = dm.info
        >>> print(info)
        """
        all_devices = jax.devices()
        device_count = len(jax.devices(self.device_type))
        available = self._get_available_devices()
        
        # Get platform name
        platform = self._current_device.platform if self._current_device else "unknown"
        
        # Try to get memory info (GPU only)
        memory_info = None
        if self.device_type == "gpu":
            try:
                # This is platform-specific and may not work on all systems
                memory_info = {
                    "note": "Memory info not available through JAX API"
                }
            except Exception:
                pass
        
        return DeviceInfo(
            device_type=self.device_type,
            device_count=device_count,
            available_devices=available,
            current_device=self._current_device,
            platform=platform,
            memory_info=memory_info
        )
    
    def get_device_count(self) -> int:
        """Get number of devices of the selected type.
        
        Returns
        -------
        int
            Number of devices
        """
        return len(jax.devices(self.device_type))
    
    def get_all_devices(self) -> List[Any]:
        """Get all devices of the selected type.
        
        Returns
        -------
        list
            List of JAX device objects
        """
        return jax.devices(self.device_type)
    
    def __repr__(self) -> str:
        return f"DeviceManager(device_type='{self.device_type}')"


# =============================================================================
# Global Device Management
# =============================================================================

# Global device manager instance
_global_device_manager: Optional[DeviceManager] = None


def set_device(device: DeviceType = "auto") -> None:
    """Set global device for OmniLSS.
    
    This sets the default device for all OmniLSS operations.
    
    Parameters
    ----------
    device : DeviceType, default="auto"
        Device type to use
        
    Examples
    --------
    >>> from omnilss.core.device import set_device
    >>> 
    >>> # Use GPU
    >>> set_device("gpu")
    >>> 
    >>> # Auto-select
    >>> set_device("auto")
    """
    global _global_device_manager
    _global_device_manager = DeviceManager(device)


def get_device_manager() -> DeviceManager:
    """Get global device manager.
    
    Returns
    -------
    DeviceManager
        Global device manager
        
    Examples
    --------
    >>> from omnilss.core.device import get_device_manager
    >>> 
    >>> dm = get_device_manager()
    >>> print(dm.device_type)
    """
    global _global_device_manager
    if _global_device_manager is None:
        _global_device_manager = DeviceManager("auto")
    return _global_device_manager


def get_device() -> str:
    """Get current global device type.
    
    Returns
    -------
    str
        Device type ("cpu", "gpu", or "tpu")
        
    Examples
    --------
    >>> from omnilss.core.device import get_device
    >>> 
    >>> device = get_device()
    >>> print(device)
    'gpu'
    """
    return get_device_manager().device_type


def get_device_info() -> DeviceInfo:
    """Get information about current device.
    
    Returns
    -------
    DeviceInfo
        Device information
        
    Examples
    --------
    >>> from omnilss.core.device import get_device_info
    >>> 
    >>> info = get_device_info()
    >>> print(info)
    """
    return get_device_manager().info


def to_device(array: Any) -> Any:
    """Move array to current global device.
    
    Parameters
    ----------
    array : array-like
        Array to move
        
    Returns
    -------
    jax.Array
        Array on device
        
    Examples
    --------
    >>> from omnilss.core.device import to_device
    >>> import jax.numpy as jnp
    >>> 
    >>> x = to_device(jnp.array([1, 2, 3]))
    """
    return get_device_manager().to_device(array)


def to_host(array: Any) -> np.ndarray:
    """Move array from device to host.
    
    Parameters
    ----------
    array : jax.Array
        Array on device
        
    Returns
    -------
    np.ndarray
        Numpy array on CPU
        
    Examples
    --------
    >>> from omnilss.core.device import to_host
    >>> 
    >>> y = to_host(x)
    """
    return get_device_manager().to_host(array)


# =============================================================================
# Utility Functions
# =============================================================================

def list_devices() -> dict:
    """List all available devices.
    
    Returns
    -------
    dict
        Dictionary with device types as keys and device lists as values
        
    Examples
    --------
    >>> from omnilss.core.device import list_devices
    >>> 
    >>> devices = list_devices()
    >>> for device_type, device_list in devices.items():
    ...     print(f"{device_type}: {len(device_list)} devices")
    """
    all_devices = jax.devices()
    
    result = {}
    for device in all_devices:
        platform = device.platform
        if platform not in result:
            result[platform] = []
        result[platform].append(device)
    
    return result


def print_device_info() -> None:
    """Print detailed device information.
    
    Examples
    --------
    >>> from omnilss.core.device import print_device_info
    >>> 
    >>> print_device_info()
    """
    print("="*70)
    print("OmniLSS Device Information")
    print("="*70)
    
    # Current device
    info = get_device_info()
    print(f"\nCurrent Device: {info.device_type}")
    print(f"Platform: {info.platform}")
    print(f"Device Count: {info.device_count}")
    
    # All devices
    print(f"\nAvailable Devices:")
    devices = list_devices()
    for device_type, device_list in devices.items():
        print(f"  {device_type}: {len(device_list)} device(s)")
        for i, device in enumerate(device_list):
            print(f"    [{i}] {device}")
    
    print("="*70)


__all__ = [
    "DeviceType",
    "DeviceInfo",
    "DeviceManager",
    "set_device",
    "get_device_manager",
    "get_device",
    "get_device_info",
    "to_device",
    "to_host",
    "list_devices",
    "print_device_info",
]
