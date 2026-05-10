"""Memory optimization utilities for GAMLSS fitting.

This module provides utilities to reduce memory usage and minimize
numpy ↔ JAX array conversions during model fitting.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

import jax.numpy as jnp
import numpy as np


class ArrayCache:
    """Cache for array conversions to minimize numpy ↔ JAX conversions.
    
    This class maintains both numpy and JAX versions of arrays,
    converting only when necessary.
    
    Examples
    --------
    >>> cache = ArrayCache()
    >>> cache.set_numpy("y", np.array([1, 2, 3]))
    >>> y_jax = cache.get_jax("y")  # Converts to JAX
    >>> y_jax2 = cache.get_jax("y")  # Returns cached version
    """
    
    def __init__(self):
        self._numpy_cache: Dict[str, np.ndarray] = {}
        self._jax_cache: Dict[str, jnp.ndarray] = {}
    
    def set_numpy(self, key: str, array: np.ndarray):
        """Store a numpy array.
        
        Parameters
        ----------
        key : str
            Key to store the array under
        array : np.ndarray
            Numpy array to store
        """
        self._numpy_cache[key] = array
        # Invalidate JAX cache for this key
        if key in self._jax_cache:
            del self._jax_cache[key]
    
    def set_jax(self, key: str, array: jnp.ndarray):
        """Store a JAX array.
        
        Parameters
        ----------
        key : str
            Key to store the array under
        array : jnp.ndarray
            JAX array to store
        """
        self._jax_cache[key] = array
        # Invalidate numpy cache for this key
        if key in self._numpy_cache:
            del self._numpy_cache[key]
    
    def get_numpy(self, key: str) -> np.ndarray:
        """Get a numpy array, converting from JAX if necessary.
        
        Parameters
        ----------
        key : str
            Key of the array to retrieve
            
        Returns
        -------
        np.ndarray
            Numpy array
        """
        if key in self._numpy_cache:
            return self._numpy_cache[key]
        elif key in self._jax_cache:
            # Convert from JAX to numpy and cache
            array = np.asarray(self._jax_cache[key], dtype=np.float64)
            self._numpy_cache[key] = array
            return array
        else:
            raise KeyError(f"Array '{key}' not found in cache")
    
    def get_jax(self, key: str) -> jnp.ndarray:
        """Get a JAX array, converting from numpy if necessary.
        
        Parameters
        ----------
        key : str
            Key of the array to retrieve
            
        Returns
        -------
        jnp.ndarray
            JAX array
        """
        if key in self._jax_cache:
            return self._jax_cache[key]
        elif key in self._numpy_cache:
            # Convert from numpy to JAX and cache
            array = jnp.asarray(self._numpy_cache[key], dtype=jnp.float64)
            self._jax_cache[key] = array
            return array
        else:
            raise KeyError(f"Array '{key}' not found in cache")
    
    def has(self, key: str) -> bool:
        """Check if an array is in the cache.
        
        Parameters
        ----------
        key : str
            Key to check
            
        Returns
        -------
        bool
            True if the key exists in either cache
        """
        return key in self._numpy_cache or key in self._jax_cache
    
    def clear(self):
        """Clear all cached arrays."""
        self._numpy_cache.clear()
        self._jax_cache.clear()
    
    def memory_usage(self) -> Dict[str, int]:
        """Get memory usage statistics.
        
        Returns
        -------
        dict
            Dictionary with memory usage in bytes
        """
        numpy_bytes = sum(arr.nbytes for arr in self._numpy_cache.values())
        jax_bytes = sum(arr.nbytes for arr in self._jax_cache.values())
        
        return {
            "numpy_arrays": len(self._numpy_cache),
            "jax_arrays": len(self._jax_cache),
            "numpy_bytes": numpy_bytes,
            "jax_bytes": jax_bytes,
            "total_bytes": numpy_bytes + jax_bytes,
        }


def convert_data_to_jax(
    data: Mapping[str, Any],
    keys: list[str] | None = None,
) -> Dict[str, jnp.ndarray]:
    """Convert data dictionary to JAX arrays.
    
    Parameters
    ----------
    data : Mapping[str, Any]
        Input data dictionary
    keys : list[str], optional
        Keys to convert. If None, convert all keys.
        
    Returns
    -------
    dict
        Dictionary with JAX arrays
    """
    if keys is None:
        keys = list(data.keys())
    
    jax_data = {}
    for key in keys:
        if key in data:
            value = data[key]
            if isinstance(value, (np.ndarray, list, tuple)):
                jax_data[key] = jnp.asarray(value, dtype=jnp.float64)
            else:
                jax_data[key] = value
    
    return jax_data


def minimize_conversions(
    arrays: Dict[str, np.ndarray],
    use_jax: bool = True,
) -> Dict[str, np.ndarray | jnp.ndarray]:
    """Minimize array conversions by converting once at the start.
    
    Parameters
    ----------
    arrays : dict
        Dictionary of numpy arrays
    use_jax : bool, default True
        If True, convert to JAX arrays. Otherwise, keep as numpy.
        
    Returns
    -------
    dict
        Dictionary of arrays (JAX if use_jax=True, numpy otherwise)
    """
    if use_jax:
        return {
            key: jnp.asarray(arr, dtype=jnp.float64)
            for key, arr in arrays.items()
        }
    else:
        return {
            key: np.asarray(arr, dtype=np.float64)
            for key, arr in arrays.items()
        }


class MemoryOptimizer:
    """Memory optimizer for GAMLSS fitting.
    
    This class tracks memory usage and provides recommendations
    for optimization.
    
    Attributes
    ----------
    conversions : int
        Number of numpy ↔ JAX conversions
    numpy_to_jax : int
        Number of numpy → JAX conversions
    jax_to_numpy : int
        Number of JAX → numpy conversions
    """
    
    def __init__(self):
        self.conversions = 0
        self.numpy_to_jax = 0
        self.jax_to_numpy = 0
        self._conversion_log = []
    
    def log_conversion(self, from_type: str, to_type: str, size: int):
        """Log an array conversion.
        
        Parameters
        ----------
        from_type : str
            Source type ('numpy' or 'jax')
        to_type : str
            Target type ('numpy' or 'jax')
        size : int
            Size of the array in bytes
        """
        self.conversions += 1
        
        if from_type == 'numpy' and to_type == 'jax':
            self.numpy_to_jax += 1
        elif from_type == 'jax' and to_type == 'numpy':
            self.jax_to_numpy += 1
        
        self._conversion_log.append({
            'from': from_type,
            'to': to_type,
            'size': size,
        })
    
    def summary(self) -> Dict[str, Any]:
        """Get conversion summary.
        
        Returns
        -------
        dict
            Summary statistics
        """
        total_bytes = sum(log['size'] for log in self._conversion_log)
        
        return {
            'total_conversions': self.conversions,
            'numpy_to_jax': self.numpy_to_jax,
            'jax_to_numpy': self.jax_to_numpy,
            'total_bytes_converted': total_bytes,
            'avg_bytes_per_conversion': total_bytes / max(self.conversions, 1),
        }
    
    def print_summary(self):
        """Print conversion summary."""
        summary = self.summary()
        
        print("\n" + "=" * 60)
        print("Memory Optimization Summary")
        print("=" * 60)
        print(f"Total Conversions: {summary['total_conversions']}")
        print(f"  numpy → JAX: {summary['numpy_to_jax']}")
        print(f"  JAX → numpy: {summary['jax_to_numpy']}")
        print(f"Total Bytes Converted: {summary['total_bytes_converted']:,}")
        print(f"Avg Bytes/Conversion: {summary['avg_bytes_per_conversion']:,.0f}")
        
        if self.conversions > 10:
            print("\n⚠ Warning: High number of conversions detected!")
            print("  Consider using ArrayCache or converting arrays once at the start.")
        
        print("=" * 60)
    
    def reset(self):
        """Reset all counters."""
        self.conversions = 0
        self.numpy_to_jax = 0
        self.jax_to_numpy = 0
        self._conversion_log.clear()


# Global memory optimizer instance
_memory_optimizer = MemoryOptimizer()


def get_memory_optimizer() -> MemoryOptimizer:
    """Get the global memory optimizer instance.
    
    Returns
    -------
    MemoryOptimizer
        Global memory optimizer
    """
    return _memory_optimizer
