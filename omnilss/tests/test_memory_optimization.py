"""Phase 7.5: memory-optimisation cache thread safety.

ArrayCache (used by the global memory optimizer singleton) must be safe under
concurrent numpy/JAX cache reads and writes.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnilss.memory_optimization import ArrayCache, get_memory_optimizer


def test_cache_sequential_correctness():
    """set/get/has/clear behave correctly in sequence."""
    cache = ArrayCache()
    cache.set_numpy("a", np.array([1.0, 2.0]))
    assert cache.has("a")
    np.testing.assert_allclose(cache.get_numpy("a"), [1.0, 2.0])
    np.testing.assert_allclose(np.asarray(cache.get_jax("a")), [1.0, 2.0])
    # set_jax invalidates the numpy copy
    cache.set_jax("a", np.array([3.0, 4.0]))
    np.testing.assert_allclose(cache.get_numpy("a"), [3.0, 4.0])
    cache.clear()
    assert not cache.has("a")


def test_cache_concurrent_access_is_safe():
    """Concurrent set/get across threads raises no exceptions."""
    cache = ArrayCache()
    errors: list[Exception] = []

    def worker(i: int) -> None:
        try:
            for j in range(50):
                key = f"k{i % 4}"
                cache.set_numpy(key, np.array([i, j]))
                _ = cache.get_jax(key)
                _ = cache.get_numpy(key)
                _ = cache.has(key)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [], f"concurrent cache access raised: {errors[:3]}"


def test_global_optimizer_is_singleton():
    """get_memory_optimizer returns the same global instance."""
    assert get_memory_optimizer() is get_memory_optimizer()
