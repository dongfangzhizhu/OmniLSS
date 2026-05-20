"""Tests for cold/hot benchmark helper behavior."""

from __future__ import annotations

import jax.numpy as jnp

from benchmarks.comprehensive_performance_test import benchmark_jax


def test_benchmark_jax_returns_non_negative_cold_and_hot_times():
    def fn(x):
        return jnp.sum(x * x)

    cold_s, hot_median_s = benchmark_jax(
        fn,
        jnp.ones((256,)),
        n_warmup=1,
        n_repeat=3,
    )

    assert cold_s >= 0.0
    assert hot_median_s >= 0.0


def test_benchmark_jax_cold_not_less_than_hot_for_jitted_function():
    # For a fresh function call path, cold time should usually include
    # compilation overhead and be no smaller than steady-state hot median.
    def fn(x):
        return jnp.tanh(x).sum()

    cold_s, hot_median_s = benchmark_jax(
        fn,
        jnp.linspace(0.0, 1.0, 4096),
        n_warmup=2,
        n_repeat=3,
    )

    assert cold_s >= hot_median_s
