"""Tests for cold/hot benchmark helper behavior."""

from __future__ import annotations

import jax.numpy as jnp
from unittest.mock import patch

from benchmarks.comprehensive_performance_test import benchmark_jax, honest_benchmark


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


def test_honest_benchmark_reports_cold_hot_fields_and_note():
    class _FakeModel:
        def __init__(self):
            self.g_dev = 1.234
            self.fitted_values = {"mu": jnp.array([0.0])}

    with (
        patch("benchmarks.comprehensive_performance_test.benchmark_jax", return_value=(0.12, 0.03)),
        patch("benchmarks.comprehensive_performance_test.gamlss", return_value=_FakeModel()),
    ):
        out = honest_benchmark("NO", n=32, n_warmup=2, n_runs=5)

    assert out["family"] == "NO"
    assert out["n"] == 32
    assert out["cold_s"] == 0.12
    assert out["median_s"] == 0.03
    assert out["runs"] == 5
    assert out["warmup_runs"] == 2
    assert isinstance(out["deviance"], float)
    assert "cold/hot separated" in str(out["note"])


def test_benchmark_jax_call_count_matches_cold_warm_hot_contract():
    calls = {"n": 0}

    def fn(x):
        calls["n"] += 1
        return jnp.sum(x)

    n_warmup = 2
    n_repeat = 4
    benchmark_jax(fn, jnp.ones((8,)), n_warmup=n_warmup, n_repeat=n_repeat)
    assert calls["n"] == 1 + n_warmup + n_repeat
