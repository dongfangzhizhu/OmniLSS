# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for omnilss.config and method='auto' selection."""

from __future__ import annotations

import math

import numpy as np
import pytest

import omnilss.config as cfg
from omnilss import gamlss, NO, GA, PO


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_no_data(n=200, seed=99):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = 2.0 + 1.5 * x + rng.standard_normal(n)
    return {"y": y, "x": x}


# ---------------------------------------------------------------------------
# config module unit tests
# ---------------------------------------------------------------------------

class TestConfigDefaults:
    def test_auto_method_enabled_default(self):
        assert cfg.AUTO_METHOD_ENABLED is True

    def test_force_jax_default(self):
        assert cfg.FORCE_JAX is False

    def test_cpu_crossover_never(self):
        assert cfg.CPU_CROSSOVER_N["default"] == math.inf

    def test_gpu_crossover_never_by_default(self):
        # All families should default to math.inf (no auto-switch)
        for fam in ["NO", "GA", "PO", "BI", "WEI", "TF"]:
            assert cfg.GPU_CROSSOVER_N.get(fam, math.inf) == math.inf

    def test_tpu_crossover_never_by_default(self):
        assert cfg.TPU_CROSSOVER_N["default"] == math.inf

    def test_jax_supported_families(self):
        assert "NO" in cfg.JAX_SUPPORTED_FAMILIES
        assert "GA" in cfg.JAX_SUPPORTED_FAMILIES
        assert "NBI" not in cfg.JAX_SUPPORTED_FAMILIES


class TestAutoSelectMethod:
    def test_auto_select_method_trace_cpu(self, monkeypatch):
        monkeypatch.setattr(cfg, "_current_backend", lambda: ("cpu", []))
        decision = cfg.auto_select_method_trace("NO", 12345)
        assert decision.method == "RS"
        assert decision.backend == "cpu"
        assert decision.reason == "cpu_backend_prefers_numpy_rs"

    def test_describe_method_routing_reason_known_and_unknown(self):
        known = cfg.describe_method_routing_reason("gpu_crossover_reached")
        unknown = cfg.describe_method_routing_reason("unknown_code")
        assert "GPU crossover threshold" in known
        assert "No explanation registered" in unknown


    def test_cpu_always_rs(self):
        """On CPU backend, auto_select_method always returns 'RS'."""
        import jax
        if jax.default_backend() != "cpu":
            pytest.skip("CPU-only test")
        for fam in ["NO", "GA", "PO", "BI", "WEI", "TF"]:
            for n in [100, 1_000, 100_000, 1_000_000]:
                assert cfg.auto_select_method(fam, n) == "RS"

    def test_unsupported_family_always_rs(self):
        """Unsupported families always get 'RS' regardless of device."""
        for fam in ["NBI", "ZAGA", "BEINF", "JSU"]:
            result = cfg.auto_select_method(fam, 1_000_000)
            assert result == "RS", f"Expected RS for {fam}, got {result}"

    def test_force_jax_overrides(self):
        """FORCE_JAX=True returns RS_JAX for supported families."""
        original = cfg.FORCE_JAX
        try:
            cfg.FORCE_JAX = True
            assert cfg.auto_select_method("NO", 100) == "RS_JAX"
            assert cfg.auto_select_method("NBI", 100) == "RS"  # unsupported
        finally:
            cfg.FORCE_JAX = original

    def test_auto_disabled_always_rs(self):
        """AUTO_METHOD_ENABLED=False always returns 'RS'."""
        original = cfg.AUTO_METHOD_ENABLED
        try:
            cfg.AUTO_METHOD_ENABLED = False
            assert cfg.auto_select_method("NO", 1_000_000) == "RS"
        finally:
            cfg.AUTO_METHOD_ENABLED = original

    def test_gpu_crossover_respected(self):
        """GPU crossover threshold is respected when backend is GPU."""
        import jax
        if jax.default_backend() != "gpu":
            pytest.skip("GPU-only test")
        original = dict(cfg.GPU_CROSSOVER_N)
        try:
            cfg.GPU_CROSSOVER_N["NO"] = 50_000
            assert cfg.auto_select_method("NO", 49_999) == "RS"
            assert cfg.auto_select_method("NO", 50_000) == "RS_JAX"
            assert cfg.auto_select_method("NO", 100_000) == "RS_JAX"
        finally:
            cfg.GPU_CROSSOVER_N.clear()
            cfg.GPU_CROSSOVER_N.update(original)

    def test_get_config_summary(self):
        summary = cfg.get_config_summary()
        assert "auto_method_enabled" in summary
        assert "jax_backend" in summary
        assert "gpu_crossover_n" in summary
        assert "tpu_crossover_n" in summary
        assert "jax_supported_families" in summary


# ---------------------------------------------------------------------------
# Integration: method='auto' via gamlss()
# ---------------------------------------------------------------------------

class TestGamlssAutoMethod:
    def test_auto_produces_valid_model(self):
        data = _make_no_data()
        model = gamlss("y ~ x", family=NO(), data=data, method="auto")
        assert math.isfinite(model.g_dev)
        assert model.g_dev > 0

    def test_auto_matches_rs_on_cpu(self):
        """method='auto' should give same result as method='RS' on CPU."""
        import jax
        if jax.default_backend() != "cpu":
            pytest.skip("CPU-only test")
        data = _make_no_data()
        m_auto = gamlss("y ~ x", family=NO(), data=data, method="auto")
        m_rs   = gamlss("y ~ x", family=NO(), data=data, method="RS")
        assert abs(m_auto.g_dev - m_rs.g_dev) < 1e-6

    def test_auto_stored_in_additional_slots(self):
        """The resolved method name should be stored in the model."""
        data = _make_no_data()
        model = gamlss("y ~ x", family=NO(), data=data, method="auto")
        # method slot should be 'RS' or 'RS_JAX', not 'AUTO'
        method_used = model.additional_slots.get("method", "")
        assert method_used in ("RS", "RS_JAX"), f"Unexpected method: {method_used}"
        routing = model.additional_slots.get("method_routing")
        if routing is not None:
            assert isinstance(routing, dict)
            assert routing.get("selected_method") in ("RS", "RS_JAX")
            assert routing.get("family") == "NO"
            assert isinstance(routing.get("reason"), str)
            assert isinstance(routing.get("reason_detail"), str)
            assert routing["reason_detail"]

    def test_auto_with_unsupported_family_uses_rs(self):
        """method='auto' with an unsupported family falls back to RS."""
        from omnilss import NBI
        rng = np.random.default_rng(42)
        x = rng.standard_normal(200)
        mu = np.exp(1.0 + 0.3 * x)
        y = rng.negative_binomial(5, 5 / (5 + mu)).astype(float)
        data = {"y": y, "x": x}
        model = gamlss("y ~ x", family=NBI(), data=data, method="auto")
        assert math.isfinite(model.g_dev)

    @pytest.mark.parametrize("family_fn,data_fn", [
        (NO,  _make_no_data),
        (GA,  lambda: {
            "y": np.random.default_rng(43).gamma(4.0, np.exp(1.0 + 0.5 * np.random.default_rng(43).standard_normal(200)) / 4.0),
            "x": np.random.default_rng(43).standard_normal(200),
        }),
        (PO,  lambda: {
            "y": np.random.default_rng(44).poisson(np.exp(1.0 + 0.3 * np.random.default_rng(44).standard_normal(200))).astype(float),
            "x": np.random.default_rng(44).standard_normal(200),
        }),
    ])
    def test_auto_multiple_families(self, family_fn, data_fn):
        data = data_fn()
        model = gamlss("y ~ x", family=family_fn(), data=data, method="auto")
        assert math.isfinite(model.g_dev)


# ---------------------------------------------------------------------------
# User-facing crossover configuration API
# ---------------------------------------------------------------------------

def test_set_crossover_gpu():
    """set_crossover() correctly updates GPU_CROSSOVER_N."""
    original = dict(cfg.GPU_CROSSOVER_N)
    try:
        cfg.set_crossover("gpu", n=50_000, family="NO")
        assert cfg.GPU_CROSSOVER_N["NO"] == 50_000
    finally:
        cfg.GPU_CROSSOVER_N.clear()
        cfg.GPU_CROSSOVER_N.update(original)


def test_set_crossover_invalid_device():
    """set_crossover("cpu", ...) raises ValueError."""
    with pytest.raises(ValueError, match="device must be"):
        cfg.set_crossover("cpu", n=10)  # type: ignore[arg-type]


def test_set_crossover_negative_threshold():
    """Negative crossover thresholds are rejected."""
    with pytest.raises(ValueError, match="non-negative"):
        cfg.set_crossover("gpu", n=-1, family="NO")


def test_crossover_context_manager_restores():
    """crossover_config() restores original values after exit."""
    original_gpu = dict(cfg.GPU_CROSSOVER_N)
    original_tpu = dict(cfg.TPU_CROSSOVER_N)
    with cfg.crossover_config(gpu={"NO": 10, "default": 20}, tpu={"GA": 30}):
        assert cfg.GPU_CROSSOVER_N["NO"] == 10
        assert cfg.GPU_CROSSOVER_N["default"] == 20
        assert cfg.TPU_CROSSOVER_N["GA"] == 30
    assert cfg.GPU_CROSSOVER_N == original_gpu
    assert cfg.TPU_CROSSOVER_N == original_tpu


def test_force_jax_overrides_cpu(monkeypatch):
    """FORCE_JAX=True returns RS_JAX even on CPU for supported families."""
    original_force = cfg.FORCE_JAX
    monkeypatch.setattr(cfg, "_current_backend", lambda: ("cpu", []))
    try:
        cfg.FORCE_JAX = True
        assert cfg.auto_select_method("NO", 1) == "RS_JAX"
    finally:
        cfg.FORCE_JAX = original_force


def test_auto_method_disabled(monkeypatch):
    """AUTO_METHOD_ENABLED=False makes auto_select_method() return RS."""
    original_auto = cfg.AUTO_METHOD_ENABLED
    original_force = cfg.FORCE_JAX
    monkeypatch.setattr(cfg, "_current_backend", lambda: ("gpu", []))
    try:
        cfg.FORCE_JAX = False
        cfg.AUTO_METHOD_ENABLED = False
        cfg.set_crossover("gpu", n=1, family="NO")
        assert cfg.auto_select_method("NO", 1_000_000) == "RS"
    finally:
        cfg.AUTO_METHOD_ENABLED = original_auto
        cfg.FORCE_JAX = original_force


def test_yaml_config_loading(tmp_path, monkeypatch):
    """YAML config files load crossover thresholds."""
    import importlib

    config_file = tmp_path / "omnilss_config.yaml"
    config_file.write_text(
        "auto_method_enabled: false\n"
        "force_jax: true\n"
        "gpu_crossover_n:\n"
        "  NO: 50000\n"
        "  default: .inf\n"
        "tpu_crossover_n:\n"
        "  GA: 10000\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OMNILSS_CONFIG_FILE", str(config_file))
    monkeypatch.delenv("OMNILSS_AUTO_METHOD", raising=False)
    monkeypatch.delenv("OMNILSS_FORCE_JAX", raising=False)
    monkeypatch.delenv("OMNILSS_GPU_CROSSOVER_N", raising=False)
    monkeypatch.delenv("OMNILSS_TPU_CROSSOVER_N", raising=False)

    reloaded = importlib.reload(cfg)
    try:
        assert reloaded.AUTO_METHOD_ENABLED is False
        assert reloaded.FORCE_JAX is True
        assert reloaded.GPU_CROSSOVER_N["NO"] == 50_000
        assert reloaded.GPU_CROSSOVER_N["default"] == math.inf
        assert reloaded.TPU_CROSSOVER_N["GA"] == 10_000
    finally:
        monkeypatch.delenv("OMNILSS_CONFIG_FILE", raising=False)
        importlib.reload(cfg)


def test_yaml_config_env_overrides_file(tmp_path, monkeypatch):
    """Environment variables take precedence over YAML config files."""
    import importlib

    config_file = tmp_path / "omnilss_config.yaml"
    config_file.write_text(
        "auto_method_enabled: false\n"
        "gpu_crossover_n:\n"
        "  NO: 50000\n"
        "  default: .inf\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OMNILSS_CONFIG_FILE", str(config_file))
    monkeypatch.setenv("OMNILSS_AUTO_METHOD", "1")
    monkeypatch.setenv("OMNILSS_GPU_CROSSOVER_N", "NO=25000,GA=80000,default=90000")

    reloaded = importlib.reload(cfg)
    try:
        assert reloaded.AUTO_METHOD_ENABLED is True
        assert reloaded.GPU_CROSSOVER_N["NO"] == 25_000
        assert reloaded.GPU_CROSSOVER_N["GA"] == 80_000
        assert reloaded.GPU_CROSSOVER_N["default"] == 90_000
    finally:
        monkeypatch.delenv("OMNILSS_CONFIG_FILE", raising=False)
        monkeypatch.delenv("OMNILSS_AUTO_METHOD", raising=False)
        monkeypatch.delenv("OMNILSS_GPU_CROSSOVER_N", raising=False)
        importlib.reload(cfg)


def test_unknown_family_warns_but_accepts():
    """set_crossover() warns for unknown families but keeps the value."""
    original = dict(cfg.GPU_CROSSOVER_N)
    try:
        with pytest.warns(UserWarning, match="Unknown family"):
            cfg.set_crossover("gpu", n=123, family="FUTURE")
        assert cfg.GPU_CROSSOVER_N["FUTURE"] == 123
    finally:
        cfg.GPU_CROSSOVER_N.clear()
        cfg.GPU_CROSSOVER_N.update(original)


def test_tpu_crossover_has_all_jax_family_placeholders():
    """TPU placeholders cover every currently JAX-supported family."""
    for family in cfg.JAX_SUPPORTED_FAMILIES:
        assert family in cfg.TPU_CROSSOVER_N
        assert cfg.TPU_CROSSOVER_N[family] == math.inf
