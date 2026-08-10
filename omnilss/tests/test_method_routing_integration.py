# SPDX-License-Identifier: GPL-3.0-or-later
"""End-to-end method routing tests for gamlss(method='auto'/'RS').

Updated: 2026-07-23 based on comprehensive audit findings.

Routing contract (post-fix):
- method='RS' = explicit NumPy RS (CPU path), always returns 'RS'
- method='auto' = device-aware routing, may return 'RS' or 'RS_JAX' based on backend + threshold + p_total
- method='RS_JAX' = deprecated, emits DeprecationWarning, falls back to NumPy RS for small p
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import omnilss.config as cfg
from omnilss import NO, NO2, gamlss
from omnilss.family_capabilities import FamilyCapabilityError
from omnilss.algorithms import jax_rs_integration, rs_algorithm


def _make_no_data(n: int = 40) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260517)
    x = rng.normal(size=n)
    y = 1.0 + 0.5 * x + rng.normal(scale=0.25, size=n)
    return {"y": y, "x": x}


def test_auto_routes_to_rs_on_cpu(monkeypatch):
    """CPU backend method='auto' calls the NumPy RS path."""
    calls: list[str] = []

    def fake_rs_fit(**kwargs):
        calls.append("rs")
        return SimpleNamespace(additional_slots={"method": "RS"})

    def fake_jax_fit(**kwargs):  # pragma: no cover - should not be called
        calls.append("jax")
        return SimpleNamespace(additional_slots={"method": "RS_JAX"})

    monkeypatch.setattr(cfg, "_current_backend", lambda: ("cpu", []))
    monkeypatch.setattr(rs_algorithm, "rs_fit", fake_rs_fit)
    monkeypatch.setattr(jax_rs_integration, "gamlss_rs_jax", fake_jax_fit)

    result = gamlss("y ~ x", family=NO(), data=_make_no_data(), method="auto")

    assert result.additional_slots["method"] == "RS"
    assert calls == ["rs"]


def test_manual_rs_works_on_gpu_env(monkeypatch):
    """GPU availability does not override explicit method='RS'."""
    calls: list[str] = []

    def fake_rs_fit(**kwargs):
        calls.append("rs")
        return SimpleNamespace(additional_slots={"method": "RS"})

    def fake_jax_fit(**kwargs):  # pragma: no cover - should not be called
        calls.append("jax")
        return SimpleNamespace(additional_slots={"method": "RS_JAX"})

    monkeypatch.setattr(cfg, "_current_backend", lambda: ("gpu", []))
    monkeypatch.setattr(rs_algorithm, "rs_fit", fake_rs_fit)
    monkeypatch.setattr(jax_rs_integration, "gamlss_rs_jax", fake_jax_fit)

    result = gamlss("y ~ x", family=NO(), data=_make_no_data(), method="RS")

    assert result.additional_slots["method"] == "RS"
    assert calls == ["rs"]


def test_auto_routes_to_jax_above_threshold(monkeypatch):
    """Auto method routes to JAX when GPU backend and n exceeds threshold, p_total > 4."""
    calls: list[str] = []

    def fake_rs_fit(**kwargs):  # pragma: no cover - should not be called
        calls.append("rs")
        return SimpleNamespace(additional_slots={"method": "RS"})

    def fake_jax_fit(**kwargs):
        calls.append("jax")
        return SimpleNamespace(additional_slots={"method": "RS_JAX"})

    monkeypatch.setattr(cfg, "_current_backend", lambda: ("gpu", []))
    monkeypatch.setattr(rs_algorithm, "rs_fit", fake_rs_fit)
    monkeypatch.setattr(jax_rs_integration, "gamlss_rs_jax", fake_jax_fit)

    with cfg.crossover_config(gpu={"NO": 10}):
        # Use large p_total (>4) to avoid small-p fallback
        result = gamlss("y ~ x1 + x2 + x3 + x4 + x5", family=NO(), data=_make_no_data(n=40), method="auto")

    assert result.additional_slots["method"] == "RS_JAX"
    assert calls == ["jax"]


def test_auto_routes_to_rs_below_threshold(monkeypatch):
    """Auto method stays on NumPy RS when n is below the GPU crossover threshold."""
    calls: list[str] = []

    def fake_rs_fit(**kwargs):
        calls.append("rs")
        return SimpleNamespace(additional_slots={"method": "RS"})

    def fake_jax_fit(**kwargs):  # pragma: no cover - should not be called
        calls.append("jax")
        return SimpleNamespace(additional_slots={"method": "RS_JAX"})

    monkeypatch.setattr(cfg, "_current_backend", lambda: ("gpu", []))
    monkeypatch.setattr(rs_algorithm, "rs_fit", fake_rs_fit)
    monkeypatch.setattr(jax_rs_integration, "gamlss_rs_jax", fake_jax_fit)

    with cfg.crossover_config(gpu={"NO": 100}):  # n=40 < threshold=100
        result = gamlss("y ~ x1 + x2 + x3 + x4 + x5", family=NO(), data=_make_no_data(n=40), method="auto")

    assert result.additional_slots["method"] == "RS"
    assert calls == ["rs"]


def test_auto_routes_to_jax_at_threshold(monkeypatch):
    """Auto method routes to JAX when n equals the GPU crossover threshold."""
    calls: list[str] = []

    def fake_rs_fit(**kwargs):  # pragma: no cover - should not be called
        calls.append("rs")
        return SimpleNamespace(additional_slots={"method": "RS"})

    def fake_jax_fit(**kwargs):
        calls.append("jax")
        return SimpleNamespace(additional_slots={"method": "RS_JAX"})

    monkeypatch.setattr(cfg, "_current_backend", lambda: ("gpu", []))
    monkeypatch.setattr(rs_algorithm, "rs_fit", fake_rs_fit)
    monkeypatch.setattr(jax_rs_integration, "gamlss_rs_jax", fake_jax_fit)

    with cfg.crossover_config(gpu={"NO": 40}):  # n=40 == threshold=40
        result = gamlss("y ~ x1 + x2 + x3 + x4 + x5", family=NO(), data=_make_no_data(n=40), method="auto")

    assert result.additional_slots["method"] == "RS_JAX"
    assert calls == ["jax"]


def test_routing_records_accelerator_available(monkeypatch):
    """Route diagnostic records accelerator_available=False without a GPU/TPU."""
    # Real CPU fit: route diagnostic must record accelerator_available=False.
    result = gamlss("y ~ x", family=NO(), data=_make_no_data(n=40), method="auto")
    routing = result.additional_slots["method_routing"]
    assert routing["accelerator_available"] is False

    # GPU backend: routing decision handed to the backend records True.
    captured = {}

    def fake_jax_fit(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(additional_slots={"method": "RS_JAX"})

    monkeypatch.setattr(cfg, "_current_backend", lambda: ("gpu", []))
    monkeypatch.setattr(jax_rs_integration, "gamlss_rs_jax", fake_jax_fit)

    with cfg.crossover_config(gpu={"NO": 1}):
        gamlss("y ~ x1 + x2 + x3 + x4 + x5", family=NO(), data=_make_no_data(n=40), method="auto")

    assert captured["routing_decision"]["accelerator_available"] is True


def test_rs_does_not_route_to_jax(monkeypatch):
    """Explicit method='RS' always uses NumPy RS, even on GPU with high threshold."""
    calls: list[str] = []

    def fake_rs_fit(**kwargs):
        calls.append("rs")
        return SimpleNamespace(additional_slots={"method": "RS"})

    def fake_jax_fit(**kwargs):  # pragma: no cover - should not be called
        calls.append("jax")
        return SimpleNamespace(additional_slots={"method": "RS_JAX"})

    monkeypatch.setattr(cfg, "_current_backend", lambda: ("gpu", []))
    monkeypatch.setattr(rs_algorithm, "rs_fit", fake_rs_fit)
    monkeypatch.setattr(jax_rs_integration, "gamlss_rs_jax", fake_jax_fit)

    with cfg.crossover_config(gpu={"NO": 10}):
        result = gamlss("y ~ x", family=NO(), data=_make_no_data(n=40), method="RS")

    assert result.additional_slots["method"] == "RS"
    assert calls == ["rs"]


def test_rs_jax_deprecated_warning():
    """Deprecated RS_JAX emits warning and may return RS or RS_JAX depending on routing."""
    import warnings

    data = _make_no_data()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = gamlss("y ~ x", family=NO(), data=data, method="RS_JAX")

    # Should emit deprecation warning
    assert any(issubclass(wi.category, DeprecationWarning) for wi in w)
    # RS_JAX may fall back to RS or remain RS_JAX depending on implementation details
    assert result.additional_slots["method"] in {"RS", "RS_JAX"}


def test_auto_rejects_placeholder_family():
    """Placeholder families are rejected by default, even with auto routing."""
    data = _make_no_data()
    with pytest.raises(FamilyCapabilityError, match="placeholder"):
        gamlss("y ~ x", family=NO2(), data=data, method="auto")


def test_rs_rejects_placeholder_family(monkeypatch):
    """Placeholder families are rejected by default for explicit RS too."""
    data = _make_no_data()
    with pytest.raises(FamilyCapabilityError, match="placeholder"):
        gamlss("y ~ x", family=NO2(), data=data, method="RS")


def test_strict_capabilities_allow_validated_no_rs_route(monkeypatch):
    calls: list[str] = []

    def fake_rs_fit(**kwargs):
        calls.append("rs")
        return SimpleNamespace(additional_slots={"method": "RS"})

    monkeypatch.setattr(rs_algorithm, "rs_fit", fake_rs_fit)

    result = gamlss(
        "y ~ x",
        family=NO(),
        data=_make_no_data(),
        method="RS",
        strict_capabilities=True,
    )

    assert result.additional_slots["method"] == "RS"
    assert calls == ["rs"]


def test_strict_capabilities_reject_experimental_ga_rs_route():
    from omnilss import GA

    with pytest.raises(FamilyCapabilityError, match="requested evidence tier"):
        gamlss(
            "y ~ x",
            family=GA(),
            data=_make_no_data(),
            method="RS",
            strict_capabilities=True,
        )


def test_auto_routes_small_p_to_rs_on_gpu(monkeypatch):
    """Small p_total (<=4) forces NumPy RS even on GPU with crossover threshold."""
    calls: list[str] = []

    def fake_rs_fit(**kwargs):
        calls.append("rs")
        return SimpleNamespace(additional_slots={"method": "RS"})

    def fake_jax_fit(**kwargs):  # pragma: no cover - should not be called
        calls.append("jax")
        return SimpleNamespace(additional_slots={"method": "RS_JAX"})

    monkeypatch.setattr(cfg, "_current_backend", lambda: ("gpu", []))
    monkeypatch.setattr(rs_algorithm, "rs_fit", fake_rs_fit)
    monkeypatch.setattr(jax_rs_integration, "gamlss_rs_jax", fake_jax_fit)

    with cfg.crossover_config(gpu={"NO": 10}):
        result = gamlss("y ~ x", family=NO(), data=_make_no_data(n=40), method="auto")

    # Small p triggers NumPy RS fallback
    assert result.additional_slots["method"] == "RS"
    assert calls == ["rs"]


def test_routing_decision_payload_for_auto(monkeypatch):
    captured = {}

    def fake_jax_fit(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(additional_slots={"method": "RS_JAX"})

    monkeypatch.setattr(cfg, "_current_backend", lambda: ("gpu", []))
    monkeypatch.setattr(jax_rs_integration, "gamlss_rs_jax", fake_jax_fit)

    with cfg.crossover_config(gpu={"NO": 1}):
        # Large p to bypass small-p fallback
        gamlss("y ~ x1 + x2 + x3 + x4 + x5", family=NO(), data=_make_no_data(n=40), method="auto")

    routing = captured.get("routing_decision")
    assert isinstance(routing, dict)
    assert routing.get("requested_method") == "AUTO"
    assert routing.get("selected_method") == "RS_JAX"


def test_manual_rs_jax_gets_routing_decision(monkeypatch):
    """Deprecated RS_JAX still receives routing decision payload."""
    captured = {}

    def fake_jax_fit(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(additional_slots={"method": "RS", "method_routing": kwargs.get("routing_decision")})

    monkeypatch.setattr(cfg, "_current_backend", lambda: ("gpu", []))
    monkeypatch.setattr(jax_rs_integration, "gamlss_rs_jax", fake_jax_fit)

    result = gamlss("y ~ x", family=NO(), data=_make_no_data(n=40), method="RS_JAX")

    routing = captured.get("routing_decision")
    assert isinstance(routing, dict)
    assert routing.get("requested_method") == "RS_JAX"
    # RS_JAX falls back to RS for small p
    assert result.additional_slots.get("method") in {"RS", "RS_JAX"}
