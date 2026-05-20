# SPDX-License-Identifier: GPL-3.0-or-later
"""End-to-end method routing tests for gamlss(method='auto')."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import omnilss.config as cfg
from omnilss import NBI, NO, gamlss
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


def test_default_rs_routes_to_jax_above_threshold(monkeypatch):
    """Default method='RS' routes through configured accelerator thresholds."""
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
        result = gamlss("y ~ x", family=NO(), data=_make_no_data(n=40), method="RS")

    assert result.additional_slots["method"] == "RS_JAX"
    assert calls == ["jax"]

def test_rs_jax_raises_for_unsupported_family():
    """method='RS_JAX' with an unsupported family raises a clear capability error."""
    data = _make_no_data()
    with pytest.raises(FamilyCapabilityError, match="not supported by method='RS_JAX'"):
        gamlss("y ~ x", family=NBI(), data=data, method="RS_JAX")


def test_rs_jax_capability_error_happens_before_backend(monkeypatch):
    """Unsupported RS_JAX routes fail before importing/calling the JAX backend."""
    data = _make_no_data()

    def fail_backend(**kwargs):  # pragma: no cover - should not be reached
        raise AssertionError("JAX backend should not be called for unsupported family")

    monkeypatch.setattr(jax_rs_integration, "gamlss_rs_jax", fail_backend)

    with pytest.raises(FamilyCapabilityError, match="Use method='RS' instead"):
        gamlss("y ~ x", family=NBI(), data=data, method="RS_JAX")


def test_auto_routes_to_jax_above_threshold(monkeypatch):
    """GPU backend switches to JAX when n exceeds the configured threshold."""
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
        result = gamlss("y ~ x", family=NO(), data=_make_no_data(n=40), method="auto")

    assert result.additional_slots["method"] == "RS_JAX"
    assert calls == ["jax"]


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


def test_rs_jax_receives_routing_decision_payload(monkeypatch):
    captured = {}

    def fake_jax_fit(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(additional_slots={"method": "RS_JAX"})

    monkeypatch.setattr(cfg, "_current_backend", lambda: ("gpu", []))
    monkeypatch.setattr(jax_rs_integration, "gamlss_rs_jax", fake_jax_fit)

    with cfg.crossover_config(gpu={"NO": 1}):
        gamlss("y ~ x", family=NO(), data=_make_no_data(n=40), method="auto")

    routing = captured.get("routing_decision")
    assert isinstance(routing, dict)
    assert routing.get("requested_method") == "AUTO"
    assert routing.get("selected_method") == "RS_JAX"



def test_manual_rs_jax_receives_explicit_routing_decision(monkeypatch):
    captured = {}

    def fake_jax_fit(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(additional_slots={"method": "RS_JAX", "method_routing": kwargs.get("routing_decision")})

    monkeypatch.setattr(cfg, "_current_backend", lambda: ("gpu", []))
    monkeypatch.setattr(jax_rs_integration, "gamlss_rs_jax", fake_jax_fit)

    result = gamlss("y ~ x", family=NO(), data=_make_no_data(n=40), method="RS_JAX")

    routing = captured.get("routing_decision")
    assert isinstance(routing, dict)
    assert routing.get("requested_method") == "RS_JAX"
    assert routing.get("selected_method") == "RS_JAX"
    assert routing.get("reason") == "explicit_method_requested"
    assert routing.get("reason_detail")
    assert result.additional_slots.get("method_routing") == routing
