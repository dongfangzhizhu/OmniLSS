"""ZIP service boundary forwarding contracts."""

from __future__ import annotations

import json

import numpy as np
import pytest


def test_grpc_fit_forwards_weights_from_data_envelope(monkeypatch):
    pytest.importorskip("grpc")
    from omnilss.api.grpc import server as grpc_server
    from omnilss.api.grpc.generated import fit_pb2

    captured = {}

    class DummyRegistry:
        def save(self, model):
            return "zip-model"

    class DummyModel:
        g_dev = 1.0
        iter = 2
        additional_slots = {"converged": True}

    def fake_gamlss(**kwargs):
        captured.update(kwargs)
        return DummyModel()

    monkeypatch.setattr(grpc_server, "gamlss", fake_gamlss)
    monkeypatch.setattr(grpc_server, "REGISTRY", DummyRegistry())
    service, *_ = grpc_server.create_service()
    weights = np.linspace(0.5, 1.5, 4).tolist()
    request = fit_pb2.FitRequest(
        formula="y ~ x",
        family="ZIP",
        data_json=json.dumps({"y": [0, 1, 0, 2], "x": [0, 1, 2, 3], "weights": weights}),
        sigma_formula="~ x",
        method="RS",
    )
    response = service.Fit(request, None)
    assert response.success is True
    np.testing.assert_allclose(captured["weights"], weights)
    assert captured["sigma_formula"] == "~ x"


def test_grpc_fit_rejects_zip_placeholder_route_before_backend(monkeypatch):
    pytest.importorskip("grpc")
    from omnilss.api.grpc import server as grpc_server
    from omnilss.api.grpc.generated import fit_pb2

    def fail_backend(**kwargs):
        raise AssertionError("backend must not be called")

    monkeypatch.setattr(grpc_server, "gamlss", fail_backend)
    service, *_ = grpc_server.create_service()
    response = service.Fit(
        fit_pb2.FitRequest(
            formula="y ~ x",
            family="GIG",
            data_json=json.dumps({"y": [1.0], "x": [0.0]}),
            method="RS",
        ),
        None,
    )
    assert response.success is False
    assert "placeholder" in response.error.lower()
