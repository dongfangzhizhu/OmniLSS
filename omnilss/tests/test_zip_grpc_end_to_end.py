"""End-to-end gRPC ZIP fit, persistence, and prediction regression."""

from __future__ import annotations

import json

import numpy as np
import pytest


grpc = pytest.importorskip("grpc")


def test_zip_fit_predict_survives_registry_rebuild(tmp_path, monkeypatch):
    from omnilss.api.grpc import server as grpc_server
    from omnilss.api.grpc.generated import fit_pb2, predict_pb2

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    monkeypatch.setattr(grpc_server, "MODEL_STORE", model_dir)
    monkeypatch.setattr(grpc_server, "MODEL_DB", model_dir / "registry.sqlite3")

    isolated_registry = grpc_server._ModelRegistry()
    monkeypatch.setattr(grpc_server, "REGISTRY", isolated_registry)
    service, *_ = grpc_server.create_service()
    rng = np.random.default_rng(2042)
    x = np.linspace(-1.0, 1.0, 48)
    mu = np.exp(0.25 + 0.3 * x)
    y = np.where(rng.random(x.size) < 0.25, 0, rng.poisson(mu)).astype(float)
    weights = np.linspace(0.75, 1.25, x.size)
    request = fit_pb2.FitRequest(
        formula="y ~ x",
        family="ZIP",
        data_json=json.dumps({"y": y.tolist(), "x": x.tolist(), "weights": weights.tolist()}),
        sigma_formula="~ x",
        method="RS",
        max_iter=20,
    )

    response = service.Fit(request, None)
    assert response.success, response.error
    assert response.model_id

    newdata = {"x": [-0.75, 0.0, 0.75]}
    first = service.Predict(
        predict_pb2.PredictRequest(
            model_id=response.model_id,
            newdata_json=json.dumps(newdata),
        ),
        None,
    )
    assert first.success, first.error
    first_params = json.loads(first.params_json)
    assert set(first_params) == {"mu", "sigma"}
    assert len(first_params["mu"]) == 3

    # Simulate a process restart: rebuild the registry from the persisted DB
    # and artifact, then predict through the same service boundary.
    previous = isolated_registry
    try:
        previous._db.close()
    except Exception:
        pass
    restarted = grpc_server._ModelRegistry()
    monkeypatch.setattr(grpc_server, "REGISTRY", restarted)
    second = service.Predict(
        predict_pb2.PredictRequest(
            model_id=response.model_id,
            newdata_json=json.dumps(newdata),
        ),
        None,
    )
    assert second.success, second.error
    np.testing.assert_allclose(
        json.loads(second.params_json)["mu"], first_params["mu"], rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        json.loads(second.params_json)["sigma"], first_params["sigma"], rtol=1e-10, atol=1e-10
    )
