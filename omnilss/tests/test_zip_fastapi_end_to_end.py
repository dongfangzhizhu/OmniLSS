"""FastAPI ZIP fit/predict service regression."""

from __future__ import annotations

import json
import numpy as np
import pytest


fastapi = pytest.importorskip("fastapi")


def test_zip_fastapi_fit_predict_with_weights(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from omnilss.api.http import fastapi_server
    from omnilss.api.grpc import server as grpc_server

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    monkeypatch.setattr(grpc_server, "MODEL_STORE", model_dir)
    monkeypatch.setattr(grpc_server, "MODEL_DB", model_dir / "registry.sqlite3")
    registry = grpc_server._ModelRegistry()
    monkeypatch.setattr(grpc_server, "REGISTRY", registry)
    monkeypatch.delenv("OMNILSS_API_TOKEN", raising=False)
    app = fastapi_server.create_app()

    rng = np.random.default_rng(2043)
    x = np.linspace(-1.0, 1.0, 36)
    y = np.where(rng.random(x.size) < 0.25, 0, rng.poisson(np.exp(0.2 + 0.25 * x))).tolist()
    response = TestClient(app).post("/fit", json={
        "formula": "y ~ x",
        "family": "ZIP",
        "method": "RS",
        "sigma_formula": "~ x",
        "max_iter": 15,
        "data": {"y": y, "x": x.tolist(), "weights": np.linspace(0.8, 1.2, x.size).tolist()},
    })
    assert response.status_code == 200, response.text
    fit = response.json()
    assert fit["success"] is True
    prediction = TestClient(app).post(
        f"/models/{fit['model_id']}/predict",
        json={"newdata": {"x": [-0.5, 0.0, 0.5]}},
    )
    assert prediction.status_code == 200, prediction.text
    payload = prediction.json()
    assert set(payload["params"]) == {"mu", "sigma"}
    assert len(payload["params"]["mu"]) == 3


def test_fastapi_rejects_placeholder_before_fit(monkeypatch):
    from fastapi.testclient import TestClient
    from omnilss.api.http import fastapi_server

    monkeypatch.delenv("OMNILSS_API_TOKEN", raising=False)
    response = TestClient(fastapi_server.create_app()).post("/fit", json={
        "formula": "y ~ x", "family": "GIG", "data": {"y": [1.0], "x": [0.0]}
    })
    assert response.status_code == 400
    assert "placeholder" in str(response.json()["detail"]).lower()


def test_fastapi_fit_and_predict_require_bearer_token(monkeypatch):
    from fastapi.testclient import TestClient
    from omnilss.api.http import fastapi_server

    monkeypatch.setenv("OMNILSS_API_TOKEN", "test-token")
    client = TestClient(fastapi_server.create_app())
    body = {"formula": "y ~ 1", "family": "NO", "data": {"y": [1.0, 2.0]}}
    assert client.post("/fit", json=body).status_code == 401
    response = client.post("/fit", json=body, headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 200, response.text
    model_id = response.json()["model_id"]
    assert client.post(
        f"/models/{model_id}/predict",
        json={"newdata": {}},
        headers={"Authorization": "Bearer test-token"},
    ).status_code == 200


def test_fastapi_capability_routes_match_runtime_report(monkeypatch):
    from fastapi.testclient import TestClient
    from omnilss.api.http import fastapi_server
    from omnilss.family_capabilities import capability_matrix, method_route_capability_report

    monkeypatch.delenv("OMNILSS_API_TOKEN", raising=False)
    client = TestClient(fastapi_server.create_app())
    capabilities = client.get("/capabilities")
    assert capabilities.status_code == 200
    assert capabilities.json() == capability_matrix()
    route = client.get("/route-capability", params={"family": "ZIP", "method": "RS", "strict": "true"})
    assert route.status_code == 200
    assert route.json() == method_route_capability_report("ZIP", "RS", strict=True, formulas={})


def test_fastapi_and_grpc_share_model_lifecycle(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from omnilss.api.grpc import server as grpc_server
    from omnilss.api.grpc.generated import fit_pb2, predict_pb2
    from omnilss.api.http import fastapi_server

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    monkeypatch.setattr(grpc_server, "MODEL_STORE", model_dir)
    monkeypatch.setattr(grpc_server, "MODEL_DB", model_dir / "registry.sqlite3")
    registry = grpc_server._ModelRegistry()
    monkeypatch.setattr(grpc_server, "REGISTRY", registry)
    monkeypatch.delenv("OMNILSS_API_TOKEN", raising=False)
    app = TestClient(fastapi_server.create_app())
    service, *_ = grpc_server.create_service()

    data = {"y": [0.0, 1.0, 0.0, 2.0, 1.0], "x": [-1.0, -0.5, 0.0, 0.5, 1.0]}
    http_fit = app.post("/fit", json={
        "formula": "y ~ x", "family": "ZIP", "data": data, "sigma_formula": "~ 1"
    })
    assert http_fit.status_code == 200, http_fit.text
    http_model_id = http_fit.json()["model_id"]
    grpc_prediction = service.Predict(
        predict_pb2.PredictRequest(
            model_id=http_model_id,
            newdata_json='{"x": [-0.25, 0.25]}',
        ),
        None,
    )
    assert grpc_prediction.success, grpc_prediction.error
    assert set(json.loads(grpc_prediction.params_json)) == {"mu", "sigma"}

    grpc_fit = service.Fit(
        fit_pb2.FitRequest(
            formula="y ~ x", family="ZIP", data_json=json.dumps(data), method="RS"
        ),
        None,
    )
    assert grpc_fit.success, grpc_fit.error
    http_prediction = app.post(
        "/models/%s/predict" % grpc_fit.model_id,
        json={"newdata": {"x": [-0.25, 0.25]}},
    )
    assert http_prediction.status_code == 200, http_prediction.text
    assert set(http_prediction.json()["params"]) == {"mu", "sigma"}

    deleted = app.delete("/models/%s" % http_model_id)
    assert deleted.status_code == 200
    after_delete = service.Predict(
        predict_pb2.PredictRequest(model_id=http_model_id, newdata_json='{"x": [0.0]}'),
        None,
    )
    assert after_delete.success is False
    assert after_delete.error == "KeyError error"
