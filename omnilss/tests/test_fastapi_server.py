import importlib.util

import pytest

FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="fastapi not installed")
def test_fastapi_models_endpoints_share_registry(monkeypatch):
    from fastapi.testclient import TestClient

    from omnilss.api.http.fastapi_server import create_app

    class DummyRegistry:
        def __init__(self):
            self.ids = ["m1", "m2"]

        def list_ids(self):
            return list(self.ids)

        def delete(self, model_id: str):
            if model_id in self.ids:
                self.ids.remove(model_id)
                return True
            return False

        def audit(self):
            return [
                {"model_id": "m1", "integrity_ok": True, "capability_ok": True},
                {"model_id": "m2", "integrity_ok": False, "capability_ok": False},
            ]

    def audit(self):
            return [
                {"model_id": "m1", "integrity_ok": True, "capability_ok": True},
                {"model_id": "m2", "integrity_ok": False, "capability_ok": False},
            ]

    import omnilss.api.grpc.server as grpc_server

    dummy = DummyRegistry()
    monkeypatch.setattr(grpc_server, "REGISTRY", dummy)

    app = create_app()
    client = TestClient(app)

    response = client.get("/models")
    assert response.status_code == 200
    assert response.json() == {"model_ids": ["m1", "m2"]}

    response = client.delete("/models/m1")
    assert response.status_code == 200
    assert response.json()["deleted"] is True

    response = client.delete("/models/missing")
    assert response.status_code == 404

    audit = client.get("/models/audit")
    assert audit.status_code == 200
    assert audit.json()["status"] == "degraded"
    assert audit.json()["unhealthy_model_count"] == 1
    assert audit.json()["metrics"]["integrity_failures"] == 1
    assert audit.json()["metrics"]["capability_failures"] == 1
    assert client.get("/health").json()["status"] == "degraded"
    metrics = client.get("/models/audit/metrics")
    assert metrics.json()["audit_requests_total"] == 1
    assert metrics.json()["metrics_requests_total"] == 1
    assert metrics.json()["unhealthy_models"] == 1
    prometheus = client.get("/metrics")
    assert prometheus.status_code == 200
    assert "omnilss_registry_unhealthy_models 1" in prometheus.text
    cleanup = client.post("/models/audit/cleanup")
    assert cleanup.status_code == 200
    assert cleanup.json()["deleted_count"] == 1
    assert cleanup.json()["reason"] == "unhealthy_artifact"
    assert client.get("/models/audit/metrics").json()["cleanup_requests_total"] == 1


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="fastapi not installed")
def test_fastapi_auth_token_enforced_when_configured(monkeypatch):
    from fastapi.testclient import TestClient
    from omnilss.api.http.fastapi_server import create_app
    import omnilss.api.grpc.server as grpc_server

    class DummyRegistry:
        def list_ids(self):
            return ["m1"]

        def delete(self, _model_id: str):
            return True

    monkeypatch.setattr(grpc_server, "REGISTRY", DummyRegistry())
    monkeypatch.setenv("OMNILSS_API_TOKEN", "test123")

    app = create_app()
    client = TestClient(app)

    assert client.get("/models").status_code == 401
    assert client.get("/models", headers={"authorization": "Bearer bad"}).status_code == 401
    assert client.get("/models", headers={"authorization": "Bearer test123"}).status_code == 200
