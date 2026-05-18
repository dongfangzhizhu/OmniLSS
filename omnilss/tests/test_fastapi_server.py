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
