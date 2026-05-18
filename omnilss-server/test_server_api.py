import pytest
pytest.importorskip("fastapi")

import numpy as np
from fastapi.testclient import TestClient

from main import app


def test_fit_predict_diagnostics_flow():
    client = TestClient(app)
    rng = np.random.default_rng(0)
    n = 40
    x = rng.normal(size=n).tolist()
    y = (1.0 + 0.5 * np.array(x) + rng.normal(scale=0.2, size=n)).tolist()

    fit_resp = client.post('/fit', json={"formula": "y ~ x", "family": "NO", "data": {"y": y, "x": x}})
    assert fit_resp.status_code == 200
    model_id = fit_resp.json()["model_id"]

    pred_resp = client.post('/predict', json={"model_id": model_id, "newdata": {"x": [0.1, -0.2]}})
    assert pred_resp.status_code == 200
    assert "mu" in pred_resp.json()["params"]

    diag_resp = client.get(f'/diagnostics/{model_id}')
    assert diag_resp.status_code == 200
    assert "deviance" in diag_resp.json()

    list_resp = client.get('/models')
    assert list_resp.status_code == 200
    assert model_id in list_resp.json()["model_ids"]

    delete_resp = client.delete(f'/models/{model_id}')
    assert delete_resp.status_code == 200
    assert delete_resp.json()["deleted"] is True

    missing_pred_resp = client.post('/predict', json={"model_id": model_id, "newdata": {"x": [0.0]}})
    assert missing_pred_resp.status_code == 404
