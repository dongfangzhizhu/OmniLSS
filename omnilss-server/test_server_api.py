# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the omnilss-server FastAPI boundary (auth, DoS, capability gate, lifecycle)."""

from __future__ import annotations

import os

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi import HTTPException  # noqa: E402

import main as server_main  # noqa: E402
from omnilss.api.grpc.server import _ModelRegistry  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_server_state(tmp_path, monkeypatch):
    """Isolate the persistent registry, auth token, and rate-limit state per test."""
    monkeypatch.setenv("OMNILSS_MODEL_STORE_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("OMNILSS_MODEL_DB_PATH", str(tmp_path / "models.sqlite3"))
    registry = _ModelRegistry()
    monkeypatch.setattr(server_main, "REGISTRY", registry)
    monkeypatch.delenv("OMNILSS_API_TOKEN", raising=False)
    server_main._RATE_LIMIT_STATE.clear()
    yield
    server_main._RATE_LIMIT_STATE.clear()


def _sample_normal_data(n: int = 40, seed: int = 0) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = 1.0 + 0.5 * x + rng.normal(scale=0.2, size=n)
    return {"y": y.tolist(), "x": x.tolist()}


def test_fit_predict_diagnostics_delete_lifecycle():
    fit_out = server_main.fit(
        server_main.FitRequest(formula="y ~ x", family="NO", data=_sample_normal_data()),
        None,
    )
    model_id = fit_out["model_id"]
    assert np.isfinite(fit_out["deviance"])
    assert fit_out["converged"] is True

    pred_out = server_main.predict(
        server_main.PredictRequest(model_id=model_id, newdata={"x": [0.1, -0.2]}),
        None,
    )
    assert "mu" in pred_out["params"]
    assert len(pred_out["params"]["mu"]) == 2

    diag_out = server_main.diagnostics(model_id, None)
    assert np.isfinite(diag_out["deviance"])
    assert diag_out["iter"] >= 1

    listed = server_main.list_models(None)
    assert model_id in listed["model_ids"]

    deleted = server_main.delete_model(model_id, None)
    assert deleted == {"deleted": True, "model_id": model_id}

    with pytest.raises(HTTPException) as exc_info:
        server_main.predict(
            server_main.PredictRequest(model_id=model_id, newdata={"x": [0.0]}),
            None,
        )
    assert exc_info.value.status_code == 404


def test_auth_open_when_token_unset():
    assert os.getenv("OMNILSS_API_TOKEN") is None
    server_main._check_auth("")
    server_main._check_auth("Bearer anything")


def test_auth_enforces_bearer_token(monkeypatch):
    monkeypatch.setenv("OMNILSS_API_TOKEN", "top-secret-token")

    with pytest.raises(HTTPException) as missing_exc:
        server_main._check_auth("")
    assert missing_exc.value.status_code == 401

    with pytest.raises(HTTPException) as wrong_exc:
        server_main._check_auth("Bearer wrong-token")
    assert wrong_exc.value.status_code == 401

    with pytest.raises(HTTPException) as scheme_exc:
        server_main._check_auth("Basic top-secret-token")
    assert scheme_exc.value.status_code == 401

    server_main._check_auth("Bearer top-secret-token")


def test_content_length_dos_guard():
    assert server_main._check_content_length(None) is True
    assert server_main._check_content_length(str(server_main.MAX_JSON_BODY_BYTES)) is True
    assert server_main._check_content_length(str(server_main.MAX_JSON_BODY_BYTES + 1)) is False


def test_rate_limit_enforces_cap_and_bounds_client_memory(monkeypatch):
    monkeypatch.setattr(server_main, "RATE_LIMIT_MAX_REQUESTS", 3)
    monkeypatch.setattr(server_main, "RATE_LIMIT_MAX_CLIENTS", 5)

    assert server_main._check_rate_limit("10.0.0.1", now=100.0) is True
    assert server_main._check_rate_limit("10.0.0.1", now=101.0) is True
    assert server_main._check_rate_limit("10.0.0.1", now=102.0) is True
    assert server_main._check_rate_limit("10.0.0.1", now=103.0) is False

    # Window slides forward: client allowed again after window expires.
    assert server_main._check_rate_limit("10.0.0.1", now=165.0) is True

    # Many rotating client IPs must never exceed RATE_LIMIT_MAX_CLIENTS.
    for i in range(20):
        assert server_main._check_rate_limit(f"192.168.0.{i}", now=200.0 + i * 0.01) is True
    assert len(server_main._RATE_LIMIT_STATE) <= 5


def test_fit_rejects_cg_with_smooth_terms():
    req = server_main.FitRequest(
        formula="y ~ pb(x)",
        family="NO",
        data=_sample_normal_data(),
        method="CG",
    )
    with pytest.raises(HTTPException) as exc_info:
        server_main.fit(req, None)
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["code"] == "smooth_terms_unsupported_for_method"


def test_fit_rejects_experimental_family_fail_closed():
    req = server_main.FitRequest(
        formula="y ~ x",
        family="DEL",
        data={"y": [0, 1, 2, 1, 3, 2, 0, 4], "x": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]},
        method="RS",
    )
    with pytest.raises(HTTPException) as exc_info:
        server_main.fit(req, None)
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["type"] == "production_family_rejected"


def test_fit_rejects_nonevidenced_cg_route_for_admitted_family():
    # GA/CG is admitted since D2.281 (live-R CG parity), so the old GA case no
    # longer exercises the gate.  NBII is production-admitted on RS but has no
    # live-R CG evidence (its R-side CG fit errors), i.e. still experimental.
    req = server_main.FitRequest(
        formula="y ~ x",
        family="NBII",
        data={"y": [1.2, 2.3, 1.8, 3.1, 2.5, 1.9, 2.8, 3.4], "x": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]},
        method="CG",
    )
    with pytest.raises(HTTPException) as exc_info:
        server_main.fit(req, None)
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["type"] == "production_family_rejected"


def test_fit_rejects_invalid_max_iter():
    req = server_main.FitRequest(
        formula="y ~ x",
        family="NO",
        data=_sample_normal_data(),
        max_iter=5000,
    )
    with pytest.raises(HTTPException) as exc_info:
        server_main.fit(req, None)
    assert exc_info.value.status_code == 400


def test_distributions_select_ranks_admitted_and_rejects_experimental():
    rng = np.random.default_rng(7)
    x = rng.normal(size=40)
    y = np.exp(0.5 + 0.3 * x + rng.normal(scale=0.2, size=40))
    req = server_main.DistributionSelectRequest(
        formula="y ~ x",
        data={"y": y.tolist(), "x": x.tolist()},
        candidate_families=["NO", "GA", "DEL"],
        method="RS",
    )
    out = server_main.distributions_select(req, None)
    assert out["selected_family"] in {"NO", "GA"}
    by_family = {row["family"]: row for row in out["candidates"]}
    assert by_family["DEL"]["error"] == "family_not_admitted"
    assert "gaic" in by_family["GA"]


def test_predict_missing_variable_returns_400():
    fit_out = server_main.fit(
        server_main.FitRequest(formula="y ~ x", family="NO", data=_sample_normal_data()),
        None,
    )
    with pytest.raises(HTTPException) as exc_info:
        server_main.predict(
            server_main.PredictRequest(model_id=fit_out["model_id"], newdata={"wrong_col": [1.0]}),
            None,
        )
    assert exc_info.value.status_code == 400
