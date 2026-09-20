"""gRPC server smoke tests (optional dependency)."""

import time
from concurrent.futures import ThreadPoolExecutor

import importlib.util

import pytest

GRPC_AVAILABLE = importlib.util.find_spec("grpc") is not None
grpc = (
    pytest.importorskip("grpc", reason="grpcio not installed")
    if GRPC_AVAILABLE
    else None
)


def test_grpc_error_text_preserves_prediction_schema_envelope() -> None:
    """Prediction schema errors should stay machine-readable over gRPC."""
    import json

    from omnilss.api.grpc import server as grpc_server
    from omnilss.prediction import PredictionSchemaError

    error = PredictionSchemaError(
        "Factor term 'factor(grp)' contains unseen levels ['c']",
        parameter="mu",
        term="factor(grp)",
        reason="unseen factor levels ['c']",
        code="unseen_factor_levels",
    )

    payload = json.loads(grpc_server._error_text(error))

    assert payload == {
        "type": "prediction_schema_error",
        "code": "unseen_factor_levels",
        "parameter": "mu",
        "term": "factor(grp)",
        "reason": "unseen factor levels ['c']",
        "message": "Factor term 'factor(grp)' contains unseen levels ['c']",
        "expected_schema": None,
        "actual_schema": None,
        "state_version": None,
    }


def test_top_level_prediction_schema_exports() -> None:
    """The schema-safe prediction API should be importable from omnilss."""
    import omnilss

    assert omnilss.PredictionSchemaError is not None
    assert callable(omnilss.build_prediction_design_matrix)


def test_grpc_runtime_gaps_reflect_optional_dependencies() -> None:
    """Runtime diagnostics should expose missing protobuf/grpc pieces."""
    from omnilss.api.grpc import server as grpc_server

    gaps = grpc_server._grpc_runtime_gaps()

    if importlib.util.find_spec("grpc") is None:
        assert "grpcio" in gaps
    else:
        assert "grpcio" not in gaps

    if importlib.util.find_spec("google") is None:
        assert "protobuf" in gaps
    elif importlib.util.find_spec("google.protobuf") is None:
        assert "protobuf" in gaps
    else:
        assert "protobuf" not in gaps


def test_create_service_reports_actionable_runtime_gap(monkeypatch) -> None:
    """create_service should fail with install guidance when runtime is incomplete."""
    from omnilss.api.grpc import server as grpc_server

    monkeypatch.setattr(grpc_server, "_grpc_runtime_gaps", lambda: ["protobuf"])

    with pytest.raises(RuntimeError) as excinfo:
        grpc_server.create_service()

    message = str(excinfo.value)
    assert "Missing: protobuf" in message
    assert "pip install 'omnilss[grpc]'" in message
    assert "omnilss.api.grpc.generated" in message


def test_model_registry_save_is_thread_safe() -> None:
    """Concurrent saves should not corrupt in-memory/db registry state."""
    from omnilss.api.grpc import server as grpc_server

    from omnilss.algorithms.rs_algorithm import rs_fit

    registry = grpc_server._ModelRegistry()
    data = {"x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0]}
    model = rs_fit(formula="y ~ x", family="NO", data=data, max_iter=2, verbose=False)
    try:
        with ThreadPoolExecutor(max_workers=10) as pool:
            model_ids = list(pool.map(lambda _: registry.save(model), range(10)))
        assert len(set(model_ids)) == 10
        assert len(registry.list_ids()) >= 10
    finally:
        for model_id in list(registry.list_ids()):
            registry.delete(model_id)


@pytest.mark.skipif(not GRPC_AVAILABLE, reason="grpcio not installed")
def test_grpc_predict_response_preserves_prediction_schema_error(monkeypatch) -> None:
    """The direct Predict service should return schema errors as JSON text."""
    import json

    from omnilss.api.grpc import server as grpc_server
    from omnilss.api.grpc.generated import predict_pb2
    from omnilss.prediction import PredictionSchemaError

    class BrokenModel:
        def predict_params(self, newdata):  # noqa: ANN001, ARG002
            raise PredictionSchemaError(
                "Factor term 'factor(grp)' contains unseen levels ['c']",
                parameter="mu",
                term="factor(grp)",
                reason="unseen factor levels ['c']",
                code="unseen_factor_levels",
            )

    try:
        service, *_ = grpc_server.create_service()
    except RuntimeError as exc:
        pytest.skip(f"gRPC stubs/runtime unavailable in environment: {exc}")

    monkeypatch.setattr(grpc_server.REGISTRY, "load", lambda model_id: BrokenModel())

    response = service.Predict(
        predict_pb2.PredictRequest(
            model_id="broken", newdata_json=json.dumps({"grp": ["c"]})
        ),
        None,
    )

    assert response.success is False
    assert json.loads(response.params_json) == {}
    payload = json.loads(response.error)
    assert payload["type"] == "prediction_schema_error"
    assert payload["code"] == "unseen_factor_levels"
    assert payload["parameter"] == "mu"


@pytest.mark.skipif(not GRPC_AVAILABLE, reason="grpcio not installed")
def test_grpc_server_starts() -> None:
    """Server should start when grpc runtime and stubs are available."""
    from omnilss.api.grpc.server import serve

    try:
        server = serve(host="127.0.0.1", port=59051)
    except RuntimeError as exc:
        pytest.skip(f"gRPC stubs/runtime unavailable in environment: {exc}")
    assert server is not None
    server.stop(grace=0)


@pytest.mark.skipif(not GRPC_AVAILABLE, reason="grpcio not installed")
def test_grpc_fit_request() -> None:
    """Fit request should return a model id on success."""
    import json

    import numpy as np

    from omnilss.api.grpc.server import serve

    try:
        from omnilss.api.grpc.generated import fit_pb2, fit_pb2_grpc
    except Exception as exc:
        pytest.skip(f"Generated protobuf stubs unavailable: {exc}")

    try:
        server = serve(host="127.0.0.1", port=59052)
    except RuntimeError as exc:
        pytest.skip(f"gRPC stubs/runtime unavailable in environment: {exc}")

    time.sleep(0.2)
    try:
        channel = grpc.insecure_channel("127.0.0.1:59052")
        stub = fit_pb2_grpc.FitServiceStub(channel)

        n = 50
        data = {"y": np.random.randn(n).tolist(), "x": np.linspace(0, 1, n).tolist()}
        req = fit_pb2.FitRequest(
            formula="y ~ x",
            family="NO",
            data_json=json.dumps(data),
            sigma_formula="~ 1",
            method="RS",
            max_iter=20,
        )
        resp = stub.Fit(req, timeout=10)
        assert resp.success, f"Fit failed: {resp.error}"
        assert len(resp.model_id) > 0
        assert resp.deviance > 0
        assert resp.iterations >= 0
    finally:
        server.stop(grace=0)


@pytest.mark.skipif(not GRPC_AVAILABLE, reason="grpcio not installed")
def test_grpc_capability_matrix_service_direct() -> None:
    """CapabilityMatrix should expose the runtime capability matrix."""
    import json

    from omnilss.family_capabilities import capability_matrix
    from omnilss.api.grpc import server as grpc_server
    from omnilss.api.grpc.generated import capability_pb2

    try:
        service, *_ = grpc_server.create_service()
    except RuntimeError as exc:
        pytest.skip(f"gRPC stubs/runtime unavailable in environment: {exc}")

    resp = service.CapabilityMatrix(capability_pb2.CapabilityMatrixRequest(), None)

    assert resp.success, resp.error
    assert json.loads(resp.matrix_json) == capability_matrix()


@pytest.mark.skipif(not GRPC_AVAILABLE, reason="grpcio not installed")
def test_grpc_capability_matrix_request() -> None:
    """CapabilityMatrix request should work over the gRPC boundary."""
    import json

    from omnilss.family_capabilities import capability_matrix
    from omnilss.api.grpc.server import serve

    try:
        from omnilss.api.grpc.generated import capability_pb2, capability_pb2_grpc
    except Exception as exc:
        pytest.skip(f"Generated protobuf stubs unavailable: {exc}")

    try:
        server = serve(host="127.0.0.1", port=59053)
    except RuntimeError as exc:
        pytest.skip(f"gRPC stubs/runtime unavailable in environment: {exc}")

    time.sleep(0.2)
    try:
        channel = grpc.insecure_channel("127.0.0.1:59053")
        stub = capability_pb2_grpc.CapabilityServiceStub(channel)
        resp = stub.CapabilityMatrix(
            capability_pb2.CapabilityMatrixRequest(), timeout=10
        )
        assert resp.success, resp.error
        assert json.loads(resp.matrix_json) == capability_matrix()
    finally:
        server.stop(grace=0)

@pytest.mark.skipif(not GRPC_AVAILABLE, reason="grpcio not installed")
def test_grpc_route_capability_response_matches_runtime_report() -> None:
    """Capability RPC should expose method-route preflight decisions."""
    import json

    from omnilss.api.grpc import server as grpc_server
    from omnilss.api.grpc.generated import capability_pb2
    from omnilss.family_capabilities import method_route_capability_report

    try:
        service, *_ = grpc_server.create_service()
    except RuntimeError as exc:
        pytest.skip(f"gRPC stubs/runtime unavailable in environment: {exc}")

    response = service.RouteCapability(
        capability_pb2.RouteCapabilityRequest(family="GA", method="RS", strict=True),
        None,
    )

    assert response.success is True
    assert response.error == ""
    assert json.loads(response.report_json) == method_route_capability_report(
        "GA", "RS", strict=True
    )


@pytest.mark.skipif(not GRPC_AVAILABLE, reason="grpcio not installed")
def test_grpc_route_capability_requires_family_and_method() -> None:
    """Capability RPC should reject empty method-route preflight keys."""
    from omnilss.api.grpc import server as grpc_server
    from omnilss.api.grpc.generated import capability_pb2

    try:
        service, *_ = grpc_server.create_service()
    except RuntimeError as exc:
        pytest.skip(f"gRPC stubs/runtime unavailable in environment: {exc}")

    response = service.RouteCapability(
        capability_pb2.RouteCapabilityRequest(family="NO", method="", strict=False),
        None,
    )

    assert response.success is False
    assert response.report_json == "{}"
    assert response.error == "ValueError error"


def test_model_registry_list_and_delete(tmp_path, monkeypatch) -> None:
    """Registry should expose list/delete helpers for service management."""
    from omnilss.api.grpc import server as grpc_server

    monkeypatch.setattr(grpc_server, "MODEL_STORE", tmp_path / "models")
    grpc_server.MODEL_STORE.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        grpc_server, "MODEL_DB", tmp_path / "models" / "registry.sqlite3"
    )

    monkeypatch.setattr(grpc_server, "save_model_json", lambda model, path: path.write_text("{}"))
    registry = grpc_server._ModelRegistry()

    class Dummy:
        g_dev = 1.0

    model_id = registry.save(Dummy())
    assert model_id in registry.list_ids()
    assert registry.delete(model_id) is True
    assert model_id not in registry.list_ids()
    assert registry.delete(model_id) is False


def test_model_registry_recovers_index_from_sqlite(tmp_path, monkeypatch) -> None:
    """Registry should restore saved model ids after process restart."""
    from omnilss.api.grpc import server as grpc_server

    monkeypatch.setattr(grpc_server, "MODEL_STORE", tmp_path / "models")
    grpc_server.MODEL_STORE.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        grpc_server, "MODEL_DB", tmp_path / "models" / "registry.sqlite3"
    )

    class Dummy:
        g_dev = 1.0

    monkeypatch.setattr(grpc_server, "save_model_json", lambda model, path: path.write_text("{}"))
    first = grpc_server._ModelRegistry()
    model_id = first.save(Dummy())

    second = grpc_server._ModelRegistry()
    assert model_id in second.list_ids()


def test_model_registry_migrates_legacy_sqlite_schema(tmp_path, monkeypatch) -> None:
    """Legacy SQLite registries should gain capability replay columns."""
    import sqlite3

    from omnilss.api.grpc import server as grpc_server

    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    db_path = model_dir / "registry.sqlite3"
    artifact_path = model_dir / "legacy.omnilss"
    artifact_path.write_text("not-a-zip")
    with sqlite3.connect(db_path) as db:
        db.execute(
            "CREATE TABLE models ("
            "id TEXT PRIMARY KEY, path TEXT NOT NULL, created_at REAL NOT NULL"
            ")"
        )
        db.execute(
            "INSERT INTO models(id, path, created_at) VALUES (?, ?, ?)",
            ("legacy", str(artifact_path), time.time()),
        )

    monkeypatch.setattr(grpc_server, "MODEL_STORE", model_dir)
    monkeypatch.setattr(grpc_server, "MODEL_DB", db_path)

    registry = grpc_server._ModelRegistry()

    assert registry.list_ids() == ["legacy"]
    with sqlite3.connect(db_path) as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(models)")}
        row = db.execute(
            "SELECT schema_version, capability_report_json, capability_ok "
            "FROM models WHERE id = 'legacy'"
        ).fetchone()

    assert {
        "schema_version",
        "family",
        "capability_matrix_version",
        "implementation_status",
        "capability_json",
        "capability_report_json",
        "capability_ok",
    }.issubset(columns)
    assert row[0] == grpc_server._MODEL_REGISTRY_SCHEMA_VERSION
    assert "capability_replay_failed" in row[1]
    assert row[2] == 0


def test_model_registry_persists_capability_snapshot_and_replay_report(
    tmp_path, monkeypatch
) -> None:
    """SQLite registry rows should include replayable family capability metadata."""
    import json
    import sqlite3

    import numpy as np

    from omnilss import CAPABILITY_MATRIX_VERSION, gamlss
    from omnilss.api.grpc import server as grpc_server

    monkeypatch.setattr(grpc_server, "MODEL_STORE", tmp_path / "models")
    grpc_server.MODEL_STORE.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        grpc_server, "MODEL_DB", tmp_path / "models" / "registry.sqlite3"
    )

    x = np.linspace(-1.0, 1.0, 24)
    y = 1.0 + 0.25 * x + np.random.default_rng(203).normal(scale=0.1, size=x.size)
    model = gamlss("y ~ x", family="NO", data={"y": y, "x": x}, max_iter=5)

    first = grpc_server._ModelRegistry()
    model_id = first.save(model)

    with sqlite3.connect(grpc_server.MODEL_DB) as db:
        row = db.execute(
            "SELECT schema_version, family, capability_matrix_version, "
            "implementation_status, capability_json, "
            "capability_report_json, capability_ok "
            "FROM models WHERE id = ?",
            (model_id,),
        ).fetchone()

    assert row is not None
    (
        schema_version,
        family,
        capability_matrix_version,
        implementation_status,
        capability_json,
        capability_report_json,
        capability_ok,
    ) = row
    assert schema_version == grpc_server._MODEL_REGISTRY_SCHEMA_VERSION
    assert family == "NO"
    assert capability_matrix_version == CAPABILITY_MATRIX_VERSION
    assert implementation_status == "implemented"
    assert json.loads(capability_json)["features"]["r_consistency"] == "validated"
    assert json.loads(capability_report_json)["ok"] is True
    assert capability_ok == 1

    second = grpc_server._ModelRegistry()
    assert model_id in second.list_ids()


def test_grpc_list_models_and_delete_model_service_direct(monkeypatch) -> None:
    """Fit service management RPCs should expose list/delete semantics."""
    from omnilss.api.grpc import server as grpc_server
    from omnilss.api.grpc.generated import fit_pb2

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

    try:
        service, *_ = grpc_server.create_service()
    except RuntimeError as exc:
        pytest.skip(f"gRPC stubs/runtime unavailable in environment: {exc}")

    dummy = DummyRegistry()
    monkeypatch.setattr(grpc_server, "REGISTRY", dummy)

    listed = service.ListModels(fit_pb2.ListModelsRequest(), None)
    assert listed.success is True
    assert sorted(listed.model_ids) == ["m1", "m2"]

    deleted = service.DeleteModel(fit_pb2.DeleteModelRequest(model_id="m1"), None)
    assert deleted.success is True
    assert deleted.deleted is True

    missing = service.DeleteModel(
        fit_pb2.DeleteModelRequest(model_id="missing"), None
    )
    assert missing.success is True
    assert missing.deleted is False


def test_grpc_batch_fit_service_direct(monkeypatch) -> None:
    """BatchFit should fit multiple requests through one Core RPC."""
    import json

    from omnilss.api.grpc import server as grpc_server
    from omnilss.api.grpc.generated import fit_pb2

    class DummyModel:
        def __init__(self, deviance: float) -> None:
            self.g_dev = deviance
            self.iter = 2
            self.additional_slots = {"rs_converged": True}

    class DummyRegistry:
        def __init__(self) -> None:
            self.saved = 0

        def save(self, model) -> str:
            self.saved += 1
            return f"model-{self.saved}"

    calls = []

    def fake_gamlss(**kwargs):
        calls.append(kwargs)
        return DummyModel(10.0 + len(calls))

    try:
        service, *_ = grpc_server.create_service()
    except RuntimeError as exc:
        pytest.skip(f"gRPC stubs/runtime unavailable in environment: {exc}")

    monkeypatch.setattr(grpc_server, "gamlss", fake_gamlss)
    monkeypatch.setattr(grpc_server, "REGISTRY", DummyRegistry())

    request = fit_pb2.BatchFitRequest(
        requests=[
            fit_pb2.FitRequest(
                formula="y ~ x",
                family="NO",
                data_json=json.dumps({"y": [1.0, 2.0], "x": [0.0, 1.0]}),
                sigma_formula="~ 1",
                method="RS",
                max_iter=3,
            ),
            fit_pb2.FitRequest(
                formula="y ~ x",
                family="GA",
                data_json=json.dumps({"y": [2.0, 3.0], "x": [0.0, 1.0]}),
                sigma_formula="~ 1",
                method="RS",
                max_iter=3,
            ),
        ]
    )

    response = service.BatchFit(request, None)

    assert response.success is True
    assert [r.model_id for r in response.responses] == ["model-1", "model-2"]
    assert [r.deviance for r in response.responses] == [11.0, 12.0]
    assert [call["family"] for call in calls] == ["NO", "GA"]



def test_grpc_batch_fit_reports_partial_failures(monkeypatch) -> None:
    """BatchFit should surface partial failures in top-level status."""
    import json

    from omnilss.api.grpc import server as grpc_server
    from omnilss.api.grpc.generated import fit_pb2

    class DummyModel:
        def __init__(self, deviance: float) -> None:
            self.g_dev = deviance
            self.iter = 1
            self.additional_slots = {"rs_converged": True}

    class DummyRegistry:
        def __init__(self) -> None:
            self.saved = 0

        def save(self, model) -> str:
            self.saved += 1
            return f"model-{self.saved}"

    def fake_gamlss(**kwargs):
        if kwargs.get("family") == "BROKEN":
            raise ValueError("simulated failure")
        return DummyModel(8.0)

    try:
        service, *_ = grpc_server.create_service()
    except RuntimeError as exc:
        pytest.skip(f"gRPC stubs/runtime unavailable in environment: {exc}")

    monkeypatch.setattr(grpc_server, "gamlss", fake_gamlss)
    monkeypatch.setattr(grpc_server, "REGISTRY", DummyRegistry())

    req = fit_pb2.BatchFitRequest(
        requests=[
            fit_pb2.FitRequest(
                formula="y ~ x",
                family="NO",
                data_json=json.dumps({"y": [1.0], "x": [0.0]}),
            ),
            fit_pb2.FitRequest(
                formula="y ~ x",
                family="BROKEN",
                data_json=json.dumps({"y": [1.0], "x": [0.0]}),
            ),
        ]
    )

    resp = service.BatchFit(req, None)

    assert resp.success is False
    assert "1 of 2 fits failed" in resp.error
    assert [r.success for r in resp.responses] == [True, False]


def test_grpc_fit_error_returns_fit_response(monkeypatch) -> None:
    """Unary Fit failures must return FitResponse, not BatchFitResponse."""
    import json

    from omnilss.api.grpc import server as grpc_server
    from omnilss.api.grpc.generated import fit_pb2

    try:
        service, *_ = grpc_server.create_service()
    except RuntimeError as exc:
        pytest.skip(f"gRPC stubs/runtime unavailable in environment: {exc}")

    def fake_gamlss(**kwargs):  # noqa: ANN003
        raise ValueError("simulated unary fit failure")

    monkeypatch.setattr(grpc_server, "gamlss", fake_gamlss)

    response = service.Fit(
        fit_pb2.FitRequest(
            formula="y ~ x",
            family="NO",
            data_json=json.dumps({"y": [1.0], "x": [0.0]}),
            method="RS",
        ),
        None,
    )

    assert isinstance(response, fit_pb2.FitResponse)
    assert response.success is False
    assert response.model_id == ""
    assert response.error == "ValueError error"
    assert "simulated unary fit failure" not in response.error


def test_grpc_fit_blocks_placeholder_family_before_backend(monkeypatch) -> None:
    """Fit should reject placeholder families at the service boundary."""
    import json

    from omnilss.api.grpc import server as grpc_server
    from omnilss.api.grpc.generated import fit_pb2

    try:
        service, *_ = grpc_server.create_service()
    except RuntimeError as exc:
        pytest.skip(f"gRPC stubs/runtime unavailable in environment: {exc}")

    def fake_gamlss(**kwargs):  # noqa: ANN003
        raise AssertionError("placeholder family should be blocked before gamlss")

    monkeypatch.setattr(grpc_server, "gamlss", fake_gamlss)

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
    payload = json.loads(response.error)
    assert payload["code"] == "placeholder_requires_opt_in"
    assert payload["implementation_status"] == "placeholder"


def test_grpc_predict_supports_column_vectors(monkeypatch) -> None:
    """Predict should accept structured column vectors without JSON payload."""
    from omnilss.api.grpc import server as grpc_server
    from omnilss.api.grpc.generated import predict_pb2

    class DummyModel:
        def predict_params(self, newdata):
            x = newdata["x"]
            return {"mu": x, "sigma": [1.0 for _ in x]}

    try:
        service, *_ = grpc_server.create_service()
    except RuntimeError as exc:
        pytest.skip(f"gRPC stubs/runtime unavailable in environment: {exc}")

    monkeypatch.setattr(grpc_server.REGISTRY, "load", lambda _model_id: DummyModel())

    response = service.Predict(
        predict_pb2.PredictRequest(
            model_id="dummy",
            newdata_columns=[predict_pb2.ColumnVector(name="x", values=[1.0, 2.0])],
        ),
        None,
    )

    assert response.success is True
    assert response.error == ""
    assert len(response.params) == 2
    by_name = {p.name: list(p.values) for p in response.params}
    assert by_name["mu"] == [1.0, 2.0]
    assert by_name["sigma"] == [1.0, 1.0]


@pytest.mark.skipif(not GRPC_AVAILABLE, reason="grpcio not installed")
def test_grpc_predict_is_restart_safe_with_persistent_registry(tmp_path, monkeypatch) -> None:
    """A model fitted before restart should remain predictable after restart."""
    import importlib
    import json
    import numpy as np

    monkeypatch.setenv("OMNILSS_MODEL_STORE_DIR", str(tmp_path / "models"))
    monkeypatch.setenv(
        "OMNILSS_MODEL_DB_PATH", str(tmp_path / "models" / "registry.sqlite3")
    )

    import omnilss.api.grpc.server as grpc_server

    grpc_server = importlib.reload(grpc_server)

    try:
        from omnilss.api.grpc.generated import fit_pb2, fit_pb2_grpc, predict_pb2, predict_pb2_grpc
    except Exception as exc:
        pytest.skip(f"Generated protobuf stubs unavailable: {exc}")

    server1 = grpc_server.serve(host="127.0.0.1", port=59056)
    time.sleep(0.2)
    try:
        channel = grpc.insecure_channel("127.0.0.1:59056")
        fit_stub = fit_pb2_grpc.FitServiceStub(channel)
        predict_stub = predict_pb2_grpc.PredictServiceStub(channel)

        n = 40
        data = {"y": np.random.randn(n).tolist(), "x": np.linspace(0, 1, n).tolist()}
        fit_resp = fit_stub.Fit(
            fit_pb2.FitRequest(
                formula="y ~ x",
                family="NO",
                data_json=json.dumps(data),
                sigma_formula="~ 1",
                method="RS",
                max_iter=10,
            ),
            timeout=10,
        )
        assert fit_resp.success, f"Fit failed: {fit_resp.error}"
        model_id = fit_resp.model_id

        pred_resp = predict_stub.Predict(
            predict_pb2.PredictRequest(
                model_id=model_id,
                newdata_json=json.dumps({"x": [0.1, 0.5, 0.9]}),
            ),
            timeout=10,
        )
        assert pred_resp.success, f"Predict before restart failed: {pred_resp.error}"
    finally:
        server1.stop(grace=0)

    grpc_server = importlib.reload(grpc_server)
    server2 = grpc_server.serve(host="127.0.0.1", port=59057)
    time.sleep(0.2)
    try:
        channel = grpc.insecure_channel("127.0.0.1:59057")
        predict_stub = predict_pb2_grpc.PredictServiceStub(channel)

        pred_resp = predict_stub.Predict(
            predict_pb2.PredictRequest(
                model_id=model_id,
                newdata_json=json.dumps({"x": [0.2, 0.4]}),
            ),
            timeout=10,
        )
        assert pred_resp.success, f"Predict after restart failed: {pred_resp.error}"
        payload = json.loads(pred_resp.params_json)
        assert "mu" in payload
        assert len(payload["mu"]) == 2
    finally:
        server2.stop(grace=0)
