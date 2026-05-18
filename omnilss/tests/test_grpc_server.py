"""gRPC server smoke tests (optional dependency)."""

import time

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
    assert "non-empty family and method" in response.error


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
