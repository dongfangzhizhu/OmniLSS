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
