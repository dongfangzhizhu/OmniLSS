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
