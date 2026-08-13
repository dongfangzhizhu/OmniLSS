"""Phase 8.1: OmniLSS Pro SDK coverage of capability/version/health endpoints.

Verifies that `omnilss_pro.client.OmniLSSCoreClient` can drive the frozen
CapabilityService contract (CapabilityMatrix / RouteCapability / Version /
Health) against the Core gRPC server.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from omnilss_pro.api.grpc.generated import capability_pb2, capability_pb2_grpc
    from omnilss_pro.client import OmniLSSCoreClient
except Exception as exc:  # pragma: no cover
    pytest.skip(f"omnilss-pro capability client unavailable: {exc}", allow_module_level=True)


@pytest.fixture()
def client():
    return OmniLSSCoreClient(host="h", port=1)


def test_capability_stub_is_created(client):
    assert isinstance(client._capability_stub, capability_pb2_grpc.CapabilityServiceStub)


def test_capability_matrix_parses_json(client):
    response = capability_pb2.CapabilityMatrixResponse(
        matrix_json='{"NO": "E3", "ZIP": "placeholder"}', success=True
    )
    client._capability_stub.CapabilityMatrix = mock.Mock(return_value=response)
    result = client.capability_matrix()
    assert result == {"NO": "E3", "ZIP": "placeholder"}
    client._capability_stub.CapabilityMatrix.assert_called_once()
    call = client._capability_stub.CapabilityMatrix.call_args
    assert call.kwargs["metadata"] == ()


def test_capability_matrix_failure_raises(client):
    response = capability_pb2.CapabilityMatrixResponse(success=False, error="gate error")
    client._capability_stub.CapabilityMatrix = mock.Mock(return_value=response)
    with pytest.raises(RuntimeError, match="gate error"):
        client.capability_matrix()


def test_route_capability_passes_args_and_parses(client):
    response = capability_pb2.RouteCapabilityResponse(
        report_json='{"admitted": false, "reason": "placeholder"}', success=True
    )
    client._capability_stub.RouteCapability = mock.Mock(return_value=response)
    result = client.route_capability("ZIP", method="RS", strict=True)
    assert result == {"admitted": False, "reason": "placeholder"}
    request = client._capability_stub.RouteCapability.call_args[0][0]
    assert request.family == "ZIP"
    assert request.method == "RS"
    assert request.strict is True


def test_route_capability_failure_raises(client):
    response = capability_pb2.RouteCapabilityResponse(success=False, error="rejected")
    client._capability_stub.RouteCapability = mock.Mock(return_value=response)
    with pytest.raises(RuntimeError, match="rejected"):
        client.route_capability("NO")


def test_version_maps_fields(client):
    response = capability_pb2.VersionResponse(
        version="0.4.0",
        python_version="3.10.16",
        jax_version="0.6.2",
        x64_enabled="true",
        success=True,
    )
    client._capability_stub.Version = mock.Mock(return_value=response)
    result = client.version()
    assert result == {
        "version": "0.4.0",
        "python_version": "3.10.16",
        "jax_version": "0.6.2",
        "x64_enabled": "true",
    }


def test_version_failure_raises(client):
    response = capability_pb2.VersionResponse(success=False, error="boom")
    client._capability_stub.Version = mock.Mock(return_value=response)
    with pytest.raises(RuntimeError, match="boom"):
        client.version()


def test_health_returns_status_and_model_count(client):
    response = capability_pb2.HealthResponse(
        status="ok", model_count=3, success=True
    )
    client._capability_stub.Health = mock.Mock(return_value=response)
    result = client.health()
    assert result == {"status": "ok", "model_count": 3, "success": True}


def test_health_degraded_is_returned_not_raised(client):
    response = capability_pb2.HealthResponse(
        status="degraded", model_count=0, success=False, error="registry unavailable"
    )
    client._capability_stub.Health = mock.Mock(return_value=response)
    result = client.health()
    assert result["status"] == "degraded"
    assert result["success"] is False


def test_capability_endpoints_send_token_metadata():
    client = OmniLSSCoreClient(host="h", port=1, api_token="sekret")
    defaults = {
        "CapabilityMatrix": {"matrix_json": "{}"},
        "RouteCapability": {"report_json": "{}"},
        "Version": {},
        "Health": {},
    }
    for method_name in ("CapabilityMatrix", "RouteCapability", "Version", "Health"):
        attr = getattr(capability_pb2, f"{method_name}Response")
        response = attr(success=True, **defaults[method_name])
        stub_method = mock.Mock(return_value=response)
        setattr(client._capability_stub, method_name, stub_method)
        client_method = getattr(client, {
            "CapabilityMatrix": "capability_matrix",
            "RouteCapability": "route_capability",
            "Version": "version",
            "Health": "health",
        }[method_name])
        if method_name == "RouteCapability":
            client_method("NO")
        else:
            client_method()
        assert stub_method.call_args.kwargs["metadata"] == (
            ("authorization", "Bearer sekret"),
        )
