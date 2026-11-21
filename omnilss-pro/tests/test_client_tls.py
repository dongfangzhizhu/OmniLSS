"""Phase 5.2: client TLS + token-injection behaviour (mock grpc)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class _FakeChannel:
    """Minimal channel exposing the RPC methods generated stubs need."""

    def __init__(
        self,
        address: str,
        secure: bool = False,
        creds: object | None = None,
    ) -> None:
        self.address = address
        self.secure = secure
        self.creds = creds

    def unary_unary(self, *args, **kwargs):
        return object()

    def unary_stream(self, *args, **kwargs):
        return object()

    def stream_unary(self, *args, **kwargs):
        return object()

    def stream_stream(self, *args, **kwargs):
        return object()


try:
    import grpc  # noqa: F401
except ImportError:
    _grpc_stub = types.ModuleType("grpc")
    _grpc_stub.insecure_channel = lambda addr: _FakeChannel(addr, secure=False)
    _grpc_stub.secure_channel = lambda addr, creds: _FakeChannel(addr, secure=True, creds=creds)
    _grpc_stub.ssl_channel_credentials = lambda root_certificates=None: ("creds", root_certificates)
    sys.modules["grpc"] = _grpc_stub

try:
    import omnilss_pro.client as client_mod
    from omnilss_pro.client import OmniLSSCoreClient
except Exception as exc:  # pragma: no cover
    pytest.skip(f"omnilss-pro client unavailable: {exc}", allow_module_level=True)


@pytest.fixture(autouse=True)
def _mock_grpc_channels(monkeypatch):
    """Always intercept channel factories so tests pass with or without real grpcio."""
    monkeypatch.setattr(
        client_mod.grpc,
        "insecure_channel",
        lambda addr: _FakeChannel(addr, secure=False),
    )
    monkeypatch.setattr(
        client_mod.grpc,
        "secure_channel",
        lambda addr, creds: _FakeChannel(addr, secure=True, creds=creds),
    )
    monkeypatch.setattr(
        client_mod.grpc,
        "ssl_channel_credentials",
        lambda root_certificates=None: ("creds", root_certificates),
    )


def test_client_insecure_by_default():
    client = OmniLSSCoreClient(host="h", port=1)
    assert client._channel.secure is False
    assert client._channel.address == "h:1"
    assert client._metadata == ()


def test_client_tls_uses_secure_channel():
    client = OmniLSSCoreClient(host="h", port=1, tls=True, root_cert=b"PEM")
    assert client._channel.secure is True
    assert client._channel.address == "h:1"
    assert client._channel.creds == ("creds", b"PEM")


def test_client_token_injected_into_metadata():
    client = OmniLSSCoreClient(host="h", port=1, api_token="sekret")
    assert ("authorization", "Bearer sekret") in client._metadata


def test_client_metadata_absent_without_token():
    client = OmniLSSCoreClient(host="h", port=1)
    assert client._metadata == ()
