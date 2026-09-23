"""Phase 5.2: TLS support (gRPC server + REST documented).

- gRPC server binds a secure port when OMNILSS_TLS_CERT/OMNILSS_TLS_KEY are
  set (ssl_server_credentials); otherwise insecure (token still enforced).
- REST: uvicorn TLS flags documented in docker-compose command comment.
"""

from __future__ import annotations

import sys
import os
import subprocess
import time
import json
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

SERVER_SOURCE = Path(SRC / "omnilss" / "api" / "grpc" / "server.py")


def test_server_supports_tls_cert_env():
    """serve() reads OMNILSS_TLS_CERT/OMNILSS_TLS_KEY and uses secure port."""
    text = SERVER_SOURCE.read_text(encoding="utf-8")
    assert 'os.getenv("OMNILSS_TLS_CERT", "")' in text
    assert 'os.getenv("OMNILSS_TLS_KEY", "")' in text
    assert "grpc.ssl_server_credentials" in text
    assert "add_secure_port" in text
    assert "add_insecure_port" in text  # fallback when no certs


def test_client_supports_tls_and_token():
    """omnilss-pro client accepts tls/root_cert/api_token and injects auth."""
    client_src = (
        Path(__file__).resolve().parents[2] / "omnilss-pro" / "src" / "omnilss_pro" / "client.py"
    )
    text = client_src.read_text(encoding="utf-8")
    assert "tls: bool = False" in text
    assert "root_cert" in text
    assert "api_token" in text
    assert "grpc.ssl_channel_credentials" in text
    assert "grpc.secure_channel" in text
    assert "metadata=self._metadata" in text


@pytest.mark.skipif(not __import__("importlib.util").util.find_spec("grpc"), reason="grpcio not installed")
def test_grpc_tls_metrics_requires_trusted_root_and_token(tmp_path, monkeypatch):
    """A real TLS channel must verify the certificate and Bearer metadata."""
    import grpc
    from omnilss.api.grpc import server as grpc_server
    from omnilss.api.grpc.generated import capability_pb2, capability_pb2_grpc

    openssl = "openssl"
    ca_key = tmp_path / "ca.key"
    ca_cert = tmp_path / "ca.crt"
    server_key = tmp_path / "server.key"
    server_csr = tmp_path / "server.csr"
    server_cert = tmp_path / "server.crt"
    ext = tmp_path / "server.ext"
    config = tmp_path / "openssl.cnf"
    config.write_text("[req]\ndistinguished_name=req_distinguished_name\n[req_distinguished_name]\n\n", encoding="utf-8")
    subprocess.run([openssl, "genrsa", "-out", str(ca_key), "2048"], check=True, capture_output=True)
    subprocess.run([openssl, "req", "-x509", "-new", "-nodes", "-config", str(config), "-key", str(ca_key), "-sha256", "-days", "1", "-subj", "/CN=OmniLSS Test CA", "-out", str(ca_cert)], check=True, capture_output=True)
    subprocess.run([openssl, "genrsa", "-out", str(server_key), "2048"], check=True, capture_output=True)
    subprocess.run([openssl, "req", "-new", "-config", str(config), "-key", str(server_key), "-subj", "/CN=localhost", "-out", str(server_csr)], check=True, capture_output=True)
    ext.write_text("subjectAltName=DNS:localhost,IP:127.0.0.1\n", encoding="utf-8")
    subprocess.run([openssl, "x509", "-req", "-in", str(server_csr), "-CA", str(ca_cert), "-CAkey", str(ca_key), "-CAcreateserial", "-out", str(server_cert), "-days", "1", "-sha256", "-extfile", str(ext)], check=True, capture_output=True)

    monkeypatch.setenv("OMNILSS_API_TOKEN", "tls-token")
    monkeypatch.setenv("OMNILSS_TLS_CERT", str(server_cert))
    monkeypatch.setenv("OMNILSS_TLS_KEY", str(server_key))
    server = grpc_server.serve(host="127.0.0.1", port=59058)
    time.sleep(0.2)
    try:
        bad = grpc.secure_channel("localhost:59058", grpc.ssl_channel_credentials())
        with pytest.raises(grpc.RpcError):
            capability_pb2_grpc.CapabilityServiceStub(bad).Metrics(capability_pb2.MetricsRequest(), timeout=3)
        good = grpc.secure_channel("localhost:59058", grpc.ssl_channel_credentials(root_certificates=ca_cert.read_bytes()))
        stub = capability_pb2_grpc.CapabilityServiceStub(good)
        with pytest.raises(grpc.RpcError) as auth_error:
            stub.Metrics(capability_pb2.MetricsRequest(), timeout=3)
        assert auth_error.value.code() == grpc.StatusCode.UNAUTHENTICATED
        response = stub.Metrics(capability_pb2.MetricsRequest(), timeout=3, metadata=(("authorization", "Bearer tls-token"),))
        assert response.success is True
        assert json.loads(response.metrics_json)["total_models"] >= 0
    finally:
        server.stop(grace=0)
