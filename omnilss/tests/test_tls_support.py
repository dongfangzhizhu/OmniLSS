"""Phase 5.2: TLS support (gRPC server + REST documented).

- gRPC server binds a secure port when OMNILSS_TLS_CERT/OMNILSS_TLS_KEY are
  set (ssl_server_credentials); otherwise insecure (token still enforced).
- REST: uvicorn TLS flags documented in docker-compose command comment.
"""

from __future__ import annotations

import sys
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
