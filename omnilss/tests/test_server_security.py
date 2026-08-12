"""Phase 5.1: authentication defaults to denied.

- gRPC server refuses to start without OMNILSS_API_TOKEN on a non-loopback
  interface (_validate_api_token).
- docker-compose sets OMNILSS_API_TOKEN for both services (production
  deployments are token-authenticated).
- FastAPI omnilss-server/main.py requires Depends(_check_auth) on every
  endpoint (same policy as api.http.fastapi_server; token empty = local dev).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from omnilss.api.grpc.server import _validate_api_token  # noqa: E402


def test_default_deny_without_token_non_loopback() -> None:
    """No token + non-loopback bind -> RuntimeError (default-deny)."""
    with pytest.raises(RuntimeError, match="OMNILSS_API_TOKEN"):
        _validate_api_token("0.0.0.0", "")
    with pytest.raises(RuntimeError, match="OMNILSS_API_TOKEN"):
        _validate_api_token("::", "")


def test_loopback_allowed_without_token() -> None:
    """No token + loopback bind is allowed (local development)."""
    for host in ("127.0.0.1", "localhost", "::1"):
        _validate_api_token(host, "")  # must not raise


def test_token_allows_non_loopback() -> None:
    """A configured token permits a non-loopback bind."""
    _validate_api_token("0.0.0.0", "secret-token")  # must not raise


def test_docker_compose_sets_token() -> None:
    """docker-compose configures OMNILSS_API_TOKEN for both services."""
    import yaml

    compose = Path(__file__).resolve().parents[2] / "docker-compose.yml"
    with compose.open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    for service in ("omnilss-core-grpc", "omnilss-core-http"):
        env = config["services"][service]["environment"]
        assert "OMNILSS_API_TOKEN" in env, f"{service} missing OMNILSS_API_TOKEN"
        assert env["OMNILSS_API_TOKEN"] != "", f"{service} token must not be empty"


def test_fastapi_endpoints_require_auth() -> None:
    """Every omnilss-server endpoint declares the auth dependency."""
    text = (Path(__file__).resolve().parents[2] / "omnilss-server" / "main.py").read_text(
        encoding="utf-8"
    )
    import re

    endpoints = re.findall(r"^@app\.(?:get|post|delete|put|patch)\(", text, re.MULTILINE)
    assert endpoints, "no @app routes found in omnilss-server/main.py"
    auth_deps = text.count("Depends(_check_auth)")
    assert auth_deps == len(endpoints), (
        f"{len(endpoints)} routes but {auth_deps} auth dependencies "
        "(every endpoint must require Depends(_check_auth))"
    )
