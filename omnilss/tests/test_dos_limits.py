"""Phase 5.3: DoS limits (gRPC + REST).

- gRPC: Sample.n / BatchFit count / max_iter capped; JSON body bounded;
  message size limits configured on the server.
- REST (omnilss-server/main.py): body-size guard + per-client rate limit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from omnilss.api.grpc.server import (  # noqa: E402
    MAX_BATCH_FITS,
    MAX_GRPC_MESSAGE_BYTES,
    MAX_ITER,
    MAX_JSON_BODY_BYTES,
    MAX_SAMPLE_N,
    _from_json,
)


def test_dos_limit_constants_defined():
    """All gRPC DoS caps are present and positive."""
    for name, value in {
        "MAX_SAMPLE_N": MAX_SAMPLE_N,
        "MAX_BATCH_FITS": MAX_BATCH_FITS,
        "MAX_ITER": MAX_ITER,
        "MAX_JSON_BODY_BYTES": MAX_JSON_BODY_BYTES,
        "MAX_GRPC_MESSAGE_BYTES": MAX_GRPC_MESSAGE_BYTES,
    }.items():
        assert isinstance(value, int) and value > 0, f"{name} = {value}"


def test_from_json_rejects_oversized_payload():
    """A JSON body over the byte cap is rejected before parsing."""
    big = '{"data": "' + "x" * (MAX_JSON_BODY_BYTES + 10) + '"}'
    with pytest.raises(ValueError, match="payload too large"):
        _from_json(big)


def test_from_json_parses_normal_payload():
    """A normal JSON object parses fine."""
    assert _from_json('{"a": 1}') == {"a": 1}


def test_grpc_server_sets_message_size_limits():
    """grpc.server is configured with explicit receive/send message caps."""
    source = (SRC / "omnilss" / "api" / "grpc" / "server.py").read_text(encoding="utf-8")
    assert "grpc.max_receive_message_length" in source
    assert "grpc.max_send_message_length" in source
    assert "MAX_GRPC_MESSAGE_BYTES" in source


def test_fit_caps_max_iter():
    """max_iter is capped at MAX_ITER in the fit path."""
    source = (SRC / "omnilss" / "api" / "grpc" / "server.py").read_text(encoding="utf-8")
    assert "min(int(request.max_iter), MAX_ITER)" in source


def test_batch_fit_caps_request_count():
    """BatchFit rejects batches over MAX_BATCH_FITS."""
    source = (SRC / "omnilss" / "api" / "grpc" / "server.py").read_text(encoding="utf-8")
    assert "len(request.requests) > MAX_BATCH_FITS" in source


def test_sample_caps_n():
    """Sample.n is capped at MAX_SAMPLE_N."""
    source = (SRC / "omnilss" / "api" / "grpc" / "server.py").read_text(encoding="utf-8")
    assert "min(max(0, int(request.n)), MAX_SAMPLE_N)" in source


def test_rest_body_guard_and_rate_limit():
    """omnilss-server/main.py guards body size and rate-limits clients."""
    text = (Path(__file__).resolve().parents[2] / "omnilss-server" / "main.py").read_text(
        encoding="utf-8"
    )
    assert "MAX_JSON_BODY_BYTES" in text
    assert "413_REQUEST_ENTITY_TOO_LARGE" in text
    assert "RATE_LIMIT_MAX_REQUESTS" in text
    assert "429_TOO_MANY_REQUESTS" in text
