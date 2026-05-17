from __future__ import annotations

import json
from urllib.request import Request, urlopen

from omnilss.api.http.server import serve
from omnilss.family_capabilities import capability_matrix


def _get_response(url: str, *, request_id: str | None = None):
    headers = {"X-Request-ID": request_id} if request_id else {}
    req = Request(url, headers=headers)
    return urlopen(req, timeout=5)  # noqa: S310 - local test server


def _get_json(url: str, *, request_id: str | None = None):
    with _get_response(url, request_id=request_id) as response:
        assert response.status == 200
        return json.loads(response.read().decode("utf-8")), response.headers


def test_http_health_endpoint_includes_request_id():
    server = serve(host="127.0.0.1", port=0)
    try:
        host, port = server.server_address
        payload, headers = _get_json(
            f"http://{host}:{port}/health", request_id="test-request-1"
        )
        assert payload == {
            "service": "omnilss",
            "status": "ok",
            "request_id": "test-request-1",
        }
        assert headers["X-Request-ID"] == "test-request-1"
    finally:
        server.shutdown()
        server.server_close()


def test_http_capability_matrix_endpoint():
    server = serve(host="127.0.0.1", port=0)
    try:
        host, port = server.server_address
        payload, headers = _get_json(f"http://{host}:{port}/capabilities")
        assert payload == capability_matrix()
        assert headers["X-Request-ID"]
    finally:
        server.shutdown()
        server.server_close()


def test_http_metrics_endpoint_tracks_metadata_requests():
    server = serve(host="127.0.0.1", port=0)
    try:
        host, port = server.server_address
        _get_json(f"http://{host}:{port}/health")
        _get_json(f"http://{host}:{port}/capability-matrix")
        with _get_response(f"http://{host}:{port}/metrics") as response:
            assert response.status == 200
            body = response.read().decode("utf-8")

        assert "omnilss_http_requests_total 2" in body
        assert "omnilss_http_health_requests_total 1" in body
        assert "omnilss_http_capability_requests_total 1" in body
    finally:
        server.shutdown()
        server.server_close()
