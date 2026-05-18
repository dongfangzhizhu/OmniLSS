from __future__ import annotations

import http.client
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from omnilss.api.http.server import serve
from omnilss.family_capabilities import capability_matrix, method_route_capability_report


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


def test_http_route_capability_endpoint_reports_admission_decision():
    server = serve(host="127.0.0.1", port=0)
    try:
        host, port = server.server_address
        payload, headers = _get_json(
            f"http://{host}:{port}/route-capability?family=GA&method=RS&strict=true",
            request_id="route-1",
        )

        assert payload == method_route_capability_report("GA", "RS", strict=True)
        assert payload["ok"] is False
        assert payload["code"] == "experimental_requires_opt_in"
        assert headers["X-Request-ID"] == "route-1"

        with _get_response(f"http://{host}:{port}/metrics") as response:
            body = response.read().decode("utf-8")
        assert "omnilss_http_route_admission_requests_total 1" in body
    finally:
        server.shutdown()
        server.server_close()


def test_http_route_capability_requires_family_and_method():
    server = serve(host="127.0.0.1", port=0)
    try:
        host, port = server.server_address
        req = Request(
            f"http://{host}:{port}/route-capability?family=NO",
            headers={"X-Request-ID": "route-missing"},
        )
        try:
            urlopen(req, timeout=5)  # noqa: S310 - local test server
            raise AssertionError("expected HTTPError")
        except HTTPError as exc:
            assert exc.code == 400
            payload = json.loads(exc.read().decode("utf-8"))
            assert payload == {
                "success": False,
                "error": {
                    "type": "http_error",
                    "code": "invalid_route_query",
                    "message": (
                        "route capability checks require non-empty 'family' "
                        "and 'method' query parameters"
                    ),
                },
                "request_id": "route-missing",
            }
            assert exc.headers["X-Request-ID"] == "route-missing"
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


def test_http_unknown_endpoint_uses_structured_error_envelope():
    server = serve(host="127.0.0.1", port=0)
    try:
        host, port = server.server_address
        req = Request(
            f"http://{host}:{port}/missing", headers={"X-Request-ID": "missing-1"}
        )
        try:
            urlopen(req, timeout=5)  # noqa: S310 - local test server
            raise AssertionError("expected HTTPError")
        except HTTPError as exc:
            assert exc.code == 404
            payload = json.loads(exc.read().decode("utf-8"))
            assert payload == {
                "success": False,
                "error": {
                    "type": "http_error",
                    "code": "not_found",
                    "message": "No OmniLSS HTTP endpoint for '/missing'",
                },
                "request_id": "missing-1",
            }
            assert exc.headers["X-Request-ID"] == "missing-1"
    finally:
        server.shutdown()
        server.server_close()


def test_http_post_is_blocked_with_structured_error_envelope():
    server = serve(host="127.0.0.1", port=0)
    try:
        host, port = server.server_address
        req = Request(
            f"http://{host}:{port}/predict",
            data=b"{}",
            method="POST",
            headers={"X-Request-ID": "post-1"},
        )
        try:
            urlopen(req, timeout=5)  # noqa: S310 - local test server
            raise AssertionError("expected HTTPError")
        except HTTPError as exc:
            assert exc.code == 405
            payload = json.loads(exc.read().decode("utf-8"))
            assert payload["success"] is False
            assert payload["error"]["type"] == "http_error"
            assert payload["error"]["code"] == "method_not_allowed"
            assert "fit/predict endpoints require authn" in payload["error"]["message"]
            assert payload["request_id"] == "post-1"
            assert exc.headers["X-Request-ID"] == "post-1"
    finally:
        server.shutdown()
        server.server_close()


def test_http_post_payload_limit_returns_structured_error():
    server = serve(host="127.0.0.1", port=0, max_request_bytes=4)
    try:
        host, port = server.server_address
        req = Request(
            f"http://{host}:{port}/predict",
            data=b"12345",
            method="POST",
            headers={"X-Request-ID": "large-1"},
        )
        try:
            urlopen(req, timeout=5)  # noqa: S310 - local test server
            raise AssertionError("expected HTTPError")
        except HTTPError as exc:
            assert exc.code == 413
            payload = json.loads(exc.read().decode("utf-8"))
            assert payload["success"] is False
            assert payload["error"]["type"] == "http_error"
            assert payload["error"]["code"] == "payload_too_large"
            assert "maximum allowed is 4 bytes" in payload["error"]["message"]
            assert payload["request_id"] == "large-1"
            assert exc.headers["X-Request-ID"] == "large-1"

        with _get_response(f"http://{host}:{port}/metrics") as response:
            body = response.read().decode("utf-8")
        assert "omnilss_http_payload_too_large_total 1" in body
    finally:
        server.shutdown()
        server.server_close()


def test_http_structured_event_sink_records_success_and_errors():
    events: list[dict[str, object]] = []
    server = serve(host="127.0.0.1", port=0, event_sink=events.append)
    try:
        host, port = server.server_address
        _get_json(f"http://{host}:{port}/health", request_id="event-ok")
        req = Request(
            f"http://{host}:{port}/predict",
            data=b"{}",
            method="POST",
            headers={"X-Request-ID": "event-blocked"},
        )
        try:
            urlopen(req, timeout=5)  # noqa: S310 - local test server
            raise AssertionError("expected HTTPError")
        except HTTPError as exc:
            assert exc.code == 405
            exc.read()

        assert len(events) == 2
        assert events[0]["event"] == "http_request"
        assert events[0]["method"] == "GET"
        assert events[0]["path"] == "/health"
        assert events[0]["status"] == 200
        assert events[0]["request_id"] == "event-ok"
        assert "duration_seconds" in events[0]
        assert "error_code" not in events[0]

        assert events[1]["method"] == "POST"
        assert events[1]["path"] == "/predict"
        assert events[1]["status"] == 405
        assert events[1]["request_id"] == "event-blocked"
        assert events[1]["error_code"] == "method_not_allowed"
    finally:
        server.shutdown()
        server.server_close()


def test_http_event_sink_failure_does_not_break_response():
    def broken_sink(event: dict[str, object]) -> None:  # noqa: ARG001
        raise RuntimeError("sink failed")

    server = serve(host="127.0.0.1", port=0, event_sink=broken_sink)
    try:
        host, port = server.server_address
        payload, headers = _get_json(
            f"http://{host}:{port}/health", request_id="sink-failure"
        )
        assert payload["status"] == "ok"
        assert headers["X-Request-ID"] == "sink-failure"
    finally:
        server.shutdown()
        server.server_close()


def test_http_invalid_content_length_returns_structured_error_and_metric():
    server = serve(host="127.0.0.1", port=0)
    try:
        host, port = server.server_address
        conn = http.client.HTTPConnection(host, port, timeout=5)
        conn.putrequest("POST", "/predict")
        conn.putheader("X-Request-ID", "bad-length-1")
        conn.putheader("Content-Length", "not-an-integer")
        conn.endheaders()
        response = conn.getresponse()
        try:
            assert response.status == 400
            payload = json.loads(response.read().decode("utf-8"))
            assert payload == {
                "success": False,
                "error": {
                    "type": "http_error",
                    "code": "invalid_content_length",
                    "message": "Content-Length must be an integer",
                },
                "request_id": "bad-length-1",
            }
            assert response.getheader("X-Request-ID") == "bad-length-1"
        finally:
            conn.close()

        with _get_response(f"http://{host}:{port}/metrics") as metrics_response:
            body = metrics_response.read().decode("utf-8")
        assert "omnilss_http_bad_request_total 1" in body
    finally:
        server.shutdown()
        server.server_close()
