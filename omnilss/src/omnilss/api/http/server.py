"""Minimal stdlib HTTP service boundary for OmniLSS.

This lightweight server intentionally avoids mandatory web-framework
dependencies.  It exposes service metadata endpoints that are useful for
orchestration and capability-aware clients while the production API is being
hardened.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

from ...family_capabilities import capability_matrix

DEFAULT_MAX_REQUEST_BYTES = 1_048_576


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")


def create_handler(
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
    event_sink: Callable[[dict[str, Any]], None] | None = None,
):
    """Return a request handler class serving OmniLSS metadata endpoints."""

    max_body_bytes = max(0, int(max_request_bytes))

    metrics = {
        "requests_total": 0,
        "capability_requests_total": 0,
        "health_requests_total": 0,
        "not_found_total": 0,
        "method_not_allowed_total": 0,
        "payload_too_large_total": 0,
        "bad_request_total": 0,
        "request_duration_seconds_sum": 0.0,
    }
    metrics_lock = threading.Lock()

    def record_metric(name: str, *, duration: float | None = None) -> None:
        with metrics_lock:
            metrics["requests_total"] += 1
            metrics[name] += 1
            if duration is not None:
                metrics["request_duration_seconds_sum"] += duration

    def emit_event(
        *,
        method: str,
        path: str,
        status: int,
        request_id: str,
        duration: float,
        error_code: str | None = None,
    ) -> None:
        if event_sink is None:
            return
        event = {
            "event": "http_request",
            "method": method,
            "path": path,
            "status": status,
            "request_id": request_id,
            "duration_seconds": float(duration),
        }
        if error_code is not None:
            event["error_code"] = error_code
        try:
            event_sink(event)
        except Exception:
            # Observability hooks must not break prototype metadata responses.
            return

    def metrics_text() -> str:
        with metrics_lock:
            snapshot = dict(metrics)
        lines = [
            "# HELP omnilss_http_requests_total Total HTTP metadata requests.",
            "# TYPE omnilss_http_requests_total counter",
            f"omnilss_http_requests_total {snapshot['requests_total']}",
            "# HELP omnilss_http_capability_requests_total Capability matrix requests.",
            "# TYPE omnilss_http_capability_requests_total counter",
            f"omnilss_http_capability_requests_total {snapshot['capability_requests_total']}",
            "# HELP omnilss_http_health_requests_total Health check requests.",
            "# TYPE omnilss_http_health_requests_total counter",
            f"omnilss_http_health_requests_total {snapshot['health_requests_total']}",
            "# HELP omnilss_http_not_found_total Unknown endpoint requests.",
            "# TYPE omnilss_http_not_found_total counter",
            f"omnilss_http_not_found_total {snapshot['not_found_total']}",
            "# HELP omnilss_http_method_not_allowed_total Unsupported method requests.",
            "# TYPE omnilss_http_method_not_allowed_total counter",
            f"omnilss_http_method_not_allowed_total {snapshot['method_not_allowed_total']}",
            "# HELP omnilss_http_payload_too_large_total Oversized request payloads.",
            "# TYPE omnilss_http_payload_too_large_total counter",
            f"omnilss_http_payload_too_large_total {snapshot['payload_too_large_total']}",
            "# HELP omnilss_http_bad_request_total Malformed HTTP metadata requests.",
            "# TYPE omnilss_http_bad_request_total counter",
            f"omnilss_http_bad_request_total {snapshot['bad_request_total']}",
            "# HELP omnilss_http_request_duration_seconds_sum Sum of request durations.",
            "# TYPE omnilss_http_request_duration_seconds_sum counter",
            f"omnilss_http_request_duration_seconds_sum {snapshot['request_duration_seconds_sum']:.12g}",
        ]
        return "\n".join(lines) + "\n"

    class OmniLSSHTTPRequestHandler(BaseHTTPRequestHandler):
        server_version = "OmniLSSHTTP/0.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            """Silence default stderr logging in embedded/test deployments."""

        def _request_id(self) -> str:
            inbound = self.headers.get("X-Request-ID", "").strip()
            return inbound or str(uuid.uuid4())

        def _send_body(
            self,
            status: int,
            body: bytes,
            *,
            content_type: str,
            request_id: str,
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Request-ID", request_id)
            for name, value in (extra_headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def _send_json(
            self,
            status: int,
            payload: Any,
            *,
            request_id: str,
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            self._send_body(
                status,
                _json_bytes(payload),
                content_type="application/json; charset=utf-8",
                request_id=request_id,
                extra_headers=extra_headers,
            )

        def _send_error(
            self,
            status: int,
            *,
            code: str,
            message: str,
            request_id: str,
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            self._send_json(
                status,
                {
                    "success": False,
                    "error": {
                        "type": "http_error",
                        "code": code,
                        "message": message,
                    },
                    "request_id": request_id,
                },
                request_id=request_id,
                extra_headers=extra_headers,
            )

        def _content_length(self) -> int:
            raw = self.headers.get("Content-Length")
            if raw in {None, ""}:
                return 0
            try:
                length = int(str(raw).strip())
            except ValueError as exc:
                raise ValueError("Content-Length must be an integer") from exc
            if length < 0:
                raise ValueError("Content-Length must be non-negative")
            return length

        def do_GET(self) -> None:  # noqa: N802
            started = time.perf_counter()
            request_id = self._request_id()
            parsed = urlparse(self.path)
            if parsed.path in {"/health", "/healthz"}:
                record_metric(
                    "health_requests_total", duration=time.perf_counter() - started
                )
                self._send_json(
                    200,
                    {"status": "ok", "service": "omnilss", "request_id": request_id},
                    request_id=request_id,
                )
                emit_event(
                    method="GET",
                    path=parsed.path,
                    status=200,
                    request_id=request_id,
                    duration=time.perf_counter() - started,
                )
                return
            if parsed.path in {"/capabilities", "/capability-matrix"}:
                record_metric(
                    "capability_requests_total",
                    duration=time.perf_counter() - started,
                )
                self._send_json(200, capability_matrix(), request_id=request_id)
                emit_event(
                    method="GET",
                    path=parsed.path,
                    status=200,
                    request_id=request_id,
                    duration=time.perf_counter() - started,
                )
                return
            if parsed.path == "/metrics":
                body = metrics_text().encode("utf-8")
                self._send_body(
                    200,
                    body,
                    content_type="text/plain; version=0.0.4; charset=utf-8",
                    request_id=request_id,
                )
                emit_event(
                    method="GET",
                    path=parsed.path,
                    status=200,
                    request_id=request_id,
                    duration=time.perf_counter() - started,
                )
                return
            record_metric("not_found_total", duration=time.perf_counter() - started)
            self._send_error(
                404,
                code="not_found",
                message=f"No OmniLSS HTTP endpoint for {parsed.path!r}",
                request_id=request_id,
            )
            emit_event(
                method="GET",
                path=parsed.path,
                status=404,
                request_id=request_id,
                duration=time.perf_counter() - started,
                error_code="not_found",
            )

        def do_POST(self) -> None:  # noqa: N802
            started = time.perf_counter()
            request_id = self._request_id()
            parsed = urlparse(self.path)
            try:
                content_length = self._content_length()
            except ValueError as exc:
                record_metric(
                    "bad_request_total", duration=time.perf_counter() - started
                )
                self._send_error(
                    400,
                    code="invalid_content_length",
                    message=str(exc),
                    request_id=request_id,
                )
                emit_event(
                    method="POST",
                    path=parsed.path,
                    status=400,
                    request_id=request_id,
                    duration=time.perf_counter() - started,
                    error_code="invalid_content_length",
                )
                return
            if content_length > max_body_bytes:
                record_metric(
                    "payload_too_large_total", duration=time.perf_counter() - started
                )
                self._send_error(
                    413,
                    code="payload_too_large",
                    message=(
                        f"Request payload is {content_length} bytes; "
                        f"maximum allowed is {max_body_bytes} bytes"
                    ),
                    request_id=request_id,
                    extra_headers={"Connection": "close"},
                )
                self.close_connection = True
                emit_event(
                    method="POST",
                    path=parsed.path,
                    status=413,
                    request_id=request_id,
                    duration=time.perf_counter() - started,
                    error_code="payload_too_large",
                )
                return
            record_metric(
                "method_not_allowed_total", duration=time.perf_counter() - started
            )
            self._send_error(
                405,
                code="method_not_allowed",
                message=f"HTTP POST is not enabled for {parsed.path!r}; fit/predict endpoints require authn, limits, and structured logging before exposure",
                request_id=request_id,
                extra_headers={"Allow": "GET"},
            )
            emit_event(
                method="POST",
                path=parsed.path,
                status=405,
                request_id=request_id,
                duration=time.perf_counter() - started,
                error_code="method_not_allowed",
            )

    return OmniLSSHTTPRequestHandler


def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    *,
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
    event_sink: Callable[[dict[str, Any]], None] | None = None,
) -> ThreadingHTTPServer:
    """Start the minimal OmniLSS HTTP metadata server.

    The returned server is already running in a background daemon thread.  Call
    ``server.shutdown()`` and ``server.server_close()`` when done.
    """

    server = ThreadingHTTPServer(
        (host, port),
        create_handler(max_request_bytes=max_request_bytes, event_sink=event_sink),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    server._omnilss_thread = thread  # type: ignore[attr-defined]
    return server
