"""Minimal stdlib HTTP service boundary for OmniLSS.

This lightweight server intentionally avoids mandatory web-framework
dependencies.  It exposes service metadata endpoints that are useful for
orchestration and capability-aware clients while the production API is being
hardened.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from ...family_capabilities import capability_matrix


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")


def create_handler():
    """Return a request handler class serving OmniLSS metadata endpoints."""

    class OmniLSSHTTPRequestHandler(BaseHTTPRequestHandler):
        server_version = "OmniLSSHTTP/0.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            """Silence default stderr logging in embedded/test deployments."""

        def _send_json(self, status: int, payload: Any) -> None:
            body = _json_bytes(payload)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path in {"/health", "/healthz"}:
                self._send_json(200, {"status": "ok", "service": "omnilss"})
                return
            if parsed.path in {"/capabilities", "/capability-matrix"}:
                self._send_json(200, capability_matrix())
                return
            self._send_json(
                404,
                {
                    "error": "not_found",
                    "message": f"No OmniLSS HTTP endpoint for {parsed.path!r}",
                },
            )

    return OmniLSSHTTPRequestHandler


def serve(host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    """Start the minimal OmniLSS HTTP metadata server.

    The returned server is already running in a background daemon thread.  Call
    ``server.shutdown()`` and ``server.server_close()`` when done.
    """

    import threading

    server = ThreadingHTTPServer((host, port), create_handler())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    server._omnilss_thread = thread  # type: ignore[attr-defined]
    return server
