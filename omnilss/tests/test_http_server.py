from __future__ import annotations

import json
from urllib.request import urlopen

from omnilss.api.http.server import serve
from omnilss.family_capabilities import capability_matrix


def _get_json(url: str):
    with urlopen(url, timeout=5) as response:  # noqa: S310 - local test server
        assert response.status == 200
        return json.loads(response.read().decode("utf-8"))


def test_http_health_endpoint():
    server = serve(host="127.0.0.1", port=0)
    try:
        host, port = server.server_address
        payload = _get_json(f"http://{host}:{port}/health")
        assert payload == {"service": "omnilss", "status": "ok"}
    finally:
        server.shutdown()
        server.server_close()


def test_http_capability_matrix_endpoint():
    server = serve(host="127.0.0.1", port=0)
    try:
        host, port = server.server_address
        payload = _get_json(f"http://{host}:{port}/capabilities")
        assert payload == capability_matrix()
    finally:
        server.shutdown()
        server.server_close()
