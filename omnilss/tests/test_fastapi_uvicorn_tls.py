"""Real Uvicorn HTTPS and Bearer-token deployment regression."""

from __future__ import annotations

import importlib.util
import os
import shutil
import socket
import ssl
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest


if importlib.util.find_spec("fastapi") is None or importlib.util.find_spec("uvicorn") is None:
    pytest.skip("FastAPI/Uvicorn server extra is not installed", allow_module_level=True)


def _run_openssl(args: list[str]) -> None:
    executable = shutil.which("openssl") or "openssl"
    config = os.environ.get("OPENSSL_CONF")
    env = dict(os.environ)
    if config:
        env["OPENSSL_CONF"] = config
    subprocess.run([executable, *args], check=True, capture_output=True, env=env)


def _free_port() -> int:
    """Return an ephemeral localhost port instead of a fixed one.

    A hard-coded 59100 can fall inside a Windows/Hyper-V excluded range or be
    taken by another process, which makes uvicorn exit before binding.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _stop_process(process: subprocess.Popen[str]) -> tuple[str, str]:
    """Terminate (then kill) uvicorn and always collect its output.

    A fixed 5s communicate() timeout used to turn a slow CI start into a
    TimeoutExpired instead of the real "server did not start" assertion.
    """
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    return process.communicate(timeout=10)


def test_uvicorn_https_requires_ca_and_token(tmp_path):
    root = Path(__file__).resolve().parents[2]
    server_root = root / "omnilss-server"
    ca_key, ca_cert = tmp_path / "ca.key", tmp_path / "ca.crt"
    key, csr, cert, ext, cfg = (tmp_path / n for n in ("server.key", "server.csr", "server.crt", "server.ext", "openssl.cnf"))
    cfg.write_text("[req]\ndistinguished_name=req_distinguished_name\n[req_distinguished_name]\n\n", encoding="utf-8")
    _run_openssl(["genrsa", "-out", str(ca_key), "2048"])
    _run_openssl(["req", "-x509", "-new", "-nodes", "-config", str(cfg), "-key", str(ca_key), "-sha256", "-days", "1", "-addext", "basicConstraints=critical,CA:TRUE", "-subj", "/CN=OmniLSS Test CA", "-out", str(ca_cert)])
    _run_openssl(["genrsa", "-out", str(key), "2048"])
    _run_openssl(["req", "-new", "-config", str(cfg), "-key", str(key), "-subj", "/CN=localhost", "-out", str(csr)])
    ext.write_text("subjectAltName=DNS:localhost,IP:127.0.0.1\n", encoding="utf-8")
    _run_openssl(["x509", "-req", "-in", str(csr), "-CA", str(ca_cert), "-CAkey", str(ca_key), "-CAcreateserial", "-out", str(cert), "-days", "1", "-sha256", "-extfile", str(ext)])

    port = _free_port()
    env = dict(os.environ, PYTHONPATH=str(root / "omnilss" / "src"), OMNILSS_API_TOKEN="uvicorn-token")
    process = subprocess.Popen([
        sys.executable, "-m", "uvicorn", "main:app",
        "--host", "127.0.0.1", "--port", str(port), "--ssl-keyfile", str(key), "--ssl-certfile", str(cert),
    ], cwd=server_root, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    context = ssl.create_default_context(cafile=str(ca_cert))
    insecure_context = ssl._create_unverified_context()
    try:
        # CI runners can spend tens of seconds importing JAX before uvicorn
        # binds; 15s was below the observed cold start.
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                try:
                    urlopen(f"https://localhost:{port}/models", context=insecure_context, timeout=1)
                except HTTPError as exc:
                    if exc.code == 401:
                        break
                    raise
            except Exception:
                time.sleep(0.2)
        else:
            stdout, stderr = _stop_process(process)
            raise AssertionError(f"Uvicorn HTTPS server did not start: {stdout}\n{stderr}")
        request = Request(f"https://localhost:{port}/models", headers={"Authorization": "Bearer uvicorn-token"})
        with urlopen(request, context=context, timeout=5) as response:
            assert response.status == 200
        with pytest.raises(HTTPError) as exc_info:
            urlopen(Request(f"https://localhost:{port}/models"), context=context, timeout=5)
        assert exc_info.value.code == 401
        with pytest.raises(Exception):
            urlopen(f"http://localhost:{port}/health", timeout=2)
    finally:
        _stop_process(process)
