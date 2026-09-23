from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("tls_readiness", ROOT / "tools" / "tls_readiness.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_tls_readiness_fails_missing_files(tmp_path):
    report = MODULE.validate(tmp_path / "missing.crt", tmp_path / "missing.key")
    assert report["ok"] is False
    assert len(report["issues"]) == 2


def test_tls_readiness_fails_non_pem_files(tmp_path):
    cert = tmp_path / "server.crt"
    key = tmp_path / "server.key"
    cert.write_text("certificate", encoding="utf-8")
    key.write_text("private key", encoding="utf-8")
    report = MODULE.validate(cert, key)
    assert report["ok"] is False
    assert any("PEM" in issue for issue in report["issues"])
