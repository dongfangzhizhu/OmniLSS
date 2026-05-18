"""Tests for local release-gate helper behavior."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "release_check.py"
_SPEC = importlib.util.spec_from_file_location("release_check_tool", _TOOL_PATH)
release_check = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(release_check)


def test_release_check_preflight_only_skips_packaging(monkeypatch, capsys):
    """Offline preflight should stop before build/twine packaging checks."""

    monkeypatch.setattr(release_check, "validate_docs_localization", lambda: [])
    monkeypatch.setattr(
        release_check,
        "validate_capability_matrix_file",
        lambda: {"ok": True, "issues": []},
    )
    monkeypatch.setattr(
        release_check,
        "run_release_gate_smoke",
        lambda: {"ok": True, "issues": []},
    )
    monkeypatch.setattr(
        release_check,
        "_run",
        lambda cmd: (_ for _ in ()).throw(AssertionError("packaging ran")),
    )

    assert release_check.main(["--preflight-only"]) == 0
    assert "Offline release preflight checks completed successfully" in capsys.readouterr().out


def test_release_check_preflight_reports_matrix_drift(monkeypatch, capsys):
    """Matrix drift should fail before packaging checks and print issue details."""

    monkeypatch.setattr(release_check, "validate_docs_localization", lambda: [])
    monkeypatch.setattr(
        release_check,
        "validate_capability_matrix_file",
        lambda: {
            "ok": False,
            "issues": [
                {
                    "severity": "error",
                    "code": "version_mismatch",
                    "path": "$.version",
                    "message": "expected version 3",
                }
            ],
        },
    )
    monkeypatch.setattr(
        release_check,
        "_run",
        lambda cmd: (_ for _ in ()).throw(AssertionError("packaging ran")),
    )

    assert release_check.main(["--preflight-only"]) == 1
    err = capsys.readouterr().err
    assert "Capability matrix validation failed" in err
    assert "version_mismatch" in err


def test_release_check_preflight_reports_smoke_failures(monkeypatch, capsys):
    """Core smoke failures should block the offline release preflight."""

    monkeypatch.setattr(release_check, "validate_docs_localization", lambda: [])
    monkeypatch.setattr(
        release_check,
        "validate_capability_matrix_file",
        lambda: {"ok": True, "issues": []},
    )
    monkeypatch.setattr(
        release_check,
        "run_release_gate_smoke",
        lambda: {
            "ok": False,
            "issues": [
                {
                    "severity": "error",
                    "code": "linear_roundtrip_prediction_mismatch",
                    "path": "$.checks.linear_fit_predict_json_roundtrip",
                    "message": "roundtrip mismatch",
                }
            ],
        },
    )
    monkeypatch.setattr(
        release_check,
        "_run",
        lambda cmd: (_ for _ in ()).throw(AssertionError("packaging ran")),
    )

    assert release_check.main(["--preflight-only"]) == 1
    err = capsys.readouterr().err
    assert "Release gate smoke checks failed" in err
    assert "linear_roundtrip_prediction_mismatch" in err
