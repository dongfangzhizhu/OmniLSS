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
        "validate_three_month_checklist",
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


def test_release_check_preflight_reports_three_month_checklist_failures(monkeypatch, capsys):
    """Three-month checklist drift should block preflight before packaging checks."""

    monkeypatch.setattr(release_check, "validate_docs_localization", lambda: [])
    monkeypatch.setattr(
        release_check,
        "validate_capability_matrix_file",
        lambda: {"ok": True, "issues": []},
    )
    monkeypatch.setattr(
        release_check,
        "validate_three_month_checklist",
        lambda: {
            "ok": False,
            "issues": [
                {
                    "code": "week_order_violation",
                    "week": "W07",
                    "message": "Week 7 started before Week 6 completion evidence.",
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
    assert "Three-month execution checklist validation failed" in err
    assert "week_order_violation" in err


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
        "validate_three_month_checklist",
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


def test_release_check_reports_build_failure_and_hint(monkeypatch, capsys):
    monkeypatch.setattr(release_check, "validate_docs_localization", lambda: [])
    monkeypatch.setattr(release_check, "validate_capability_matrix_file", lambda: {"ok": True, "issues": []})
    monkeypatch.setattr(release_check, "validate_three_month_checklist", lambda: {"ok": True, "issues": []})
    monkeypatch.setattr(release_check, "run_release_gate_smoke", lambda: {"ok": True, "issues": []})

    calls = []

    def _fake_run(cmd):
        calls.append(cmd)
        if "build" in cmd:
            return 1
        return 0

    monkeypatch.setattr(release_check, "_run", _fake_run)

    assert release_check.main([]) == 1
    err = capsys.readouterr().err
    assert "Build step failed" in err
    assert "pip install build" in err
    assert any("build" in c for c in calls)


def test_release_check_reports_twine_failure_and_hint(monkeypatch, capsys):
    monkeypatch.setattr(release_check, "validate_docs_localization", lambda: [])
    monkeypatch.setattr(release_check, "validate_capability_matrix_file", lambda: {"ok": True, "issues": []})
    monkeypatch.setattr(release_check, "validate_three_month_checklist", lambda: {"ok": True, "issues": []})
    monkeypatch.setattr(release_check, "run_release_gate_smoke", lambda: {"ok": True, "issues": []})

    calls = []

    def _fake_run(cmd):
        calls.append(cmd)
        if "twine" in cmd:
            return 1
        return 0

    monkeypatch.setattr(release_check, "_run", _fake_run)

    assert release_check.main([]) == 1
    err = capsys.readouterr().err
    assert "Twine check failed" in err
    assert "pip install twine" in err
    assert any("twine" in c for c in calls)
    assert any("build" in c for c in calls)
    assert next(i for i, c in enumerate(calls) if "build" in c) < next(i for i, c in enumerate(calls) if "twine" in c)


def test_release_check_full_success_runs_build_and_twine(monkeypatch, capsys):
    monkeypatch.setattr(release_check, "validate_docs_localization", lambda: [])
    monkeypatch.setattr(release_check, "validate_capability_matrix_file", lambda: {"ok": True, "issues": []})
    monkeypatch.setattr(release_check, "validate_three_month_checklist", lambda: {"ok": True, "issues": []})
    monkeypatch.setattr(release_check, "run_release_gate_smoke", lambda: {"ok": True, "issues": []})

    calls = []

    def _fake_run(cmd):
        calls.append(cmd)
        return 0

    monkeypatch.setattr(release_check, "_run", _fake_run)

    assert release_check.main([]) == 0
    out = capsys.readouterr().out
    assert "Release checks completed successfully." in out
    assert any("build" in c for c in calls)
    assert any("twine" in c for c in calls)


def test_release_check_reports_missing_python_executable(monkeypatch, capsys):
    monkeypatch.setattr(release_check.shutil, "which", lambda name: None if name == "python" else "/usr/bin/" + name)

    assert release_check.main([]) == 2
    err = capsys.readouterr().err
    assert "Missing required executable: python" in err


def test_release_check_preflight_only_still_requires_python_executable(monkeypatch, capsys):
    monkeypatch.setattr(release_check.shutil, "which", lambda name: None if name == "python" else "/usr/bin/" + name)

    assert release_check.main(["--preflight-only"]) == 2
    err = capsys.readouterr().err
    assert "Missing required executable: python" in err


def test_release_check_reports_missing_python_before_preflight_calls(monkeypatch):
    monkeypatch.setattr(release_check.shutil, "which", lambda name: None if name == "python" else "/usr/bin/" + name)

    called = {"docs": False, "matrix": False, "checklist": False, "smoke": False}

    monkeypatch.setattr(release_check, "validate_docs_localization", lambda: called.__setitem__("docs", True) or [])
    monkeypatch.setattr(release_check, "validate_capability_matrix_file", lambda: called.__setitem__("matrix", True) or {"ok": True, "issues": []})
    monkeypatch.setattr(release_check, "validate_three_month_checklist", lambda: called.__setitem__("checklist", True) or {"ok": True, "issues": []})
    monkeypatch.setattr(release_check, "run_release_gate_smoke", lambda: called.__setitem__("smoke", True) or {"ok": True, "issues": []})

    assert release_check.main([]) == 2
    assert called == {"docs": False, "matrix": False, "checklist": False, "smoke": False}


def test_release_check_preflight_only_missing_python_short_circuits_downstream(monkeypatch):
    monkeypatch.setattr(release_check.shutil, "which", lambda name: None if name == "python" else "/usr/bin/" + name)

    called = {"docs": False, "matrix": False, "checklist": False, "smoke": False}

    monkeypatch.setattr(release_check, "validate_docs_localization", lambda: called.__setitem__("docs", True) or [])
    monkeypatch.setattr(release_check, "validate_capability_matrix_file", lambda: called.__setitem__("matrix", True) or {"ok": True, "issues": []})
    monkeypatch.setattr(release_check, "validate_three_month_checklist", lambda: called.__setitem__("checklist", True) or {"ok": True, "issues": []})
    monkeypatch.setattr(release_check, "run_release_gate_smoke", lambda: called.__setitem__("smoke", True) or {"ok": True, "issues": []})

    assert release_check.main(["--preflight-only"]) == 2
    assert called == {"docs": False, "matrix": False, "checklist": False, "smoke": False}


def test_release_check_missing_python_never_runs_packaging(monkeypatch):
    monkeypatch.setattr(release_check.shutil, "which", lambda name: None if name == "python" else "/usr/bin/" + name)

    called = {"run": False}

    def _fail_if_called(cmd):
        called["run"] = True
        raise AssertionError("_run should not be called when python is missing")

    monkeypatch.setattr(release_check, "_run", _fail_if_called)

    assert release_check.main([]) == 2
    assert called["run"] is False


def test_release_check_preflight_only_missing_python_never_runs_packaging(monkeypatch):
    monkeypatch.setattr(release_check.shutil, "which", lambda name: None if name == "python" else "/usr/bin/" + name)

    called = {"run": False}

    def _fail_if_called(cmd):
        called["run"] = True
        raise AssertionError("_run should not be called in preflight-only when python is missing")

    monkeypatch.setattr(release_check, "_run", _fail_if_called)

    assert release_check.main(["--preflight-only"]) == 2
    assert called["run"] is False


def test_release_check_uses_python_interpreter_for_build_and_twine(monkeypatch):
    monkeypatch.setattr(release_check, "validate_docs_localization", lambda: [])
    monkeypatch.setattr(release_check, "validate_capability_matrix_file", lambda: {"ok": True, "issues": []})
    monkeypatch.setattr(release_check, "validate_three_month_checklist", lambda: {"ok": True, "issues": []})
    monkeypatch.setattr(release_check, "run_release_gate_smoke", lambda: {"ok": True, "issues": []})

    calls = []

    def _fake_run(cmd):
        calls.append(cmd)
        return 0

    monkeypatch.setattr(release_check, "_run", _fake_run)

    assert release_check.main([]) == 0
    assert any(cmd[:3] == [release_check.sys.executable, "-m", "build"] for cmd in calls)
    assert any(cmd[:3] == [release_check.sys.executable, "-m", "twine"] for cmd in calls)
