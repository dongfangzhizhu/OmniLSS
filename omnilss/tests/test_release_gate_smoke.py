"""Tests for Month 1 release-gate smoke checks."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "release_gate_smoke.py"
_SPEC = importlib.util.spec_from_file_location("release_gate_smoke_tool", _TOOL_PATH)
release_gate_smoke = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(release_gate_smoke)


def test_release_gate_smoke_passes_core_roundtrip_and_schema_errors(tmp_path):
    report = release_gate_smoke.run_release_gate_smoke(tmp_path)

    assert report["ok"] is True
    assert report["issues"] == []
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["linear_fit_predict_json_roundtrip"]["ok"] is True
    assert checks["linear_fit_predict_json_roundtrip"]["max_abs_error"] <= 1e-10
    assert checks["model_artifact_validator"]["ok"] is True
    assert checks["missing_variable_prediction_error"]["ok"] is True
    assert checks["unseen_factor_prediction_error"]["ok"] is True
