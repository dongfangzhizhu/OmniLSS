"""Tests for the fail-closed capability admission gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("validate_capability_admission", ROOT / "tools" / "validate_capability_admission.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _coverage() -> dict:
    spec = importlib.util.spec_from_file_location("generate_r_export_coverage", ROOT / "tools" / "generate_r_export_coverage.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.build_coverage()


def test_runtime_coverage_and_route_gate_are_consistent():
    report = MODULE.validate(_coverage())
    assert report["ok"] is True, report["issues"]
    assert report["runtime_family_count"] == report["coverage_family_count"]
    assert report["routes"]["BB"]["production_rejected"] is True


def test_missing_and_drifted_family_are_reported():
    payload = _coverage()
    name = next(iter(payload["families"]))
    del payload["families"][name]
    payload["families"]["EXTRA"] = {"implementation_status": "experimental"}
    report = MODULE.validate(payload)
    codes = {issue["code"] for issue in report["issues"]}
    assert "coverage_missing_family" in codes
    assert "coverage_extra_family" in codes


def test_status_drift_is_fail_closed():
    payload = _coverage()
    payload["families"]["NO"]["implementation_status"] = "placeholder"
    report = MODULE.validate(payload)
    assert any(issue["code"] == "implementation_status_drift" and issue["family"] == "NO" for issue in report["issues"])
