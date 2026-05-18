from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from omnilss.family_capabilities import (
    CAPABILITY_MATRIX_VERSION,
    FEATURES,
    capability_matrix,
)

_TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
_GENERATE_TOOL_PATH = _TOOLS_ROOT / "generate_capability_matrix.py"
_GENERATE_SPEC = importlib.util.spec_from_file_location(
    "generate_capability_matrix_tool", _GENERATE_TOOL_PATH
)
generate_capability_matrix = importlib.util.module_from_spec(_GENERATE_SPEC)
assert _GENERATE_SPEC.loader is not None
_GENERATE_SPEC.loader.exec_module(generate_capability_matrix)

_VALIDATE_TOOL_PATH = _TOOLS_ROOT / "validate_capability_matrix.py"
_VALIDATE_SPEC = importlib.util.spec_from_file_location(
    "validate_capability_matrix_tool", _VALIDATE_TOOL_PATH
)
validate_capability_matrix = importlib.util.module_from_spec(_VALIDATE_SPEC)
assert _VALIDATE_SPEC.loader is not None
_VALIDATE_SPEC.loader.exec_module(validate_capability_matrix)


def test_generate_capability_matrix_writes_runtime_snapshot(tmp_path):
    output = tmp_path / "matrix.json"

    written = generate_capability_matrix.write_capability_matrix(output)
    payload = json.loads(written.read_text())

    assert written == output
    assert payload == capability_matrix()
    assert payload["version"] == CAPABILITY_MATRIX_VERSION
    assert payload["features"] == list(FEATURES)
    assert payload["method_capability_features"]["RS"] == "rs_fit"
    assert payload["families"]["NO"]["features"]["production_safe"] == "validated"


def test_generated_development_artifact_is_current():
    artifact = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "development"
        / "family-capability-matrix-2026-05-18.json"
    )

    payload = json.loads(artifact.read_text())

    assert payload == capability_matrix()


def test_capability_matrix_is_public_top_level_api():
    import omnilss

    assert omnilss.CAPABILITY_MATRIX_VERSION == CAPABILITY_MATRIX_VERSION
    assert omnilss.capability_matrix() == capability_matrix()
    assert dict(omnilss.METHOD_ROUTE_FEATURES) == omnilss.method_capability_features()
    assert omnilss.method_capability_features()["CG"] == "cg_fit"
    assert omnilss.method_route_feature("RS") == "rs_fit"
    assert (
        omnilss.method_route_capability_report("NO", "RS", strict=True)["ok"]
        is True
    )
    assert (
        omnilss.validate_capability_matrix_payload(capability_matrix())["ok"]
        is True
    )


def test_validate_capability_matrix_file_accepts_generated_artifact(tmp_path):
    output = tmp_path / "matrix.json"
    generate_capability_matrix.write_capability_matrix(output)

    report = validate_capability_matrix.validate_capability_matrix_file(output)

    assert report["ok"] is True
    assert report["path"] == str(output)
    assert report["version"] == CAPABILITY_MATRIX_VERSION
    assert report["issues"] == []


def test_validate_capability_matrix_file_reports_drift(tmp_path):
    output = tmp_path / "matrix-drift.json"
    payload = capability_matrix()
    payload["version"] = CAPABILITY_MATRIX_VERSION - 1
    payload["method_routes"] = {"RS": "rs_fit"}
    output.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_capability_matrix.validate_capability_matrix_file(output)

    assert report["ok"] is False
    assert {issue["code"] for issue in report["issues"]} >= {
        "version_mismatch",
        "method_routes_mismatch",
    }
