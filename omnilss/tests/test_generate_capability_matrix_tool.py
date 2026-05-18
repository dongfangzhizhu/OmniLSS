from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from omnilss.family_capabilities import (
    CAPABILITY_MATRIX_VERSION,
    FEATURES,
    capability_matrix,
)

_TOOL_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "generate_capability_matrix.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "generate_capability_matrix_tool", _TOOL_PATH
)
generate_capability_matrix = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(generate_capability_matrix)


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
    assert omnilss.method_route_capability_report("NO", "RS", strict=True)["ok"] is True
