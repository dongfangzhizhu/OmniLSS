from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

_TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "validate_model_artifact.py"
_SPEC = importlib.util.spec_from_file_location("validate_model_artifact_tool", _TOOL_PATH)
validate_model_artifact = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(validate_model_artifact)


def _write_minimal_artifact(path: Path, *, training_data_included: bool = False) -> None:
    meta = {
        "omnilss_version": "0.3.0",
        "parameters": ["mu"],
        "training_data_included": training_data_included,
        "design_matrix_schema": {
            "version": 2,
            "artifact_version": 2,
            "parameters": {
                "mu": {
                    "parameter": "mu",
                    "formula": "y ~ x",
                    "term_order": ["x"],
                    "n_columns": 2,
                }
            },
        },
    }
    import io
    import numpy as np

    arrays = io.BytesIO()
    np.savez(arrays, coef__mu=np.array([0.0, 1.0]))
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("meta.json", json.dumps(meta))
        zf.writestr("arrays.npz", arrays.getvalue())


def test_validate_artifact_tool_reports_valid_artifact(tmp_path):
    artifact = tmp_path / "valid.omnilss"
    _write_minimal_artifact(artifact)

    report = validate_model_artifact.validate_artifact(artifact)

    assert report["ok"] is True
    assert report["errors"] == []


def test_validate_artifact_tool_main_exit_codes(tmp_path, capsys):
    artifact = tmp_path / "warning.omnilss"
    _write_minimal_artifact(artifact, training_data_included=True)

    assert validate_model_artifact.main([str(artifact)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is True
    assert output["warnings"][0]["code"] == "training_data_included"

    assert validate_model_artifact.main([str(artifact), "--fail-on-warning"]) == 2


def test_validate_artifact_tool_main_fails_invalid_artifact(tmp_path, capsys):
    artifact = tmp_path / "invalid.omnilss"
    artifact.write_text("not a zip")

    assert validate_model_artifact.main([str(artifact)]) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["errors"][0]["code"] == "invalid_zip"
