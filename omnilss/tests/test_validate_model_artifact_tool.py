from __future__ import annotations

import ast
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


def _write_schema_artifact(
    path: Path, param_schema: dict, *, coef_count: int = 2
) -> None:
    meta = {
        "omnilss_version": "0.3.0",
        "parameters": ["mu"],
        "training_data_included": False,
        "design_matrix_schema": {
            "version": 2,
            "artifact_version": 2,
            "parameters": {"mu": {"parameter": "mu", **param_schema}},
        },
    }
    import io
    import numpy as np

    arrays = io.BytesIO()
    np.savez(arrays, coef__mu=np.zeros(coef_count))
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("meta.json", json.dumps(meta))
        zf.writestr("arrays.npz", arrays.getvalue())


def test_validate_artifact_tool_reports_valid_artifact(tmp_path):
    artifact = tmp_path / "valid.omnilss"
    _write_minimal_artifact(artifact)

    report = validate_model_artifact.validate_artifact(artifact)

    assert report["type"] == "artifact_validation_report"
    assert report["version"] == 1
    assert report["ok"] is True
    assert report["error_count"] == 0
    assert report["errors"] == []


def test_validate_artifact_tool_main_exit_codes(tmp_path, capsys):
    artifact = tmp_path / "warning.omnilss"
    _write_minimal_artifact(artifact, training_data_included=True)

    assert validate_model_artifact.main([str(artifact)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is True
    assert output["warning_count"] == 1
    assert output["warnings"][0]["type"] == "artifact_validation_issue"
    assert output["warnings"][0]["severity"] == "warning"
    assert output["warnings"][0]["code"] == "training_data_included"

    assert validate_model_artifact.main([str(artifact), "--fail-on-warning"]) == 2


def test_validate_artifact_tool_main_fails_invalid_artifact(tmp_path, capsys):
    artifact = tmp_path / "invalid.omnilss"
    artifact.write_text("not a zip")

    assert validate_model_artifact.main([str(artifact)]) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["type"] == "artifact_validation_report"
    assert output["error_count"] == 1
    assert output["errors"][0]["type"] == "artifact_validation_issue"
    assert output["errors"][0]["severity"] == "error"
    assert output["errors"][0]["code"] == "invalid_zip"


def test_validate_artifact_tool_requires_factor_levels(tmp_path):
    artifact = tmp_path / "missing-factor-levels.omnilss"
    _write_schema_artifact(
        artifact,
        {
            "formula": "y ~ factor(grp)",
            "term_order": ["factor(grp)"],
            "factor_levels": {},
            "n_columns": 2,
        },
    )

    report = validate_model_artifact.validate_artifact(artifact)

    assert report["ok"] is False
    assert any(error["code"] == "missing_factor_levels" for error in report["errors"])


def test_validate_artifact_tool_accepts_factor_level_schema(tmp_path):
    artifact = tmp_path / "factor-levels.omnilss"
    _write_schema_artifact(
        artifact,
        {
            "formula": "y ~ factor(grp)",
            "term_order": ["factor(grp)"],
            "factor_levels": {"grp": ["a", "b"]},
            "n_columns": 2,
        },
    )

    report = validate_model_artifact.validate_artifact(artifact)

    assert report["ok"] is True
    assert report["errors"] == []


def test_validate_artifact_tool_requires_numeric_transform_ast(tmp_path):
    artifact = tmp_path / "missing-transform-ast.omnilss"
    _write_schema_artifact(
        artifact,
        {
            "formula": "y ~ x + x * x",
            "term_order": ["x * x"],
            "numeric_transform_ast": {},
            "n_columns": 2,
        },
    )

    report = validate_model_artifact.validate_artifact(artifact)

    assert report["ok"] is False
    assert any(
        error["code"] == "missing_numeric_transform_ast"
        for error in report["errors"]
    )


def test_validate_artifact_tool_accepts_numeric_transform_ast(tmp_path):
    artifact = tmp_path / "transform-ast.omnilss"
    term = "x * x"
    _write_schema_artifact(
        artifact,
        {
            "formula": "y ~ x + x * x",
            "term_order": [term],
            "numeric_transform_ast": {
                term: ast.dump(ast.parse(term, mode="eval"), include_attributes=False)
            },
            "n_columns": 2,
        },
    )

    report = validate_model_artifact.validate_artifact(artifact)

    assert report["ok"] is True
    assert report["errors"] == []
