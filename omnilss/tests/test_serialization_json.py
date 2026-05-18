import json
import zipfile
import numpy as np
import pytest

from omnilss import (
    artifact_schema_policy as top_level_artifact_schema_policy,
    compare_model_capability_snapshot as top_level_compare_model_capability_snapshot,
    gamlss,
    validate_model_json as top_level_validate_model_json,
)
from omnilss.serialization import load_model_json, save_model_json


def test_json_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    n = 80
    x = rng.normal(size=n)
    y = 1.5 + 0.4 * x + rng.normal(scale=0.2, size=n)
    model = gamlss("y ~ x", family="NO", data={"y": y, "x": x})

    path = tmp_path / "m.omnilss"
    save_model_json(model, path)
    loaded = load_model_json(path)

    assert loaded.additional_slots.get("loaded_from_json") is True
    assert abs(float(loaded.g_dev) - float(model.g_dev)) < 1e-8
    assert np.max(np.abs(np.asarray(loaded.fitted_values["mu"]) - np.asarray(model.fitted_values["mu"]))) < 1e-8


def test_json_roundtrip_preserves_linear_prediction(tmp_path):
    rng = np.random.default_rng(22)
    n = 80
    x = rng.normal(size=n)
    y = 1.5 + 0.4 * x + rng.normal(scale=0.2, size=n)
    model = gamlss("y ~ x", family="NO", data={"y": y, "x": x})

    path = tmp_path / "m_predict.omnilss"
    save_model_json(model, path)
    loaded = load_model_json(path)

    newdata = {"x": np.array([-1.0, 0.0, 1.0])}
    original = model.predict_params(newdata)
    restored = loaded.predict_params(newdata)
    for key in original:
        np.testing.assert_allclose(restored[key], original[key], rtol=1e-10, atol=1e-10)


def test_json_artifact_contains_design_matrix_schema(tmp_path):
    rng = np.random.default_rng(23)
    n = 50
    x = rng.normal(size=n)
    y = 1.5 + 0.4 * x + rng.normal(scale=0.2, size=n)
    model = gamlss("y ~ x", family="NO", data={"y": y, "x": x})

    path = tmp_path / "m_schema.omnilss"
    save_model_json(model, path)

    with zipfile.ZipFile(path, "r") as zf:
        meta = json.loads(zf.read("meta.json").decode("utf-8"))
    assert meta["design_matrix_schema"]["version"] == 2
    assert meta["design_matrix_schema"]["artifact_version"] == 2
    schema = meta["design_matrix_schema"]["parameters"]["mu"]
    assert schema["formula"] == "y ~ x"
    assert schema["term_labels"] == ["x"]
    assert schema["term_order"] == ["x"]
    assert schema["column_names"] == ["(Intercept)", "x"]
    assert schema["n_columns"] == 2
    assert schema["coefficient_count"] == 2

    loaded = load_model_json(path)
    assert loaded.additional_slots["design_matrix_schema"]["parameters"]["mu"] == schema


def test_json_artifact_contains_family_capability_snapshot(tmp_path):
    rng = np.random.default_rng(25)
    n = 50
    x = rng.normal(size=n)
    y = 1.5 + 0.4 * x + rng.normal(scale=0.2, size=n)
    model = gamlss("y ~ x", family="NO", data={"y": y, "x": x})

    path = tmp_path / "m_capability.omnilss"
    save_model_json(model, path)

    with zipfile.ZipFile(path, "r") as zf:
        meta = json.loads(zf.read("meta.json").decode("utf-8"))

    capability = meta["family_capability"]
    assert capability["name"] == "NO"
    assert capability["features"]["production_safe"] == "validated"
    assert capability["features"]["r_consistency"] == "validated"

    loaded = load_model_json(path)
    assert loaded.additional_slots["family_capability"] == capability


def test_fit_attaches_design_matrix_schema_before_serialization():
    rng = np.random.default_rng(24)
    n = 40
    x = rng.normal(size=n)
    y = 1.5 + 0.4 * x + rng.normal(scale=0.2, size=n)
    model = gamlss("y ~ x", family="NO", data={"y": y, "x": x})

    schema = model.additional_slots["design_matrix_schema"]["parameters"]["mu"]
    assert schema["raw_formula"] == "y ~ x"
    assert schema["term_order"] == ["x"]
    assert schema["training_column_checksum"]


def test_json_artifact_omits_training_data_by_default(tmp_path):
    rng = np.random.default_rng(26)
    n = 40
    x = rng.normal(size=n)
    y = 1.5 + 0.4 * x + rng.normal(scale=0.2, size=n)
    model = gamlss("y ~ x", family="NO", data={"y": y, "x": x})

    path = tmp_path / "m_no_training_data.omnilss"
    save_model_json(model, path)

    with zipfile.ZipFile(path, "r") as zf:
        meta = json.loads(zf.read("meta.json").decode("utf-8"))
        import io

        arrays = np.load(io.BytesIO(zf.read("arrays.npz")))
        array_names = set(arrays.files)

    assert meta["training_data_included"] is False
    assert meta["call"]["training_data_omitted"] is True
    assert "data" not in meta["call"]
    assert "y" not in array_names

    loaded = load_model_json(path)
    assert loaded.additional_slots["training_data_included"] is False
    assert len(np.asarray(loaded.y)) == 0


def test_json_artifact_can_include_training_response_explicitly(tmp_path):
    rng = np.random.default_rng(27)
    n = 30
    x = rng.normal(size=n)
    y = 1.5 + 0.4 * x + rng.normal(scale=0.2, size=n)
    model = gamlss("y ~ x", family="NO", data={"y": y, "x": x})

    path = tmp_path / "m_with_training_data.omnilss"
    save_model_json(model, path, include_training_data=True)

    with zipfile.ZipFile(path, "r") as zf:
        meta = json.loads(zf.read("meta.json").decode("utf-8"))
        import io

        arrays = np.load(io.BytesIO(zf.read("arrays.npz")))
        array_names = set(arrays.files)

    assert meta["training_data_included"] is True
    assert "data" in meta["call"]
    assert "y" in array_names

    loaded = load_model_json(path)
    np.testing.assert_allclose(np.asarray(loaded.y), y)


def test_json_artifact_preserves_df_fit(tmp_path):
    rng = np.random.default_rng(28)
    n = 40
    x = rng.normal(size=n)
    y = 1.5 + 0.4 * x + rng.normal(scale=0.2, size=n)
    model = gamlss("y ~ x", family="NO", data={"y": y, "x": x})
    model.df_fit = 1.25

    path = tmp_path / "m_df_fit.omnilss"
    save_model_json(model, path)
    loaded = load_model_json(path)

    assert loaded.df_fit == 1.25


def test_json_artifact_preserves_smooth_metadata_and_prediction(tmp_path):
    rng = np.random.default_rng(29)
    n = 70
    x = np.linspace(0.0, 1.0, n)
    y = 0.2 + np.sin(2 * np.pi * x) + rng.normal(scale=0.05, size=n)
    model = gamlss("y ~ pb(x)", family="NO", data={"y": y, "x": x}, max_iter=3)

    path = tmp_path / "m_smooth.omnilss"
    save_model_json(model, path)

    with zipfile.ZipFile(path, "r") as zf:
        meta = json.loads(zf.read("meta.json").decode("utf-8"))

    smooth_meta = meta["smooth_infos"]["mu"][0]
    assert smooth_meta["variable"] == "x"
    assert smooth_meta["smoother"] == "pb"
    assert smooth_meta["basis_smoother"] == "pb"
    assert smooth_meta["knots"]

    schema_meta = meta["design_matrix_schema"]["parameters"]["mu"]
    assert schema_meta["smooth_metadata_required"] is True
    assert schema_meta["smooth_basis_metadata"][0]["variable"] == "x"

    newdata = {"x": np.array([0.1, 0.3, 0.7])}
    original = model.predict_params(newdata)["mu"]
    loaded = load_model_json(path)
    restored = loaded.predict_params(newdata)["mu"]
    np.testing.assert_allclose(restored, original, rtol=1e-7, atol=1e-7)




def test_artifact_schema_policy_is_public() -> None:
    from omnilss.serialization import artifact_schema_policy

    policy = artifact_schema_policy()

    assert top_level_artifact_schema_policy() == policy
    assert policy["type"] == "artifact_schema_policy"
    assert policy["current_design_schema_version"] == 2
    assert policy["supported_artifact_schema_versions"] == [2]

def test_validate_model_json_accepts_schema_safe_artifact(tmp_path):
    rng = np.random.default_rng(32)
    n = 50
    x = rng.normal(size=n)
    y = 1.5 + 0.4 * x + rng.normal(scale=0.2, size=n)
    model = gamlss("y ~ x", family="NO", data={"y": y, "x": x})

    path = tmp_path / "valid.omnilss"
    save_model_json(model, path)

    from omnilss.serialization import validate_model_json

    result = validate_model_json(path)
    assert top_level_validate_model_json(path)["ok"] is True
    assert result["type"] == "artifact_validation_report"
    assert result["version"] == 1
    assert result["ok"] is True
    assert result["error_count"] == 0
    assert result["errors"] == []


def test_validate_model_json_reports_schema_mismatch(tmp_path):
    rng = np.random.default_rng(33)
    n = 50
    x = rng.normal(size=n)
    y = 1.5 + 0.4 * x + rng.normal(scale=0.2, size=n)
    model = gamlss("y ~ x", family="NO", data={"y": y, "x": x})

    path = tmp_path / "invalid.omnilss"
    save_model_json(model, path)

    with zipfile.ZipFile(path, "r") as zf:
        meta = json.loads(zf.read("meta.json").decode("utf-8"))
        arrays_payload = zf.read("arrays.npz")
    meta["design_matrix_schema"]["parameters"]["mu"]["n_columns"] = 99
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("meta.json", json.dumps(meta))
        zf.writestr("arrays.npz", arrays_payload)

    from omnilss.serialization import validate_model_json

    result = validate_model_json(path)
    assert result["ok"] is False
    assert result["error_count"] == 1
    mismatch = next(
        error
        for error in result["errors"]
        if error["code"] == "coefficient_schema_mismatch"
    )
    assert mismatch["type"] == "artifact_validation_issue"
    assert mismatch["severity"] == "error"


def test_validate_model_json_warns_when_training_data_included(tmp_path):
    rng = np.random.default_rng(34)
    n = 30
    x = rng.normal(size=n)
    y = 1.5 + 0.4 * x + rng.normal(scale=0.2, size=n)
    model = gamlss("y ~ x", family="NO", data={"y": y, "x": x})

    path = tmp_path / "with_training.omnilss"
    save_model_json(model, path, include_training_data=True)

    from omnilss.serialization import validate_model_json

    result = validate_model_json(path)
    assert result["ok"] is True
    assert result["warning_count"] == 1
    warning = next(
        warning
        for warning in result["warnings"]
        if warning["code"] == "training_data_included"
    )
    assert warning["type"] == "artifact_validation_issue"
    assert warning["severity"] == "warning"


def test_json_artifact_preserves_terms_for_validation_wrappers(tmp_path):
    rng = np.random.default_rng(39)
    x = np.linspace(0.0, 1.0, 40)
    y = 1.0 + x + rng.normal(scale=0.05, size=len(x))
    model = gamlss("y ~ x", family="NO", data={"y": y, "x": x}, max_iter=2)

    path = tmp_path / "terms.omnilss"
    save_model_json(model, path)
    loaded = load_model_json(path)

    assert loaded.terms["mu"]["response"] == "y"
    assert loaded.terms["mu"]["term_labels"] == ["x"]


def test_model_capability_snapshot_report_matches_runtime_registry(tmp_path):
    from omnilss.serialization import compare_model_capability_snapshot

    rng = np.random.default_rng(40)
    x = np.linspace(0.0, 1.0, 40)
    y = 0.5 + x + rng.normal(scale=0.05, size=len(x))
    model = gamlss("y ~ x", family="NO", data={"y": y, "x": x}, max_iter=2)

    path = tmp_path / "capability.omnilss"
    save_model_json(model, path)

    report_from_path = compare_model_capability_snapshot(path)
    loaded = load_model_json(path)
    report_from_model = compare_model_capability_snapshot(loaded)

    assert report_from_path["ok"] is True
    assert top_level_compare_model_capability_snapshot(path) == report_from_path
    assert report_from_path["family"] == "NO"
    assert report_from_path["changes"] == []
    assert report_from_model == report_from_path


def test_model_capability_snapshot_report_detects_drift(tmp_path):
    from omnilss.serialization import compare_model_capability_snapshot

    rng = np.random.default_rng(41)
    x = np.linspace(0.0, 1.0, 40)
    y = 0.5 + x + rng.normal(scale=0.05, size=len(x))
    model = gamlss("y ~ x", family="NO", data={"y": y, "x": x}, max_iter=2)

    path = tmp_path / "capability-drift.omnilss"
    save_model_json(model, path)
    loaded = load_model_json(path)
    saved = dict(loaded.additional_slots["family_capability"])
    features = dict(saved["features"])
    features["prediction"] = "experimental"
    saved["features"] = features
    loaded.additional_slots = {**loaded.additional_slots, "family_capability": saved}

    report = compare_model_capability_snapshot(loaded)

    assert report["ok"] is False
    assert {
        "code": "capability_status_changed",
        "feature": "prediction",
        "artifact_status": "experimental",
        "runtime_status": "validated",
    } in report["changes"]


def test_load_model_json_rejects_future_artifact_schema(tmp_path):
    rng = np.random.default_rng(47)
    n = 30
    x = rng.normal(size=n)
    y = 1.5 + 0.4 * x + rng.normal(scale=0.2, size=n)
    model = gamlss("y ~ x", family="NO", data={"y": y, "x": x})

    path = tmp_path / "future-schema.omnilss"
    save_model_json(model, path)

    with zipfile.ZipFile(path, "r") as zf:
        meta = json.loads(zf.read("meta.json").decode("utf-8"))
        arrays_payload = zf.read("arrays.npz")
    meta["design_matrix_schema"]["artifact_version"] = 99
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("meta.json", json.dumps(meta))
        zf.writestr("arrays.npz", arrays_payload)

    with pytest.raises(ValueError, match="newer than this runtime supports"):
        load_model_json(path)
