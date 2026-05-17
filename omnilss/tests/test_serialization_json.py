import json
import zipfile
import numpy as np

from omnilss import gamlss
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
