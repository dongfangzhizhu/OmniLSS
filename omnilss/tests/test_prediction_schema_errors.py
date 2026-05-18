import numpy as np
import pytest

from omnilss import gamlss
from omnilss.distributions import resolve_family
from omnilss.model import GAMLSSModel
from omnilss.prediction import PredictionSchemaError
from omnilss.serialization import load_model_json, save_model_json


def test_prediction_schema_error_for_unseen_factor_level():
    family = resolve_family("NO")
    model = GAMLSSModel(
        par=family.parameters,
        family=family,
        df_fit=2.0,
        g_dev=0.0,
        n=4,
        y=np.array([], dtype=np.float64),
        coefficients={"mu": np.array([0.0, 1.0])},
        formulas={"mu": "y ~ factor(grp)"},
        parameters=family.parameters,
        additional_slots={
            "design_matrix_schema": {
                "version": 2,
                "artifact_version": 2,
                "parameters": {
                    "mu": {
                        "parameter": "mu",
                        "formula": "y ~ factor(grp)",
                        "term_order": ["factor(grp)"],
                        "has_intercept": True,
                        "factor_levels": {"grp": ["a", "b"]},
                        "n_columns": 2,
                        "coefficient_count": 2,
                    }
                },
            }
        },
    )

    with pytest.raises(PredictionSchemaError) as exc_info:
        model.predict_params({"grp": np.array(["c", "a"])})

    err = exc_info.value
    assert err.code == "unseen_factor_levels"
    assert err.parameter == "mu"
    assert err.term == "factor(grp)"
    assert "c" in err.reason


def test_prediction_schema_error_for_missing_smooth_metadata(tmp_path):
    rng = np.random.default_rng(30)
    x = np.linspace(0.0, 1.0, 50)
    y = np.sin(2 * np.pi * x) + rng.normal(scale=0.05, size=len(x))
    model = gamlss("y ~ pb(x)", family="NO", data={"y": y, "x": x}, max_iter=2)

    path = tmp_path / "smooth.omnilss"
    save_model_json(model, path)
    loaded = load_model_json(path)
    loaded.additional_slots = {**loaded.additional_slots, "smooth_infos": {}}

    with pytest.raises(PredictionSchemaError) as exc_info:
        loaded.predict_params({"x": np.array([0.1, 0.2])})

    err = exc_info.value
    assert err.code == "missing_smooth_metadata"
    assert err.parameter == "mu"
    assert err.term == "pb(x)"


def test_prediction_schema_error_for_column_count_mismatch(tmp_path):
    rng = np.random.default_rng(31)
    x = rng.normal(size=40)
    y = 1.0 + 0.5 * x + rng.normal(scale=0.1, size=40)
    model = gamlss("y ~ x", family="NO", data={"y": y, "x": x}, max_iter=2)

    path = tmp_path / "linear.omnilss"
    save_model_json(model, path)
    loaded = load_model_json(path)
    schema = loaded.additional_slots["design_matrix_schema"]["parameters"]["mu"]
    schema["n_columns"] = 3

    with pytest.raises(PredictionSchemaError) as exc_info:
        loaded.predict_params({"x": np.array([0.0, 1.0])})

    err = exc_info.value
    assert err.code == "schema_column_mismatch"
    assert err.parameter == "mu"
    assert "column count" in err.reason


def test_prediction_schema_error_to_dict_envelope():
    err = PredictionSchemaError(
        "bad prediction schema",
        parameter="mu",
        term="factor(grp)",
        reason="unseen factor levels ['c']",
        code="unseen_factor_levels",
    )

    assert err.to_dict() == {
        "code": "unseen_factor_levels",
        "parameter": "mu",
        "term": "factor(grp)",
        "reason": "unseen factor levels ['c']",
        "message": "bad prediction schema",
    }


def test_legacy_predict_reuses_schema_safe_factor_validation():
    from omnilss.predict_gamlss_23_12_21 import predict

    family = resolve_family("NO")
    model = GAMLSSModel(
        par=("mu",),
        family=family,
        df_fit=2.0,
        g_dev=0.0,
        n=4,
        y=np.array([], dtype=np.float64),
        coefficients={"mu": np.array([0.0, 1.0])},
        formulas={"mu": "y ~ factor(grp)"},
        terms={"mu": {"term_labels": ["factor(grp)"], "intercept": True}},
        parameters=("mu",),
        additional_slots={
            "design_matrix_schema": {
                "version": 2,
                "artifact_version": 2,
                "parameters": {
                    "mu": {
                        "parameter": "mu",
                        "formula": "y ~ factor(grp)",
                        "term_order": ["factor(grp)"],
                        "has_intercept": True,
                        "factor_levels": {"grp": ["a", "b"]},
                        "n_columns": 2,
                        "coefficient_count": 2,
                    }
                },
            }
        },
    )

    with pytest.raises(PredictionSchemaError) as exc_info:
        predict(model, what="mu", newdata={"grp": np.array(["c", "a"])})

    assert exc_info.value.code == "unseen_factor_levels"


def test_legacy_predict_all_reuses_schema_safe_smooth_validation(tmp_path):
    from omnilss.predictAll_22_08_22 import predict_all

    rng = np.random.default_rng(32)
    x = np.linspace(0.0, 1.0, 50)
    y = np.sin(2 * np.pi * x) + rng.normal(scale=0.05, size=len(x))
    model = gamlss("y ~ pb(x)", family="NO", data={"y": y, "x": x}, max_iter=2)

    path = tmp_path / "smooth.omnilss"
    save_model_json(model, path)
    loaded = load_model_json(path)
    loaded.additional_slots = {**loaded.additional_slots, "smooth_infos": {}}

    with pytest.raises(PredictionSchemaError) as exc_info:
        predict_all(loaded, newdata={"x": np.array([0.1, 0.2])}, output="data.frame")

    assert exc_info.value.code == "missing_smooth_metadata"


def test_prodist_reuses_schema_safe_prediction_errors(tmp_path):
    from omnilss.prodist import prodist_data

    rng = np.random.default_rng(33)
    x = np.linspace(0.0, 1.0, 50)
    y = np.cos(2 * np.pi * x) + rng.normal(scale=0.05, size=len(x))
    model = gamlss("y ~ pb(x)", family="NO", data={"y": y, "x": x}, max_iter=2)

    path = tmp_path / "smooth-prodist.omnilss"
    save_model_json(model, path)
    loaded = load_model_json(path)
    loaded.additional_slots = {**loaded.additional_slots, "smooth_infos": {}}

    with pytest.raises(PredictionSchemaError) as exc_info:
        prodist_data(loaded, newdata={"x": np.array([0.1, 0.2])})

    assert exc_info.value.code == "missing_smooth_metadata"


def test_get_pef_reuses_schema_safe_prediction_errors():
    from omnilss.getPEF import get_pef_data

    rng = np.random.default_rng(34)
    x = np.linspace(0.0, 1.0, 50)
    y = np.sin(2 * np.pi * x) + rng.normal(scale=0.05, size=len(x))
    model = gamlss("y ~ pb(x)", family="NO", data={"y": y, "x": x}, max_iter=2)
    model.additional_slots = {**model.additional_slots, "smooth_infos": {}}

    with pytest.raises(PredictionSchemaError) as exc_info:
        get_pef_data(model, term="x", n_points=5)

    assert exc_info.value.code == "missing_smooth_metadata"
