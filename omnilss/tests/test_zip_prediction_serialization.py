"""End-to-end ZIP prediction, newdata, weights, and artifact contracts."""

from __future__ import annotations

import numpy as np

from omnilss import gamlss
from omnilss.serialization import load_model_json, save_model_json, validate_model_json


def _fit_zip():
    rng = np.random.default_rng(2026)
    x = np.linspace(-1.5, 1.5, 90)
    mu = np.exp(0.35 + 0.25 * x)
    is_extra_zero = rng.random(x.size) < 0.25
    y = np.where(is_extra_zero, 0, rng.poisson(mu)).astype(float)
    weights = np.linspace(0.75, 1.25, x.size)
    return gamlss(
        "y ~ x",
        family="ZIP",
        data={"y": y, "x": x},
        sigma_formula="~1",
        weights=weights,
        max_iter=30,
    )


def test_zip_newdata_prediction_preserves_parameter_shapes():
    model = _fit_zip()
    newdata = {"x": np.array([-1.0, 0.0, 1.0])}
    params = model.predict_params(newdata)
    assert set(params) == {"mu", "sigma"}
    assert params["mu"].shape == (3,)
    assert params["sigma"].shape == (3,)
    assert np.all(params["mu"] > 0)
    assert np.all((params["sigma"] > 0) & (params["sigma"] < 1))
    assert model.call["parameter_formulas"]["sigma"] == "y ~1"
    response = model.predict(newdata)
    assert np.asarray(response).shape == (3,)


def test_zip_weights_are_retained_and_artifact_roundtrip_predicts(tmp_path):
    model = _fit_zip()
    assert np.asarray(model.weights).shape == (90,)
    newdata = {"x": np.array([-1.0, 0.0, 1.0])}
    original = model.predict_params(newdata)
    path = tmp_path / "zip.omnilss"
    save_model_json(model, path)
    report = validate_model_json(path)
    assert report["ok"] is True, report
    restored = load_model_json(path)
    np.testing.assert_allclose(restored.weights, model.weights)
    predicted = restored.predict_params(newdata)
    for parameter in original:
        np.testing.assert_allclose(predicted[parameter], original[parameter], rtol=1e-8, atol=1e-8)


def test_zip_sigma_parameter_formula_rebuilds_on_newdata(tmp_path):
    model = _fit_zip()
    # The public sigma_formula is the ZIP second-parameter formula path.
    model = gamlss(
        "y ~ x",
        family="ZIP",
        data={"y": np.asarray(model.y), "x": np.linspace(-1.5, 1.5, 90)},
        sigma_formula="~ x",
        max_iter=15,
    )
    assert len(model.coefficients["sigma"]) == 2
    assert "x" in model.call["parameter_formulas"]["sigma"]
    newdata = {"x": np.array([-1.0, 0.0, 1.0])}
    original = model.predict_params(newdata)["sigma"]
    path = tmp_path / "zip_sigma_formula.omnilss"
    save_model_json(model, path)
    restored = load_model_json(path)
    np.testing.assert_allclose(restored.predict_params(newdata)["sigma"], original)


def test_zip_prediction_schema_rejects_missing_newdata_column():
    model = _fit_zip()
    try:
        model.predict_params({"wrong": np.array([1.0])})
    except Exception as exc:
        assert "x" in str(exc)
    else:
        raise AssertionError("missing ZIP predictor column was silently accepted")
