"""Public-API contracts from the v0.1 execution plan (E05/E06).

The plain-user path must be: fit -> summary() -> predict_mean ->
predict_distribution -> predict_quantiles, without reaching into
additional_slots for deviance/AIC/EDF/convergence.
"""

import numpy as np

from omnilss import gamlss


def _fit_no():
    rng = np.random.default_rng(20261122)
    n = 200
    x = rng.uniform(0.0, 10.0, n)
    y = 2.0 + 3.0 * x + rng.normal(0.0, 1.0 + 0.2 * x)
    model = gamlss(
        "y ~ x",
        sigma_formula="~ x",
        family="NO",
        data={"y": y, "x": x},
    )
    return model, x


def test_model_summary_exposes_release_metrics():
    model, _ = _fit_no()
    result = model.summary()

    assert result.family == "NO"
    assert result.converged is True
    assert np.isfinite(result.global_deviance)
    assert np.isfinite(result.aic)
    assert result.edf > 0.0
    assert isinstance(result.warnings, tuple)
    assert set(result.coefficients) == {"mu", "sigma"}
    for block in result.coefficients.values():
        assert np.isfinite(block["estimate"]).all()


def test_model_prediction_helpers_agree_with_predict_params():
    model, x = _fit_no()
    newdata = {"x": x[:8]}

    params = model.predict_params(newdata)
    mean = np.asarray(model.predict_mean(newdata))
    distribution = model.predict_distribution(newdata)
    quantiles = model.predict_quantiles(newdata, quantiles=[0.05, 0.5, 0.95])

    # Normal mean is mu.
    np.testing.assert_allclose(mean, np.asarray(params["mu"]), rtol=1e-10)
    assert set(distribution) == {"family", "params"}
    assert set(distribution["params"]) == {"mu", "sigma"}
    assert set(quantiles) == {0.05, 0.5, 0.95}
    assert all(np.asarray(v).shape == (8,) for v in quantiles.values())


def test_model_save_load_round_trip_is_pickle_free(tmp_path):
    import zipfile

    model, x = _fit_no()
    newdata = {"x": x[:6]}
    expected = np.asarray(model.predict_params(newdata)["mu"])

    path = tmp_path / "model.oml"
    model.save(path)

    # The artifact is a ZIP container with JSON + npz, never a pickle stream.
    assert zipfile.is_zipfile(path)
    with zipfile.ZipFile(path) as zf:
        assert {"meta.json", "arrays.npz"} <= set(zf.namelist())
    assert b"pickle" not in path.read_bytes()[:4096]

    restored = type(model).load(path)
    np.testing.assert_allclose(
        np.asarray(restored.predict_params(newdata)["mu"]), expected, rtol=1e-12
    )
    assert restored.family.name == model.family.name
