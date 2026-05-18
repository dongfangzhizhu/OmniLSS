from __future__ import annotations

import numpy as np

from omnilss import gamlss
from omnilss.prediction_runtime import (
    predict_distribution,
    predict_interval,
    predict_mean,
    predict_quantile,
)


def _make_no_model():
    rng = np.random.default_rng(20260517)
    x = np.linspace(-1.0, 1.0, 32)
    y = 1.0 + 0.4 * x + rng.normal(scale=0.05, size=x.size)
    return gamlss("y ~ x", family="NO", data={"y": y, "x": x}, max_iter=2)


def test_prediction_runtime_wrappers_delegate_to_schema_safe_params():
    model = _make_no_model()
    newdata = {"x": np.array([-0.5, 0.0, 0.5])}

    mean = predict_mean(model, newdata)
    distribution = predict_distribution(model, newdata)

    np.testing.assert_allclose(mean, model.predict_params(newdata)["mu"])
    assert distribution["family"] == "NO"
    assert set(distribution["params"]) >= {"mu", "sigma"}
    np.testing.assert_allclose(distribution["params"]["mu"], mean)


def test_prediction_runtime_quantiles_use_all_distribution_parameters():
    model = _make_no_model()
    newdata = {"x": np.array([-0.25, 0.25])}

    median = predict_quantile(model, newdata, q=0.5)
    lower, upper = predict_interval(model, newdata, alpha=0.2)

    expected = model.predict_quantiles(newdata, quantiles=[0.1, 0.5, 0.9])
    np.testing.assert_allclose(median, expected[0.5])
    np.testing.assert_allclose(lower, expected[0.1])
    np.testing.assert_allclose(upper, expected[0.9])
