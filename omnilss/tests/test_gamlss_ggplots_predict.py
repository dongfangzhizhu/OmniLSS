# SPDX-License-Identifier: GPL-3.0-or-later
"""gamlss.ggplots prediction helpers (D2.260): predict_pdf / predict_cdf /
quantile_gamlss.

All three are compositions of family d/p/q at predicted parameters, so the
contracts are the distribution identities: densities integrate to 1, discrete
pmfs sum to 1, CDFs are monotone in [0, 1], lower_tail=FALSE is the complement,
and quantile_gamlss inverts the fitted quantile.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from omnilss import (
    gamlss,
    predict_cdf,
    predict_pdf,
    quantile_gamlss,
)
from omnilss.prediction import predict_params


@pytest.fixture(scope="module")
def normal_model():
    rng = np.random.default_rng(2)
    n = 100
    x = rng.normal(size=n)
    y = 1.0 + 0.5 * x + rng.normal(scale=0.4, size=n)
    model = gamlss("y ~ x", family="NO", data={"y": y, "x": x}, max_iter=30)
    return model, np.array([-1.0, 0.0, 1.0])


@pytest.fixture(scope="module")
def poisson_model():
    rng = np.random.default_rng(4)
    n = 80
    x = rng.normal(size=n)
    y = rng.poisson(np.exp(1.0 + 0.3 * x)).astype(float)
    model = gamlss("y ~ x", family="PO", data={"y": y, "x": x}, max_iter=30)
    return model


def test_predict_pdf_is_a_density_per_newdata_row(normal_model):
    model, x_new = normal_model
    result = predict_pdf(model, {"x": x_new}, from_=-3.0, to=5.0, no_points=201, plot=False)
    assert result["grid"].size == 201
    assert result["grid"][0] == pytest.approx(-3.0)
    assert result["grid"][-1] == pytest.approx(5.0)
    assert result["discrete"] is False
    assert set(result["density"]) == {"1", "2", "3"}

    params = predict_params(model, {"x": x_new})
    for index in range(3):
        expected = stats.norm.pdf(
            result["grid"], params["mu"][index], params["sigma"][index]
        )
        np.testing.assert_allclose(result["density"][str(index + 1)], expected, atol=1e-12)
        area = float(np.trapz(result["density"][str(index + 1)], result["grid"]))
        assert area == pytest.approx(1.0, abs=0.01)


def test_predict_pdf_uses_unit_steps_for_discrete_families(poisson_model):
    short = predict_pdf(
        poisson_model, {"x": np.array([-1.0, 1.0])}, from_=0, to=8, plot=False
    )
    assert short["discrete"] is True
    np.testing.assert_allclose(short["grid"], np.arange(0.0, 9.0))

    # The grid is finite, so the pmf only sums to 1 over a wide enough range.
    result = predict_pdf(
        poisson_model, {"x": np.array([-1.0, 1.0])}, from_=0, to=30, plot=False
    )
    np.testing.assert_allclose(result["grid"], np.arange(0.0, 31.0))
    for curve in result["density"].values():
        assert np.all(curve >= 0.0)
        assert float(np.sum(curve)) == pytest.approx(1.0, abs=1e-4)


def test_predict_cdf_is_monotone_and_complements_itself(normal_model):
    model, x_new = normal_model
    lower = predict_cdf(model, {"x": x_new}, from_=-3.0, to=5.0, plot=False)
    upper = predict_cdf(
        model, {"x": x_new}, from_=-3.0, to=5.0, lower_tail=False, plot=False
    )
    assert lower["lower_tail"] is True and upper["lower_tail"] is False
    for key in lower["cdf"]:
        curve = lower["cdf"][key]
        assert np.all((curve >= 0.0) & (curve <= 1.0))
        assert np.all(np.diff(curve) >= -1e-12)
        assert curve[-1] == pytest.approx(1.0, abs=1e-6)
        np.testing.assert_allclose(upper["cdf"][key], 1.0 - curve, atol=1e-12)


def test_predict_handles_the_binomial_fixed_parameter():
    rng = np.random.default_rng(1)
    n = 60
    x = rng.normal(size=n)
    size = 20
    y = rng.binomial(size, 1.0 / (1.0 + np.exp(-(0.2 + 0.4 * x))))
    model = gamlss(
        "y ~ x", family="BI",
        data={"y": y.astype(float), "x": x, "bd": np.full(n, float(size))},
        max_iter=20,
    )
    newdata = {"x": np.array([0.0]), "bd": np.array([float(size)])}
    pmf = predict_pdf(model, newdata, from_=0, to=size, plot=False)
    assert pmf["grid"].size == size + 1
    assert float(np.sum(pmf["density"]["1"])) == pytest.approx(1.0, abs=1e-6)
    cdf = predict_cdf(model, newdata, from_=0, to=size, plot=False)
    assert float(cdf["cdf"]["1"][-1]) == pytest.approx(1.0, abs=1e-6)

    with pytest.raises(ValueError, match="fixed parameter 'bd'"):
        predict_pdf(model, {"x": np.array([0.0])}, from_=0, to=size, plot=False)


def test_quantile_gamlss_inverts_the_fitted_quantile(normal_model):
    model, x_new = normal_model
    quantiles = quantile_gamlss(model, 0.9, {"x": x_new})
    params = predict_params(model, {"x": x_new})
    expected = params["mu"] + params["sigma"] * stats.norm.ppf(0.9)
    np.testing.assert_allclose(quantiles, expected, atol=1e-12)

    # In-sample it must agree with the fitted quantile vector.
    in_sample = quantile_gamlss(model, 0.5)
    expected_median = np.asarray(model.fitted_values["mu"], dtype=float)
    np.testing.assert_allclose(in_sample, expected_median, atol=1e-9)

    # The fitted quantile maps back to the same probability.
    probability = model.family.p(
        np.asarray(quantiles),
        np.asarray(params["mu"]),
        np.asarray(params["sigma"]),
    )
    np.testing.assert_allclose(probability, 0.9, atol=1e-9)


def test_prediction_helpers_validate_their_inputs(normal_model):
    model, _ = normal_model
    with pytest.raises(ValueError, match="newdata is required"):
        predict_pdf(model)
    with pytest.raises(ValueError, match="newdata is required"):
        predict_cdf(model)
    with pytest.raises(ValueError, match="gamlss model"):
        predict_pdf("not a model", {"x": np.array([0.0])})
    with pytest.raises(ValueError, match="between 0 and 1"):
        quantile_gamlss(model, 1.5)
    with pytest.raises(ValueError, match="gamlss model"):
        quantile_gamlss("not a model", 0.5)
