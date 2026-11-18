# SPDX-License-Identifier: GPL-3.0-or-later
"""gamlss.ggplots final four (D2.263): fit_PB / pcr_coef_path / pcr_path /
histSmo_plot.

fit_PB is verified against live R (sourced R/gamlss.ggplots/R/fit_PB.R) on the
fixture below: lambda 0.601065104688, df 13.546668990737, sigma 0.183720760905,
deviance -178.786515793578, and the first four coefficients agree to 1e-9.

pcr_coef_path's long frame is R's c(row(da)), c(da), c(col(da)) with
da = t(coefficients[1:M, 1:M]): for matrix(1:9, 3, 3) R gives
lambda 1,2,3,1,2,3,1,2,3 / beta 1,4,7,2,5,8,3,6,9 / group 1,1,1,2,2,2,3,3,3.
"""
from __future__ import annotations

import numpy as np
import pytest

from omnilss import (
    PSplineFit,
    fit_PB,
    gamlss,
    histSmo,
    histSmo_plot,
    pcr_coef_path,
    pcr_path,
)
from omnilss.pcr_series import PCRFit

R_LAMBDA = 0.601065104688
R_DF = 13.546668990737
R_SIGMA = 0.183720760905
R_DEVIANCE = -178.786515793578
R_COEF = (-0.3372582516, 0.0864571091, 0.4982823195, 0.8941918976)


def _pb_data():
    rng = np.random.default_rng(7)
    n = 300
    x = np.sort(rng.uniform(0, 10, size=n))
    y = np.sin(x) + rng.normal(scale=0.2, size=n)
    return x, y


def test_fit_pb_matches_live_r():
    x, y = _pb_data()
    result = fit_PB(x=x, y=y, nseg=20, lambda_=10, order=2, degree=3)
    assert isinstance(result, PSplineFit)
    assert result.lambda_ == pytest.approx(R_LAMBDA, rel=1e-9)
    assert result.df == pytest.approx(R_DF, rel=1e-9)
    assert result.sigma == pytest.approx(R_SIGMA, rel=1e-9)
    assert result.deviance == pytest.approx(R_DEVIANCE, rel=1e-9)
    np.testing.assert_allclose(result.coef[:4], R_COEF, rtol=1e-8, atol=1e-9)
    assert result.grid.size == 500
    assert float(np.min(result.smooth)) == pytest.approx(-1.0528734374, abs=1e-8)
    assert float(np.max(result.smooth)) == pytest.approx(1.0185214458, abs=1e-8)


def test_fit_pb_structure_and_band():
    x, y = _pb_data()
    result = fit_PB(x=x, y=y, nseg=15, lambda_=5, plot=False)
    assert result.fitted_values.shape == y.shape
    assert result.lower.shape == y.shape and result.upper.shape == y.shape
    assert np.all(result.lower <= result.fitted_values + 1e-12)
    assert np.all(result.fitted_values <= result.upper + 1e-12)
    assert result.df < x.size
    assert result.penalty >= 0.0
    assert result.sigma_b > 0.0
    assert result.deviance_increment.shape == y.shape
    assert np.sum(result.deviance_increment) == pytest.approx(result.deviance)
    assert result.knots.size > 0
    assert result.figure is None  # plot=False


def test_fit_pb_weights_and_validation():
    x, y = _pb_data()
    weights = np.ones_like(y)
    weights[:10] = 0.0
    weighted = fit_PB(x=x, y=y, weights=weights, nseg=12, lambda_=3, plot=False)
    assert np.all(np.isfinite(weighted.fitted_values))
    with pytest.raises(ValueError, match="same length"):
        fit_PB(x=x, y=y[:-1])
    with pytest.raises(ValueError, match="weights should be the same length"):
        fit_PB(x=x, y=y, weights=np.ones(3))
    with pytest.raises(ValueError, match="needs x and y"):
        fit_PB()


def test_fit_pb_respects_max_df():
    x, y = _pb_data()
    result = fit_PB(x=x, y=y, nseg=20, lambda_=1e-4, max_df=5.0, plot=False)
    assert result.df <= 5.0 + 0.15


def _pcr_object(coefficients, pc=2):
    matrix = np.asarray(coefficients, dtype=np.float64)
    size = matrix.shape[0]
    return PCRFit(
        coefficients=matrix,
        scores=np.zeros((5, size)),
        loadings=np.zeros((5, size)),
        gamma=np.arange(1.0, size + 1),
        se_gamma=np.ones(size),
        center=np.zeros(5),
        scale=np.ones(5),
        fitted_values=np.zeros(5),
        xvar=np.arange(5.0),
        gaic=np.zeros(size),
        pc=pc,
        k=1.0,
        M=size,
        mean_y=0.0,
        dim_x=(5, size),
        sigma=np.ones(5),
        residuals=np.zeros(5),
    )


def test_pcr_coef_path_matches_r_long_frame():
    # R's matrix(1:9, 3, 3) is column-major: [[1,4,7],[2,5,8],[3,6,9]].
    coefficients = np.array([[1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]])
    result = pcr_coef_path(_pcr_object(coefficients), plot=False)
    frame = result["data"]
    assert list(frame.columns) == ["lambda", "beta", "group"]
    assert frame["lambda"].tolist() == [1, 2, 3, 1, 2, 3, 1, 2, 3]
    assert frame["beta"].tolist() == [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]
    assert frame["group"].tolist() == [1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert result["pc"] == 2
    assert pcr_coef_path(_pcr_object(coefficients)).figure is not None


def test_pcr_coef_path_validates_the_object():
    with pytest.raises(ValueError, match="PCR class"):
        pcr_coef_path(None)
    with pytest.raises(ValueError, match="PCR class"):
        pcr_coef_path({"coefficients": np.eye(3)})


def test_pcr_path_requires_a_pcr_smoother_and_a_valid_parameter():
    rng = np.random.default_rng(1)
    x = rng.normal(size=50)
    y = 1.0 + 0.5 * x + rng.normal(scale=0.3, size=50)
    model = gamlss("y ~ x", family="NO", data={"y": y, "x": x}, max_iter=10)
    with pytest.raises(NotImplementedError, match="no PCR smoother"):
        pcr_path(model)
    with pytest.raises(ValueError, match="parameter"):
        pcr_path(model, parameter="omega")
    with pytest.raises(ValueError, match="gamlss model"):
        pcr_path("not a model")


def test_hist_smo_plot_from_the_reduced_payload():
    rng = np.random.default_rng(4)
    y = rng.normal(size=500)
    payload = histSmo(y, bins=15)
    result = histSmo_plot(payload, plot=False)
    bars = result["bars"]
    assert bars.shape == (15, 2)
    assert result["discrete"] is False
    assert result["xname"] == "x"
    # Continuous branch: R plots the density (not the relative frequency).
    np.testing.assert_allclose(bars["freq"], payload.density)
    width = payload.bin_edges[1] - payload.bin_edges[0]
    assert float(np.sum(bars["freq"]) * width) == pytest.approx(1.0, abs=0.02)
    assert result["curve"].shape == (15, 2)
    assert histSmo_plot(payload).figure is not None


def test_hist_smo_plot_accepts_r_like_payloads():
    discrete = {
        "discrete": True,
        "counts": np.array([3.0, 5.0, 2.0]),
        "x": np.array([1.0, 2.0, 3.0]),
        "density": np.array([0.3, 0.5, 0.2]),
    }
    result = histSmo_plot(discrete, plot=False)
    assert result["discrete"] is True
    np.testing.assert_allclose(result["bars"]["freq"], [0.3, 0.5, 0.2])
    np.testing.assert_allclose(result["curve"]["x"], [1.0, 2.0, 3.0])
    assert result["title"] == "smooth histogram of x"

    continuous = {
        "discrete": False,
        "hist": {
            "counts": np.array([2.0, 4.0, 3.0]),
            "mids": np.array([0.5, 1.5, 2.5]),
            "density": np.array([0.2, 0.4, 0.3]),
            "xname": "height",
        },
        "x": np.linspace(0.0, 3.0, 20),
        "density": np.linspace(0.1, 0.5, 20),
    }
    result = histSmo_plot(continuous, plot=False, title="custom")
    assert result["xname"] == "height"
    assert result["title"] == "custom"
    np.testing.assert_allclose(result["bars"]["freq"], [0.2, 0.4, 0.3])
    assert result["curve"].shape == (20, 2)

    with pytest.raises(ValueError, match="histSmo"):
        histSmo_plot()
