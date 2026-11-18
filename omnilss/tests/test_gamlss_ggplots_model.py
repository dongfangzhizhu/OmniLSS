# SPDX-License-Identifier: GPL-3.0-or-later
"""gamlss.ggplots model_* comparisons and diagnostics (D2.257).

GAIC/TD ordering follows R exactly:

    delta  = max(value) - value        (worst model gets 0, best gets the span)
    scaled = delta / (max - min)

verified against live R's model_GAIC on the matrix branch
(matrix(c(100, 101, 110)) -> delta 10/9/0, scaled 1.0/0.9/0.0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from omnilss import (
    GGPlot,
    gamlss,
    model_GAIC,
    model_GAIC_lollipop,
    model_TD,
    model_TD_lollipop,
    model_centiles,
    model_devianceIncr_diff,
    model_mom_bucket,
    model_pca,
    model_wp,
    model_wp_wrap,
)
from omnilss.DevianceIncr import devianceIncr
from omnilss.ggplots import _ppoints
from omnilss.gamlssVGD_23_12_21 import get_tgd


@pytest.fixture(scope="module")
def models():
    rng = np.random.default_rng(3)
    n = 120
    x = rng.normal(size=n)
    z = rng.uniform(-1.0, 1.0, size=n)
    y = 1.0 + 0.5 * x + 0.3 * z + rng.normal(scale=0.4, size=n)
    data = {"y": y, "x": x, "z": z}
    small = gamlss("y ~ x", family="NO", data=data, max_iter=40)
    full = gamlss("y ~ x + z", family="NO", data=data, max_iter=40)
    return small, full, data


def test_model_gaic_single_model_is_deviance_plus_penalty(models):
    small, _, _ = models
    value = model_GAIC(small, k=2)
    assert isinstance(value, float)
    assert value == pytest.approx(float(small.g_dev) + 2.0 * float(small.df_fit))
    corrected = model_GAIC(small, k=2, c=True)
    expected = float(small.g_dev) + 2.0 * float(small.df_fit) + (
        2.0 * small.df_fit * (small.df_fit + 1.0) / (small.n - small.df_fit - 1.0)
    )
    assert corrected == pytest.approx(expected)
    assert model_GAIC(small, k=3.84) == pytest.approx(
        float(small.g_dev) + 3.84 * float(small.df_fit)
    )


def test_model_gaic_matrix_matches_live_r():
    matrix = np.array([[100.0, 1.0], [101.0, 2.0], [110.0, 3.0]])
    table = model_GAIC(matrix, plot=False)
    assert list(table.columns) == ["GAIC", "delta", "scaled"]
    np.testing.assert_allclose(table["GAIC"], [100.0, 101.0, 110.0])
    np.testing.assert_allclose(table["delta"], [10.0, 9.0, 0.0])
    np.testing.assert_allclose(table["scaled"], [1.0, 0.9, 0.0])
    with pytest.raises(ValueError, match="which"):
        model_GAIC(matrix, which=5, plot=False)
    with pytest.raises(ValueError, match="scale"):
        model_GAIC(matrix, scale="[0,1", plot=False)


def test_model_gaic_model_list_keeps_r_delta_direction(models):
    small, full, _ = models
    table = model_GAIC(small, full, plot=False)
    assert list(table.columns) == ["df", "AIC", "delta", "scaled"]
    assert list(table.index) == ["model_1", "model_2"]
    assert table.loc["model_1", "AIC"] > table.loc["model_2", "AIC"]
    # Worse model (higher AIC) gets delta 0; better model gets the full span.
    assert table.loc["model_1", "delta"] == pytest.approx(0.0)
    assert table.loc["model_2", "delta"] > 0.0
    assert table.loc["model_1", "scaled"] == pytest.approx(0.0)
    assert table.loc["model_2", "scaled"] == pytest.approx(1.0)
    assert model_GAIC_lollipop(small, full).kind == "model_GAIC_lollipop"


def test_model_td_matches_get_tgd(models):
    small, full, data = models
    table = model_TD(small, full, newdata=data, plot=False)
    expected = [
        float(get_tgd(model, newdata=data).test_global_deviance)
        for model in (small, full)
    ]
    np.testing.assert_allclose(table["TD"], expected, rtol=0, atol=0)
    assert table.loc["model_1", "delta"] == pytest.approx(0.0)
    assert model_TD_lollipop(small, full, newdata=data).kind == "model_TD_lollipop"
    with pytest.raises(ValueError, match="more than one model"):
        model_TD(small, newdata=data)
    with pytest.raises(ValueError, match="newdata"):
        model_TD(small, full)


def test_model_centiles_evaluates_the_fitted_quantiles(models):
    small, full, data = models
    table = model_centiles(small, full, cent=[10, 50, 90], plot=False)
    assert isinstance(table, pd.DataFrame)
    assert set(table["Y"]) == {"model_1", "model_2"}
    assert table.shape[0] == 2 * 3 * data["y"].size
    for model, label in ((small, "model_1"), (full, "model_2")):
        order = np.argsort(data["x"])
        mu = np.asarray(model.fitted_values["mu"], dtype=float)[order]
        sigma = np.asarray(model.fitted_values["sigma"], dtype=float)[order]
        for centile in (10, 50, 90):
            subset = table[(table["Y"] == label) & (table["centiles"] == str(centile))]
            subset = subset.sort_values("x")
            expected = mu + sigma * stats.norm.ppf(centile / 100.0)
            np.testing.assert_allclose(subset["c"].to_numpy(), expected, atol=1e-9)
    assert model_centiles(small, full, plot=True).kind == "model_centiles"


def test_model_wp_builds_the_worm_ordinates(models):
    small, full, _ = models
    result = model_wp(small, full, plot=False)
    data = result["data"]
    assert set(data["model"]) == {"model_1", "model_2"}
    for model, label in ((small, "model_1"), (full, "model_2")):
        residuals = np.asarray(model.residuals, dtype=float)
        expected_x = stats.norm.ppf(_ppoints(residuals.size))[
            np.argsort(np.argsort(residuals))
        ]
        subset = data[data["model"] == label]
        np.testing.assert_allclose(subset["x"].to_numpy(), expected_x, atol=1e-12)
        np.testing.assert_allclose(
            subset["rqres"].to_numpy(), residuals - expected_x, atol=1e-12
        )
    band = result["band"]
    assert np.all(band["low"] <= 0.0) and np.all(band["high"] >= 0.0)
    assert model_wp(small, full).kind == "model_wp"
    with pytest.raises(ValueError, match="more than two models"):
        model_wp(small)


def test_model_wp_wrap_facets_by_covariate_interval(models):
    small, full, data = models
    result = model_wp_wrap(small, full, xvar="x", n_inter=4, plot=False)
    data_frame = result["data"]
    assert data_frame["Z"].nunique() == 4
    assert data_frame.shape[0] == 2 * data["y"].size
    assert model_wp_wrap(small, full, xvar="x", n_inter=2).kind == "model_wp_wrap"
    with pytest.raises(ValueError, match="xvar"):
        model_wp_wrap(small, full)


def test_model_pca_returns_scores_and_loadings(models):
    small, full, _ = models
    result = model_pca(small, full, plot=False)
    assert result["scores"].shape[1] == 2
    assert result["arrows"].shape == (2, 3)
    assert list(result["arrows"]["label"]) == ["model_1", "model_2"]
    # The first component explains at least as much as the second.
    assert result["sdev"][0] >= result["sdev"][1]
    assert model_pca(small, full).kind == "model_pca"
    with pytest.raises(ValueError, match="more than two models"):
        model_pca(small)


def test_model_deviance_incr_diff_matches_direct_increments(models):
    small, full, data = models
    table = model_devianceIncr_diff(small, full, data=data, plot=False)
    expected = (
        np.asarray(devianceIncr(small), dtype=float)
        - np.asarray(devianceIncr(full), dtype=float)
    )
    np.testing.assert_allclose(table["diff"].to_numpy(), expected, atol=1e-12)
    assert list(table["index"]) == list(range(1, data["y"].size + 1))
    assert set(table["col"]) == {"normal"}
    assert table["txt"].isna().all()

    strict = model_devianceIncr_diff(small, full, data=data, tol=0.0, plot=False)
    assert set(strict["col"]) == {"outlier"}
    assert strict["txt"].notna().all()
    assert model_devianceIncr_diff(small, full, data=data).kind == "model_devianceIncr_diff"


def test_model_mom_bucket_is_the_moment_bucket_alias(models):
    small, _, _ = models
    result = model_mom_bucket(small, no_bootstrap=0)
    assert isinstance(result, GGPlot)
    assert result.kind == "model_mom_bucket"
    assert len(result.data["rows"]) == 1
