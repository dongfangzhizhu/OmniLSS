# SPDX-License-Identifier: GPL-3.0-or-later
"""gamlss.ggplots parameterized diagnostics (D2.258).

Contract: the data layer of every diagnostic, plus the rules R applies:
weights == 0 observations are dropped (1-based indices preserved), outliers are
flagged by |quantile residual| or by the requested quantile threshold, worm-plot
ordinates are residual - qnorm(ppoints)[rank], and fitted_survival plots
1 - CDF when lower_tail is TRUE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from omnilss import (
    GGPlot,
    fitted_centiles,
    fitted_centiles_gap,
    fitted_centiles_legend,
    fitted_devianceIncr,
    fitted_leverage,
    fitted_survival,
    fitted_survival_data,
    fitted_terms,
    gamlss,
    resid_dtop,
    resid_median,
    resid_mu,
    resid_param,
    resid_quantile,
    resid_wp,
    resid_wp_wrap,
    resid_xvar,
    resp_mu,
    resp_param,
    resp_quantile,
)
from omnilss.DevianceIncr import devianceIncr
from omnilss.ggplots import _ppoints


@pytest.fixture(scope="module")
def model():
    rng = np.random.default_rng(3)
    n = 120
    x = rng.normal(size=n)
    y = 1.0 + 0.5 * x + rng.normal(scale=0.4, size=n)
    return gamlss("y ~ x", family="NO", data={"y": y, "x": x}, max_iter=40), x, y


def test_resid_mu_flags_outliers_and_keeps_indices(model):
    fitted, _, _ = model
    result = resid_mu(fitted, value=2.0)
    data = result.data["data"]
    assert list(data.columns) == ["obs", "sdres", "fv", "color", "txt"]
    assert data.shape[0] == np.asarray(fitted.residuals).size
    np.testing.assert_allclose(data["sdres"], np.asarray(fitted.residuals))
    np.testing.assert_allclose(
        data["fv"], np.asarray(fitted.fitted_values["mu"])
    )
    expected = (data["sdres"].abs() >= 2.0)
    assert (data["color"] == "outlier").to_numpy().tolist() == expected.tolist()
    assert data.loc[expected, "txt"].notna().all()
    unplotted = resid_mu(fitted, plot=False)
    assert unplotted["threshold"] == 2.0
    assert set(unplotted["outliers"].columns) == {"observation", "quan_resid"}


def test_resid_diagnostics_drop_zero_weights():
    rng = np.random.default_rng(11)
    x = rng.normal(size=60)
    y = 1.0 + 0.4 * x + rng.normal(scale=0.3, size=60)
    weights = np.ones(60)
    weights[[0, 5]] = 0.0
    fitted = gamlss("y ~ x", family="NO", data={"y": y, "x": x}, weights=weights, max_iter=20)
    data = resid_mu(fitted, plot=False)["data"]
    assert 1 not in set(data["obs"]) and 6 not in set(data["obs"])
    assert data.shape[0] == 58


def test_resid_median_uses_the_fitted_median(model):
    fitted, _, _ = model
    data = resid_median(fitted).data["data"]
    expected = np.asarray(fitted.fitted_values["mu"]) + 0.0 * np.asarray(
        fitted.fitted_values["sigma"]
    )
    np.testing.assert_allclose(data["fv"], expected, atol=1e-12)


def test_resid_param_and_quantile_report_the_correlation(model):
    fitted, _, _ = model
    by_param = resid_param(fitted, "mu")
    data = by_param.data["data"]
    assert list(data.columns) == ["obs", "resid", "fv"]
    expected = float(np.corrcoef(data["resid"], data["fv"])[0, 1])
    assert by_param.data["correlation"] == pytest.approx(expected)

    by_quantile = resid_quantile(fitted, 0.5)
    frame = by_quantile.data["data"]
    assert list(frame.columns) == ["obs", "y", "fv"]
    assert by_quantile.data["correlation"] == pytest.approx(
        float(np.corrcoef(frame["y"], frame["fv"])[0, 1])
    )
    with pytest.raises(ValueError, match="parameter"):
        resid_param(fitted, "tau")


def test_resid_xvar_handles_numeric_and_factor(model):
    fitted, x, _ = model
    numeric = resid_xvar(fitted, "x", plot=False)
    assert numeric["xname"] == "x"
    assert numeric["is_factor"] is False
    np.testing.assert_allclose(numeric["data"]["x"], x)

    groups = np.where(x > 0, "pos", "neg")
    categorical = resid_xvar(fitted, groups, plot=False)
    assert categorical["xname"] == "xvar"
    assert categorical["is_factor"] is True
    assert set(categorical["data"]["xvar"]) == {"pos", "neg"}

    with pytest.raises(ValueError, match="xvar"):
        resid_xvar(fitted)


def test_resid_wp_matches_the_worm_ordinates(model):
    fitted, _, _ = model
    result = resid_wp(fitted)
    data = result.data["data"]
    residuals = np.asarray(fitted.residuals)
    expected_x = stats.norm.ppf(_ppoints(residuals.size))[np.argsort(np.argsort(residuals))]
    np.testing.assert_allclose(data["x"], expected_x, atol=1e-12)
    np.testing.assert_allclose(data["rqres"], residuals - expected_x, atol=1e-12)
    np.testing.assert_allclose(
        data["fv"].to_numpy(), np.polyval(np.polyfit(expected_x, residuals - expected_x, 3), expected_x),
        atol=1e-10,
    )
    assert set(result.data["band"].columns) == {"high", "low", "z"}

    from_residuals = resid_wp(resid=residuals)
    np.testing.assert_allclose(
        from_residuals.data["data"]["rqres"], data["rqres"], atol=1e-12
    )
    with pytest.raises(ValueError, match="fitted object or the argument resid"):
        resid_wp()


def test_resid_wp_wrap_facets_the_covariate(model):
    fitted, _, _ = model
    result = resid_wp_wrap(fitted, xvar="x", n_inter=3)
    assert result.data["n_inter"] == 3
    assert result.data["data"]["Z"].nunique() == 3
    assert set(result.data["band"]["Z"]) == set(result.data["data"]["Z"])
    with pytest.raises(ValueError, match="xvar"):
        resid_wp_wrap(fitted)


def test_resid_dtop_detrends_the_empirical_cdf(model):
    fitted, _, _ = model
    result = resid_dtop(fitted, type="Owen", conf_level="95")
    data = result.data["data"]
    residuals = np.asarray(fitted.residuals)
    ordered = np.sort(residuals)
    mm = np.searchsorted(ordered, residuals, side="right") / residuals.size
    expected = stats.norm.ppf(np.clip(mm, 1e-300, 1 - 1e-16)) - residuals
    order = np.argsort(residuals)
    np.testing.assert_allclose(data["dzscores"], expected[order], atol=1e-12)
    band = result.data["band"]
    assert np.all(band["lower"] <= band["upper"])
    assert band.shape[0] == np.unique(residuals).size

    jw = resid_dtop(fitted, type="JW", conf_level="99")
    assert jw.data["type"] == "JW" and jw.data["conf_level"] == "99"
    with pytest.raises(ValueError, match="type"):
        resid_dtop(fitted, type="nope")
    with pytest.raises(ValueError, match="conf.level"):
        resid_dtop(fitted, conf_level="90")


def test_resp_plots_compare_response_and_fitted_values(model):
    fitted, _, _ = model
    mu_plot = resp_mu(fitted)
    frame = mu_plot.data["data"]
    assert list(frame.columns) == ["obs", "y", "fv"]
    np.testing.assert_allclose(frame["y"], np.asarray(fitted.y))
    assert mu_plot.data["correlation"] == pytest.approx(
        float(np.corrcoef(frame["y"], frame["fv"])[0, 1])
    )
    assert resp_param(fitted, "sigma").data["data"].shape[0] == frame.shape[0]
    assert resp_quantile(fitted, 0.9).data["quantile"] == 0.9
    with pytest.raises(ValueError, match="parameter"):
        resp_param(fitted, "tau")


def test_fitted_leverage_sums_to_the_design_rank(model):
    fitted, _, _ = model
    result = fitted_leverage(fitted, quan_val=0.99)
    leverage = result.data["leverage"]
    assert leverage.shape == (np.asarray(fitted.residuals).size,)
    assert float(np.sum(leverage)) == pytest.approx(2.0, abs=1e-6)
    assert result.data["threshold"] == pytest.approx(float(np.quantile(leverage, 0.99)))
    assert result.data["rn2"] == pytest.approx(4.0 / leverage.size)


def test_fitted_deviance_increment_matches_the_model(model):
    fitted, _, _ = model
    result = fitted_devianceIncr(fitted, quan_val=0.99)
    expected = np.asarray(devianceIncr(fitted), dtype=float)
    np.testing.assert_allclose(result.data["data"]["DIncr"], expected, atol=1e-12)
    assert result.data["threshold"] == pytest.approx(float(np.quantile(expected, 0.99)))


def test_fitted_survival_is_one_minus_the_cdf(model):
    fitted, _, _ = model
    result = fitted_survival(fitted, obs=[1, 2], from_=-2.0, to=4.0, no_points=101, plot=False)
    grid = result["grid"]
    assert grid.size == 101
    assert set(result["survival"]) == {"1", "2"}
    for key, index in (("1", 0), ("2", 1)):
        args = [
            np.full(grid.size, float(np.asarray(fitted.fitted_values[name])[index]))
            for name in fitted.family.parameters
        ]
        expected = 1.0 - np.asarray(fitted.family.p(grid, *args))
        np.testing.assert_allclose(result["survival"][key], expected, atol=1e-12)
        assert np.all(np.diff(result["survival"][key]) <= 1e-12)

    survival_data = fitted_survival_data(fitted, obs=[1], from_=-2.0, to=4.0, plot=False)
    assert survival_data["observed"][0] == pytest.approx(float(fitted.y[0]))


def test_fitted_centiles_and_variants(model):
    fitted, x, y = model
    table = fitted_centiles(fitted, cent=[10, 50, 90], plot=False)
    assert table.shape[0] == 3 * x.size
    assert set(table["centiles"]) == {"10", "50", "90"}
    order = np.argsort(x)
    mu = np.asarray(fitted.fitted_values["mu"])[order]
    sigma = np.asarray(fitted.fitted_values["sigma"])[order]
    subset = table[table["centiles"] == "50"].sort_values("x")
    np.testing.assert_allclose(subset["c"], mu, atol=1e-9)
    np.testing.assert_allclose(subset["y"], y[order], atol=0)

    legend = fitted_centiles_legend(fitted, cent=[50], save_data=True)
    assert isinstance(legend, pd.DataFrame) and legend.shape[0] == x.size

    gap = fitted_centiles_gap(fitted, cent=[50], plot=False)
    expected_grid = np.linspace(x.min(), x.max(), x.size)
    np.testing.assert_allclose(gap.sort_values("x")["x"], expected_grid, atol=1e-9)
    assert fitted_centiles_gap(fitted, cent=[50]).kind == "fitted_centiles_gap"


def test_fitted_terms_delegates_to_the_partial_effects(model):
    fitted, _, _ = model
    result = fitted_terms(fitted)
    assert set(result) == {"x"}
    chart = result["x"]
    assert isinstance(chart, GGPlot) and chart.kind == "fitted_terms"
    slope = float(np.polyfit(chart.data["x"], chart.data["effect"], 1)[0])
    assert slope == pytest.approx(float(np.ravel(fitted.coefficients["mu"])[1]), rel=1e-6)


def test_validation_errors(model):
    fitted, _, _ = model
    with pytest.raises(ValueError, match="gamlss model"):
        resid_mu("not a model")
    with pytest.raises(ValueError, match="fitted object"):
        resid_mu()
    with pytest.raises(ValueError, match="gamlss model"):
        fitted_leverage("not a model")
    with pytest.raises(ValueError, match="gamlss model"):
        fitted_devianceIncr()
    with pytest.raises(ValueError, match="1-based"):
        fitted_survival(fitted, obs=[0], plot=False)
