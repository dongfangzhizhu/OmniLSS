"""Dedicated evidence module for gamlss.ggplots (P4-6, D2.255-D2.263).

The coverage layer (109/109 public_api + tested) is checked first, then the four
families the port groups the package into:

1. model diagnostics and prediction curves;
2. model comparison, worm plots, centiles and bootstrap coefficients;
3. bucket plots (moment / centile) and bivariate surfaces;
4. the standalone numerical helpers (fit_PB, PCR paths, smoothed histogram).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

import omnilss
from omnilss import (
    GGPlot,
    PSplineFit,
    bivar_fun,
    bivar_pdf,
    boot_coef,
    boot_coef_one,
    centile_bucket,
    centile_colour_half,
    fit_PB,
    fitted_leverage,
    fitted_survival,
    gamlss,
    histSmo,
    histSmo_plot,
    model_centiles,
    model_GAIC,
    model_TD,
    model_wp,
    moment_bucket,
    moment_gray_both,
    pcr_coef_path,
    pe_param,
    predict_cdf,
    predict_pdf,
    quantile_gamlss,
    resid_plots,
    resid_qqplot,
    y_dots,
    y_hist,
)
from omnilss.pcr_series import PCRFit

_REPO = Path(__file__).resolve().parents[2]
_COVERAGE = _REPO / "docs" / "development" / "r-symbol-coverage.json"
_EXPECTED_SYMBOLS = 109


def _package_symbols() -> list[dict]:
    payload = json.loads(_COVERAGE.read_text(encoding="utf-8"))
    return [row for row in payload["symbols"] if row.get("r_package") == "gamlss.ggplots"]


@pytest.fixture(scope="module")
def models():
    rng = np.random.default_rng(3)
    n = 90
    x = rng.normal(size=n)
    z = rng.uniform(-1.0, 1.0, size=n)
    y = 1.0 + 0.5 * x + 0.3 * z + rng.normal(scale=0.4, size=n)
    data = {"y": y, "x": x, "z": z}
    small = gamlss("y ~ x", family="NO", data={"y": y, "x": x}, max_iter=30)
    full = gamlss("y ~ x + z", family="NO", data=data, max_iter=30)
    return small, full, data


def test_every_ggplots_symbol_is_exported_and_tested():
    symbols = _package_symbols()
    assert len(symbols) == _EXPECTED_SYMBOLS
    for row in symbols:
        layers = row["layers"]
        assert layers["surface"] == "public_api", row["r_symbol"]
        assert layers["kernel"] == "implemented", row["r_symbol"]
        assert layers["parity"] in {"tested", "r_consistency_test"}, row["r_symbol"]
        assert callable(getattr(omnilss, row["r_symbol"].replace(".", "_"))), row["r_symbol"]


def test_diagnostics_and_prediction_workflow(models):
    small, full, data = models
    hist = y_hist(data["y"])
    assert np.sum(hist.data["counts"]) == data["y"].size
    assert 0 <= int(y_dots(data["y"]).data["outlier"].sum()) <= 3

    qq = resid_qqplot(full)
    assert qq.data["slope"] == pytest.approx(1.0, abs=0.35)
    assert set(resid_plots(full)) == {"qqplot", "density", "ecdf"}

    leverage = fitted_leverage(full, plot=False)
    assert float(np.sum(leverage["leverage"])) == pytest.approx(3.0, abs=1e-6)
    survival = fitted_survival(full, obs=[1], from_=-2.0, to=4.0, plot=False)
    assert set(survival["survival"]) == {"1"}

    effect = pe_param(full, "x")
    slope = float(np.polyfit(effect.data["x"], effect.data["effect"], 1)[0])
    assert slope == pytest.approx(float(np.ravel(full.coefficients["mu"])[1]), rel=1e-6)

    newdata = {"x": np.array([-1.0, 0.0, 1.0]), "z": np.array([0.0, 0.0, 0.0])}
    pdf = predict_pdf(full, newdata, from_=-3.0, to=5.0, plot=False)
    area = float(np.trapz(pdf["density"]["1"], pdf["grid"]))
    assert area == pytest.approx(1.0, abs=0.02)
    cdf = predict_cdf(full, newdata, from_=-3.0, to=5.0, plot=False)
    assert cdf["cdf"]["1"][-1] == pytest.approx(1.0, abs=1e-6)

    quantiles = quantile_gamlss(small, 0.9, {"x": newdata["x"]})
    from omnilss.prediction import predict_params

    params = predict_params(small, {"x": newdata["x"]})
    expected = params["mu"] + params["sigma"] * stats.norm.ppf(0.9)
    np.testing.assert_allclose(quantiles, expected, atol=1e-12)


def test_comparison_worm_centiles_and_bootstrap_workflow(models):
    small, full, data = models
    assert isinstance(model_GAIC(small), float)
    table = model_GAIC(small, full, plot=False)
    assert list(table.columns) == ["df", "AIC", "delta", "scaled"]
    assert table.loc["model_1", "delta"] == pytest.approx(0.0)

    td = model_TD(small, full, newdata=data, plot=False)
    assert np.all(np.isfinite(td["TD"].to_numpy()))

    worm = model_wp(small, full, plot=False)
    assert set(worm["data"]["model"]) == {"model_1", "model_2"}

    centiles = model_centiles(small, full, cent=[10, 50, 90], plot=False)
    assert centiles.shape[0] == 2 * 3 * data["y"].size

    from omnilss import NonParametricBoot

    bootstrap = NonParametricBoot(small, B=5, seed=1)
    one = boot_coef_one(bootstrap, par=2, plot=False)
    assert one["values"].size == bootstrap.trueB
    charts = boot_coef(bootstrap, plot=False)
    assert list(charts) == list(bootstrap.coef_names)


def test_bucket_and_bivariate_workflow(models):
    small, _, _ = models
    moments = moment_bucket(small, no_bootstrap=2, seed=1)
    row = moments.data["rows"][0]
    assert set(row["polygon"].columns) == {"X", "Y"}
    assert moment_gray_both().boundary.shape == (1000, 2)
    assert centile_colour_half().type == "tail"
    bucket = centile_bucket(
        np.asarray(small.residuals), no_bootstrap=2, seed=1
    )
    assert isinstance(bucket, GGPlot)
    assert bucket.data["points"].shape[0] == 1

    density = lambda x, y: np.exp(-(x**2 + y**2) / 2.0) / (2.0 * np.pi)
    surface = bivar_fun(density, xlim=(-2, 2), ylim=(-2, 2), n=21, plot=False)
    assert surface.z.shape == (21, 21)
    scalar = lambda x, y: float(stats.multivariate_normal.pdf([x, y], [0, 0], np.eye(2)))
    hdr = bivar_pdf(scalar, xlim=(-3, 3), ylim=(-3, 3), n=80, plot=False)
    assert hdr.mass == pytest.approx(1.0, abs=0.02)
    assert hdr.level == pytest.approx(1.0 / (4.0 * np.pi), rel=5e-3)


def test_numerical_helpers_workflow(models):
    _, _, data = models
    x = np.sort(data["x"])
    y = np.sin(x) + 0.1 * data["y"][np.argsort(data["x"])]
    pb = fit_PB(x=x, y=y, nseg=10, lambda_=5, plot=False)
    assert isinstance(pb, PSplineFit)
    assert 1.0 < pb.df < x.size
    assert np.all(np.isfinite(pb.smooth))

    matrix = np.array([[1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]])
    pcr = PCRFit(
        coefficients=matrix, scores=np.zeros((5, 3)), loadings=np.zeros((5, 3)),
        gamma=np.arange(1.0, 4.0), se_gamma=np.ones(3), center=np.zeros(5),
        scale=np.ones(5), fitted_values=np.zeros(5), xvar=np.arange(5.0),
        gaic=np.zeros(3), pc=2, k=1.0, M=3, mean_y=0.0, dim_x=(5, 3),
        sigma=np.ones(5), residuals=np.zeros(5),
    )
    path = pcr_coef_path(pcr, plot=False)
    assert path["data"]["beta"].tolist() == [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]

    payload = histSmo(data["y"], bins=10)
    histogram = histSmo_plot(payload, plot=False)
    assert histogram["bars"].shape == (10, 2)
    assert histogram["discrete"] is False
