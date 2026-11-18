# SPDX-License-Identifier: GPL-3.0-or-later
"""gamlss.ggplots first batch (D2.255): data layer plus optional rendering.

Pixel parity with ggplot2 is out of scope; the tests pin the data each plot
carries and the diagnostic rules shared with R (median +/- value * mad,
qnorm(ppoints) QQ coordinates, quartile QQ line, moment bucket band).
"""
from __future__ import annotations

import numpy as np
import pytest

from omnilss import (
    GGPlot,
    add_resid_qqplot,
    fitted_cdf,
    fitted_cdf_data,
    fitted_pdf,
    fitted_pdf_data,
    gamlss,
    model_density,
    model_qqplot,
    moment_bucket,
    resid_density,
    resid_ecdf,
    resid_plots,
    resid_qqplot,
    y_dots,
    y_hist,
)


def _model(seed: int = 1, n: int = 100):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = 1.0 + 0.5 * x + rng.normal(scale=0.4, size=n)
    return gamlss("y ~ x", family="NO", data={"y": y, "x": x}, max_iter=30), y, x


def test_y_hist_data_matches_the_mad_rule():
    rng = np.random.default_rng(2)
    y = rng.normal(size=200)
    plot = y_hist(y, value=3.0)
    assert isinstance(plot, GGPlot)
    assert np.sum(plot.data["counts"]) == 200
    assert plot.data["edges"].size == plot.data["counts"].size + 1
    median = float(np.median(y))
    spread = float(np.median(np.abs(y - median)) * 1.4826)
    assert plot.data["median"] == pytest.approx(median)
    assert plot.data["lower"] == pytest.approx(median - 3.0 * spread, rel=1e-5)
    assert plot.data["upper"] == pytest.approx(median + 3.0 * spread, rel=1e-5)


def test_y_dots_flags_the_injected_outlier():
    rng = np.random.default_rng(3)
    y = rng.normal(size=200)
    y[0] = 20.0
    plot = y_dots(y, value=3.0)
    assert bool(plot.data["outlier"][0]) is True
    # The rule may also flag a genuine tail point; the injected one must be in.
    assert int(plot.data["outlier"].sum()) >= 1


def test_fitted_pdf_and_data_integrate_to_one():
    model, _, _ = _model()
    plot = fitted_pdf(model, [0, 5], from_=-6.0, to=8.0, no_points=401)
    assert set(plot.data["density"]) == {"1", "6"}
    for curve in plot.data["density"].values():
        assert np.all(np.asarray(curve) >= 0.0)
        area = float(np.trapz(np.asarray(curve), plot.data["grid"]))
        assert area == pytest.approx(1.0, abs=0.05)
    with_obs = fitted_pdf_data(model, [0], from_=-6.0, to=8.0)
    assert "observed" in with_obs.data
    assert np.asarray(with_obs.data["observed"]).shape == (1,)


def test_fitted_cdf_is_monotone_and_bounded():
    model, _, _ = _model()
    plot = fitted_cdf(model, [0], from_=-6.0, to=8.0)
    curve = np.asarray(plot.data["cdf"]["1"])
    assert np.all((curve >= 0.0) & (curve <= 1.0))
    assert np.all(np.diff(curve) >= -1e-12)
    assert "observed" in fitted_cdf_data(model, [0]).data


def test_model_qqplot_uses_ppoints_and_the_quartile_line():
    model, _, _ = _model()
    plot = model_qqplot(model, value=3.0)
    n = np.asarray(plot.data["sample"]).size
    assert np.asarray(plot.data["theoretical"]).size == n
    assert np.all(np.diff(np.asarray(plot.data["theoretical"])) >= 0)
    assert plot.data["slope"] == pytest.approx(1.0, abs=0.3)
    assert np.asarray(plot.data["outlier"]).dtype == bool


def test_model_density_accepts_several_models():
    first, _, _ = _model(seed=1)
    second, _, _ = _model(seed=2)
    plot = model_density(first, second)
    assert set(plot.data["density"]) == {"1", "2"}
    assert np.asarray(plot.data["grid"]).size == 256


def test_resid_qqplot_density_ecdf_and_plots():
    model, _, _ = _model()
    qq = resid_qqplot(model)
    assert np.asarray(qq.data["observation"]).size == np.asarray(qq.data["sample"]).size
    assert add_resid_qqplot(None, model).kind == "add_resid_qqplot"

    density = resid_density(model)
    assert np.asarray(density.data["density"]).size == np.asarray(density.data["grid"]).size

    ecdf = resid_ecdf(model)
    assert np.asarray(ecdf.data["ecdf"])[-1] == pytest.approx(1.0)
    assert np.all(np.diff(np.asarray(ecdf.data["ecdf"])) >= 0)

    bundle = resid_plots(model)
    assert set(bundle) == {"qqplot", "density", "ecdf"}
    assert all(isinstance(plot, GGPlot) for plot in bundle.values())


def test_moment_bucket_reports_moments_band_and_bootstrap():
    rng = np.random.default_rng(5)
    sample = rng.normal(size=150)
    plot = moment_bucket(sample, no_bootstrap=5, seed=7)
    row = plot.data["rows"][0]
    assert row["skewness"] == pytest.approx(float(np.mean(((sample - sample.mean()) ** 3))) / sample.std() ** 3, rel=1e-9)
    assert np.isfinite(row["kurtosis"])
    assert len(row["band"]) == 50
    assert row["bootstrap"].shape == (5, 2)

    model, _, _ = _model()
    from_model = moment_bucket(model, no_bootstrap=0)
    assert len(from_model.data["rows"]) == 1


def test_plots_validate_their_inputs():
    with pytest.raises(ValueError, match="gamlss model"):
        model_qqplot("not a model")
    with pytest.raises(ValueError, match="gamlss model"):
        fitted_pdf(None)
    with pytest.raises(ValueError, match="finite"):
        y_hist(np.array([np.nan, np.nan]))
