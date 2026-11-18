# SPDX-License-Identifier: GPL-3.0-or-later
"""gamlss.ggplots bootstrap coefficient plots (D2.262): boot_coef / boot_coef_one.

R names the bootstrap columns names(unlist(coef(model))), i.e.
"mu.(Intercept)", "mu.x", "sigma.x"; the port keeps those names on BootResult.
The tests also pin the refit fix: a model with sigma_formula="~ x" must keep
four coefficients in every replicate (before the fix the sigma model was
silently dropped and boot had fewer columns than orig_coef).
"""
from __future__ import annotations

import numpy as np
import pytest

from omnilss import (
    BayesianBoot,
    GGPlot,
    NonParametricBoot,
    boot_coef,
    boot_coef_one,
    gamlss,
)


@pytest.fixture(scope="module")
def bootstrap():
    rng = np.random.default_rng(1)
    n = 80
    x = rng.normal(size=n)
    y = 1.0 + 0.5 * x + rng.normal(scale=0.3 + 0.1 * np.abs(x), size=n)
    model = gamlss(
        "y ~ x", sigma_formula="~ x", family="NO",
        data={"y": y, "x": x}, max_iter=40,
    )
    return NonParametricBoot(model, B=12, seed=1)


def test_names_and_shapes_stay_consistent(bootstrap):
    assert bootstrap.boot.shape[1] == bootstrap.orig_coef.size
    assert bootstrap.coef_names == (
        "mu.(Intercept)", "mu.x", "sigma.(Intercept)", "sigma.x"
    )


def test_boot_coef_one_reports_values_and_limits(bootstrap):
    result = boot_coef_one(bootstrap, par=2, plot=False)
    assert result["name"] == "mu.x"
    np.testing.assert_allclose(result["values"], bootstrap.boot[:, 1], rtol=0, atol=0)
    assert result["orig"] == pytest.approx(float(bootstrap.orig_coef[1]))
    np.testing.assert_allclose(
        result["quantiles"], np.quantile(bootstrap.boot[:, 1], [0.025, 0.5, 0.975]),
        rtol=0, atol=0,
    )
    assert result["title"] == "mu.x"
    assert boot_coef_one(bootstrap, par=1, title="custom").data["title"] == "custom"
    assert boot_coef_one(bootstrap, par=1).figure is not None


def test_boot_coef_returns_one_chart_per_coefficient(bootstrap):
    charts = boot_coef(bootstrap, plot=False)
    assert set(charts) == set(bootstrap.coef_names)
    assert all(isinstance(chart, GGPlot) for chart in charts.values())
    assert all(chart.kind == "boot_coef" for chart in charts.values())
    np.testing.assert_allclose(
        charts["sigma.x"].data["values"], bootstrap.boot[:, 3], rtol=0, atol=0
    )


def test_boot_coef_selects_terms_by_name_or_position(bootstrap):
    by_name = boot_coef(bootstrap, terms=["sigma.x"], plot=False)
    assert list(by_name) == ["sigma.x"]

    by_index = boot_coef(bootstrap, terms=[1, 4], plot=False)
    assert list(by_index) == ["mu.(Intercept)", "sigma.x"]

    single = boot_coef(bootstrap, terms="mu.x", plot=False)
    assert list(single) == ["mu.x"]

    with pytest.raises(ValueError, match="not a bootstrap coefficient"):
        boot_coef(bootstrap, terms=["nope"], plot=False)
    with pytest.raises(ValueError, match="outside the columns"):
        boot_coef(bootstrap, terms=[9], plot=False)


def test_boot_coef_works_for_the_bayesian_bootstrap():
    rng = np.random.default_rng(3)
    n = 60
    x = rng.normal(size=n)
    y = 1.0 + 0.4 * x + rng.normal(scale=0.3, size=n)
    model = gamlss("y ~ x", family="NO", data={"y": y, "x": x}, max_iter=30)
    result = BayesianBoot(model, B=6, seed=2)
    charts = boot_coef(result, plot=False)
    assert list(charts) == list(result.coef_names)
    assert charts["mu.x"].data["values"].size == result.trueB


def test_bootstrap_plots_validate_their_inputs(bootstrap):
    with pytest.raises(ValueError, match="NonParametric.Boot"):
        boot_coef("not a bootstrap")
    with pytest.raises(ValueError, match="NonParametric.Boot"):
        boot_coef_one({"boot": np.zeros((2, 2))})
    with pytest.raises(ValueError, match="outside the bootstrap"):
        boot_coef_one(bootstrap, par=9)
    with pytest.raises(ValueError, match="outside the bootstrap"):
        boot_coef_one(bootstrap, par=0)
