# SPDX-License-Identifier: GPL-3.0-or-later
"""gamlss.ggplots pe_* partial effects (D2.256).

The contract is the effect curve's data: varying one covariate over its range
with the others held at the scenario value, centring at the mean, and reporting
the parameter (or its linear predictor).  On a NO fit the effect of x is exactly
mu's x coefficient times (x - mean(x)).
"""
from __future__ import annotations

import numpy as np
import pytest

from omnilss import (
    GGPlot,
    gamlss,
    pe_1_parameter,
    pe_2_parameter,
    pe_eta,
    pe_eta_grid,
    pe_param,
    pe_param_grid,
    pe_pdf,
    pe_pdf_grid,
    pe_terms,
)


@pytest.fixture(scope="module")
def model():
    rng = np.random.default_rng(2)
    n = 120
    x = rng.normal(size=n)
    z = rng.uniform(-1.0, 1.0, size=n)
    y = 1.0 + 0.5 * x + 0.3 * z + rng.normal(scale=0.4, size=n)
    return gamlss("y ~ x + z", family="NO", data={"y": y, "x": x, "z": z}, max_iter=40), x, z


def test_pe_param_matches_the_model_coefficient(model):
    fitted, x, _ = model
    plot = pe_param(fitted, "x")
    assert isinstance(plot, GGPlot)
    grid = np.asarray(plot.data["x"])
    effect = np.asarray(plot.data["effect"])
    center = int(np.argmin(np.abs(grid - grid.mean())))
    assert effect[center] == pytest.approx(0.0, abs=1e-10)
    slope = float(np.polyfit(grid, effect, 1)[0])
    assert slope == pytest.approx(float(np.ravel(fitted.coefficients["mu"])[1]), rel=1e-6, abs=1e-8)
    assert plot.data["term"] == "x" and plot.data["parameter"] == "mu"


def test_pe_eta_equals_pe_param_under_the_identity_link(model):
    fitted, _, _ = model
    by_parameter = pe_param(fitted, "x")
    by_eta = pe_eta(fitted, "x")
    np.testing.assert_allclose(by_parameter.data["effect"], by_eta.data["effect"], atol=1e-12)
    assert by_eta.data["type"] == "eta"


def test_pe_eta_applies_the_link_for_sigma(model):
    fitted, x, z = model
    # An intercept-only sigma does not vary with x: both curves are flat.
    flat = pe_eta(fitted, "x", parameter="sigma")
    np.testing.assert_allclose(np.asarray(flat.data["effect"]), 0.0, atol=1e-10)

    rng = np.random.default_rng(9)
    y = 1.0 + 0.5 * x + 0.3 * z + rng.normal(scale=np.exp(0.2 + 0.6 * x) * 0.2, size=x.size)
    varying = gamlss(
        "y ~ x + z", sigma_formula="~ x",
        family="NO", data={"y": y, "x": x, "z": z}, max_iter=60,
    )
    parameter = pe_param(varying, "x", parameter="sigma")
    eta = pe_eta(varying, "x", parameter="sigma")
    raw = np.asarray(parameter.data["raw"])
    expected = np.log(raw)
    expected = expected - expected[int(np.argmin(np.abs(np.asarray(parameter.data["x"]) - np.asarray(parameter.data["x"]).mean())))]
    np.testing.assert_allclose(np.asarray(eta.data["effect"]), expected, atol=1e-8)


def test_pe_two_and_grids(model):
    fitted, _, _ = model
    two = pe_2_parameter(fitted, ["x", "z"])
    assert set(two) == {"x", "z"}
    assert set(pe_param_grid(fitted, ["x", "z"])) == {"x", "z"}
    assert set(pe_eta_grid(fitted, ["x", "z"])) == {"x", "z"}
    assert pe_1_parameter(fitted, "z").data["term"] == "z"


def test_pe_pdf_and_grid(model):
    fitted, _, _ = model
    plot = pe_pdf(fitted, "x", points=3)
    assert len(plot.data["density"]) == 3
    for curve in plot.data["density"].values():
        assert np.all(np.asarray(curve) >= 0.0)
        assert np.all(np.isfinite(np.asarray(curve)))
    assert set(pe_pdf_grid(fitted, ["x", "z"])) == {"x", "z"}


def test_pe_terms_defaults_to_the_formula_covariates(model):
    fitted, _, _ = model
    terms = pe_terms(fitted)
    assert set(terms) == {"x", "z"}


def test_scenario_changes_the_curve(model):
    fitted, _, _ = model
    default = pe_param(fitted, "x")
    shifted = pe_param(fitted, "x", scenario={"z": 0.9})
    assert not np.allclose(default.data["raw"], shifted.data["raw"])
    # Centred curves agree: only the intercept moves with the scenario.
    np.testing.assert_allclose(default.data["effect"], shifted.data["effect"], atol=1e-8)


def test_pe_validation(model):
    fitted, _, _ = model
    with pytest.raises(ValueError, match="gamlss model"):
        pe_param("not a model", "x")
    with pytest.raises(ValueError, match="term"):
        pe_param(fitted, None)
    with pytest.raises(ValueError, match="not in the data"):
        pe_param(fitted, "w")
    with pytest.raises(ValueError, match="parameter"):
        pe_param(fitted, "x", parameter="tau")
    with pytest.raises(ValueError, match="type"):
        pe_1_parameter(fitted, "x", type="raw")
    with pytest.raises(ValueError, match="terms"):
        pe_2_parameter(fitted, None)
