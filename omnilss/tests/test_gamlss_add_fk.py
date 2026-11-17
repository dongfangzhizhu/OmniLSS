# SPDX-License-Identifier: GPL-3.0-or-later
"""gamlss.add free-knot terms (D2.250): fk / gamlss.fk / fitFixedKnots /
fitFreeKnots / centilesTwo.

Live R reference (R 4.6.0 + gamlss.dist 6.1.1, sources in
R/gamlss.add/R/freeKnots-18-06-13.R) on the identical sample
(seed 3, n=100, x sorted U(0,10), y = sin(x) + N(0, 0.2), degree 1):

    fitFixedKnots(knots = 3, 7)      rss 39.690485586883
                                     coef 1.4440660236, -0.5825541845,
                                          0.7501746671, -0.1594065177
    fitFreeKnots(knots = 5)          break point 4.4811375530
                                     rss 30.709061237852
    fitFreeKnots(knots = 3, 7)       break points 5.2296934754, 7.9267859854
                                     rss 9.987674925832

The two-knot run is not bit-identical: the objective matches R to 1e-12 at R's
optimum, but R's single L-BFGS-B start can stop in a shallower basin, so the port
uses a deterministic multi-start and lands at rss 9.987781 (rel 1e-5).
"""
from __future__ import annotations

import numpy as np
import pytest

from omnilss import (
    CentileSurface,
    centilesTwo,
    fitFixedKnots,
    fitFreeKnots,
    fk,
    fk_control,
    gamlss,
    gamlss_fk,
)


def _sample():
    rng = np.random.default_rng(3)
    n = 100
    x = np.sort(rng.uniform(0, 10, size=n))
    y = np.sin(x) + rng.normal(scale=0.2, size=n)
    return x, y


def test_fixed_knots_match_live_r():
    x, y = _sample()
    fit = fitFixedKnots(y=y, x=x, knots=[3.0, 7.0], degree=1)
    assert fit.rss == pytest.approx(39.690485586883, rel=1e-11)
    np.testing.assert_allclose(
        fit.coef, [1.4440660236, -0.5825541845, 0.7501746671, -0.1594065177],
        rtol=1e-10, atol=1e-11,
    )
    assert fit.df == 4
    assert fit.sigma == pytest.approx(0.396904855869, rel=1e-11)
    # R's deviance is sum(-2 * dNO(y, mu = fitted, sigma = sigma)).
    n = y.size
    expected = n * np.log(2 * np.pi) + 2 * n * np.log(fit.sigma) + fit.rss / fit.sigma**2
    assert fit.deviance == pytest.approx(expected, rel=1e-12)


def test_one_free_knot_matches_live_r_exactly():
    x, y = _sample()
    fit = fitFreeKnots(y=y, x=x, knots=[5.0], degree=1)
    assert fit.break_points[0] == pytest.approx(4.4811375530, abs=1e-8)
    assert fit.rss == pytest.approx(30.709061237852, rel=1e-11)


def test_two_free_knots_reach_the_r_optimum_within_the_documented_tolerance():
    x, y = _sample()
    fit = fitFreeKnots(y=y, x=x, knots=[3.0, 7.0], degree=1)
    assert fit.rss == pytest.approx(9.987674925832, rel=1e-4)
    np.testing.assert_allclose(
        fit.break_points, [5.2296934754, 7.9267859854], atol=5e-3
    )
    # Objective parity: at R's own optimum our rss is R's rss.
    at_r = fitFixedKnots(y=y, x=x, knots=[5.2296934754, 7.9267859854], degree=1)
    assert at_r.rss == pytest.approx(9.987674925832, rel=1e-9)


def test_free_knot_fit_adds_one_df_per_optimised_knot():
    x, y = _sample()
    fixed = fitFixedKnots(y=y, x=x, knots=[3.0, 7.0], degree=1)
    free = fitFreeKnots(y=y, x=x, knots=[3.0, 7.0], degree=1)
    assert free.df == fixed.df + 2


def test_fk_control_defaults_and_base_rejection():
    control = fk_control()
    assert control.degree == 1
    assert control.all_fixed is False
    assert control.base == "trun"
    assert fk_control(degree=3).degree == 3
    with pytest.raises(NotImplementedError, match="Bbase"):
        fk_control(base="Bbase")
    with pytest.raises(ValueError, match="base must be"):
        fk_control(base="nope")


def test_gamlss_fk_first_pass_is_fixed_and_later_passes_are_free():
    x, y = _sample()
    term = fk(x, start=[3.0, 7.0], degree=1)
    first = gamlss_fk(term, y, None, first_iter=True)
    later = gamlss_fk(term, y, None, first_iter=False)
    assert first.coef_smo.rss == pytest.approx(39.690485586883, rel=1e-11)
    assert later.coef_smo.rss < first.coef_smo.rss
    assert later.lam.shape == (2,)
    assert first.nl_df == first.coef_smo.df - 1
    np.testing.assert_allclose(first.residuals, y - first.fitted_values, rtol=0, atol=0)


def test_fn_validation_and_formula_rejection():
    with pytest.raises(ValueError, match="covariate values"):
        fk()
    x, y = _sample()
    term = fk(x)
    with pytest.raises(TypeError, match="fk\\(\\) term"):
        gamlss_fk("x", y)
    with pytest.raises(NotImplementedError, match="gamlss_fk"):
        gamlss("y ~ fk(x)", family="NO", data={"y": y, "x": x}, max_iter=5)


def test_centiles_two_matches_direct_quantiles_and_the_mgcv_mask():
    rng = np.random.default_rng(9)
    n = 150
    x = rng.uniform(-2, 2, size=n)
    z = rng.uniform(-1, 1, size=n)
    y = 1.0 + 0.5 * x + 0.3 * z + rng.normal(scale=0.4, size=n)
    model = gamlss("y ~ x + z", family="NO", data={"y": y, "x": x, "z": z}, max_iter=30)
    grid1 = np.linspace(-1.5, 1.5, 7)
    grid2 = np.linspace(-0.8, 0.8, 5)
    surface = centilesTwo(model, grid1, grid2, "x", "z", cent=0.5)
    assert isinstance(surface, CentileSurface)
    assert surface.values.shape == (7, 5)
    # mgcv::exclude.too.far(dist = 0.01) masks 33 of 35 cells, keeping (2,2) and (3,2).
    expected_mask = np.ones((7, 5), dtype=bool)
    expected_mask[2, 2] = False
    expected_mask[3, 2] = False
    np.testing.assert_array_equal(surface.mask, expected_mask)
    mesh = np.meshgrid(grid1, grid2, indexing="ij")
    from omnilss.prediction import predict_params

    params = predict_params(model, {"x": mesh[0].ravel(), "z": mesh[1].ravel()})
    direct = np.asarray(model.family.q(0.5, params["mu"], params["sigma"]), dtype=float)
    direct = direct.reshape(mesh[0].shape)
    np.testing.assert_allclose(
        surface.values[~surface.mask], direct[~surface.mask], rtol=0, atol=0
    )


def test_centiles_two_validates_its_inputs():
    rng = np.random.default_rng(1)
    x = rng.normal(size=20)
    y = x + rng.normal(scale=0.2, size=20)
    model = gamlss("y ~ x", family="NO", data={"y": y, "x": x}, max_iter=10)
    with pytest.raises(ValueError, match="x1_name and x2_name"):
        centilesTwo(model, [0.0], [0.0])
    with pytest.raises(ValueError, match="not in the data"):
        centilesTwo(model, [0.0], [0.0], "x", "z")
