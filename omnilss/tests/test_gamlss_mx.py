# SPDX-License-Identifier: GPL-3.0-or-later
"""gamlss.mx: gamlssMX / gamlssMXfits / gamlssNP / plotMP (D2.247).

Live R reference (R 4.6.0 + gamlss 5.5.0, R/gamlss.mx sources):

    gamlssMX(y ~ x, family="NO", K=2, seed=1, n.cyc=100)
        dev 686.74424530, prob 0.553768/0.446232,
        mu1 0.001249/0.314287, mu2 2.855662/0.426983

    gamlssNP(y ~ x, random=~1, family=NO(), K=2, EMn.cyc=60)
        dev 687.91093142, prob 0.56557878/0.43442122,
        mass points 0.03128966/2.89773759, mu coef 0.03128966/0.34488750/2.86644793,
        sigma -0.31294180, df 5, EM iterations 13

The data generator is deterministic (seed 11), so these are stable fixtures.
Branches R supports but the port does not (modelled priors, zero component, GQ
mixtures, random slopes) must raise, never approximate.
"""
from __future__ import annotations

import numpy as np
import pytest

from omnilss import gamlssMX, gamlssMXfits, gamlssNP, get_pdfMX, get_pdfNP, plotMP
from omnilss.mixtures import dmx, mx_control, np_control


def _data():
    rng = np.random.default_rng(11)
    n = 200
    x = rng.normal(size=n)
    component = (rng.random(n) < 0.4).astype(int)
    y = np.where(
        component == 1,
        3.0 + 0.5 * x + rng.normal(scale=0.8, size=n),
        0.0 + 0.3 * x + rng.normal(scale=0.6, size=n),
    )
    return {"y": y, "x": x}


def test_gamlss_mx_matches_live_r():
    fit = gamlssMX(
        "y ~ x", family="NO", K=2, data=_data(),
        control=mx_control(seed=1, n_cyc=100, trace=False, plot=False),
    )
    assert fit.deviance == pytest.approx(686.74424530, rel=1e-6)
    np.testing.assert_allclose(fit.prob, [0.553768, 0.446232], atol=1e-3)
    np.testing.assert_allclose(
        np.ravel(np.asarray(fit.models[0].coefficients["mu"], dtype=float)),
        [0.001249, 0.314287], atol=2e-3,
    )
    np.testing.assert_allclose(
        np.ravel(np.asarray(fit.models[1].coefficients["mu"], dtype=float)),
        [2.855662, 0.426983], atol=2e-3,
    )
    assert fit.df_fit == pytest.approx(7.0)  # 2 * 3 df + (K - 1)
    assert fit.aic == pytest.approx(fit.deviance + 2 * fit.df_fit)
    assert fit.post_prob.shape == (fit.N, fit.K)
    np.testing.assert_allclose(fit.post_prob.sum(axis=1), 1.0, atol=1e-12)


def test_gamlss_mx_fit_is_a_usable_mixture_object():
    fit = gamlssMX(
        "y ~ x", family="NO", K=2, data=_data(),
        control=mx_control(seed=1, n_cyc=30, trace=False, plot=False),
    )
    density = get_pdfMX(fit, 1)
    assert float(np.ravel(density(0.0))[0]) > 0.0
    assert fit.residuals.shape == (fit.N,)


def test_gamlss_mx_keeps_the_best_random_start():
    fit = gamlssMXfits(n=3, seed=7, formula="y ~ x", family="NO", K=2, data=_data())
    assert fit.extra["which"] == int(np.argmin(fit.extra["dev"]))
    assert fit.extra["dev"][fit.extra["which"]] == pytest.approx(fit.deviance)
    assert fit.deviance <= min(fit.extra["dev"]) + 1e-9


def test_gamlss_np_matches_live_r():
    fit = gamlssNP(
        "y ~ x", random="~1", family="NO", data=_data(), K=2,
        control=np_control(EMn_cyc=60, trace=False, plot_opt=0),
    )
    assert fit.deviance == pytest.approx(687.91093142, rel=1e-8)
    np.testing.assert_allclose(fit.prob, [0.56557878, 0.43442122], atol=1e-3)
    np.testing.assert_allclose(fit.mass_points, [0.03128966, 2.89773759], atol=1e-5)
    np.testing.assert_allclose(
        np.ravel(np.asarray(fit.models[0].coefficients["mu"], dtype=float)),
        [0.03128966, 0.34488750, 2.86644793], atol=1e-5,
    )
    np.testing.assert_allclose(
        np.ravel(np.asarray(fit.models[0].coefficients["sigma"], dtype=float)),
        [-0.31294180], atol=1e-5,
    )
    assert fit.df_fit == pytest.approx(5.0)
    assert fit.em_iter == 13


def test_gamlss_np_fit_is_a_usable_mixture_object():
    fit = gamlssNP(
        "y ~ x", family="NO", data=_data(), K=2,
        control=np_control(EMn_cyc=60, trace=False, plot_opt=0),
    )
    density = get_pdfNP(fit, 1)
    assert float(np.ravel(density(0.0))[0]) > 0.0
    assert fit.ebp.shape == (fit.N,)
    assert fit.residuals.shape == (fit.N,)


def test_plot_mp_returns_the_persp_geometry():
    result = plotMP([0.0, 3.0], [0.0, 0.5], [0.55, 0.45], theta=30, phi=40)
    assert result.x_grid.size == 20 and result.y_grid.size == 20
    assert result.z.shape == (20, 20)
    assert result.points.shape == (2, 3)
    assert len(result.stems) == 2
    assert result.stems[0][0][2] == 0.0
    assert result.stems[0][1][2] == pytest.approx(0.55)
    assert (result.theta, result.phi) == (30.0, 40.0)
    with pytest.raises(ValueError, match="equal length"):
        plotMP([0.0], [0.0, 1.0], [0.5, 0.5])


def test_dmx_accepts_a_scalar_response():
    value = dmx(0.0, mu=[0.0, 3.0], sigma=[1.0, 1.0], pi=[0.5, 0.5], family=["NO", "NO"])
    assert np.asarray(value).shape == (1,)
    assert float(np.ravel(value)[0]) > 0.0


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"pi_formula": "~x"}, "multinom"),
        ({"zero_component": True}, "point mass at zero"),
    ],
)
def test_gamlss_mx_rejects_unported_branches(kwargs, match):
    with pytest.raises(NotImplementedError, match=match):
        gamlssMX("y ~ x", family="NO", K=2, data=_data(), **kwargs)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"mixture": "gq"}, "gq"),
        ({"random": "~x"}, "random=~1"),
    ],
)
def test_gamlss_np_rejects_unported_branches(kwargs, match):
    with pytest.raises(NotImplementedError, match=match):
        gamlssNP("y ~ x", family="NO", K=2, data=_data(), **kwargs)


def test_gamlss_mx_validates_component_lists_and_missing_data():
    with pytest.raises(ValueError, match="formula"):
        gamlssMX(["y ~ x"], family=["NO", "NO"], K=2, data=_data())
    with pytest.raises(ValueError, match="NA"):
        gamlssMX("y ~ x", family="NO", K=2, data={"y": [1.0, np.nan], "x": [0.0, 1.0]})
