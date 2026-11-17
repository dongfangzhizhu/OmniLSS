# SPDX-License-Identifier: GPL-3.0-or-later
"""gamlss.foreach: fitQR / fitRolling / bootstrap helpers (D2.253).

fitQR is verified against live R: SEP3 with sigma = tau = 1 and
nu = sqrt((1 - q) / q) on the same sample gives

    q = 0.25  deviance 413.1560604055
    q = 0.50  deviance 382.6045908747
    q = 0.75  deviance 413.8929784546

Coefficients are not asserted: the asymmetric Laplace objective is piecewise
linear and both R and this port report the deviance of the last inner IWLS fit
while the outer coefficients can sit at a neighbouring kink.  The tests pin the
likelihood (deviance) and the quantile behaviour instead.

The bootstraps are random procedures; the contract is shape, reproducibility
for a fixed seed, and bootstrap means near the original fit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from omnilss import (
    BayesianBoot,
    NonParametricBoot,
    centiles_boot,
    fitQR,
    fitRolling,
    gamlss,
)

QR_FIXTURE = Path(__file__).parent / "fixtures" / "foreach" / "qr_sample.json"
R_DEVIANCE = {0.25: 413.1560604055, 0.5: 382.6045908747, 0.75: 413.8929784546}


@pytest.fixture(scope="module")
def qr_data() -> dict:
    payload = json.loads(QR_FIXTURE.read_text(encoding="utf-8"))
    return {"y": np.asarray(payload["y"], dtype=float), "x": np.asarray(payload["x"], dtype=float)}


def _series(seed: int = 2, n: int = 80):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    y = 1.0 + 0.02 * t + rng.normal(scale=0.3, size=n)
    return {"y": y, "t": t}


def test_fitqr_matches_live_r_deviance(qr_data):
    models = fitQR("y ~ x", qr_data, quantile=[0.25, 0.5, 0.75], max_iter=100)
    assert len(models) == 3
    for model, quantile in zip(models, (0.25, 0.5, 0.75)):
        assert float(model.g_dev) == pytest.approx(R_DEVIANCE[quantile], rel=1e-5)
        # Only mu is estimated; the three other parameters stay fixed.
        assert list(model.coefficients) == ["mu"]


def test_fitqr_has_quantile_behaviour(qr_data):
    for quantile in (0.25, 0.5, 0.75):
        model = fitQR("y ~ x", qr_data, quantile=[quantile], max_iter=100)[0]
        fitted = np.asarray(model.fitted_values["mu"], dtype=float)
        fraction = float(np.mean(qr_data["y"] < fitted))
        assert abs(fraction - quantile) < 0.1, (quantile, fraction)


def test_fitqr_rejects_invalid_quantiles(qr_data):
    for bad in (0.0, 1.0, -0.1, 1.2):
        with pytest.raises(ValueError, match="between 0 and 1"):
            fitQR("y ~ x", qr_data, quantile=[bad])


def test_fitrolling_matches_a_direct_window_fit():
    data = _series()
    model = gamlss("y ~ t", family="NO", data=data, max_iter=30)
    rolled = fitRolling(model, window=50)
    assert isinstance(rolled, pd.DataFrame)
    assert rolled.shape == (30, 3)
    assert list(rolled.columns) == ["mu", "sigma", "GDincr"]
    assert list(rolled.index[:3]) == [51, 52, 53]

    window_data = {"y": data["y"][0:50], "t": data["t"][0:50]}
    direct = gamlss("y ~ t", family="NO", data=window_data, max_iter=30)
    from omnilss.prediction import predict_params

    predicted = predict_params(direct, {"t": np.array([50.0])})
    assert rolled.iloc[0]["mu"] == pytest.approx(float(np.ravel(predicted["mu"])[0]), rel=1e-10)
    assert rolled.iloc[0]["sigma"] == pytest.approx(
        float(np.ravel(predicted["sigma"])[0]), rel=1e-10
    )
    increment = float(
        np.ravel(
            direct.family.g_dev_inc(
                y=float(data["y"][50]), mu=float(np.ravel(predicted["mu"])[0]),
                sigma=float(np.ravel(predicted["sigma"])[0]),
            )
        )[0]
    )
    assert rolled.iloc[0]["GDincr"] == pytest.approx(increment, rel=1e-10)


def test_fitrolling_validates_window_and_time_index():
    data = _series()
    model = gamlss("y ~ t", family="NO", data=data, max_iter=10)
    with pytest.raises(ValueError, match="window"):
        fitRolling(model, window=0)
    with pytest.raises(ValueError, match="window"):
        fitRolling(model, window=80)
    rolled = fitRolling(model, window=70, as_time=data["t"])
    assert rolled.index[0] == pytest.approx(70.0)


@pytest.mark.parametrize("bootstrap", (NonParametricBoot, BayesianBoot))
def test_bootstrap_shape_reproducibility_and_location(bootstrap):
    data = _series()
    model = gamlss("y ~ t", family="NO", data=data, max_iter=30)
    first = bootstrap(model, B=5, seed=3)
    second = bootstrap(model, B=5, seed=3)
    assert first.boot.shape == (first.trueB, first.orig_coef.size)
    assert first.trueB <= first.B
    np.testing.assert_array_equal(first.boot, second.boot)
    # The bootstrap mean of the coefficients sits near the original fit.
    assert np.allclose(first.boot.mean(axis=0), first.orig_coef, rtol=0.25, atol=0.05)
    assert set(first.param) == {"mu", "sigma"}
    assert first.param["mu"].shape[0] == first.trueB


def test_bootstrap_rejects_bad_inputs():
    data = _series()
    model = gamlss("y ~ t", family="NO", data=data, max_iter=10)
    with pytest.raises(ValueError, match="gamlss object"):
        NonParametricBoot("not a model", B=2)
    with pytest.raises(ValueError, match="positive"):
        NonParametricBoot(model, B=0)


def test_centiles_boot_matches_direct_quantiles():
    data = _series()
    model = gamlss("y ~ t", family="NO", data=data, max_iter=30)
    xvalues = np.array([20.0, 40.0, 60.0])
    result = centiles_boot(model, xname="t", xvalues=xvalues, cent=[25, 50, 75], B=3, seed=1)
    assert isinstance(result.boot0, np.ndarray)
    assert result.boot0.shape == (3, 3)
    assert result.trueB <= 3 and result.failed == 3 - result.trueB
    from omnilss.prediction import predict_params

    params = predict_params(model, {"t": xvalues})
    direct = np.column_stack([
        np.asarray(model.family.q(value / 100.0, params["mu"], params["sigma"]), dtype=float)
        for value in (25, 50, 75)
    ])
    np.testing.assert_allclose(result.boot0, direct, rtol=1e-12, atol=1e-12)


def test_centiles_boot_validates_its_inputs():
    data = _series()
    model = gamlss("y ~ t", family="NO", data=data, max_iter=10)
    with pytest.raises(ValueError, match="xvalues"):
        centiles_boot(model, xname="t")
    with pytest.raises(ValueError, match="xname"):
        centiles_boot(model, xvalues=np.array([1.0]))
    with pytest.raises(ValueError, match="not included"):
        centiles_boot(model, xname="z", xvalues=np.array([1.0]))
