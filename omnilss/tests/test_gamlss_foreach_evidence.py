"""Dedicated evidence module for gamlss.foreach (P4-4, D2.253).

Two layers of evidence:

1. surface: every gamlss.foreach symbol in the coverage artifact is exported and
   marked public_api / tested;
2. behaviour: the five ported entry points run end to end (quantile fits,
   rolling forecasts, both bootstraps and the bootstrap centile curves).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import omnilss
from omnilss import (
    BayesianBoot,
    NonParametricBoot,
    centiles_boot,
    fitQR,
    fitRolling,
    gamlss,
)

_REPO = Path(__file__).resolve().parents[2]
_COVERAGE = _REPO / "docs" / "development" / "r-symbol-coverage.json"
_EXPECTED_SYMBOLS = 13


def _package_symbols() -> list[dict]:
    payload = json.loads(_COVERAGE.read_text(encoding="utf-8"))
    return [row for row in payload["symbols"] if row.get("r_package") == "gamlss.foreach"]


def _series(seed: int = 2, n: int = 60) -> dict:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    y = 1.0 + 0.02 * t + rng.normal(scale=0.3, size=n)
    return {"y": y, "t": t}


@pytest.fixture(scope="module")
def model():
    data = _series()
    return gamlss("y ~ t", family="NO", data=data, max_iter=30), data


def test_every_foreach_symbol_is_exported_and_tested():
    symbols = _package_symbols()
    assert len(symbols) == _EXPECTED_SYMBOLS
    for row in symbols:
        layers = row["layers"]
        assert layers["surface"] == "public_api", row["r_symbol"]
        assert layers["kernel"] == "implemented", row["r_symbol"]
        assert layers["parity"] in {"tested", "r_consistency_test"}, row["r_symbol"]
        python_name = row["r_symbol"].replace(".", "_")
        assert hasattr(omnilss, python_name), row["r_symbol"]


def test_quantile_and_rolling_workflow():
    rng = np.random.default_rng(3)
    n = 90
    x = rng.normal(size=n)
    y = 1.0 + 0.5 * x + (0.4 + 0.2 * np.abs(x)) * rng.normal(size=n)
    models = fitQR("y ~ x", {"y": y, "x": x}, quantile=[0.25, 0.5, 0.75], max_iter=60)
    assert len(models) == 3
    for fitted_model, quantile in zip(models, (0.25, 0.5, 0.75)):
        values = np.asarray(fitted_model.fitted_values["mu"], dtype=float)
        fraction = float(np.mean(y < values))
        assert abs(fraction - quantile) < 0.12
        assert np.isfinite(float(fitted_model.g_dev))

    data = _series()
    simple = gamlss("y ~ t", family="NO", data=data, max_iter=30)
    rolled = fitRolling(simple, window=45)
    assert rolled.shape[0] == data["y"].size - 45
    assert list(rolled.columns) == ["mu", "sigma", "GDincr"]
    assert np.all(np.isfinite(rolled["GDincr"].to_numpy()))
    assert np.all(rolled["sigma"].to_numpy() > 0.0)


def test_bootstrap_and_centile_workflow(model):
    fitted, data = model
    nonparametric = NonParametricBoot(fitted, B=4, seed=1)
    bayesian = BayesianBoot(fitted, B=4, seed=1)
    for result in (nonparametric, bayesian):
        assert result.boot.shape[1] == result.orig_coef.size
        assert result.trueB + result.failed == result.B
        assert np.all(np.isfinite(result.boot))

    repeat = NonParametricBoot(fitted, B=4, seed=1)
    np.testing.assert_array_equal(nonparametric.boot, repeat.boot)

    xvalues = np.array([10.0, 30.0, 50.0])
    centiles = centiles_boot(
        fitted, xname="t", xvalues=xvalues, cent=[50], B=2, seed=2
    )
    assert centiles.boot0.shape == (3, 1)
    assert centiles.trueB + centiles.failed == centiles.B
    from omnilss.prediction import predict_params

    params = predict_params(fitted, {"t": xvalues})
    direct = np.asarray(fitted.family.q(0.5, params["mu"], params["sigma"]), dtype=float)
    np.testing.assert_allclose(centiles.boot0[:, 0], direct, rtol=1e-12, atol=1e-12)


def test_foreach_entry_points_validate_their_arguments(model):
    fitted, data = model
    with pytest.raises(ValueError, match="between 0 and 1"):
        fitQR("y ~ t", data, quantile=[1.5])
    with pytest.raises(ValueError, match="positive"):
        NonParametricBoot(fitted, B=0)
    with pytest.raises(ValueError, match="window"):
        fitRolling(fitted, window=0)
    with pytest.raises(ValueError, match="xvalues"):
        centiles_boot(fitted, xname="t")
