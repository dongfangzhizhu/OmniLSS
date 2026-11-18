"""Live-R consistency evidence for gamlss.mx (D2.265).

Runs the vendored gamlss.mx sources under Rscript and compares the EM fits
(gamlssMX / gamlssNP) and the mixture density dMX against this port.
Skipped when Rscript or the vendored sources are unavailable.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from omnilss import gamlssMX, gamlssNP
from omnilss.mixtures import dmx, mx_control, np_control
from tests.r_consistency_scripts import MX_R_SCRIPT

_REPO = Path(__file__).resolve().parents[2]
_R_SCRIPT = MX_R_SCRIPT


def _data(n: int = 200, seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    component = (rng.random(n) < 0.4).astype(int)
    y = np.where(
        component == 1,
        3.0 + 0.5 * x + rng.normal(scale=0.8, size=n),
        0.0 + 0.3 * x + rng.normal(scale=0.6, size=n),
    )
    return {"y": y.tolist(), "x": x.tolist()}


def _live_r(payload: dict) -> dict:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    source = _REPO / "R" / "gamlss.mx" / "R" / "gamlssMX-10-11-07.R"
    if not source.exists():
        pytest.skip("vendored gamlss.mx source not present")
    with tempfile.TemporaryDirectory() as tmp:
        data_file = Path(tmp) / "mx.json"
        data_file.write_text(json.dumps(payload), encoding="utf-8")
        script = Path(tmp) / "ref.R"
        script.write_text(_R_SCRIPT, encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(script), str(data_file)],
            capture_output=True,
            text=True,
            timeout=900,
            cwd=str(_REPO),
        )
    if proc.returncode != 0:
        pytest.skip(f"R failed: {proc.stderr[-300:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def reference() -> dict:
    return _live_r(_data())


@pytest.mark.slow
def test_gamlss_mx_matches_live_r(reference):
    data = _data()
    fit = gamlssMX(
        "y ~ x", family="NO", K=2, data=data,
        control=mx_control(seed=1, n_cyc=100, trace=False, plot=False),
    )
    expected = reference["mx"]
    assert fit.deviance == pytest.approx(float(expected["dev"]), rel=1e-6)
    np.testing.assert_allclose(fit.prob, np.asarray(expected["prob"], dtype=float), atol=1e-4)
    np.testing.assert_allclose(
        np.ravel(np.asarray(fit.models[0].coefficients["mu"], dtype=float)),
        np.asarray(expected["mu1"], dtype=float), atol=2e-3,
    )
    np.testing.assert_allclose(
        np.ravel(np.asarray(fit.models[1].coefficients["mu"], dtype=float)),
        np.asarray(expected["mu2"], dtype=float), atol=2e-3,
    )
    sigma = np.concatenate([
        np.ravel(np.asarray(model.coefficients["sigma"], dtype=float))
        for model in fit.models
    ])
    np.testing.assert_allclose(sigma, np.asarray(expected["sigma"], dtype=float), atol=2e-3)


@pytest.mark.slow
def test_gamlss_np_matches_live_r(reference):
    if "np" not in reference:
        pytest.skip("R gamlssNP branch unavailable")
    data = _data()
    fit = gamlssNP(
        "y ~ x", random="~1", family="NO", data=data, K=2,
        control=np_control(EMn_cyc=60, trace=False, plot_opt=0),
    )
    expected = reference["np"]
    assert fit.deviance == pytest.approx(float(expected["dev"]), rel=1e-8)
    np.testing.assert_allclose(fit.prob, np.asarray(expected["prob"], dtype=float), atol=1e-4)
    np.testing.assert_allclose(
        np.asarray(fit.mass_points, dtype=float),
        np.asarray(expected["mass"], dtype=float), atol=1e-5,
    )
    np.testing.assert_allclose(
        np.ravel(np.asarray(fit.models[0].coefficients["mu"], dtype=float)),
        np.asarray(expected["mu"], dtype=float), atol=1e-5,
    )
    np.testing.assert_allclose(
        np.ravel(np.asarray(fit.models[0].coefficients["sigma"], dtype=float)),
        np.asarray(expected["sigma"], dtype=float), atol=1e-5,
    )
    assert fit.em_iter == int(expected["emiter"])


@pytest.mark.slow
def test_dmx_matches_live_r(reference):
    grid = np.array([-1.0, 0.0, 1.0])
    mine = np.asarray(
        dmx(
            grid,
            mu=[np.zeros(3), np.full(3, 3.0)],
            sigma=[np.ones(3), np.ones(3)],
            pi=[0.6, 0.4],
            family=["NO", "NO"],
        ),
        dtype=float,
    )
    np.testing.assert_allclose(mine, np.asarray(reference["dmx"], dtype=float), rtol=1e-12)
