"""Live-R consistency evidence for gamlss.tr (D2.266).

Compares the truncated density, CDF, quantile, family deviance and the generated
functions against the vendored sources for left, right and both-side truncation.
The random generator and the tail-fitting wrappers are not compared numerically,
so they stay at "tested".
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from omnilss import gen_trun, trun, trun_d, trun_p, trun_q
from tests.r_consistency_scripts import TR_R_SCRIPT

_REPO = Path(__file__).resolve().parents[2]
_R_SCRIPT = TR_R_SCRIPT


def _payload() -> dict:
    return {
        "x": [-1.0, 0.0, 0.5, 1.0, 2.0],
        "mu": [0.0, 0.2, -0.3, 0.5, -0.5],
        "sigma": [1.0, 1.2, 0.8, 1.1, 0.9],
        "probs": [0.1, 0.5, 0.9],
    }


def _live_r(payload: dict) -> dict:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss.tr" / "R" / "trun.d-30-03-24.R").exists():
        pytest.skip("vendored gamlss.tr source not present")
    with tempfile.TemporaryDirectory() as tmp:
        data_file = Path(tmp) / "tr.json"
        data_file.write_text(json.dumps(payload), encoding="utf-8")
        script = Path(tmp) / "ref.R"
        script.write_text(_R_SCRIPT, encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(script), str(data_file)],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(_REPO),
        )
    if proc.returncode != 0:
        pytest.skip(f"R failed: {proc.stderr[-300:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _array(values) -> np.ndarray:
    return np.asarray(
        [np.nan if item is None else float(item) for item in values], dtype=float
    )


@pytest.fixture(scope="module")
def reference() -> dict:
    return _live_r(_payload())


@pytest.mark.slow
@pytest.mark.parametrize("truncation", ["left", "right", "both"])
def test_trun_density_cdf_quantile_match_live_r(reference, truncation):
    payload = _payload()
    par = (-0.5, 1.5) if truncation == "both" else 0.0
    x = np.asarray(payload["x"], dtype=float)
    mu = np.asarray(payload["mu"], dtype=float)
    sigma = np.asarray(payload["sigma"], dtype=float)
    probs = np.asarray(payload["probs"], dtype=float)

    mine_d = np.asarray(trun_d(par, "NO", truncation)(x, mu=mu, sigma=sigma), dtype=float)
    mine_p = np.asarray(trun_p(par, "NO", truncation)(x, mu=mu, sigma=sigma), dtype=float)
    mine_q = np.asarray(
        trun_q(par, "NO", truncation)(probs, mu=mu[: probs.size], sigma=sigma[: probs.size]),
        dtype=float,
    )
    expected = reference[truncation]
    np.testing.assert_allclose(mine_d, _array(expected["d"]), atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(mine_p, _array(expected["p"]), atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(mine_q, _array(expected["q"]), atol=1e-12, equal_nan=True)


@pytest.mark.slow
def test_trun_family_deviance_matches_live_r(reference):
    payload = _payload()
    x = np.asarray(payload["x"], dtype=float)
    mu = np.asarray(payload["mu"], dtype=float)
    sigma = np.asarray(payload["sigma"], dtype=float)
    family = trun(par=0.0, family="NO", type="left")
    mine = np.asarray(family.g_dev_inc(y=x, mu=mu, sigma=sigma), dtype=float)
    np.testing.assert_allclose(
        mine, _array(reference["left"]["fam_dev"]), atol=1e-10, equal_nan=True
    )


@pytest.mark.slow
def test_gen_trun_generates_the_same_functions(reference):
    payload = _payload()
    generated = gen_trun(par=0.0, family="NO", type="left")
    x = np.asarray(payload["x"], dtype=float)
    mu = np.asarray(payload["mu"], dtype=float)
    sigma = np.asarray(payload["sigma"], dtype=float)
    probs = np.asarray(payload["probs"], dtype=float)
    expected = reference["gen"]
    np.testing.assert_allclose(
        np.asarray(generated["dNOtr"](x, mu=mu, sigma=sigma), dtype=float),
        _array(expected["d"]), atol=1e-12, equal_nan=True,
    )
    np.testing.assert_allclose(
        np.asarray(generated["pNOtr"](x, mu=mu, sigma=sigma), dtype=float),
        _array(expected["p"]), atol=1e-12, equal_nan=True,
    )
    np.testing.assert_allclose(
        np.asarray(
            generated["qNOtr"](probs, mu=mu[: probs.size], sigma=sigma[: probs.size]),
            dtype=float,
        ),
        _array(expected["q"]), atol=1e-12, equal_nan=True,
    )
