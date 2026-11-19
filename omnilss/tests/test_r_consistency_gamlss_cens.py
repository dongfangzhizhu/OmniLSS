"""Live-R consistency evidence for gamlss.cens (D2.266).

Compares the density, CDF, quantile and generator helpers against the vendored
sources under Rscript. The family constructor is covered by test_gamlss_cens.py
instead: its R counterpart returns a family list whose internals are not a
numeric contract. Non-event rows of the CDF use R's runif branch, so only the
death rows are compared numerically.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from omnilss import cens_d, cens_p, cens_q, gen_cens
from tests.r_consistency_scripts import CENS_R_SCRIPT

_REPO = Path(__file__).resolve().parents[2]
_R_SCRIPT = CENS_R_SCRIPT


def _payload() -> dict:
    time = [1.0, 2.0, 3.0, 4.0, 5.0]
    status = [1, 0, 1, 0, 1]
    mu = [0.0, 0.5, -0.5, 1.0, -1.0]
    sigma = [1.0, 1.2, 0.8, 1.1, 0.9]
    cases = {
        "right": {"time": time, "status": status, "mu": mu, "sigma": sigma},
        "left": {"time": time, "status": status, "mu": mu, "sigma": sigma},
        "interval": {
            "time": time,
            "time2": [2.0, 3.0, 4.0, 5.0, 6.0],
            "status": [0, 1, 2, 3, 1],
            "mu": mu,
            "sigma": sigma,
        },
    }
    return {
        "cases": cases,
        "probs": [0.1, 0.5, 0.9],
        "gen_time": time,
        "gen_status": status,
        "gen_mu": mu,
        "gen_sigma": sigma,
    }


def _live_r(payload: dict) -> dict:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss.cens" / "R" / "cens.d.R").exists():
        pytest.skip("vendored gamlss.cens source not present")
    with tempfile.TemporaryDirectory() as tmp:
        data_file = Path(tmp) / "cens.json"
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


@pytest.fixture(scope="module")
def reference() -> dict:
    return _live_r(_payload())


@pytest.mark.slow
@pytest.mark.parametrize("censor_type", ["right", "left", "interval"])
def test_cens_density_and_cdf_match_live_r(reference, censor_type):
    payload = _payload()
    case = payload["cases"][censor_type]
    y = {"time": case["time"], "status": case["status"]}
    if censor_type == "interval":
        y["time2"] = case["time2"]
    mine_d = np.asarray(
        cens_d("NO", censor_type)(y, mu=case["mu"], sigma=case["sigma"], log=True),
        dtype=float,
    )
    mine_p = np.asarray(
        cens_p("NO", censor_type)(y, mu=case["mu"], sigma=case["sigma"]),
        dtype=float,
    )
    expected = reference[censor_type]
    np.testing.assert_allclose(mine_d, np.asarray(expected["d"], dtype=float), atol=1e-12)
    events = np.asarray(case["status"]) == 1
    np.testing.assert_allclose(
        mine_p[events], np.asarray(expected["p"], dtype=float)[events], atol=1e-12
    )


@pytest.mark.slow
def test_cens_quantile_matches_live_r(reference):
    payload = _payload()
    mine = np.asarray(cens_q("NO")(payload["probs"], 0.0, 1.0), dtype=float)
    np.testing.assert_allclose(mine, np.asarray(reference["q"], dtype=float), atol=1e-12)


@pytest.mark.slow
def test_gen_cens_generates_the_same_functions(reference):
    payload = _payload()
    generated = gen_cens("NO", "right")
    y = {"time": payload["gen_time"], "status": payload["gen_status"]}
    mine_d = np.asarray(
        generated["dNOrc"](y, mu=payload["gen_mu"], sigma=payload["gen_sigma"], log=True),
        dtype=float,
    )
    mine_p = np.asarray(
        generated["pNOrc"](y, mu=payload["gen_mu"], sigma=payload["gen_sigma"]),
        dtype=float,
    )
    expected = reference["gen"]
    np.testing.assert_allclose(mine_d, np.asarray(expected["d"], dtype=float), atol=1e-12)
    events = np.asarray(payload["gen_status"]) == 1
    np.testing.assert_allclose(
        mine_p[events], np.asarray(expected["p"], dtype=float)[events], atol=1e-12
    )
