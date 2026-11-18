"""Live-R consistency evidence for gamlss.prepdata (D2.265).

momentSK / centileSK are compared exactly (they are closed-form); y_zscores is
compared distributionally, because R routes it through gamlssML while the port
uses the RS path (they reach the same optimum up to the optimiser path).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from omnilss import centileSK, momentSK, y_zscores

_REPO = Path(__file__).resolve().parents[2]
_R_SCRIPT = """
suppressPackageStartupMessages(library(gamlss))
suppressPackageStartupMessages(library(gamlss.dist))
suppressPackageStartupMessages(library(jsonlite))
source("R/gamlss.dist/R/Functions_for_SK_in_gamlss.R")
args <- commandArgs(trailingOnly = TRUE)
d <- fromJSON(args[1])
m <- momentSK(d$y, weights = d$w)
cs <- centileSK(d$y, weights = d$w)
ml <- gamlssML(d$z, family = SHASHo)
out <- list(
  moment = list(skew = m$mom.skew, tkurt = m$trans.mom.kurt,
                jb = m$jarque.bera.test),
  centile = list(s025 = cs$S0.25, s001 = cs$S0.01, k001 = cs$K0.01,
                 tk = cs$trans.K0.01),
  zscores = list(resid = as.numeric(resid(ml)), params = as.numeric(fitted(ml)))
)
cat(toJSON(out, digits = 17, auto_unbox = TRUE))
"""


def _payload() -> dict:
    rng = np.random.default_rng(5)
    y = rng.gamma(2.0, 1.0, size=200) * 2.0 - 3.0
    w = rng.uniform(0.5, 2.0, size=200)
    rng2 = np.random.default_rng(0)
    z = rng2.normal(size=120)
    return {"y": y.tolist(), "w": w.tolist(), "z": z.tolist()}


def _live_r(payload: dict) -> dict:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss.dist" / "R" / "Functions_for_SK_in_gamlss.R").exists():
        pytest.skip("vendored gamlss.dist SK helper not present")
    with tempfile.TemporaryDirectory() as tmp:
        data_file = Path(tmp) / "prep.json"
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
    return _live_r(_payload())


@pytest.mark.slow
def test_moment_sk_matches_live_r(reference):
    payload = _payload()
    y = np.asarray(payload["y"], dtype=float)
    w = np.asarray(payload["w"], dtype=float)
    mine = momentSK(y, weights=w)
    expected = reference["moment"]
    assert mine.skewness == pytest.approx(float(expected["skew"]), rel=1e-12)
    assert mine.trans_kurt == pytest.approx(float(expected["tkurt"]), rel=1e-12)
    assert mine.jarque_bera == pytest.approx(float(expected["jb"]), rel=1e-12)


@pytest.mark.slow
def test_centile_sk_matches_live_r(reference):
    payload = _payload()
    y = np.asarray(payload["y"], dtype=float)
    w = np.asarray(payload["w"], dtype=float)
    mine = centileSK(y, weights=w)
    expected = reference["centile"]
    assert mine.s025 == pytest.approx(float(expected["s025"]), rel=1e-12)
    assert mine.s001 == pytest.approx(float(expected["s001"]), rel=1e-12)
    assert mine.k001 == pytest.approx(float(expected["k001"]), rel=1e-12)
    assert mine.trans_k001 == pytest.approx(float(expected["tk"]), rel=1e-12)


@pytest.mark.slow
def test_y_zscores_agrees_with_gamlss_ml(reference):
    payload = _payload()
    z = np.asarray(payload["z"], dtype=float)
    mine = np.asarray(y_zscores(z), dtype=float)
    expected = np.asarray(reference["zscores"]["resid"], dtype=float)
    assert mine.size == expected.size
    correlation = float(np.corrcoef(mine, expected)[0, 1])
    assert correlation > 0.999
    ratio = float(np.std(mine, ddof=1) / np.std(expected, ddof=1))
    assert 0.98 <= ratio <= 1.02
    assert int(np.sum(np.abs(mine) > 2.5)) == int(np.sum(np.abs(expected) > 2.5))
