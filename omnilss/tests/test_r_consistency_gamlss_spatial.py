"""Live-R consistency evidence for gamlss.spatial (D2.265).

The vendored MRF/MRFA sources only use spam for matrix storage, so identity
wrappers let them run without the spam package; the fits are then compared with
this port (MRF Q-function and MRFA alternating/df/lambda modes).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from omnilss import MRF, MRFA

_REPO = Path(__file__).resolve().parents[2]
_R_SCRIPT = """
suppressPackageStartupMessages(library(gamlss.dist))
suppressPackageStartupMessages(library(jsonlite))
as.spam <- function(x) x
diag.spam <- function(x) diag(x)
spam <- function(x, nrow) matrix(x, nrow = nrow)
source("R/gamlss.spatial/R/MRF.R")
source("R/gamlss.spatial/R/MRFA.R")
args <- commandArgs(trailingOnly = TRUE)
d <- fromJSON(args[1])
x <- factor(d$area, levels = d$areas)
q <- MRF(y = d$y, x = x, precision = d$G)
a <- MRFA(y = d$y, x = x, precision = d$G)
a_df <- MRFA(y = d$y, x = x, precision = d$G, df = 6)
a_lam <- MRFA(y = d$y, x = x, precision = d$G, lambda = 2)
out <- list(
  q = list(beta = as.numeric(q$beta), sig2e = q$sig2e, sig2b = q$sig2b,
           df = q$df, dev = q$deviance),
  a = list(beta = as.numeric(a$beta), sig2e = a$sig2e, sig2b = a$sig2b,
           df = a$df, lam = a$lambda, dev = a$deviance),
  a_df = list(beta = as.numeric(a_df$beta), lam = a_df$lambda, df = a_df$df,
              dev = a_df$deviance),
  a_lam = list(beta = as.numeric(a_lam$beta), lam = a_lam$lambda, df = a_lam$df,
               dev = a_lam$deviance)
)
cat(toJSON(out, digits = 17, auto_unbox = TRUE))
"""


def _fixture() -> dict:
    areas = [f"A{index}" for index in range(1, 10)]
    precision = np.zeros((9, 9))
    for index in range(9):
        row, column = divmod(index, 3)
        neighbours = []
        if row > 0:
            neighbours.append(index - 3)
        if row < 2:
            neighbours.append(index + 3)
        if column > 0:
            neighbours.append(index - 1)
        if column < 2:
            neighbours.append(index + 1)
        precision[index, index] = float(len(neighbours))
        for neighbour in neighbours:
            precision[index, neighbour] = -1.0
    rng = np.random.default_rng(7)
    n = 200
    area = np.array([areas[index] for index in rng.integers(0, 9, size=n)])
    effects = dict(zip(areas, [-1.0, -0.5, 0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5]))
    y = np.array([effects[item] for item in area]) + rng.normal(scale=0.4, size=n)
    return {
        "y": y.tolist(),
        "area": area.tolist(),
        "areas": areas,
        "G": precision.tolist(),
    }


def _live_r(payload: dict) -> dict:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    source = _REPO / "R" / "gamlss.spatial" / "R" / "MRF.R"
    if not source.exists():
        pytest.skip("vendored gamlss.spatial source not present")
    with tempfile.TemporaryDirectory() as tmp:
        data_file = Path(tmp) / "spatial.json"
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
    return _live_r(_fixture())


@pytest.fixture(scope="module")
def inputs():
    payload = _fixture()
    return (
        np.asarray(payload["y"], dtype=float),
        np.asarray(payload["area"]),
        np.asarray(payload["G"], dtype=float),
    )


@pytest.mark.slow
def test_mrf_matches_live_r(reference, inputs):
    y, area, precision = inputs
    fit = MRF(y=y, x=area, precision=precision)
    expected = reference["q"]
    np.testing.assert_allclose(
        np.asarray(fit.beta, dtype=float), np.asarray(expected["beta"], dtype=float), atol=1e-6
    )
    assert fit.sig2e == pytest.approx(float(expected["sig2e"]), rel=1e-5)
    assert fit.sig2b == pytest.approx(float(expected["sig2b"]), rel=1e-5)
    assert fit.df == pytest.approx(float(expected["df"]), rel=1e-5)
    assert fit.deviance == pytest.approx(float(expected["dev"]), rel=1e-7)


@pytest.mark.slow
def test_mrfa_matches_live_r_in_all_modes(reference, inputs):
    y, area, precision = inputs
    auto = MRFA(y=y, x=area, precision=precision)
    expected = reference["a"]
    np.testing.assert_allclose(
        np.asarray(auto.beta, dtype=float), np.asarray(expected["beta"], dtype=float), atol=1e-9
    )
    assert auto.sig2e == pytest.approx(float(expected["sig2e"]), rel=1e-9)
    assert auto.sig2b == pytest.approx(float(expected["sig2b"]), rel=1e-9)
    assert auto.df == pytest.approx(float(expected["df"]), rel=1e-9)
    assert auto.lam == pytest.approx(float(expected["lam"]), rel=1e-9)

    by_df = MRFA(y=y, x=area, precision=precision, df=6)
    # The df mode solves edf(lambda) = df; R's uniroot and our brentq agree to the
    # solver tolerance, so the coefficients inherit ~1e-7 rather than 1e-9.
    np.testing.assert_allclose(
        np.asarray(by_df.beta, dtype=float),
        np.asarray(reference["a_df"]["beta"], dtype=float), atol=2e-6,
    )
    assert by_df.lam == pytest.approx(float(reference["a_df"]["lam"]), rel=2e-6)
    assert by_df.df == pytest.approx(float(reference["a_df"]["df"]), abs=2e-6)
    assert by_df.deviance == pytest.approx(float(reference["a_df"]["dev"]), rel=5e-7)

    by_lambda = MRFA(y=y, x=area, precision=precision, lambda_=2.0)
    np.testing.assert_allclose(
        np.asarray(by_lambda.beta, dtype=float),
        np.asarray(reference["a_lam"]["beta"], dtype=float), atol=1e-9,
    )
    assert by_lambda.df == pytest.approx(float(reference["a_lam"]["df"]), rel=1e-9)
    assert by_lambda.deviance == pytest.approx(float(reference["a_lam"]["dev"]), rel=1e-9)
