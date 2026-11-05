"""R gamlss.foreach helpers: getSVD() and which.Data.Corr().

getSVD is checked against live R on the *same* matrix (the RNGs differ), and the
singular vectors are compared in absolute value because their sign is not
identifiable.  which.Data.Corr is compared directly.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from omnilss.pcr import SVDResult, getSVD, which_Data_Corr

_REPO = Path(__file__).resolve().parents[2]

_SVD_SCRIPT = """suppressPackageStartupMessages(library(gamlss))
source("R/gamlss.foreach/R/PCR-GAMLSS.R")
library(jsonlite)
a <- commandArgs(trailingOnly = TRUE)
x <- as.matrix(fromJSON(a[1]))
S <- getSVD(x)
cat(toJSON(list(d = as.numeric(S$d), X = as.numeric(S$X), center = as.numeric(S$center),
                scale = as.numeric(S$scale), u = as.numeric(S$u), vt = as.numeric(S$vt)),
           auto_unbox = FALSE, digits = 17))
"""

_CORR_SCRIPT = """suppressPackageStartupMessages(library(gamlss))
source("R/gamlss.foreach/R/PCR-GAMLSS.R")
df <- data.frame(a = c(1, 2, 3, 4, 5), b = c(2, 4, 6, 8, 10), c = c(5, 1, 4, 2, 3))
cat(jsonlite::toJSON(which.Data.Corr(df, r = 0.9, digits = 3), auto_unbox = FALSE, digits = 17))
"""


def _matrix() -> np.ndarray:
    return np.random.RandomState(1).normal(size=(10, 4))


def test_svd_returns_the_scaled_decomposition() -> None:
    result = getSVD(_matrix())
    assert isinstance(result, SVDResult)
    assert result.d.shape == (4,)
    assert result.X.shape == (10, 4)
    assert result.u.shape == (10, 4)
    assert result.vt.shape == (4, 4)


def test_svd_centres_and_scales_like_r_scale() -> None:
    x = _matrix()
    result = getSVD(x)
    np.testing.assert_allclose(result.X.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(result.X.std(axis=0, ddof=1), 1.0, rtol=1e-12)
    np.testing.assert_allclose(result.center, x.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(result.scale, x.std(axis=0, ddof=1), rtol=1e-12)


def test_svd_reconstructs_the_scaled_matrix() -> None:
    result = getSVD(_matrix())
    rebuilt = result.u @ np.diag(result.d) @ result.vt
    np.testing.assert_allclose(rebuilt, result.X, rtol=1e-10, atol=1e-12)


def test_svd_rejects_non_matrices() -> None:
    with pytest.raises(ValueError, match="x must be a matrix"):
        getSVD(np.arange(5.0))


def test_which_data_corr_reports_the_correlated_pair() -> None:
    frame = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10], "c": [5, 1, 4, 2, 3]})
    result = which_Data_Corr(frame, 0.9, print=False)
    assert result.to_dict("records") == [{"name1": "a", "name2": "b", "corrs": 1.0}]


def test_which_data_corr_returns_none_when_nothing_passes() -> None:
    frame = pd.DataFrame({"a": [1, 2, 3, 4, 5], "c": [5, 1, 4, 2, 3]})
    assert which_Data_Corr(frame, 0.99, print=False) is None


def test_which_data_corr_skips_factors_and_characters() -> None:
    frame = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [2, 4, 6, 8, 10],
            "g": pd.Categorical(["x", "y", "x", "y", "x"]),
            "s": ["p", "q", "p", "q", "p"],
        }
    )
    result = which_Data_Corr(frame, 0.9, print=False)
    assert set(result["name1"]) | set(result["name2"]) == {"a", "b"}


def test_which_data_corr_validates_the_threshold() -> None:
    frame = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]})
    for bad in (0.0, 1.0, -1.5):
        with pytest.raises(ValueError, match="greater than  0 and lass than 1"):
            which_Data_Corr(frame, bad, print=False)


@pytest.mark.slow
def test_svd_matches_live_r() -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss.foreach" / "R" / "PCR-GAMLSS.R").exists():
        pytest.skip("vendored gamlss.foreach not present")
    x = _matrix()
    with tempfile.TemporaryDirectory() as tmp:
        payload = Path(tmp) / "x.json"
        payload.write_text(json.dumps(x.tolist()), encoding="utf-8")
        script = Path(tmp) / "svd.R"
        script.write_text(_SVD_SCRIPT, encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(script), str(payload)],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(_REPO),
        )
    if proc.returncode != 0:
        pytest.skip(f"R failed: {proc.stderr[-300:]}")
    ref = json.loads(proc.stdout.strip().splitlines()[-1])
    result = getSVD(x)
    np.testing.assert_allclose(result.d, np.asarray(ref["d"], dtype=float), rtol=1e-12)
    np.testing.assert_allclose(
        result.X,
        np.asarray(ref["X"], dtype=float).reshape(result.X.shape, order="F"),
        rtol=1e-12,
    )
    np.testing.assert_allclose(result.center, np.asarray(ref["center"], dtype=float), rtol=1e-12)
    np.testing.assert_allclose(result.scale, np.asarray(ref["scale"], dtype=float), rtol=1e-12)
    reference_u = np.asarray(ref["u"], dtype=float).reshape(result.u.shape, order="F")
    np.testing.assert_allclose(np.abs(result.u), np.abs(reference_u), rtol=1e-11, atol=1e-12)


@pytest.mark.slow
def test_which_data_corr_matches_live_r() -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss.foreach" / "R" / "PCR-GAMLSS.R").exists():
        pytest.skip("vendored gamlss.foreach not present")
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "corr.R"
        script.write_text(_CORR_SCRIPT, encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(script)], capture_output=True, text=True, timeout=300, cwd=str(_REPO)
        )
    if proc.returncode != 0:
        pytest.skip(f"R failed: {proc.stderr[-300:]}")
    reference = json.loads(proc.stdout.strip().splitlines()[-1])
    frame = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10], "c": [5, 1, 4, 2, 3]})
    result = which_Data_Corr(frame, 0.9, print=False)
    assert result.shape[0] == len(reference)
    assert result["name1"].tolist() == [row[0] for row in reference]
    assert result["name2"].tolist() == [row[1] for row in reference]
