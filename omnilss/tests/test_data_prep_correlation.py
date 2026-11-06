"""R gamlss.prepdata correlation / empty-space helpers.

data_cor(), data_pcor() and data_void() share R's preamble (data_cut, na.omit,
factor removal, size checks) and differ in what they compute and in *when* the
diag.off / lower.tri.off flags are applied -- data_cor returns before applying
them, the other two do not.  Both behaviours are pinned here and checked against
live R.
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

from omnilss.data_prep import data_cor, data_dim, data_pcor, data_void, void

_REPO = Path(__file__).resolve().parents[2]

_FOREACH = """foreach <- function(...) {
  args <- list(...); nm <- names(args)
  iter <- nm[nzchar(nm) & !startsWith(nm, ".")][1]
  structure(list(name = iter, values = args[[iter]],
                 combine = if (".combine" %in% nm) args[[".combine"]] else NULL),
            class = "foreachStub")
}
`%dopar%` <- function(obj, expr) {
  e <- substitute(expr); env <- parent.frame()
  out <- lapply(obj$values, function(v) { assign(obj$name, v, envir = env); eval(e, env) })
  if (is.null(obj$combine)) out else do.call(obj$combine, out)
}
"""

_RSCRIPT = _FOREACH + """suppressPackageStartupMessages(library(gamlss))
library(jsonlite)
source("R/gamlss.prepdata/R/data_part.R")
source("R/gamlss.prepdata/R/data_cor.R")
source("R/gamlss.prepdata/R/data_pcor.R")
source("R/gamlss.prepdata/R/empty_space.R")
a <- commandArgs(trailingOnly = TRUE); d <- fromJSON(a[1])
df <- data.frame(x1 = d$x1, x2 = d$x2, x3 = d$x3, g = factor(d$g))
cc <- suppressWarnings(data_cor(df, plot = FALSE))
pc <- suppressWarnings(data_pcor(df, plot = FALSE))
pc2 <- suppressWarnings(data_pcor(df, plot = FALSE, diag.off = FALSE))
vv <- suppressWarnings(data_void(df, plot = FALSE))
vv2 <- suppressWarnings(data_void(df, plot = FALSE, diag.off = FALSE))
out <- list(v = void(d$x1, d$x2), v2 = void(d$x1, d$x3),
  cor = as.numeric(cc), cor_dim = as.integer(dim(cc)), cor_names = colnames(cc),
  pcor = as.numeric(pc), pcor2 = as.numeric(pc2),
  voidm = as.numeric(vv), voidm2 = as.numeric(vv2))
cat("JSONSTART"); cat(toJSON(out, auto_unbox = FALSE, digits = 17))
"""


def _data(seed: int = 5, n: int = 50):
    rs = np.random.RandomState(seed)
    x1 = rs.normal(size=n)
    x2 = 0.8 * x1 + rs.normal(size=n)
    x3 = rs.normal(size=n)
    g = rs.choice(["a", "b"], n)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "g": pd.Categorical(g)})


def test_void_uses_the_r_table_length() -> None:
    rs = np.random.RandomState(0)
    x = rs.normal(size=50)
    # floor(exp(-0.5165698 + 0.4983922 * log(50))) = 4 bins per variable.
    # Binning a variable against itself fills the diagonal, so 12 of 16 cells
    # stay empty.
    assert void(x, x) == pytest.approx(12.0 / 16.0)
    assert void(x, x, table_length=2) == pytest.approx(1.0 / 2.0)


def test_void_requires_matching_lengths() -> None:
    with pytest.raises(ValueError, match="the same length"):
        void(np.arange(10.0), np.arange(9.0))


def test_void_prints_when_asked(capsys) -> None:
    rs = np.random.RandomState(1)
    x = rs.normal(size=40)
    value = void(x, -x, print=True)
    captured = capsys.readouterr().out
    assert "% void" in captured and "table length =" in captured
    assert 0.0 <= value <= 1.0


def test_data_cor_returns_the_rounded_matrix() -> None:
    frame = _data()
    matrix = data_cor(frame, plot=False, print_info=False)
    assert list(matrix.columns) == ["x1", "x2", "x3"]
    assert list(matrix.index) == ["x1", "x2", "x3"]
    expected = frame[["x1", "x2", "x3"]].corr(method="pearson").round(3)
    np.testing.assert_allclose(matrix.to_numpy(), expected.to_numpy(), atol=5e-4)
    # R returns the matrix before applying diag.off, so the diagonal is 1.
    assert np.all(np.diag(matrix.to_numpy()) == 1.0)


def test_data_cor_supports_the_other_types() -> None:
    frame = _data()
    spearman = data_cor(frame, type="spearman", plot=False, print_info=False)
    expected = frame[["x1", "x2", "x3"]].corr(method="spearman").round(3)
    np.testing.assert_allclose(spearman.to_numpy(), expected.to_numpy(), atol=5e-4)
    with pytest.raises(ValueError, match="should be one of"):
        data_cor(frame, type="cosine", plot=False, print_info=False)


def test_data_cor_validates_the_frame() -> None:
    frame = _data()
    with pytest.raises(ValueError, match="the data is list"):
        data_cor({"a": [1, 2, 3]}, plot=False, print_info=False)
    with pytest.raises(ValueError, match="too small"):
        data_cor(frame.head(10), plot=False, print_info=False)
    with pytest.raises(ValueError, match="no variable is left"):
        data_cor(frame[["g"]], plot=False, print_info=False)
    with pytest.raises(ValueError, match="only one variable is left"):
        data_cor(frame[["x1"]], plot=False, print_info=False)


def test_data_cor_warns_about_omitted_factors() -> None:
    frame = _data()
    with pytest.warns(UserWarning, match="factors have been omited"):
        data_cor(frame, plot=False, print_info=False)


def test_data_pcor_applies_diag_off_before_returning() -> None:
    frame = _data()
    matrix = data_pcor(frame, plot=False, print_info=False)
    assert np.all(np.isnan(np.diag(matrix.to_numpy())))
    kept = data_pcor(frame, plot=False, diag_off=False, print_info=False)
    np.testing.assert_allclose(np.diag(kept.to_numpy()), 1.0)
    # The off-diagonal entries are the partial correlations.
    correlation = frame[["x1", "x2", "x3"]].corr().to_numpy()
    precision = np.linalg.inv(correlation)
    scaling = 1.0 / np.sqrt(np.diag(precision))
    expected = 2.0 * np.eye(3) - precision * np.outer(scaling, scaling)
    np.testing.assert_allclose(kept.to_numpy(), np.round(expected, 3), atol=5e-4)


def test_data_void_is_symmetric_and_masks_the_diagonal() -> None:
    frame = _data()
    matrix = data_void(frame, plot=False, print_info=False).to_numpy()
    assert np.all(np.isnan(np.diag(matrix)))
    assert np.allclose(matrix, matrix.T, equal_nan=True)
    assert np.all((matrix[~np.isnan(matrix)] >= 0.0) & (matrix[~np.isnan(matrix)] <= 1.0))
    kept = data_void(frame, plot=False, diag_off=False, print_info=False).to_numpy()
    np.testing.assert_allclose(np.diag(kept), 1.0)
    lower = data_void(frame, plot=False, lower_tri_off=True, print_info=False).to_numpy()
    assert np.all(np.isnan(lower[np.tril_indices(3, -1)]))


def test_prepdata_summaries_run_with_the_default_print_flag() -> None:
    """The print= parameter shadows the builtin, so every summary uses _emit()."""

    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]})
    assert data_dim(frame)["nrow"] == 3


@pytest.mark.slow
def test_matches_live_r() -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss.prepdata" / "R" / "data_cor.R").exists():
        pytest.skip("vendored gamlss.prepdata not present")
    frame = _data()
    payload = {
        "x1": frame["x1"].tolist(),
        "x2": frame["x2"].tolist(),
        "x3": frame["x3"].tolist(),
        "g": frame["g"].astype(str).tolist(),
    }
    with tempfile.TemporaryDirectory() as tmp:
        data_path = Path(tmp) / "d.json"
        data_path.write_text(json.dumps(payload), encoding="utf-8")
        script_path = Path(tmp) / "cor.R"
        script_path.write_text(_RSCRIPT, encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(script_path), str(data_path)],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(_REPO),
        )
    if "JSONSTART" not in proc.stdout:
        pytest.skip(f"R failed: {proc.stderr[-300:]}")
    ref = json.loads(proc.stdout.split("JSONSTART", 1)[1].strip())

    def difference(mine, key):
        expected = np.asarray([np.nan if value == "NA" else value for value in ref[key]], dtype=float)
        mine = np.asarray(mine, dtype=float)
        expected = expected.reshape(mine.shape, order="F")
        both = np.isnan(mine) & np.isnan(expected)
        return float(np.max(np.where(both, 0.0, np.abs(np.nan_to_num(mine) - np.nan_to_num(expected)))))

    assert difference(void(frame["x1"], frame["x2"]), "v") == 0.0
    assert difference(void(frame["x1"], frame["x3"]), "v2") == 0.0

    shape = tuple(int(value) for value in ref["cor_dim"])
    correlation = data_cor(frame, plot=False, print_info=False)
    assert correlation.shape == shape
    assert list(correlation.columns) == [str(value) for value in ref["cor_names"]]
    assert difference(correlation.to_numpy(), "cor") == 0.0

    partial = data_pcor(frame, plot=False, print_info=False)
    assert difference(partial.to_numpy(), "pcor") == 0.0
    partial_kept = data_pcor(frame, plot=False, diag_off=False, print_info=False)
    assert difference(partial_kept.to_numpy(), "pcor2") == 0.0

    empties = data_void(frame, plot=False, print_info=False)
    assert difference(empties.to_numpy(), "voidm") == 0.0
    empties_kept = data_void(frame, plot=False, diag_off=False, print_info=False)
    assert difference(empties_kept.to_numpy(), "voidm2") == 0.0